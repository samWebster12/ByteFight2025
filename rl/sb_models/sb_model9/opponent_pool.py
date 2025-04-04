import os
import random
import copy
import torch
import numpy as np
from gymnasium import spaces
from stable_baselines3.common.policies import ActorCriticPolicy

from custom_policy3 import ByteFightMaskedPolicy, ByteFightFeaturesExtractor

class OpponentPool:
    def __init__(self, snapshot_dir, initial_policy=None, initial_rating=1000):
        self.snapshot_dir = snapshot_dir
        os.makedirs(snapshot_dir, exist_ok=True)
        self.snapshots = []  # each element: (policy_object, rating, step)
        if initial_policy is not None:
            self.add_snapshot(initial_policy, rating=initial_rating, step=0)

    def add_snapshot(self, main_policy, rating=1000, step=0):
        """
        main_policy: The SB3 policy object from your trained PPO (model.policy).
        We create a new ByteFightMaskedPolicy with the same architecture,
        load the state dict from `main_policy`, and store it.
        """
        # 1) Create dummy observation and action spaces matching your environment.
        dummy_obs_space = spaces.Dict({
            "board_image": spaces.Box(low=0, high=1, shape=(9, 64, 64), dtype=float),
            "features": spaces.Box(low=-1e6, high=1e6, shape=(15,), dtype=float),
            "action_mask": spaces.Box(low=0, high=1, shape=(10,), dtype=int),
        })
        dummy_act_space = spaces.Discrete(10)

        # 2) Define policy_kwargs (must match what your main policy uses).
        policy_kwargs = {
            "net_arch": dict(pi=[256, 128], vf=[256, 128]),
            "activation_fn": torch.nn.ReLU,
            "features_extractor_class": ByteFightFeaturesExtractor,
            "features_extractor_kwargs": {"features_dim": 256},
        }

        # 3) Supply a dummy lr_schedule (SB3 requires this).
        def dummy_lr_schedule(_):
            return 3e-4

        # 4) Construct the policy instance.
        snapshot_policy = ByteFightMaskedPolicy(
            observation_space=dummy_obs_space,
            action_space=dummy_act_space,
            lr_schedule=dummy_lr_schedule,
            **policy_kwargs
        )

        # 5) Load the main policy's state dictionary into the snapshot.
        snapshot_policy.load_state_dict(main_policy.state_dict())

        # 6) (Optional: you can also save the state dict to disk here.)
        self.snapshots.append((snapshot_policy, rating, step))
        print(f"[OPPONENT POOL] Added snapshot at step {step} with rating {rating}")

    def sample_opponent(self, main_rating=None):
        """
        Sample an opponent from the pool.
        If main_rating is provided, we weight snapshots so that those with ratings
        closer to main_rating are more likely to be chosen.
        Otherwise, uniform sampling is used.
        """
        if not self.snapshots:
            raise ValueError("Opponent pool is empty!")
        
        # If a main_rating is provided, compute weights
        if main_rating is not None:
            tau = 50.0  # sensitivity parameter; adjust as needed
            ratings = np.array([snapshot[1] for snapshot in self.snapshots])
            # Compute weight as exponential decay of absolute rating difference:
            weights = np.exp(-np.abs(ratings - main_rating) / tau)
            weights = weights / np.sum(weights)
            idx = int(np.random.choice(len(self.snapshots), p=weights))
        else:
            idx = random.randint(0, len(self.snapshots) - 1)

        opp_policy, rating, step = self.snapshots[idx]
        #print(f"[OPPONENT POOL] Sampling snapshot {idx}: step {step}, rating {rating}")
        return opp_policy, idx

    def update_rating(self, idxA, idxB, result, k_factor=32):
        """
        Basic ELO update (optional).
        result: 1.0 if snapshot A wins, 0.0 if snapshot B wins, 0.5 for a tie.
        """
        snapA, ratingA, stepA = self.snapshots[idxA]
        snapB, ratingB, stepB = self.snapshots[idxB]
        expectedA = 1 / (1 + 10 ** ((ratingB - ratingA) / 400))
        expectedB = 1 - expectedA

        newA = ratingA + k_factor * (result - expectedA)
        newB = ratingB + k_factor * ((1 - result) - expectedB)

        self.snapshots[idxA] = (snapA, newA, stepA)
        self.snapshots[idxB] = (snapB, newB, stepB)
        print(f"[OPPONENT POOL] Updated ratings: Snapshot {idxA} -> {newA}, Snapshot {idxB} -> {newB}")
