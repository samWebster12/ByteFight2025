import os
import random
import torch
import numpy as np
import json  # We'll use JSON to save ratings/metadata.
from gymnasium import spaces
from custom_policy3 import ByteFightMaskedPolicy, ByteFightFeaturesExtractor

from controllers.custom1_controller import PlayerControllerCustom1
from controllers.custom2_controller import PlayerControllerCustom2
from controllers.lect_controller import PlayerControllerLect
from controllers.mcts1_controller import PlayerControllerMCTS1
from controllers.mcts3controller import PlayerControllerMCTS3
from controllers.minimax_controller import PlayerControllerMinimax
from controllers.sample_controller import PlayerControllerSample

# Build a list of safe globals for controller unpickling.
safe_globals_list = [
    PlayerControllerLect,
    PlayerControllerSample,
    PlayerControllerCustom1,
    PlayerControllerCustom2,
    PlayerControllerMCTS1,
    PlayerControllerMCTS3,
    PlayerControllerMinimax
]


class OpponentPool:
    def __init__(self, snapshot_dir, initial_policy=None, initial_rating=1000):
        """
        snapshot_dir: directory in which to save snapshot files and metadata.
        initial_policy: if provided, a policy snapshot is added.
        """
        self.snapshot_dir = snapshot_dir
        os.makedirs(snapshot_dir, exist_ok=True)
        # snapshots_meta will be a list of dicts with keys:
        # "id", "type" (either "policy" or "controller"), "rating", "step", "filename"
        self.snapshots_meta = []
        self.metadata_file = os.path.join(self.snapshot_dir, "ratings.json")
        self._load_metadata()
        # If there is an initial policy and no snapshots have been loaded, add it.
        if initial_policy is not None and not self.snapshots_meta:
            self.add_policy_snapshot(initial_policy, rating=initial_rating, step=0)

    def add_policy_snapshot(self, main_policy, rating=1000, step=0):
        """
        Save a deep RL policy snapshot.
        We construct a new ByteFightMaskedPolicy, save its state dict to a .pt file,
        and record metadata with type "policy".
        """
        snapshot_id = len(self.snapshots_meta)
        filename = f"snapshot_{snapshot_id}.pt"
        filepath = os.path.join(self.snapshot_dir, filename)
        # Save the state dict of main_policy to disk.
        torch.save(main_policy.state_dict(), filepath)
        # Record metadata.
        meta = {
            "id": snapshot_id,
            "type": "policy",
            "rating": rating,
            "step": step,
            "filename": filename
        }
        self.snapshots_meta.append(meta)
        self._save_metadata()
        print(f"[OPPONENT POOL] Added policy snapshot id {snapshot_id} at step {step} with rating {rating}")

    def add_controller_snapshot(self, controller_obj, rating=1000, step=0):
        """
        Save a snapshot of a controller object (which is not necessarily a deep RL policy)
        by pickling the entire object.
        """
        snapshot_id = len(self.snapshots_meta)
        filename = f"controller_snapshot_{snapshot_id}.pt"
        filepath = os.path.join(self.snapshot_dir, filename)
        # Save the entire controller object to disk.
        # (This pickles the object including its class reference.)
        torch.save(controller_obj, filepath)
        meta = {
            "id": snapshot_id,
            "type": "controller",
            "rating": rating,
            "step": step,
            "filename": filename
        }
        self.snapshots_meta.append(meta)
        self._save_metadata()
        print(f"[OPPONENT POOL] Added controller snapshot id {snapshot_id} at step {step} with rating {rating}")

    def _save_metadata(self):
        with open(self.metadata_file, "w") as f:
            json.dump(self.snapshots_meta, f)
        # Uncomment the next line if you want to log metadata saves.
        # print(f"[OPPONENT POOL] Metadata saved to {self.metadata_file}")

    def _load_metadata(self):
        if os.path.exists(self.metadata_file):
            with open(self.metadata_file, "r") as f:
                self.snapshots_meta = json.load(f)
            print(f"[OPPONENT POOL] Loaded metadata from {self.metadata_file}")
        else:
            self.snapshots_meta = []

    def sample_opponent(self, main_rating=None, exclude_id=None):
        """
        Sample an opponent snapshot based on ratings.
        Optionally, exclude a specific snapshot id.
        Returns a controller object and its snapshot id.
        For type "policy", we load and reconstruct a ByteFightMaskedPolicy.
        For type "controller", we load the controller using torch.load (or pickle.load).
        """
        self._load_metadata()
        if not self.snapshots_meta:
            raise ValueError("Opponent pool is empty!")

        available_ids = [meta["id"] for meta in self.snapshots_meta]
        if exclude_id is not None and len(available_ids) > 1:
            if exclude_id in available_ids:
                available_ids.remove(exclude_id)

        if main_rating is not None:
            tau = 350  # sensitivity parameter; adjust as needed
            ratings = np.array([meta["rating"] for meta in self.snapshots_meta if meta["id"] in available_ids])
            weights = np.exp(-np.abs(ratings - main_rating) / tau)
            weights = weights / np.sum(weights)
            #print("[OPPONENT POOL DEBUG] Sampling opponents (weighted):")
            #for meta, w in zip([m for m in self.snapshots_meta if m["id"] in available_ids], weights):
                #print(f"  Snapshot id {meta['id']}: type = {meta['type']}, rating = {meta['rating']}, probability = {w:.4f}")
            chosen_id = int(np.random.choice(available_ids, p=weights))
        else:
            print("[OPPONENT POOL DEBUG] No main rating provided, selecting randomly.")
            chosen_id = random.choice(available_ids)

        chosen_meta = next(meta for meta in self.snapshots_meta if meta["id"] == chosen_id)
        filepath = os.path.join(self.snapshot_dir, chosen_meta["filename"])
        # Depending on type, load the snapshot accordingly.
        if chosen_meta["type"] == "policy":
            # Reconstruct a new ByteFightMaskedPolicy.
            dummy_obs_space = spaces.Dict({
                "board_image": spaces.Box(low=0, high=1, shape=(9, 64, 64), dtype=float),
                "features": spaces.Box(low=-1e6, high=1e6, shape=(15,), dtype=float),
                "action_mask": spaces.Box(low=0, high=1, shape=(10,), dtype=int),
            })
            dummy_act_space = spaces.Discrete(10)
            policy_kwargs = {
                "net_arch": dict(pi=[256, 128], vf=[256, 128]),
                "activation_fn": torch.nn.ReLU,
                "features_extractor_class": ByteFightFeaturesExtractor,
                "features_extractor_kwargs": {"features_dim": 256},
            }
            def dummy_lr_schedule(_):
                return 3e-4
            controller = ByteFightMaskedPolicy(
                observation_space=dummy_obs_space,
                action_space=dummy_act_space,
                lr_schedule=dummy_lr_schedule,
                **policy_kwargs
            )
            state_dict = torch.load(filepath)
            controller.load_state_dict(state_dict)
        elif chosen_meta["type"] == "controller":
            # For a controller snapshot, we assume it was saved (e.g., via torch.save or pickle)
            # and we load it directly.
            with torch.serialization.safe_globals(safe_globals_list):
                controller = torch.load(filepath, map_location=torch.device('cpu'), weights_only=False)
        else:
            raise ValueError(f"Unknown snapshot type: {chosen_meta['type']}")

        print(f"[OPPONENT POOL DEBUG] Selected snapshot id {chosen_id}: type = {chosen_meta['type']}, rating = {chosen_meta['rating']}, step = {chosen_meta['step']}")
        return controller, chosen_id, chosen_meta["type"]

    def update_rating(self, idxA, idxB, result, k_factor=32):
        """
        Basic ELO update.
        idxA and idxB are snapshot ids (the "id" field in metadata).
        result: 1.0 if snapshot A wins, 0.0 if snapshot B wins, 0.5 for a tie.
        Updates the ratings in metadata and saves the file.
        """
        self._load_metadata()
        metaA = next((m for m in self.snapshots_meta if m["id"] == idxA), None)
        metaB = next((m for m in self.snapshots_meta if m["id"] == idxB), None)
        if metaA is None or metaB is None:
            print(f"[OPPONENT POOL] Could not find metadata for snapshot ids: {idxA}, {idxB}")
            return
        ratingA = metaA["rating"]
        ratingB = metaB["rating"]
        expectedA = 1 / (1 + 10 ** ((ratingB - ratingA) / 400))
        expectedB = 1 - expectedA
        newA = ratingA + k_factor * (result - expectedA)
        newB = ratingB + k_factor * ((1 - result) - expectedB)
        metaA["rating"] = newA
        metaB["rating"] = newB
        self._save_metadata()
        print(f"[OPPONENT POOL] Updated ratings: Snapshot {idxA} -> {newA}, Snapshot {idxB} -> {newB}")
