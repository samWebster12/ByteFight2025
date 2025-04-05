import os
import random
import torch
import numpy as np
import json  # We'll use JSON to save ratings/metadata.
from gymnasium import spaces
from custom_policy3 import ByteFightMaskedPolicy, ByteFightFeaturesExtractor

class OpponentPool:
    def __init__(self, snapshot_dir, initial_policy=None, initial_rating=1000):
        self.snapshot_dir = snapshot_dir
        os.makedirs(snapshot_dir, exist_ok=True)
        # This list holds metadata for each snapshot. Each element is a dict with keys:
        # "id", "rating", "step", "filename"
        self.snapshots_meta = []
        self.metadata_file = os.path.join(self.snapshot_dir, "ratings.json")
        # If there is already a metadata file, load it.
        self._load_metadata()
        # Optionally, add an initial policy.
        if initial_policy is not None and not self.snapshots_meta:
            self.add_snapshot(initial_policy, rating=initial_rating, step=0)

    def add_snapshot(self, main_policy, rating=1000, step=0):
        """
        Save a snapshot of the current main_policy (a SB3 policy object) to disk.
        A new ByteFightMaskedPolicy is constructed and its state_dict is saved to a .pt file.
        The metadata (unique id, rating, step, filename) is appended to snapshots_meta and saved.
        """
        # Create a unique snapshot id using the next available integer.
        snapshot_id = len(self.snapshots_meta)
        filename = f"snapshot_{snapshot_id}.pt"
        filepath = os.path.join(self.snapshot_dir, filename)
        # Save the state dict of main_policy to disk.
        torch.save(main_policy.state_dict(), filepath)
        # Append metadata.
        meta = {"id": snapshot_id, "rating": rating, "step": step, "filename": filename}
        self.snapshots_meta.append(meta)
        self._save_metadata()
        print(f"[OPPONENT POOL] Added snapshot id {snapshot_id} at step {step} with rating {rating}")

    def _save_metadata(self):
        """
        Save the current snapshots_meta list to the metadata JSON file.
        """
        with open(self.metadata_file, "w") as f:
            json.dump(self.snapshots_meta, f)
        #print(f"[OPPONENT POOL] Metadata saved to {self.metadata_file}")

    def _load_metadata(self):
        """
        Load snapshots_meta from the metadata JSON file, if it exists.
        """
        if os.path.exists(self.metadata_file):
            with open(self.metadata_file, "r") as f:
                self.snapshots_meta = json.load(f)
            #print(f"[OPPONENT POOL] Loaded metadata from {self.metadata_file}")
        else:
            self.snapshots_meta = []

    def sample_opponent(self, main_rating=None, exclude_id=None):
        """
        Sample an opponent from the pool.
        - If main_rating is provided, weight the snapshots so that those with ratings
          closer to main_rating are more likely to be chosen.
        - If exclude_id is provided (and there’s more than one snapshot), that snapshot
          is excluded from the candidate pool.
        Returns a freshly constructed policy object (loaded from the saved .pt file)
        and the chosen snapshot id.
        """
        # Reload the metadata from disk.
        self._load_metadata()
        if not self.snapshots_meta:
            raise ValueError("Opponent pool is empty!")

        # Build a list of available snapshot ids.
        available_ids = [meta["id"] for meta in self.snapshots_meta]
      #  print("[OPPONENT POOL DEBUG] Initial available snapshot IDs:", available_ids)
        if exclude_id is not None and len(available_ids) > 1:
            if exclude_id in available_ids:
       #         print(f"[OPPONENT POOL DEBUG] Excluding snapshot id: {exclude_id}")
                available_ids.remove(exclude_id)
        #print("[OPPONENT POOL DEBUG] Final available IDs after exclusion:", available_ids)

        if main_rating is not None:
            tau = 50  # sensitivity parameter; adjust as needed
            # Get ratings for available snapshots.
            ratings = np.array([meta["rating"] for meta in self.snapshots_meta if meta["id"] in available_ids])
            weights = np.exp(-np.abs(ratings - main_rating) / tau)
            weights = weights / np.sum(weights)
            #print("[OPPONENT POOL DEBUG] Sampling opponents (weighted):")
         #   for meta, w in zip([meta for meta in self.snapshots_meta if meta["id"] in available_ids], weights):
          #      print(f"  Snapshot id {meta['id']}: rating = {meta['rating']}, probability = {w:.4f}")
            chosen_id = int(np.random.choice(available_ids, p=weights))
        else:
            print("[OPPONENT POOL DEBUG] No main rating provided, selecting randomly.")
            chosen_id = random.choice(available_ids)

        # Find the metadata entry for the chosen id.
        chosen_meta = next(meta for meta in self.snapshots_meta if meta["id"] == chosen_id)
        filepath = os.path.join(self.snapshot_dir, chosen_meta["filename"])
        # Load the saved state_dict.
        state_dict = torch.load(filepath)
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
        opp_policy = ByteFightMaskedPolicy(
            observation_space=dummy_obs_space,
            action_space=dummy_act_space,
            lr_schedule=dummy_lr_schedule,
            **policy_kwargs
        )
        opp_policy.load_state_dict(state_dict)
        print(f"[OPPONENT POOL DEBUG] Selected snapshot id {chosen_id}: rating = {chosen_meta['rating']}, step = {chosen_meta['step']}")
        return opp_policy, chosen_id

    def update_rating(self, idxA, idxB, result, k_factor=32):
        """
        Basic ELO update.
          - idxA and idxB are the snapshot ids (the "id" field in metadata).
          - result: 1.0 if snapshot A wins, 0.0 if snapshot B wins, 0.5 for a tie.
        Updates the ratings in the metadata and saves the file.
        """
        self._load_metadata()
        metaA = next((m for m in self.snapshots_meta if m["id"] == idxA), None)
        metaB = next((m for m in self.snapshots_meta if m["id"] == idxB), None)
        if metaA is None or metaB is None:
            print(f"[OPPONENT POOL] Could not find metadata for given snapshot ids: {idxA}, {idxB}.")
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
