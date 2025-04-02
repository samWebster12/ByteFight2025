import os
import time
import numpy as np
import torch
import argparse
import json
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.env_util import make_vec_env
import gymnasium as gym

# Import your custom components
from bytefight_env import ByteFightSnakeEnv
from custom_policy2 import ImprovedByteFightMaskedPolicy, ImprovedByteFightFeaturesExtractor
from opp_controller import OppController

# Setup training parameters
RANDOM_SEED = 42
NUM_ENV = 8         # Number of environments for parallel training
TOTAL_TIMESTEPS = 1_600_000
SAVE_FREQ = 50_000  # Save model every 100k steps
EVAL_FREQ = 20_000   # Evaluate model every 20k steps
LOGS_DIR = "./logs"
MODELS_DIR = "./models"
LOG_LEVEL = 1      # 0: No output, 1: Info, 2: Debug

# Create required directories
os.makedirs(LOGS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(f"{LOGS_DIR}/eval", exist_ok=True)

import os, sys
parent_dir = os.path.abspath(os.path.join(__file__, "../../../.."))
sys.path.insert(0, parent_dir)

# ByteFight imports
from game.game_map import Map


def get_map_string(map_name):
    if not os.path.exists("maps.json"):
        raise FileNotFoundError("maps.json file not found. Please make sure it exists in the current directory.")
    
    with open("maps.json", "r") as f:
        maps = json.load(f)

    if map_name not in maps:
        available_maps = ", ".join(maps.keys())
        raise KeyError(f"Map '{map_name}' not found in maps.json. Available maps: {available_maps}")
    
    return maps[map_name]

# Define environment creation function
def make_bytefight_env(rank, seed=0, map_name="empty"):
    """
    Create a ByteFight environment with a random opponent, 
    loading the requested map from maps.json with no fallback.
    """
    def _init():
        map_string = get_map_string(map_name)
        print(f"Loaded map '{map_name}' for environment {rank}")

        # Create Map, Board, and PlayerBoard
        game_map = Map(map_string)

        # Create opponent controller
        dummy_data = 1
        opponent = OppController(dummy_data)

        # Create the environment
        env = ByteFightSnakeEnv(game_map, opponent, render_mode=None)

        # Wrap environment with Monitor for statistics
        env = Monitor(env, f"{LOGS_DIR}/env_{rank}")

        # Set environment seed
        env.reset(seed=seed + rank)
        return env

    set_random_seed(seed)
    return _init

if __name__ == "__main__":
    # Set the map to use for training
    MAP_NAME = "empty"  # We'll load the "empty" map from maps.json

    # Create vectorized environment
    print(f"Creating environments with map '{MAP_NAME}'...")
    vec_env = SubprocVecEnv([make_bytefight_env(i, RANDOM_SEED, MAP_NAME) for i in range(NUM_ENV)])

    # Create separate environment for evaluation
    print(f"Creating evaluation environment with map '{MAP_NAME}'...")
    # MODIFIED: No fallback; we expect "empty" to be in maps.json
    eval_map_string = get_map_string(MAP_NAME)
    eval_game_map = Map(eval_map_string)
    print(eval_game_map.dim_x, eval_game_map.dim_y)
    eval_opponent = OppController(1)

    eval_env = ByteFightSnakeEnv(
        game_map=eval_game_map,
        opponent_controller=eval_opponent,
        render_mode=None,
    )
    eval_env = Monitor(eval_env, f"{LOGS_DIR}/eval_env")

    # Setup callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=SAVE_FREQ // NUM_ENV,  # Divide by number of environments
        save_path=MODELS_DIR,
        name_prefix="bytefight_ppo",
        save_replay_buffer=True,
        save_vecnormalize=True,
    )

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=f"{MODELS_DIR}/best",
        log_path=f"{LOGS_DIR}/eval",
        eval_freq=EVAL_FREQ // NUM_ENV,  # Divide by number of environments
        deterministic=True,
        render=False,
        n_eval_episodes=10
    )

    # Define PPO policy kwargs for our improved feature extractor
    policy_kwargs = {
        "net_arch": dict(pi=[256, 128], vf=[256, 128]),  # Fixed net_arch format
        "activation_fn": torch.nn.ReLU,
        "features_extractor_class": ImprovedByteFightFeaturesExtractor,
        "features_extractor_kwargs": {
            "features_dim": 384  # This is what the improved extractor expects
        }
    }

    # Create PPO agent with custom policy
    print("Creating PPO agent...")
    model = PPO(
        policy=ImprovedByteFightMaskedPolicy,
        env=vec_env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        clip_range_vf=None,
        normalize_advantage=True,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        use_sde=False,
        sde_sample_freq=-1,
        target_kl=None,
        tensorboard_log=LOGS_DIR,
        policy_kwargs=policy_kwargs,
        verbose=LOG_LEVEL,
        seed=RANDOM_SEED,
        device="auto"  # Will use CUDA if available
    )

    # Train the agent
    print(f"Starting training for {TOTAL_TIMESTEPS} steps...")
    start_time = time.time()

    # Try to load from checkpoint if it exists
    checkpoint_path = f"{MODELS_DIR}/checkpoint.zip"
    if os.path.exists(checkpoint_path):
        print(f"Loading model from checkpoint: {checkpoint_path}")
        model = PPO.load(
            checkpoint_path,
            env=vec_env,
            custom_objects={
                "policy_class": ImprovedByteFightMaskedPolicy,
                "features_extractor_class": ImprovedByteFightFeaturesExtractor
            }
        )
        print("Checkpoint loaded successfully!")

    # Start or continue training
    model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        callback=[checkpoint_callback, eval_callback],
        reset_num_timesteps=True,
        progress_bar=True
    )

    # Save final model
    final_model_path = f"{MODELS_DIR}/bytefight_ppo_final2"
    model.save(final_model_path)
    print(f"Training completed in {time.time() - start_time:.2f} seconds")
    print(f"Final model saved to: {final_model_path}")

    # Close environments
    vec_env.close()
    eval_env.close()