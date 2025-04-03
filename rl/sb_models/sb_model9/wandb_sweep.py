import os
import time
import numpy as np
import torch
import argparse
import json
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.atari_wrappers import ClipRewardEnv
import gymnasium as gym

# Import your custom components
from bytefight_env import ByteFightSnakeEnv
from custom_policy3 import ByteFightMaskedPolicy, ByteFightFeaturesExtractor
from opp_controller import OppController

# Import wandb and its SB3 callback
import wandb
from wandb.integration.sb3 import WandbCallback

# Setup training parameters defaults (these will be overridden by wandb.config)
RANDOM_SEED = 42
NUM_ENV = 8         # Number of environments for parallel training
TOTAL_TIMESTEPS = 300_000
SAVE_FREQ = 50_000  # Save model every 50k steps
EVAL_FREQ = 20_000  # Evaluate model every 20k steps
LOGS_DIR = "./logs"
MODELS_DIR = "./models"
LOG_LEVEL = 1       # 0: No output, 1: Info, 2: Debug

# Create required directories
os.makedirs(LOGS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(f"{LOGS_DIR}/eval", exist_ok=True)

import os, sys
parent_dir = os.path.abspath(os.path.join(__file__, "../../../.."))
sys.path.insert(0, parent_dir)

from game.game_map import Map

# List of all available maps
AVAILABLE_MAP_NAMES = [
    "pillars", "great_divide", "cage", "empty", "empty_large", "ssspline",
    "combustible_lemons", "arena", "ladder", "compasss", "recurve",
    "diamonds", "ssspiral", "lol", "attrition"
]

def get_map_string(map_name):
    if not os.path.exists("maps.json"):
        raise FileNotFoundError("maps.json file not found. Please make sure it exists in the current directory.")
    with open("maps.json", "r") as f:
        maps = json.load(f)
    if map_name not in maps:
        available_maps = ", ".join(maps.keys())
        raise KeyError(f"Map '{map_name}' not found in maps.json. Available maps: {available_maps}")
    return maps[map_name]

def make_bytefight_env(rank, seed=0):
    """
    Create a ByteFight environment.
    Randomly select a map from AVAILABLE_MAP_NAMES for each environment instance.
    """
    def _init():
        # Create opponent controller
        dummy_data = 1
        opponent = OppController(dummy_data)
        # Create the environment (use_opponent flag can be set as desired)
        env = ByteFightSnakeEnv(AVAILABLE_MAP_NAMES, opponent, render_mode=None, use_opponent=False)
        # Wrap environment with Monitor for statistics
        env = Monitor(env, f"{LOGS_DIR}/env_{rank}")
        # Set environment seed
        env.reset(seed=seed + rank)
        return env
    set_random_seed(seed)
    return _init

if __name__ == "__main__":
    # Initialize wandb run (this will be overwritten by sweep values)
    wandb.init(project="bytefight-sweep", config={
        "learning_rate": 3e-4,
        "n_steps": 2048,
        "batch_size": 64,
        "n_epochs": 10,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "clip_range": 0.2,
        "ent_coef": 0.01,
        "vf_coef": 0.5,
        "max_grad_norm": 0.5,
        
    })
    config = wandb.config

    print(f"Creating {NUM_ENV} environments with random maps from AVAILABLE_MAP_NAMES...")
    vec_env = SubprocVecEnv([make_bytefight_env(i, RANDOM_SEED) for i in range(NUM_ENV)])
    vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=True, clip_obs=10, clip_reward=3)

    # Create evaluation environment similarly
    eval_opponent = OppController(1)
    eval_env = ByteFightSnakeEnv(
        map_names=AVAILABLE_MAP_NAMES,
        opponent_controller=eval_opponent,
        render_mode=None,
        verbose=False,
        use_opponent=False
    )
    eval_env = Monitor(eval_env, f"{LOGS_DIR}/eval_env")

    checkpoint_callback = CheckpointCallback(
        save_freq=SAVE_FREQ // NUM_ENV,
        save_path=MODELS_DIR,
        name_prefix="bytefight_ppo_wandb",
        save_replay_buffer=True,
        save_vecnormalize=True,
    )

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=f"{MODELS_DIR}/best_wandb",
        log_path=f"{LOGS_DIR}/eval",
        eval_freq=EVAL_FREQ // NUM_ENV,
        deterministic=True,
        render=False,
        n_eval_episodes=10
    )

    policy_kwargs = {
        "net_arch": dict(pi=[256, 128], vf=[256, 128]),
        "activation_fn": torch.nn.ReLU,
        "features_extractor_class": ByteFightFeaturesExtractor,
        "features_extractor_kwargs": {
            "features_dim": 256
        }
    }

    print("Creating PPO agent...")
    model = PPO(
        policy=ByteFightMaskedPolicy,
        env=vec_env,
        learning_rate=config.learning_rate,
        n_steps=config.n_steps,
        batch_size=config.batch_size,
        n_epochs=config.n_epochs,
        gamma=config.gamma,
        gae_lambda=config.gae_lambda,
        clip_range=config.clip_range,
        normalize_advantage=True,
        ent_coef=config.ent_coef,
        vf_coef=config.vf_coef,
        max_grad_norm=config.max_grad_norm,
        use_sde=False,
        sde_sample_freq=-1,
        target_kl=None,
        tensorboard_log=LOGS_DIR,
        policy_kwargs=policy_kwargs,
        verbose=LOG_LEVEL,
        seed=RANDOM_SEED,
        device="auto"
    )

    print(f"Starting training for {TOTAL_TIMESTEPS} timesteps...")
    start_time = time.time()

    checkpoint_path = f"{MODELS_DIR}/checkpoint.zip"
    if os.path.exists(checkpoint_path):
        print(f"Loading model from checkpoint: {checkpoint_path}")
        model = PPO.load(
            checkpoint_path,
            env=vec_env,
            custom_objects={
                "policy_class": ByteFightMaskedPolicy,
                "features_extractor_class": ByteFightFeaturesExtractor
            }
        )
        print("Checkpoint loaded successfully!")

    model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        callback=[checkpoint_callback, eval_callback, WandbCallback()],
        reset_num_timesteps=True,
        progress_bar=True
    )

    final_model_path = f"{MODELS_DIR}/bytefight_ppo_final"
    model.save(final_model_path)
    print(f"Training completed in {time.time() - start_time:.2f} seconds")
    print(f"Final model saved to: {final_model_path}")

    vec_env.close()
    eval_env.close()
