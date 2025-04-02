import json
import os
import numpy as np
import random
import gymnasium as gym

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor
from stable_baselines3.common.utils import set_random_seed

# Import your environment from its file
from bytefight_env_single import SingleProcessByteFightEnv

def make_env(map_string, opponent_module, submission_dir, rank=0):
    """
    Helper function to create and return a single env instance.
    """
    def _init():
        env = SingleProcessByteFightEnv(
            map_string=map_string,
            opponent_module=opponent_module,
            submission_dir=submission_dir,
            render_mode=None
        )
        return env
    
    set_random_seed(rank)
    return _init

def main():
     # 1) Load map from maps.json
    with open("maps.json") as f:
        map_dict = json.load(f)
    # Using the "empty" map for simplicity
    map_string = map_dict["empty"]
    
    # 2) Opponent code - make sure this exists in the specified path
    opponent_module = "sample_player"
    submission_dir = "../../../workspace"
    
    # 3) Load the pretrained model
    model_path = "bytefight_logs/ppo_bytefight_final.zip"
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
        return
        
    print(f"Loading model from {model_path}")
    model = PPO.load(model_path)
    print("Model loaded successfully!")
    
    # 10) Run a few test episodes
    test_env = SingleProcessByteFightEnv(
        map_string=map_string,
        opponent_module=opponent_module,
        submission_dir=submission_dir,
        render_mode="human"
    )
    
    for episode in range(5):
        obs, _ = test_env.reset()
        done = False
        total_reward = 0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = test_env.step(action)
            total_reward += reward
        
        print(f"Episode {episode+1}: winner={info.get('winner')}, total_reward={total_reward:.2f}, turn_count={info.get('turn_count')}\n")
    test_env.close()

if __name__ == "__main__":
    main()