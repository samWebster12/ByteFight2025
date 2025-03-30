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
    
    # 3) Logging directory
    log_dir = "bytefight_logs"
    os.makedirs(log_dir, exist_ok=True)
    
    # 4) Create environment (using just a single environment for simplicity)
    env_fn = make_env(map_string, opponent_module, submission_dir)
    env = DummyVecEnv([env_fn])
    env = VecMonitor(env)
    
    # 5) Setup checkpoint callback
    checkpoint_callback = CheckpointCallback(
        save_freq=10000,
        save_path=f"{log_dir}/checkpoints/",
        name_prefix="ppo_bytefight"
    )
    
    # 6) Create a simpler network architecture
    policy_kwargs = {
        "net_arch": dict(pi=[64, 64], vf=[64, 64])  # Smaller network
    }
    
    # 7) Create PPO model with simpler parameters
    model = PPO(
        policy="MultiInputPolicy",
        env=env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        verbose=1,
        tensorboard_log=f"{log_dir}/tensorboard/" if os.path.exists("/usr/bin/tensorboard") else None,
        policy_kwargs=policy_kwargs
    )
    
    # 8) Train the model
    total_timesteps = 100000  # Reduced number of steps
    model.learn(
        total_timesteps=total_timesteps,
        callback=checkpoint_callback,
        progress_bar=True
    )
    
    # 9) Save final model
    final_model_path = f"{log_dir}/ppo_bytefight_final"
    model.save(final_model_path)
    print("Final model saved to", final_model_path)
    
    # 10) Run a few test episodes
    test_env = SingleProcessByteFightEnv(
        map_string=map_string,
        opponent_module=opponent_module,
        submission_dir=submission_dir,
        render_mode="human"
    )
    
    for episode in range(3):
        obs, _ = test_env.reset()
        done = False
        total_reward = 0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = test_env.step(action)
            total_reward += reward
        
        print(f"Episode {episode+1}: winner={info.get('winner')}, total_reward={total_reward:.2f}, turn_count={info.get('turn_count')}")
    test_env.close()

if __name__ == "__main__":
    main()