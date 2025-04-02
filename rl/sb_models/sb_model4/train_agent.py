import json
import os
import numpy as np
import random
import gymnasium as gym
import torch.nn as nn

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback, BaseCallback
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.env_checker import check_env

# Import your environment and custom model
from bytefight_env_single import SingleProcessByteFightEnv
from sb_arch_10M import SnakeCNN, MaskedActionWrapper

# Attempt to import tensorboard
try:
    import tensorboard
    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False

class PrintCallback(BaseCallback):
    """
    Callback to print training progress after episodes complete.
    """
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_count = 0
        self.episode_rewards = []
        self.win_count = 0
        self.lose_count = 0
        self.tie_count = 0
        
    def _on_step(self):
        # Process episode completions
        for i, done in enumerate(self.locals["dones"]):
            if done:
                info = self.locals["infos"][i]
                reward = self.locals["rewards"][i]
                
                self.episode_rewards.append(reward)
                self.episode_count += 1
                
                winner = info.get("winner")
                if winner == "PLAYER_A":
                    self.win_count += 1
                elif winner == "PLAYER_B":
                    self.lose_count += 1
                elif winner == "TIE":
                    self.tie_count += 1
                
                # Print statistics every 10 episodes
                if self.episode_count % 10 == 0:
                    last_10 = self.episode_rewards[-10:]
                    avg_reward = sum(last_10) / len(last_10)
                    win_rate = (self.win_count / self.episode_count) * 100
                    #print(f"Episode {self.episode_count}, Avg Reward: {avg_reward:.2f}, Win Rate: {win_rate:.1f}%")
                    #print(f"Wins: {self.win_count}, Losses: {self.lose_count}, Ties: {self.tie_count}")
                    #print("-"*40)
        return True

def make_env(map_string, opponent_module, submission_dir, rank=0):
    """
    Create a single environment instance with action masking.
    """
    def _init():
        env = SingleProcessByteFightEnv(
            map_string=map_string,
            opponent_module=opponent_module,
            submission_dir=submission_dir,
            render_mode=None
        )
        # Apply the action masking wrapper
        env = MaskedActionWrapper(env)
        return env
    
    set_random_seed(rank)
    return _init

def linear_schedule(initial_value):
    """
    Linear learning rate schedule.
    """
    def schedule(progress_remaining):
        return progress_remaining * initial_value
    return schedule

def main():
    # 1) Load map
    with open("maps.json") as f:
        map_dict = json.load(f)
    map_string = map_dict["empty"]
    
    # 2) Opponent code
    opponent_module = "sample_player"
    submission_dir = "../../../workspace"
    
    # 3) Logging directory
    log_dir = "bytefight_logs"
    os.makedirs(log_dir, exist_ok=True)
    
    # 4) Create parallel environments
    n_envs = 8  # Increased from 4 to 8 for more diverse experience
    env_fns = [make_env(map_string, opponent_module, submission_dir, rank=i) for i in range(n_envs)]
    env = SubprocVecEnv(env_fns)
    env = VecMonitor(env)
    
    # 5) Create separate eval environment
    eval_env = make_env(map_string, opponent_module, submission_dir)()
    
    # 6) Setup callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=50000 // n_envs,
        save_path=f"{log_dir}/checkpoints/",
        name_prefix="ppo_bytefight",
        save_vecnormalize=True
    )
    
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=f"{log_dir}/best_model/",
        log_path=f"{log_dir}/eval_results/",
        eval_freq=25000 // n_envs,
        deterministic=True,
        render=False,
        n_eval_episodes=10  # Increased from 5 to 10
    )
    
    print_callback = PrintCallback()
    
    # 7) TensorBoard logging
    if TENSORBOARD_AVAILABLE:
        tensorboard_log = f"{log_dir}/tensorboard/"
    else:
        print("TensorBoard not installed. Disabling tensorboard_log.")
        tensorboard_log = None
    
    # 8) Configure network architecture with our custom CNN
    policy_kwargs = {
        "net_arch": dict(
                pi=[256, 128],  # Policy network architecture  
                vf=[256, 128]   # Value function network architecture
            ),
            "features_extractor_class": SnakeCNN,
            "features_extractor_kwargs": {
                "features_dim": 256  # Changed from 512 to 256 to match our new model
            }
    }
    
    # 9) Create PPO model with improved hyperparameters
    model = PPO(
        policy="MultiInputPolicy",
        env=env,
        learning_rate=linear_schedule(3e-4),  # Learning rate with decay
        n_steps=2048,
        batch_size=128,  # Increased batch size for better learning
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        policy_kwargs=policy_kwargs,
        verbose=1,
        device="cuda",
        tensorboard_log=tensorboard_log
    )
    
    # 10) Train the model with increased timesteps
    total_timesteps = 300000  # Increased from 300000
    model.learn(
        total_timesteps=total_timesteps,
        callback=[checkpoint_callback, eval_callback, print_callback],
        progress_bar=True
    )
    
    # 11) Save final model
    final_model_path = f"{log_dir}/ppo_bytefight_final"
    model.save(final_model_path)
    print("Final model saved to", final_model_path)
    
    # 12) Test the trained model
    test_env = SingleProcessByteFightEnv(
        map_string=map_string,
        opponent_module=opponent_module,
        submission_dir=submission_dir,
        render_mode="human"
    )
    test_env = MaskedActionWrapper(test_env)
    
    print("\nEvaluating final model:")
    for episode in range(5):  # Increased from 3 to 5
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