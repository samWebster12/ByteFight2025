import json
import os
import numpy as np
import random
import gymnasium as gym
import torch.nn as nn
import glob
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback, BaseCallback
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.env_checker import check_env

# Import your environment and custom model
from bytefight_env_single import SingleProcessByteFightEnv
from sb_arch_10M import SnakeCNN
from wrappers import OpponentWrapper, MaskedActionWrapper  # Import from the separate file

from schedules import linear_schedule_fn
from functools import partial

class SelfPlayCallback(BaseCallback):
    """
    Callback for implementing self-play during training.
    """
    def __init__(self, eval_env, save_path="./opponents", eval_freq=10000, n_eval_episodes=5, verbose=1):
        super().__init__(verbose)
        self.eval_env = eval_env
        self.save_path = save_path
        self.eval_freq = eval_freq
        self.n_eval_episodes = n_eval_episodes
        self.best_mean_reward = -np.inf
        self.current_opponent = None
        os.makedirs(save_path, exist_ok=True)
    
    def _on_step(self):
        if self.num_timesteps % self.eval_freq == 0:
            # Save current model as potential opponent
            model_path = os.path.join(self.save_path, f"opponent_{self.num_timesteps}")
            self.model.save(model_path)
            
            # Evaluate if this model is better than the current best
            mean_reward = self._evaluate_model()
            
            if mean_reward > self.best_mean_reward:
                print(f"New best model with reward {mean_reward:.2f} > {self.best_mean_reward:.2f}")
                self.best_mean_reward = mean_reward
                
                # Save as best model
                best_path = os.path.join(self.save_path, "best_model")
                self.model.save(best_path)
                
                # Update opponent in evaluation environment
                self.current_opponent = None  # Clear references
                if hasattr(self.eval_env, "update_opponent"):
                    self.eval_env.update_opponent(None)  # Clear existing model
                    self.current_opponent = PPO.load(best_path)
                    self.eval_env.update_opponent(self.current_opponent)
                
                # Tell worker environments to load the model from the path
                try:
                    if hasattr(self.model.env, "env_method"):
                        # Pass the path instead of the model
                        self.model.env.env_method("load_opponent_from_path", best_path)
                except Exception as e:
                    print(f"Error updating opponent in training envs: {e}")
                
                print(f"Updated opponent model to timestep {self.num_timesteps}")
        
        return True
    
    def _evaluate_model(self):
        """
        Evaluate model performance against current best opponent.
        """
        total_reward = 0
        for _ in range(self.n_eval_episodes):
            obs, _ = self.eval_env.reset()
            done = False
            episode_reward = 0
            while not done:
                action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, done, _, _ = self.eval_env.step(action)
                episode_reward += reward
            total_reward += episode_reward
        
        mean_reward = total_reward / self.n_eval_episodes
        print(f"Evaluation: mean reward = {mean_reward:.2f} over {self.n_eval_episodes} episodes")
        return mean_reward


def make_env(map_string, submission_dir, opponent_model=None, rank=0):
    """
    Create a single environment instance with self-play capability.
    """
    def _init():
        env = SingleProcessByteFightEnv(
            map_string=map_string,
            opponent_module="sample_player",  # Fallback opponent
            submission_dir=submission_dir,
            render_mode=None
        )
        # Wrap with opponent model
        env = OpponentWrapper(env, opponent_model)
        # Apply action masking
        env = MaskedActionWrapper(env)
        return env
    
    set_random_seed(rank)
    return _init


def linear_schedule_fn(progress_remaining, initial_value):
    """
    Global function for linear learning rate schedule.
    """
    return progress_remaining * initial_value



def main():
    # 1) Load map
    with open("maps.json") as f:
        map_dict = json.load(f)
    map_string = map_dict["empty"]
    
    # 2) Submission directory
    submission_dir = "../../../workspace"
    
    # 3) Logging directory
    log_dir = "bytefight_logs"
    opponents_dir = os.path.join(log_dir, "opponents")
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(opponents_dir, exist_ok=True)
    
    # 4) Initialize with default opponent or best previous model
    initial_opponent = None
    best_model_path = os.path.join(opponents_dir, "best_model.zip")
    if os.path.exists(best_model_path):
        print("Loading previous best model as initial opponent...")
        try:
            initial_opponent = PPO.load(best_model_path)
            print("Successfully loaded initial opponent!")
        except Exception as e:
            print(f"Failed to load opponent: {e}")
            initial_opponent = None
    
    # 5) Create parallel environments
    n_envs = 8
    env_fns = [make_env(map_string, submission_dir, initial_opponent, rank=i) for i in range(n_envs)]
    env = SubprocVecEnv(env_fns)
    env = VecMonitor(env)
    
    # 6) Create evaluation environment
    eval_env = make_env(map_string, submission_dir, initial_opponent)()
    
    # 7) Setup callbacks
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
        n_eval_episodes=10
    )
    
    # New callback for self-play
    self_play_callback = SelfPlayCallback(
        eval_env=eval_env,
        save_path=opponents_dir,
        eval_freq=25000 // n_envs,
        n_eval_episodes=10
    )
    
    # 8) TensorBoard logging
    try:
        import tensorboard
        tensorboard_log = f"{log_dir}/tensorboard/"
    except ImportError:
        print("TensorBoard not installed. Disabling tensorboard_log.")
        tensorboard_log = None

    # 9) Configure network architecture
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

    # 10) Create or load model
    model_path = os.path.join(log_dir, "latest_model.zip")
    if os.path.exists(model_path):
        print(f"Loading existing model from {model_path}")
        model = PPO.load(
            model_path,
            env=env,
            tensorboard_log=tensorboard_log
        )
    else:
        print("Creating new model")
        model = PPO(
            policy="MultiInputPolicy",
            env=env,
            learning_rate=partial(linear_schedule_fn, initial_value=3e-4),
            n_steps=2048,
            batch_size=64,
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
    
    # 11) Train the model
    total_timesteps = 1000000
    model.learn(
        total_timesteps=total_timesteps,
        callback=[checkpoint_callback, eval_callback, self_play_callback],
        progress_bar=True,
        reset_num_timesteps=False  # Continue from previous training
    )
    
    # 12) Save final model
    final_model_path = f"{log_dir}/ppo_bytefight_final"
    model.save(final_model_path)
    print("Final model saved to", final_model_path)
    
    # 13) Test against best opponent
    print("\nTesting final model against best opponent...")
    test_env = SingleProcessByteFightEnv(
        map_string=map_string,
        opponent_module="sample_player",
        submission_dir=submission_dir,
        render_mode="human"
    )
    
    # Try to load best opponent
    best_opponent = None
    if os.path.exists(os.path.join(opponents_dir, "best_model.zip")):
        try:
            best_opponent = PPO.load(os.path.join(opponents_dir, "best_model.zip"))
        except Exception:
            best_opponent = None
    
    test_env = OpponentWrapper(test_env, best_opponent)
    test_env = MaskedActionWrapper(test_env)
    
    wins, losses, ties = 0, 0, 0
    for episode in range(10):
        obs, _ = test_env.reset()
        done = False
        total_reward = 0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = test_env.step(action)
            total_reward += reward
        
        print(f"Episode {episode+1}: winner={info.get('winner')}, reward={total_reward:.2f}, turns={info.get('turn_count')}")
        
        if info.get('winner') == "PLAYER_A":
            wins += 1
        elif info.get('winner') == "PLAYER_B":
            losses += 1
        else:
            ties += 1
    
    print(f"\nFinal results: {wins} wins, {losses} losses, {ties} ties")
    test_env.close()


if __name__ == "__main__":
    main()