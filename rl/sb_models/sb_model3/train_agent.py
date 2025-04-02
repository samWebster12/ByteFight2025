import json
import os
import numpy as np
import random
import torch
import gymnasium as gym
from gymnasium import spaces
from typing import Dict, List, Optional, Tuple, Union

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback, BaseCallback
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor, DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import set_random_seed

# Import your environment and custom components
from bytefight_env import ByteFightEnv
from sb_arch import SnakeCNN, linear_schedule
from game.enums import Action, Cell

# Import tensorboard for logging if available
try:
    import tensorboard
    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False


class TrackStatsCallback(BaseCallback):
    """
    Callback to track and print training statistics.
    """
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_count = 0
        self.episode_rewards = []
        self.win_count = 0
        self.lose_count = 0
        self.tie_count = 0
        self.best_reward = -float('inf')
        
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
                
                # Update best reward
                if reward > self.best_reward:
                    self.best_reward = reward
                
                # Print progress every 10 episodes
                if self.episode_count % 10 == 0:
                    last_10 = self.episode_rewards[-10:]
                    avg_reward = sum(last_10) / len(last_10)
                    win_rate = (self.win_count / self.episode_count) * 100
                    
                    # Log to tensorboard
                    self.logger.record("train/avg_reward_last10", avg_reward)
                    self.logger.record("train/win_rate", win_rate)
                    self.logger.record("train/loss_rate", (self.lose_count / self.episode_count) * 100)
                    self.logger.record("train/tie_rate", (self.tie_count / self.episode_count) * 100)
                    self.logger.record("train/best_reward", self.best_reward)
                    
                    if self.verbose:
                        print(f"Episode {self.episode_count}, Avg Reward: {avg_reward:.2f}, Win Rate: {win_rate:.1f}%")
                        print(f"Wins: {self.win_count}, Losses: {self.lose_count}, Ties: {self.tie_count}")
                        print(f"Best reward so far: {self.best_reward:.2f}")
                        print("-"*40)
        
        return True


class SimplifiedByteFightEnv(gym.Wrapper):
    """
    A wrapper that simplifies the ByteFight environment for early training
    and progressively introduces complexity.
    """
    def __init__(self, env, difficulty=0):
        """
        Initialize the wrapper with a difficulty level:
        0 = Very simplified (single moves only, simplified bidding)
        1 = Basic (allow dual moves with enhanced rewards)
        2 = Intermediate (full moves with standard rewards)
        3 = Full (complete original environment)
        """
        super().__init__(env)
        self.difficulty = difficulty
        self.original_env = env
        
        # The action space remains the same
        self.action_space = env.action_space
        
        # Keep the observation space the same to maintain compatibility
        self.observation_space = env.observation_space
    
    def step(self, action):
        """
        Process the action according to the current difficulty level
        """
        multi_action = np.array(action, dtype=np.int64).copy()
        
        # For the simplest difficulty (0): force single moves and help with bidding
        if self.difficulty == 0:
            # If in bidding phase, use a safe bid value (1)
            if hasattr(self.env, '_bid_phase') and self.env._bid_phase:
                # Use a safe bid value
                multi_action[0] = 1  # Bid of 1 is generally safe
            else:
                # Force only single moves by setting first element to 0
                multi_action[0] = 0
                
                # Ensure we're making a valid move - prefer cardinal directions
                multi_action[1] = min(multi_action[1], 7)  # Stick to directions, no traps yet
        
        # For difficulty 1: allow up to two moves but with guided selection
        elif self.difficulty == 1:
            # If not in bidding phase
            if not (hasattr(self.env, '_bid_phase') and self.env._bid_phase):
                # Limit to at most 2 moves (0-1 in the action space)
                multi_action[0] = min(1, multi_action[0])
        
        # Call the original environment's step with our processed action
        obs, reward, done, truncated, info = self.env.step(multi_action)
        
        # Modify rewards based on difficulty
        if not done:
            # For the simplest difficulty, boost all rewards to encourage initial learning
            if self.difficulty == 0:
                # Bonus just for surviving a step
                reward += 0.5
                
                # If positive reward (ate apple, etc.), boost it further
                if reward > 0:
                    reward *= 2.0
            
            # For difficulty 1, smaller boost to rewards
            elif self.difficulty == 1:
                # Smaller bonus for surviving
                reward += 0.2
                
                # If positive reward, boost it a bit
                if reward > 0:
                    reward *= 1.5
        
        return obs, reward, done, truncated, info
    
    def reset(self, **kwargs):
        """Reset and apply simplified rules based on difficulty"""
        obs, info = self.env.reset(**kwargs)
        
        # Simplify bidding at the lowest difficulty
        if self.difficulty == 0 and hasattr(self.env, '_bid_phase') and self.env._bid_phase:
            # Auto-bid a safe value
            self.env._board.resolve_bid(1, 0)  # Always bid 1, opponent bids 0
            self.env._bid_phase = False
            
            # Reset observation after auto-bidding
            obs = self.env._make_observation()
        
        return obs, info
    
    def increase_difficulty(self):
        """Increase the difficulty level up to the maximum"""
        if self.difficulty < 3:
            prev_difficulty = self.difficulty
            self.difficulty += 1
            print(f"Difficulty increased from {prev_difficulty} to {self.difficulty}")
            return True
        return False


# Custom policy that properly handles the MultiDiscrete action space
class ImprovedActorCriticPolicy(torch.nn.Module):
    """
    Actor Critic policy for ByteFight that includes built-in action guidance
    to help the agent learn valid actions early in training.
    """
    def __init__(self, observation_space, action_space, difficulty=0):
        super().__init__()
        self.observation_space = observation_space
        self.action_space = action_space
        self.difficulty = difficulty
        
        # Create feature extractor
        self.features_extractor = SnakeCNN(observation_space, features_dim=256)
        
        # Create value network
        self.value_net = torch.nn.Sequential(
            torch.nn.Linear(256, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 1)
        )
        
        # Create policy networks for each dimension of the MultiDiscrete space
        self.policy_nets = torch.nn.ModuleList()
        for n_cat in action_space.nvec:
            self.policy_nets.append(torch.nn.Sequential(
                torch.nn.Linear(256, 128),
                torch.nn.ReLU(),
                torch.nn.Linear(128, n_cat)
            ))
            
    def forward(self, obs, deterministic=False):
        """Forward pass that applies guidance based on difficulty level"""
        # Extract features
        features = self.features_extractor(obs)
        
        # Get action logits for each part of the MultiDiscrete space
        action_logits = []
        for policy_net in self.policy_nets:
            logits = policy_net(features)
            action_logits.append(logits)
        
        # Get value prediction
        value = self.value_net(features).squeeze(-1)
        
        # Sample actions
        actions = []
        log_probs = []
        
        # Determine if we're in bidding phase (turn_count = 0)
        is_bidding = obs["turn_count"][0, 0] == 0
        
        # Apply guidance based on difficulty and phase
        for i, logits in enumerate(action_logits):
            # Simplified logic for bidding phase
            if is_bidding and i == 0 and self.difficulty <= 1:
                # Force a safe bid value (1) at low difficulties
                action = torch.ones_like(logits[:, 0], dtype=torch.int64)
                
            # Simplified logic for moves at difficulty 0
            elif i == 0 and self.difficulty == 0 and not is_bidding:
                # Force single moves only (0)
                action = torch.zeros_like(logits[:, 0], dtype=torch.int64)
                
            # Normal action sampling
            else:
                if deterministic:
                    action = torch.argmax(logits, dim=-1)
                else:
                    probs = torch.nn.functional.softmax(logits, dim=-1)
                    action = torch.multinomial(probs, 1).squeeze(-1)
            
            actions.append(action)
            # Dummy log_prob
            log_prob = torch.zeros_like(action, dtype=torch.float32)
            log_probs.append(log_prob)
        
        # Combine actions and log_probs
        actions = torch.stack(actions, dim=-1)
        log_prob = torch.stack(log_probs, dim=-1).sum(dim=-1)
        
        return actions, value, log_prob
    
    def set_difficulty(self, difficulty):
        """Update the policy's difficulty level"""
        self.difficulty = difficulty


def make_env(map_string, opponent_module, submission_dir, difficulty=0, rank=0):
    """
    Helper function to create a simplified ByteFight environment with seed.
    """
    def _init():
        env = ByteFightEnv(
            map_string=map_string,
            opponent_module=opponent_module,
            submission_dir=submission_dir,
            render_mode=None
        )
        # Wrap with simplified environment with specified difficulty
        env = SimplifiedByteFightEnv(env, difficulty=difficulty)
        return env
    
    set_random_seed(rank)
    return _init


def create_model(env, log_dir=None, tensorboard_log=None, difficulty=0):
    """
    Create a PPO model with architecture suitable for the current difficulty.
    """
    # Define policy architecture - simpler for lower difficulties
    if difficulty <= 1:
        policy_kwargs = {
            "net_arch": dict(pi=[128, 64], vf=[128, 64]),  # Simpler network
            "features_extractor_class": SnakeCNN,
            "features_extractor_kwargs": dict(features_dim=128),  # Fewer features
            "activation_fn": torch.nn.ReLU,
        }
        
        # Higher entropy coefficient for more exploration at lower difficulties
        ent_coef = 0.05
        lr = 5e-4  # Higher learning rate
        n_steps = 1024  # Shorter rollouts
        
    else:
        # More complex architecture for higher difficulties
        policy_kwargs = {
            "net_arch": dict(pi=[256, 128], vf=[256, 128]),  # Deeper network
            "features_extractor_class": SnakeCNN,
            "features_extractor_kwargs": dict(features_dim=256),  # More features
            "activation_fn": torch.nn.ReLU,
        }
        
        # Lower entropy as we want more exploitation at higher difficulties
        ent_coef = 0.01
        lr = 3e-4  # Standard learning rate
        n_steps = 2048  # Longer rollouts
    
    # Create PPO model with hyperparameters adjusted to difficulty
    model = PPO(
        policy="MultiInputPolicy",  # Use SB3's MultiInputPolicy which works with Dict observations
        env=env,
        learning_rate=linear_schedule(lr),
        n_steps=n_steps,
        batch_size=64 if difficulty <= 1 else 256,  # Smaller batches for simpler tasks
        n_epochs=4 if difficulty <= 1 else 10,  # Fewer epochs early on
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=ent_coef,
        vf_coef=0.5,
        max_grad_norm=0.5,
        policy_kwargs=policy_kwargs,
        verbose=1,
        device="cuda",
        tensorboard_log=tensorboard_log
    )
    
    return model


def curriculum_train(
    map_dict,
    opponent_module="sample_player",
    submission_dir="../../../workspace",
    log_dir="bytefight_curriculum_logs",
    n_envs=4,
    total_timesteps=500000,
):
    """
    Train with curriculum learning - progressively increase difficulty.
    """
    os.makedirs(log_dir, exist_ok=True)
    
    # Setup tensorboard logging
    if TENSORBOARD_AVAILABLE:
        tensorboard_log = f"{log_dir}/tensorboard/"
    else:
        print("TensorBoard not installed. Disabling tensorboard_log.")
        tensorboard_log = None
    
    # Define our difficulty levels and timesteps per level
    difficulties = [0, 1, 2, 3]  # Increasing complexity
    timesteps_per_difficulty = total_timesteps // len(difficulties)
    
    # Performance criteria for progression
    min_avg_reward = [-3.0, -2.0, -1.0, None]  # Min average reward to advance
    min_win_rate = [0.05, 0.15, 0.25, None]    # Min win rate to advance
    max_timesteps_per_difficulty = [
        timesteps_per_difficulty * 2,  # Level 0: More time if needed
        timesteps_per_difficulty * 1.5, # Level 1: Some extra time 
        timesteps_per_difficulty * 1.2, # Level 2: A bit extra time
        timesteps_per_difficulty       # Level 3: Standard time
    ]
    
    # Select map for training (just use empty map for now)
    map_string = map_dict["empty"]
    
    # Create model and train through each difficulty level
    model = None
    total_timesteps_used = 0
    difficulty_idx = 0
    
    while difficulty_idx < len(difficulties) and total_timesteps_used < total_timesteps:
        difficulty = difficulties[difficulty_idx]
        print(f"\n===== Starting training at difficulty level {difficulty} =====\n")
        
        # Create environments with current difficulty
        env_fns = [make_env(map_string, opponent_module, submission_dir, 
                           difficulty=difficulty, rank=i) for i in range(n_envs)]
        env = SubprocVecEnv(env_fns)
        env = VecMonitor(env)
        
        # Create evaluation environment with same difficulty
        eval_env = Monitor(make_env(map_string, opponent_module, submission_dir, difficulty=difficulty)())
        eval_env = DummyVecEnv([lambda: eval_env])
        
        # Setup callbacks for this difficulty level
        checkpoint_callback = CheckpointCallback(
            save_freq=timesteps_per_difficulty // 4,
            save_path=f"{log_dir}/checkpoints/difficulty_{difficulty}/",
            name_prefix="ppo_bytefight",
            save_vecnormalize=True
        )
        
        eval_callback = EvalCallback(
            eval_env,
            best_model_save_path=f"{log_dir}/best_model/difficulty_{difficulty}/",
            log_path=f"{log_dir}/eval_results/difficulty_{difficulty}/",
            eval_freq=min(10000, timesteps_per_difficulty // 10),
            deterministic=True,
            render=False,
            n_eval_episodes=10
        )
        
        stats_callback = TrackStatsCallback(verbose=0)
        
        # If first difficulty level, create new model
        if model is None:
            model = create_model(env, log_dir, tensorboard_log, difficulty=difficulty)
        else:
            # For subsequent levels, update model with the new environment
            model.set_env(env)
            
            # Adjust learning parameters for higher difficulties if needed
            if difficulty >= 2:
                model.ent_coef = 0.01  # Reduce entropy for more exploitation
                
        # Train at current difficulty
        timesteps_for_this_level = min(
            max_timesteps_per_difficulty[difficulty_idx], 
            total_timesteps - total_timesteps_used
        )
        
        print(f"Training at difficulty {difficulty} for {timesteps_for_this_level} timesteps")
        
        model.learn(
            total_timesteps=timesteps_for_this_level,
            callback=[checkpoint_callback, eval_callback, stats_callback],
            progress_bar=True,
            reset_num_timesteps=(difficulty_idx == 0)  # Only reset on first difficulty
        )
        
        # Update total timesteps used
        total_timesteps_used += timesteps_for_this_level
        
        # Save model for this difficulty
        model_path = f"{log_dir}/model_difficulty_{difficulty}"
        model.save(model_path)
        print(f"Saved model at difficulty {difficulty} to {model_path}")
        
        # Check performance to see if we should advance difficulty
        if difficulty_idx < len(difficulties) - 1:  # Skip check for final difficulty
            # Calculate performance metrics from the last 100 episodes
            recent_rewards = stats_callback.episode_rewards[-100:] if len(stats_callback.episode_rewards) >= 100 else stats_callback.episode_rewards
            avg_reward = sum(recent_rewards) / max(1, len(recent_rewards))
            
            episodes = stats_callback.episode_count
            win_rate = stats_callback.win_count / max(1, episodes)
            
            print(f"Performance after {timesteps_for_this_level} timesteps:")
            print(f"  Average reward: {avg_reward:.2f}")
            print(f"  Win rate: {win_rate:.2%}")
            
            # Check if we should advance to the next difficulty
            reward_criterion = min_avg_reward[difficulty_idx] is None or avg_reward >= min_avg_reward[difficulty_idx]
            win_criterion = min_win_rate[difficulty_idx] is None or win_rate >= min_win_rate[difficulty_idx]
            
            if reward_criterion or win_criterion:
                print(f"Performance criteria met! Moving to next difficulty level.")
                difficulty_idx += 1
            else:
                # Check if we've spent too much time at this difficulty
                if timesteps_for_this_level >= max_timesteps_per_difficulty[difficulty_idx]:
                    print(f"Maximum timesteps for difficulty {difficulty} reached. Advancing anyway.")
                    difficulty_idx += 1
                else:
                    # Train for a bit longer at this difficulty (another 25% of timesteps)
                    extra_timesteps = min(
                        timesteps_per_difficulty // 4,
                        total_timesteps - total_timesteps_used,
                        max_timesteps_per_difficulty[difficulty_idx] - timesteps_for_this_level
                    )
                    
                    if extra_timesteps > 0:
                        print(f"Performance criteria not yet met. Training for additional {extra_timesteps} timesteps.")
                        model.learn(
                            total_timesteps=extra_timesteps,
                            callback=[checkpoint_callback, eval_callback, stats_callback],
                            progress_bar=True,
                            reset_num_timesteps=False
                        )
                        total_timesteps_used += extra_timesteps
                        
                        # Check again after extra training
                        recent_rewards = stats_callback.episode_rewards[-100:] if len(stats_callback.episode_rewards) >= 100 else stats_callback.episode_rewards
                        avg_reward = sum(recent_rewards) / max(1, len(recent_rewards))
                        win_rate = stats_callback.win_count / max(1, stats_callback.episode_count)
                        
                        print(f"Performance after additional training:")
                        print(f"  Average reward: {avg_reward:.2f}")
                        print(f"  Win rate: {win_rate:.2%}")
                        
                        reward_criterion = min_avg_reward[difficulty_idx] is None or avg_reward >= min_avg_reward[difficulty_idx]
                        win_criterion = min_win_rate[difficulty_idx] is None or win_rate >= min_win_rate[difficulty_idx]
                        
                        if reward_criterion or win_criterion:
                            print(f"Performance criteria now met! Moving to next difficulty level.")
                        else:
                            print(f"Still not meeting criteria, but advancing to prevent getting stuck.")
                    
                    # Advance to next difficulty regardless
                    difficulty_idx += 1
        else:
            # Final difficulty, move on
            difficulty_idx += 1
    
    # Save final model
    final_model_path = f"{log_dir}/final_model"
    model.save(final_model_path)
    print(f"Final model saved to {final_model_path}")
    
    return model


def evaluate_model(
    model,
    map_string,
    opponent_module="sample_player",
    submission_dir="../../../workspace",
    num_episodes=10,
    render_mode="human",
    difficulty=3  # Evaluate at full difficulty by default
):
    """Evaluate a trained model."""
    # Create evaluation environment
    eval_env = make_env(map_string, opponent_module, submission_dir, difficulty=difficulty)()
    
    # Set render mode if specified
    eval_env.render_mode = render_mode
    
    # Track metrics
    rewards = []
    win_count = 0
    lose_count = 0
    tie_count = 0
    turn_counts = []
    
    for episode in range(num_episodes):
        obs, _ = eval_env.reset()
        done = False
        total_reward = 0
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = eval_env.step(action)
            total_reward += reward
        
        rewards.append(total_reward)
        turn_counts.append(info.get('turn_count', 0))
        
        winner = info.get("winner")
        if winner == "PLAYER_A":
            win_count += 1
        elif winner == "PLAYER_B":
            lose_count += 1
        else:
            tie_count += 1
        
        print(f"Episode {episode+1}: winner={winner}, reward={total_reward:.2f}, turn_count={info.get('turn_count')}")
    
    eval_env.close()
    
    # Calculate metrics
    win_rate = win_count / num_episodes
    avg_reward = sum(rewards) / num_episodes
    avg_turns = sum(turn_counts) / num_episodes if turn_counts else 0
    
    print(f"\nEvaluation Results (Difficulty {difficulty}):")
    print(f"Win Rate: {win_rate * 100:.1f}%")
    print(f"Average Reward: {avg_reward:.2f}")
    print(f"Average Game Length: {avg_turns:.1f} turns")
    print(f"Wins: {win_count}, Losses: {lose_count}, Ties: {tie_count}")
    
    return {
        "win_rate": win_rate,
        "avg_reward": avg_reward,
        "avg_turns": avg_turns,
        "win_count": win_count,
        "lose_count": lose_count,
        "tie_count": tie_count
    }


def main():
    # 1) Load maps
    with open("maps.json") as f:
        map_dict = json.load(f)
    
    # 2) Opponent module
    opponent_module = "sample_player"
    submission_dir = "../../../workspace"
    
    # 3) Training parameters
    log_dir = "bytefight_curriculum_logs"
    n_envs = 4
    total_timesteps = 250000
    
    # 4) Train with curriculum learning
    model = curriculum_train(
        map_dict,
        opponent_module,
        submission_dir,
        log_dir,
        n_envs,
        total_timesteps
    )
    
    # 5) Evaluate the final model at full difficulty
    evaluate_model(
        model,
        map_dict["empty"],
        opponent_module,
        submission_dir,
        num_episodes=5,
        render_mode="human",
        difficulty=3  # Full difficulty
    )


if __name__ == "__main__":
    main()