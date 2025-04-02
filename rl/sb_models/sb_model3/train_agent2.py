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
                    if hasattr(self, 'logger') and self.logger is not None:
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


class CurriculumByteFightEnv(gym.Wrapper):
    """
    A wrapper that implements curriculum learning for ByteFight by providing
    appropriate action masks and reward shaping based on difficulty level.
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
        
        # Keep the original action space
        self.action_space = env.action_space
        
        # Modify observation space to include proper action masks for the multi-discrete action space
        # We need a mask for [3 moves count options, 9 first move options, 9 second move options, 9 third move options]
        self.observation_space = spaces.Dict({
            # Keep all original observation components
            **env.observation_space.spaces,
            # Replace the original action_mask with our expanded one
            "action_mask": spaces.Box(low=0, high=1, shape=(30,), dtype=np.uint8)
        })
    
    def step(self, action):
        """
        Process action and apply reward shaping based on difficulty level
        """
        # Let the environment process the action as-is - no modifications
        obs, reward, done, truncated, info = self.env.step(action)
        
        # Modify rewards based on difficulty to encourage learning
        if not done:
            # For the simplest difficulty, boost rewards to encourage learning
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
        
        # Update the action mask in the observation
        obs = self._update_action_mask(obs)
        
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
        
        # Update the action mask in the observation
        obs = self._update_action_mask(obs)
        
        return obs, info
    
    def _update_action_mask(self, obs):
        """
        Create an expanded action mask that properly covers the multi-discrete action space.
        This ensures the agent only selects valid actions based on the current difficulty level.
        """
        # Create a full action mask for the multi-discrete space
        # Format: [3 move counts, 9 first move options, 9 second move options, 9 third move options]
        mask = np.zeros(30, dtype=np.uint8)
        
        # Get the original action mask (used for individual moves)
        original_mask = obs.get("action_mask", np.ones(9, dtype=np.uint8))
        
        # Determine if we're in bidding phase
        is_bidding = obs["turn_count"].item() == 0
        
        if is_bidding:
            # In bidding phase, we only need to mask the first part (move count which is used as bid)
            if self.difficulty == 0:
                # At difficulty 0, only allow bid of 1
                mask[1] = 1  # Allow only bid value 1
            elif self.difficulty == 1:
                # At difficulty 1, allow bids 0-2
                mask[0:3] = 1  # Allow bid values 0, 1, 2
            else:
                # At higher difficulties, allow all bid values
                mask[0:3] = 1  # Allow all bid values
                
            # Mask for moves is not relevant in bidding phase
            mask[3:] = 0
        
        else:
            # Normal move phase - mask based on difficulty
            try:
                # Get the board state and player board if available
                if hasattr(self.env, '_board'):
                    board = self.env._board
                    pb_a = None
                    try:
                        # Try to get player board
                        pb_a = self.env._board.get_player_board(True)
                    except:
                        # If get_player_board is not available, try a different approach
                        if hasattr(self.env, '_make_observation'):
                            pass  # We'll use original_mask for move validity
            except:
                pass  # We'll use original_mask for move validity
            
            # Mask move counts based on difficulty
            if self.difficulty == 0:
                # Only allow single moves at difficulty 0
                mask[0] = 1  # Allow only one move
                mask[1:3] = 0  # No multiple moves
            elif self.difficulty == 1:
                # Allow up to two moves at difficulty 1
                mask[0:2] = 1  # Allow one or two moves
                mask[2] = 0  # No three moves
            else:
                # Allow all move counts at higher difficulties
                mask[0:3] = 1  # Allow all move counts
            
            # Mask individual moves using the original mask
            # First move options (index 3-11)
            mask[3:12] = original_mask
            
            # For second and third moves, we simplify by allowing the same moves as the first
            # In a real implementation, you would compute valid follow-up moves based on the first move
            if self.difficulty >= 1:
                # Second move options (index 12-20)
                mask[12:21] = original_mask
                
                if self.difficulty >= 2:
                    # Third move options (index 21-29)
                    mask[21:30] = original_mask
        
        # Update the observation with our expanded mask
        obs_dict = dict(obs)
        obs_dict["action_mask"] = mask
        
        return obs_dict
    
    def increase_difficulty(self):
        """Increase the difficulty level up to the maximum"""
        if self.difficulty < 3:
            prev_difficulty = self.difficulty
            self.difficulty += 1
            print(f"Difficulty increased from {prev_difficulty} to {self.difficulty}")
            return True
        return False


# Custom PPO policy that properly handles action masking
class MaskedMultiDiscretePolicy(torch.nn.Module):
    """
    Custom policy that applies action masks for a multi-discrete action space.
    """
    def __init__(self, observation_space, action_space, features_dim=256):
        super().__init__()
        self.observation_space = observation_space
        self.action_space = action_space
        
        # Create the feature extractor
        self.features_extractor = SnakeCNN(observation_space, features_dim=features_dim)
        
        # Create separate policy networks for each part of the multi-discrete action space
        self.policy_nets = torch.nn.ModuleList()
        for n_actions in action_space.nvec:
            self.policy_nets.append(torch.nn.Sequential(
                torch.nn.Linear(features_dim, 128),
                torch.nn.ReLU(),
                torch.nn.Linear(128, n_actions)
            ))
        
        # Create value network
        self.value_net = torch.nn.Sequential(
            torch.nn.Linear(features_dim, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 1)
        )
    
    def forward(self, obs, deterministic=False):
        """
        Forward pass that applies action masking before action selection.
        """
        # Extract features from the observation
        features = self.features_extractor(obs)
        
        # Get action logits for each part of the multi-discrete space
        logits_list = []
        for policy_net in self.policy_nets:
            logits = policy_net(features)
            logits_list.append(logits)
        
        # Get value prediction
        values = self.value_net(features).squeeze(-1)
        
        # Apply action masks to the logits
        if "action_mask" in obs:
            mask = obs["action_mask"]
            
            # Reshape the mask to match the multi-discrete dimensions
            mask_parts = []
            start_idx = 0
            for i, n_actions in enumerate(self.action_space.nvec):
                end_idx = start_idx + n_actions
                mask_parts.append(mask[:, start_idx:end_idx])
                start_idx = end_idx
            
            # Apply each mask to its corresponding logits
            for i, (logits, mask_part) in enumerate(zip(logits_list, mask_parts)):
                # Set masked actions to large negative values
                masked_logits = torch.where(
                    mask_part > 0,
                    logits,
                    torch.tensor(-1e8, device=logits.device, dtype=logits.dtype)
                )
                logits_list[i] = masked_logits
        
        # Sample actions
        if deterministic:
            # Select most likely action for each dimension
            actions = torch.stack([torch.argmax(logits, dim=1) for logits in logits_list], dim=1)
        else:
            # Sample from masked probabilities
            actions = []
            for logits in logits_list:
                probs = torch.nn.functional.softmax(logits, dim=1)
                action = torch.multinomial(probs, 1).squeeze(-1)
                actions.append(action)
            actions = torch.stack(actions, dim=1)
        
        # Compute log probabilities (dummy for compatibility)
        log_probs = torch.zeros_like(actions[:, 0], dtype=torch.float32)
        
        return actions, values, log_probs


def make_env(map_string, opponent_module, submission_dir, difficulty=0, rank=0):
    """
    Helper function to create a curriculum ByteFight environment with seed.
    """
    def _init():
        env = ByteFightEnv(
            map_string=map_string,
            opponent_module=opponent_module,
            submission_dir=submission_dir,
            render_mode=None
        )
        # Wrap with curriculum environment with specified difficulty
        env = CurriculumByteFightEnv(env, difficulty=difficulty)
        return env
    
    set_random_seed(rank)
    return _init


class MaskedPPO(PPO):
    """
    Custom PPO implementation that uses our masked multi-discrete policy.
    """
    def __init__(
        self,
        policy,
        env,
        difficulty=0,
        learning_rate=0.0003,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        clip_range_vf=None,
        normalize_advantage=True,
        ent_coef=0.0,
        vf_coef=0.5,
        max_grad_norm=0.5,
        use_sde=False,
        sde_sample_freq=-1,
        target_kl=None,
        stats_window_size=100,
        tensorboard_log=None,
        policy_kwargs=None,
        verbose=0,
        seed=None,
        device="auto",
        _init_setup_model=True,
    ):
        # Use the custom policy if not specified otherwise
        if policy == "MaskedMultiDiscretePolicy":
            policy = MaskedMultiDiscretePolicy
        
        # Initialize with parent class
        super().__init__(
            policy=policy,
            env=env,
            learning_rate=learning_rate,
            n_steps=n_steps,
            batch_size=batch_size,
            n_epochs=n_epochs,
            gamma=gamma,
            gae_lambda=gae_lambda,
            clip_range=clip_range,
            clip_range_vf=clip_range_vf,
            normalize_advantage=normalize_advantage,
            ent_coef=ent_coef,
            vf_coef=vf_coef,
            max_grad_norm=max_grad_norm,
            use_sde=use_sde,
            sde_sample_freq=sde_sample_freq,
            target_kl=target_kl,
            stats_window_size=stats_window_size,
            tensorboard_log=tensorboard_log,
            policy_kwargs=policy_kwargs,
            verbose=verbose,
            seed=seed,
            device=device,
            _init_setup_model=_init_setup_model,
        )
        self.difficulty = difficulty
    
    def set_difficulty(self, difficulty):
        """Update the difficulty level."""
        self.difficulty = difficulty
        
        # Also update all sub-environments if they support difficulty
        try:
            for env in self.env.envs:
                if hasattr(env, 'increase_difficulty'):
                    while env.difficulty < difficulty:
                        env.increase_difficulty()
        except:
            pass


def create_model(env, log_dir=None, tensorboard_log=None, difficulty=0):
    """
    Create a masked PPO model with architecture suitable for the current difficulty.
    """
    # Define policy architecture - simpler for lower difficulties
    if difficulty <= 1:
        features_dim = 128
        
        # Higher entropy coefficient for more exploration at lower difficulties
        ent_coef = 0.05
        lr = 5e-4  # Higher learning rate
        n_steps = 1024  # Shorter rollouts
        batch_size = 64
        n_epochs = 4
        
    else:
        features_dim = 256
        
        # Lower entropy as we want more exploitation at higher difficulties
        ent_coef = 0.01
        lr = 3e-4  # Standard learning rate
        n_steps = 2048  # Longer rollouts
        batch_size = 256
        n_epochs = 10
    
    policy_kwargs = {
        "features_dim": features_dim,
    }
    
    # Create masked PPO model with hyperparameters adjusted to difficulty
    model = MaskedPPO(
        policy="MaskedMultiDiscretePolicy",
        env=env,
        difficulty=difficulty,
        learning_rate=linear_schedule(lr),
        n_steps=n_steps,
        batch_size=batch_size,
        n_epochs=n_epochs,
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
        eval_env_fn = make_env(map_string, opponent_module, submission_dir, difficulty=difficulty)
        eval_env = Monitor(eval_env_fn())
        eval_env = DummyVecEnv([lambda: eval_env])
        eval_env = VecMonitor(eval_env)
        
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
        
        stats_callback = TrackStatsCallback(verbose=1)
        
        # If first difficulty level, create new model
        if model is None:
            model = create_model(env, log_dir, tensorboard_log, difficulty=difficulty)
        else:
            # For subsequent levels, update model with the new environment
            model.set_env(env)
            model.set_difficulty(difficulty)
            
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
    total_timesteps = 500000
    
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