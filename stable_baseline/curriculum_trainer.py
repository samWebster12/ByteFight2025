import json
import os
import numpy as np
import random
import shutil
import time
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback, BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor

# Import ByteFight environment
from bytefight_env_single import SingleProcessByteFightEnv

class CurriculumBytefightEnv(SingleProcessByteFightEnv):
    """
    ByteFight environment that supports curriculum learning with custom opponents.
    """
    def __init__(
        self,
        map_string,
        opponent_level=1,
        submission_dir="./workspace",
        max_steps=2000,
        render_mode=None
    ):
        # We'll still use the opponent_module parameter, but we'll override
        # the opponent controller after initialization
        super().__init__(
            map_string=map_string,
            opponent_module="sample_player",  # This will be replaced
            submission_dir=submission_dir,
            max_steps=max_steps,
            render_mode=render_mode
        )
        self.opponent_level = opponent_level
        
        # Replace the opponent controller with our smart controller
        from smart_opponent import SmartPlayerController
        time_left_callable = lambda: 5.0
        self._opponent_controller = SmartPlayerController(
            time_left_callable, 
            intelligence_level=opponent_level
        )

class PrintStatsCallback(BaseCallback):
    """
    Callback for printing training statistics.
    """
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_count = 0
        self.episode_rewards = []
        self.win_count = 0
        self.lose_count = 0
        self.tie_count = 0
        self.max_length = 0
        self.start_time = time.time()
        
    def _on_step(self):
        # Check if episode ended
        if self.locals.get("dones")[0]:
            # Get episode reward
            reward = sum(self.locals.get("episode_rewards")[0])
            self.episode_rewards.append(reward)
            self.episode_count += 1
            
            # Get winner info from info dict
            info = self.locals.get("infos")[0]
            winner = info.get("winner")
            if winner == "PLAYER_A":
                self.win_count += 1
            elif winner == "PLAYER_B":
                self.lose_count += 1
            elif winner == "TIE":
                self.tie_count += 1
            
            # Track max length achieved
            agent_length = info.get("agent_max_length", 0)
            if agent_length > self.max_length:
                self.max_length = agent_length
            
            # Print every 10 episodes
            if self.episode_count % 10 == 0:
                elapsed_time = time.time() - self.start_time
                avg_reward = sum(self.episode_rewards[-10:]) / 10
                win_rate = self.win_count / self.episode_count * 100
                print(f"Episodes: {self.episode_count}, Time: {elapsed_time:.1f}s")
                print(f"Avg Reward: {avg_reward:.2f}, Win Rate: {win_rate:.1f}%")
                print(f"Wins: {self.win_count}, Losses: {self.lose_count}, Ties: {self.tie_count}")
                print(f"Max Length: {self.max_length}")
                print("-" * 40)
        
        return True

def make_env(map_string, opponent_level, submission_dir, rank=0):
    """
    Helper function to create a wrapped environment.
    """
    def _init():
        # Create environment with appropriate opponent level
        env = CurriculumBytefightEnv(
            map_string=map_string,
            opponent_level=opponent_level,
            submission_dir=submission_dir,
            render_mode=None
        )
        # Wrap with monitor for statistics
        return Monitor(env)
    
    # Set random seed for reproducibility
    set_random_seed(rank)
    return _init

def curriculum_training():
    """
    Train a ByteFight agent using curriculum learning with increasingly difficult opponents.
    """
    # Load map
    with open("maps.json") as f:
        map_dict = json.load(f)
    
    # Select different maps for different stages of training
    map_strings = {
        "empty": map_dict.get("empty", ""),
        "obstacles": map_dict.get("cage", ""),  # Use "cage" as an obstacle map
        "complex": map_dict.get("combustible_lemons", "")  # Use this as a complex map
    }
    
    submission_dir = "./workspace"
    log_dir = "bytefight_curriculum_logs"
    os.makedirs(log_dir, exist_ok=True)
    
    # Curriculum stages configuration
    stages = [
        # Start with basic opponent on empty map
        {
            "opponent_level": 1, 
            "steps": 200000, 
            "name": "stage1_random",
            "map": "empty"
        },
        # Progress to apple-seeking opponent
        {
            "opponent_level": 2, 
            "steps": 200000, 
            "name": "stage2_apple_seeking",
            "map": "empty"
        },
        # Add obstacle map with moderate opponent
        {
            "opponent_level": 2, 
            "steps": 200000, 
            "name": "stage3_obstacles",
            "map": "obstacles"
        },
        # Harder opponent that avoids traps
        {
            "opponent_level": 3, 
            "steps": 300000, 
            "name": "stage4_smart",
            "map": "obstacles"
        },
        # Advanced opponent on complex map
        {
            "opponent_level": 4, 
            "steps": 300000, 
            "name": "stage5_advanced",
            "map": "complex"
        },
    ]
    
    model = None
    total_steps = sum(stage["steps"] for stage in stages)
    
    # Create placeholder for the final model
    final_model_path = f"{log_dir}/final_model/ppo_bytefight_final"
    os.makedirs(os.path.dirname(final_model_path), exist_ok=True)
    
    # Progress through curriculum stages
    for i, stage in enumerate(stages):
        print(f"\n{'='*50}")
        print(f"Starting curriculum stage {i+1}/{len(stages)}: {stage['name']}")
        print(f"Opponent Level: {stage['opponent_level']}, Map: {stage['map']}")
        print(f"{'='*50}\n")
        
        # Get map for this stage
        map_string = map_strings.get(stage["map"], map_strings["empty"])
        
        # Number of environments to run in parallel
        n_envs = 4
        
        # Create environment creators for each parallel environment
        env_fns = [make_env(map_string, stage["opponent_level"], submission_dir, rank=i) for i in range(n_envs)]
        
        # Create vectorized environment
        env = SubprocVecEnv(env_fns)
        env = VecMonitor(env)
        
        # Create evaluation environment with same difficulty
        eval_env = CurriculumBytefightEnv(
            map_string=map_string,
            opponent_level=stage["opponent_level"],
            submission_dir=submission_dir
        )
        eval_env = Monitor(eval_env)
        
        # Setup callbacks
        stage_dir = f"{log_dir}/{stage['name']}"
        os.makedirs(f"{stage_dir}/checkpoints", exist_ok=True)
        os.makedirs(f"{stage_dir}/best_model", exist_ok=True)
        os.makedirs(f"{stage_dir}/results", exist_ok=True)
        
        checkpoint_callback = CheckpointCallback(
            save_freq=20000 // n_envs,
            save_path=f"{stage_dir}/checkpoints/",
            name_prefix=f"ppo_bytefight_{stage['name']}",
            save_vecnormalize=True
        )
        
        eval_callback = EvalCallback(
            eval_env,
            best_model_save_path=f"{stage_dir}/best_model/",
            log_path=f"{stage_dir}/results/",
            eval_freq=20000 // n_envs,
            deterministic=True,
            render=False,
            n_eval_episodes=5
        )
        
        print_callback = PrintStatsCallback()
        
        # Create custom network architecture
        policy_kwargs = {
            "net_arch": [dict(pi=[256, 256], vf=[256, 256])],
            "features_extractor_kwargs": {"features_dim": 512}
        }
        
        # If we already have a model, load it and continue training
        # Otherwise create a new model
        if model is None:
            model = PPO(
                policy="MultiInputPolicy", 
                env=env, 
                learning_rate=3e-4,
                n_steps=2048,
                batch_size=64,
                n_epochs=10,
                gamma=0.99,
                gae_lambda=0.95,
                clip_range=0.2,
                ent_coef=0.01,  # Encourage exploration
                vf_coef=0.5,
                max_grad_norm=0.5,
                tensorboard_log=f"{log_dir}/tensorboard/",
                policy_kwargs=policy_kwargs,
                verbose=1, 
                device="cuda"
            )
        else:
            # Continue training with existing model but on new environment
            model.set_env(env)
            
            # Adjust learning rate to be lower in later stages
            if i >= len(stages) // 2:
                model.learning_rate = 1e-4
            
            # Reduce entropy coefficient in later stages for more exploitation
            if i >= 2:
                model.ent_coef = 0.005
        
        # Train for the specified number of steps
        model.learn(
            total_timesteps=stage["steps"],
            callback=[checkpoint_callback, eval_callback, print_callback],
            progress_bar=True
        )
        
        # Save the model after this stage
        stage_model_path = f"{stage_dir}/model_after_stage"
        model.save(stage_model_path)
        print(f"Completed curriculum stage {i+1}: {stage['name']}")
        print(f"Model saved to {stage_model_path}")
        
        # After final stage, save as the final model
        if i == len(stages) - 1:
            model.save(final_model_path)
            print(f"Final model saved to {final_model_path}")
            
            # Also save to the expected path for the controller
            controller_model_path = "ppo_bytefight_final"
            model.save(controller_model_path)
            print(f"Model saved for controller at {controller_model_path}")
    
    print("\nCurriculum training complete!")
    
    # Evaluate the final model against all opponent levels
    print("\nEvaluating final model against all opponent levels:")
    
    for level in range(1, 5):
        print(f"\n{'-'*40}")
        print(f"Testing against opponent level {level}:")
        
        # Use the complex map for final evaluation
        map_string = map_strings["complex"]
        
        test_env = CurriculumBytefightEnv(
            map_string=map_string,
            opponent_level=level,
            submission_dir=submission_dir,
            render_mode="human"
        )
        
        # Run 3 test episodes per level
        wins = 0
        losses = 0
        ties = 0
        total_turns = 0
        total_length = 0
        
        for episode in range(3):
            obs, _ = test_env.reset()
            done = False
            total_reward = 0
            
            while not done:
                # Use the action mask to choose only valid actions
                action_mask = obs["action_mask"]
                
                # Get action from model
                action, _states = model.predict(obs, deterministic=True)
                
                # If the chosen action is invalid, replace it with a valid one
                if action_mask[action] == 0:
                    valid_actions = np.where(action_mask == 1)[0]
                    if len(valid_actions) > 0:
                        action = random.choice(valid_actions)
                
                obs, reward, done, truncated, info = test_env.step(action)
                total_reward += reward
            
            winner = info.get("winner")
            if winner == "PLAYER_A":
                wins += 1
            elif winner == "PLAYER_B":
                losses += 1
            elif winner == "TIE":
                ties += 1
            
            total_turns += info.get("turn_count", 0)
            total_length += info.get("agent_length", 0)
            
            print(f"Episode {episode+1} Results:")
            print(f"Winner: {winner}")
            print(f"Turns: {info.get('turn_count')}")
            print(f"Agent length: {info.get('agent_length')}")
            print(f"Opponent length: {info.get('opponent_length')}")
            print(f"Total reward: {total_reward:.2f}")
            print("-" * 30)
        
        # Print summary for this opponent level
        print(f"\nLevel {level} Summary:")
        print(f"Wins: {wins}, Losses: {losses}, Ties: {ties}")
        print(f"Win Rate: {wins/3*100:.1f}%")
        print(f"Average Game Length: {total_turns/3:.1f} turns")
        print(f"Average Agent Length: {total_length/3:.1f}")
    
    print("\nFinal model evaluation complete!")

if __name__ == "__main__":
    curriculum_training()