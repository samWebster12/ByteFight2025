import json
import os
import numpy as np
import gymnasium as gym
from stable_baselines3 import PPO

# Import your environment and wrappers
from bytefight_env_single import SingleProcessByteFightEnv
from wrappers import MaskedActionWrapper
from sb_arch_10M import SnakeCNN  # Make sure to import your CNN architecture

def main():
    # 1) Load map
    with open("maps.json") as f:
        map_dict = json.load(f)
    map_string = map_dict["empty"]
    
    # 2) Define paths
    submission_dir = "../../../workspace"
    log_dir = "bytefight_logs"
    opponents_dir = os.path.join(log_dir, "opponents")
    
    # 3) Load best model
    best_model_path = os.path.join(opponents_dir, "best_model.zip")
    if not os.path.exists(best_model_path):
        print(f"Error: No model found at {best_model_path}")
        return
        
    print(f"Loading model from {best_model_path}")
    model = PPO.load(best_model_path)
    print("Model loaded successfully!")
    
    # 4) Configure test environment
    test_env = SingleProcessByteFightEnv(
        map_string=map_string,
        opponent_module="sample_player",
        submission_dir=submission_dir,
        render_mode="human"  # Set to "human" for visualization
    )
    test_env = MaskedActionWrapper(test_env)
    
    # 5) Run test episodes
    num_episodes = 10
    wins, losses, ties = 0, 0, 0
    
    for episode in range(num_episodes):
        obs, _ = test_env.reset()
        done = False
        total_reward = 0
        turn_count = 0
        
        while not done:
            # Get action from trained model
            action, _ = model.predict(obs, deterministic=True)
            
            # Apply action to environment
            obs, reward, done, truncated, info = test_env.step(action)
            total_reward += reward
            turn_count += 1
        
        # Record result
        result = info.get('winner')
        print("RESULT: ", result)
        if result == "PLAYER_A":
            wins += 1
            result_str = "WIN"
        elif result == "PLAYER_B":
            losses += 1
            result_str = "LOSS"
        else:
            ties += 1
            result_str = "TIE"
            
        print(f"Episode {episode+1}: {result_str}, Reward: {total_reward:.2f}, Turns: {info.get('turn_count')}")
    
    # 6) Print final statistics
    print("\n===== Test Results =====")
    print(f"Total Episodes: {num_episodes}")
    print(f"Wins: {wins} ({wins/num_episodes*100:.1f}%)")
    print(f"Losses: {losses} ({losses/num_episodes*100:.1f}%)")
    print(f"Ties: {ties} ({ties/num_episodes*100:.1f}%)")
    
    test_env.close()

if __name__ == "__main__":
    main()