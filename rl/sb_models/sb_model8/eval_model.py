import os
import numpy as np
import time
import json
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from stable_baselines3 import PPO

# Import your custom components (updated for 10-action absolute space)
from bytefight_env import ByteFightSnakeEnv
from custom_policy3 import ByteFightMaskedPolicy, ByteFightFeaturesExtractor
from opp_controller import OppController

# ByteFight imports
import sys
parent_dir = os.path.abspath(os.path.join(__file__, "../../../.."))
sys.path.insert(0, parent_dir)
from game.game_map import Map
from game.board import Board
from game.player_board import PlayerBoard
from game.enums import Result

def get_map_string(map_name):
    """Load map string from maps.json file."""
    if not os.path.exists("maps.json"):
        raise FileNotFoundError("maps.json file not found. Please make sure it exists in the current directory.")
    
    with open("maps.json", "r") as f:
        maps = json.load(f)

    if map_name not in maps:
        available_maps = ", ".join(maps.keys())
        raise KeyError(f"Map '{map_name}' not found in maps.json. Available maps: {available_maps}")
    
    return maps[map_name]

def create_environment(map_names, render_mode="rgb_array"):
    """Create a ByteFight environment with the specified map."""
    # Load map string
    # Create opponent controller (same as in training)
    opponent = OppController(1)
    
    # Create the environment (it now uses an absolute action space of size 10)
    env = ByteFightSnakeEnv(
        map_names, 
        opponent, 
        render_mode=render_mode,
        verbose=False,
        use_opponent=True
    )
    
    return env

AVAILABLE_MAP_NAMES = [
    "pillars", "great_divide", "cage", "empty", "empty_large", "ssspline",
    "combustible_lemons", "arena", "ladder", "compasss", "recurve",
    "diamonds", "ssspiral", "lol", "attrition"
]

def evaluate_model(model_path, map_name="empty", num_episodes=5, render=True, save_video=False):
    """
    Evaluate a trained model on the ByteFight environment.
    
    The environment now uses an absolute action space (8 directions, TRAP, END_TURN).
    
    Args:
        model_path: Path to the trained model file.
        map_name: Name of the map to use.
        num_episodes: Number of episodes to run.
        render: Whether to render the environment.
        save_video: Whether to save a video of the gameplay.
    """
    # Create environment
    env = create_environment(map_name, render_mode="rgb_array" if render or save_video else None)
    
    # Load trained model with the updated policy components
    print(f"Loading model from {model_path}")
    try:
        model = PPO.load(
            model_path,
            env=env,
            custom_objects={
                "policy_class": ByteFightMaskedPolicy,
                "features_extractor_class": ByteFightFeaturesExtractor
            }
        )
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    # Statistics
    episode_rewards = []
    episode_lengths = []
    win_count = 0
    loss_count = 0
    tie_count = 0
    all_frames = []
    
    # Run evaluation episodes
    for episode in range(num_episodes):
        obs, info = env.reset()
        
        if not info.get("bidding_successful", True):
            print(f"Episode {episode+1}: Bidding failed, skipping...")
            continue
            
        episode_reward = 0
        episode_length = 0
        done = False
        episode_frames = []
        
        print(f"Starting episode {episode+1}/{num_episodes}")
        
        while not done:
            # Get action from model (actions now in 0..9)
            action, _states = model.predict(obs, deterministic=True)
                        
            # Convert numpy array to scalar if needed
            if isinstance(action, np.ndarray):
                action = action.item()  # Convert to Python scalar
                        
            # Step the environment
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            episode_reward += reward
            episode_length += 1
            
            # Render if needed
            if render or save_video:
                frame = env.render()
                if frame is not None and save_video:
                    episode_frames.append(frame)
                    
                # Print info about the current step
                print(f"Episode {episode+1}, Step {episode_length}")
                print(f"  Action: {action}")
                print(f"  Turn: {info['turn_counter']}, Move: {info['move_counter']}")
                
                if render and not save_video and frame is not None:
                    plt.figure(figsize=(8, 8))
                    plt.imshow(frame)
                    plt.title(f"Episode {episode+1}, Step {episode_length}")
                    plt.pause(0.01)
                    plt.close()
        
        # Record outcome
        if info.get("winner") == Result.PLAYER_A:
            outcome = "Win"
            win_count += 1
        elif info.get("winner") == Result.PLAYER_B:
            outcome = "Loss"
            loss_count += 1
        else:
            outcome = "Tie"
            tie_count += 1
            
        print(f"Episode {episode+1} finished: {outcome}")
        print(f"  Reward: {episode_reward:.2f}")
        print(f"  Length: {episode_length}")
        print(f"  Apples - Agent: {info['player_a_apples']}, Opponent: {info['player_b_apples']}")
        print()
        
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        
        if save_video:
            all_frames.append(episode_frames)
    
    print("\nEvaluation Results:")
    print(f"Episodes: {num_episodes}")
    print(f"Wins: {win_count} ({win_count/num_episodes:.1%})")
    print(f"Losses: {loss_count} ({loss_count/num_episodes:.1%})")
    print(f"Ties: {tie_count} ({tie_count/num_episodes:.1%})")
    print(f"Average Episode Length: {np.mean(episode_lengths):.1f} steps")
    print(f"Average Reward: {np.mean(episode_rewards):.2f}")
    
    if save_video and all_frames:
        save_gameplay_video(all_frames, f"bytefight_{map_name}_gameplay.mp4")
    
    env.close()

def save_gameplay_video(frames_list, filename, fps=10):
    """
    Save gameplay video from a list of episode frames.
    """
    print(f"Saving video to {filename}...")
    frames = frames_list[0]  # Save first episode for video
    
    fig, ax = plt.subplots(figsize=(10, 10))
    
    def update(frame_idx):
        ax.clear()
        ax.imshow(frames[frame_idx])
        ax.set_title(f"Frame {frame_idx}")
        ax.axis('off')
        return [ax]
    
    ani = FuncAnimation(fig, update, frames=len(frames), blit=True)
    ani.save(filename, fps=fps, dpi=100)
    print(f"Video saved to {filename}")

if __name__ == "__main__":
    # Path to the trained model
    model_path = "models/bytefight_ppo_600000_steps"
    
    # Evaluation settings
    #map_names = ["empty"]
    num_episodes = 100
    render = False
    save_video = False
    
    evaluate_model(
        model_path=model_path,
        map_name=AVAILABLE_MAP_NAMES,
        num_episodes=num_episodes,
        render=render,
        save_video=save_video
    )
