import os
import numpy as np
import time
import json
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from stable_baselines3 import PPO

# Import your custom components
from bytefight_env import ByteFightSnakeEnv
from custom_policy import ByteFightMaskedPolicy, ByteFightFeaturesExtractor
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
    """Load map string from maps.json file"""
    if not os.path.exists("maps.json"):
        raise FileNotFoundError("maps.json file not found. Please make sure it exists in the current directory.")
    
    with open("maps.json", "r") as f:
        maps = json.load(f)

    if map_name not in maps:
        available_maps = ", ".join(maps.keys())
        raise KeyError(f"Map '{map_name}' not found in maps.json. Available maps: {available_maps}")
    
    return maps[map_name]


def create_environment(map_name="empty", render_mode="rgb_array"):
    """Create a ByteFight environment with the specified map"""
    # Load map string
    map_string = get_map_string(map_name)
    print(f"Loaded map '{map_name}' for evaluation")
    
    # Create Map, Board, and PlayerBoard
    game_map = Map(map_string)
    
    # Create opponent controller (same as in training)
    opponent = OppController(1)
    
    # Create the environment
    env = ByteFightSnakeEnv(
        game_map, 
        opponent, 
        render_mode=render_mode,
        handle_bidding=True,
        verbose=False
    )
    
    return env


def evaluate_model(model_path, map_name="empty", num_episodes=5, render=True, save_video=False):
    """
    Evaluate a trained model on ByteFight environment
    
    Args:
        model_path: Path to the trained model file
        map_name: Name of the map to use
        num_episodes: Number of episodes to run
        render: Whether to render the environment
        save_video: Whether to save a video of the gameplay
    """
    # Create environment
    env = create_environment(map_name, render_mode="rgb_array" if render or save_video else None)
    
    # Load trained model
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
            # Get action from model
            # Get action from model
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
                
                # Optional visualization
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
        
        # Save episode statistics
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        
        # Save frames for this episode
        if save_video:
            all_frames.append(episode_frames)
    
    # Print summary statistics
    print("\nEvaluation Results:")
    print(f"Episodes: {num_episodes}")
    print(f"Wins: {win_count} ({win_count/num_episodes:.1%})")
    print(f"Losses: {loss_count} ({loss_count/num_episodes:.1%})")
    print(f"Ties: {tie_count} ({tie_count/num_episodes:.1%})")
    print(f"Average Episode Length: {np.mean(episode_lengths):.1f} steps")
    print(f"Average Reward: {np.mean(episode_rewards):.2f}")
    
    # Save video if requested
    if save_video and all_frames:
        save_gameplay_video(all_frames, f"bytefight_{map_name}_gameplay.mp4")
    
    # Close the environment
    env.close()


def save_gameplay_video(frames_list, filename, fps=10):
    """
    Save gameplay video from a list of episode frames
    """
    print(f"Saving video to {filename}...")
    
    # Select first episode for video (can be modified to save all episodes)
    frames = frames_list[0]
    
    # Create figure for animation
    fig, ax = plt.subplots(figsize=(10, 10))
    
    def update(frame_idx):
        ax.clear()
        ax.imshow(frames[frame_idx])
        ax.set_title(f"Frame {frame_idx}")
        ax.axis('off')
        return [ax]
    
    # Create animation
    ani = FuncAnimation(fig, update, frames=len(frames), blit=True)
    
    # Save animation
    ani.save(filename, fps=fps, dpi=100)
    print(f"Video saved to {filename}")


if __name__ == "__main__":
    # Path to the trained model
    model_path = "models/bytefight_ppo_final.zip"
    
    # Evaluation settings
    map_name = "empty"  # Same map used for training
    num_episodes = 5    # Number of episodes to evaluate
    render = False       # Whether to render the environment
    save_video = False   # Whether to save a video of gameplay
    
    # Run evaluation
    evaluate_model(
        model_path=model_path,
        map_name=map_name,
        num_episodes=num_episodes,
        render=render,
        save_video=save_video
    )