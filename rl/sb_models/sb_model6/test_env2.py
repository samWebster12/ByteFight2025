import os
import json
import gymnasium as gym
import numpy as np

import os, sys
parent_dir = os.path.abspath(os.path.join(__file__, "../../../.."))
sys.path.insert(0, parent_dir)

# Import game components
from game.enums import Action, Result
from game.game_map import Map
from game.player_board import PlayerBoard

# Import your Gym environment
# (Adjust the import if your environment is located in a different module)
from bytefight_env import ByteFightSnakeEnv

# --- Dummy Opponent Controller ---
class DummyOppController:
    def bid(self, board, time_left):
        # Always bid 0 for simplicity.
        return 0

    def play(self, board, time_left):
        # Always play FORWARD (discrete action id 2)
        return 2

# --- Helper to load a map from maps.json ---
def load_map_from_json(map_name, json_path="maps.json"):
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"Could not find {json_path}")
    with open(json_path, "r") as f:
        maps_dict = json.load(f)
    if map_name not in maps_dict:
        raise ValueError(f"Map {map_name} not found in {json_path}")
    map_string = maps_dict[map_name]
    return Map(map_string)

# --- Evaluation Test Functions ---

def evaluate_bidding(env):
    print("=== Evaluating Bidding Resolution ===")
    obs, info = env.reset()
    print("Bidding info:", info)
    print("Environment started after bidding.\n")

def evaluate_moves(env):
    print("=== Evaluating Move Sequence & Sacrifice Rules ===")
    # Reset environment and get initial state.
    obs, info = env.reset()
    pb = PlayerBoard(True, env.board)
    initial_length = pb.get_length(enemy=False)
    print("Initial snake length:", initial_length)

    # Make the first move: FORWARD (action id 2).
    print("\nPerforming first move (FORWARD).")
    obs, reward, done, truncated, info = env.step(2)
    pb = PlayerBoard(True, env.board)
    length_after_first = pb.get_length(enemy=False)
    print("After first move, snake length:", length_after_first)
    print("Current sacrifice value:", env.current_sacrifice)
    
    # Second move: FORWARD.
    print("\nPerforming second move (FORWARD).")
    obs, reward, done, truncated, info = env.step(2)
    pb = PlayerBoard(True, env.board)
    length_after_second = pb.get_length(enemy=False)
    print("After second move, snake length:", length_after_second)
    print("Current sacrifice value:", env.current_sacrifice)
    
    # Third move: FORWARD.
    print("\nPerforming third move (FORWARD).")
    obs, reward, done, truncated, info = env.step(2)
    pb = PlayerBoard(True, env.board)
    length_after_third = pb.get_length(enemy=False)
    print("After third move, snake length:", length_after_third)
    print("Current sacrifice value (should have increased by 2):", env.current_sacrifice)
    
    # End turn.
    print("\nEnding turn (END_TURN).")
    obs, reward, done, truncated, info = env.step(6)
    print("After ending turn, reward:", reward, "done:", done, "\n")

def evaluate_trap(env):
    print("=== Evaluating Trap Placement Rules ===")
    obs, info = env.reset()
    pb = PlayerBoard(True, env.board)
    initial_length = pb.get_length(enemy=False)
    print("Initial snake length:", initial_length)
    
    # Make two moves so that trap placement becomes available.
    print("Performing two moves (FORWARD) to enable trap action.")
    env.step(2)
    env.step(2)
    
    # Now try placing a trap (action id 5).
    print("Placing trap (TRAP action).")
    obs, reward, done, truncated, info = env.step(5)
    print("After trap action, reward:", reward, "done:", done, "\n")
    # (If the snake did not have enough unqueued length for a trap,
    # the game should end with a loss.)

def evaluate_invalid_move(env):
    print("=== Evaluating Invalid Move (Expecting Game Over) ===")
    obs, info = env.reset()
    print("Attempting an invalid move (e.g., moving LEFT into a wall).")
    # Action id 0 (LEFT) may be invalid depending on starting position.
    obs, reward, done, truncated, info = env.step(0)
    if done:
        print("Game ended as expected due to an invalid move. Reward:", reward)
    else:
        print("Invalid move did not end the game as expected (test failed).")
    print("")

def main():
    # Load a map from maps.json (choose one of your keys, e.g., "pillars")
    try:
        game_map = load_map_from_json("pillars", json_path="maps.json")
    except Exception as e:
        print("Error loading map:", e)
        return

    # Instantiate the dummy opponent.
    dummy_opp = DummyOppController()

    # Create the ByteFight environment with verbose debug enabled.
    env = ByteFightSnakeEnv(game_map, dummy_opp, render_mode="human", verbose=True)
    
    # Run evaluation tests.
    evaluate_bidding(env)
    evaluate_moves(env)
    evaluate_trap(env)
    evaluate_invalid_move(env)
    
    print("=== Evaluation Complete ===")

if __name__ == "__main__":
    main()
