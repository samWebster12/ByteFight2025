import numpy as np
import sys
import os
from typing import List, Tuple, Dict, Any

# Adjust these paths as needed to import your game files
parent_dir = os.path.abspath(os.path.join(__file__, "../../../.."))
sys.path.insert(0, parent_dir)

from game.enums import Action, Result
from game.game_map import Map
from game.player_board import PlayerBoard
from game.board import Board
from opp_controller import OppController
from bytefight_env import ByteFightSnakeEnv  # Import your environment

class DummyOpponent:
    """A simple opponent that always moves forward and rarely places traps"""
    
    def bid(self, board, time_left):
        """Always bid 0"""
        return 0
    
    def play(self, board, time_left):
        """Just move forward once and end turn"""
        # Get current direction
        current_dir = board.get_direction(enemy=False)
        if current_dir is None:
            # If no direction, go NORTH
            return [Action.NORTH]
        return [current_dir]

def print_observation(obs, info, title="Current Observation"):
    """Print observation data from the environment"""
    print(f"\n=== {title} ===")
    
    # Print scalar features
    features = obs['features']
    print(f"Move count: {int(features[0])}")
    print(f"Turn count: {int(features[1])}")
    print(f"Current sacrifice: {int(features[2])}")
    print(f"Player length: {int(features[5])} (queued: {int(features[8])})")
    print(f"Enemy length: {int(features[6])} (queued: {int(features[9])})")
    
    # Print info
    print("\nInfo:")
    print(f"  Current actions: {info.get('current_actions', [])}")
    print(f"  Player length: {info.get('player_a_length', 'N/A')}")
    print(f"  Enemy length: {info.get('player_b_length', 'N/A')}")
    
    # Print valid actions
    action_mask = obs['action_mask']
    valid_actions = [i for i, valid in enumerate(action_mask) if valid]
    print(f"Valid actions: {valid_actions}")

def run_sacrifice_test():
    """Run tests to verify sacrifice mechanics using observations"""
    # Create the environment with verbose mode on
    test_map = Map("19,19#3,9#15,9#9#2##100,7,Vertical#0000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000#0")
    opponent = DummyOpponent()
    env = ByteFightSnakeEnv(test_map, opponent, verbose=True)
    
    # Reset the environment
    obs, info = env.reset()
    print_observation(obs, info, "Initial State")
    
    # Test 1: Take two consecutive moves (should apply sacrifice on the second)
    print("\n=== TEST 1: Basic Sacrifice Mechanics ===")
    
    # First move (no sacrifice)
    action = 2  # FORWARD
    print(f"Taking action: {action} ({AGENT_ACTIONS[action]})")
    obs, reward, done, truncated, info = env.step(action)
    print_observation(obs, info, "After First Move")
    
    # Second move (should have sacrifice 2)
    action = 2  # FORWARD again
    print(f"Taking action: {action} ({AGENT_ACTIONS[action]})")
    obs, reward, done, truncated, info = env.step(action)
    print_observation(obs, info, "After Second Move (Should Apply Sacrifice)")
    
    # End turn
    action = 6  # END_TURN
    print(f"Taking action: {action} ({AGENT_ACTIONS[action]})")
    obs, reward, done, truncated, info = env.step(action)
    print_observation(obs, info, "After First Turn")
    
    # Test 2: Longer sequence with increasing sacrifices
    print("\n=== TEST 2: Increasing Sacrifices ===")
    
    # Take 3 moves in a row to see sacrifices increase
    actions_to_take = [2, 2, 2, 6]  # FORWARD, FORWARD, FORWARD, END_TURN
    expected_sacrifices = [0, 2, 4]  # Expected sacrifices for each move
    
    for i, action in enumerate(actions_to_take):
        sacrifice_note = f" (Expected sacrifice: {expected_sacrifices[i]})" if i < 3 else ""
        print(f"Taking action: {action} ({AGENT_ACTIONS[action]}){sacrifice_note}")
        obs, reward, done, truncated, info = env.step(action)
        print_observation(obs, info, f"After Action {i+1}")
    
    # Test 3: Verify that trap consumes length correctly
    print("\n=== TEST 3: Trap Length Consumption ===")
    
    # Move forward once, then place a trap
    actions_to_take = [2, 5, 6]  # FORWARD, TRAP, END_TURN
    
    for i, action in enumerate(actions_to_take):
        print(f"Taking action: {action} ({AGENT_ACTIONS[action]})")
        obs, reward, done, truncated, info = env.step(action)
        print_observation(obs, info, f"After Action {i+1}")
    
    # Clean up
    env.close()
    print("\n=== Tests completed! ===")

if __name__ == "__main__":
    # Define the mapping for readable output
    AGENT_ACTIONS = {
        0: "LEFT",
        1: "FORWARD_LEFT",
        2: "FORWARD",
        3: "FORWARD_RIGHT",
        4: "RIGHT",
        5: "TRAP",
        6: "END_TURN",
    }
    
    # Run the test
    run_sacrifice_test()