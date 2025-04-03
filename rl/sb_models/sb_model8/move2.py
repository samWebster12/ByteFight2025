import os
import sys
import numpy as np
import random
import matplotlib.pyplot as plt


parent_dir = os.path.abspath(os.path.join(__file__, "../../../.."))
sys.path.insert(1, parent_dir)

# Import game components
from game.enums import Action, Result
from game.game_map import Map
from game.player_board import PlayerBoard
from bytefight_env import ByteFightSnakeEnv
from opp_controller import OppController
from train_agent import get_map_string

class SimpleOpponent():
    """A simple opponent that always moves north."""
    def bid(self, player_board, time_left):
        return 0
        
    def play(self, player_board, time_left):
        return Action.NORTH
    

def move_obs(env, i, dir):
    print(f"\n=== {i+1} Move (FORWARD) ===")
    obs, reward, done, truncated, info = env.step(dir)

    # Extract scalar features and board image from the observation
    features = obs["features"]
    board_img = obs["board_image"]

    # From _build_observation(), we know:
    # Index 5 of features is player A's total length,
    # index 7 is player A's queued length,
    # index 6 is opponent's total length,
    # index 8 is opponent's queued length.
    my_length = features[5]
    my_queued_length = features[7]
    opp_length = features[6]
    opp_queued_length = features[8]

    # Head masks (channels 4 and 5)
    my_head_mask = board_img[4]
    opp_head_mask = board_img[5]

    # Find the coordinate where the head mask equals 1
    my_head_coords = np.argwhere(my_head_mask == 1)
    opp_head_coords = np.argwhere(opp_head_mask == 1)

    my_head_pos = my_head_coords[0] if len(my_head_coords) > 0 else None
    opp_head_pos = opp_head_coords[0] if len(opp_head_coords) > 0 else None

    # Body masks (channels 2 and 3)
    my_body_mask = board_img[2]
    opp_body_mask = board_img[3]

    # Find all coordinates where the snake bodies are present (nonzero)
    my_body_coords = np.argwhere(my_body_mask > 0)
    opp_body_coords = np.argwhere(opp_body_mask > 0)

    print(f"Player A Length after {i} move: {my_length}, queued: {my_queued_length}")
    print(f"Player A Head Position after {i} move: {my_head_pos}")
    print(f"Player A Body Coordinates:\n{my_body_coords}")

    print(f"\nOpponent Length after first move: {opp_length}, queued: {opp_queued_length}")
    print(f"Opponent Head Position after first move: {opp_head_pos}")
    print(f"Opponent Body Coordinates:\n{opp_body_coords}")

    print("\n=== End Turn ===")
    obs, reward, done, truncated, info = env.step(9)  # END_TURN



def debug_sacrifice_mechanic(verbose=True):
    """Debug the sacrifice mechanic implementation."""
    print("======= Sacrifice Mechanic Debugger =======")
    
    # Create a map with plenty of space
        
    # Create the environment
    env = ByteFightSnakeEnv(
        map_names=['empty_large'],
        opponent_controller=OppController(1),
        render_mode=None,
        verbose=verbose,
        use_opponent=False
    )
    
    # Reset the environment
    obs, info = env.reset()

    for i in range(12):
        if env.done:
            print("GAME OVER")
            break
        move_obs(env, i, 0)

    
    # Check the state after ending turn
    pb_a = PlayerBoard(True, env.board)
    length_after_end_turn = pb_a.get_length(enemy=False)
    
    print(f"Length after ending turn: {length_after_end_turn}")
    #print(f"Current sacrifice value: {env.current_sacrifice}")
    
    # Expected behavior:
    # 1. First move: No sacrifice, length unchanged
    # 2. Second move: 2 length sacrificed (3 at tail, 1 at head)
    # 3. Third move: 4 length sacrificed (5 at tail, 1 at head)
    # 4. End turn: Sacrifice counter reset to 0

if __name__ == "__main__":
    debug_sacrifice_mechanic()