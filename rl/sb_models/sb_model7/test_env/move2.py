import os
import sys
import numpy as np
import random
import matplotlib.pyplot as plt


# Set up parent directories for module imports
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, parent_dir)

parent_dir = os.path.abspath(os.path.join(__file__, "../../../../.."))
sys.path.insert(1, parent_dir)

# Import game components
from game.enums import Action, Result
from game.game_map import Map
from game.player_board import PlayerBoard
from bytefight_env import ByteFightSnakeEnv
from opp_controller import OppController

class SimpleOpponent():
    """A simple opponent that always moves north."""
    def bid(self, player_board, time_left):
        return 0
        
    def play(self, player_board, time_left):
        return Action.NORTH
    

def choose_random_dir():
    acts = [0, 1, 2, 3, 4]
    return random.choice(acts)


def display_obs_single_grid(obs):
    """
    Creates a single 2D color-coded image from all 9 channels in obs["board_image"].
    
    Each channel is assigned a distinct color. If multiple channels overlap in the same cell,
    the later channel in the list overrides the earlier channels.

    Example channel order (ByteFight):
      0: Walls
      1: Apples
      2: My Snake Body
      3: Opponent Snake Body
      4: My Head
      5: Opponent Head
      6: My Trap
      7: Opponent Trap
      8: Portal

    The final result is a single H×W RGB image where each channel appears in its own color.
    """
    if "board_image" not in obs:
        print("No board_image found in obs!")
        return
    
    board_img = obs["board_image"]  # shape (9, H, W)
    num_channels, height, width = board_img.shape
    
    # Define a color for each channel as (R, G, B) in 0..255
    # Feel free to adjust these colors to your liking.
    channel_colors = {
        0: (150, 150, 150),  # Walls  (gray)
        1: (255,   0,   0),  # Apples (red)
        2: (  0, 200,   0),  # My Body (green)
        3: (  0, 120, 255),  # Opp Body (blue-ish)
        4: (  0, 255,   0),  # My Head (bright green)
        5: (  0, 180, 255),  # Opp Head (lighter blue)
        6: (255, 255,   0),  # My Trap (yellow)
        7: (255, 128,   0),  # Opp Trap (orange)
        8: (255,   0, 255),  # Portal (magenta)
    }
    
    # If any channel is missing a color, default to white
    default_color = (255, 255, 255)
    
    # Create an empty RGB canvas, initialized to black
    # We'll do 3 channels (R,G,B) in 0..255
    merged_image = np.zeros((height, width, 3), dtype=np.uint8)
    
    # For each channel in ascending order:
    #  - For each pixel that is non-zero, set that pixel's color
    #  - Later channels override earlier ones at overlapping pixels
    for c in range(num_channels):
        color = channel_colors.get(c, default_color)
        mask = board_img[c] > 0
        merged_image[mask] = color
    
    # Plot the merged image
    plt.figure(figsize=(8, 8))
    plt.title("Single-Grid Observation Overlay")
    plt.imshow(merged_image)
    plt.axis("off")
    plt.show()
    
    # Optionally, print scalar features and action mask
    if "features" in obs:
        print("Features:\n", obs["features"])
    if "action_mask" in obs:
        print("Action mask:\n", obs["action_mask"])

def move_obs(env, i, dir):
    print(f"\n=== {i+1} Move (FORWARD) ===")
    obs, reward, done, truncated, info = env.step(dir)
    display_obs_single_grid(obs)

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
    obs, reward, done, truncated, info = env.step(6)  # END_TURN



def move(env, i, dir):

    print(f"\n=== {i+1} Move (FORWARD) ===")
    obs, reward, done, truncated, info = env.step(dir)

    # Check the state after sixth move
    pb_a = PlayerBoard(True, env.board)
    length_after_sixth = pb_a.get_length(enemy=False)
    head_after_sixth = pb_a.get_head_location(enemy=False)
    
    print(f"Length after {i} move: {length_after_sixth}")
    print(f"Head position after {i} move: {head_after_sixth}")
    #print(f"Current sacrifice value: {env.current_sacrifice}")
    
    # Debug snake's internal state
    snake = env.board.snake_a
    print(f"Physical length: {snake.get_unqueued_length()}")
    print(f"Total length: {snake.get_length()}")
    print(f"Direction: {snake.get_direction()}")

    pb_b = PlayerBoard(False, env.board)
    length_after_first = pb_b.get_length(enemy=False)
    head_after_first = pb_b.get_head_location(enemy=False)
    
    print(f"\n\nOppponent Length after first move: {length_after_first}")
    print(f"Oppponent Head position after first move: {head_after_first}")
    #print(f"Oppponent Current sacrifice value: {env.current_sacrifice}")

    # Debug snake's internal state
    snake = env.board.snake_b
    print(f"Oppponent Physical length: {snake.get_unqueued_length()}")
    print(f"OppponentTotal length: {snake.get_length()}")
    print(f"Oppponent Direction: {snake.get_direction()}")
    
    print("\n=== End Turn ===")
    obs, reward, done, truncated, info = env.step(6)  # END_TURN

def move_without_end_turn(env, i, dir):

    print(f"\n=== {i+1} Move (FORWARD) ===")
    obs, reward, done, truncated, info = env.step(dir)  # FORWARD again

    # Check the state after sixth move
    pb_a = PlayerBoard(True, env.board)
    length_after_sixth = pb_a.get_length(enemy=False)
    head_after_sixth = pb_a.get_head_location(enemy=False)
    
    print(f"Length after {i} move: {length_after_sixth}")
    print(f"Head position after {i} move: {head_after_sixth}")
    #print(f"Current sacrifice value: {env.current_sacrifice}")
    
    # Debug snake's internal state
    snake = env.board.snake_a
    print(f"Physical length: {snake.get_unqueued_length()}")
    print(f"Total length: {snake.get_length()}")
    print(f"Direction: {snake.get_direction()}")

    pb_b = PlayerBoard(False, env.board)
    length_after_first = pb_b.get_length(enemy=False)
    head_after_first = pb_b.get_head_location(enemy=False)
    
    print(f"\n\nOppponent Length after first move: {length_after_first}")
    print(f"Oppponent Head position after first move: {head_after_first}")
    #print(f"Oppponent Current sacrifice value: {env.current_sacrifice}")

    # Debug snake's internal state
    snake = env.board.snake_b
    print(f"Oppponent Physical length: {snake.get_unqueued_length()}")
    print(f"OppponentTotal length: {snake.get_length()}")
    print(f"Oppponent Direction: {snake.get_direction()}")



def debug_sacrifice_mechanic(verbose=True):
    """Debug the sacrifice mechanic implementation."""
    print("======= Sacrifice Mechanic Debugger =======")
    
    # Create a map with plenty of space
    map_string = "19,19#3,9#15,9#9#2##100,7,Vertical#0000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000#0"
    
    game_map = Map(map_string)
    
    # Create the environment
    env = ByteFightSnakeEnv(
        game_map=game_map,
        opponent_controller=OppController(1),
        render_mode=None,
        verbose=verbose
    )
    
    # Force snake to have a specific length for easier testing
    env.board.snake_a.start(env.board.snake_a.get_head_loc(), 10)
    
    # Reset the environment
    obs, info = env.reset()
    
    # Log initial state
    pb_a = PlayerBoard(True, env.board)
    initial_length = pb_a.get_length(enemy=False)
    initial_head_pos = pb_a.get_head_location(enemy=False)
    initial_unqueued_length = pb_a.get_unqueued_length(enemy=False)
    
    print(f"Initial snake length: {initial_length}")
    print(f"Initial head position: {initial_head_pos}")
    print(f"Initial unqueued length: {initial_unqueued_length}")
    #print(f"Initial sacrifice value: {env.current_sacrifice}")
    
    # Debug the snake's internal structure
    print("\nSnake internal structure:")
    snake = env.board.snake_a
    # Removed the queue attribute access
    print(f"Physical length: {snake.get_unqueued_length()}")
    print(f"Total length: {snake.get_length()}")
    print(f"Direction: {snake.get_direction()}")
    
    for i in range(12):
        if env.done:
            print("GAME OVER")
            break
        move_obs(env, i, choose_random_dir())

    
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