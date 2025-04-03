import os
import sys
import numpy as np

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

def sac(env: ByteFightSnakeEnv, i):

    print(f"\n=== {i+1} Move (FORWARD) ===")
    obs, reward, done, truncated, info = env.step(2)  # FORWARD again

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

    current_sacrifice = features[2]

    # Find all coordinates where the snake bodies are present (nonzero)
    my_heading = features[3]
    opp_heading = features[4]
    
    print(f"Length after {i} move: {my_length}")
    print(f"Head position after {i} move: {my_head_coords}")
    #print(f"Current sacrifice value: {env.current_sacrifice}")
    

    print(f"Physical length: {my_length - my_queued_length}")
    print(f"Total length: {my_length}")
    print(f"Direction: {my_heading}")


    print(f"\n\nOppponent Length after first move: {opp_length}")
    print(f"Oppponent Head position after first move: {opp_head_coords}")

    # Debug snake's internal state
    print(f"Oppponent Physical length: {opp_length - opp_queued_length}")
    print(f"OppponentTotal length: {opp_length}")
    print(f"Oppponent Direction: {opp_heading}")

    print(f"\nCurrent Sacrifice: ", current_sacrifice)


def move(env, i):

    print(f"\n=== {i+1} Move (FORWARD) ===")
    obs, reward, done, truncated, info = env.step(0)  # FORWARD again
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

    current_sacrifice = features[2]

    # Find all coordinates where the snake bodies are present (nonzero)
    my_heading = features[3]
    opp_heading = features[4]
    
    print(f"Length after {i} move: {my_length}")
    print(f"Head position after {i} move: {my_head_coords}")
    #print(f"Current sacrifice value: {env.current_sacrifice}")
    

    print(f"Physical length: {my_length - my_queued_length}")
    print(f"Total length: {my_length}")
    print(f"Direction: {my_heading}")


    print(f"\n\nOppponent Length after first move: {opp_length}")
    print(f"Oppponent Head position after first move: {opp_head_coords}")

    # Debug snake's internal state
    print(f"Oppponent Physical length: {opp_length - opp_queued_length}")
    print(f"OppponentTotal length: {opp_length}")
    print(f"Oppponent Direction: {opp_heading}")

    print(f"\nCurrent Sacrifice: ", current_sacrifice)
    
    print("\n=== End Turn ===")
    obs, reward, done, truncated, info = env.step(9)  # END_TURN



def trap(env, i):

    print(f"\n=== {i+1} Trap ===")
    obs, reward, done, truncated, info = env.step(8)  # FORWARD again
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

    current_sacrifice = features[2]


    # Find all coordinates where the snake bodies are present (nonzero)
    my_heading = features[3]
    opp_heading = features[4]
    
    print(f"Length after {i} move: {my_length}")
    print(f"Head position after {i} move: {my_head_coords}")
    #print(f"Current sacrifice value: {env.current_sacrifice}")
    

    print(f"Physical length: {my_length - my_queued_length}")
    print(f"Total length: {my_length}")
    print(f"Direction: {my_heading}")


    print(f"\n\nOppponent Length after first move: {opp_length}")
    print(f"Oppponent Head position after first move: {opp_head_coords}")

    # Debug snake's internal state
    print(f"Oppponent Physical length: {opp_length - opp_queued_length}")
    print(f"OppponentTotal length: {opp_length}")
    print(f"Oppponent Direction: {opp_heading}")

    print(f"\nCurrent Sacrifice: ", current_sacrifice)

    
    print("\n=== End Turn ===")
\
def debug_sacrifice_mechanic(verbose=True):
    """Debug the sacrifice mechanic implementation."""
    print("======= Sacrifice Mechanic Debugger =======")
    
    # Create a map with plenty of space
    map_string = "19,19#3,9#15,9#9#2##100,7,Vertical#0000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000#0"
    
    game_map = Map(map_string)
    
    # Create the environment
    env = ByteFightSnakeEnv(
        game_map=game_map,
        opponent_controller=SimpleOpponent(),
        render_mode=None,
        verbose=verbose,
        use_opponent=False
    )
    
    # Reset the environment
    obs, info = env.reset()
    
    # Log initial state
    pb = PlayerBoard(True, env.board)
    initial_length = pb.get_length(enemy=False)
    initial_head_pos = pb.get_head_location(enemy=False)
    initial_unqueued_length = pb.get_unqueued_length(enemy=False)
    
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
    
    for i in range(4):
        move(env, i)

    for j in range(4, 6):
        sac(env, j)
    
    trap(env, 6)

    print("\n=== End Turn ===")
    obs, reward, done, truncated, info = env.step(6)  # END_TURN
    # Expected behavior:
    # 1. First move: No sacrifice, length unchanged
    # 2. Second move: 2 length sacrificed (3 at tail, 1 at head)
    # 3. Third move: 4 length sacrificed (5 at tail, 1 at head)
    # 4. End turn: Sacrifice counter reset to 0
    

if __name__ == "__main__":
    debug_sacrifice_mechanic()