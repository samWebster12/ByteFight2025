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

    # Check the state after sixth move
    pb_a = PlayerBoard(True, env.forecast_board)
    
    print(f"Length after {i} move: {env.forecast_board.snake_a.get_length()}")
    print(f"Head position after {i} move: {env.forecast_board.snake_a.get_head_loc()}")
    #print(f"Current sacrifice value: {env.current_sacrifice}")
    
    # Debug snake's internal state
    snake = env.board.snake_a
    print(f"Physical length: {snake.get_unqueued_length()}")
    print(f"Total length: {snake.get_length()}")
    print(f"Direction: {snake.get_direction()}")


def move(env, i):

    print(f"\n=== {i+1} Move (FORWARD) ===")
    obs, reward, done, truncated, info = env.step(2)  # FORWARD again

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
        verbose=verbose
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

    for j in range(4, 8):
        sac(env, j)

    print("\n=== End Turn ===")
    obs, reward, done, truncated, info = env.step(6)  # END_TURN
    # Expected behavior:
    # 1. First move: No sacrifice, length unchanged
    # 2. Second move: 2 length sacrificed (3 at tail, 1 at head)
    # 3. Third move: 4 length sacrificed (5 at tail, 1 at head)
    # 4. End turn: Sacrifice counter reset to 0
    

if __name__ == "__main__":
    debug_sacrifice_mechanic()