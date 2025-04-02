import os
import sys
import numpy as np
import random

# Set up parent directories for module imports
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, parent_dir)
parent_dir = os.path.abspath(os.path.join(__file__, "../../../.."))
sys.path.insert(0, parent_dir)

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

def move_without_end_turn(env, move_num):
    """
    Makes a directional move (FORWARD, action 2) without ending the turn.
    This accumulates the action in the current turn.
    """
    print(f"\n=== Move {move_num}: FORWARD (without ending turn) ===")
    acts = [0, 1, 2, 3, 4]
    move_num = random.choice(acts)
    obs, reward, done, truncated, info = env.step(move_num)  # Action 2: FORWARD
    pb_a = PlayerBoard(True, env.forecast_board)
    print(f"Agent Length after move {move_num}: {pb_a.get_length(enemy=False)}")
    print(f"Agent Head position after move {move_num}: {pb_a.get_head_location(enemy=False)}")
    return done

def manual_end_turn(env):
    """Ends the current turn by issuing the END_TURN action (action 6)."""
    print("\n=== Manually Ending Turn ===")
    obs, reward, done, truncated, info = env.step(6)  # Action 6: END_TURN
    return done

def attempt_trap(env, attempt_num):
    """
    Attempts to place a trap (action 5) after at least one directional move.
    Checks forecast validity via pb.is_valid_trap() before proceeding.
    """
    print(f"\n=== Trap Attempt {attempt_num} ===")
    pb = PlayerBoard(True, env.forecast_board)
    trap_valid = pb.is_valid_trap()
    print(f"Forecast: is_valid_trap() returns: {trap_valid}")
    
    done = False  # Initialize 'done' with a default value
    if trap_valid:
        obs, reward, done, truncated, info = env.step(5)  # Action 5: TRAP
        print("Trap attempted.")
    else:
        print("Trap forecast invalid; trap not attempted.")
    
    pb_a = PlayerBoard(True, env.forecast_board)
    print(f"Agent Length after trap placement: {pb_a.get_length(enemy=False)}")
    print(f"Agent Head position after trap placement: {pb_a.get_head_location(enemy=False)}")
        
    return done

def reset_env(game_map, verbose):
    # Create the environment with our SimpleOpponent
    env = ByteFightSnakeEnv(
        game_map=game_map,
        opponent_controller=SimpleOpponent(),
        render_mode=None,
        verbose=verbose
    )
    
    # Force agent snake to have a starting length of 10 for easier testing    
    # Reset the environment
    obs, info = env.reset()
    
    pb_a = PlayerBoard(True, env.board)
    print(f"Initial agent snake length: {pb_a.get_length(enemy=False)}")
    print(f"Initial agent head position: {pb_a.get_head_location(enemy=False)}")
    print(f"Initial agent unqueued length: {pb_a.get_unqueued_length(enemy=False)}")
    
    print("\nAgent snake internal structure:")
    snake = env.board.snake_a
    print(f"Physical length: {snake.get_unqueued_length()}")
    print(f"Total length: {snake.get_length()}")
    print(f"Direction: {snake.get_direction()}")

    return env

def verify_trap_presence(env):
    """Prints whether a trap placed by the agent is on the map."""
    pb = PlayerBoard(True, env.board)
    # Get a mask for traps placed by your snake.
    trap_mask = pb.get_trap_mask(my_traps=True, enemy_traps=False)
    # If any cell in the trap mask is nonzero, then a trap exists.
    if np.any(trap_mask > 0):
        print("Trap is present on the map.")
    else:
        print("No trap found on the map.")
    
    # Alternatively, you can print a board string if available:
    board_str = env.board.get_board_string()
    #print("Board state:")
    #print(board_str)


def debug_trap_mechanic(verbose=True):
    """Test trap placement after at least one directional move in a turn."""
    print("======= Trap Mechanic Debugger =======")
    
    # Create a spacious map
    map_string = ("19,19#3,9#15,9#9#2##100,7,Vertical#0000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000#0")
    game_map = Map(map_string)
    
    
    # --- Test 1: Attempt Trap on First Move (should be disallowed) ---
    print("\n--- Test 1: Attempt Trap on First Move (should be disallowed) ---")
    env = reset_env(game_map, verbose)

    obs, reward, done, truncated, info = env.step(5)  # TRAP on first move
    if done:
        print("Trap on first move correctly disallowed; episode ended with loss.")
    else:
        print("ERROR: Trap on first move was allowed unexpectedly.")
        return

    # Reset environment for further testing
    
    # --- Test 2: Make one directional move (without ending turn), then attempt trap ---
    print("\n--- Test 2: Move without ending turn, then attempt trap ---")
    env = reset_env(game_map, verbose)
    move_without_end_turn(env, 1)
    # Now attempt trap placement
    attempt_trap(env, 1)
    manual_end_turn(env)
    
    # --- Test 3: Multiple moves in one turn before attempting trap ---
    
    print("\n--- Test 3: Multiple moves in same turn then trap ---")
    env = reset_env(game_map, verbose)
    move_without_end_turn(env, 1)
    move_without_end_turn(env, 1)
    move_without_end_turn(env, 1)
    verify_trap_presence(env)
    attempt_trap(env, 2)
    manual_end_turn(env)
    verify_trap_presence(env)
    
    # Final state check
    pb_a = PlayerBoard(True, env.board)
    print(f"\nFinal agent snake length: {pb_a.get_length(enemy=False)}")

if __name__ == "__main__":
    debug_trap_mechanic()
