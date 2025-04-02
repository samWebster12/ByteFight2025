from game import player_board
from game.enums import Action
from collections.abc import Callable
import random

class OppController:
    # for the controller to read
    def __init__(self, time_left: Callable):
        print("[OPP DEBUG] Opponent controller initialized")
        return

    def bid(self, board:player_board.PlayerBoard, time_left:Callable):
        return 0

    def play(self, board:player_board.PlayerBoard, time_left:Callable):
        verbose = False

        # Debug board state
        head_loc = board.get_head_location()
        direction = board.get_direction()
        length = board.get_length()
        
        if verbose:
            print(f"[OPP DEBUG] Head location: {head_loc}, Direction: {direction}, Length: {length}")
        
        possible_moves = board.get_possible_directions()
        if verbose:
            print(f"[OPP DEBUG] Possible directions: {possible_moves}")
        
        # Check each possible move
        final_moves = []
        for move in possible_moves:
            is_valid = board.is_valid_move(move)
            if is_valid:
                # Check where this move would lead
                try:
                    next_loc = board.get_loc_after_move(move)
                    in_bounds = board.cell_in_bounds(next_loc)
                    occupied = False
                    if in_bounds:
                        x, y = next_loc[0], next_loc[1]
                        occupied = board.is_occupied(x, y)
                    
                    if verbose:
                        print(f"[OPP DEBUG] Move {move} would lead to {next_loc}, In bounds: {in_bounds}, Occupied: {occupied}")
                    final_moves.append(move)
                except Exception as e:
                    if verbose:
                        print(f"[OPP DEBUG] Error checking move {move}: {str(e)}")
            else:
                if verbose:
                    print(f"[OPP DEBUG] Move {move} is not valid")
        
        if verbose:
            print(f"[OPP DEBUG] Final valid moves: {final_moves}")
        if len(final_moves) == 0:
            if verbose:
                print("[OPP DEBUG] NO VALID MOVES! Forfeiting...")
            return Action.FF
        
        final_move = random.choice(final_moves)
        if verbose:
            print(f"[OPP DEBUG] Chosen move: {final_move}")
        return [final_move]