from game import player_board
from game.enums import Action, Result
from collections.abc import Callable
import utils

class PlayerController:
    def __init__(self, time_left: Callable):
        pass

    def bid(self, board: player_board.PlayerBoard, time_left: Callable):
        return 0

    def play(self, board: player_board.PlayerBoard, time_left: Callable):
        # 1) Find all valid single-step moves
        possible_moves = board.get_possible_directions()
        valid_moves = [mv for mv in possible_moves if board.is_valid_move(mv)]

        # If none are valid, forfeit
        if not valid_moves:
            return Action.FF

        # 2) Evaluate each valid move and pick the best one
        best_move = None
        best_score = -float('inf')

        for mv in valid_moves:
            # Forecast applying this move on a copy
            forecasted_board, success = board.forecast_move(mv, check_validity=True)
            if not success:
                continue  # Skip if something made it invalid

            # Evaluate that forecasted board
            score = utils.evaluate_position(forecasted_board)

            # Track the best
            if score > best_score:
                best_score = score
                best_move = mv

        # 3) Return the single best move
        if best_move is None:
            return Action.FF

        return [best_move]

    
