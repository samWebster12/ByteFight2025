from game import player_board
from game.enums import Action
from collections.abc import Callable
import random
import math

class PlayerControllerMinimax:
    def __init__(self, time_left: Callable):
        # Set a maximum search depth. Adjust this value based on performance testing.
        self.max_depth = 5

    def bid(self, board: player_board.PlayerBoard, time_left: Callable):
        # This bot always bids 0.
        return 0

    def play(self, board: player_board.PlayerBoard, time_left: Callable):
        best_move = self.minimax_decision(board, self.max_depth)
        if best_move is None:
            return Action.FF  # Forfeit if no valid move found
        return [best_move]

    def minimax_decision(self, board: player_board.PlayerBoard, depth: int):
        best_value = -math.inf
        best_move = None
        alpha = -math.inf
        beta = math.inf
        for move in board.get_possible_directions():
            if board.is_valid_move(move):
                # Create a copy of the board state.
                board_copy = board.get_copy()
                # Forecast the turn with the move (non-mutating simulation).
                board_copy, valid = board_copy.forecast_turn([move])
                if not valid:
                    continue
                value = self.min_value(board_copy, depth - 1, alpha, beta)
                if value > best_value:
                    best_value = value
                    best_move = move
                alpha = max(alpha, best_value)
        return best_move

    def max_value(self, board: player_board.PlayerBoard, depth: int, alpha: float, beta: float):
        if depth == 0 or self.is_terminal(board):
            return self.evaluate_board(board)
        value = -math.inf
        for move in board.get_possible_directions():
            if board.is_valid_move(move):
                board_copy = board.get_copy()
                board_copy, valid = board_copy.forecast_turn([move])
                if not valid:
                    continue
                value = max(value, self.min_value(board_copy, depth - 1, alpha, beta))
                if value >= beta:
                    return value  # Beta cutoff
                alpha = max(alpha, value)
        return value

    def min_value(self, board: player_board.PlayerBoard, depth: int, alpha: float, beta: float):
        if depth == 0 or self.is_terminal(board):
            return self.evaluate_board(board)
        value = math.inf
        for move in board.get_possible_directions():
            if board.is_valid_move(move):
                board_copy = board.get_copy()
                board_copy, valid = board_copy.forecast_turn([move])
                if not valid:
                    continue
                value = min(value, self.max_value(board_copy, depth - 1, alpha, beta))
                if value <= alpha:
                    return value  # Alpha cutoff
                beta = min(beta, value)
        return value

    def evaluate_board(self, board: player_board.PlayerBoard):
        # Get basic snake metrics
        my_length = board.get_length()
        enemy_length = board.get_length(enemy=True)
        length_diff = my_length - enemy_length

        # Apple metrics
        my_apples = board.get_apples_eaten()
        enemy_apples = board.get_apples_eaten(enemy=True)
        apple_diff = my_apples - enemy_apples

        # Calculate Manhattan distance to the closest apple
        my_head = board.get_head_location()
        apples = board.get_current_apples()  # Expects a numpy array of apple coordinates (x,y)
        if apples.size > 0:
            distances = [abs(my_head[0] - a[0]) + abs(my_head[1] - a[1]) for a in apples]
            closest_apple_distance = min(distances)
        else:
            closest_apple_distance = 0

        # Adjust apple distance weight if snake is small.
        # When near the minimum allowed size, apples are even more important.
        min_size = board.get_min_player_size()
        apple_distance_weight = (board.get_length(enemy=True) - board.get_length()) / 3

        if my_length <= 10:
            apple_distance_weight = 2.0  # Increase weight when small
        else:
            apple_distance_weight = 0.1

        # Compute the distance to the nearest wall (using board boundaries)
        x, y = my_head
        dim_x = board.get_dim_x()
        dim_y = board.get_dim_y()
        # Distance to the closest wall is the minimum distance from the head to any board edge.
        dist_to_wall = min(x, dim_x - x - 1, y, dim_y - y - 1)
        wall_penalty = 0
        # If too close to a wall (for example, less than 2 cells away), apply a penalty.
        if dist_to_wall < 2:
            wall_penalty = (2 - dist_to_wall) * 1.0  # Adjust weight as necessary

        # Combine all factors into a single score.
        # Higher score means a better board state.
        score = (length_diff * 1.0) + (apple_diff * 0.5) - (closest_apple_distance * apple_distance_weight) - wall_penalty
        return score
    
    def evaluate_board_2(self, board: player_board.PlayerBoard):
        # Basic metrics: snake lengths and apples eaten
        my_length = board.get_length()
        enemy_length = board.get_length(enemy=True)
        length_diff = my_length - enemy_length
        
        # Additional factors
        my_apples = board.get_apples_eaten()
        enemy_apples = board.get_apples_eaten(enemy=True)
        apple_diff = my_apples - enemy_apples

        # Distance to closest apple (using Manhattan distance)
        my_head = board.get_head_location()
        apples = board.get_current_apples()  # Returns a numpy array of apple coordinates
        if apples.size > 0:
            distances = [abs(my_head[0] - a[0]) + abs(my_head[1] - a[1]) for a in apples]
            closest_apple_distance = min(distances)
        else:
            closest_apple_distance = 0
        
        # Combine factors using weights (tune these as needed)
        score = (length_diff * 1.0) + (apple_diff * 0.5) - (closest_apple_distance * 0.1)
        return score
        
    def is_terminal(self, board: player_board.PlayerBoard):
        # The game is terminal if either snake's length drops below the minimum allowed size.
        min_size = board.get_min_player_size()
        return board.get_length() < min_size or board.get_length(enemy=True) < min_size
