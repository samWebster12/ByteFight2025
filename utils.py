from game import player_board
from game.enums import Action, Cell, Result
from collections.abc import Callable
import random
import numpy as np
import uuid
import os

def evaluate_position(b) -> float:
        if b.is_game_over():
            winner = b.game_board.winner
            if winner is not None:
                if winner == Result.TIE:
                    return 0.0
                is_current_player_A = b.get_am_player_a(enemy=False)
                if (winner == Result.PLAYER_A and is_current_player_A) or \
                   (winner == Result.PLAYER_B and not is_current_player_A):
                    return 9999.0
                else:
                    return -9999.0
            return 0.0

        my_apples  = b.get_apples_eaten(enemy=False)
        opp_apples = b.get_apples_eaten(enemy=True)

        my_len     = b.get_length(enemy=False)
        opp_len    = b.get_length(enemy=True)

        my_time  = b.get_time_left(enemy=False)
        opp_time = b.get_time_left(enemy=True)

        score  = 15.0 * (my_apples - opp_apples)
        score +=  5.0 * (my_len    - opp_len)
        score +=  0.1 * (my_time   - opp_time)

        return score