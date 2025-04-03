from game import player_board
from game.enums import Action, Cell, Result
from collections.abc import Callable
import random
import numpy as np
import uuid
import os
import torch
import torch.nn as nn

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

def board_to_features(b: player_board.PlayerBoard) -> np.ndarray:
    H, W = b.get_dim_y(), b.get_dim_x()
    walls    = np.zeros((H, W), dtype=np.float32)
    my_snake = np.zeros((H, W), dtype=np.float32)
    en_snake = np.zeros((H, W), dtype=np.float32)
    apples   = np.zeros((H, W), dtype=np.float32)
    my_traps = np.zeros((H, W), dtype=np.float32)
    en_traps = np.zeros((H, W), dtype=np.float32)

    walls[b.game_board.map.cells_walls > 0] = 1

    # Always from *current* player's perspective
    p_mask = b.get_snake_mask(my_snake=True, enemy_snake=False)
    my_snake[(p_mask == Cell.PLAYER_BODY.value) | (p_mask == Cell.PLAYER_HEAD.value)] = 1

    e_mask = b.get_snake_mask(my_snake=False, enemy_snake=True)
    en_snake[(e_mask == Cell.ENEMY_BODY.value) | (e_mask == Cell.ENEMY_HEAD.value)] = 1

    apples[b.get_apple_mask() > 0] = 1

    my_traps[b.get_trap_mask(my_traps=True, enemy_traps=False) > 0] = 1
    en_traps[b.get_trap_mask(my_traps=False, enemy_traps=True) > 0] = 1

    return np.stack([walls, my_snake, en_snake, apples, my_traps, en_traps], axis=0)  # shape (6, H, W)

class TinyCNN(nn.Module):
    def __init__(self, input_shape):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(6, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))  # Global avg pool
        )
        self.fc = nn.Linear(64, 1)

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)
