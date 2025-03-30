from game import player_board
from game.enums import Action, Cell, Result
from collections.abc import Callable
import random
import numpy as np
import uuid
import os
import utils

class PlayerController:
    def __init__(self, time_left: Callable):
        self.minimum_moves = 5
        self.maximum_moves = 20
        self.save_interval = random.randint(self.minimum_moves, self.maximum_moves)
        self.move_count = 0
        self.output_dir = "saved_boards"
        os.makedirs(self.output_dir, exist_ok=True)

    def bid(self, board: player_board.PlayerBoard, time_left: Callable):
        return 0

    def play(self, board: player_board.PlayerBoard, time_left: Callable):
        self.move_count += 1

        if self.move_count >= self.save_interval:
            features = self.board_to_features(board)
            label = utils.evaluate_position(board)
            self.save_numpy_to_file(features, label)
            self.move_count = 0
            self.save_interval = random.randint(self.minimum_moves, self.maximum_moves)

        possible_moves = board.get_possible_directions()
        final_moves = [m for m in possible_moves if board.is_valid_move(m)]

        if not final_moves:
            return Action.FF

        return [random.choice(final_moves)]

    def board_to_features(self, b) -> np.ndarray:
        H, W = b.get_dim_y(), b.get_dim_x()
        walls    = np.zeros((H, W), dtype=np.float32)
        my_snake = np.zeros((H, W), dtype=np.float32)
        en_snake = np.zeros((H, W), dtype=np.float32)
        apples   = np.zeros((H, W), dtype=np.float32)
        my_traps = np.zeros((H, W), dtype=np.float32)
        en_traps = np.zeros((H, W), dtype=np.float32)

        walls[b.game_board.map.cells_walls > 0] = 1

        p_mask = b.get_snake_mask(my_snake=True, enemy_snake=False)
        my_snake[(p_mask == Cell.PLAYER_BODY.value) | (p_mask == Cell.PLAYER_HEAD.value)] = 1

        e_mask = b.get_snake_mask(my_snake=False, enemy_snake=True)
        en_snake[(e_mask == Cell.ENEMY_BODY.value) | (e_mask == Cell.ENEMY_HEAD.value)] = 1

        apples[b.get_apple_mask() > 0] = 1

        my_traps[b.get_trap_mask(my_traps=True, enemy_traps=False) > 0] = 1
        en_traps[b.get_trap_mask(my_traps=False, enemy_traps=True) > 0] = 1

        return np.stack([walls, my_snake, en_snake, apples, my_traps, en_traps], axis=0)

    def save_numpy_to_file(self, features: np.ndarray, label: float):
        unique_id = str(uuid.uuid4())
        filename = f"{unique_id}.npz"
        filepath = os.path.join(self.output_dir, filename)
        np.savez_compressed(filepath, x=features, y=label)

    
