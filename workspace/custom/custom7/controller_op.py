import math
import random
import collections
import numpy as np
from collections.abc import Callable

import numba
from numba import njit

from game import player_board
from game.enums import Action, Cell

INF = 999999

def board_to_key(board: player_board.PlayerBoard) -> tuple:
    """
    Create a simple hashable key for transposition.
    We'll store:
      - Our head position
      - Enemy head position
      - Our length
      - Enemy length
      - The turn_count
      - Whether it's our turn
    """
    my_head = board.get_head_location(enemy=False)
    enemy_head = board.get_head_location(enemy=True)
    my_len = board.get_length(enemy=False)
    enemy_len = board.get_length(enemy=True)
    turn_count = board.get_turn_count()
    am_i_turn = 1 if board.is_my_turn() else 0

    return (
        int(my_head[0]), int(my_head[1]),
        int(enemy_head[0]), int(enemy_head[1]),
        my_len, enemy_len,
        turn_count,
        am_i_turn
    )

# Precompute allowed directions table (for headings 0-7, allowed moves are ±45° and ±90°).
allowed_dirs_table = np.empty((8, 5), dtype=np.int32)
for h in range(8):
    allowed = [ (h + diff) % 8 for diff in [-2, -1, 0, 1, 2] ]
    allowed_dirs_table[h, :] = allowed

# Precompute direction offsets as a (8,2) numpy array.
direction_offsets_np = np.array([
    [ 0, -1],  # NORTH
    [ 1, -1],  # NORTHEAST
    [ 1,  0],  # EAST
    [ 1,  1],  # SOUTHEAST
    [ 0,  1],  # SOUTH
    [-1,  1],  # SOUTHWEST
    [-1,  0],  # WEST
    [-1, -1]   # NORTHWEST
], dtype=np.int32)

@njit
def numba_bfs_with_turn(occupancy, dim_x, dim_y, start_x, start_y, start_heading, allowed_dirs_table, direction_offsets):
    # Force start_heading to a plain int by doing arithmetic:
    sh = start_heading % 8
    dist = np.full((dim_y, dim_x, 8), INF, dtype=np.int32)
    dist[start_y, start_x, sh] = 0

    max_states = dim_x * dim_y * 8
    queue = np.empty((max_states, 3), dtype=np.int32)
    head_idx = 0
    tail_idx = 0
    queue[tail_idx, 0] = start_x
    queue[tail_idx, 1] = start_y
    queue[tail_idx, 2] = sh
    tail_idx += 1

    while head_idx < tail_idx:
        cx = queue[head_idx, 0]
        cy = queue[head_idx, 1]
        chead = queue[head_idx, 2]
        head_idx += 1
        cur_cost = dist[cy, cx, chead]
        for j in range(5):
            new_heading = allowed_dirs_table[chead, j]
            dx = direction_offsets[new_heading, 0]
            dy = direction_offsets[new_heading, 1]
            nx = cx + dx
            ny = cy + dy
            if nx < 0 or nx >= dim_x or ny < 0 or ny >= dim_y:
                continue
            if occupancy[ny, nx] != 0:
                continue
            if dist[ny, nx, new_heading] > cur_cost + 1:
                dist[ny, nx, new_heading] = cur_cost + 1
                queue[tail_idx, 0] = nx
                queue[tail_idx, 1] = ny
                queue[tail_idx, 2] = new_heading
                tail_idx += 1
    return dist

@njit
def numba_union_reachable(dist_3d):
    dim_y, dim_x, _ = dist_3d.shape
    mask = np.zeros((dim_y, dim_x), dtype=np.uint8)
    for y in range(dim_y):
        for x in range(dim_x):
            min_val = INF
            for h in range(8):
                if dist_3d[y, x, h] < min_val:
                    min_val = dist_3d[y, x, h]
            if min_val < INF:
                mask[y, x] = 1
    return mask

def build_occupancy(board: player_board.PlayerBoard) -> np.ndarray:
    dim_x = board.get_dim_x()
    dim_y = board.get_dim_y()
    occ = np.zeros((dim_y, dim_x), dtype=np.int32)
    for y in range(dim_y):
        for x in range(dim_x):
            cell = board.cell_occupied_by(x, y)
            if cell != Cell.SPACE:
                occ[y, x] = 1
    return occ

class PlayerController:
    def __init__(self, time_left: Callable):
        self.time_left = time_left
        self.lookahead_depth = 15
        self.survival_cache = {}
        self.max_enemy_moves = 4

    def bid(self, board: player_board.PlayerBoard, time_left: Callable):
        return 0

    def play(self, board: player_board.PlayerBoard, time_left: Callable):
        phase = self.determine_phase(board)
        moves = board.get_possible_directions()
        valid_moves = [m for m in moves if board.is_valid_move(m)]
        if not valid_moves:
            return [Action.FF]
        safe_moves = []
        for mv in valid_moves:
            next_board, valid = board.forecast_turn([mv], check_validity=True)
            if not valid:
                continue
            if self.is_losing_state(next_board):
                continue
            if self.can_survive_floodfill(next_board):
                safe_moves.append(mv)
        if not safe_moves:
            return [random.choice(valid_moves)]
        if phase == "FARMING":
            chosen = self.farming_phase(board, safe_moves)
        elif phase == "ATTACK":
            chosen = self.attack_phase(board, safe_moves)
        else:
            chosen = self.defense_phase(board, safe_moves)
        return [chosen]

    # Numba-accelerated BFS-based survival check with turning restrictions
    def can_survive_floodfill(self, board: player_board.PlayerBoard) -> bool:
        if board.is_game_over():
            return not self.is_losing_state(board)
        my_length = board.get_length(enemy=False)
        if my_length < board.get_min_player_size():
            return False
        my_head = board.get_head_location(enemy=False)
        my_heading = board.get_direction(enemy=False)
        if my_heading is None:
            my_heading = Action.NORTH
        if isinstance(my_heading, int):
            my_heading_val = my_heading
        else:
            my_heading_val = my_heading.value
        enemy_head = board.get_head_location(enemy=True)
        enemy_heading = board.get_direction(enemy=True)
        if enemy_heading is None:
            enemy_heading = Action.NORTH
        if isinstance(enemy_heading, int):
            enemy_heading_val = enemy_heading
        else:
            enemy_heading_val = enemy_heading.value

        dim_x = board.get_dim_x()
        dim_y = board.get_dim_y()
        occ = build_occupancy(board)
        my_dist_3d = numba_bfs_with_turn(occ, dim_x, dim_y, int(my_head[0]), int(my_head[1]), my_heading_val,
                                          allowed_dirs_table, direction_offsets_np)
        enemy_dist_3d = numba_bfs_with_turn(occ, dim_x, dim_y, int(enemy_head[0]), int(enemy_head[1]), enemy_heading_val,
                                             allowed_dirs_table, direction_offsets_np)
        my_min = np.min(my_dist_3d, axis=2)
        enemy_min = np.min(enemy_dist_3d, axis=2)
        safe_mask = (my_min < enemy_min) & (my_min < INF)
        visited = np.zeros((dim_y, dim_x), dtype=np.uint8)
        sx, sy = int(my_head[0]), int(my_head[1])
        if not safe_mask[sy, sx]:
            return False
        queue = collections.deque()
        queue.append((sx, sy))
        visited[sy, sx] = 1
        count = 1
        while queue:
            cx, cy = queue.popleft()
            for i in range(8):
                dx = direction_offsets_np[i, 0]
                dy = direction_offsets_np[i, 1]
                nx = cx + dx
                ny = cy + dy
                if 0 <= nx < dim_x and 0 <= ny < dim_y:
                    if safe_mask[ny, nx] and visited[ny, nx] == 0:
                        visited[ny, nx] = 1
                        queue.append((nx, ny))
                        count += 1
        return count >= my_length

    # Phase Decision
    def determine_phase(self, board: player_board.PlayerBoard) -> str:
        my_apples = board.get_apples_eaten()
        enemy_apples = board.get_apples_eaten(enemy=True)
        my_len = board.get_length()
        enemy_len = board.get_length(enemy=True)
        min_size = board.get_min_player_size()
        if (my_apples < enemy_apples) or (my_len <= min_size + 2):
            return "FARMING"
        if (my_apples > enemy_apples) or (my_len >= enemy_len + 2):
            return "ATTACK"
        return "FARMING"

    def farming_phase(self, board: player_board.PlayerBoard, safe_moves: list[Action]) -> Action:
        apples = board.get_current_apples()
        if apples.size == 0:
            return random.choice(safe_moves)
        head = board.get_head_location()
        nearest_apple = None
        nearest_dist = 999999
        for ax, ay in apples:
            dist = abs(ax - head[0]) + abs(ay - head[1])
            if dist < nearest_dist:
                nearest_dist = dist
                nearest_apple = (ax, ay)
        best_move = None
        best_dist = nearest_dist
        for mv in safe_moves:
            next_board, ok = board.forecast_turn([mv], check_validity=True)
            if not ok:
                continue
            new_head = next_board.get_head_location()
            dist_new = abs(nearest_apple[0] - new_head[0]) + abs(nearest_apple[1] - new_head[1])
            if dist_new < best_dist:
                best_move = mv
                best_dist = dist_new
        if best_move is None:
            return random.choice(safe_moves)
        return best_move

    def defense_phase(self, board: player_board.PlayerBoard, safe_moves: list[Action]) -> Action:
        return random.choice(safe_moves)

    def attack_phase(self, board: player_board.PlayerBoard, safe_moves: list[Action]) -> Action:
        enemy_head = board.get_head_location(enemy=True)
        best_move = None
        best_score = None
        for mv in safe_moves:
            nxt_board, ok = board.forecast_turn([mv], check_validity=True)
            if not ok:
                continue
            enemy_safe = self.count_enemy_safe_moves(nxt_board)
            my_new_head = nxt_board.get_head_location(enemy=False)
            dist = abs(my_new_head[0] - enemy_head[0]) + abs(my_new_head[1] - enemy_head[1])
            move_score = (-enemy_safe, -dist)
            if best_score is None or move_score > best_score:
                best_score = move_score
                best_move = mv
        if not best_move:
            return random.choice(safe_moves)
        return best_move

    def count_enemy_safe_moves(self, board: player_board.PlayerBoard) -> int:
        enemy_moves = board.get_possible_directions(enemy=True)
        safe_count = 0
        for em in enemy_moves:
            if board.is_valid_move(em, enemy=True):
                enemy_forecast, ok = board.forecast_turn([em], check_validity=True)
                if ok and not self.is_losing_state(enemy_forecast):
                    if self.can_survive_floodfill(enemy_forecast):
                        safe_count += 1
        return safe_count

    def is_losing_state(self, board: player_board.PlayerBoard) -> bool:
        if board.is_game_over():
            return True
        my_len = board.get_length()
        min_len = board.get_min_player_size()
        return my_len < min_len
