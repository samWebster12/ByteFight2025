import math
import random
import numpy as np
from collections.abc import Callable

# Import from numba.experimental to ensure jitclass is available
from numba.experimental import jitclass
import numba
from numba import njit

from game import player_board
from game.enums import Action, Cell


# --------------------------------------------------------------------------
# 1) A JIT-Class specification for a numeric BoardState
# --------------------------------------------------------------------------
spec = [
    ('my_head_x', numba.int32),
    ('my_head_y', numba.int32),
    ('enemy_head_x', numba.int32),
    ('enemy_head_y', numba.int32),
    ('my_length', numba.int32),
    ('enemy_length', numba.int32),
    ('turn_count', numba.int32),
    ('is_my_turn_flag', numba.boolean),
    # A 2D numpy array for collisions, walls, etc.
    ('grid', numba.int8[:, :])
]

@jitclass(spec)
class BoardState:
    """
    A minimal numeric representation of the board for Numba-based recursion.
    We'll store:
      - (my_head_x, my_head_y)
      - (enemy_head_x, enemy_head_y)
      - my_length, enemy_length
      - turn_count, is_my_turn_flag
      - grid (2D) indicating walls, player snake, enemy snake, etc.
    """
    def __init__(
        self,
        my_head_x,
        my_head_y,
        enemy_head_x,
        enemy_head_y,
        my_length,
        enemy_length,
        turn_count,
        is_my_turn_flag,
        grid
    ):
        self.my_head_x = my_head_x
        self.my_head_y = my_head_y
        self.enemy_head_x = enemy_head_x
        self.enemy_head_y = enemy_head_y
        self.my_length = my_length
        self.enemy_length = enemy_length
        self.turn_count = turn_count
        self.is_my_turn_flag = is_my_turn_flag
        self.grid = grid

    def copy(self):
        """Returns a copy of this state for forecast logic."""
        return BoardState(
            self.my_head_x,
            self.my_head_y,
            self.enemy_head_x,
            self.enemy_head_y,
            self.my_length,
            self.enemy_length,
            self.turn_count,
            self.is_my_turn_flag,
            self.grid.copy()
        )


# --------------------------------------------------------------------------
# 2) Convert the PlayerBoard -> BoardState with correct dimension methods
# --------------------------------------------------------------------------
def convert_board(board: player_board.PlayerBoard) -> BoardState:
    """
    Convert your PlayerBoard into a numeric BoardState for use by Numba.
    We'll demonstrate building a grid with:
      - -1 for walls
      -  1 for your snake
      -  2 for enemy snake
      -  0 for empty space
    """
    my_head = board.get_head_location(enemy=False)
    enemy_head = board.get_head_location(enemy=True)

    my_len = board.get_length(enemy=False)
    enemy_len = board.get_length(enemy=True)

    turn_count = board.get_turn_count()
    is_my_turn_flag = board.is_my_turn()

    # Dimensions from docs
    width = board.get_dim_x()
    height = board.get_dim_y()

    # Create a 2D array to store walls/snake
    grid = np.zeros((height, width), dtype=np.int8)

    # Fill in walls
    wall_mask = board.get_wall_mask()  # 2D array with 1 where there's a wall
    for y in range(height):
        for x in range(width):
            if wall_mask[y, x] == 1:
                grid[y, x] = -1

    # Fill in snake positions using get_snake_mask
    #   According to docs, snake positions are indicated by various Cell enum values.
    #   We'll treat *my* snake as 1, enemy snake as 2, if not a wall already.
    snake_mask = board.get_snake_mask(my_snake=True, enemy_snake=True)
    for y in range(height):
        for x in range(width):
            if grid[y, x] == -1:
                continue  # Already a wall
            val = snake_mask[y, x]
            # If val is PLAYER_HEAD or PLAYER_BODY
            if val == Cell.PLAYER_HEAD or val == Cell.PLAYER_BODY:
                grid[y, x] = 1
            # If val is ENEMY_HEAD or ENEMY_BODY
            elif val == Cell.ENEMY_HEAD or val == Cell.ENEMY_BODY:
                grid[y, x] = 2
            else:
                # 0 = empty
                pass

    return BoardState(
        my_head_x=int(my_head[0]),
        my_head_y=int(my_head[1]),
        enemy_head_x=int(enemy_head[0]),
        enemy_head_y=int(enemy_head[1]),
        my_length=my_len,
        enemy_length=enemy_len,
        turn_count=turn_count,
        is_my_turn_flag=is_my_turn_flag,
        grid=grid
    )


# --------------------------------------------------------------------------
# 3) Numba helper functions: losing state, move generation, forecast, etc.
# --------------------------------------------------------------------------
@njit
def is_losing_state_numba(state: BoardState) -> bool:
    """
    Simple check: if my_length < 2 or other conditions, it's losing.
    You can expand this based on your actual game rules.
    """
    if state.my_length < 2:
        return True
    return False

@njit
def is_my_turn_numba(state: BoardState) -> bool:
    return state.is_my_turn_flag

@njit
def get_possible_moves_numba(state: BoardState, for_me: bool) -> numba.typed.List:
    """
    Return up to 8 possible directions (N,NE,E,SE,S,SW,W,NW), 
    or maybe just 4 if you prefer cardinal only.
    We rely on the docs that PlayerBoard can handle diagonal moves,
    but it's your choice whether you allow them or not.
    
    For demonstration, we return the list of all 8 directions from Action:
      0: NORTH
      1: NORTHEAST
      2: EAST
      3: SOUTHEAST
      4: SOUTH
      5: SOUTHWEST
      6: WEST
      7: NORTHWEST
    In your real code, you might want to filter out invalid directions 
    early if you want.
    """
    moves = numba.typed.List.empty_list(numba.int32)
    for m in range(8):
        moves.append(m)
    return moves

@njit
def forecast_move_numba(state: BoardState, move: int, for_me: bool) -> (BoardState, bool):
    """
    Apply 'move' to the board state, return (new_state, ok).
    - If out of bounds or hits a wall => ok=False
    - We do not handle partial sacrifice logic here because that is 
      typically done by PlayerBoard. This is just a demonstration 
      of how you might track collisions in numeric form.
    """

    new_state = state.copy()
    # The grid shape
    height, width = new_state.grid.shape

    # Decide whose head to move
    if for_me:
        x = new_state.my_head_x
        y = new_state.my_head_y
    else:
        x = new_state.enemy_head_x
        y = new_state.enemy_head_y

    # Map each move int to dx,dy
    # 0:N,1:NE,2:E,3:SE,4:S,5:SW,6:W,7:NW
    # You can adjust these to match your actual Action enums
    dir_map = [
        (0, -1),  # N
        (1, -1),  # NE
        (1,  0),  # E
        (1,  1),  # SE
        (0,  1),  # S
        (-1, 1),  # SW
        (-1,0),   # W
        (-1,-1)   # NW
    ]
    dx, dy = dir_map[move]
    nx, ny = x + dx, y + dy

    # Check bounds or wall
    if nx < 0 or nx >= width or ny < 0 or ny >= height:
        return (new_state, False)
    if new_state.grid[ny, nx] == -1:  # It's a wall in our numeric grid
        return (new_state, False)

    # Apply the "move"
    if for_me:
        new_state.my_head_x = nx
        new_state.my_head_y = ny
    else:
        new_state.enemy_head_x = nx
        new_state.enemy_head_y = ny

    # Toggle turn
    new_state.is_my_turn_flag = not new_state.is_my_turn_flag
    new_state.turn_count += 1

    # We'll say it's always "ok" if we didn't collide with walls or OOB
    return (new_state, True)


@njit
def can_survive_numba(state: BoardState, depth: int, max_enemy_moves: int) -> bool:
    """
    Returns True if there's a sequence of moves leading to survival 
    for 'depth' steps (my turn + enemy turn), using worst-case logic 
    for the enemy. No dictionary caching here (pure nopython).
    """
    # Base
    if depth <= 0:
        return True
    if is_losing_state_numba(state):
        return False

    # My turn
    if is_my_turn_numba(state):
        my_moves = get_possible_moves_numba(state, for_me=True)
        if len(my_moves) == 0:
            return False

        # If ANY move leads to survival => True
        for mv in my_moves:
            nxt_state, ok = forecast_move_numba(state, mv, for_me=True)
            if not ok:
                continue
            if is_losing_state_numba(nxt_state):
                continue
            if can_survive_numba(nxt_state, depth - 1, max_enemy_moves):
                return True
        return False

    # Enemy turn => worst-case
    else:
        enemy_moves = get_possible_moves_numba(state, for_me=False)
        if len(enemy_moves) == 0:
            return True

        # partial pruning
        if len(enemy_moves) > max_enemy_moves:
            # For demonstration, just slice the first N
            enemy_moves = enemy_moves[:max_enemy_moves]

        # If the enemy can choose ANY move that kills us => we lose
        for em in enemy_moves:
            opp_state, ok = forecast_move_numba(state, em, for_me=False)
            if not ok:
                # could skip or treat as kill
                continue
            if is_losing_state_numba(opp_state):
                return False
            if not can_survive_numba(opp_state, depth - 1, max_enemy_moves):
                return False

        # No forced kill => survive
        return True


# --------------------------------------------------------------------------
# 4) A lightweight board_to_key function for your Python caching
# --------------------------------------------------------------------------
def board_to_key(board: player_board.PlayerBoard) -> tuple:
    """
    A simple key for (board, depth) caching.
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

# --------------------------------------------------------------------------
# 5) Main PlayerController
# --------------------------------------------------------------------------
class PlayerController:
    def __init__(self, time_left: Callable):
        self.time_left = time_left
        self.lookahead_depth = 12
        self.survival_cache = {}   # (board_key, depth) -> bool
        self.max_enemy_moves = 4   # partial pruning

    def bid(self, board: player_board.PlayerBoard, time_left: Callable):
        # For demonstration, always bid 0
        return 0

    def play(self, board: player_board.PlayerBoard, time_left: Callable):
        phase = self.determine_phase(board)

        moves = board.get_possible_directions()
        valid_moves = [m for m in moves if board.is_valid_move(m)]
        if not valid_moves:
            # No valid moves => must forfeit
            return [Action.FF]

        safe_moves = []
        for mv in valid_moves:
            nxt_board, ok = board.forecast_turn([mv], check_validity=True)
            if not ok:
                continue
            if self.is_losing_state(nxt_board):
                continue
            # Check multi-step survival
            if self.can_survive_wrapper(nxt_board, self.lookahead_depth):
                safe_moves.append(mv)

        if not safe_moves:
            # fallback if none pass the test
            return [random.choice(valid_moves)]

        if phase == "FARMING":
            chosen = self.farming_phase(board, safe_moves)
        else:
            chosen = self.defense_phase(board, safe_moves)

        return [chosen]

    # -------------------------------------------------------------------------
    # Phase Decision
    # -------------------------------------------------------------------------
    def determine_phase(self, board: player_board.PlayerBoard) -> str:
        my_apples = board.get_apples_eaten()
        enemy_apples = board.get_apples_eaten(enemy=True)
        my_len = board.get_length()
        min_size = board.get_min_player_size()

        # simplistic logic
        if (my_apples < enemy_apples) or (my_len <= min_size + 2):
            return "FARMING"
        else:
            return "DEFENSE"

    # -------------------------------------------------------------------------
    # Farming / Defense
    # -------------------------------------------------------------------------
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
            nxt_board, ok = board.forecast_turn([mv], check_validity=True)
            if not ok:
                continue
            new_head = nxt_board.get_head_location()
            dist_new = abs(nearest_apple[0] - new_head[0]) + abs(nearest_apple[1] - new_head[1])
            if dist_new < best_dist:
                best_move = mv
                best_dist = dist_new

        if best_move is None:
            return random.choice(safe_moves)
        return best_move

    def defense_phase(self, board: player_board.PlayerBoard, safe_moves: list[Action]) -> Action:
        return random.choice(safe_moves)

    # -------------------------------------------------------------------------
    # Multi-Step Survival Check with a Python-level cache
    # -------------------------------------------------------------------------
    def can_survive_wrapper(self, board: player_board.PlayerBoard, depth: int) -> bool:
        """
        1) Build a key from the board
        2) If in cache, return it
        3) Otherwise convert board -> numeric BoardState
           and run can_survive_numba
        """
        key = (board_to_key(board), depth)
        if key in self.survival_cache:
            return self.survival_cache[key]

        state = convert_board(board)
        result = can_survive_numba(state, depth, self.max_enemy_moves)
        self.survival_cache[key] = result
        return result

    # -------------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------------
    def is_losing_state(self, board: player_board.PlayerBoard) -> bool:
        # Simple check: game over or length < min size
        if board.is_game_over():
            return True
        my_len = board.get_length()
        min_len = board.get_min_player_size()
        return (my_len < min_len)