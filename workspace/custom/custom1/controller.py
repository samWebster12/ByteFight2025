import math
import random
from collections.abc import Callable

from game import player_board
from game.enums import Action

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
    This is not a perfect representation of the entire board, but 
    it's a decent approximation that helps caching repeated states.
    """
    # My head (x, y)
    my_head = board.get_head_location(enemy=False)
    # Enemy head (x, y)
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


class PlayerController:
    def __init__(self, time_left: Callable):
        self.time_left = time_left
        self.lookahead_depth = 10  # We can do up to 10 steps. This is large!
        # Transposition table for can_survive: (board_key, depth) -> bool
        self.survival_cache = {}
        # We'll also prune the enemy's moves if they have too many
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
            # Multi-step check up to 'self.lookahead_depth'
            if self.can_survive(next_board, self.lookahead_depth):
                safe_moves.append(mv)

        if not safe_moves:
            # if none pass the multi-step test, fallback to a random valid
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

        #if (my_apples < enemy_apples) or (my_len <= min_size + 2):
        #if my_len < 7:
        #    return "FARMING"
        #else:
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
            next_board, valid = board.forecast_turn([mv], check_validity=True)
            if not valid:
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

    # -------------------------------------------------------------------------
    # Multi-Step Survival Check (with transposition + partial pruning)
    # -------------------------------------------------------------------------
    def can_survive(self, board: player_board.PlayerBoard, depth: int) -> bool:
        """
        Returns True if there's a sequence of moves leading to survival 
        for 'depth' steps (yours + enemy's). 
        We do a worst-case approach for enemy.
        Potentially expensive, so we add caching + partial pruning.
        """
        # 1) Transposition
        key = (board_to_key(board), depth)
        if key in self.survival_cache:
            return self.survival_cache[key]

        # 2) Base
        if depth <= 0:
            # survived enough steps
            self.survival_cache[key] = True
            return True
        if self.is_losing_state(board):
            self.survival_cache[key] = False
            return False

        # 3) My turn
        if board.is_my_turn():
            my_moves = []
            possible = board.get_possible_directions()
            for m in possible:
                if board.is_valid_move(m):
                    my_moves.append(m)

            if not my_moves:
                self.survival_cache[key] = False
                return False

            # If ANY move leads to survival => True
            for mv in my_moves:
                nxt_board, ok = board.forecast_turn([mv], check_validity=True)
                if not ok:
                    continue
                if self.is_losing_state(nxt_board):
                    continue
                if self.can_survive(nxt_board, depth - 1):
                    self.survival_cache[key] = True
                    return True
            self.survival_cache[key] = False
            return False
        else:
            # 4) Enemy turn => worst-case
            # We'll do partial pruning: limit the # of enemy moves
            bcopy = board.get_copy()
            bcopy.reverse_perspective()
            enemy_moves = []
            for m in bcopy.get_possible_directions(enemy=False):
                if bcopy.is_valid_move(m, enemy=False):
                    enemy_moves.append(m)

            # no enemy moves => they are stuck => we survive
            if not enemy_moves:
                self.survival_cache[key] = True
                return True

            # partial pruning
            if len(enemy_moves) > self.max_enemy_moves:
                enemy_moves = random.sample(enemy_moves, self.max_enemy_moves)

            # If the enemy can choose ANY move that kills us => we die
            for em in enemy_moves:
                opp_board, ok = board.forecast_turn([em], check_validity=True)
                if not ok:
                    # skip it or consider it a kill?
                    continue
                if self.is_losing_state(opp_board):
                    self.survival_cache[key] = False
                    return False
                if not self.can_survive(opp_board, depth - 1):
                    self.survival_cache[key] = False
                    return False

            # If no forced kill found => we survive
            self.survival_cache[key] = True
            return True

    # -------------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------------
    def is_losing_state(self, board: player_board.PlayerBoard) -> bool:
        if board.is_game_over():
            return True
        my_len = board.get_length()
        min_len = board.get_min_player_size()
        return (my_len < min_len)
