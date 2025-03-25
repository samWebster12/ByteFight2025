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

class PlayerController:
    def __init__(self, time_left: Callable):
        self.time_left = time_left
        self.lookahead_depth = 15  # used in survival checks
        self.survival_cache = {}  # transposition table for can_survive
        self.max_enemy_moves = 4   # partial pruning for enemy in can_survive

        # We'll define a separate depth for the Attack minimax
        self.attack_minimax_depth = 6  # can specify as needed

    def bid(self, board: player_board.PlayerBoard, time_left: Callable):
        return 0

    def play(self, board: player_board.PlayerBoard, time_left: Callable):
        phase = self.determine_phase(board)

        moves = board.get_possible_directions()
        valid_moves = [m for m in moves if board.is_valid_move(m)]
        if not valid_moves:
            return [Action.FF]

        # 1) Filter to "safe" moves
        safe_moves = []
        for mv in valid_moves:
            nxt_board, valid = board.forecast_turn([mv], check_validity=True)
            if not valid:
                continue
            if self.is_losing_state(nxt_board):
                continue
            # Multi-step check up to 'self.lookahead_depth'
            if self.can_survive(nxt_board, self.lookahead_depth):
                safe_moves.append(mv)

        if not safe_moves:
            # If none pass the survival test, fallback to a random valid
            return [random.choice(valid_moves)]

        # 2) Choose a move based on the phase
        self.lookahead_depth = min(16, board.get_length() + 1)
        if phase == "FARMING":
            chosen = self.farming_phase(board, safe_moves)
        elif phase == "ATTACK":
            # *** multi-step minimax approach with alpha-beta pruning ***
            chosen = self.attack_phase(board, safe_moves)
        else:
            chosen = self.defense_phase(board, safe_moves)

        return [chosen]

    # -------------------------------------------------------------------------
    # Phase Decision
    # -------------------------------------------------------------------------
    def determine_phase(self, board: player_board.PlayerBoard) -> str:
        """
        Decide whether to FARM, ATTACK, or DEFEND.
        - If behind on apples or length near min => FARM
        - If significantly ahead => ATTACK
        - Else => DEFENSE
        """
        my_apples = board.get_apples_eaten()
        enemy_apples = board.get_apples_eaten(enemy=True)
        my_len = board.get_length()
        enemy_len = board.get_length(enemy=True)
        min_size = board.get_min_player_size()

        # If behind on apples or just above min size => farm
        if (my_apples < enemy_apples) or (my_len <= min_size + 2):
            return "FARMING"

        # If I'm ahead in length or apples => attack
        # (You wrote >= enemy_len - 2 previously, but let's keep it as you like)
        if (my_apples > enemy_apples) or (my_len >= enemy_len + 2):
            return "ATTACK"

        return "FARMING"

    # -------------------------------------------------------------------------
    # Farming
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
            nxt_board, valid = board.forecast_turn([mv], check_validity=True)
            if not valid:
                continue
            new_head = nxt_board.get_head_location()
            dist_new = abs(nearest_apple[0] - new_head[0]) + abs(nearest_apple[1] - new_head[1])
            if dist_new < best_dist:
                best_dist = dist_new
                best_move = mv

        if best_move is None:
            return random.choice(safe_moves)
        return best_move

    # -------------------------------------------------------------------------
    # Defense (unchanged)
    # -------------------------------------------------------------------------
    def defense_phase(self, board: player_board.PlayerBoard, safe_moves: list[Action]) -> Action:
        return random.choice(safe_moves)

    # -------------------------------------------------------------------------
    # Attack Phase with MULTI-STEP MINIMAX + ALPHA-BETA
    # -------------------------------------------------------------------------
    def attack_phase(self, board: player_board.PlayerBoard, safe_moves: list[Action]) -> Action:
        """
        We'll do a minimax with 'attack_minimax_depth' plies. We are Minimizer, enemy is Maximizer.
        We'll pick the move that leads to the minimal final # of safe positions
        for the enemy, after both snakes move up to 'attack_minimax_depth' times.

        Now with alpha-beta pruning to skip losing branches early.
        """
        if not safe_moves:
            return Action.FF

        best_move = None
        best_value = math.inf  # Minimizer => we want to find min
        alpha = -math.inf
        beta  = math.inf

        for mv in safe_moves:
            nxt_board, ok = board.forecast_turn([mv], check_validity=True)
            if not ok:
                continue
            val = self.attack_minimax_ab(nxt_board,
                                         depth=self.attack_minimax_depth - 1,
                                         is_me=False,
                                         alpha=alpha,
                                         beta=beta)
            if val < best_value:
                best_value = val
                best_move = mv
            # For a minimizer at the root, we also can update 'beta'
            # but typically alpha is for maximizing layer, beta for minimizing
            # We'll treat it as we do in deeper layers. 
            if best_value < beta:
                beta = best_value
            if beta <= alpha:
                break  # alpha-beta cutoff

        if best_move is None:
            return random.choice(safe_moves)
        return best_move

    def attack_minimax_ab(self, board: player_board.PlayerBoard, depth: int,
                          is_me: bool, alpha: float, beta: float) -> float:
        """
        Multi-step minimax with alpha-beta pruning:

        - If depth <= 0 or game over => evaluate # of enemy safe moves
        - If is_me => Minimizer layer => pick min
        - If not is_me => Maximizer layer => pick max
        """
        if depth <= 0 or board.is_game_over():
            return float(self.count_enemy_safe_moves(board))

        if is_me:
            # Minimizer
            best_val = math.inf
            my_moves = []
            possible = board.get_possible_directions()
            for m in possible:
                if board.is_valid_move(m):
                    nxt_board, ok = board.forecast_turn([m], check_validity=True)
                    if ok and not self.is_losing_state(nxt_board):
                        if self.can_survive(nxt_board, 1):
                            my_moves.append(m)

            if not my_moves:
                return float(self.count_enemy_safe_moves(board))

            for mv in my_moves:
                nxt_board, ok = board.forecast_turn([mv], check_validity=True)
                if not ok:
                    continue
                val = self.attack_minimax_ab(nxt_board, depth - 1,
                                             is_me=False,
                                             alpha=alpha, beta=beta)
                if val < best_val:
                    best_val = val
                # Update beta since we're Minimizer
                if best_val < beta:
                    beta = best_val
                if beta <= alpha:
                    break  # alpha-beta cutoff
            return best_val
        else:
            # Maximizer (enemy)
            best_val = -math.inf
            bcopy = board.get_copy()
            bcopy.reverse_perspective()
            possible = bcopy.get_possible_directions(enemy=False)
            enemy_moves = []
            for em in possible:
                if bcopy.is_valid_move(em, enemy=False):
                    pass_board, ok = board.forecast_turn([em], check_validity=True)
                    if ok and not self.is_losing_state(pass_board):
                        if self.can_survive(pass_board, 1):
                            enemy_moves.append(em)

            if not enemy_moves:
                return float(self.count_enemy_safe_moves(board))

            for em in enemy_moves:
                pass_board, ok = board.forecast_turn([em], check_validity=True)
                if not ok:
                    continue
                val = self.attack_minimax_ab(pass_board, depth - 1,
                                             is_me=True,
                                             alpha=alpha, beta=beta)
                if val > best_val:
                    best_val = val
                # Update alpha since we're Maximizer
                if best_val > alpha:
                    alpha = best_val
                if beta <= alpha:
                    break  # alpha-beta cutoff
            return best_val

    # -------------------------------------------------------------------------
    # Evaluate: how many safe moves the enemy has next turn
    # -------------------------------------------------------------------------
    def count_enemy_safe_moves(self, board: player_board.PlayerBoard) -> int:
        enemy_moves = board.get_possible_directions(enemy=True)
        safe_count = 0
        for em in enemy_moves:
            if board.is_valid_move(em, enemy=True):
                eboard, eok = board.forecast_turn([em], check_validity=True)
                if eok and not self.is_losing_state(eboard):
                    if self.can_survive(eboard, 1):
                        safe_count += 1
        return safe_count

    # -------------------------------------------------------------------------
    # can_survive etc. from old code
    # -------------------------------------------------------------------------
    def can_survive(self, board: player_board.PlayerBoard, depth: int) -> bool:
        key = (board_to_key(board), depth)
        if key in self.survival_cache:
            return self.survival_cache[key]

        if depth <= 0:
            self.survival_cache[key] = True
            return True
        if self.is_losing_state(board):
            self.survival_cache[key] = False
            return False

        if board.is_my_turn():
            my_moves = []
            possible = board.get_possible_directions()
            for m in possible:
                if board.is_valid_move(m):
                    my_moves.append(m)

            if not my_moves:
                self.survival_cache[key] = False
                return False

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
            bcopy = board.get_copy()
            bcopy.reverse_perspective()
            enemy_moves = []
            for m in bcopy.get_possible_directions(enemy=False):
                if bcopy.is_valid_move(m, enemy=False):
                    enemy_moves.append(m)

            if not enemy_moves:
                self.survival_cache[key] = True
                return True

            if len(enemy_moves) > self.max_enemy_moves:
                enemy_moves = random.sample(enemy_moves, self.max_enemy_moves)

            for em in enemy_moves:
                opp_board, ok = board.forecast_turn([em], check_validity=True)
                if not ok:
                    continue
                if self.is_losing_state(opp_board):
                    self.survival_cache[key] = False
                    return False
                if not self.can_survive(opp_board, depth - 1):
                    self.survival_cache[key] = False
                    return False

            self.survival_cache[key] = True
            return True

    def is_losing_state(self, board: player_board.PlayerBoard) -> bool:
        if board.is_game_over():
            return True
        my_len = board.get_length()
        min_len = board.get_min_player_size()
        return (my_len < min_len)