import math
import random
from collections.abc import Callable
from game import player_board
from game.enums import Action

def board_to_key(board: player_board.PlayerBoard) -> tuple:
    """
    A simple hashable key for transposition in can_survive.
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
        # We'll store our current phase so we can add a "bonus" for continuity
        self.current_phase = "FARMING"  # or default to something
        self.attack_threshold = 0.0     # The "score" above which we Attack, else we Farm

        # Additional variables from your existing code:
        self.lookahead_depth = 15
        self.survival_cache = {}
        self.max_enemy_moves = 4
        self.attack_lookahead = 3

    def bid(self, board: player_board.PlayerBoard, time_left: Callable):
        return 0

    def play(self, board: player_board.PlayerBoard, time_left: Callable):
        # 1) Evaluate a “score” that suggests how strongly we want to Attack
        score_attack = self.compute_attack_score(board)

        # 2) Add a "phase continuity bonus" if we’re already in Attack
        if self.current_phase == "ATTACK":
            # e.g. +3 bonus means we keep attacking unless the Farm reason is substantially bigger
            score_attack += 3.0

        # 3) If the final score_attack > self.attack_threshold => Attack, else => Farm
        if score_attack > self.attack_threshold:
            phase = "ATTACK"
        else:
            phase = "FARMING"

        # Update current_phase
        self.current_phase = phase

        # Now we gather possible turns, filter them with survival, then pick final turn
        turns = self.gather_possible_turns(board)
        valid_turns = []
        for t in turns:
            nxt_board, ok = board.forecast_turn(t, check_validity=True)
            if not ok:
                continue
            if self.is_losing_state(nxt_board):
                continue
            if self.can_survive(nxt_board, self.lookahead_depth):
                valid_turns.append(t)

        if not valid_turns:
            # fallback
            single_moves = board.get_possible_directions()
            single_valid = [m for m in single_moves if board.is_valid_move(m)]
            if single_valid:
                return [random.choice(single_valid)]
            return [Action.FF]

        # Phase-based final pick
        if self.current_phase == "ATTACK":
            chosen_turn = self.attack_phase(board, valid_turns)
        else:
            chosen_turn = self.farming_phase(board, valid_turns)

        return chosen_turn

    # -------------------------------------------------------------------------
    # Weighted Attack Score
    # -------------------------------------------------------------------------
    def compute_attack_score(self, board: player_board.PlayerBoard) -> float:
        """
        Example function that returns a float "attack_score".
        Larger means we are more inclined to Attack.

        We'll do a simple combination:
          - length_diff
          - apple_diff
          - BFS area difference or distance to enemy
          - Possibly a time factor
        """
        my_len = board.get_length()
        enemy_len = board.get_length(enemy=True)
        length_diff = my_len - enemy_len

        my_apples = board.get_apples_eaten()
        enemy_apples = board.get_apples_eaten(enemy=True)
        apple_diff = my_apples - enemy_apples

        # BFS-based area difference if you want
        my_area = self.compute_reachable_area(board, enemy=False)
        enemy_area = self.compute_reachable_area(board, enemy=True)
        area_diff = my_area - enemy_area

        # distance to enemy
        dist = self.distance_to_enemy(board)

        # Weighted sum
        W_len = 2.0
        W_app = 1.5
        W_area = 1.0
        W_dist = -0.3   # negative weight for distance => prefer smaller distance => bigger score

        # final
        score = (length_diff * W_len) + (apple_diff * W_app) + (area_diff * W_area) + (dist * W_dist)
        return score

    def distance_to_enemy(self, board: player_board.PlayerBoard) -> float:
        my_head = board.get_head_location(enemy=False)
        enemy_head = board.get_head_location(enemy=True)
        return abs(my_head[0] - enemy_head[0]) + abs(my_head[1] - enemy_head[1])

    def compute_reachable_area(self, board: player_board.PlayerBoard, enemy: bool) -> int:
        """
        BFS from the chosen snake's head to see how many squares are reachable,
        ignoring walls.
        """
        dim_x = board.get_dim_x()
        dim_y = board.get_dim_y()
        head = board.get_head_location(enemy=enemy)
        if head[0] < 0 or head[0] >= dim_x or head[1] < 0 or head[1] >= dim_y:
            return 0
        visited = set()
        queue = [ (head[0], head[1]) ]
        visited.add( (head[0], head[1]) )

        wall_mask = board.get_wall_mask()  # shape is (dim_y, dim_x)

        while queue:
            x, y = queue.pop()
            for nx, ny in self.neighbors_4(x,y):
                if 0 <= nx < dim_x and 0 <= ny < dim_y:
                    if (nx, ny) not in visited:
                        # skip if it's a wall
                        if wall_mask[ny, nx] == 1:
                            continue
                        visited.add((nx, ny))
                        queue.append((nx, ny))
        return len(visited)

    def neighbors_4(self, x, y):
        return [(x+1,y), (x-1,y), (x,y+1), (x,y-1)]

    # -------------------------------------------------------------------------
    # The rest is your code from before
    # -------------------------------------------------------------------------
    def gather_possible_turns(self, board: player_board.PlayerBoard) -> list[list[Action]]:
        turns = []
        moves = board.get_possible_directions()
        for mv in moves:
            if board.is_valid_move(mv):
                turns.append([mv])
        if self.can_place_trap(board):
            for mv in moves:
                if board.is_valid_move(mv):
                    turns.append([mv, Action.TRAP])
        return turns

    def can_place_trap(self, board: player_board.PlayerBoard) -> bool:
        if not board.is_valid_trap():
            return False
        traps_left = board.get_traps_until_limit(enemy=False)
        return (traps_left > 0)

    def farming_phase(self, board: player_board.PlayerBoard, valid_turns: list[list[Action]]) -> list[Action]:
        apples = board.get_current_apples()
        if apples.size == 0:
            return random.choice(valid_turns)

        head = board.get_head_location()
        nearest_apple = None
        nearest_dist = 999999
        for ax, ay in apples:
            dist = abs(ax - head[0]) + abs(ay - head[1])
            if dist < nearest_dist:
                nearest_dist = dist
                nearest_apple = (ax, ay)

        best_turn = None
        best_dist2 = nearest_dist
        for t in valid_turns:
            nxt_board, ok = board.forecast_turn(t, check_validity=True)
            if not ok:
                continue
            new_head = nxt_board.get_head_location()
            dist_new = abs(nearest_apple[0] - new_head[0]) + abs(nearest_apple[1] - new_head[1])
            if dist_new < best_dist2:
                best_dist2 = dist_new
                best_turn = t

        if best_turn is None:
            return random.choice(valid_turns)
        return best_turn

    def attack_phase(self, board: player_board.PlayerBoard, valid_turns: list[list[Action]]) -> list[Action]:
        """
        Among the 'valid_turns' (which may include single steps, multi-step dashes, 
        and/or trap actions), pick the one that yields:
        1) minimal enemy reachable area
        2) minimal distance to the enemy (as a tiebreak)
        We do this by maximizing (-enemy_area, -dist).
        """
        # Where is the enemy's head right now (for distance calculation)?
        enemy_head = board.get_head_location(enemy=True)

        best_turn = None
        best_score = None  # We'll store a tuple ( -enemy_area, -distance )

        for turn_actions in valid_turns:
            # Forecast applying all actions in 'turn_actions'
            nxt_board, ok = board.forecast_turn(turn_actions, check_validity=True)
            if not ok or self.is_losing_state(nxt_board):
                continue

            # (A) Compute how many squares the enemy can reach from this new state
            enemy_area = self.compute_reachable_area(nxt_board, enemy=True)

            # (B) Compute distance from our new head to the enemy's current head
            #     If your code wants the *enemy's head in the new state*, 
            #     you'd do 'nxt_board.get_head_location(enemy=True)' instead. 
            #     But typically you want our new head vs. the current known enemy position,
            #     or you can forecast the enemy's next move if you prefer.
            my_new_head = nxt_board.get_head_location(enemy=False)
            dist_to_enemy = abs(my_new_head[0] - enemy_head[0]) + abs(my_new_head[1] - enemy_head[1])

            # (C) We want to minimize enemy_area, then minimize dist => 
            #     "maximize" the tuple ( -enemy_area, -dist_to_enemy )
            move_score = (-enemy_area, -dist_to_enemy)

            if (best_score is None) or (move_score > best_score):
                best_score = move_score
                best_turn = turn_actions

        # Fallback if none chosen
        if not best_turn:
            return random.choice(valid_turns)

        return best_turn


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
            for m in board.get_possible_directions():
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
