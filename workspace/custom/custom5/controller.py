import math
import random
from collections.abc import Callable
from game import player_board
from game.enums import Action

def board_to_key(board: player_board.PlayerBoard) -> tuple:
    """
    Same as before.
    """
    my_head = board.get_head_location(enemy=False)
    enemy_head = board.get_head_location(enemy=True)
    my_len = board.get_length(enemy=False)
    enemy_len = board.get_length(enemy=True)
    turn_count = board.get_turn_count()
    am_i_turn = 1 if board.is_my_turn() else 0
    return (int(my_head[0]), int(my_head[1]),
            int(enemy_head[0]), int(enemy_head[1]),
            my_len, enemy_len, turn_count, am_i_turn)


class MCTSNode:
    """
    A basic MCTS node to store:
      - the board state
      - is_me or not
      - parent, children
      - visits, total_reward
      - untried actions
    """
    def __init__(self, board: player_board.PlayerBoard, is_me: bool, parent=None):
        self.board = board
        self.is_me = is_me  # whether it's "my" turn or the enemy's
        self.parent = parent
        self.children = []
        self.visits = 0
        self.total_reward = 0.0
        self.untried_moves = self.get_actions()

    def get_actions(self) -> list[Action]:
        """
        Return the possible single-step moves for the current player.
        If is_me==True, gather my moves; else gather enemy moves.
        """
        if self.is_me:
            possible = self.board.get_possible_directions(enemy=False)
            valid = []
            for m in possible:
                if self.board.is_valid_move(m, enemy=False):
                    valid.append(m)
            return valid
        else:
            # enemy
            bcopy = self.board.get_copy()
            bcopy.reverse_perspective()
            possible = bcopy.get_possible_directions(enemy=False)
            valid = []
            for em in possible:
                if bcopy.is_valid_move(em, enemy=False):
                    valid.append(em)
            return valid

    def ucb_child(self, c=1.4):
        """
        Return the child with best UCB score.
        UCB = (child.total_reward / child.visits) + c * sqrt(ln(self.visits)/child.visits)
        """
        best = None
        best_value = -99999999
        for child in self.children:
            if child.visits == 0:
                return child  # immediately pick unvisited child
            avg = child.total_reward / child.visits
            ucb = avg + c * math.sqrt(math.log(self.visits) / child.visits)
            if ucb > best_value:
                best_value = ucb
                best = child
        return best


class PlayerController:
    def __init__(self, time_left: Callable):
        self.time_left = time_left
        self.lookahead_depth = 15  # used in survival checks
        self.survival_cache = {}
        self.max_enemy_moves = 4
        # We'll do N MCTS iterations for Attack
        self.mcts_iterations = 200

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
            if self.can_survive(nxt_board, self.lookahead_depth):
                safe_moves.append(mv)

        if not safe_moves:
            return [random.choice(valid_moves)]

        if phase == "FARMING":
            chosen = self.farming_phase(board, safe_moves)
        elif phase == "ATTACK":
            # run MCTS
            chosen = self.attack_phase_mcts(board, safe_moves)
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
        enemy_len = board.get_length(enemy=True)
        min_size = board.get_min_player_size()

        if (my_apples < enemy_apples) or (my_len <= min_size + 2):
            return "FARMING"

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
    # Defense
    # -------------------------------------------------------------------------
    def defense_phase(self, board: player_board.PlayerBoard, safe_moves: list[Action]) -> Action:
        return random.choice(safe_moves)

    # -------------------------------------------------------------------------
    # Attack with MCTS
    # -------------------------------------------------------------------------
    def attack_phase_mcts(self, board: player_board.PlayerBoard, safe_moves: list[Action]) -> Action:
        """
        Instead of Minimizer/Maximizer, we do MCTS:
          - Root node = (board, is_me=True).
          - We do self.mcts_iterations iteration of (selection -> expansion -> simulation -> backprop)
          - Then pick child with highest # of visits or best avg reward
        We'll define a reward = - count_enemy_safe_moves in the simulation final state,
        so we want to maximize negative => minimize enemy safe moves.
        """
        # 1) Build root
        root = MCTSNode(board, is_me=True, parent=None)
        # We only consider your safe_moves from the root. So to speed up,
        # let's manually set the untried moves = safe_moves
        root.untried_moves = safe_moves

        for _ in range(self.mcts_iterations):
            # a) SELECT
            node = self.mcts_select(root)

            # b) EXPAND
            node = self.mcts_expand(node)

            # c) SIMULATE
            reward = self.mcts_simulate(node.board, sim_depth=5)

            # d) BACKPROP
            self.mcts_backprop(node, reward)

        # pick best child => e.g. highest visits
        # each child corresponds to a move from the root
        best_child = None
        best_visits = -1
        for child in root.children:
            if child.visits > best_visits:
                best_visits = child.visits
                best_child = child

        if not best_child:
            # fallback
            return random.choice(safe_moves)

        # the move that leads to that child
        # we know each child was created by a single action from the root
        # so we can store that action
        return best_child.action_from_parent

    # --------------- MCTS Helpers ---------------
    def mcts_select(self, node: MCTSNode) -> MCTSNode:
        """
        Descend the tree with UCB until we hit a node with untried moves or no children.
        """
        while True:
            if node.untried_moves:
                # we can expand right away
                return node
            # if no untried => choose child with best UCB
            if not node.children:
                return node
            node = node.ucb_child()

    def mcts_expand(self, node: MCTSNode) -> MCTSNode:
        """
        Take one untried move from node.untried_moves, create a child, apply that move on a new board, return child
        """
        if not node.untried_moves:
            return node  # can't expand

        move = random.choice(node.untried_moves)
        node.untried_moves.remove(move)

        # forecast
        nxt_board, ok = node.board.forecast_turn([move], check_validity=True)
        if not ok:
            # if invalid or leads to losing, just create a terminal node anyway
            child = MCTSNode(node.board, is_me=not node.is_me, parent=node)
            child.action_from_parent = move
            node.children.append(child)
            return child

        child = MCTSNode(nxt_board, is_me=not node.is_me, parent=node)
        child.action_from_parent = move
        node.children.append(child)
        return child

    def mcts_simulate(self, board: player_board.PlayerBoard, sim_depth: int) -> float:
        """
        A quick random playout up to sim_depth moves or until game over.
        We'll define reward = -count_enemy_safe_moves at the end.
        Negative => if enemy has many safe moves, reward is smaller.
        We'll do random moves for both sides.
        """
        sim_board = board.get_copy()
        for _ in range(sim_depth):
            if sim_board.is_game_over():
                break
            # if it's my turn
            if sim_board.is_my_turn():
                # gather possible directions
                my_moves = sim_board.get_possible_directions(enemy=False)
                valids = []
                for m in my_moves:
                    if sim_board.is_valid_move(m, enemy=False):
                        valids.append(m)
                if not valids:
                    break
                mv = random.choice(valids)
                sim_board, ok = sim_board.forecast_turn([mv], check_validity=True)
                if not ok:
                    break
            else:
                # enemy turn
                bcopy = sim_board.get_copy()
                bcopy.reverse_perspective()
                enemy_moves = bcopy.get_possible_directions(enemy=False)
                valids = []
                for em in enemy_moves:
                    if bcopy.is_valid_move(em, enemy=False):
                        valids.append(em)
                if not valids:
                    break
                em = random.choice(valids)
                sim_board, ok = sim_board.forecast_turn([em], check_validity=True)
                if not ok:
                    break

        # final measure
        return -float(self.count_enemy_safe_moves(sim_board))

    def mcts_backprop(self, node: MCTSNode, reward: float):
        """
        Climb back up the tree, increment visits, add reward
        """
        while node is not None:
            node.visits += 1
            node.total_reward += reward
            node = node.parent

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
