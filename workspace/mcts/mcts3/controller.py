import math
import random
import numpy as np
from numba import njit
from collections.abc import Callable
from game import player_board
from game.enums import Action

######################################################################
# 1) Numba-JIT'ed Numeric Helper
######################################################################
@njit
def combine_score(
    length_diff: float,
    apple_diff: float,
    dist_apple: float,
    dist_wall: float,
    my_len: float,
    min_size: float
) -> float:
    """
    Minimal numeric function that Numba can compile to machine code.
    This combines your partial scoring logic.
    """
    # Weighted more if we're near minimum size
    if my_len <= min_size + 2:
        w_apple = 0.3
    else:
        w_apple = 0.1

    wall_penalty = 0.0
    if dist_wall < 2:
        wall_penalty = (2 - dist_wall) * 1.0

    # Combine the partial scores
    score = (length_diff * 1.0) \
            + (apple_diff * 0.5) \
            - (dist_apple * w_apple) \
            - wall_penalty

    return score

######################################################################
# 2) Board Hash (unchanged but shown for completeness)
######################################################################
def board_to_key(board: player_board.PlayerBoard) -> int:
    """
    Return a hashable key representing the board state.
    """
    board_str = board.get_board_string()  # returns tuple with strings & lengths
    turn_count = board.get_turn_count()
    am_i_turn = 1 if board.is_my_turn() else 0
    full_str = (str(board_str) + f"_turn_{turn_count}_mine_{am_i_turn}")
    return hash(full_str)

######################################################################
# 3) MCTSNode
######################################################################
class MCTSNode:
    __slots__ = (
        "board", "parent", "move", "children",
        "visits", "wins", "untried_moves"
    )
    def __init__(self, board: player_board.PlayerBoard, parent=None, move=None):
        self.board = board
        self.parent = parent
        self.move = move
        self.children = {}
        self.visits = 0
        self.wins = 0.0

        # For simplicity, only consider single-step moves:
        possible_moves = []
        for m in board.get_possible_directions(enemy=False):
            if board.is_valid_move(m, enemy=False):
                possible_moves.append(m)
        self.untried_moves = possible_moves

    def is_fully_expanded(self):
        return len(self.untried_moves) == 0

    def best_child(self, C=1.4):
        best_score = float("-inf")
        best_node = None
        for child in self.children.values():
            if child.visits == 0:
                continue
            avg_reward = child.wins / child.visits
            # UCT formula
            uct = avg_reward + C * math.sqrt(math.log(self.visits) / child.visits)
            if uct > best_score:
                best_score = uct
                best_node = child
        return best_node

    def update(self, reward):
        self.visits += 1
        self.wins += reward

######################################################################
# 4) PlayerController with MCTS + partial Numba usage
######################################################################
class PlayerController:
    def __init__(self, time_left: Callable):
        self.iterations = 150
        self.sim_depth = 10

        # Example usage for storing partial info about visited states (not used below)
        self.transposition = {}

    ##################################################################
    # Bidding
    ##################################################################
    def bid(self, board: player_board.PlayerBoard, time_left: Callable):
        return 0

    ##################################################################
    # Main Interface: play()
    ##################################################################
    def play(self, board: player_board.PlayerBoard, time_left: Callable):
        best_move = self.mcts_search(board, self.iterations)
        if best_move is None:
            return [Action.FF]
        return [best_move]

    ##################################################################
    # MCTS
    ##################################################################
    def mcts_search(self, root_board: player_board.PlayerBoard, iters: int):
        root_node = MCTSNode(root_board.get_copy())

        for _ in range(iters):
            node = root_node
            temp_board = root_board.get_copy()

            # 1. SELECTION
            while node.is_fully_expanded() and node.children:
                node = node.best_child()
                temp_board, valid = temp_board.forecast_turn([node.move], check_validity=True)
                if not valid:
                    break

            # 2. EXPANSION
            if node.untried_moves:
                move = random.choice(node.untried_moves)
                node.untried_moves.remove(move)

                new_board, valid = temp_board.forecast_turn([move], check_validity=True)
                if not valid:
                    continue

                # Double-forecast "suicide" check
                if self.is_suicidal_future(new_board):
                    continue

                child_node = MCTSNode(new_board, parent=node, move=move)
                node.children[move] = child_node
                node = child_node
                temp_board = new_board

            # 3. SIMULATION (rollout)
            reward = self.simulate_two_player(temp_board)

            # 4. BACKPROP
            while node is not None:
                node.update(reward)
                node = node.parent

        # Choose child with highest visits
        best_move, best_visits = None, -1
        for m, c in root_node.children.items():
            if c.visits > best_visits:
                best_visits = c.visits
                best_move = m
        return best_move

    ##################################################################
    # Double-Forecast “Suicide” Check
    ##################################################################
    def is_suicidal_future(self, board: player_board.PlayerBoard) -> bool:
        bcopy = board.get_copy()
        if bcopy.is_game_over():
            return True

        if bcopy.is_my_turn():
            my_moves = self.get_valid_moves(bcopy, enemy=False)
            return (len(my_moves) == 0)

        if bcopy.is_enemy_turn():
            bcopy.reverse_perspective()
            enemy_moves = self.get_valid_moves(bcopy, enemy=False)
            # flip perspective back if needed
            bcopy.reverse_perspective()

            if not enemy_moves:
                # enemy is stuck => good for us => not suicidal
                return False

            all_future_dead = True
            for em in enemy_moves:
                opp_board, valid = board.forecast_turn([em], check_validity=True)
                if not valid:
                    continue
                if opp_board.is_game_over():
                    all_future_dead = False
                    break
                if opp_board.is_my_turn():
                    my_moves = self.get_valid_moves(opp_board, enemy=False)
                    if my_moves:
                        all_future_dead = False
                        break
            return all_future_dead

        # fallback
        return False

    ##################################################################
    # Rollout: two-player simulation
    ##################################################################
    def simulate_two_player(self, board: player_board.PlayerBoard) -> float:
        sim_board = board.get_copy()
        for _ in range(self.sim_depth):
            if self.is_terminal(sim_board):
                break

            # Our move
            if sim_board.is_my_turn():
                my_move = self.select_rollout_move(sim_board, enemy=False)
                if my_move is None:
                    break
                new_board, valid = sim_board.forecast_turn([my_move], check_validity=True)
                if not valid:
                    break
                sim_board = new_board

            if self.is_terminal(sim_board):
                break

            # Opponent move
            if sim_board.is_enemy_turn():
                sim_board.reverse_perspective()
                opp_move = self.select_rollout_move(sim_board, enemy=False)
                sim_board.reverse_perspective()

                if opp_move is None:
                    break
                new_board, valid = sim_board.forecast_turn([opp_move], check_validity=True)
                if not valid:
                    break
                sim_board = new_board

        return self.evaluate_board(sim_board)

    ##################################################################
    # Rollout policy: pick a move that tries to avoid instant death
    ##################################################################
    def select_rollout_move(self, board: player_board.PlayerBoard, enemy: bool) -> Action | None:
        valid_moves = self.get_valid_moves(board, enemy=enemy)
        if not valid_moves:
            return None

        safe_moves = []
        losing_moves = []
        for mv in valid_moves:
            new_b, ok = board.forecast_turn([mv], check_validity=True)
            if not ok:
                losing_moves.append(mv)
                continue
            if self.is_terminal(new_b):
                # If terminal and our length < min => losing
                if new_b.get_length() < new_b.get_min_player_size():
                    losing_moves.append(mv)
                else:
                    safe_moves.append(mv)
            else:
                if self.is_suicidal_future(new_b):
                    losing_moves.append(mv)
                else:
                    safe_moves.append(mv)

        if safe_moves:
            return random.choice(safe_moves)
        return random.choice(losing_moves)

    ##################################################################
    # Basic Helpers
    ##################################################################
    def get_valid_moves(self, board: player_board.PlayerBoard, enemy=False) -> list[Action]:
        moves = []
        for m in board.get_possible_directions(enemy=enemy):
            if board.is_valid_move(m, enemy=enemy):
                moves.append(m)
        return moves

    def is_terminal(self, board: player_board.PlayerBoard) -> bool:
        if board.is_game_over():
            return True
        if board.get_length() < board.get_min_player_size():
            return True
        if board.get_length(enemy=True) < board.get_min_player_size():
            return True
        return False

    ##################################################################
    # Evaluate board with partial NumPy/Numba usage
    ##################################################################
    def evaluate_board(self, board: player_board.PlayerBoard) -> float:
        my_len = board.get_length()
        enemy_len = board.get_length(enemy=True)
        length_diff = my_len - enemy_len

        my_apples = board.get_apples_eaten()
        enemy_apples = board.get_apples_eaten(enemy=True)
        apple_diff = my_apples - enemy_apples

        # Head & current apples
        head = board.get_head_location(enemy=False)  # e.g., np.array([x, y])
        apples = board.get_current_apples()          # shape: (N,2)

        # Vectorized Manhattan distance to apples
        if apples.size > 0:
            # dist_arr = |apples[:,0] - head[0]| + |apples[:,1] - head[1]|
            dist_arr = np.abs(apples[:, 0] - head[0]) + np.abs(apples[:, 1] - head[1])
            dist_apple = dist_arr.min()
        else:
            dist_apple = 0.0

        x, y = head
        dim_x = board.get_dim_x()
        dim_y = board.get_dim_y()

        # distance to the closest wall
        dist_wall = min(x, dim_x - 1 - x, y, dim_y - 1 - y)

        # Now we combine using the Numba-jitted function
        score = combine_score(
            length_diff,
            apple_diff,
            dist_apple,
            dist_wall,
            float(my_len),
            float(board.get_min_player_size())
        )
        return score



