from game import player_board
from game.enums import Action
from collections.abc import Callable
import random
import math

def compute_apple_distance(board: player_board.PlayerBoard):
    """Compute the Manhattan distance from our snake's head to the closest apple."""
    head = board.get_head_location()  # (x, y) for player's snake
    apples = board.get_current_apples()  # numpy array of apple coordinates in (x, y)
    if apples.size > 0:
        distances = [abs(head[0] - a[0]) + abs(head[1] - a[1]) for a in apples]
        return min(distances)
    else:
        return 0

class MCTSNode:
    def __init__(self, board: player_board.PlayerBoard, parent=None, move=None):
        self.board = board                # A deep copy of the board state.
        self.parent = parent              # Parent node in the tree.
        self.move = move                  # The move that led to this state.
        self.children = {}                # Dictionary mapping moves to child nodes.
        self.visits = 0                   # Number of times node was visited.
        self.wins = 0.0                   # Cumulative reward from simulations.
        # Build the list of untried moves: valid directions...
        self.untried_moves = [m for m in board.get_possible_directions() if board.is_valid_move(m)]
        # ...and include TRAP if the board allows it.
        if board.is_valid_trap():
            self.untried_moves.append(Action.TRAP)

    def is_fully_expanded(self):
        return len(self.untried_moves) == 0

    def best_child(self, C=1.4):
        best_score = -math.inf
        best_child = None
        for child in self.children.values():
            avg_reward = child.wins / child.visits if child.visits > 0 else 0
            uct = avg_reward + C * math.sqrt(math.log(self.visits) / child.visits)
            if uct > best_score:
                best_score = uct
                best_child = child
        return best_child

    def update(self, reward):
        self.visits += 1
        self.wins += reward

class PlayerController:
    def __init__(self, time_left: Callable):
        # Adjust iterations and simulation depth based on testing.
        self.iterations = 200       # Number of MCTS iterations.
        self.simulation_depth = 7   # Base depth for random playout simulation.

    def bid(self, board: player_board.PlayerBoard, time_left: Callable):
        return 0

    def play(self, board: player_board.PlayerBoard, time_left: Callable):
        best_move = self.mcts_search(board, self.iterations)
        if best_move is None:
            return Action.FF
        return [best_move]

    def mcts_search(self, root_board: player_board.PlayerBoard, iterations: int):
        root_node = MCTSNode(root_board.get_copy())
        for _ in range(iterations):
            node = root_node
            board_copy = root_board.get_copy()

            # 1. Selection: Descend until a node with untried moves or terminal state is reached.
            while node.is_fully_expanded() and node.children:
                node = node.best_child()
                board_copy, valid = board_copy.forecast_turn([node.move])
                if not valid:
                    break

            # 2. Expansion: Expand one untried move.
            if node.untried_moves:
                move = random.choice(node.untried_moves)
                node.untried_moves.remove(move)
                new_board, valid = board_copy.forecast_turn([move])
                if not valid:
                    continue  # Skip this move.
                child_node = MCTSNode(new_board, parent=node, move=move)
                node.children[move] = child_node
                node = child_node
                board_copy = new_board

            # 3. Simulation: Run a biased random playout.
            reward = self.simulate(board_copy)

            # 4. Backpropagation: Update all nodes along the path.
            while node is not None:
                node.update(reward)
                node = node.parent

        # Choose the move from the root with the most visits.
        best_move = None
        best_visits = -1
        for move, child in root_node.children.items():
            if child.visits > best_visits:
                best_visits = child.visits
                best_move = move
        return best_move

    def biased_choice(self, board: player_board.PlayerBoard, possible_moves):
        """
        Chooses a move from possible_moves, weighting moves that are safer, improve apple access,
        or are aggressive (attack moves) higher.
        """
        weights = []
        # Get enemy head location for attack bias.
        enemy_head = board.get_head_location(enemy=True)
        for move in possible_moves:
            weight = 1.0
            new_board, valid = board.forecast_turn([move])
            if not valid:
                weight = 0
            else:
                head = new_board.get_head_location()
                # Penalize moves that put us too close to walls.
                dim_x, dim_y = new_board.get_dim_x(), new_board.get_dim_y()
                dist_to_wall = min(head[0], dim_x - head[0] - 1, head[1], dim_y - head[1] - 1)
                if dist_to_wall < 2:
                    weight *= 0.5
                # When our snake is near the minimum size, prefer moves that reduce apple distance.
                my_length = new_board.get_length()
                if my_length <= board.get_min_player_size() + 2:
                    apple_distance = compute_apple_distance(new_board)
                    weight *= 1.0 / (1.0 + apple_distance)
                # Attack bias: if the move is TRAP and enemy head is nearby, boost its weight.
                if move == Action.TRAP:
                    if enemy_head is not None:
                        # Compute Manhattan distance between enemy head and our head.
                        dist_enemy = abs(head[0] - enemy_head[0]) + abs(head[1] - enemy_head[1])
                        if dist_enemy < 4:
                            weight *= 1.5  # Increase weight if enemy is close.
                # Also, bias toward moves that leave the enemy with fewer options.
                my_options = len(new_board.get_possible_directions())
                enemy_options = len(new_board.get_possible_directions(enemy=True))
                if enemy_options < my_options:
                    weight *= 1.0 + 0.1 * (my_options - enemy_options)
            weights.append(weight)
        total = sum(weights)
        if total == 0:
            return random.choice(possible_moves)
        norm_weights = [w / total for w in weights]
        return random.choices(possible_moves, weights=norm_weights, k=1)[0]

    def simulate(self, board: player_board.PlayerBoard):
        """
        Performs a random (but biased) playout from the given board state.
        The simulation depth is dynamically adjusted based on turn count.
        """
        turn_count = board.get_turn_count() if hasattr(board, "get_turn_count") else 0
        sim_depth = self.simulation_depth
        if turn_count > 1000:
            sim_depth = max(3, self.simulation_depth - 2)
        simulation_board = board.get_copy()
        for _ in range(sim_depth):
            if self.is_terminal(simulation_board):
                break
            possible_moves = [m for m in simulation_board.get_possible_directions() if simulation_board.is_valid_move(m)]
            if not possible_moves:
                break
            move = self.biased_choice(simulation_board, possible_moves)
            simulation_board, valid = simulation_board.forecast_turn([move])
            if not valid:
                break
        return self.evaluate_board(simulation_board)

    def evaluate_board(self, board: player_board.PlayerBoard):
        """
        Evaluation function enhanced with attack considerations.
          - Snake length difference.
          - Apple count difference.
          - Distance to closest apple (weighted more if the enemy is ahead).
          - Wall proximity penalty.
          - Attack bonus: bonus if enemy has fewer moves and if our head is close to enemy's head.
        """
        my_length = board.get_length()
        enemy_length = board.get_length(enemy=True)
        length_diff = my_length - enemy_length

        my_apples = board.get_apples_eaten()
        enemy_apples = board.get_apples_eaten(enemy=True)
        apple_diff = my_apples - enemy_apples

        my_head = board.get_head_location()
        apples = board.get_current_apples()  # numpy array of (x,y) apple coordinates.
        if apples.size > 0:
            distances = [abs(my_head[0] - a[0]) + abs(my_head[1] - a[1]) for a in apples]
            closest_apple_distance = min(distances)
        else:
            closest_apple_distance = 0

        # Dynamically adjust apple distance weight based on opponent advantage.
        opp_advantage = max(0, board.get_length(enemy=True) - (my_length - 2))
        apple_distance_weight = 0.2 + 0.05 * opp_advantage

        # Wall penalty.
        x, y = my_head
        dim_x, dim_y = board.get_dim_x(), board.get_dim_y()
        dist_to_wall = min(x, dim_x - x - 1, y, dim_y - y - 1)
        wall_penalty = 0
        if dist_to_wall < 2:
            wall_penalty = (2 - dist_to_wall) * 1.0

        # Attack bonus 1: Favor states where enemy has fewer options.
        my_options = len(board.get_possible_directions())
        enemy_options = len(board.get_possible_directions(enemy=True))
        option_diff = my_options - enemy_options
        attack_option_bonus = option_diff * 0.2  # Tune this weight as needed.

        # Attack bonus 2: Favor states where our head is closer to enemy's head.
        enemy_head = board.get_head_location(enemy=True)
        if enemy_head is not None:
            head_distance = abs(my_head[0] - enemy_head[0]) + abs(my_head[1] - enemy_head[1])
            # Bonus is higher when distance is small (but avoid division by zero).
            attack_proximity_bonus = (5 - head_distance) * 0.1 if head_distance < 5 else 0
        else:
            attack_proximity_bonus = 0

        score = (length_diff * 1.0) + (apple_diff * 0.5) \
                - (closest_apple_distance * apple_distance_weight) \
                - wall_penalty \
                + attack_option_bonus + attack_proximity_bonus
        return score

    def is_terminal(self, board: player_board.PlayerBoard):
        """Terminal state if either snake's length falls below the minimum allowed size."""
        min_size = board.get_min_player_size()
        return board.get_length() < min_size or board.get_length(enemy=True) < min_size
