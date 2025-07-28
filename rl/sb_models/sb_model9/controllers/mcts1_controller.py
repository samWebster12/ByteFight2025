from game import player_board
from game.enums import Action
from collections.abc import Callable
import random
import math

class MCTSNode:
    def __init__(self, board: player_board.PlayerBoard, parent=None, move=None):
        self.board = board                # The board state at this node.
        self.parent = parent              # Parent node in the tree.
        self.move = move                  # The move that led to this state.
        self.children = {}                # Dictionary mapping moves to child nodes.
        self.visits = 0                   # Number of times node was visited.
        self.wins = 0.0                   # Cumulative reward from simulations.
        # Initialize untried moves using available directions that are valid.
        self.untried_moves = [m for m in board.get_possible_directions() if board.is_valid_move(m)]

    def is_fully_expanded(self):
        return len(self.untried_moves) == 0

    def best_child(self, C=1.4):
        # Select the child with the highest UCT (Upper Confidence Bound for Trees) value.
        best_score = -math.inf
        best_child = None
        for child in self.children.values():
            # Average reward from this child.
            avg_reward = child.wins / child.visits if child.visits > 0 else 0
            # UCT value calculation.
            uct = avg_reward + C * math.sqrt(math.log(self.visits) / child.visits)
            if uct > best_score:
                best_score = uct
                best_child = child
        return best_child

    def update(self, reward):
        self.visits += 1
        self.wins += reward

class PlayerControllerMCTS1:
    def __init__(self, time_left: Callable):
        # You can adjust the number of MCTS iterations and simulation depth as needed.
        self.iterations = 100      # Number of MCTS iterations.
        self.simulation_depth = 4  # Depth for random playout simulation.

    def bid(self, board: player_board.PlayerBoard, time_left: Callable):
        return 0

    def play(self, board: player_board.PlayerBoard, time_left: Callable):
        # Run MCTS from the current board state.
        best_move = self.mcts_search(board, self.iterations)
        if best_move is None:
            return Action.FF
        return [best_move]

    def mcts_search(self, root_board: player_board.PlayerBoard, iterations: int):
        root_node = MCTSNode(root_board.get_copy())
        for _ in range(iterations):
            node = root_node
            board_copy = root_board.get_copy()

            # 1. Selection: Descend tree until a node with untried moves or terminal state is reached.
            while node.is_fully_expanded() and node.children:
                node = node.best_child()
                board_copy, valid = board_copy.forecast_turn([node.move])
                if not valid:
                    break

            # 2. Expansion: If node has untried moves, try one.
            if node.untried_moves:
                move = random.choice(node.untried_moves)
                node.untried_moves.remove(move)
                # Forecast move on a copy of the board.
                new_board, valid = board_copy.forecast_turn([move])
                if not valid:
                    continue  # Skip invalid moves.
                child_node = MCTSNode(new_board, parent=node, move=move)
                node.children[move] = child_node
                node = child_node
                board_copy = new_board

            # 3. Simulation: Run a random playout from the current state.
            reward = self.simulate(board_copy)

            # 4. Backpropagation: Update the nodes on the path with the simulation result.
            while node is not None:
                node.update(reward)
                node = node.parent

        # After iterations, choose the move from the root with the most visits.
        best_move = None
        best_visits = -1
        for move, child in root_node.children.items():
            if child.visits > best_visits:
                best_visits = child.visits
                best_move = move
        return best_move

    def simulate(self, board: player_board.PlayerBoard):
        # Run a random simulation for a fixed number of moves.
        simulation_board = board.get_copy()
        for _ in range(self.simulation_depth):
            # Check terminal conditions.
            if self.is_terminal(simulation_board):
                break
            possible_moves = [m for m in simulation_board.get_possible_directions() if simulation_board.is_valid_move(m)]
            if not possible_moves:
                break
            move = random.choice(possible_moves)
            simulation_board, valid = simulation_board.forecast_turn([move])
            if not valid:
                break
        # Use the evaluation function as the reward.
        return self.evaluate_board(simulation_board)

    def evaluate_board(self, board: player_board.PlayerBoard):
        # Improved evaluation heuristic.
        my_length = board.get_length()
        enemy_length = board.get_length(enemy=True)
        length_diff = my_length - enemy_length

        my_apples = board.get_apples_eaten()
        enemy_apples = board.get_apples_eaten(enemy=True)
        apple_diff = my_apples - enemy_apples

        my_head = board.get_head_location()
        apples = board.get_current_apples()  # Expects a numpy array of apple coordinates.
        if apples.size > 0:
            distances = [abs(my_head[0] - a[0]) + abs(my_head[1] - a[1]) for a in apples]
            closest_apple_distance = min(distances)
        else:
            closest_apple_distance = 0

        # Increase apple importance if snake is near minimum size.
        min_size = board.get_min_player_size()
        if my_length <= min_size + 2:
            apple_distance_weight = 0.3
        else:
            apple_distance_weight = 0.1

        # Penalize proximity to walls.
        x, y = my_head
        dim_x = board.get_dim_x()
        dim_y = board.get_dim_y()
        dist_to_wall = min(x, dim_x - x - 1, y, dim_y - y - 1)
        wall_penalty = 0
        if dist_to_wall < 2:
            wall_penalty = (2 - dist_to_wall) * 1.0

        score = (length_diff * 1.0) + (apple_diff * 0.5) - (closest_apple_distance * apple_distance_weight) - wall_penalty
        return score

    def is_terminal(self, board: player_board.PlayerBoard):
        # Terminal if either snake's length is below the minimum.
        min_size = board.get_min_player_size()
        return board.get_length() < min_size or board.get_length(enemy=True) < min_size
