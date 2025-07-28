from game import player_board
from game.enums import Action
from collections.abc import Callable
import random
import math

def compute_apple_distance(board: player_board.PlayerBoard):
    """Compute the Manhattan distance from our snake's head to the closest apple."""
    head = board.get_head_location()  # (x, y) for player's snake
    apples = board.get_current_apples()  # numpy array of (x, y) apple coordinates
    if apples.size > 0:
        distances = [abs(head[0] - a[0]) + abs(head[1] - a[1]) for a in apples]
        return min(distances)
    else:
        return 0

class MCTSNode:
    def __init__(self, board: player_board.PlayerBoard, parent=None, move=None):
        self.board = board            # A deep copy of the board state.
        self.parent = parent          # Parent node in the tree.
        self.move = move              # The move that led to this state.
        self.children = {}            # Map moves to child nodes.
        self.visits = 0               # Number of times node was visited.
        self.wins = 0.0               # Cumulative reward from simulations.
        # Initialize untried moves: valid moves from this state.
        self.untried_moves = [m for m in board.get_possible_directions() if board.is_valid_move(m)]
        # Optionally add TRAP if valid.
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

class PlayerControllerLect:
    def __init__(self, time_left: Callable):
        # MCTS parameters
        self.iterations = 200       # Total number of MCTS iterations.
        self.simulation_depth = 7   # Base simulation (rollout) depth.
        # Finite state for our bot. Possible states: "FARM", "ATTACK", "EXPLORE"
        self.state = "EXPLORE"

    def bid(self, board: player_board.PlayerBoard, time_left: Callable):
        return 0

    def play(self, board: player_board.PlayerBoard, time_left: Callable):
        # Sense: Update our state based on board conditions.
        self.update_state(board)
        # Think: Use MCTS (with biases based on state) to choose a move.
        best_move = self.mcts_search(board, self.iterations)
        if best_move is None:
            return Action.FF
        # Act: Return the chosen move.
        return [best_move]

    def update_state(self, board: player_board.PlayerBoard):
        """
        Updates the bot's internal state based on board conditions.
        Example:
         - If our snake is nearly at minimum length, we enter FARM mode.
         - If enemy's head is within a short distance and we are longer, we enter ATTACK mode.
         - Otherwise, we default to EXPLORE.
        """
        my_length = board.get_length()
        min_size = board.get_min_player_size()
        enemy_head = board.get_head_location(enemy=True)
        my_head = board.get_head_location()

        # Default to exploring.
        self.state = "EXPLORE"
        # If we're very short, prioritize farming.
        if my_length <= min_size + 2:
            self.state = "FARM"
        # If enemy is very close and we're in a favorable position, attack.
        elif enemy_head is not None:
            distance = abs(my_head[0] - enemy_head[0]) + abs(my_head[1] - enemy_head[1])
            if distance <= 3 and my_length > board.get_length(enemy=True):
                self.state = "ATTACK"

    def mcts_search(self, root_board: player_board.PlayerBoard, iterations: int):
        root_node = MCTSNode(root_board.get_copy())
        for _ in range(iterations):
            node = root_node
            board_copy = root_board.get_copy()

            # 1. Selection: descend until a node with untried moves or terminal state is reached.
            while node.is_fully_expanded() and node.children:
                node = node.best_child()
                board_copy, valid = board_copy.forecast_turn([node.move])
                if not valid:
                    break

            # 2. Expansion: if node has untried moves, expand one.
            if node.untried_moves:
                move = random.choice(node.untried_moves)
                node.untried_moves.remove(move)
                new_board, valid = board_copy.forecast_turn([move])
                if not valid:
                    continue
                child_node = MCTSNode(new_board, parent=node, move=move)
                node.children[move] = child_node
                node = child_node
                board_copy = new_board

            # 3. Simulation: run a biased random playout.
            reward = self.simulate(board_copy)

            # 4. Backpropagation: update nodes on the path.
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
        Choose a move from possible_moves with weights based on the current state.
        - In FARM mode: favor moves that reduce distance to apples.
        - In ATTACK mode: favor moves that bring our head closer to the enemy or allow trap placement.
        - In EXPLORE mode: slightly favor moves that lead to more open space.
        """
        weights = []
        enemy_head = board.get_head_location(enemy=True)
        for move in possible_moves:
            weight = 1.0
            new_board, valid = board.forecast_turn([move])
            if not valid:
                weight = 0
            else:
                head = new_board.get_head_location()
                # Penalize moves near walls.
                dim_x, dim_y = new_board.get_dim_x(), new_board.get_dim_y()
                dist_to_wall = min(head[0], dim_x - head[0] - 1, head[1], dim_y - head[1] - 1)
                if dist_to_wall < 2:
                    weight *= 0.5

                if self.state == "FARM":
                    # In FARM mode, favor moves that decrease apple distance.
                    apple_distance = compute_apple_distance(new_board)
                    weight *= 1.0 / (1.0 + apple_distance)
                elif self.state == "ATTACK":
                    # In ATTACK mode, favor moves that reduce distance to enemy head.
                    if enemy_head is not None:
                        dist_enemy = abs(head[0] - enemy_head[0]) + abs(head[1] - enemy_head[1])
                        # Increase weight if enemy is close.
                        if dist_enemy < 4:
                            weight *= 1.5
                    # Also favor trap moves.
                    if move == Action.TRAP:
                        weight *= 1.3
                else:
                    # In EXPLORE mode, slightly favor moves that yield more open space.
                    open_spaces = self.count_open_spaces(new_board, head)
                    weight *= 1.0 + 0.05 * open_spaces
            weights.append(weight)
        total = sum(weights)
        if total == 0:
            return random.choice(possible_moves)
        norm_weights = [w / total for w in weights]
        return random.choices(possible_moves, weights=norm_weights, k=1)[0]

    def count_open_spaces(self, board: player_board.PlayerBoard, head):
        """
        Count free cells around the snake's head within a 5x5 grid.
        Uses board.is_valid_cell() to check if a cell is within bounds and not occupied.
        """
        open_count = 0
        dim_x, dim_y = board.get_dim_x(), board.get_dim_y()
        for dx in range(-2, 3):
            for dy in range(-2, 3):
                x, y = head[0] + dx, head[1] + dy
                if board.cell_in_bounds((x, y)) and not board.is_occupied(x, y):
                    open_count += 1
        return open_count

    def simulate(self, board: player_board.PlayerBoard):
        """
        Run a biased random playout from the current board state.
        The simulation depth is dynamically adjusted based on the turn count.
        """
        turn_count = board.get_turn_count() if hasattr(board, "get_turn_count") else 0
        sim_depth = self.simulation_depth if turn_count <= 1000 else max(3, self.simulation_depth - 2)
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
        Evaluate the board state using a heuristic that considers:
          - Difference in snake lengths.
          - Difference in apples eaten.
          - Distance to the closest apple (weighted dynamically).
          - Penalty for proximity to walls.
          - Attack bonuses: extra reward if our head is close to the enemy, and if the enemy has fewer moves.
        """
        my_length = board.get_length()
        enemy_length = board.get_length(enemy=True)
        length_diff = my_length - enemy_length

        my_apples = board.get_apples_eaten()
        enemy_apples = board.get_apples_eaten(enemy=True)
        apple_diff = my_apples - enemy_apples

        my_head = board.get_head_location()
        apples = board.get_current_apples()
        if apples.size > 0:
            distances = [abs(my_head[0] - a[0]) + abs(my_head[1] - a[1]) for a in apples]
            closest_apple_distance = min(distances)
        else:
            closest_apple_distance = 0

        opp_advantage = max(0, board.get_length(enemy=True) - (my_length - 2))
        apple_distance_weight = 0.2 + 0.05 * opp_advantage

        x, y = my_head
        dim_x, dim_y = board.get_dim_x(), board.get_dim_y()
        dist_to_wall = min(x, dim_x - x - 1, y, dim_y - y - 1)
        wall_penalty = (2 - dist_to_wall) * 1.0 if dist_to_wall < 2 else 0

        # Attack bonuses.
        my_moves = len(board.get_possible_directions())
        enemy_moves = len(board.get_possible_directions(enemy=True))
        move_diff_bonus = (my_moves - enemy_moves) * 0.2

        enemy_head = board.get_head_location(enemy=True)
        if enemy_head is not None:
            head_distance = abs(my_head[0] - enemy_head[0]) + abs(my_head[1] - enemy_head[1])
            proximity_bonus = (5 - head_distance) * 0.1 if head_distance < 5 else 0
        else:
            proximity_bonus = 0

        score = (length_diff * 1.0) + (apple_diff * 0.5) \
                - (closest_apple_distance * apple_distance_weight) \
                - wall_penalty + move_diff_bonus + proximity_bonus
        return score

    def is_terminal(self, board: player_board.PlayerBoard):
        """A terminal state is reached if either snake's length falls below the minimum allowed size."""
        min_size = board.get_min_player_size()
        return board.get_length() < min_size or board.get_length(enemy=True) < min_size
