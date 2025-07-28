import math
import random
from collections.abc import Callable

import numpy as np

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

import math
import random

class MCTSNode:
    def __init__(self, board: player_board.PlayerBoard, move_seq: list[Action] = None, parent: "MCTSNode" = None):
        self.board = board
        self.move_seq = move_seq  # The move sequence that led to this node (a list of Actions)
        self.parent = parent
        self.children: list[MCTSNode] = []
        self.untried_moves: list[list[Action]] = []  # Candidate multi-move sequences from this state
        self.visits = 0
        self.total_reward = 0.0

    def ucb1(self, exploration: float = 1.414) -> float:
        # If a node hasn't been visited, return infinity to ensure it's tried
        if self.visits == 0:
            return float('inf')
        return self.total_reward / self.visits + exploration * math.sqrt(math.log(self.parent.visits) / self.visits)

def select_child(node: MCTSNode) -> MCTSNode:
    """Return the child with the highest UCB1 value."""
    return max(node.children, key=lambda child: child.ucb1())

def generate_candidate_moves(board: player_board.PlayerBoard) -> list[list[Action]]:
    """
    Combine standard and extended sequences from generate_sequences.
    This returns a list of candidate multi-move sequences.
    """
    standard, extended = generate_sequences(board)
    return standard + extended

def rollout(board: player_board.PlayerBoard, depth: int, weights: dict) -> float:
    """
    Perform a random rollout (simulation) from the current board state.
    At each step, choose a random valid move until the simulation depth is reached
    or the game ends. Then use the evaluation function to compute a reward.
    """
    current_board = board
    for _ in range(depth):
        if current_board.is_game_over():
            break
        moves = current_board.get_possible_directions()
        valid_moves = [m for m in moves if current_board.is_valid_move(m)]
        if not valid_moves:
            break
        chosen_move = random.choice(valid_moves)
        next_board, ok = current_board.forecast_turn([chosen_move], check_validity=True)
        if not ok:
            break
        current_board = next_board
    # In the rollout, we use an empty sequence (since we’re not incurring extra sacrifice costs)
    return evaluate_state(current_board, sequence=[], weights=weights)

def mcts_search(root_board: player_board.PlayerBoard, weights: dict, iterations: int = 1000, simulation_depth: int = 10) -> list[Action]:
    """
    Run MCTS from the current board state using the provided evaluation weights.
    Returns the best move sequence (i.e. the sequence of Actions for the turn)
    based on the simulations.
    """
    # Create the root node
    root = MCTSNode(root_board)
    # Initialize candidate moves for the root node.
    root.untried_moves = generate_candidate_moves(root_board)
    
    for _ in range(iterations):
        node = root
        # Make a copy of the board for simulation purposes.
        board_state = root_board.get_copy()
        
        # SELECTION: Traverse the tree using UCB until reaching a node with untried moves.
        while not node.untried_moves and node.children:
            node = select_child(node)
            # Forecast the move sequence that led to this node.
            board_state, ok = board_state.forecast_turn(node.move_seq, check_validity=True)
            if not ok:
                break
        
        # EXPANSION: If we have candidate moves at this node, expand one.
        if node.untried_moves:
            move_seq = node.untried_moves.pop()
            new_board, ok = board_state.forecast_turn(move_seq, check_validity=True)
            if not ok:
                continue  # Skip if the move sequence is invalid.
            child = MCTSNode(new_board, move_seq=move_seq, parent=node)
            child.untried_moves = generate_candidate_moves(new_board)
            node.children.append(child)
            node = child
            board_state = new_board
        
        # SIMULATION: Rollout from the expanded node.
        reward = rollout(board_state, simulation_depth, weights)
        
        # BACKPROPAGATION: Propagate the reward up the tree.
        while node is not None:
            node.visits += 1
            node.total_reward += reward
            node = node.parent
    
    # After iterations, pick the child of the root with the highest visit count.
    if not root.children:
        # Fallback: choose a random candidate move if no children were expanded.
        candidate_moves = generate_candidate_moves(root_board)
        return random.choice(candidate_moves) if candidate_moves else []
    
    best_child = max(root.children, key=lambda child: child.visits)
    return best_child.move_seq


# All 8 cardinal/diagonal directions.
ALL_DIRECTIONS = [
    Action.NORTH,
    Action.NORTHEAST,
    Action.EAST,
    Action.SOUTHEAST,
    Action.SOUTH,
    Action.SOUTHWEST,
    Action.WEST,
    Action.NORTHWEST
]

# Mapping from directional Action to its angle (in degrees).
# We assume 0° is East, 90° is North, 180° is West, 270° is South.
angle_map = {
    Action.NORTH: 90,
    Action.NORTHEAST: 45,
    Action.EAST: 0,
    Action.SOUTHEAST: 315,  # equivalent to -45°
    Action.SOUTH: 270,
    Action.SOUTHWEST: 225,
    Action.WEST: 180,
    Action.NORTHWEST: 135,
}

# Mapping from directional Action to displacement (Δx, Δy).
# (For example, Action.NORTH moves (0, -1) if y decreases upward.)
MOVE_DISPLACEMENTS = {
    Action.NORTH: (0, -1),
    Action.NORTHEAST: (1, -1),
    Action.EAST: (1, 0),
    Action.SOUTHEAST: (1, 1),
    Action.SOUTH: (0, 1),
    Action.SOUTHWEST: (-1, 1),
    Action.WEST: (-1, 0),
    Action.NORTHWEST: (-1, -1)
}

# The trap action (assumed to be defined in your Action enum).
TRAP_ACTION = Action.TRAP

def allowed_directions_from_heading(current_heading: Action) -> list[Action]:
    """
    Given the current heading, return a list of allowed directional moves (non-trap)
    that are within 90° of the current heading.
    """
    if current_heading == None:
        return ALL_DIRECTIONS
    
    allowed = []
    current_angle = angle_map[current_heading]
    for action in ALL_DIRECTIONS:
        candidate_angle = angle_map[action]
        diff = abs(current_angle - candidate_angle)
        diff = min(diff, 360 - diff)
        if diff <= 90:
            allowed.append(action)
    return allowed


def generate_sequences(board: player_board.PlayerBoard) -> tuple[list[list[Action]], list[list[Action]]]:
    """
    Generate move sequences for one turn based on current board state.

    Two types of sequences are generated:
      1. Standard sequences: up to 3 sub-moves, subject to the snake’s available length.
      2. Extended enemy sequences: up to min(8, allowed-by-length) sub-moves,
         but only include sequences in which every directional move does not
         increase the Manhattan distance to the enemy.

    A sequence is only generated if the sacrifice cost (number of extra moves)
    does not reduce the snake's length below 2.

    This version limits the use of trap actions to at most 2 per turn.
    
    Returns:
        A tuple (standard_sequences, extended_sequences) where each is a list of sequences,
        and each sequence is a list of Action values.
    """
    current_length = board.get_length()
    if current_length < 2:
        return [], []  # Not enough length to move.

    # A sequence of k moves costs (k - 1) length. So we require:
    #   current_length - (k - 1) >= 2   =>   k <= current_length - 1.
    max_moves_allowed = current_length  # k - 1 <= current_length - 2  --> k <= current_length - 1
    standard_max = min(3, max_moves_allowed)
    extended_max = min(5, max_moves_allowed)

    start_pos = board.get_head_location()
    enemy_pos = board.get_head_location(enemy=True)

    standard_sequences = []
    extended_sequences = []

    def manhattan(pos1, pos2):
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

    def recurse(seq: list[Action],
                move_count: int,
                current_pos: tuple[int, int],
                current_heading: Action | None,
                trap_count: int,
                max_moves: int,
                is_extended: bool):
        """
        Recursively build move sequences.

        Args:
            seq: List of Actions taken so far.
            move_count: Number of moves in the current sequence.
            current_pos: Current head position (x, y) after these moves.
            current_heading: The current heading (None for the first move).
            trap_count: Number of traps used so far in this sequence.
            max_moves: Maximum number of sub-moves to generate in this branch.
            is_extended: If True, only allow moves that do not increase distance to enemy.
        """
        if seq:
            if is_extended:
                extended_sequences.append(seq.copy())
            else:
                standard_sequences.append(seq.copy())

        if move_count >= max_moves:
            return

        # Determine allowed directional actions based on current heading.
        if current_heading is None:
            allowed_directional = ALL_DIRECTIONS
        else:
            allowed_directional = allowed_directions_from_heading(current_heading)

        # Only allow the trap action if we haven't used 2 traps yet.
        allowed_actions = allowed_directional + ([TRAP_ACTION] if trap_count < 2 else [])

        for action in allowed_actions:
            if action == TRAP_ACTION:
                new_pos = current_pos
                new_heading = current_heading  # Heading remains unchanged.
                new_trap_count = trap_count + 1
            else:
                disp = MOVE_DISPLACEMENTS.get(action)
                if disp is None:
                    continue
                new_pos = (current_pos[0] + disp[0], current_pos[1] + disp[1])
                new_heading = action
                new_trap_count = trap_count

            # For extended sequences, ensure directional moves don't increase Manhattan distance.
            if is_extended and action != TRAP_ACTION:
                if manhattan(new_pos, enemy_pos) > manhattan(current_pos, enemy_pos):
                    continue

            # Check sacrifice cost: a sequence of (move_count+1) moves costs move_count.
            if current_length - move_count < 2:
                continue

            seq.append(action)
            recurse(seq, move_count + 1, new_pos, new_heading, new_trap_count, max_moves, is_extended)
            seq.pop()

    # Generate standard sequences.
    recurse(seq=[], move_count=0, current_pos=start_pos, current_heading=None,
            trap_count=0, max_moves=standard_max, is_extended=False)

    # Generate extended enemy-directed sequences.
    recurse(seq=[], move_count=0, current_pos=start_pos, current_heading=None,
            trap_count=15, max_moves=extended_max, is_extended=True)

    return standard_sequences, extended_sequences

def get_nearest_apple_distance(board: player_board.PlayerBoard) -> int:
    """
    Returns the Manhattan distance from our snake's head to the nearest apple.
    If no apple exists, return a high value (e.g. 1000).
    """
    apples = board.get_current_apples()
    if apples.size == 0:
        return 1000  # No apples; arbitrarily high distance.
    print(apples)
    head = board.get_head_location()
    # Compute Manhattan distance for each apple.
    distances = [abs(ax - head[0]) + abs(ay - head[1]) for ax, ay in apples]
    return min(distances)


def count_safe_moves(board: player_board.PlayerBoard) -> int:
    """
    Returns the number of moves from the current board state that are "safe".
    This uses a shallow lookahead: for each possible move, we forecast the board
    and ensure that it is valid and not immediately losing.
    """
    safe_count = 0
    moves = board.get_possible_directions()
    for move in moves:
        if board.is_valid_move(move):
            # Forecast the move to see if it leads to a non-losing state.
            next_board, ok = board.forecast_turn([move], check_validity=True)
            if ok and not board.is_game_over() and next_board.get_length() >= board.get_min_player_size():
                safe_count += 1
    return safe_count

def count_enemy_safe_moves(board: player_board.PlayerBoard) -> int:
    """
    Count how many moves the enemy can make that are "safe".
    We perform a shallow one-ply lookahead for each possible enemy move:
      - Only count moves that are valid.
      - Only count moves where forecasting the move does not result in a losing state.
    """
    safe_count = 0
    enemy_moves = board.get_possible_directions(enemy=True)
    
    for move in enemy_moves:
        if board.is_valid_move(move, enemy=True):
            # Forecast the enemy's move.
            enemy_board, ok = board.forecast_turn([move], check_validity=True, reverse=True)
            if not ok:
                continue
            # Check if enemy is not in an immediate losing state.
            if enemy_board.is_game_over() or enemy_board.get_length(enemy=True) < board.get_min_player_size():
                continue
            safe_count += 1
            
    return safe_count



def evaluate_state(board: player_board.PlayerBoard, sequence: list[Action], weights: dict) -> float:
    if board.is_game_over() or board.get_length() < board.get_min_player_size():
        return -1e6

    score = 0.0

    # 1. Enemy kill potential (using your existing count_enemy_safe_moves).
    enemy_safe_moves = count_enemy_safe_moves(board)
    max_enemy_moves = 10  # example value
    enemy_factor = max_enemy_moves - enemy_safe_moves
    score += weights.get('enemy', 1.0) * enemy_factor

    # 2. Apple proximity.
    apple_distance = get_nearest_apple_distance(board)
    max_distance = 100  # example bound
    apple_factor = max_distance - apple_distance
    score += weights.get('apple', 1.0) * (1.0 / (get_nearest_apple_distance(board) + 1))
    
    # 3. Sacrifice penalty.
    sacrifice_cost = max(0, len(sequence) - 1)
    score -= weights.get('sacrifice', 1.0) * sacrifice_cost

    # 4. Optional bonuses.
    snake_length = board.get_length()
    score += weights.get('length', 0.0) * snake_length

    safe_moves = count_safe_moves(board)
    score += weights.get('mobility', 0.0) * safe_moves

    if safe_moves > 0:
        score += weights.get('survival', 100.0)

    return score




class PlayerController:
    def __init__(self, time_left: Callable):
        self.time_left = time_left
        self.lookahead_depth = 15  # We can do up to 10 steps
        # Transposition table for can_survive: (board_key, depth) -> bool
        self.survival_cache = {}
        # We'll also prune the enemy's moves if they have too many
        self.max_enemy_moves = 4

    def bid(self, board: player_board.PlayerBoard, time_left: Callable):
        return 0

    def play(self, board: player_board.PlayerBoard, time_left: Callable):
        stand_seqs, ext_seqs = generate_sequences(board)
        samp = stand_seqs[0]

        weights = {
            'enemy': 0.2,
            'apple': 50,
            'sacrifice': 2,
            'length': 0.4,
            'mobility': 2,
            'survival': 2
        }
        seqs = stand_seqs + ext_seqs

        best = mcts_search(board, weights, 500, 10)
        print("Best Sequence: ", best)
        
        return best

    # -------------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------------
    def is_losing_state(self, board: player_board.PlayerBoard) -> bool:
        if board.is_game_over():
            return True
        my_len = board.get_length()
        min_len = board.get_min_player_size()
        return (my_len < min_len)