from game import player_board
from game.enums import Action
from collections.abc import Callable
import random
import numpy as np

class SmartPlayerController:
    """
    A smarter opponent for training your ByteFight agent.
    
    This controller has different intelligence levels:
    - Level 1: Random valid moves
    - Level 2: Prefers moves toward apples
    - Level 3: Seeks apples and avoids traps
    - Level 4: Advanced strategy with trap placement
    """
    
    def __init__(self, time_left: Callable, intelligence_level: int = 1):
        """
        Initialize the smart controller.
        
        Args:
            time_left: A callable that returns the time left for this player
            intelligence_level: 1-4, with 4 being the smartest
        """
        self.intelligence_level = min(max(intelligence_level, 1), 4)
        print(f"Initialized SmartPlayerController with intelligence level {self.intelligence_level}")
    
    def bid(self, board: player_board.PlayerBoard, time_left: Callable):
        """
        Make a bid at the start of the game.
        
        Args:
            board: The game board
            time_left: A callable that returns the time left for this player
            
        Returns:
            int: The bid amount
        """
        initial_length = board.get_length()
        min_size = board.get_min_player_size()
        
        if self.intelligence_level == 1:
            # Level 1: Random small bid
            return random.randint(0, 1)
        
        elif self.intelligence_level == 2:
            # Level 2: Moderate bid
            max_bid = min(initial_length - min_size, initial_length // 4)
            return random.randint(0, max_bid)
            
        elif self.intelligence_level == 3:
            # Level 3: Calculated bid based on length
            max_bid = min(initial_length - min_size, initial_length // 3)
            return random.randint(max_bid//2, max_bid)
            
        elif self.intelligence_level == 4:
            # Level 4: Strategic bid based on length
            max_bid = min(initial_length - min_size, initial_length // 2)
            return random.randint(max_bid//2, max_bid)
    
    def play(self, board: player_board.PlayerBoard, time_left: Callable):
        """
        Make a move based on the intelligence level.
        
        Args:
            board: The game board
            time_left: A callable that returns the time left for this player
            
        Returns:
            Action or list[Action]: The action(s) to take
        """
        # Get valid moves for any intelligence level
        valid_moves = []
        for move in range(8):  # 8 directional moves
            action = Action(move)
            if board.is_valid_move(action):
                valid_moves.append(action)
        
        # Check if trap placement is valid
        can_trap = board.is_valid_trap()
        if can_trap:
            valid_moves.append(Action.TRAP)
        
        # If no valid moves, forfeit
        if not valid_moves:
            return Action.FF
        
        # Level 1: Completely random
        if self.intelligence_level == 1:
            return [random.choice(valid_moves)]
        
        # For higher levels, need apple positions
        apple_positions = board.get_apple_positions() if hasattr(board, 'get_apple_positions') else []
        head_pos = board.get_head_location()
        
        # Level 2: Basic apple-seeking
        if self.intelligence_level == 2:
            if apple_positions and random.random() < 0.7:  # 70% chance to seek apples
                # Get closest apple
                closest_apple = min(apple_positions, key=lambda pos: 
                                  abs(pos[0] - head_pos[0]) + abs(pos[1] - head_pos[1]))
                
                # Move toward the apple
                return [self._move_toward_target(closest_apple, head_pos, valid_moves)]
            else:
                return [random.choice(valid_moves)]
        
        # Level 3: Smarter with trap avoidance
        if self.intelligence_level == 3:
            # Get enemy trap positions to avoid
            enemy_traps = []
            for y in range(board.get_dim_y()):
                for x in range(board.get_dim_x()):
                    if board.has_enemy_trap(x, y):
                        enemy_traps.append((x, y))
            
            # Score each possible move
            move_scores = {}
            for move in valid_moves:
                if move == Action.TRAP:
                    # Only place traps if we're long enough
                    if board.get_unqueued_length() > 8 and random.random() < 0.3:
                        move_scores[move] = 10
                    else:
                        move_scores[move] = -5
                    continue
                
                # Get new position after move
                new_pos = self._get_position_after_move(head_pos, move)
                
                # Start with a neutral score
                score = 0
                
                # Heavy penalty for moving into a trap
                if new_pos in enemy_traps:
                    score -= 100
                
                # Reward for moving toward an apple
                if apple_positions:
                    closest_apple = min(apple_positions, key=lambda pos: 
                                      abs(pos[0] - new_pos[0]) + abs(pos[1] - new_pos[1]))
                    distance = abs(closest_apple[0] - new_pos[0]) + abs(closest_apple[1] - new_pos[1])
                    score += 20 / (1 + distance)  # Higher score for closer apples
                
                move_scores[move] = score
            
            # Choose move with highest score
            best_score = max(move_scores.values())
            best_moves = [move for move, score in move_scores.items() if score == best_score]
            return [random.choice(best_moves)]
        
        # Level 4: Advanced strategy with multiple moves and trap placement
        if self.intelligence_level == 4:
            # Similar to level 3 but with more complex decision-making
            # Get enemy trap positions and snake positions to avoid
            enemy_traps = []
            for y in range(board.get_dim_y()):
                for x in range(board.get_dim_x()):
                    if board.has_enemy_trap(x, y):
                        enemy_traps.append((x, y))
            
            # Get enemy snake body and head positions (to avoid)
            enemy_snake_mask = board.get_snake_mask(my_snake=False, enemy_snake=True)
            enemy_positions = []
            for y in range(board.get_dim_y()):
                for x in range(board.get_dim_x()):
                    if enemy_snake_mask[y, x] in [5, 6]:  # ENEMY_HEAD or ENEMY_BODY
                        enemy_positions.append((x, y))
            
            # Decision to place trap strategically
            if can_trap:
                # Place trap if we have enough length and are in a good position
                enemy_head = board.get_head_location(enemy=True)
                if enemy_head is not None:
                    my_tail = board.get_tail_location(enemy=False)
                    
                    # Check proximity to enemy head (good place for trap)
                    if (abs(my_tail[0] - enemy_head[0]) + abs(my_tail[1] - enemy_head[1])) < 5:
                        if board.get_unqueued_length() > 5 and random.random() < 0.7:
                            return [Action.TRAP]
            
            # Score moves based on multiple factors
            move_scores = {}
            for move in valid_moves:
                if move == Action.TRAP:
                    continue  # Already handled trap decision above
                
                # Get new position after move
                new_pos = self._get_position_after_move(head_pos, move)
                
                # Start with a neutral score
                score = 0
                
                # Heavy penalty for moving into a trap or snake
                if new_pos in enemy_traps:
                    score -= 100
                
                if new_pos in enemy_positions:
                    score -= 50
                
                # Reward for moving toward an apple
                if apple_positions:
                    closest_apple = min(apple_positions, key=lambda pos: 
                                      abs(pos[0] - new_pos[0]) + abs(pos[1] - new_pos[1]))
                    distance = abs(closest_apple[0] - new_pos[0]) + abs(closest_apple[1] - new_pos[1])
                    score += 30 / (1 + distance)  # Higher score for closer apples
                
                # Penalty for moving too close to walls if not necessary
                wall_mask = board.get_wall_mask()
                wall_neighbors = 0
                for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                    nx, ny = new_pos[0] + dx, new_pos[1] + dy
                    if 0 <= ny < board.get_dim_y() and 0 <= nx < board.get_dim_x():
                        if wall_mask[ny, nx] == 1:  # WALL
                            wall_neighbors += 1
                
                # Penalize moves that put us next to walls (limited movement options)
                score -= wall_neighbors * 2
                
                move_scores[move] = score
            
            # If no valid directional moves scored
            if not move_scores:
                # Try to place a trap if possible, otherwise random
                if can_trap:
                    return [Action.TRAP]
                else:
                    return [random.choice(valid_moves)]
            
            # Choose move with highest score
            best_score = max(move_scores.values())
            best_moves = [move for move, score in move_scores.items() if score == best_score]
            
            # Consider chaining multiple moves
            chosen_move = random.choice(best_moves)
            
            # Sometimes do multiple moves in one turn if length permits
            # Only do this if we're heading toward an apple and have sufficient length
            my_length = board.get_length(enemy=False)
            if (my_length > 10 and apple_positions and random.random() < 0.3):
                # Check if we can do another move after this one
                test_board = board.forecast_move(chosen_move)[0]
                next_valid_moves = []
                for next_move in range(8):
                    if test_board.is_valid_move(Action(next_move)):
                        next_valid_moves.append(Action(next_move))
                
                if next_valid_moves:
                    # Choose a second move toward the apple
                    next_head_pos = self._get_position_after_move(head_pos, chosen_move)
                    closest_apple = min(apple_positions, key=lambda pos: 
                                      abs(pos[0] - next_head_pos[0]) + abs(pos[1] - next_head_pos[1]))
                    second_move = self._move_toward_target(closest_apple, next_head_pos, next_valid_moves)
                    
                    # Make sure the second move doesn't cause us to lose too much length
                    if my_length > 15:  # Only do multi-move with plenty of length
                        return [chosen_move, second_move]
            
            # Default to single move
            return [chosen_move]
    
    def _move_toward_target(self, target_pos, current_pos, valid_moves):
        """
        Choose a move that gets closer to the target position.
        
        Args:
            target_pos: (x, y) position of the target
            current_pos: (x, y) current position
            valid_moves: List of valid Action enums
            
        Returns:
            Action: Best move toward the target
        """
        # Calculate direction to target
        dx = target_pos[0] - current_pos[0]
        dy = target_pos[1] - current_pos[1]
        
        # Filter valid_moves to only include Action enums (not TRAP)
        valid_directions = [move for move in valid_moves if move != Action.TRAP]
        if not valid_directions:
            return random.choice(valid_moves)
        
        # Try primary directions first
        if dx > 0 and Action.EAST in valid_directions:
            return Action.EAST
        elif dx < 0 and Action.WEST in valid_directions:
            return Action.WEST
        elif dy > 0 and Action.SOUTH in valid_directions:
            return Action.SOUTH
        elif dy < 0 and Action.NORTH in valid_directions:
            return Action.NORTH
        
        # Try diagonal directions
        if dx > 0 and dy < 0 and Action.NORTHEAST in valid_directions:
            return Action.NORTHEAST
        elif dx < 0 and dy < 0 and Action.NORTHWEST in valid_directions:
            return Action.NORTHWEST
        elif dx > 0 and dy > 0 and Action.SOUTHEAST in valid_directions:
            return Action.SOUTHEAST
        elif dx < 0 and dy > 0 and Action.SOUTHWEST in valid_directions:
            return Action.SOUTHWEST
        
        # Fall back to any valid directional move
        return random.choice(valid_directions)
    
    def _get_position_after_move(self, current_pos, move):
        """
        Calculate new position after making a move.
        
        Args:
            current_pos: (x, y) current position
            move: Action enum representing the move
            
        Returns:
            tuple: (x, y) new position
        """
        x, y = current_pos
        
        if move == Action.NORTH:
            return (x, y - 1)
        elif move == Action.SOUTH:
            return (x, y + 1)
        elif move == Action.EAST:
            return (x + 1, y)
        elif move == Action.WEST:
            return (x - 1, y)
        elif move == Action.NORTHEAST:
            return (x + 1, y - 1)
        elif move == Action.NORTHWEST:
            return (x - 1, y - 1)
        elif move == Action.SOUTHEAST:
            return (x + 1, y + 1)
        elif move == Action.SOUTHWEST:
            return (x - 1, y + 1)
        else:
            return current_pos  # TRAP or invalid move