from game import player_board
from game.enums import Action, Cell
from collections.abc import Callable
import random
import numpy as np
import os
import sys

# Try to import PPO from stable_baselines3 if available
try:
    from stable_baselines3 import PPO
    MODEL_AVAILABLE = True
except ImportError:
    MODEL_AVAILABLE = False

class PlayerController:
    """
    A reinforcement learning agent controller for ByteFight.
    """
    def __init__(self, time_left: Callable):
        """
        Initialize the controller.
        
        Args:
            time_left: A callable that returns the time left for this player
        """
        self.model = None
        self.model_path = "ppo_bytefight_final"
        
        # Try to load the model if available
        if MODEL_AVAILABLE:
            try:
                if os.path.exists(self.model_path + ".zip"):
                    self.model = PPO.load(self.model_path)
                    print("Loaded RL model from", self.model_path)
                else:
                    print(f"Model file {self.model_path}.zip not found")
            except Exception as e:
                print(f"Error loading model: {e}")
                self.model = None
        else:
            print("stable_baselines3 not available, using fallback strategy")

    def bid(self, board: player_board.PlayerBoard, time_left: Callable):
        """
        Make a bid at the start of the game.
        
        Args:
            board: The game board
            time_left: A callable that returns the time left for this player
            
        Returns:
            int: The bid amount
        """
        # Simple bidding strategy: bid ~20% of initial length
        initial_length = board.get_length()
        max_bid = min(initial_length - board.get_min_player_size(), initial_length // 5)
        
        # Random variation to avoid tie bids
        return random.randint(max(0, max_bid-1), max_bid)

    def play(self, board: player_board.PlayerBoard, time_left: Callable):
        """
        Make a move.
        
        Args:
            board: The game board
            time_left: A callable that returns the time left for this player
            
        Returns:
            Action or list[Action]: The action(s) to take
        """
        # Get all legal moves
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
        
        # Use the model to choose an action if available
        if self.model is not None:
            try:
                # Create an observation similar to what the model was trained on
                obs = self._create_observation(board)
                
                # Get action from model
                action, _states = self.model.predict(obs, deterministic=True)
                
                # Convert to Action enum
                chosen_action = Action.TRAP if action == 8 else Action(action)
                
                # Make sure the action is valid
                if chosen_action in valid_moves:
                    return [chosen_action]
                else:
                    # Fall back to random valid move
                    return [random.choice(valid_moves)]
            except Exception as e:
                print(f"Error using model: {e}")
                # Fall back to random strategy
                return [random.choice(valid_moves)]
        else:
            # Fallback strategy if model isn't available:
            # Simple heuristic: prefer to move toward apples, or place traps if beneficial
            
            # Check if there are any apples
            apple_positions = board.get_apple_positions() if hasattr(board, 'get_apple_positions') else []
            
            if apple_positions and random.random() < 0.8:  # 80% chance to move toward apples
                # Move toward closest apple
                head_pos = board.get_head_location()
                
                # Find closest apple
                closest_apple = None
                min_dist = float('inf')
                for apple_pos in apple_positions:
                    dist = abs(apple_pos[0] - head_pos[0]) + abs(apple_pos[1] - head_pos[1])
                    if dist < min_dist:
                        min_dist = dist
                        closest_apple = apple_pos
                
                if closest_apple:
                    # Determine direction to move based on relative position
                    dx = closest_apple[0] - head_pos[0]
                    dy = closest_apple[1] - head_pos[1]
                    
                    # Try primary directions first
                    if dx > 0 and Action.EAST in valid_moves:
                        return [Action.EAST]
                    elif dx < 0 and Action.WEST in valid_moves:
                        return [Action.WEST]
                    elif dy > 0 and Action.SOUTH in valid_moves:
                        return [Action.SOUTH]
                    elif dy < 0 and Action.NORTH in valid_moves:
                        return [Action.NORTH]
                    
                    # Try diagonal directions if primary not available
                    if dx > 0 and dy < 0 and Action.NORTHEAST in valid_moves:
                        return [Action.NORTHEAST]
                    elif dx < 0 and dy < 0 and Action.NORTHWEST in valid_moves:
                        return [Action.NORTHWEST]
                    elif dx > 0 and dy > 0 and Action.SOUTHEAST in valid_moves:
                        return [Action.SOUTHEAST]
                    elif dx < 0 and dy > 0 and Action.SOUTHWEST in valid_moves:
                        return [Action.SOUTHWEST]
            
            # Consider placing a trap if we're long enough and in a good position
            if can_trap and board.get_unqueued_length() > 5 and random.random() < 0.2:
                return [Action.TRAP]
            
            # Default to random valid move
            return [random.choice(valid_moves)]

    def _create_observation(self, board: player_board.PlayerBoard):
        """
        Create an observation dictionary compatible with the trained model.
        
        Args:
            board: The game board
            
        Returns:
            dict: Observation dictionary with 'image' and 'action_mask' keys
        """
        # Get board dimensions
        dim_x = board.get_dim_x()
        dim_y = board.get_dim_y()
        
        # Create observation channels (9 channels, padded to 64x64)
        channels = np.zeros((9, 64, 64), dtype=np.uint8)
        
        # Channel 0: walls
        wall_mask = board.get_wall_mask()
        channels[0, :dim_y, :dim_x] = (wall_mask == Cell.WALL) * 255
        
        # Channel 1: apples
        apple_mask = board.get_apple_mask()
        channels[1, :dim_y, :dim_x] = (apple_mask == Cell.APPLE) * 255
        
        # Channel 2: my snake head
        my_snake_mask = board.get_snake_mask(my_snake=True, enemy_snake=False)
        channels[2, :dim_y, :dim_x] = (my_snake_mask == Cell.PLAYER_HEAD) * 255
        
        # Channel 3: my snake body
        channels[3, :dim_y, :dim_x] = (my_snake_mask == Cell.PLAYER_BODY) * 255
        
        # Channel 4: enemy snake head
        enemy_snake_mask = board.get_snake_mask(my_snake=False, enemy_snake=True)
        channels[4, :dim_y, :dim_x] = (enemy_snake_mask == Cell.ENEMY_HEAD) * 255
        
        # Channel 5: enemy snake body
        channels[5, :dim_y, :dim_x] = (enemy_snake_mask == Cell.ENEMY_BODY) * 255
        
        # Channel 6: my traps
        my_trap_mask = board.get_trap_mask(my_traps=True, enemy_traps=False)
        channels[6, :dim_y, :dim_x] = (my_trap_mask > 0) * 255
        
        # Channel 7: enemy traps
        enemy_trap_mask = board.get_trap_mask(my_traps=False, enemy_traps=True)
        channels[7, :dim_y, :dim_x] = (enemy_trap_mask > 0) * 255
        
        # Channel 8: portals
        portal_mask = board.get_portal_mask()
        channels[8, :dim_y, :dim_x] = (portal_mask == 1) * 255
        
        # Create action mask (which actions are valid)
        mask = np.zeros(9, dtype=np.uint8)
        
        # Set valid moves in mask
        for move in range(8):  # 8 directional moves
            action = Action(move)
            if board.is_valid_move(action):
                mask[move] = 1
        
        # Check if trap placement is valid
        if board.is_valid_trap():
            mask[8] = 1
        
        return {
            "image": channels,
            "action_mask": mask
        }