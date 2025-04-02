from game import player_board
from game.enums import Action, Cell
from collections.abc import Callable
import numpy as np
import os
import random

# Import your custom layers and model
from .composite_layers import MultiInputActorCriticPolicy

class PlayerController:
    def __init__(self, time_left: Callable):
        # Path to the weights file (in the same directory)
        weights_path = os.path.join(os.path.dirname(__file__), "weights.npz")
        
        # Initialize the model
        self.model = MultiInputActorCriticPolicy()
        
        # Load weights if the file exists
        if os.path.exists(weights_path):
            try:
                weights = np.load(weights_path)
                self.model.load_weights(weights)
                self.model_loaded = True
                print("Successfully loaded NumPy model weights")
            except Exception as e:
                print(f"Error loading model weights: {e}")
                self.model_loaded = False
        else:
            print(f"Weights file not found at {weights_path}")
            self.model_loaded = False

    def bid(self, board: player_board.PlayerBoard, time_left: Callable):
        # Simple bidding strategy - just bid 0
        return 0

    def play(self, board: player_board.PlayerBoard, time_left: Callable):
        # Fallback to random valid move if model not loaded
        if not hasattr(self, 'model_loaded') or not self.model_loaded:
            return self._random_valid_move(board)
        
        try:
            # Convert board state to model observation
            observation = self._board_to_observation(board)
            
            # Get model prediction
            logits, _ = self.model.forward(observation)
            
            # Apply action mask (although this should already be done in the forward method)
            masked_logits = logits * observation['action_mask']
            
            # Get the action with the highest probability
            # Note: action 8 is TRAP, actions 0-7 are directional moves
            best_action_idx = np.argmax(masked_logits)
            
            # Convert to Action enum
            if best_action_idx == 8:
                if board.is_valid_trap():
                    return Action.TRAP
                else:
                    # If trap is not valid, choose a movement action
                    return self._select_best_movement(board, masked_logits)
            else:
                action = Action(best_action_idx)
                if board.is_valid_move(action):
                    return [action]
                else:
                    # If selected action is not valid, fall back to best valid action
                    return self._select_best_movement(board, masked_logits)
        
        except Exception as e:
            print(f"Error during model prediction: {e}")
            return self._random_valid_move(board)

    def _board_to_observation(self, board: player_board.PlayerBoard):
        """Convert board state to the observation format expected by the model"""
        dim_x = board.board.map.dim_x
        dim_y = board.board.map.dim_y
        
        # Create 9-channel observation
        channels = np.zeros((9, dim_y, dim_x), dtype=np.uint8)
        
        # Channel 0: Walls
        wall_mask = board.get_wall_mask()
        channels[0] = np.where(wall_mask == Cell.WALL, 255, 0)
        
        # Channel 1: Apples
        apple_mask = board.get_apple_mask()
        channels[1] = np.where(apple_mask == Cell.APPLE, 255, 0)
        
        # Channel 2: My snake head
        a_snake_mask = board.get_snake_mask(my_snake=True, enemy_snake=False)
        channels[2] = np.where(a_snake_mask == Cell.PLAYER_HEAD, 255, 0)
        
        # Channel 3: My snake body
        channels[3] = np.where(a_snake_mask == Cell.PLAYER_BODY, 255, 0)
        
        # Channel 4: Enemy snake head
        b_snake_mask = board.get_snake_mask(my_snake=False, enemy_snake=True)
        channels[4] = np.where(b_snake_mask == Cell.ENEMY_HEAD, 255, 0)
        
        # Channel 5: Enemy snake body
        channels[5] = np.where(b_snake_mask == Cell.ENEMY_BODY, 255, 0)
        
        # Channel 6: My traps
        my_trap_mask = board.get_trap_mask(my_traps=True, enemy_traps=False)
        channels[6] = np.where(my_trap_mask > 0, 255, 0)
        
        # Channel 7: Enemy traps
        enemy_trap_mask = board.get_trap_mask(my_traps=False, enemy_traps=True)
        channels[7] = np.where(enemy_trap_mask > 0, 255, 0)
        
        # Channel 8: Portals
        try:
            portal_mask = board.get_portal_mask(descriptive=False)
            # Sometimes, even with descriptive=False, the mask might have an extra dimension
            if portal_mask.ndim == 3:
                portal_mask = portal_mask[:, :, 0]
            channels[8] = np.where(portal_mask == 1, 255, 0)
        except Exception as e:
            print(f"Warning: Error processing portal mask - {e}")
            channels[8] = 0
        
        # Pad to 64x64 if needed
        image = np.zeros((9, 64, 64), dtype=np.uint8)
        image[:, :dim_y, :dim_x] = channels[:, :dim_y, :dim_x]
        
        # Create action mask
        mask = np.zeros((1, 9), dtype=np.float32)
        
        # Set valid moves
        for i in range(8):
            action = Action(i)
            if board.is_valid_move(action):
                mask[0, i] = 1
        
        # Set trap validity
        if board.is_valid_trap():
            mask[0, 8] = 1
        
        # Ensure there's at least one valid action
        if np.sum(mask) == 0:
            mask[0, 0] = 1
        
        return {
            "image": image.astype(np.float32) / 255.0,  # Normalize to [0,1]
            "action_mask": mask
        }

    def _select_best_movement(self, board: player_board.PlayerBoard, logits):
        """Select the best valid movement action based on logits"""
        # Only consider movement actions (indices 0-7)
        movement_logits = logits[0, :8]
        valid_moves = []
        
        # Get valid moves with their corresponding logits
        for i in range(8):
            action = Action(i)
            if board.is_valid_move(action):
                valid_moves.append((action, movement_logits[i]))
        
        if not valid_moves:
            return self._random_valid_move(board)
        
        # Sort by logit value (descending) and take the best
        valid_moves.sort(key=lambda x: x[1], reverse=True)
        return [valid_moves[0][0]]

    def _random_valid_move(self, board: player_board.PlayerBoard):
        """Fallback to random valid move selection"""
        possible_moves = board.get_possible_directions()
        final_moves = []
        
        for move in possible_moves:
            if board.is_valid_move(move):
                final_moves.append(move)
        
        if len(final_moves) == 0:
            return Action.FF
        
        final_move = random.choice(final_moves)
        return [final_move]