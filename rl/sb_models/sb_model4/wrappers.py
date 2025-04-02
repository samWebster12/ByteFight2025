import gymnasium as gym
import numpy as np
from game.enums import Action, Cell
from game.player_board import PlayerBoard

class OpponentWrapper(gym.Wrapper):
    """
    Wrapper to use a PPO model as the opponent in the ByteFight environment.
    """
    def __init__(self, env, opponent_model=None):
        super().__init__(env)
        self.opponent_model = opponent_model
        
    def step(self, action):
        # If it's player A's turn, use the provided action
        if self.env.unwrapped._board.is_as_turn():
            return self.env.step(action)
        
        # Otherwise, if we have an opponent model, use it for player B
        elif self.opponent_model is not None:
            # Create PlayerBoard for player B
            pb_b = PlayerBoard(False, self.env.unwrapped._board)
            obs_b = self._make_opponent_observation(pb_b)
            
            # Get action from opponent model
            opponent_action, _ = self.opponent_model.predict(obs_b, deterministic=True)
            
            # Apply action to environment
            return self.env.step(opponent_action)
        
        # Fall back to default opponent
        else:
            return self.env.step(action)
    
    def load_opponent_from_path(self, model_path):
        """
        Load an opponent model from a specified path.
        """
        try:
            from stable_baselines3 import PPO
            self.opponent_model = None  # Clear any existing reference
            self.opponent_model = PPO.load(model_path)
            return True
        except Exception as e:
            print(f"Error loading opponent model: {e}")
            return False
    
    def _make_opponent_observation(self, player_board):
        """
        Create observation for opponent model from player B's perspective.
        """
        # Get dimensions
        dim_x = player_board.get_dim_x()
        dim_y = player_board.get_dim_y()
        
        channels = np.zeros((9, dim_y, dim_x), dtype=np.uint8)
        
        # Wall mask
        wall_mask = player_board.get_wall_mask()
        channels[0] = np.where(wall_mask == Cell.WALL, 255, 0)
        
        # Apple mask
        apple_mask = player_board.get_apple_mask()
        channels[1] = np.where(apple_mask == Cell.APPLE, 255, 0)
        
        # Snake masks (from B's perspective)
        b_snake_mask = player_board.get_snake_mask(my_snake=True, enemy_snake=False)
        channels[2] = np.where(b_snake_mask == Cell.PLAYER_HEAD, 255, 0)
        channels[3] = np.where(b_snake_mask == Cell.PLAYER_BODY, 255, 0)
        
        a_snake_mask = player_board.get_snake_mask(my_snake=False, enemy_snake=True)
        channels[4] = np.where(a_snake_mask == Cell.ENEMY_HEAD, 255, 0)
        channels[5] = np.where(a_snake_mask == Cell.ENEMY_BODY, 255, 0)
        
        # Trap masks
        b_trap_mask = player_board.get_trap_mask(my_traps=True, enemy_traps=False)
        channels[6] = np.where(b_trap_mask > 0, 255, 0)
        
        a_trap_mask = player_board.get_trap_mask(my_traps=False, enemy_traps=True)
        channels[7] = np.where(a_trap_mask > 0, 255, 0)
        
        # Portal mask
        try:
            portal_mask = player_board.get_portal_mask(descriptive=False)
            if portal_mask.ndim == 3:
                portal_mask = portal_mask[:, :, 0]
            channels[8] = np.where(portal_mask == 1, 255, 0)
        except Exception:
            channels[8] = 0
        
        image = np.zeros((9, 64, 64), dtype=np.uint8)
        image[:, :dim_y, :dim_x] = channels[:, :dim_y, :dim_x]
        
        # Action mask for opponent
        mask = np.zeros((9,), dtype=np.uint8)
        try:
            valid_moves = []
            for move in range(8):
                action = Action(move)
                if player_board.is_valid_move(action):
                    valid_moves.append(action)
            for move in valid_moves:
                mask[int(move)] = 1
            if player_board.is_valid_trap():
                mask[8] = 1
            if sum(mask) == 0:
                mask[0] = 1
        except Exception:
            mask = np.ones((9,), dtype=np.uint8)
        
        return {"image": image, "action_mask": mask}
    
    def update_opponent(self, new_opponent_model):
        """
        Update the opponent model during training.
        """
        self.opponent_model = new_opponent_model


class MaskedActionWrapper(gym.ActionWrapper):
    """
    Wrapper to ensure the agent only selects valid actions based on the action mask.
    """
    def __init__(self, env):
        super().__init__(env)
    
    def action(self, action):
        """
        Apply the action mask to ensure only valid actions are taken.
        """
        observation = self.env.unwrapped._make_observation()
        action_mask = observation["action_mask"]
        
        # If the chosen action is invalid and there are valid actions available
        if action_mask[action] == 0 and np.sum(action_mask) > 0:
            # Choose a random valid action instead
            valid_actions = np.where(action_mask == 1)[0]
            action = np.random.choice(valid_actions)
        
        return action