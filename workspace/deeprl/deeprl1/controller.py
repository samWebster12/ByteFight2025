# controller.py

import numpy as np
import random
from collections.abc import Callable

from game.player_board import PlayerBoard
from game.enums import Action, Cell

from .model import NumpyPolicy


import os
base_dir = os.path.dirname(os.path.abspath(__file__))
weights_path = os.path.join(base_dir, "weights.npz")

# We assume your environment / training used the 9 discrete indices:
#   0=NORTH,1=NORTHEAST,2=EAST,3=SOUTHEAST,4=SOUTH,5=SOUTHWEST,6=WEST,7=NORTHWEST,8=TRAP
INDEX_TO_ACTION = [
    Action.NORTH, Action.NORTHEAST, Action.EAST, Action.SOUTHEAST,
    Action.SOUTH, Action.SOUTHWEST, Action.WEST, Action.NORTHWEST,
    Action.TRAP
]

class PlayerController:
    def __init__(self, time_left: Callable):
        """
        time_left: a callable returning how many seconds remain (float).
        In ByteFight, you have ~90 seconds total for your moves (chess-clock style).
        """
        self.time_left = time_left

        # Load your SB3->NumPy policy that you put in model.py
        # Make sure 'weights.npz' is in the same directory, or update the path below.
        self.model = NumpyPolicy(weights_path)


    def bid(self, board: PlayerBoard, time_left: Callable) -> int:
        """
        Return how much length you're willing to sacrifice to move first.
        We'll just return 0 for this example.
        """
        return 0

    def play(self, board: PlayerBoard, time_left: Callable):
        """
        Called each time it's your turn. Return a list of Actions (>=1).
        We'll do exactly one action per turn in this sample.
        """
        # 1) Encode the 9-channel observation as done by the gym env
        obs_image = self._encode_observation(board)

        # 2) Build the 9‑entry action_mask => 1 if valid, else 0
        #    (We only do this for the current player, i.e. "is_player_a=board.is_my_turn()")
        #    Since the gym env always checks if it's "A"’s turn or not, we assume
        #    your code is for player A. If you are B, you'd adapt similarly.
        #    We'll replicate what SingleProcessByteFightEnv does for the agent’s turn.

        action_mask = np.zeros((9,), dtype=np.float32)

        # Mark directions as valid if board says they're valid
        # get_possible_directions() checks which directions are physically possible
        possible_dirs = board.get_possible_directions(enemy=False)
        for move in possible_dirs:
            # If the board returns an int, convert to Action
            if isinstance(move, int):
                move_enum = Action(move)
            else:
                move_enum = move  # already an Action

            # Now you can do move_enum.value safely
            action_index = int(move_enum.value)
            if board.is_valid_move(move_enum):
                action_mask[action_index] = 1.0


        # If trap is valid, mark index 8
        if board.is_valid_trap(enemy=False):
            action_mask[8] = 1.0

        # If somehow no valid actions, we must forfeit (FF) or do a fallback
        # The environment code uses if sum ==0, then mask[0]=1, but that’s arbitrary.
        # We'll do the same:
        if action_mask.sum() == 0:
            # No valid moves: must yield or something
            return [Action.FF]

        # 3) Model forward pass => pick discrete action
        # The environment used a uint8 image in shape (9,64,64). That’s what we have.
        # Our model expects float, or it can handle uint8. If you want to convert to float
        #   do: obs_image.astype(np.float32) / 255.
        # But if your training used raw 0/255 inputs, keep as is.
        logits, _ = self.model.forward(obs_image, action_mask)

        # 4) Argmax policy
        discrete_act = int(np.argmax(logits))
        return [discrete_act]


    def _encode_observation(self, board: PlayerBoard) -> np.ndarray:
        """
        Replicates SingleProcessByteFightEnv._make_observation() for the 'image' portion,
        i.e. a (9,64,64) array with the channels:

          0: Walls
          1: Apples
          2: My snake head
          3: My snake body
          4: Enemy snake head
          5: Enemy snake body
          6: My traps
          7: Enemy traps
          8: Portals

        We'll fill channels up to dim_x, dim_y in the top-left.
        The rest is zero if the board is smaller than 64x64.
        """
        # Board size
        dim_x = board.get_dim_x()
        dim_y = board.get_dim_y()

        # 9 channels, each up to 64x64
        channels = np.zeros((9, dim_y, dim_x), dtype=np.uint8)

        # 0) Walls
        wall_mask = board.get_wall_mask()  # shape=(dim_y,dim_x), ==Cell.WALL if walls
        channels[0] = np.where(wall_mask == Cell.WALL, 255, 0)

        # 1) Apples
        apple_mask = board.get_apple_mask()  # shape=(dim_y,dim_x), ==Cell.APPLE if apples
        channels[1] = np.where(apple_mask == Cell.APPLE, 255, 0)

        # 2 & 3) My snake head & body
        my_snake = board.get_snake_mask(my_snake=True, enemy_snake=False)
        channels[2] = np.where(my_snake == Cell.PLAYER_HEAD, 255, 0)
        channels[3] = np.where(my_snake == Cell.PLAYER_BODY, 255, 0)

        # 4 & 5) Enemy snake head & body
        enemy_snake = board.get_snake_mask(my_snake=False, enemy_snake=True)
        channels[4] = np.where(enemy_snake == Cell.ENEMY_HEAD, 255, 0)
        channels[5] = np.where(enemy_snake == Cell.ENEMY_BODY, 255, 0)

        # 6) My traps
        my_traps = board.get_trap_mask(my_traps=True, enemy_traps=False)
        channels[6] = np.where(my_traps > 0, 255, 0)

        # 7) Enemy traps
        enemy_traps = board.get_trap_mask(my_traps=False, enemy_traps=True)
        channels[7] = np.where(enemy_traps > 0, 255, 0)

        # 8) Portals
        try:
            portal_mask = board.get_portal_mask(descriptive=False)
            # If descriptive=False, hopefully shape=(dim_y,dim_x), with 1=portal,0=not
            if portal_mask.ndim == 3:
                # Some versions might return (dim_y, dim_x, 2). We'll just pick the first slice:
                portal_mask = portal_mask[:, :, 0]
            channels[8] = np.where(portal_mask == 1, 255, 0)
        except:
            # If something breaks, just zero it out
            channels[8] = 0

        # Construct the final (9,64,64)
        image = np.zeros((9,64,64), dtype=np.uint8)
        # Copy the actual map data into top-left
        image[:, :dim_y, :dim_x] = channels[:, :dim_y, :dim_x]
        return image
