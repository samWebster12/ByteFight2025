import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Optional, Dict, Any, Tuple

import os, sys
parent_dir = os.path.abspath(os.path.join(__file__, "../../../.."))
sys.path.insert(0, parent_dir)

from game.enums import Action, Result, Cell
from game.player_board import PlayerBoard
from game.board import Board
from opp_controller import OppController
from game.game_map import Map

# Mapping from discrete action space to relative directions
AGENT_ACTIONS = {
    0: "LEFT",         # 90 degrees left of current heading
    1: "FORWARD_LEFT", # 45 degrees left of current heading
    2: "FORWARD",      # Continue in current heading
    3: "FORWARD_RIGHT",# 45 degrees right of current heading
    4: "RIGHT",        # 90 degrees right of current heading
    5: "TRAP",         # Place trap
    6: "END_TURN",     # End current turn
}

def map_discrete_to_bytefight(action_id: int, heading: Action) -> Action:
    """Convert from gym's discrete action space to ByteFight's action system"""
    name = AGENT_ACTIONS[action_id]
    if name == "TRAP":
        return Action.TRAP
    elif name == "END_TURN":
        return "END_TURN"

    # Map relative direction to absolute direction based on current heading
    relative_to_absolute = {}
    relative_to_absolute["FORWARD"] = heading
    relative_to_absolute["LEFT"] = Action((heading.value - 2) % 8)
    relative_to_absolute["RIGHT"] = Action((heading.value + 2) % 8)
    relative_to_absolute["FORWARD_LEFT"] = Action((heading.value - 1) % 8)
    relative_to_absolute["FORWARD_RIGHT"] = Action((heading.value + 1) % 8)
    return relative_to_absolute[name]

def get_heading_value(board: PlayerBoard, enemy: bool) -> int:
    """Get the current heading of the specified snake"""
    heading = board.get_direction(enemy=enemy)
    if heading is None:
        return 0
    return int(heading.value) if not isinstance(heading, int) else heading

class ByteFightSnakeEnv(gym.Env):
    """
    A single-agent ByteFight environment using forecast methods:
      - The agent uses discrete RELATIVE moves
      - Actions are accumulated and only applied at END_TURN
      - Forecasting is used to provide observations without modifying the main board
    """
    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(
        self,
        game_map: Map,
        opponent_controller: OppController,
        render_mode: Optional[str] = None,
        verbose: bool = False
    ):
        super().__init__()
        self.verbose = verbose
        self.game_map = game_map
        self.opponent_controller = opponent_controller
        self.render_mode = render_mode
        
        # Discrete action space with 7 possible actions
        self.action_space = spaces.Discrete(7)

        # Initialize the main game board
        self.board = Board(game_map, time_to_play=110)
        
        # Board dimensions
        max_width = 64  # Max from the requirements
        max_height = 64 # Max from the requirements

        # Observation space definition
        self.observation_image_space = spaces.Box(
            low=0,
            high=1,
            shape=(9, max_height, max_width),
            dtype=np.float32
        )
        self.observation_scalar_space = spaces.Box(
            low=-1e6,
            high=1e6,
            shape=(15,),
            dtype=np.float32
        )
        self.observation_action_mask_space = spaces.Box(
            low=0,
            high=1,
            shape=(7,),
            dtype=np.uint8
        )
        self.observation_space = spaces.Dict({
            "board_image": self.observation_image_space,
            "features": self.observation_scalar_space,
            "action_mask": self.observation_action_mask_space,
        })

        # State variables
        self.current_actions = []  # Store actions until END_TURN
        self.done = False
        self.winner = None
        self.forecast_board = None  # Board for forecasting without changing main board
        self._last_obs = None

    def _handle_bidding(self) -> bool:
        """Handle the bidding phase at the start of the game"""
        try:
            # Get opponent's bid
            opponent_bid = 0
            if hasattr(self.opponent_controller, 'bid'):
                pb_b = PlayerBoard(False, self.board)
                opponent_bid = self.opponent_controller.bid(pb_b, 5.0)
                if not self.board.is_valid_bid(opponent_bid):
                    self.done = True
                    self.winner = Result.PLAYER_A
                    if self.verbose:
                        print(f"[DEBUG] Invalid opponent bid: {opponent_bid}")
                    return False
            
            # Our bid is 0 for simplicity
            our_bid = 0
            
            # Resolve the bid
            self.board.resolve_bid(our_bid, opponent_bid)
            if self.verbose:
                print(f"[DEBUG] Bidding completed: our_bid={our_bid}, opponent_bid={opponent_bid}")
            return True
        except Exception as e:
            self.done = True
            self.winner = Result.PLAYER_A
            if self.verbose:
                print(f"[DEBUG] Bidding error: {e}")
            return False

    def _tiebreak(self) -> Result:
        """Determine the winner based on tiebreak rules"""
        pb_a = PlayerBoard(True, self.board)
        
        player_a_apples = pb_a.get_apples_eaten(enemy=False)
        player_b_apples = pb_a.get_apples_eaten(enemy=True)
        player_a_length = pb_a.get_length(enemy=False)
        player_b_length = pb_a.get_length(enemy=True)
        player_a_time = pb_a.get_time_left(enemy=False)
        player_b_time = pb_a.get_time_left(enemy=True)

        # Follow ByteFight tiebreak rules
        if player_a_apples > player_b_apples:
            return Result.PLAYER_A
        elif player_b_apples > player_a_apples:
            return Result.PLAYER_B
        elif player_a_length > player_b_length:
            return Result.PLAYER_A
        elif player_b_length > player_a_length:
            return Result.PLAYER_B
        elif abs(player_a_time - player_b_time) > 5.0:
            if player_a_time > player_b_time:
                return Result.PLAYER_A
            else:
                return Result.PLAYER_B
        else:
            return Result.TIE

    def _get_valid_action_mask(self) -> np.ndarray:
        """Create a mask for valid actions in the current state"""
        mask = np.ones(7, dtype=np.uint8)
        if not self.forecast_board.is_as_turn():
            self.forecast_board.next_turn()
        
        # Create a PlayerBoard for checking valid actions
        pb = PlayerBoard(True, self.forecast_board)
        
        # If it's the first move in a turn, certain actions are invalid
        if not self.current_actions:
            # Can't place trap as first action if not enough unqueued length
            if pb.get_unqueued_length(enemy=False) <= 2:
                mask[5] = 0  # TRAP

        else:
            # END_TURN is always valid after first move
            # Other actions depend on validity checks
            pass
            
        # Check valid movement directions
        my_heading = get_heading_value(pb, enemy=False)
        for i in range(5):  # Check the 5 movement directions
            mapped = map_discrete_to_bytefight(i, Action(my_heading))
            # Create a copy of current forecast board and actions to test this move

            test_board = self.forecast_board.get_copy()
            test_actions = self.current_actions.copy()
            test_actions.append(mapped)
            # Check if this move would be valid
            if not test_board.is_valid_turn(test_actions, a_to_play=True):
                mask[i] = 0
        
        # Check trap validity
        if pb.get_unqueued_length(enemy=False) <= 2 or not pb.is_valid_trap():
            mask[5] = 0  # TRAP

        if len(self.current_actions) == 0:
            mask[5] = 0
            mask[6] = 0

        return mask

    def _build_observation(self) -> Dict[str, np.ndarray]:
        """Build observation from the current forecasted board state"""
        pb_a = PlayerBoard(True, self.forecast_board)

        # Get board dimensions
        height = pb_a.get_dim_y()
        width = pb_a.get_dim_x()

        # Create feature masks
        wall_mask = pb_a.get_wall_mask().astype(np.float32)
        apple_mask = pb_a.get_apple_mask().astype(np.float32)
        my_snake_mask = pb_a.get_snake_mask(my_snake=True, enemy_snake=False)
        opp_snake_mask = pb_a.get_snake_mask(my_snake=False, enemy_snake=True)

        # Extract head positions
        my_head_mask = np.zeros_like(my_snake_mask)
        opp_head_mask = np.zeros_like(opp_snake_mask)
        my_head_mask[my_snake_mask == Cell.PLAYER_HEAD] = 1
        opp_head_mask[opp_snake_mask == Cell.ENEMY_HEAD] = 1

        # Remove heads from body masks
        my_snake_mask = np.logical_and(my_snake_mask > 0, my_snake_mask != Cell.PLAYER_HEAD).astype(np.float32)
        opp_snake_mask = np.logical_and(opp_snake_mask > 0, opp_snake_mask != Cell.ENEMY_HEAD).astype(np.float32)

        # Get trap masks
        my_trap_mask = (pb_a.get_trap_mask(my_traps=True, enemy_traps=False) > 0).astype(np.float32)
        opp_trap_mask_raw = pb_a.get_trap_mask(my_traps=False, enemy_traps=True)
        opp_trap_mask = (opp_trap_mask_raw < 0).astype(np.float32)

        # Get portal mask
        portal_mask_3d = pb_a.get_portal_mask(descriptive=False)
        if portal_mask_3d.ndim == 3 and portal_mask_3d.shape[-1] == 2:
            portal_mask_2d = np.all(portal_mask_3d >= 0, axis=-1).astype(np.float32)
        else:
            portal_mask_2d = portal_mask_3d.astype(np.float32)
        portal_mask = portal_mask_2d

        # Function to pad or crop masks to maximum dimension
        def _pad_or_crop(arr, th=64, tw=64):
            h, w = arr.shape
            padded = np.zeros((th, tw), dtype=np.float32)
            h_to_copy = min(h, th)
            w_to_copy = min(w, tw)
            padded[:h_to_copy, :w_to_copy] = arr[:h_to_copy, :w_to_copy]
            return padded

        # Stack all feature channels
        channels = []
        for m in [
            wall_mask, apple_mask, my_snake_mask, opp_snake_mask,
            my_head_mask, opp_head_mask, my_trap_mask, opp_trap_mask, portal_mask
        ]:
            channels.append(_pad_or_crop(m))
        board_image = np.stack(channels, axis=0).astype(np.float32)

        # Get scalar features
        turn_count = self.forecast_board.turn_count
        move_count = len(self.current_actions) + 1  # +1 because we're about to make another move
        my_heading = get_heading_value(pb_a, enemy=False)
        opp_heading = get_heading_value(pb_a, enemy=True)
        my_length = pb_a.get_length(enemy=False)
        opp_length = pb_a.get_length(enemy=True)
        my_queued = pb_a.get_queued_length(enemy=False)
        opp_queued = pb_a.get_queued_length(enemy=True)
        my_apples = pb_a.get_apples_eaten(enemy=False)
        opp_apples = pb_a.get_apples_eaten(enemy=True)
        my_max_len = pb_a.get_max_length(enemy=False)
        opp_max_len = pb_a.get_max_length(enemy=True)
        
        # Decay information
        is_decaying = float(pb_a.currently_decaying())
        decay_interval = pb_a.get_current_decay_interval() or 0
        
        # Calculate current sacrifice (though not needed with forecast approach)
        current_sacrifice = 0  # First move has no sacrifice
        if move_count > 1:
            current_sacrifice = (move_count - 1) * 2  # Each subsequent move increases by 2

        # Build scalar features array
        scalars = np.array([
            float(move_count),
            float(turn_count),
            float(current_sacrifice),
            float(my_heading),
            float(opp_heading),
            float(my_length),
            float(opp_length),
            float(my_queued),
            float(opp_queued),
            float(my_apples),
            float(opp_apples),
            float(my_max_len),
            float(opp_max_len),
            is_decaying,
            float(decay_interval)
        ], dtype=np.float32)

        # Get valid action mask
        action_mask = self._get_valid_action_mask()

        # Combine everything into the observation
        obs = {
            "board_image": board_image,
            "features": scalars,
            "action_mask": action_mask
        }
        return obs

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        super().reset(seed=seed)
        
        # Create a completely new board instance from scratch
        self.board = Board(self.game_map, time_to_play=110)
        if not self.board.is_as_turn():
            self.board.next_turn()
        
        # Reset all internal state
        self.current_actions = []
        self.done = False
        self.winner = None
        pb_main = PlayerBoard(True, self.board)
        pb_opp = PlayerBoard(False, self.board)
        self._turn_start_agent_length = pb_main.get_length(enemy=False)
        self._turn_start_agent_apples = pb_main.get_apples_eaten(enemy=False)
        self._turn_start_opponent_length = pb_opp.get_length(enemy=False)
        self._turn_start_opponent_apples = pb_opp.get_apples_eaten(enemy=False)
        
        # Handle bidding phase
        bidding_success = self._handle_bidding()
        
        # Create a forecast board that will be updated with each action
        self.forecast_board = self.board.get_copy()
        
        # Generate initial observation
        obs = self._build_observation()
        self._last_obs = obs
        
        info = {"bidding_successful": bidding_success}
        return obs, info

    def step(self, action: int) -> Tuple[Dict[str, np.ndarray], float, bool, bool, Dict[str, Any]]:
        """Take an action with improved reward shaping, where the move penalty equals the sacrifice cost.
        No move penalty is applied on the first move of a turn, on END_TURN, or on trap actions.
        """
        if self.done:
            return self._last_obs, 0.0, True, False, {}

        reward = 0.0
        truncated = False
        info = {}
        

        # Get ByteFight-formatted action from the discrete action
        pb_a = PlayerBoard(True, self.forecast_board)
        my_heading = get_heading_value(pb_a, enemy=False)
        mapped_action = map_discrete_to_bytefight(action, Action(my_heading))
        
        if self.verbose:
            print(f"[DEBUG] Action {action} mapped to {mapped_action}, current actions: {self.current_actions}")
        
        # Handle END_TURN separately (no move penalty here)
        if mapped_action == "END_TURN":
            # Apply the accumulated actions to the main board
            if self.current_actions:
                if not self.board.is_as_turn():
                    self.board.next_turn()
                success = self.board.apply_turn(self.current_actions, a_to_play=True)
                if not success:
                    if self.verbose:
                        print("[DEBUG] Failed to apply our turn")
                    reward -= 5.0
                    self.done = True
                    self.winner = Result.PLAYER_B
                    return self._last_obs, reward, self.done, truncated, info

                # (Optional) Check if our snake’s length has fallen below the minimum after our turn
                pb_main = PlayerBoard(True, self.board)
                if pb_main.get_length(enemy=False) < pb_main.get_min_player_size():
                    if self.verbose:
                        print("[DEBUG] Our snake length < min after our turn")
                    reward -= 5.0
                    self.done = True
                    self.winner = Result.PLAYER_B
                    return self._last_obs, reward, self.done, truncated, info
            
            # Process the opponent’s turn (same as before)
            try:
                pb_b = PlayerBoard(False, self.board)
                opp_move = self.opponent_controller.play(pb_b, pb_b.get_time_left(enemy=False))
                if opp_move is None or (hasattr(opp_move, 'value') and opp_move.value == Action.FF.value):
                    if self.verbose:
                        print("[DEBUG] Opponent forfeits => We win")
                    reward += 5.0
                    self.done = True
                    self.winner = Result.PLAYER_A
                else:
                    if isinstance(opp_move, Action) or isinstance(opp_move, int):
                        opp_move = [opp_move]
                    if self.board.is_as_turn():
                        self.board.next_turn()
                    success = self.board.apply_turn(opp_move, a_to_play=False)
                    if not success:
                        if self.verbose:
                            print(f"[DEBUG] Opponent made invalid move: {opp_move}")
                        reward += 5.0
                        self.done = True
                        self.winner = Result.PLAYER_A
            except Exception as e:
                if self.verbose:
                    print(f"[DEBUG] Opponent crashed: {e}")
                reward += 5.0
                self.done = True
                self.winner = Result.PLAYER_A

            # End-of-turn shaping: (you can keep your survival bonus and delta rewards here)
            reward += 0.05  # survival bonus

            # Compute deltas from turn start (assuming these were stored at turn start)
            pb_main = PlayerBoard(True, self.board)
            curr_agent_length = pb_main.get_length(enemy=False)
            curr_agent_apples = pb_main.get_apples_eaten(enemy=False)
            agent_length_delta = curr_agent_length - self._turn_start_agent_length
            agent_apple_delta = curr_agent_apples - self._turn_start_agent_apples

            pb_opp = PlayerBoard(False, self.board)
            curr_opp_length = pb_opp.get_length(enemy=False)
            opp_length_delta = self._turn_start_opponent_length - curr_opp_length

            if agent_length_delta > 0:
                reward += 0.3 * agent_length_delta
            elif agent_length_delta < 0:
                reward += 0.3 * agent_length_delta

            if agent_apple_delta > 0:
                reward += 0.1 * agent_apple_delta  # you might reduce this to keep apple bonus lower

            if opp_length_delta > 0:
                reward += 0.3 * opp_length_delta

            # Reset turn state for the next turn
            self.forecast_board = self.board.get_copy()
            self.current_actions = []
            pb_main = PlayerBoard(True, self.board)
            pb_opp = PlayerBoard(False, self.board)
            self._turn_start_agent_length = pb_main.get_length(enemy=False)
            self._turn_start_agent_apples = pb_main.get_apples_eaten(enemy=False)
            self._turn_start_opponent_length = pb_opp.get_length(enemy=False)
            self._turn_start_opponent_apples = pb_opp.get_apples_eaten(enemy=False)

        else:
            # For non-END_TURN actions:
            # If the action is a directional move (i.e. not a TRAP), add a move penalty corresponding to the extra sacrifice.
            if mapped_action != Action.TRAP:
                # Only apply a penalty if there are already some actions in this turn
                if len(self.current_actions) >= 1:
                    # According to rules, the first move has a sacrifice of 0,
                    # the second move sacrifices 3 (net cost 2), the third sacrifices 5 (net cost 4), etc.
                    # We'll use the extra sacrifice beyond the base 1 cell:
                    non_trap_actions = [act for act in self.current_actions if act != Action.TRAP]
                    additional_sacrifice = max(0, len(non_trap_actions) - 1) * 2
                    reward -= additional_sacrifice * 0.1
            
            else:
                reward -= 0.1

            # Accumulate the action in the current turn
            self.current_actions.append(mapped_action)
            
            # Update forecast board
            if not self.board.is_as_turn():
                self.board.next_turn()
            forecast_board, forecast_success = self.board.forecast_turn(self.current_actions)
            if not forecast_success:
                if self.verbose:
                    print(f"[DEBUG] Invalid forecast for action: {mapped_action}")
                reward -= 5.0
                self.done = True
                self.winner = Result.PLAYER_B
            else:
                if self.verbose:
                    print("[DEBUG] Forecast Success, updating board")
                if not forecast_board.is_as_turn():
                    forecast_board.next_turn()
                self.forecast_board = forecast_board

            # Update incremental rewards immediately after this move:
            pb_a = PlayerBoard(True, self.forecast_board)
            curr_length = pb_a.get_length(enemy=False)
            curr_apples = pb_a.get_apples_eaten(enemy=False)
            length_delta = curr_length - self._turn_start_agent_length
            apple_delta = curr_apples - self._turn_start_agent_apples
            if length_delta > 0:
                reward += 0.3 * length_delta
            elif length_delta < 0:
                reward += 0.3 * length_delta
            if apple_delta > 0:
                reward += 0.1 * apple_delta

        # Global game-over checks
        if not self.done:
            if self.board.turn_count >= 2000:
                self.done = True
                self.winner = self._tiebreak()
                if self.winner == Result.PLAYER_A:
                    reward += 1.0
                elif self.winner == Result.PLAYER_B:
                    reward -= 1.0
                else:
                    reward += 0.0
            pb_main = PlayerBoard(True, self.board)
            if pb_main.is_game_over():
                self.done = True
                self.winner = self.board.get_winner()
                if self.winner == Result.PLAYER_A:
                    reward += 1.0
                elif self.winner == Result.PLAYER_B:
                    reward -= 1.0

        if self.done:
            self.forecast_board = self.board.get_copy()
            self.current_actions = []

        obs = self._build_observation()
        self._last_obs = obs
        pb_main = PlayerBoard(True, self.forecast_board)
        info = {
            "winner": self.winner,
            "turn_counter": self.board.turn_count,
            "move_counter": len(self.current_actions),
            "player_a_apples": pb_main.get_apples_eaten(enemy=False),
            "player_b_apples": pb_main.get_apples_eaten(enemy=True),
            "player_a_length": pb_main.get_length(enemy=False),
            "player_b_length": pb_main.get_length(enemy=True),
            "current_actions": self.current_actions.copy()
        }

        return obs, reward, self.done, False, info




    def render(self):
        """Render the environment (placeholder)"""
        pass

    def close(self):
        """Clean up the environment (placeholder)"""
        pass