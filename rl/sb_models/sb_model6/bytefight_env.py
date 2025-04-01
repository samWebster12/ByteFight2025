import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Optional, Dict, Any, Tuple

import os, sys
parent_dir = os.path.abspath(os.path.join(__file__, "../../../.."))
sys.path.insert(0, parent_dir)

from game.enums import Action, Result
from game.player_board import PlayerBoard, Board
from opp_controller import OppController
from game.game_map import Map

AGENT_ACTIONS = {
    0: "LEFT",
    1: "FORWARD_LEFT",
    2: "FORWARD",
    3: "FORWARD_RIGHT",
    4: "RIGHT",
    5: "TRAP",
    6: "END_TURN",
}

print(f"[DEBUG] AGENT_ACTIONS mapping: {AGENT_ACTIONS}")


def map_discrete_to_bytefight(action_id: int, heading: Action) -> Action:
    name = AGENT_ACTIONS[action_id]
    if name == "TRAP":
        return Action.TRAP
    elif name == "END_TURN":
        return "END_TURN"

    relative_to_absolute = {}
    relative_to_absolute["FORWARD"] = heading
    relative_to_absolute["LEFT"] = Action((heading.value - 2) % 8)
    relative_to_absolute["RIGHT"] = Action((heading.value + 2) % 8)
    relative_to_absolute["FORWARD_LEFT"] = Action((heading.value - 1) % 8)
    relative_to_absolute["FORWARD_RIGHT"] = Action((heading.value + 1) % 8)
    return relative_to_absolute[name]

def get_heading_value(board: PlayerBoard, enemy: bool) -> int:
    heading = board.get_direction(enemy=enemy)
    if heading is None:
        return 0
    return int(heading.value) if not isinstance(heading, int) else heading

class ByteFightSnakeEnv(gym.Env):
    """
    A single-agent ByteFight environment:
      - The agent uses discrete RELATIVE moves.
      - The opponent's moves are interpreted as absolute directions.
      - Debug prints the mask each step.
    """
    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(
        self,
        game_map: Map,
        opponent_controller: OppController,
        render_mode: Optional[str] = None,
        handle_bidding: bool = True,
        verbose = False
    ):
        super().__init__()
        self.verbose = verbose
        self.game_map = game_map
        self.opponent_controller = opponent_controller
        self.render_mode = render_mode
        self.handle_bidding = handle_bidding
        self.bidding_done = not handle_bidding

        self.action_space = spaces.Discrete(7)

        self.board = Board(game_map, time_to_play=100)

        max_width = 32
        max_height = 32
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

        self.turn_counter = 1
        self.move_counter = 1
        self.current_sacrifice = 0
        self.done = False
        self.winner = None

        self.player_a_apples = 0
        self.player_b_apples = 0
        self.player_a_time_left = 110.0
        self.player_b_time_left = 110.0
        self._last_obs = None

    def _handle_bidding(self):
        try:
            opponent_bid = 0
            if hasattr(self.opponent_controller, 'bid'):
                opponent_bid = self.opponent_controller.bid(self.board, 5.0)
                if not self.board.is_valid_bid(opponent_bid):
                    self.done = True
                    self.winner = Result.PLAYER_A
                    return False
            our_bid = 0
            pb_a = PlayerBoard(True, self.board)
            pb_a.apply_bid(our_bid, opponent_bid)
            self.bidding_done = True
            return True
        except Exception as e:
            self.done = True
            self.winner = Result.PLAYER_A
            return False

    def _tiebreak(self):
        pb_a = PlayerBoard(True, self.board)
        self.player_a_apples = pb_a.get_apples_eaten(enemy=False)
        self.player_b_apples = pb_a.get_apples_eaten(enemy=True)
        player_a_length = pb_a.get_length(enemy=False)
        player_b_length = pb_a.get_length(enemy=True)
        self.player_a_time_left = pb_a.get_time_left(enemy=False)
        self.player_b_time_left = pb_a.get_time_left(enemy=True)

        if self.player_a_apples > self.player_b_apples:
            return Result.PLAYER_A
        elif self.player_b_apples > self.player_a_apples:
            return Result.PLAYER_B
        elif player_a_length > player_b_length:
            return Result.PLAYER_A
        elif player_b_length > player_a_length:
            return Result.PLAYER_B
        elif abs(self.player_a_time_left - self.player_b_time_left) > 5.0:
            if self.player_a_time_left > self.player_b_time_left:
                return Result.PLAYER_A
            else:
                return Result.PLAYER_B
        else:
            return Result.TIE

    def _get_valid_action_mask(self) -> np.ndarray:
        mask = np.ones(7, dtype=bool)

        if self.move_counter == 1:
            mask[5] = False
            mask[6] = False

        pb_a = PlayerBoard(True, self.board)
        my_unqueued_len = pb_a.get_unqueued_length(enemy=False)
        if my_unqueued_len <= 2:
            mask[5] = False

        my_heading = get_heading_value(pb_a, enemy=False)
        for i in range(5):
            mapped = map_discrete_to_bytefight(i, Action(my_heading))
            if not self.board.is_valid_move(mapped, sacrifice=self.current_sacrifice, a_to_play=True):
                mask[i] = False

        return mask

    def _build_observation(self) -> Dict[str, np.ndarray]:
        pb_a = PlayerBoard(True, self.board)

        height = pb_a.get_dim_y()
        width = pb_a.get_dim_x()

        wall_mask = pb_a.get_wall_mask().astype(np.float32)
        apple_mask = pb_a.get_apple_mask().astype(np.float32)
        my_snake_mask = pb_a.get_snake_mask(my_snake=True, enemy_snake=False)
        opp_snake_mask = pb_a.get_snake_mask(my_snake=False, enemy_snake=True)

        my_head_mask = np.zeros_like(my_snake_mask)
        opp_head_mask = np.zeros_like(opp_snake_mask)
        my_head_mask[my_snake_mask == 3] = 1
        opp_head_mask[opp_snake_mask == 5] = 1

        my_snake_mask = np.logical_and(my_snake_mask > 0, my_snake_mask != 3).astype(np.float32)
        opp_snake_mask = np.logical_and(opp_snake_mask > 0, opp_snake_mask != 5).astype(np.float32)

        my_trap_mask = (pb_a.get_trap_mask(my_traps=True, enemy_traps=False) > 0).astype(np.float32)
        opp_trap_mask = (pb_a.get_trap_mask(my_traps=False, enemy_traps=True) > 0).astype(np.float32)

        portal_mask_3d = pb_a.get_portal_mask(descriptive=False)
        if portal_mask_3d.ndim == 3 and portal_mask_3d.shape[-1] == 2:
            portal_mask_2d = np.all(portal_mask_3d >= 0, axis=-1).astype(np.float32)
        else:
            portal_mask_2d = portal_mask_3d.astype(np.float32)
        portal_mask = portal_mask_2d

        def _pad_or_crop(arr, th=32, tw=32):
            h, w = arr.shape
            padded = np.zeros((th, tw), dtype=np.float32)
            h_to_copy = min(h, th)
            w_to_copy = min(w, tw)
            padded[:h_to_copy, :w_to_copy] = arr[:h_to_copy, :w_to_copy]
            return padded

        channels = []
        for m in [
            wall_mask, apple_mask, my_snake_mask, opp_snake_mask,
            my_head_mask, opp_head_mask, my_trap_mask, opp_trap_mask, portal_mask
        ]:
            channels.append(_pad_or_crop(m))
        board_image = np.stack(channels, axis=0).astype(np.float32)

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

        is_decaying = float(pb_a.currently_decaying())
        decay_interval = pb_a.get_current_decay_interval() or 0
        decay_rate = float(decay_interval)

        scalars = np.array([
            float(self.move_counter),
            float(self.turn_counter),
            float(self.current_sacrifice),
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
            decay_rate
        ], dtype=np.float32)

        action_mask = self._get_valid_action_mask().astype(np.uint8)

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
        
        # Create a completely new board instance from scratch for each episode
        self.board = Board(self.game_map)
        
        # Reset all internal state
        self.done = False
        self.winner = None
        self.turn_counter = 1
        self.move_counter = 1
        self.current_sacrifice = 0
        self.player_a_apples = 0
        self.player_b_apples = 0
        self.player_a_time_left = 110.0
        self.player_b_time_left = 110.0
        
        # Handle bidding
        pb_b = PlayerBoard(False, self.board)
        bidA, bidB = 0, self.opponent_controller.bid(pb_b, self.player_b_time_left)
        if self.board.is_valid_bid(bidA) and self.board.is_valid_bid(bidB):
            self.board.resolve_bid(bidA, bidB)
            if self.verbose:
                print("[DEBUG] Bidding Successful")
        else:
            self.done = True
            self.winner = Result.ERROR
            if self.verbose:
                print("[DEBUG] Bidding not Successful")
        
        obs = self._build_observation()
        self._last_obs = obs
        info = {"bidding_successful": True}
        return obs, info

    def step(self, action: int) -> Tuple[Dict[str, np.ndarray], float, bool, bool, Dict[str, Any]]:
        if self.done:
            return self._last_obs, 0.0, True, False, {}

        # Print out the previously computed mask from self._last_obs
        obs_mask = self._last_obs["action_mask"]  # shape=(7,) uint8
        bool_mask = obs_mask.astype(bool)
        if self.verbose:
            print(f"[DEBUG] Current mask={bool_mask}, chosen action={action}")

        pb_a = PlayerBoard(True, self.board)
        my_heading = get_heading_value(pb_a, enemy=False)
        mapped_action = map_discrete_to_bytefight(action, Action(my_heading))

        reward = 0.0
        truncated = False

        if self.verbose:
            print(f"[DEBUG STEP] turn={self.turn_counter} move={self.move_counter} "
                f"agent_action={action} => mapped_action={mapped_action} heading={my_heading} "
                f"sacrifice={self.current_sacrifice}")

        if mapped_action == "END_TURN":
            # After player A finishes their turn
            self.turn_counter += 1
            self.move_counter = 1
            self.current_sacrifice = 0
            
            # Apply decay if needed
            self._apply_decay_if_needed()
            
            # Now it's player B's turn
            try:
                # Create a PlayerBoard for player B
                opp_board = PlayerBoard(False, self.board.get_copy(False))
                opp_move = self.opponent_controller.play(opp_board, self.player_b_time_left)
                
                if opp_move is None or (hasattr(opp_move, 'value') and opp_move.value == Action.FF.value):
                    if self.verbose:
                        print("[DEBUG STEP] Opponent forfeits => We win")
                    reward = 1.0
                    self.done = True
                    self.winner = Result.PLAYER_A
                else:
                    # Process the opponent's move
                    if isinstance(opp_move, Action) or isinstance(opp_move, int):
                        opp_move = [opp_move]
                        
                    if self.verbose:
                        print(f"[DEBUG] Applying opponent move: {opp_move}")
                    
                    # DEBUG: Check if it's the opponent's turn according to the board
                    if self.verbose:
                        print(f"[DEBUG] Is it player A's turn? {self.board.is_as_turn()}")
                    # If this prints "False", it's player B's turn as expected
                    
                    # Try to apply the opponent's move
                    success = False
                    if not self.board.is_as_turn():  # If it's player B's turn
                        success = self.board.apply_turn(opp_move)
                    else:
                        # If it's still player A's turn, we need to end A's turn first
                        if self.verbose:
                            print("[DEBUG] Ending player A's turn first")
                        self.board.next_turn()  # This should switch turns
                        success = self.board.apply_turn(opp_move)
                    
                    if not success:
                        if self.verbose:
                            print(f"[DEBUG] Failed to apply move {opp_move}")
                        reward = 1.0
                        self.done = True
                        self.winner = Result.PLAYER_A
                    else:
                        self.turn_counter += 1
                        self._apply_decay_if_needed()
            except Exception as e:
                if self.verbose:
                    print(f"[DEBUG STEP] Opponent crashed => we win: {e}")
                reward = 1.0
                self.done = True
                self.winner = Result.PLAYER_A

        else:
            if mapped_action == Action.TRAP:
                success = self.board.apply_trap(a_to_play=True)
                if not success:
                    if self.verbose:
                        print("[DEBUG STEP] Trap invalid => we lose")
                    reward = -1.0
                    self.done = True
                    self.winner = Result.PLAYER_B
            else:
                success = self.board.apply_move(
                    mapped_action,
                    sacrifice=self.current_sacrifice,
                    a_to_play=True
                )
                if not success:
                    if self.verbose:
                        print("[DEBUG STEP] Move invalid => we lose")
                    reward = -1.0
                    self.done = True
                    self.winner = Result.PLAYER_B
                else:
                    self.move_counter += 1
                    if self.move_counter > 2:
                        self.current_sacrifice += 2

            if pb_a.get_length(enemy=False) < pb_a.get_min_player_size():
                if self.verbose:
                    print("[DEBUG STEP] Snake length < min => we died")
                reward = -1.0
                self.done = True
                self.winner = Result.PLAYER_B

        obs = self._build_observation()
        self._last_obs = obs

        info = {
            "winner": self.winner if self.done else None,
            "turn_counter": self.turn_counter,
            "move_counter": self.move_counter,
            "player_a_apples": pb_a.get_apples_eaten(enemy=False),
            "player_b_apples": pb_a.get_apples_eaten(enemy=True),
            "player_a_time_left": pb_a.get_time_left(enemy=False),
            "player_b_time_left": pb_a.get_time_left(enemy=True)
        }

        if self.turn_counter >= 2000 and not self.done:
            self.done = True
            self.winner = self._tiebreak()
            if self.winner == Result.PLAYER_A:
                reward = 1.0
            elif self.winner == Result.PLAYER_B:
                reward = -1.0
            else:
                reward = 0.0
            if self.verbose:
                print(f"[DEBUG STEP] 2000 turn limit => Tiebreak => winner={self.winner}")

        if not self.done and pb_a.is_game_over():
            self.done = True
            final_winner = pb_a.game_board.get_winner()
            if final_winner == Result.PLAYER_A:
                reward = 1.0
            elif final_winner == Result.PLAYER_B:
                reward = -1.0
            else:
                reward = 0.0
            self.winner = final_winner
            if self.verbose:
                print(f"[DEBUG STEP] Board says game over => winner={self.winner}")

        return obs, reward, self.done, truncated, info

    def _apply_decay_if_needed(self):
        decay_applied = False
        if self.turn_counter >= 1950:
            decay_applied = True
        elif self.turn_counter >= 1800 and self.turn_counter % 5 == 0:
            decay_applied = True
        elif self.turn_counter >= 1600 and self.turn_counter % 10 == 0:
            decay_applied = True
        elif self.turn_counter >= 1000 and self.turn_counter % 15 == 0:
            decay_applied = True

        pb_a = PlayerBoard(True, self.board)
        if decay_applied:
            try:
                before_len = pb_a.get_length(enemy=False)
                self.board.apply_decay(check_validity=True, a_to_play=True)
                after_len = pb_a.get_length(enemy=False)
                if self.verbose:
                    print(f"[DEBUG] Decay applied: length {before_len} -> {after_len}")
            except AttributeError:
                pass

            if pb_a.get_length(enemy=False) < pb_a.get_min_player_size():
                if self.verbose:
                    print("[DEBUG] Died from decay => losing")
                self.done = True
                self.winner = Result.PLAYER_B

    def render(self):
        pass

    def close(self):
        pass
