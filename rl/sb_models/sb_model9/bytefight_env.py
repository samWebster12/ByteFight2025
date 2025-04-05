import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Optional, Dict, Any, Tuple, List
import os
import sys
import json

# Adjust path for your local environment
parent_dir = os.path.abspath(os.path.join(__file__, "../../../.."))
sys.path.insert(0, parent_dir)

from game.enums import Action, Result, Cell
from game.player_board import PlayerBoard
from game.board import Board
from opp_controller import OppController
from game.game_map import Map
from normalizer import RunningNormalizer
from opponent_pool import OpponentPool
from selfplay_controller import SelfPlayOppController


# A lookup from discrete int => ByteFight Action
# We have 8 directions + TRAP + special "END_TURN" sentinel
ACTION_LOOKUP = {
    0: Action.NORTH,
    1: Action.NORTHEAST,
    2: Action.EAST,
    3: Action.SOUTHEAST,
    4: Action.SOUTH,
    5: Action.SOUTHWEST,
    6: Action.WEST,
    7: Action.NORTHWEST,
    8: Action.TRAP,
    9: "END_TURN"       # We'll treat this specially in the step function
}

def get_heading_value(board: PlayerBoard, enemy: bool) -> int:
    """
    Get the current heading of the specified snake (0..7).
    If heading is None, just return 0.
    """
    heading = board.get_direction(enemy=enemy)
    if heading is None:
        return 0
    return int(heading.value) if not isinstance(heading, int) else heading

def get_map_string(map_name):
    if not os.path.exists("maps.json"):
        raise FileNotFoundError("maps.json file not found. Please make sure it exists in the current directory.")
    
    with open("maps.json", "r") as f:
        maps = json.load(f)

    if map_name not in maps:
        available_maps = ", ".join(maps.keys())
        raise KeyError(f"Map '{map_name}' not found in maps.json. Available maps: {available_maps}")
    
    return maps[map_name]


class ByteFightSnakeEnv(gym.Env):
    """
    A single-agent ByteFight environment that uses absolute directions as discrete actions:
      0 - NORTH
      1 - NORTHEAST
      2 - EAST
      3 - SOUTHEAST
      4 - SOUTH
      5 - SOUTHWEST
      6 - WEST
      7 - NORTHWEST
      8 - TRAP
      9 - END_TURN
    """

    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(
        self,
        map_names: List[Map],
        opponent_pool: OpponentPool,
        obs_normalizer: RunningNormalizer,
        render_mode: Optional[str] = None,
        use_opponent = True,
        verbose: bool = False,
        env_id=0
    ):
        super().__init__()
        self.verbose = verbose
        self.map_names = map_names
        self.opponent_pool = opponent_pool
        self.obs_normalizer = obs_normalizer
        self.render_mode = render_mode
        self.use_opponent = use_opponent
        self.env_id = env_id

        self.map_counts = {map_name: 0 for map_name in self.map_names}
        self.map_wins = {map_name: 0 for map_name in self.map_names}
        self.current_map_name = None

        # Discrete action space with 10 possible actions
        self.action_space = spaces.Discrete(10)
        
        # Board dimensions
        max_width = 64   # per ByteFight specs
        max_height = 64

        # Observation space
        self.observation_image_space = spaces.Box(
            low=0, high=1, shape=(9, max_height, max_width), dtype=np.float32
        )
        self.observation_scalar_space = spaces.Box(
            low=-1e6, high=1e6, shape=(15,), dtype=np.float32
        )
        self.observation_action_mask_space = spaces.Box(
            low=0, high=1, shape=(10,), dtype=np.uint8
        )
        self.observation_space = spaces.Dict({
            "board_image": self.observation_image_space,
            "features": self.observation_scalar_space,
            "action_mask": self.observation_action_mask_space,
        })

        # Internal state
        self.current_actions = []       # Accumulate actions until END_TURN
        self.done = False
        self.winner = None
        self.forecast_board = None      # For forecasting moves
        self._last_obs = None

        # Extra bookkeeping for shaping
        self._turn_start_agent_length = 0
        self._turn_start_agent_apples = 0
        self._turn_start_opponent_length = 0
        self._turn_start_opponent_apples = 0

        # Observation Normalization
        #self.obs_normalizer = RunningNormalizer(dim=15, clip_value=5.0)

        # Reward normalization variables
        self.reward_normalizer_initialized = False
        self.reward_mean = 0.0
        self.reward_var = 0.0
        self.reward_count = 0
        self.reward_clip_value = 3.0  # For example, clip normalized rewards to [-3, 3]

    def _handle_bidding(self) -> bool:
        """
        Handle the bidding phase at the start of the game (we default to 0).
        """
        try:
            # Opponent bid
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
            
            our_bid = 0  # we always bid 0 for simplicity
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
        """
        ByteFight 2025 Snake tiebreak logic.
        1) Compare apples eaten
        2) Compare final snake length
        3) Compare time on clock (5s margin)
        4) Else TIE
        """
        pb_a = PlayerBoard(True, self.board)
        a_apples = pb_a.get_apples_eaten(enemy=False)
        b_apples = pb_a.get_apples_eaten(enemy=True)
        a_len = pb_a.get_length(enemy=False)
        b_len = pb_a.get_length(enemy=True)
        a_time = pb_a.get_time_left(enemy=False)
        b_time = pb_a.get_time_left(enemy=True)

        if a_apples > b_apples:
            return Result.PLAYER_A
        elif b_apples > a_apples:
            return Result.PLAYER_B
        elif a_len > b_len:
            return Result.PLAYER_A
        elif b_len > a_len:
            return Result.PLAYER_B
        elif abs(a_time - b_time) > 5.0:
            return Result.PLAYER_A if a_time > b_time else Result.PLAYER_B
        else:
            return Result.TIE

    def _get_valid_action_mask(self) -> np.ndarray:
        """
        Create a mask for valid actions out of the 10 absolute actions.
        In ByteFight:
          - You can only move in directions up to 90° away from current heading
            but we simplify that by letting the environment handle invalid moves
            or you can do additional logic here if you want.
          - TRAP is valid only if you have unqueued length > 2
          - END_TURN is invalid if no moves have been made yet, because you must move
        We'll do a simple approach where we check if a forecast turn is valid.
        """
        mask = np.ones(10, dtype=np.uint8)  # default all valid

        if not self.forecast_board.is_as_turn():
            self.forecast_board.next_turn()

        pb = PlayerBoard(True, self.forecast_board)

        # If we haven't moved yet this turn, we can't do "END_TURN" or "TRAP" if length <= 2
        if len(self.current_actions) == 0:
            mask[9] = 0  # can't END_TURN as first action
            if pb.get_unqueued_length(enemy=False) <= 2:
                mask[8] = 0  # can't TRAP if physically length <= 2

        # Also check if trap is invalid in general
        if pb.get_unqueued_length(enemy=False) <= 2 or not pb.is_valid_trap():
            mask[8] = 0

        # For each directional action, test whether it would be valid with the forecast:
        # (We can do a minimal check here or rely on forecast to see if it fails.)
        # We'll do a quick forecast check for each direction, ignoring TRAP & END_TURN
        for act_id in range(8):  # directions only
            test_actions = self.current_actions.copy()
            test_actions.append(ACTION_LOOKUP[act_id])

            # forecast
            test_board = self.forecast_board.get_copy()
            success_board, valid = test_board.forecast_turn(test_actions)
            if not valid:
                mask[act_id] = 0

        # That’s enough basic logic to keep you from crashing on obviously invalid moves
        return mask

    def _build_observation(self) -> Dict[str, np.ndarray]:
        """
        Build the observation from the forecasted board perspective.
        Returns a dictionary of:
          - board_image: shape (9, H, W)
          - features: shape (15,)
          - action_mask: shape (10,)
        """
        pb_a = PlayerBoard(True, self.forecast_board)

        # Board dimensions
        height = pb_a.get_dim_y()
        width = pb_a.get_dim_x()

        # 1) Build the channel masks
        wall_mask = pb_a.get_wall_mask().astype(np.float32)
        apple_mask = pb_a.get_apple_mask().astype(np.float32)
        my_snake_mask = pb_a.get_snake_mask(my_snake=True, enemy_snake=False)
        opp_snake_mask = pb_a.get_snake_mask(my_snake=False, enemy_snake=True)

        # Extract head positions
        my_head_mask = np.zeros_like(my_snake_mask, dtype=np.float32)
        opp_head_mask = np.zeros_like(opp_snake_mask, dtype=np.float32)
        my_head_mask[my_snake_mask == Cell.PLAYER_HEAD] = 1
        opp_head_mask[opp_snake_mask == Cell.ENEMY_HEAD] = 1

        # Convert snake_mask to just the body
        my_body_mask = np.logical_and(my_snake_mask > 0, my_snake_mask != Cell.PLAYER_HEAD).astype(np.float32)
        opp_body_mask = np.logical_and(opp_snake_mask > 0, opp_snake_mask != Cell.ENEMY_HEAD).astype(np.float32)

        # Traps
        my_trap_mask = (pb_a.get_trap_mask(my_traps=True, enemy_traps=False) > 0).astype(np.float32)
        opp_trap_mask_raw = pb_a.get_trap_mask(my_traps=False, enemy_traps=True)
        opp_trap_mask = (opp_trap_mask_raw < 0).astype(np.float32)

        # Portals
        portal_mask_3d = pb_a.get_portal_mask(descriptive=False)
        if portal_mask_3d.ndim == 3 and portal_mask_3d.shape[-1] == 2:
            # Means it's giving us destination coords, so we use a simpler 0/1 approach
            portal_mask_2d = np.all(portal_mask_3d >= 0, axis=-1).astype(np.float32)
        else:
            portal_mask_2d = portal_mask_3d.astype(np.float32)

        # For consistent naming
        portal_mask = portal_mask_2d

        # Helper to ensure final shape is (64,64)
        def _pad_or_crop(arr, th=64, tw=64):
            h, w = arr.shape
            padded = np.zeros((th, tw), dtype=np.float32)
            h_to_copy = min(h, th)
            w_to_copy = min(w, tw)
            padded[:h_to_copy, :w_to_copy] = arr[:h_to_copy, :w_to_copy]
            return padded
        
        # Stack in the standard channel order
        #  0: Walls
        #  1: Apples
        #  2: My snake body
        #  3: Opp snake body
        #  4: My head
        #  5: Opp head
        #  6: My traps
        #  7: Opp traps
        #  8: Portals
        channels = [
            wall_mask,
            apple_mask,
            my_body_mask,
            opp_body_mask,
            my_head_mask,
            opp_head_mask,
            my_trap_mask,
            opp_trap_mask,
            portal_mask
        ]
        # Pad or crop each
        channels = [_pad_or_crop(ch) for ch in channels]
        board_image = np.stack(channels, axis=0).astype(np.float32)

            # 2) Scalar features
        turn_count = self.forecast_board.turn_count
        move_count = len(self.current_actions) + 1
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

        # Decay
        is_decaying = float(pb_a.currently_decaying())
        decay_interval = pb_a.get_current_decay_interval() or 0

        # Current sacrifice: first directional move 0, second move net 2, etc.
        current_sacrifice = 0
        if move_count > 1:
            non_trap_moves = [act for act in self.current_actions if act != Action.TRAP]
            current_sacrifice = (len(non_trap_moves))

        # Build the raw scalar features vector (15,)
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

        # ----- Online Normalization -----
        # Update stats with scalars
        self.obs_normalizer.update(scalars)

        # Normalize (no second update needed)
        norm_scalars = self.obs_normalizer.normalize(scalars, update_stats=False)

        # 3) Valid action mask
        action_mask = self._get_valid_action_mask()

        # Bundle everything
        obs = {
            "board_image": board_image,
            "features": norm_scalars,
            "action_mask": action_mask
        }
        return obs

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        super().reset(seed=seed)

        # Weighted map selection:
        alpha = 1.0
        beta = 1.0

        weights = []
        for map_name in self.map_names:
            plays = self.map_counts[map_name]
            wins = self.map_wins[map_name]
            # Compute estimated win rate with a Beta prior
            estimated_win_rate = (wins + alpha) / (plays + alpha + beta)
            # Weight maps with higher loss rate (i.e., lower estimated win rate) more heavily
            weight = 1 - estimated_win_rate
            weights.append(weight)

        weights = np.array(weights)
        weights = weights / weights.sum()  # Normalize the weights

        selected_map_name = np.random.choice(self.map_names, p=weights)
        self.current_map_name = selected_map_name

        game_map = Map(get_map_string(self.current_map_name))

        #if self.verbose:
        print(f"[DEBUG] Evaluation map: {self.current_map_name} (dimensions: {game_map.dim_x} x {game_map.dim_y})")

        self.board = Board(game_map, time_to_play=110)
        if not self.board.is_as_turn():
            self.board.next_turn()
        

        if self.opponent_pool.snapshots_meta:
            # Use the rating from the newest snapshot as main_rating
            main_rating = self.opponent_pool.snapshots_meta[-1]["rating"]
            # Exclude the newest snapshot (if there’s more than one)
            if len(self.opponent_pool.snapshots_meta) > 1:
                exclude_id = self.opponent_pool.snapshots_meta[-1]["id"]
            else:
                exclude_id = None
        else:
            main_rating = None
            exclude_id = None

        opp_policy, opp_index = self.opponent_pool.sample_opponent(main_rating=main_rating, exclude_id=exclude_id)

        # 2) Create a new SelfPlayOppController with that snapshot 
        self.opponent_controller = SelfPlayOppController(
            policy=opp_policy, 
            obs_normalizer=self.obs_normalizer
        )

        self._opponent_index = opp_index  # store so we know which snapshot we used

        self.current_actions = []
        self.done = False
        self.winner = None

        # Bidding
        bidding_success = self._handle_bidding()

        # Make a copy for forecasting
        self.forecast_board = self.board.get_copy()

        # Build obs
        obs = self._build_observation()
        self._last_obs = obs
        info = {"bidding_successful": bidding_success, "opponent_index": self._opponent_index}
        return obs, info

    def step(self, action: int) -> Tuple[Dict[str, np.ndarray], float, bool, bool, Dict[str, Any]]:
        """
        Step function using absolute moves. If action=9 => END_TURN, we apply
        the accumulated moves to the real board, then let the opponent move, etc.
        """
        if self.done:
            return self._last_obs, 0.0, True, False, {}

        reward = 0.0
        truncated = False
        info = {}

        pb_main = PlayerBoard(True, self.forecast_board)
        pb_opp = PlayerBoard(False, self.forecast_board)
        turn_start_agent_length = pb_main.get_length(enemy=False)
        turn_start_agent_apples = pb_main.get_apples_eaten(enemy=False)
        turn_start_opponent_length = pb_opp.get_length(enemy=False)
        turn_start_opponent_apples = pb_opp.get_apples_eaten(enemy=False)

        win_reward = 3
        loss_penalty = -3
        apple_reward = 0.05
        trap_penalty = -0.05
        length_change_reward = 0.05
        sacrifice_penalty = -0.05
        survival_bonus = 0.01


        # Convert discrete action => ByteFight action
        if action < 0 or action > 9:
            # out of bounds
            if self.verbose:
                print("[DEBUG] Invalid discrete action, out of range.")
            reward += loss_penalty
            self.done = True
            self.winner = Result.PLAYER_B
            return self._last_obs, reward, self.done, truncated, info

        bytefight_action = ACTION_LOOKUP[action]
        if self.verbose:
            print(f"[DEBUG] Action {action} => {bytefight_action}; current_actions={self.current_actions}")

        # If it's END_TURN
        if bytefight_action == "END_TURN":
            # Apply the accumulated actions to the main board
            if self.current_actions:
                if not self.board.is_as_turn():
                    self.board.next_turn()
                success = self.board.apply_turn(self.current_actions, a_to_play=True)
                if not success:
                    if self.verbose:
                        print("[DEBUG] Failed to apply our turn.")
                    reward += loss_penalty
                    self.done = True
                    self.winner = Result.PLAYER_B
                    return self._last_obs, reward, self.done, truncated, info

                # Check if we died from that move
                pb_main = PlayerBoard(True, self.board)
                if pb_main.get_length(enemy=False) < pb_main.get_min_player_size():
                    if self.verbose:
                        print("[DEBUG] Our snake length < min after turn")
                    reward += loss_penalty
                    self.done = True
                    self.winner = Result.PLAYER_B
                    return self._last_obs, reward, self.done, truncated, info
            
            # Opponent's turn
            if self.use_opponent:
                try:
                    pb_b = PlayerBoard(False, self.board)
                    #print("WITHIN ENV: Player B head: ", pb_b.get_head_location())
                    opp_move = self.opponent_controller.play(pb_b, pb_b.get_time_left(enemy=False))
                    if opp_move is None or (hasattr(opp_move, 'value') and opp_move.value == Action.FF.value):
                        #print("Opponnent Forfeited")
                        # Opponent forfeits => we win
                        if self.verbose:
                            print("[DEBUG] Opponent forfeits => We win.")
                        reward += win_reward
                        self.done = True
                        self.winner = Result.PLAYER_A
                    else:
                        # Opponent move could be single Action or list
                        if isinstance(opp_move, Action) or isinstance(opp_move, int):
                            opp_move = [opp_move]
                        if self.board.is_as_turn():
                            self.board.next_turn()
                        #print("Is a's turn (before): ", self.board.is_as_turn())
                        
                        #print("opp move: ", opp_move)
                        success = self.board.apply_turn(opp_move, a_to_play=False)
                        if not success:
                            #print("Is a's turn (after): ", self.board.is_as_turn())
                            #print("Oppponent made invalid move (forecasted)")
                            if self.verbose:
                                print(f"[DEBUG] Opponent made invalid move: {opp_move}")
                            reward += win_reward
                            self.done = True
                            self.winner = Result.PLAYER_A
                        else:
                            #print("opp move sucessful")
                            if self.verbose:
                                print(f"[DEBUG] Opponent successfully made move: {opp_move}")

                except Exception as e:
                    print("opponent crasehd")
                    if self.verbose:
                        print(f"[DEBUG] Opponent crashed: {e}")
                    reward += win_reward
                    self.done = True
                    self.winner = Result.PLAYER_A

            else:
                if self.verbose:
                    print(f"[DEBUG] No opponent turn. use_opponent set to {self.use_opponent}")
                if self.board.is_as_turn():
                    self.board.next_turn()

            pb_opp = PlayerBoard(False, self.board)
            curr_opp_length = pb_opp.get_length(enemy=False)
            opp_length_delta = curr_opp_length - turn_start_opponent_length

            if opp_length_delta > 0:
                reward -= length_change_reward * opp_length_delta

            # Reset for next turn
            self.forecast_board = self.board.get_copy()
            self.current_actions = []

        else:
            # It's a direction or TRAP
            if bytefight_action != Action.TRAP:
                # Movement => sacrifice penalty if not first move
                # (this is just an example shaping)
                if len(self.current_actions) >= 1:
                    non_trap_moves = [act for act in self.current_actions if act != Action.TRAP]
                    additional_sacrifice = max(0, len(non_trap_moves) - 1)
                    reward += additional_sacrifice * sacrifice_penalty
            else:
                # TRAP penalty
                reward += trap_penalty

            # Accumulate the action
            self.current_actions.append(bytefight_action)

            # Forecast
            if not self.board.is_as_turn():
                self.board.next_turn()

            forecast_board, forecast_success = self.board.forecast_turn(self.current_actions)
            if not forecast_success:
                if self.verbose:
                    print(f"[DEBUG] Invalid forecast for action: {bytefight_action}")
                reward += loss_penalty
                self.done = True
                self.winner = Result.PLAYER_B
            else:
                if self.verbose:
                    print("[DEBUG] Forecast success => updating forecast board.")
                if not forecast_board.is_as_turn():
                    forecast_board.next_turn()
                self.forecast_board = forecast_board
                reward += survival_bonus #survival bonus

            if not self.forecast_board.is_as_turn():
                self.forecast_board.next_turn()

            # Small incremental shaping
            pb_main = PlayerBoard(True, self.forecast_board)
            curr_agent_length = pb_main.get_length(enemy=False)
            curr_agent_apples = pb_main.get_apples_eaten(enemy=False)
            agent_length_delta = curr_agent_length - turn_start_agent_length
            agent_apple_delta = curr_agent_apples - turn_start_agent_apples

            reward += length_change_reward * agent_length_delta
            reward += apple_reward * agent_apple_delta

            #print("Length delta (Not END_TURN): ", agent_length_delta)
            #print("Apple delta (Not END_TURN): ", agent_apple_delta)



        # Global checks
        if not self.done:
            # Tiebreak at 2000
            if self.board.turn_count >= 2000:
                self.done = True
                self.winner = self._tiebreak()
                if self.winner == Result.PLAYER_A:
                    reward += 1.0
                elif self.winner == Result.PLAYER_B:
                    reward -= 1.0

            pb_main = PlayerBoard(True, self.board)
            if pb_main.is_game_over():
                self.done = True
                self.winner = self.board.get_winner()
                if self.winner == Result.PLAYER_A:
                    reward += 1.0
                elif self.winner == Result.PLAYER_B:
                    reward -= 1.0

        if self.done:
            # Cleanup
            self.forecast_board = self.board.get_copy()
            self.current_actions = []

            # Update map statistics here:
            self.map_counts[self.current_map_name] += 1
            # Assuming Result.PLAYER_A means our main agent wins (so we “lost” on that map),
            # update wins for the opponent or vice-versa depending on your definition.
            if self.winner == Result.PLAYER_A:
                self.map_wins[self.current_map_name] += 1
            # Optionally, save these statistics to a file.
            with open(f"map_stats/map_stats_env_{self.env_id}.json", "w") as f:
                json.dump({"map_counts": self.map_counts, "map_wins": self.map_wins}, f)


        obs = self._build_observation()
        self._last_obs = obs

        pb_main = PlayerBoard(True, self.forecast_board)
        info = {
            "winner": self.winner,
            "opponent_index": self._opponent_index,
            "turn_counter": self.board.turn_count,
            "move_counter": len(self.current_actions),
            "player_a_apples": pb_main.get_apples_eaten(enemy=False),
            "player_b_apples": pb_main.get_apples_eaten(enemy=True),
            "player_a_length": pb_main.get_length(enemy=False),
            "player_b_length": pb_main.get_length(enemy=True),
            "current_actions": self.current_actions.copy(),
            "current_map": self.current_map_name
        }

        # --- Reward Normalization & Clipping ---
        # Update running statistics for reward using Welford's algorithm
        '''
        if not self.reward_normalizer_initialized:
            self.reward_mean = reward
            self.reward_var = 0.0
            self.reward_count = 1
            self.reward_normalizer_initialized = True
        else:
            self.reward_count += 1
            delta = reward - self.reward_mean
            self.reward_mean += delta / self.reward_count
            self.reward_var += delta * (reward - self.reward_mean)
        
        # Calculate standard deviation
        reward_std = np.sqrt(self.reward_var / self.reward_count + 1e-8)
        # Normalize the reward
        norm_reward = (reward - self.reward_mean) / reward_std
        # Clip the normalized reward to [-reward_clip_value, reward_clip_value]
        norm_reward = np.clip(norm_reward, -self.reward_clip_value, self.reward_clip_value)
        # ----------------------------------------
        '''
        return obs, reward, self.done, truncated, info

    def render(self):
        """No built-in render; implement as needed."""
        pass

    def close(self):
        pass
