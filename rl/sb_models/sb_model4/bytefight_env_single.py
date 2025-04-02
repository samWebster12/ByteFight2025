import gymnasium as gym
from gymnasium import spaces
import numpy as np
import sys, os, traceback, importlib
import random
from collections.abc import Callable


parent_dir = os.path.abspath(os.path.join(__file__, "../../../.."))
sys.path.insert(0, parent_dir)

from game.game_map import Map
from game.board import Board
from game.player_board import PlayerBoard
from game.enums import Action, Result, Cell

class SingleProcessByteFightEnv(gym.Env):
    """
    A single-process Gymnasium environment for ByteFight: Snake with improved reward shaping:
      - Reward for increasing snake length
      - Reward for eating apples
      - Reward for opponent length decrease (trap hits)
      - Reward for opponent hitting your traps
      - Penalty for decreasing your own length through multiple moves
      - Heavy penalty for invalid moves
      - Win/loss rewards at episode end
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        map_string,
        opponent_module,
        submission_dir,
        max_steps=2000,  # match the ByteFight's typical tiebreak limit
        render_mode=None
    ):
        super().__init__()
        self.map_string = map_string
        self.opponent_module = opponent_module
        self.submission_dir = submission_dir
        self.render_mode = render_mode
        self.max_steps = max_steps
        self.current_step = 0

        # 9 discrete actions: 8 directions + TRAP
        self.action_space = spaces.Discrete(9)

        # Observation is a dict with "image" (9 channels) and "action_mask"(9).
        self.observation_space = spaces.Dict({
            "image": spaces.Box(low=0, high=255, shape=(9,64,64), dtype=np.uint8),
            "action_mask": spaces.Box(low=0, high=1, shape=(9,), dtype=np.uint8)
        })

        # Attempt to load opponent
        sys.path.append(self.submission_dir)
        try:
            opp_mod = importlib.import_module(self.opponent_module + ".controller")
            time_left_callable = lambda: 5.0
            self._opponent_controller = opp_mod.PlayerController(time_left_callable)
        except Exception as e:
            traceback.print_exc()
            raise ImportError(f"Could not load opponent module {self.opponent_module}") from e

        self._board = None
        self._done = False
        self._winner = None

        # Track agent/opponent stats for shaping
        self._agent_prev_length = 0
        self._agent_prev_apples = 0
        self._opp_prev_length = 0
        self._opp_prev_apples = 0
        self._agent_max_length = 0
        self._agent_traps_placed = 0
        self._last_turn_count = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._done = False
        self._winner = None
        self.current_step = 0
        self._last_turn_count = 0

        map_obj = Map(self.map_string)
        self._board = Board(map_obj, time_to_play=100, build_history=False)

        # Both players bid 0
        bidA, bidB = 0, 0
        if self._board.is_valid_bid(bidA) and self._board.is_valid_bid(bidB):
            self._board.resolve_bid(bidA, bidB)
        else:
            self._done = True
            self._winner = Result.ERROR

        # Initialize agent/opponent stats
        pb_a = PlayerBoard(True, self._board)
        pb_b = PlayerBoard(False, self._board)

        self._agent_prev_length = pb_a.get_length()
        self._agent_prev_apples = pb_a.get_apples_eaten()
        self._agent_max_length = self._agent_prev_length

        self._opp_prev_length = pb_b.get_length()
        self._opp_prev_apples = pb_b.get_apples_eaten()
        
        self._agent_traps_placed = 0

        return self._make_observation(), {}

    def step(self, action):
        self.current_step += 1
        if self.current_step >= self.max_steps:
            print(f"Game ending: reached max steps ({self.max_steps})")
            self._done = True
        
        if self._done or self._board.get_winner() is not None:
            raw_winner = self._board.get_winner()
            if raw_winner is not None:
                self._winner = raw_winner if isinstance(raw_winner, Result) else Result(raw_winner)
                print(f"Game ending: board reports winner {self._winner.name}")
                win_reason = self._board.get_win_reason() if hasattr(self._board, 'get_win_reason') else "unknown"
                print(f"Win reason: {win_reason}")
            self._done = True
            return self._null_step_return()

        step_reward = 0.0

        if self._board.is_as_turn():
            # Agent's turn
            pb_a = PlayerBoard(True, self._board)
            agent_prev_length = pb_a.get_length()
            agent_prev_apples = pb_a.get_apples_eaten()

            chosen_action = Action.TRAP if action == 8 else Action(action)
            placing_trap = (chosen_action == Action.TRAP)

            # Check validity
            if placing_trap:
                if not pb_a.is_valid_trap():
                    print(f"Invalid move detected: Agent attempted invalid trap")
                    step_reward = -10.0
                    self._board.set_winner(Result.PLAYER_B, "Agent invalid trap")
                    self._done = True
                    return self._null_step_return(step_reward)
            else:
                if not pb_a.is_valid_move(chosen_action):
                    print(f"Invalid move detected: Agent attempted invalid move {chosen_action.name}")
                    step_reward = -10.0
                    self._board.set_winner(Result.PLAYER_B, "Agent invalid move")
                    self._done = True
                    return self._null_step_return(step_reward)

            success = self._board.apply_turn([chosen_action], timer=0.05)
            if not success:
                print(f"Invalid move detected: Agent's turn failed to apply")
                step_reward = -5.0
                self._board.set_winner(Result.PLAYER_B, "Agent invalid action")
                self._done = True
                return self._null_step_return(step_reward)

            # Post-move stats
            pb_a = PlayerBoard(True, self._board)
            agent_new_length = pb_a.get_length()
            agent_new_apples = pb_a.get_apples_eaten()

            # Survival reward
            step_reward += 0.01

            # Length changes
            length_delta = agent_new_length - agent_prev_length
            if length_delta > 0:
                step_reward += 1.0 * length_delta
            elif length_delta < 0 and not placing_trap:
                step_reward += 0.2 * length_delta

            # Apple eaten
            apple_delta = agent_new_apples - agent_prev_apples
            if apple_delta > 0:
                step_reward += 2.0 * apple_delta

            # New max length
            if agent_new_length > self._agent_max_length:
                step_reward += 0.5 * (agent_new_length - self._agent_max_length)
                self._agent_max_length = agent_new_length

            # Trap placement success
            if placing_trap and success:
                self._agent_traps_placed += 1
                step_reward += 0.2

            # Update stats
            self._agent_prev_length = agent_new_length
            self._agent_prev_apples = agent_new_apples

        else:
            # Opponent's turn
            pb_b = PlayerBoard(False, self._board)
            opp_prev_length = pb_b.get_length()

            # Opponent's move
            opp_board = PlayerBoard(False, self._board.get_copy(False))
            try:
                moves = self._opponent_controller.play(opp_board, lambda: 5.0)
            except Exception as e:
                print(f"Invalid move detected: Opponent crashed with error {str(e)}")
                traceback.print_exc()
                moves = None
            
            if moves is None:
                print(f"Invalid move detected: Opponent returned None")
                self._board.set_winner(Result.PLAYER_A, "Opponent crashed/time-out")
            else:
                if isinstance(moves, Action) or isinstance(moves, int):
                    moves = [moves]
                success = self._board.apply_turn(moves, timer=0.05)
                if not success:
                    print(f"Invalid move detected: Opponent's move failed to apply: {moves}")
                    self._board.set_winner(Result.PLAYER_A, "Opponent invalid move")

            pb_b = PlayerBoard(False, self._board)
            opp_new_length = pb_b.get_length()
            length_delta = opp_new_length - opp_prev_length

            if length_delta < 0:
                step_reward += 0.3 * abs(length_delta)
                if abs(length_delta) >= 2:
                    step_reward += 0.5

            self._opp_prev_length = opp_new_length

        raw_winner = self._board.get_winner()
        if raw_winner is not None:
            self._done = True
            self._winner = raw_winner if isinstance(raw_winner, Result) else Result(raw_winner)
            print(f"Game ending after turn: board reports winner {self._winner.name}")
        elif self._done and self._winner is None:
            print(f"Game ending without a winner after {self._board.turn_count} turns")
            print(f"Agent length: {self._agent_prev_length}, Opponent length: {self._opp_prev_length}")
            print(f"Agent apples: {self._agent_prev_apples}, Opponent apples: {self._opp_prev_apples}")
            # Explicitly set as a tie
            self._winner = Result.TIE

        obs = self._make_observation()
        done = self._done
        if done:
            if self._winner == Result.PLAYER_A:
                # Win bonus
                step_reward += 5.0 + 0.01 * self.current_step
            elif self._winner == Result.PLAYER_B:
                step_reward -= 2.0
            else:
                step_reward += 1.0  # tie

        info = {
            "winner": self._winner.name if self._winner else None,
            "turn_count": self._board.turn_count,
            "agent_length": self._agent_prev_length,
            "opponent_length": self._opp_prev_length,
            "agent_max_length": self._agent_max_length,
            "traps_placed": self._agent_traps_placed
        }

        if self.render_mode == "human":
            self.render()

        return obs, step_reward, done, False, info

    def render(self):
        turn_str = f"Turn {self._board.turn_count}"
        length_str = f"Agent length: {self._agent_prev_length}, Opponent length: {self._opp_prev_length}"
        if self._winner is None:
            print(f"{turn_str}. {length_str}")
        else:
            print(f"{turn_str}. Winner={self._winner.name}. {length_str}")

    def close(self):
        pass

    def _null_step_return(self, reward_override=None):
        obs = self._make_observation()
        if reward_override is not None:
            final_reward = reward_override
        else:
            final_reward = 0.0
            if self._winner == Result.PLAYER_A:
                final_reward = 5.0 + 0.01 * self.current_step
            elif self._winner == Result.PLAYER_B:
                final_reward = -2.0
            else:
                final_reward = 1.0
        info = {
            "winner": self._winner.name if self._winner else None,
            "turn_count": self._board.turn_count,
            "agent_length": self._agent_prev_length,
            "opponent_length": self._opp_prev_length,
            "agent_max_length": self._agent_max_length,
            "traps_placed": self._agent_traps_placed
        }
        return obs, final_reward, True, False, info

    def _make_observation(self):
        """
        Build a multi-channel observation for "image" (9 channels) + action_mask.
        Channels:
        0: Walls
        1: Apples
        2: My snake head
        3: My snake body
        4: Enemy snake head
        5: Enemy snake body
        6: My traps
        7: Enemy traps
        8: Portals
        """
        if self._board is None:
            image = np.zeros((9, 64, 64), dtype=np.uint8)
        else:
            dim_x = self._board.map.dim_x
            dim_y = self._board.map.dim_y
            channels = np.zeros((9, dim_y, dim_x), dtype=np.uint8)
            pb_A = PlayerBoard(True, self._board)

            # Channel 0: Walls
            wall_mask = pb_A.get_wall_mask()
            channels[0] = np.where(wall_mask == Cell.WALL, 255, 0)

            # Channel 1: Apples
            apple_mask = pb_A.get_apple_mask()
            channels[1] = np.where(apple_mask == Cell.APPLE, 255, 0)

            # Channel 2: My snake head
            a_snake_mask = pb_A.get_snake_mask(my_snake=True, enemy_snake=False)
            channels[2] = np.where(a_snake_mask == Cell.PLAYER_HEAD, 255, 0)

            # Channel 3: My snake body
            channels[3] = np.where(a_snake_mask == Cell.PLAYER_BODY, 255, 0)

            # Channel 4: Enemy snake head
            b_snake_mask = pb_A.get_snake_mask(my_snake=False, enemy_snake=True)
            channels[4] = np.where(b_snake_mask == Cell.ENEMY_HEAD, 255, 0)

            # Channel 5: Enemy snake body
            channels[5] = np.where(b_snake_mask == Cell.ENEMY_BODY, 255, 0)

            # Channel 6: My traps
            my_trap_mask = pb_A.get_trap_mask(my_traps=True, enemy_traps=False)
            channels[6] = np.where(my_trap_mask > 0, 255, 0)

            # Channel 7: Enemy traps
            enemy_trap_mask = pb_A.get_trap_mask(my_traps=False, enemy_traps=True)
            channels[7] = np.where(enemy_trap_mask > 0, 255, 0)

            # Channel 8: Portals
            try:
                portal_mask = pb_A.get_portal_mask(descriptive=False)
                # Sometimes, even with descriptive=False, the mask might have an extra dimension.
                if portal_mask.ndim == 3:
                    portal_mask = portal_mask[:, :, 0]
                channels[8] = np.where(portal_mask == 1, 255, 0)
            except Exception as e:
                print(f"Warning: Error processing portal mask - {e}")
                channels[8] = 0

            image = np.zeros((9, 64, 64), dtype=np.uint8)
            image[:, :dim_y, :dim_x] = channels[:, :dim_y, :dim_x]

        # Build action mask (which actions are valid)
        mask = np.zeros((9,), dtype=np.uint8)
        try:
            if self._board and self._board.is_as_turn():
                pb_A = PlayerBoard(True, self._board.get_copy(False))
                valid_moves = []
                for move in range(8):  # 8 directional moves
                    action = Action(move)
                    if pb_A.is_valid_move(action):
                        valid_moves.append(action)
                for move in valid_moves:
                    mask[int(move)] = 1
                if pb_A.is_valid_trap():
                    mask[8] = 1
                if sum(mask) == 0:
                    mask[0] = 1
        except Exception as e:
            mask = np.ones((9,), dtype=np.uint8)
            print(f"Error computing action mask: {e}")

        return {"image": image, "action_mask": mask}
