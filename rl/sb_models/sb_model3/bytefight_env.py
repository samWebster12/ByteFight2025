import gymnasium as gym
from gymnasium import spaces
import numpy as np
import sys, os, traceback, importlib
import random
from collections.abc import Callable
import time


parent_dir = os.path.abspath(os.path.join(__file__, "../../../.."))
sys.path.insert(0, parent_dir)

from game.game_map import Map
from game.board import Board
from game.player_board import PlayerBoard
from game.enums import Action, Result, Cell

class ByteFightEnv(gym.Env):
    """
    A comprehensive Gymnasium environment for ByteFight: Snake that accurately reflects
    all aspects of the actual game including:
    - Multi-move turns with proper length sacrifice mechanics
    - Length queuing system for apples
    - Bidding system for first turn
    - Decay mechanism after turn 1000
    - Portal mechanics
    - Time management

    Reward shaping:
    - Reward for increasing snake length
    - Reward for eating apples
    - Reward for opponent length decrease
    - Reward for opponent hitting your traps
    - Penalty for decreasing your own length through multiple moves
    - Heavy penalty for invalid moves
    - Win/loss rewards at episode end
    """

    metadata = {"render_modes": ["human", "ansi"]}

    def __init__(
        self,
        map_string,
        opponent_module,
        submission_dir,
        max_steps=2000,
        render_mode=None,
        time_limit=110,
        bid_time=5,
        init_time=5
    ):
        super().__init__()
        self.map_string = map_string
        self.opponent_module = opponent_module
        self.submission_dir = submission_dir
        self.render_mode = render_mode
        self.max_steps = max_steps
        self.current_step = 0
        self.time_limit = time_limit
        self.bid_time = bid_time
        self.init_time = init_time
        
        # Agent times (chess clock style)
        self.agent_time_left = time_limit
        self.opponent_time_left = time_limit

        # Define action space - multi-move capability
        # Each action is a sequence of up to 3 moves
        # First element is the number of moves (1-3)
        # Next three elements are the moves themselves (valid values 0-8)
        # Where 0-7 are directions and 8 is TRAP
        self.action_space = spaces.MultiDiscrete([3, 9, 9, 9])

        # Observation is a dict with multiple components
        self.observation_space = spaces.Dict({
            # 9 channels for board state
            "image": spaces.Box(low=0, high=255, shape=(9, 64, 64), dtype=np.uint8),
            # Valid action mask for each possible move
            "action_mask": spaces.Box(low=0, high=1, shape=(9,), dtype=np.uint8),
            # Game state info
            "turn_count": spaces.Box(low=0, high=2000, shape=(1,), dtype=np.int32),
            "my_length": spaces.Box(low=0, high=1000, shape=(1,), dtype=np.int32),
            "my_queued_length": spaces.Box(low=0, high=10, shape=(1,), dtype=np.int32),
            "opponent_length": spaces.Box(low=0, high=1000, shape=(1,), dtype=np.int32),
            "opponent_queued_length": spaces.Box(low=0, high=10, shape=(1,), dtype=np.int32),
            "max_traps_allowed": spaces.Box(low=0, high=500, shape=(1,), dtype=np.int32),
            "time_left": spaces.Box(low=0, high=self.time_limit, shape=(1,), dtype=np.float32),
            "opponent_time_left": spaces.Box(low=0, high=self.time_limit, shape=(1,), dtype=np.float32),
            "is_decaying": spaces.Box(low=0, high=1, shape=(1,), dtype=np.uint8),
            "decay_rate": spaces.Box(low=0, high=15, shape=(1,), dtype=np.int32)
        })

        # Attempt to load opponent
        sys.path.append(self.submission_dir)
        try:
            opp_mod = importlib.import_module(self.opponent_module + ".controller")
            self._opponent_controller = opp_mod.PlayerController(self._opponent_time_left_func)
        except Exception as e:
            traceback.print_exc()
            raise ImportError(f"Could not load opponent module {self.opponent_module}") from e

        self._board = None
        self._done = False
        self._winner = None
        self._bid_phase = True

        # Track agent/opponent stats for shaping
        self._agent_prev_length = 0
        self._agent_prev_apples = 0
        self._agent_max_length = 0
        self._opp_prev_length = 0
        self._opp_prev_apples = 0
        self._agent_traps_placed = 0
        self._opp_traps_hit = 0
        
        # For managing decay
        self._is_decaying = False
        self._decay_rate = 0

    def _agent_time_left_func(self):
        return self.agent_time_left
    
    def _opponent_time_left_func(self):
        return self.opponent_time_left

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._done = False
        self._winner = None
        self.current_step = 0
        self._bid_phase = True
        
        # Reset time
        self.agent_time_left = self.time_limit
        self.opponent_time_left = self.time_limit
        
        # Reset decay state
        self._is_decaying = False
        self._decay_rate = 0

        # Create new board
        map_obj = Map(self.map_string)
        self._board = Board(map_obj, time_to_play=self.time_limit, build_history=False)

        # Initialize agent/opponent stats
        pb_a = PlayerBoard(True, self._board)
        pb_b = PlayerBoard(False, self._board)

        self._agent_prev_length = pb_a.get_length()
        self._agent_prev_apples = pb_a.get_apples_eaten()
        self._agent_max_length = self._agent_prev_length
        self._agent_queued_length = 0

        self._opp_prev_length = pb_b.get_length()
        self._opp_prev_apples = pb_b.get_apples_eaten()
        self._opp_queued_length = 0
        
        self._agent_traps_placed = 0
        self._opp_traps_hit = 0

        return self._make_observation(), {}

    def step(self, action):
        self.current_step += 1
        if self.current_step >= self.max_steps:
            self._done = True
        
        # If already done, return immediately
        if self._done or self._board.get_winner() is not None:
            raw_winner = self._board.get_winner()
            if raw_winner is not None:
                self._winner = raw_winner if isinstance(raw_winner, Result) else Result(raw_winner)
            self._done = True
            return self._null_step_return()

        # Initialize reward
        step_reward = 0.0
        
        # Handle bidding phase
        if self._bid_phase:
            pb_a = PlayerBoard(True, self._board)
            
            # Agent's bid is the first element of the action
            agent_bid = int(action[0])  # We'll use first action element as bid
            
            # Track time for agent's bid
            start_time = time.time()
            agent_bid_time = min(0.05, self.bid_time)  # Simulate small time usage
            self.agent_time_left -= agent_bid_time
            
            # Get opponent's bid
            opp_board = PlayerBoard(False, self._board.get_copy(False))
            try:
                opponent_start_time = time.time()
                opponent_bid = self._opponent_controller.bid(opp_board, self._opponent_time_left_func)
                opponent_bid_time = time.time() - opponent_start_time
                opponent_bid_time = min(opponent_bid_time, self.bid_time)
                self.opponent_time_left -= opponent_bid_time
            except Exception as e:
                traceback.print_exc()
                opponent_bid = 0
                self._board.set_winner(Result.PLAYER_A, "Opponent crashed during bid")
                self._done = True
                return self._null_step_return(1.0)  # Win by default
            
            # Check bid validity
            if not self._board.is_valid_bid(agent_bid):
                step_reward = -2.0
                self._board.set_winner(Result.PLAYER_B, "Agent invalid bid")
                self._done = True
                return self._null_step_return(step_reward)
                
            if not self._board.is_valid_bid(opponent_bid):
                step_reward = 2.0
                self._board.set_winner(Result.PLAYER_A, "Opponent invalid bid")
                self._done = True
                return self._null_step_return(step_reward)
            
            # Resolve bid
            self._board.resolve_bid(agent_bid, opponent_bid)
            self._bid_phase = False
            
            # Small reward for winning bid, small penalty for losing
            if self._board.is_as_turn():
                step_reward += 0.5  # Won the bid
            else:
                step_reward -= 0.1  # Lost the bid
                
            # If you bid much less than opponent and still won, bigger reward
            if self._board.is_as_turn() and agent_bid < opponent_bid - 1:
                step_reward += 0.5
                
            # Skip to next step if we're not the first player
            if not self._board.is_as_turn():
                return self._process_opponent_turn()
                
        # Determine if it's time for decay to activate
        if self._board.turn_count >= 1000 and not self._is_decaying:
            self._is_decaying = True
            self._decay_rate = 15  # Decay every 15 turns initially
        
        # Update decay rate based on turn count
        if self._is_decaying:
            if self._board.turn_count >= 1950:
                self._decay_rate = 1  # Every turn
            elif self._board.turn_count >= 1800:
                self._decay_rate = 5  # Every 5 turns
            elif self._board.turn_count >= 1600:
                self._decay_rate = 10  # Every 10 turns
                
        # Apply decay if necessary
        if self._is_decaying and self._board.turn_count % self._decay_rate == 0:
            # Decay is applied at the beginning of a player's turn
            if self._board.is_as_turn():
                # Manually apply decay to agent - decrease length by 1
                pb_a = PlayerBoard(True, self._board)
                curr_length = pb_a.get_length()
                if curr_length > 2:  # Can't go below minimum size
                    self._board.apply_decay_to_a()
                    step_reward -= 0.2  # Small penalty for decay
            
        # Process agent's turn if it's our turn
        if self._board.is_as_turn():
            return self._process_agent_turn(action)
        else:
            return self._process_opponent_turn()

    def _process_agent_turn(self, action):
        # Get pre-move stats
        pb_a = PlayerBoard(True, self._board)
        agent_prev_length = pb_a.get_length()
        agent_prev_apples = pb_a.get_apples_eaten()
        agent_prev_max_length = self._agent_max_length
        
        # Parse action
        num_moves = action[0] + 1  # 0-2 + 1 = 1-3 moves
        moves = []
        for i in range(num_moves):
            move_idx = action[i+1]
            if move_idx < 8:  # Directional move
                moves.append(Action(move_idx))
            elif move_idx == 8 and pb_a.is_valid_trap():  # Trap placement
                moves.append(Action.TRAP)
        
        # Start timing
        start_time = time.time()
        
        # Validate and execute moves
        invalid_move = False
        if not moves:
            invalid_move = True
        else:
            # Check first move validity
            first_move = moves[0]
            if first_move == Action.TRAP:
                if not pb_a.is_valid_trap():
                    invalid_move = True
            elif not pb_a.is_valid_move(first_move):
                invalid_move = True
                
        if invalid_move:
            step_reward = -10.0  # Heavy penalty for invalid move
            self._board.set_winner(Result.PLAYER_B, "Agent invalid move")
            self._done = True
            
            # Update time
            move_time = time.time() - start_time
            self.agent_time_left -= move_time
            
            return self._null_step_return(step_reward)
        
        # Apply moves - the game engine will validate subsequent moves
        success = self._board.apply_turn(moves, timer=0.05)
        
        # Update time used
        move_time = time.time() - start_time
        self.agent_time_left -= move_time
        
        # Check for time expiration
        if self.agent_time_left <= 0:
            self._board.set_winner(Result.PLAYER_B, "Agent timeout")
            self._done = True
            return self._null_step_return(-5.0)  # Heavy penalty for timeout
            
        # Check move success
        if not success:
            step_reward = -5.0  # Penalty for invalid move sequence
            self._board.set_winner(Result.PLAYER_B, "Agent invalid move sequence")
            self._done = True
            return self._null_step_return(step_reward)
            
        # Get post-move stats
        pb_a = PlayerBoard(True, self._board)
        agent_new_length = pb_a.get_length()
        agent_new_apples = pb_a.get_apples_eaten()
        
        # Calculate reward
        
        # Base survival reward
        step_reward = 0.01
        
        # Length changes
        length_delta = agent_new_length - agent_prev_length
        if length_delta > 0:
            # Length increase is good
            step_reward += 0.5 * length_delta
        elif length_delta < 0 and Action.TRAP not in moves:
            # Length decrease from multiple moves (not from trap placement)
            # Small penalty but not too harsh as this is strategic
            step_reward += 0.1 * length_delta
            
        # Apple consumption reward
        apple_delta = agent_new_apples - agent_prev_apples
        if apple_delta > 0:
            step_reward += 1.0 * apple_delta
            
        # Track trap placement
        for move in moves:
            if move == Action.TRAP:
                self._agent_traps_placed += 1
                step_reward += 0.2  # Small reward for placing trap
                
        # New max length achievement
        if agent_new_length > agent_prev_max_length:
            step_reward += 0.3 * (agent_new_length - agent_prev_max_length)
            self._agent_max_length = agent_new_length
            
        # Multi-move strategy reward - if we made multiple moves strategically
        if len(moves) > 1 and agent_new_length >= agent_prev_length:
            step_reward += 0.1 * len(moves)  # Small bonus for strategic multi-moves
            
        # Update stats
        self._agent_prev_length = agent_new_length
        self._agent_prev_apples = agent_new_apples
        
        # Get observation and check for end of game
        obs = self._make_observation()
        
        # Check for winner
        raw_winner = self._board.get_winner()
        if raw_winner is not None:
            self._done = True
            self._winner = raw_winner if isinstance(raw_winner, Result) else Result(raw_winner)
            
        # Process opponent's turn next if game is still ongoing
        if not self._done:
            return self._process_opponent_turn()
            
        # Final reward if game is over
        if self._done:
            if self._winner == Result.PLAYER_A:
                step_reward += 10.0  # Big win bonus
            elif self._winner == Result.PLAYER_B:
                step_reward -= 5.0  # Loss penalty
            else:
                step_reward += 1.0  # Tie
                
        info = {
            "winner": self._winner.name if self._winner else None,
            "turn_count": self._board.turn_count,
            "agent_length": self._agent_prev_length,
            "opponent_length": self._opp_prev_length,
            "agent_max_length": self._agent_max_length,
            "traps_placed": self._agent_traps_placed,
            "time_left": self.agent_time_left,
            "opponent_time_left": self.opponent_time_left,
            "is_decaying": int(self._is_decaying),
            "decay_rate": self._decay_rate
        }
        
        if self.render_mode == "human" or self.render_mode == "ansi":
            self.render()
            
        return obs, step_reward, self._done, False, info

    def _process_opponent_turn(self):
        # Get pre-move stats
        pb_b = PlayerBoard(False, self._board)
        opp_prev_length = pb_b.get_length()
        opp_prev_apples = pb_b.get_apples_eaten()
        
        # Initialize reward
        step_reward = 0.0
        
        # Apply decay if necessary for opponent
        if self._is_decaying and self._board.turn_count % self._decay_rate == 0:
            if not self._board.is_as_turn():
                # Apply decay to opponent
                if opp_prev_length > 2:
                    self._board.apply_decay_to_b()
                    step_reward += 0.05  # Tiny reward for opponent decay
        
        # Get opponent's move
        opp_board = PlayerBoard(False, self._board.get_copy(False))
        try:
            opponent_start_time = time.time()
            moves = self._opponent_controller.play(opp_board, self._opponent_time_left_func)
            opponent_move_time = time.time() - opponent_start_time
            self.opponent_time_left -= opponent_move_time
            
            # Check for timeout
            if self.opponent_time_left <= 0:
                self._board.set_winner(Result.PLAYER_A, "Opponent timeout")
                step_reward += 5.0  # Win by opponent timeout
                self._done = True
                return self._null_step_return(step_reward)
                
        except Exception as e:
            traceback.print_exc()
            moves = None
            self._board.set_winner(Result.PLAYER_A, "Opponent crashed")
            step_reward += 5.0  # Win by opponent error
            self._done = True
            return self._null_step_return(step_reward)
            
        # Apply opponent moves
        if moves is None:
            self._board.set_winner(Result.PLAYER_A, "Opponent returned None")
            step_reward += 5.0  # Win by opponent error
            self._done = True
        else:
            # Convert to list if single move
            if isinstance(moves, Action) or isinstance(moves, int):
                moves = [moves]
                
            # Check for forfeit
            if Action.FF in moves:
                self._board.set_winner(Result.PLAYER_A, "Opponent forfeit")
                step_reward += 5.0  # Win by forfeit
                self._done = True
                return self._null_step_return(step_reward)
                
            # Apply moves
            success = self._board.apply_turn(moves, timer=opponent_move_time)
            
            if not success:
                self._board.set_winner(Result.PLAYER_A, "Opponent invalid move")
                step_reward += 5.0  # Win by opponent error
                self._done = True
                return self._null_step_return(step_reward)
                
        # Get post-move stats
        pb_b = PlayerBoard(False, self._board)
        opp_new_length = pb_b.get_length()
        opp_new_apples = pb_b.get_apples_eaten()
        
        # Calculate rewards based on opponent's move outcome
        
        # Length changes - if opponent lost length, small reward for us
        length_delta = opp_new_length - opp_prev_length
        if length_delta < 0:
            step_reward += 0.2 * abs(length_delta)
            # Big loss likely means they hit a trap
            if abs(length_delta) >= 2:
                step_reward += 0.5
                self._opp_traps_hit += 1
                
        # Update opponent stats
        self._opp_prev_length = opp_new_length
        self._opp_prev_apples = opp_new_apples
        
        # Check for winner
        raw_winner = self._board.get_winner()
        if raw_winner is not None:
            self._done = True
            self._winner = raw_winner if isinstance(raw_winner, Result) else Result(raw_winner)
            
        # Prepare observation
        obs = self._make_observation()
        
        # Final reward adjustments if game is done
        if self._done:
            if self._winner == Result.PLAYER_A:
                step_reward += 10.0  # Big win bonus
            elif self._winner == Result.PLAYER_B:
                step_reward -= 5.0  # Loss penalty
            else:  # Tie
                step_reward += 1.0
                
        info = {
            "winner": self._winner.name if self._winner else None,
            "turn_count": self._board.turn_count,
            "agent_length": self._agent_prev_length,
            "opponent_length": self._opp_prev_length,
            "agent_max_length": self._agent_max_length,
            "traps_placed": self._agent_traps_placed,
            "opponent_traps_hit": self._opp_traps_hit,
            "time_left": self.agent_time_left,
            "opponent_time_left": self.opponent_time_left,
            "is_decaying": int(self._is_decaying),
            "decay_rate": self._decay_rate
        }
        
        if self.render_mode == "human" or self.render_mode == "ansi":
            self.render()
            
        return obs, step_reward, self._done, False, info

    def render(self):
        """Render the game state"""
        if self.render_mode is None:
            return
            
        turn_str = f"Turn {self._board.turn_count}"
        length_str = f"Agent: {self._agent_prev_length} | Opponent: {self._opp_prev_length}"
        time_str = f"Time - Agent: {self.agent_time_left:.1f}s | Opponent: {self.opponent_time_left:.1f}s"
        decay_str = f"Decay: {'ON' if self._is_decaying else 'OFF'} Rate: {self._decay_rate}"
        
        if self._winner is None:
            status = f"{turn_str}. {length_str}. {time_str}. {decay_str}"
        else:
            status = f"{turn_str}. Winner={self._winner.name}. {length_str}. {time_str}"
            
        if self.render_mode == "human" or self.render_mode == "ansi":
            print(status)
            
            # Additionally print the board state if in ansi mode
            if self.render_mode == "ansi" and self._board is not None:
                player_map, _, _, _, _ = self._board.get_board_string()
                print(player_map)

    def close(self):
        """Clean up resources"""
        pass
        
    @staticmethod
    def create_training_env(map_string, opponent_module, submission_dir, **kwargs):
        """
        Factory method to create a properly configured environment for training.
        
        Args:
            map_string: String representation of the game map
            opponent_module: Name of the opponent module
            submission_dir: Directory containing the opponent module
            **kwargs: Additional arguments for the environment
            
        Returns:
            A wrapped environment ready for training with stable-baselines3
        """
        from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor
        from stable_baselines3.common.monitor import Monitor
        
        def make_env():
            env = ByteFightEnv(
                map_string=map_string,
                opponent_module=opponent_module,
                submission_dir=submission_dir,
                **kwargs
            )
            return Monitor(env)
            
        env = DummyVecEnv([make_env])
        env = VecMonitor(env)
        
        return env

    def _null_step_return(self, reward_override=None):
        """Return for when the episode is already done"""
        obs = self._make_observation()
        
        if reward_override is not None:
            final_reward = reward_override
        else:
            final_reward = 0.0
            if self._winner == Result.PLAYER_A:
                final_reward = 10.0  # Win
            elif self._winner == Result.PLAYER_B:
                final_reward = -5.0  # Loss
            else:
                final_reward = 1.0  # Tie
                
        info = {
            "winner": self._winner.name if self._winner else None,
            "turn_count": self._board.turn_count,
            "agent_length": self._agent_prev_length,
            "opponent_length": self._opp_prev_length,
            "agent_max_length": self._agent_max_length,
            "traps_placed": self._agent_traps_placed,
            "time_left": self.agent_time_left,
            "opponent_time_left": self.opponent_time_left,
            "is_decaying": int(self._is_decaying),
            "decay_rate": self._decay_rate
        }
        
        return obs, final_reward, True, False, info

    def _make_observation(self):
        """
        Build a complete observation with:
        - Multi-channel board representation
        - Action mask
        - Game state information
        """
        # Initialize empty observation if board is None
        if self._board is None:
            image = np.zeros((9, 64, 64), dtype=np.uint8)
            mask = np.zeros((9,), dtype=np.uint8)
            return {
                "image": image,
                "action_mask": mask,
                "turn_count": np.array([0], dtype=np.int32),
                "my_length": np.array([0], dtype=np.int32),
                "my_queued_length": np.array([0], dtype=np.int32),
                "opponent_length": np.array([0], dtype=np.int32),
                "opponent_queued_length": np.array([0], dtype=np.int32),
                "max_traps_allowed": np.array([0], dtype=np.int32),
                "time_left": np.array([self.agent_time_left], dtype=np.float32),
                "opponent_time_left": np.array([self.opponent_time_left], dtype=np.float32),
                "is_decaying": np.array([0], dtype=np.uint8),
                "decay_rate": np.array([0], dtype=np.int32)
            }
            
        # Build image channels from board state
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
            # Handle potential extra dimension
            if portal_mask.ndim == 3:
                portal_mask = portal_mask[:, :, 0]
            channels[8] = np.where(portal_mask == 1, 255, 0)
        except Exception as e:
            print(f"Warning: Error processing portal mask - {e}")
            channels[8] = 0
            
        # Pad to 64x64 if necessary
        image = np.zeros((9, 64, 64), dtype=np.uint8)
        image[:, :dim_y, :dim_x] = channels[:, :dim_y, :dim_x]
        
        # Build action mask (which actions are valid)
        mask = np.zeros((9,), dtype=np.uint8)
        try:
            if self._board and (self._board.is_as_turn() or self._bid_phase):
                if self._bid_phase:
                    # All bid values are valid
                    mask = np.ones((9,), dtype=np.uint8)
        except Exception as e:
            print(f"Error computing action mask: {e}")
            mask = np.ones((9,), dtype=np.uint8)
            
        # Get game state info
        pb_a = PlayerBoard(True, self._board)
        pb_b = PlayerBoard(False, self._board)
        
        # Calculate queued length
        my_len = pb_a.get_length()
        my_queued = pb_a.get_queued_length() if hasattr(pb_a, 'get_queued_length') else 0
        
        opp_len = pb_b.get_length()
        opp_queued = pb_b.get_queued_length() if hasattr(pb_b, 'get_queued_length') else 0
        
        # Calculate max traps allowed (half of max achieved length, rounded down)
        max_traps = max(0, self._agent_max_length // 2)
        
        # Complete observation dictionary
        return {
            "image": image,
            "action_mask": mask,
            "turn_count": np.array([self._board.turn_count], dtype=np.int32),
            "my_length": np.array([my_len], dtype=np.int32),
            "my_queued_length": np.array([my_queued], dtype=np.int32),
            "opponent_length": np.array([opp_len], dtype=np.int32),
            "opponent_queued_length": np.array([opp_queued], dtype=np.int32),
            "max_traps_allowed": np.array([max_traps], dtype=np.int32),
            "time_left": np.array([self.agent_time_left], dtype=np.float32),
            "opponent_time_left": np.array([self.opponent_time_left], dtype=np.float32),
            "is_decaying": np.array([int(self._is_decaying)], dtype=np.uint8),
            "decay_rate": np.array([self._decay_rate], dtype=np.int32)
        }