import gymnasium as gym
from gymnasium import spaces
import numpy as np
import multiprocessing
import sys, os, traceback

from collections.abc import Iterable

# If needed:
# sys.path.insert(0, os.path.join(os.getcwd(), "game"))
# sys.path.insert(0, os.path.join(os.getcwd(), "workspace"))

from game.game_map import Map
from game.board import Board
from game.player_board import PlayerBoard
from game.enums import Action, Result


def run_player_process(player_name, submission_dir, player_queue, return_queue):
    """
    Minimal version of run_player_process from gameplay.py (no resource-limiting).
    """
    try:
        import importlib
        
        sys.path.append(submission_dir)
        module = importlib.import_module(player_name + ".controller")

        # Mark success
        return_queue.put(True)
    except:
        # Mark fail
        traceback.print_exc()
        return_queue.put(False)
        return

    player = None
    while True:
        func = player_queue.get()
        if func == "construct":
            temp_board, time_left = player_queue.get()
            try:
                player = module.PlayerController(lambda: time_left)
                return_queue.put((True, 0.0))
            except:
                traceback.print_exc()
                return_queue.put((False, -1))

        elif func == "bid":
            temp_board, time_left = player_queue.get()
            try:
                bid_value = player.bid(temp_board, lambda: time_left)
                return_queue.put((bid_value, 0.05))
            except:
                traceback.print_exc()
                return_queue.put((None, -1))

        elif func == "play":
            temp_board, time_left = player_queue.get()
            try:
                actions = player.play(temp_board, lambda: time_left)
                return_queue.put((actions, 0.1))
            except:
                traceback.print_exc()
                return_queue.put((None, -1))
        else:
            # unknown
            pass


class ByteFightTwoBotsEnv(gym.Env):
    """
    A Gym environment that runs two separate user-submitted bots 
    (like run_game.py does with processes). 
    The environment ends as soon as one side wins or it's a tie.
    
    On each .step(), we figure out whose turn it is, ask that bot for moves,
    and then apply those moves. If the game ends, we record the winner.
    """
    metadata = {"render_modes": ["human"]}

    def __init__(self, map_string, a_name, b_name, submission_dir, render_mode=None):
        super().__init__()
        self.map_string = map_string
        self.a_name = a_name
        self.b_name = b_name
        self.submission_dir = submission_dir
        self.render_mode = render_mode

        # Observations: 64x64 integer-coded array (dummy for demonstration)
        self.action_space = spaces.Discrete(1)  
        self.observation_space = spaces.Box(
            low=0,
            high=10,
            shape=(64,64),
            dtype=np.int32
        )

        # We'll store processes, queues, board, etc.
        self._procA = None
        self._procB = None
        self._queueA = None
        self._queueB = None
        self._returnQueueA = None
        self._returnQueueB = None

        self._board = None
        self._done = False
        self._winner = None  # Will be a Result enum or None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._done = False
        self._winner = None

        # 1) Create the Board
        map_obj = Map(self.map_string)
        self._board = Board(map_obj, time_to_play=100, build_history=False)

        # 2) Launch processes for each player's code
        self._queueA = multiprocessing.Queue()
        self._queueB = multiprocessing.Queue()
        self._returnQueueA = multiprocessing.Queue()
        self._returnQueueB = multiprocessing.Queue()

        self._procA = multiprocessing.Process(
            target=run_player_process,
            args=(self.a_name, self.submission_dir, self._queueA, self._returnQueueA)
        )
        self._procB = multiprocessing.Process(
            target=run_player_process,
            args=(self.b_name, self.submission_dir, self._queueB, self._returnQueueB)
        )
        self._procA.start()
        self._procB.start()

        # Wait for each to confirm initialization
        success_a = self._returnQueueA.get(block=True, timeout=10)
        success_b = self._returnQueueB.get(block=True, timeout=10)
        if not (success_a and success_b):
            # One or both crashed on import
            self._done = True
            self._winner = Result.ERROR
            return self._make_observation(), {}

        # 3) Construct each player's PlayerController
        self._queueA.put("construct")
        self._queueA.put((PlayerBoard(True, self._board.get_copy(False)), 5.0))
        self._queueB.put("construct")
        self._queueB.put((PlayerBoard(False, self._board.get_copy(False)), 5.0))

        # get results
        a_ok, _ = self._returnQueueA.get(block=True, timeout=10)
        b_ok, _ = self._returnQueueB.get(block=True, timeout=10)
        if not a_ok:
            self._done = True
            self._winner = Result.PLAYER_B
        if not b_ok:
            self._done = True
            self._winner = Result.PLAYER_A

        # 4) For demonstration, skip or do zero-bids if the board hasn't resolved them
        if not self._board.get_bid_resolved() and not self._done:
            bidA = 0
            bidB = 0
            if self._board.is_valid_bid(bidA) and self._board.is_valid_bid(bidB):
                self._board.resolve_bid(bidA, bidB)
            else:
                self._done = True
                self._winner = Result.ERROR

        return self._make_observation(), {}

    def step(self, action):
        """
        On each step, we do exactly one turn: whichever player's turn it is.
        The 'action' parameter is ignored because each bot is controlling itself.
        """
        # If we're done or the board says it has a winner, return a terminal step
        if self._done:
            return self._null_step_return()

        raw_winner = self._board.get_winner()
        if raw_winner is not None:
            # Convert to enum if needed
            self._done = True
            if isinstance(raw_winner, Result):
                self._winner = raw_winner
            else:
                self._winner = Result(raw_winner)
            return self._null_step_return()

        # 1) Identify whose turn it is
        is_a_turn = self._board.is_as_turn()
        queue = self._queueA if is_a_turn else self._queueB
        return_q = self._returnQueueA if is_a_turn else self._returnQueueB

        # 2) Ask that bot to "play"
        pb = PlayerBoard(is_a_turn, self._board.get_copy(False))
        queue.put("play")
        queue.put((pb, 5.0))

        # 3) Receive their moves
        moves, t = return_q.get(block=True, timeout=10)
        if moves is None:
            # They crashed or timed out
            if is_a_turn:
                self._board.set_winner(Result.PLAYER_B, "A crashed/time-out")
            else:
                self._board.set_winner(Result.PLAYER_A, "B crashed/time-out")
        else:
            # Possibly multiple moves or single
            if isinstance(moves, Action) or isinstance(moves, int):
                moves = [moves]
            success = self._board.apply_turn(moves, timer=0.05)
            if not success:
                if is_a_turn:
                    self._board.set_winner(Result.PLAYER_B, "A invalid turn")
                else:
                    self._board.set_winner(Result.PLAYER_A, "B invalid turn")

        # 4) See if the game ended
        raw_winner = self._board.get_winner()
        if raw_winner is not None:
            self._done = True
            if isinstance(raw_winner, Result):
                self._winner = raw_winner
            else:
                self._winner = Result(raw_winner)

        # Build obs, reward, etc.
        obs = self._make_observation()
        done = self._done
        reward = 0
        if done:
            # +1 if A wins, -1 if B wins, 0 otherwise
            if self._winner == Result.PLAYER_A:
                reward = +1
            elif self._winner == Result.PLAYER_B:
                reward = -1

        info = {
            "winner": self._winner.name if self._winner is not None else None,
            "turn_count": self._board.turn_count
        }

        return obs, reward, done, False, info

    def render(self):
        """
        Print the current turn and the environment's stored winner (self._winner).
        This means we never see a raw integer from board.get_winner().
        """
        turn_str = f"Turn {self._board.turn_count}"
        if self._winner is None:
            print(f"{turn_str}. Winner=None")
        else:
            print(f"{turn_str}. Winner={self._winner.name}")  
        # If you prefer printing the entire enum, do `{self._winner}`

    def close(self):
        if self._procA is not None:
            self._procA.terminate()
        if self._procB is not None:
            self._procB.terminate()

    def _null_step_return(self):
        """
        Returns a terminal-step tuple if the environment is done.
        """
        obs = np.zeros((64,64), dtype=np.int32)
        reward = 0
        if self._winner == Result.PLAYER_A:
            reward = +1
        elif self._winner == Result.PLAYER_B:
            reward = -1

        done = True
        truncated = False
        info = {
            "winner": self._winner.name if self._winner is not None else None,
            "turn_count": self._board.turn_count
        }
        if self.render_mode == "human":
            self.render()
        return obs, reward, done, truncated, info

    def _make_observation(self):
        """
        Build a 64x64 integer-coded array from the board.
        For brevity, we just return zeros here, but you can
        encode walls, apples, traps, snake positions, etc.
        """
        if self._board is None:
            return np.zeros((64,64), dtype=np.int32)
        dim_x = self._board.map.dim_x
        dim_y = self._board.map.dim_y
        arr = np.zeros((dim_y, dim_x), dtype=np.int32)
        # If you want to do more advanced logic to mark walls, apples, etc., do it here.
        full_obs = np.zeros((64,64), dtype=np.int32)
        full_obs[:dim_y, :dim_x] = arr
        return full_obs
