import numpy as np
import torch
from game import player_board
from game.enums import Action, Cell
from normalizer import RunningNormalizer
from game.player_board import PlayerBoard, Board

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

class SelfPlayOppController:
    def __init__(self, policy, obs_normalizer: RunningNormalizer):
        self.policy = policy
        self.obs_normalizer = obs_normalizer

    def bid(self, board: player_board.PlayerBoard, time_left: callable):
        return 0

    def play(self, board: player_board.PlayerBoard, time_left: callable, current_actions=None):
        """
        Called each time the environment wants the opponent's entire turn.
        We'll catch any exceptions to debug potential crashes.
        """
        from game.enums import Action as ByteAction

        try:
            actions = self._play_impl(board, time_left, current_actions)
            return actions
        except Exception as e:
            # Log the error, optionally with traceback
            import traceback
            print("[SELF-PLAY OPP] Exception in 'play':", e)
            traceback.print_exc()
            # Return a forfeit action so the environment knows we crashed
            return [ByteAction.FF]

    def _play_impl(self, board: player_board.PlayerBoard, time_left: callable, current_actions=None):
        """
        The real logic of the 'play' method. If an exception is raised, 
        we catch it in 'play()' so we can log it and return forfeit.
        """
        #print("WITHIN SELF PLAY CONTROLLER: PlayerBoard B head: ", board.get_head_location())
        if current_actions is None:
            current_actions = []
        
        local_discrete_actions = []
        board_instance = board.game_board
        forecast_board = board_instance.get_copy()
        #print("WITHIN SELF PLAY CONTROLLER: PlayerBoard B head: ", forecast_board.snake_b.get_head_loc())
        i = 0
        while True:
            #print("IS A's Turn: ", forecast_board.is_as_turn())
            # Make sure it's the opponent's turn
            if forecast_board.is_as_turn():
                forecast_board.next_turn()

            obs_dict = self._build_observation(board, local_discrete_actions, forecast_board)
            obs_batch = {k: np.expand_dims(v, axis=0) for k, v in obs_dict.items()}
            discrete_action, _ = self.policy.predict(obs_batch, deterministic=False)
            discrete_action = int(discrete_action)
            
            # If we got END_TURN, add to local_discrete_actions and break
            if discrete_action == 9:
                #local_discrete_actions.append(discrete_action)
                break
            
            #print(f"head location {i}: ", forecast_board.snake_b.get_head_loc())
            i += 1
            # Attempt forecast
            if not self._forecast_single_action(discrete_action, local_discrete_actions, forecast_board):
                from game.enums import Action as ByteAction
                # Return immediate forfeit on invalid
                return [ByteAction.FF]

            # Accumulate discrete action
            local_discrete_actions.append(discrete_action)

        # Convert discrete to ByteFight
        final_actions = [self._discrete_to_bytefight(a) for a in local_discrete_actions]
        return final_actions

    def _forecast_single_action(self, discrete_action, discrete_history, forecast_board: Board):
        if discrete_action < 0 or discrete_action > 9:
            return False

        from game.enums import Action as ByteAction
        partial_actions = [self._discrete_to_bytefight(a) for a in discrete_history]
        partial_actions.append(self._discrete_to_bytefight(discrete_action))

        tmp_board = forecast_board.get_copy()
        success_board, valid = tmp_board.forecast_turn(partial_actions)
        #print("Opp Attempted move: ", discrete_action, "\t\tForecast is valid: ", valid)
        return valid

    def _build_observation(self, pb: player_board.PlayerBoard, discrete_history, forecast_board):
        """
        Builds an observation dict (board_image, features, action_mask)
        from the perspective of the opponent. The length of discrete_history
        helps compute move_count, current_sacrifice, etc.
        """
        board_instance = pb.game_board
        pb_opp = player_board.PlayerBoard(False, board_instance)

        # Board dims
        max_height, max_width = 64, 64

        # 1) Channel masks
        wall_mask = pb_opp.get_wall_mask().astype(np.float32)
        apple_mask = pb_opp.get_apple_mask().astype(np.float32)
        my_snake_mask = pb_opp.get_snake_mask(my_snake=True, enemy_snake=False)
        opp_snake_mask = pb_opp.get_snake_mask(my_snake=False, enemy_snake=True)
        my_head_mask = (my_snake_mask == Cell.PLAYER_HEAD).astype(np.float32)
        opp_head_mask = (opp_snake_mask == Cell.ENEMY_HEAD).astype(np.float32)
        my_body_mask = np.logical_and(my_snake_mask > 0, my_snake_mask != Cell.PLAYER_HEAD).astype(np.float32)
        opp_body_mask = np.logical_and(opp_snake_mask > 0, opp_snake_mask != Cell.ENEMY_HEAD).astype(np.float32)
        my_trap_mask = (pb_opp.get_trap_mask(my_traps=True, enemy_traps=False) > 0).astype(np.float32)
        opp_trap_mask_raw = pb_opp.get_trap_mask(my_traps=False, enemy_traps=True)
        opp_trap_mask = (opp_trap_mask_raw < 0).astype(np.float32)
        portal_mask_3d = pb_opp.get_portal_mask(descriptive=False)
        if portal_mask_3d.ndim == 3 and portal_mask_3d.shape[-1] == 2:
            portal_mask = np.all(portal_mask_3d >= 0, axis=-1).astype(np.float32)
        else:
            portal_mask = portal_mask_3d.astype(np.float32)

        def _pad_or_crop(arr):
            h, w = arr.shape
            padded = np.zeros((64, 64), dtype=np.float32)
            h_to_copy = min(h, 64)
            w_to_copy = min(w, 64)
            padded[:h_to_copy, :w_to_copy] = arr[:h_to_copy, :w_to_copy]
            return padded

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
        channels = [_pad_or_crop(ch) for ch in channels]
        board_image = np.stack(channels, axis=0).astype(np.float32)

        # 2) Scalar features
        turn_count = board_instance.turn_count
        move_count = len(discrete_history) + 1
        # current_sacrifice: each non-TRAP after first => +2
        if move_count > 1:
            non_trap_moves = [act for act in discrete_history if act != 8]
            current_sacrifice = len(non_trap_moves) * 2
        else:
            current_sacrifice = 0

        my_heading = pb_opp.get_direction()
        my_heading_val = 0 if my_heading is None else int(my_heading.value)
        opp_heading_val = 0
        my_length = pb_opp.get_length(enemy=True)
        opp_length = pb_opp.get_length(enemy=False)
        my_queued = pb_opp.get_queued_length(enemy=True)
        opp_queued = pb_opp.get_queued_length(enemy=False)
        my_apples = pb_opp.get_apples_eaten(enemy=True)
        opp_apples = pb_opp.get_apples_eaten(enemy=False)
        my_max_len = pb_opp.get_max_length(enemy=True)
        opp_max_len = pb_opp.get_max_length(enemy=False)
        is_decaying = float(pb_opp.currently_decaying())
        decay_interval = pb_opp.get_current_decay_interval() or 0

        scalars = np.array([
            float(move_count),
            float(turn_count),
            float(current_sacrifice),
            float(my_heading_val),
            float(opp_heading_val),
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

        # Opponent doesn't update stats: update_stats=False
        norm_scalars = self.obs_normalizer.normalize(scalars, update_stats=False)

        # 3) Action mask
        action_mask = self._get_valid_action_mask(forecast_board, discrete_history)

        return {
            "board_image": board_image,
            "features": norm_scalars,
            "action_mask": action_mask
        }
    
    def _get_valid_action_mask(self, forecast_board, local_discrete_actions) -> np.ndarray:
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

        pb = PlayerBoard(False, forecast_board)

        # If we haven't moved yet this turn, we can't do "END_TURN" or "TRAP" if length <= 2
        if len(local_discrete_actions) == 0:
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
            test_actions = local_discrete_actions.copy()
            test_actions.append(ACTION_LOOKUP[act_id])

            # forecast
            test_board = forecast_board.get_copy()
            success_board, valid = test_board.forecast_turn(test_actions)
            if not valid:
                mask[act_id] = 0

        return mask

    def _discrete_to_bytefight(self, discrete_action):
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
            9: "END_TURN"
        }
        return ACTION_LOOKUP.get(int(discrete_action), Action.FF)
