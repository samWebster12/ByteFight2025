import os
import json
import numpy as np
import time
import argparse
from typing import Dict, List, Any, Tuple, Optional

import sys
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, parent_dir)

parent_dir = os.path.abspath(os.path.join(__file__, "../../../.."))
sys.path.insert(0, parent_dir)

# Import game components
from game.enums import Action, Result, Cell
from game.game_map import Map
from game.player_board import PlayerBoard, Board
from opp_controller import OppController
from bytefight_env import ByteFightSnakeEnv, map_discrete_to_bytefight, get_heading_value, AGENT_ACTIONS

class ConfigurableOpponent(OppController):
    """An opponent controller that can be configured for testing specific scenarios."""
    
    def __init__(self, bid_value=0, actions=None, move_delay=0.0):
        """
        Initialize the configurable opponent.
        
        Args:
            bid_value: Bid value to return for the initial bidding phase
            actions: List of actions to take sequentially (or single action to repeat)
            move_delay: Time in seconds to delay before returning a move (testing time limit)
        """
        self.bid_value = bid_value
        self.actions = actions or [Action.NORTH]  # Default to always moving north
        self.action_idx = 0
        self.move_delay = move_delay
    
    def bid(self, player_board, time_left):
        """Return the predetermined bid value."""
        return self.bid_value
    
    def play(self, player_board, time_left):
        """Return the predetermined actions or a default action if none specified."""
        if self.move_delay > 0:
            time.sleep(self.move_delay)
            
        if isinstance(self.actions, list):
            if not self.actions or self.action_idx >= len(self.actions):
                # Default to a safe action if we've exhausted our list
                return Action.NORTH
            
            action = self.actions[self.action_idx]
            self.action_idx += 1
            return action
        else:
            # If actions is a single action, always return that
            return self.actions

class ByteFightTester:
    """Comprehensive test suite for the ByteFight environment."""
    
    def __init__(self, verbose=True):
        self.verbose = verbose
        self.tests = {}
        self.passed = 0
        self.failed = 0
        self.failed_tests = []
        
    def log(self, message):
        """Log a message if verbose mode is enabled."""
        if self.verbose:
            print(message)
            
    def assert_test(self, test_name, condition, message=""):
        """Assert a test condition and record the result."""
        self.tests[test_name] = condition
        
        if condition:
            self.passed += 1
            status = "PASSED"
        else:
            self.failed += 1
            self.failed_tests.append(test_name)
            status = "FAILED"
            
        self.log(f"[{status}] {test_name}: {message}")
        return condition
    
    def print_summary(self):
        """Print a summary of all test results."""
        print("\n===== TEST SUMMARY =====")
        if self.passed + self.failed > 0:
            print(f"Passed: {self.passed}/{self.passed + self.failed} ({self.passed/(self.passed + self.failed)*100:.1f}%)")
        else:
            print("No tests were run!")
            
        if self.failed > 0:
            print("\nFailed tests:")
            for test_name in self.failed_tests:
                print(f"- {test_name}")
        print("")
        
    def load_map_from_json(self, map_name, json_path="maps.json"):
        """Load a map from a JSON file."""
        if os.path.exists(json_path):
            with open(json_path, "r") as f:
                maps_dict = json.load(f)
            if map_name in maps_dict:
                map_string = maps_dict[map_name]
                return Map(map_string)
        
        # Use hardcoded map data if file not found
        maps_data = {
            "pillars": "17,17#2,8#14,8#5#2#2,2,14,2_14,2,2,2_2,14,14,14_14,14,2,14#100,7,Vertical#0000000000000000001010101010101010000000000000000000101010101010101000000000000000000010101010101010100000000000000000001010101010101010000000000000000000101010101010101000000000000000000010101010101010100000000000000000001010101010101010000000000000000000101010101010101000000000000000000#0",
            "great_divide": "31,32#16,28#16,3#9#2#0,0,30,31_16,0,16,31_30,0,0,31_0,31,30,0_16,31,16,0_30,31,0,0#100,9,Horizontal#00000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000111111111111111111111111100000000000000000000000000000000000000000000000000000000000000000000111111111111111111111111100000011111111111111111111111110000000000000000000000000000000000000000000000000000000000000000000011111111111111111111111110000001111111111111111111111111000000111111111111111111111111100000000000000000000000000000000000000000000000000000000000000000000111111111111111111111111100000011111111111111111111111110000001111111111111111111111111000000000000000000000000000000000000000000000000000000000000000000001111111111111111111111111000000111111111111111111111111100000000000000000000000000000000000000000000000000000000000000000000111111111111111111111111100000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000#0",
            "cage":"11,11#1,5#9,5#5#2##30,1,Vertical#1010101010101010101010101010101010101010101000101010100000101010000010101010001010101010101010101010101010101010101010101#0",
            "empty":"9,9#1,4#7,4#5#2##20,1,Vertical#000000000000000000000000000000000000000000000000000000000000000000000000000000000#0",
            "empty_large":"19,19#3,9#15,9#9#2##100,7,Vertical#0000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000#0",
            "ssspline":"24,19#7,13#16,5#7#2##50,3,Origin#111110000000000000000000000010000000000000000000000001000000000000111100000000100000000000100000000000100000000000100000000000010000000000111100000000010000000000000100000000001000000000000100000000000100000000111100000000000011110000000000001111000000001000000000001000000000000100000000001000000000000010000000001111000000000010000000000001000000000001000000000001000000000001000000001111000000000000100000000000000000000000010000000000000000000000011111#0",
            "combustible_lemons":"21,20#10,1#10,18#5#2#4,4,4,15_16,4,16,15_0,8,0,11_20,8,20,11_0,11,0,8_20,11,20,8_4,15,4,4_16,15,16,4#100,4,Horizontal#000000000000000000000000000000000000000000000000000000000000000000110000000000011000000100000000000001000000000000000000000000000000000000000000000000000000000000000000000000000000000000000111111111111111111111111111111111111111111000000000000000000000000000000000000000000000000000000000000000000000000000000000000000100000000000001000000110000000000011000000000000000000000000000000000000000000000000000000000000000000#0"
        }
        
        if map_name in maps_data:
            map_string = maps_data[map_name]
            return Map(map_string)
            
        # Default empty map
        return Map(maps_data["empty"])
            
    def create_env(self, map_name="empty", opponent=None, verbose=False):
        """Create a ByteFight environment with the specified map and opponent."""
        game_map = self.load_map_from_json(map_name)
        
        if opponent is None:
            opponent = ConfigurableOpponent()
            
        env = ByteFightSnakeEnv(
            game_map=game_map,
            opponent_controller=opponent,
            render_mode=None,
            handle_bidding=True,
            verbose=verbose
        )
        
        return env
    
    def test_bidding_system(self):
        """Test that the bidding system works correctly."""
        print("\n=== Testing Bidding System ===")
        
        # Case 1: Player bids 0, Opponent bids 0 (should be random resolution)
        opponent = ConfigurableOpponent(bid_value=0)
        env = self.create_env(opponent=opponent)
        obs, info = env.reset()
        
        self.assert_test(
            "bidding_resolution_tie",
            'bidding_successful' in info and info['bidding_successful'],
            "Bidding should be successfully resolved for equal bids"
        )
        
        # Case 2: Player bids less than opponent (opponent should go first)
        opponent = ConfigurableOpponent(bid_value=3)
        env = self.create_env(opponent=opponent)
        env.board.snake_a.start(env.board.snake_a.get_head_loc(), 6)
        env.board.snake_b.start(env.board.snake_b.get_head_loc(), 6)
        
        # Force player A bid to be less
        env.board.resolve_bid(1, 3)
        
        pb = PlayerBoard(True, env.board)
        self.assert_test(
            "bidding_higher_bid_goes_first",
            not pb.is_my_turn(),
            "Opponent with higher bid should go first"
        )
        
    def test_basic_movement(self):
        """Test that basic movement works correctly."""
        print("\n=== Testing Basic Movement ===")
        env = self.create_env()
        obs, info = env.reset()
        
        # Get the initial position of our snake's head
        pb = PlayerBoard(True, env.board)
        initial_head_pos = pb.get_head_location(enemy=False)
        self.log(f"Initial head position: {initial_head_pos}")
        
        # First move: FORWARD (2)
        obs, reward, done, truncated, info = env.step(2)  # FORWARD
        
        # Check that we've moved
        pb = PlayerBoard(True, env.board)
        new_head_pos = pb.get_head_location(enemy=False)
        self.log(f"New head position after FORWARD: {new_head_pos}")
        
        self.assert_test(
            "basic_movement_changes_position",
            not np.array_equal(initial_head_pos, new_head_pos),
            "Snake should have moved after FORWARD action"
        )
        
        # Second move in the same turn: RIGHT (4)
        prev_head_pos = new_head_pos.copy()
        obs, reward, done, truncated, info = env.step(4)  # RIGHT
        
        # Check that we've moved again
        pb = PlayerBoard(True, env.board)
        new_head_pos = pb.get_head_location(enemy=False)
        self.log(f"New head position after RIGHT: {new_head_pos}")
        
        self.assert_test(
            "basic_movement_consecutive_moves",
            not np.array_equal(prev_head_pos, new_head_pos),
            "Snake should have moved after second action"
        )
        
        # End turn and check counter reset
        obs, reward, done, truncated, info = env.step(6)  # END_TURN
        
        self.assert_test(
            "basic_movement_end_turn",
            env.move_counter == 1 and env.turn_counter > 1,
            f"After END_TURN, expected move_counter=1, turn_counter>1, got move_counter={env.move_counter}, turn_counter={env.turn_counter}"
        )
        
    def test_direction_constraints(self):
        """Test that direction constraints are correctly enforced."""
        print("\n=== Testing Direction Constraints ===")
        env = self.create_env()
        obs, info = env.reset()
        
        # Get the initial heading
        pb = PlayerBoard(True, env.board)
        initial_heading = pb.get_direction(enemy=False)
        self.log(f"Initial heading: {initial_heading.name if hasattr(initial_heading, 'name') else initial_heading}")
        
        # Make a forward move to establish direction
        obs, reward, done, truncated, info = env.step(2)  # FORWARD
        
        pb = PlayerBoard(True, env.board)
        heading = pb.get_direction(enemy=False)
        self.log(f"Heading after first move: {heading.name if hasattr(heading, 'name') else heading}")
        
        # Check if LEFT, FORWARD, and RIGHT moves are available (should be)
        # But BACKWARD (opposite) move should not be available
        available_actions = [i for i, val in enumerate(obs["action_mask"]) if val == 1]
        self.log(f"Available actions: {[AGENT_ACTIONS[i] for i in available_actions]}")
        
        # Check LEFT and RIGHT are available
        self.assert_test(
            "direction_constraint_lateral_available",
            0 in available_actions and 4 in available_actions,
            "LEFT and RIGHT should be available"
        )
        
        # Forward should be available
        self.assert_test(
            "direction_constraint_forward_available",
            2 in available_actions,
            "FORWARD should be available"
        )
        
        # Check END_TURN is available after a move
        self.assert_test(
            "end_turn_available_after_move",
            6 in available_actions,
            "END_TURN should be available after a move"
        )
        
    def test_sacrifice_mechanic(self):
        """Test that the sacrifice mechanic works correctly."""
        print("\n=== Testing Sacrifice Mechanic ===")
        env = self.create_env(map_name="empty_large")
        obs, info = env.reset()
        
        # Force snake to have a specific length for easier testing
        env.board.snake_a.start(env.board.snake_a.get_head_loc(), 10)
        
        # Reset again to apply changes
        obs, info = env.reset()
        
        # Get the initial length of our snake
        pb = PlayerBoard(True, env.board)
        initial_length = pb.get_length(enemy=False)
        self.log(f"Initial snake length: {initial_length}")
        
        # First move: should have sacrifice = 0 (no length change, except if apple eaten)
        obs, reward, done, truncated, info = env.step(2)  # FORWARD
        
        # Check that length is unchanged after first move
        pb = PlayerBoard(True, env.board)
        length_after_first = pb.get_length(enemy=False)
        self.log(f"Length after first move: {length_after_first}")
        
        # Calculate expected length (could be +2 if apple eaten)
        expected_length = initial_length
        if length_after_first > initial_length:
            # Account for apple being eaten
            expected_length = initial_length + 2
            self.log("Apple eaten during first move")
        
        self.assert_test(
            "sacrifice_first_move_no_length_loss",
            length_after_first == expected_length,
            f"Expected length to remain {expected_length} after first move, got {length_after_first}"
        )
        
        # Check sacrifice counter after first move
        self.assert_test(
            "sacrifice_counter_after_first_move",
            env.current_sacrifice == 0,
            f"Expected sacrifice=0 after first move, got {env.current_sacrifice}"
        )
        
        # Second move: should sacrifice 3 at tail, gain 1 at head (net loss of 2)
        expected_length = length_after_first - 2
        obs, reward, done, truncated, info = env.step(2)  # FORWARD again
        
        # Check that length decreased by 2
        pb = PlayerBoard(True, env.board)
        length_after_second = pb.get_length(enemy=False)
        self.log(f"Length after second move: {length_after_second}")
        
        self.assert_test(
            "sacrifice_second_move_length_loss_of_2",
            length_after_second == expected_length,
            f"Expected length to decrease to {expected_length} after second move, got {length_after_second}"
        )
        
        # Check sacrifice counter after second move
        self.assert_test(
            "sacrifice_counter_after_second_move",
            env.current_sacrifice == 2,
            f"Expected sacrifice=2 after second move, got {env.current_sacrifice}"
        )
        
        # Third move: should sacrifice 5 at tail, gain 1 at head (net loss of 4)
        expected_length = length_after_second - 4
        obs, reward, done, truncated, info = env.step(2)  # FORWARD again
        
        # Check that length decreased by 4 more
        pb = PlayerBoard(True, env.board)
        length_after_third = pb.get_length(enemy=False)
        self.log(f"Length after third move: {length_after_third}")
        
        self.assert_test(
            "sacrifice_third_move_length_loss_of_4",
            length_after_third == expected_length,
            f"Expected length to decrease to {expected_length} after third move, got {length_after_third}"
        )
        
        # Check sacrifice counter after third move
        self.assert_test(
            "sacrifice_counter_after_third_move",
            env.current_sacrifice == 4,
            f"Expected sacrifice=4 after third move, got {env.current_sacrifice}"
        )
        
    def test_trap_placement(self):
        """Test that trap placement works correctly."""
        print("\n=== Testing Trap Placement ===")
        env = self.create_env(map_name="empty_large")
        
        # Force snake to have a specific length for trap testing
        env.board.snake_a.start(env.board.snake_a.get_head_loc(), 8)
        
        obs, info = env.reset()
        
        # Get the initial length and tail position
        pb = PlayerBoard(True, env.board)
        initial_length = pb.get_length(enemy=False)
        initial_tail_pos = pb.get_tail_location(enemy=False)
        self.log(f"Initial snake length: {initial_length}")
        self.log(f"Initial tail position: {initial_tail_pos}")
        
        # FORWARD moves first to set up for a trap
        obs, reward, done, truncated, info = env.step(2)  # FORWARD
        
        # Try to place a trap (TRAP = 5)
        obs, reward, done, truncated, info = env.step(5)  # TRAP
        
        # If the snake has enough unqueued length, the trap should be placed successfully
        pb = PlayerBoard(True, env.board)
        new_length = pb.get_length(enemy=False)
        
        # Check if length decreased by 1
        self.assert_test(
            "trap_placement_decreases_length",
            new_length == initial_length - 1 or done,
            f"Expected length to decrease by 1 after trap placement, from {initial_length} to {initial_length-1}, got {new_length}"
        )
        
        # Check if a trap exists at the tail position
        has_trap = False
        if not done:  # Only check if game didn't end
            has_trap = pb.has_my_trap(int(initial_tail_pos[0]), int(initial_tail_pos[1]))
            
        self.assert_test(
            "trap_exists_at_tail_position",
            has_trap or done,
            f"Expected to find a trap at the initial tail position {initial_tail_pos}"
        )
        
        # Check trap counter
        if not done:
            traps_placed = pb.get_traps_placed(enemy=False)
            self.assert_test(
                "trap_counter_incremented",
                traps_placed == 1,
                f"Expected traps_placed to be 1, got {traps_placed}"
            )
            
            # Check trap lifetime
            trap_life = pb.get_my_trap_life(int(initial_tail_pos[0]), int(initial_tail_pos[1]))
            self.assert_test(
                "trap_has_lifetime",
                trap_life > 0,
                f"Expected trap to have positive lifetime, got {trap_life}"
            )
        
    def test_apple_eating(self):
        """Test that eating apples works correctly."""
        print("\n=== Testing Apple Eating ===")
        env = self.create_env(map_name="empty_large")
        obs, info = env.reset()
        
        # Place an apple in front of the snake's head
        pb = PlayerBoard(True, env.board)
        head_pos = pb.get_head_location(enemy=False)
        heading = pb.get_direction(enemy=False)
        
        # Calculate position in front of snake
        next_pos = None
        if heading == Action.NORTH:
            next_pos = (head_pos[0], head_pos[1] - 1)
        elif heading == Action.EAST:
            next_pos = (head_pos[0] + 1, head_pos[1])
        elif heading == Action.SOUTH:
            next_pos = (head_pos[0], head_pos[1] + 1)
        elif heading == Action.WEST:
            next_pos = (head_pos[0] - 1, head_pos[1])
        
        # Place an apple
        if next_pos:
            env.board.cells_apples[next_pos[1], next_pos[0]] = 1
            self.log(f"Placed apple at {next_pos}")
        
        # Get the initial length
        initial_length = pb.get_length(enemy=False)
        self.log(f"Initial snake length: {initial_length}")
        
        # Move forward to eat the apple
        obs, reward, done, truncated, info = env.step(2)  # FORWARD
        
        # Check if length increased
        pb = PlayerBoard(True, env.board)
        new_length = pb.get_length(enemy=False)
        
        self.assert_test(
            "apple_increases_length",
            new_length > initial_length,
            f"Expected length to increase after eating apple, from {initial_length} to something larger, got {new_length}"
        )
        
        # Check queued length
        queued_length = pb.get_queued_length(enemy=False)
        self.assert_test(
            "apple_creates_queued_length",
            queued_length > 0,
            f"Expected queued_length > 0 after eating apple, got {queued_length}"
        )
        
        # Check that apple count increased
        apples_eaten = pb.get_apples_eaten(enemy=False)
        self.assert_test(
            "apple_counter_incremented",
            apples_eaten > 0,
            f"Expected apples_eaten > 0 after eating apple, got {apples_eaten}"
        )
        
    def test_portal_interaction(self):
        """Test that portal interactions work correctly."""
        print("\n=== Testing Portal Interaction ===")
        env = self.create_env(map_name="combustible_lemons")
        obs, info = env.reset()
        
        # Find portal locations
        pb = PlayerBoard(True, env.board)
        portal_mask = pb.get_portal_mask()
        
        # Check if portals exist
        portals_exist = np.any(portal_mask)
        self.assert_test(
            "portals_exist_on_map",
            portals_exist,
            "Expected to find portals on the map"
        )
        
        if portals_exist:
            # Get portal coordinates
            portal_coords = list(zip(*np.where(portal_mask)))
            self.log(f"Portal coordinates: {portal_coords}")
            
            # Save snake's starting position
            head_pos = pb.get_head_location(enemy=False)
            
            # Try to navigate to a portal
            # This is hard to automate, so we'll just check portal functionality
            for portal_pos in portal_coords:
                # Get the destination portal
                y, x = portal_pos
                dest = pb.get_portal_dest(x, y)
                
                self.assert_test(
                    f"portal_dest_{x}_{y}_valid",
                    dest is not None and not np.array_equal(dest, np.array([-1, -1])),
                    f"Expected valid destination for portal at ({x}, {y}), got {dest}"
                )
            
    def test_decay_mechanic(self):
        """Test that the decay mechanic works correctly."""
        print("\n=== Testing Decay Mechanic ===")
        env = self.create_env()
        obs, info = env.reset()
        
        # Override the turn counter to simulate later game stages
        initial_turn = 990  # Just before decay starts at 1000
        env.turn_counter = initial_turn
        
        # Get the initial length
        pb = PlayerBoard(True, env.board)
        initial_length = pb.get_length(enemy=False)
        self.log(f"Initial snake length at turn {initial_turn}: {initial_length}")
        
        # Set currently_decaying flag in the environment
        self.assert_test(
            "decay_flag_exists",
            hasattr(env, '_apply_decay_if_needed'),
            "Environment should have a decay function"
        )
        
        # Advance turns to trigger decay
        for i in range(20):
            # Make a simple move and end turn
            obs, reward, done, truncated, info = env.step(2)  # FORWARD
            obs, reward, done, truncated, info = env.step(6)  # END_TURN
            
            if done:
                break
        
        # Check if turn counter advanced past 1000 (decay zone)
        self.assert_test(
            "decay_mechanic_turn_advancement",
            env.turn_counter >= 1000,
            f"Turn counter should have advanced to 1000+, got {env.turn_counter}"
        )
        
        # Validate decay intervals
        pb = PlayerBoard(True, env.board)
        is_decaying = pb.currently_decaying()
        decay_interval = pb.get_current_decay_interval()
        
        self.assert_test(
            "decay_flag_activates_after_turn_1000",
            is_decaying,
            f"Expected snake to be decaying after turn 1000, got {is_decaying}"
        )
        
        # Check that the decay interval is valid
        self.assert_test(
            "decay_interval_set_correctly",
            decay_interval is not None and decay_interval > 0,
            f"Expected positive decay interval, got {decay_interval}"
        )
        
    def test_invalid_moves(self):
        """Test that invalid moves are correctly detected and handled."""
        print("\n=== Testing Invalid Move Detection ===")
        env = self.create_env()
        obs, info = env.reset()
        
        # Get the action mask from the observation
        action_mask = obs["action_mask"]
        valid_actions = [i for i, is_valid in enumerate(action_mask) if is_valid]
        invalid_actions = [i for i, is_valid in enumerate(action_mask) if not is_valid]
        
        self.log(f"Valid actions: {[AGENT_ACTIONS[i] for i in valid_actions]}")
        self.log(f"Invalid actions: {[AGENT_ACTIONS[i] for i in invalid_actions]}")
        
        # First, check if the empty action (6, or END_TURN) at the beginning is invalid
        # (it should be because no move has been made yet)
        self.assert_test(
            "end_turn_invalid_as_first_action",
            6 in invalid_actions,
            "END_TURN should be invalid as the first action"
        )
        
        # Try to take an invalid move and verify the game ends
        if invalid_actions:
            invalid_action = invalid_actions[0]
            self.log(f"Attempting invalid action: {AGENT_ACTIONS[invalid_action]}")
            
            obs, reward, done, truncated, info = env.step(invalid_action)
            
            self.assert_test(
                "invalid_move_ends_game",
                done and reward < 0,
                f"Game should end with negative reward when making invalid move, got done={done}, reward={reward}"
            )
            
            if done:
                self.assert_test(
                    "invalid_move_correct_winner",
                    info["winner"] == Result.PLAYER_B,
                    f"Expected winner to be PLAYER_B after invalid move, got {info['winner']}"
                )
        
    def test_end_game_conditions(self):
        """Test all the end game conditions."""
        print("\n=== Testing End Game Conditions ===")
        
        # Test 1: Game ends when snake length becomes less than minimum
        env = self.create_env(map_name="empty_large")
        
        # Force snake to have minimum+1 length
        env.board.snake_a.start(env.board.snake_a.get_head_loc(), 3)
        
        obs, info = env.reset()
        
        # Get the minimum length
        pb = PlayerBoard(True, env.board)
        min_length = pb.get_min_player_size()
        
        # Perform 2 consecutive moves to reduce length below minimum
        obs, reward, done, truncated, info = env.step(2)  # FORWARD
        obs, reward, done, truncated, info = env.step(2)  # FORWARD
        
        self.assert_test(
            "minimum_length_violation_ends_game",
            done and reward < 0,
            f"Game should end when snake length falls below minimum ({min_length})"
        )
        
        if done:
            self.assert_test(
                "minimum_length_correct_winner",
                info["winner"] == Result.PLAYER_B,
                f"Expected winner to be PLAYER_B after minimum length violation, got {info['winner']}"
            )
            
        # Test 2: Game ends when snake hits a wall
        env = self.create_env()
        obs, info = env.reset()
        
        # Get the dimensions of the board
        pb = PlayerBoard(True, env.board)
        dim_x = pb.get_dim_x()
        dim_y = pb.get_dim_y()
        
        # Get the head location
        head_pos = pb.get_head_location(enemy=False)
        
        # Try to move into a wall
        # Check if head is near a wall and try to move into it
        wall_directions = []
        if head_pos[0] <= 1:  # Near left wall
            wall_directions.append(0)  # LEFT
        if head_pos[0] >= dim_x - 2:  # Near right wall
            wall_directions.append(4)  # RIGHT
        if head_pos[1] <= 1:  # Near top wall
            wall_directions.append(1)  # FORWARD_LEFT if facing east
        if head_pos[1] >= dim_y - 2:  # Near bottom wall
            wall_directions.append(3)  # FORWARD_RIGHT if facing east
            
        moved_to_wall = False
        for direction in wall_directions:
            # Try the direction
            obs, reward, done, truncated, info = env.step(direction)
            
            if done:
                moved_to_wall = True
                self.assert_test(
                    "wall_collision_ends_game",
                    done and reward < 0,
                    f"Game should end when snake hits a wall"
                )
                
                self.assert_test(
                    "wall_collision_correct_winner",
                    info["winner"] == Result.PLAYER_B,
                    f"Expected winner to be PLAYER_B after wall collision, got {info['winner']}"
                )
                break
                
        if not moved_to_wall:
            self.log("No suitable wall direction found for testing")
            
        # Test 3: Game ends when turn limit is reached
        env = self.create_env()
        obs, info = env.reset()
        
        # Set turn counter close to the limit
        env.turn_counter = 1998
        
        # Make a couple more moves to reach turn limit
        obs, reward, done, truncated, info = env.step(2)  # FORWARD
        obs, reward, done, truncated, info = env.step(6)  # END_TURN
        
        self.assert_test(
            "turn_limit_ends_game",
            done and env.turn_counter >= 2000,
            f"Game should end after 2000 turns, turn counter: {env.turn_counter}, done: {done}"
        )
        
        if done and env.turn_counter >= 2000:
            self.assert_test(
                "turn_limit_triggers_tiebreak",
                info["winner"] is not None,
                f"Turn limit should trigger tiebreak with valid winner, got: {info['winner']}"
            )
            
    def test_action_masking(self):
        """Test that action masking correctly reflects available actions."""
        print("\n=== Testing Action Masking ===")
        env = self.create_env()
        obs, info = env.reset()
        
        # Initial mask - END_TURN and TRAP should be disabled
        initial_mask = obs["action_mask"]
        
        self.assert_test(
            "initial_mask_disables_end_turn",
            initial_mask[6] == 0,
            "END_TURN should be disabled on first move"
        )
        
        self.assert_test(
            "initial_mask_disables_trap",
            initial_mask[5] == 0,
            "TRAP should be disabled on first move"
        )
        
        # Make a move and check updated mask
        obs, reward, done, truncated, info = env.step(2)  # FORWARD
        updated_mask = obs["action_mask"]
        
        # After a move, END_TURN should be enabled
        self.assert_test(
            "end_turn_enabled_after_move",
            updated_mask[6] == 1,
            "END_TURN should be enabled after making a move"
        )
        
        # Set up a scenario where TRAP should be enabled
        env = self.create_env(map_name="empty_large")
        
        # Force snake to have sufficient length for trap
        env.board.snake_a.start(env.board.snake_a.get_head_loc(), 8)
        
        obs, info = env.reset()
        
        # Make a move to set up for trap
        obs, reward, done, truncated, info = env.step(2)  # FORWARD
        trap_mask = obs["action_mask"]
        
        # Check if TRAP is enabled
        self.assert_test(
            "trap_enabled_with_sufficient_length",
            trap_mask[5] == 1,
            "TRAP should be enabled when snake has sufficient unqueued length"
        )
        
    def test_opponent_interaction(self):
        """Test that opponent's actions affect the game state correctly."""
        print("\n=== Testing Opponent Interaction ===")
        
        # Create opponent that performs specific actions
        opponent = ConfigurableOpponent(
            bid_value=0,
            actions=[Action.TRAP, Action.NORTH, Action.NORTH]
        )
        
        env = self.create_env(map_name="empty_large", opponent=opponent)
        
        # Force both snakes to have sufficient length
        env.board.snake_a.start(env.board.snake_a.get_head_loc(), 8)
        env.board.snake_b.start(env.board.snake_b.get_head_loc(), 8)
        
        obs, info = env.reset()
        
        # Take a turn and end it to let opponent play
        obs, reward, done, truncated, info = env.step(2)  # FORWARD
        obs, reward, done, truncated, info = env.step(6)  # END_TURN
        
        # Check if opponent's actions were processed
        pb = PlayerBoard(True, env.board)
        enemy_traps = np.sum(pb.get_trap_mask(my_traps=False, enemy_traps=True) > 0)
        
        self.assert_test(
            "opponent_trap_placed",
            enemy_traps > 0,
            f"Opponent should have placed a trap, found {enemy_traps} traps"
        )
        
        # Take another turn and check that opponent moved as expected
        obs, reward, done, truncated, info = env.step(2)  # FORWARD
        obs, reward, done, truncated, info = env.step(6)  # END_TURN
        
        # Check that game is still in progress
        self.assert_test(
            "game_continues_after_opponent_move",
            not done,
            "Game should continue after opponent's move"
        )
        
    def test_trap_interaction(self):
        """Test that traps interact with snakes correctly."""
        print("\n=== Testing Trap Interaction ===")
        
        # Create a custom opponent that stays in place
        opponent = ConfigurableOpponent(
            bid_value=0,
            actions=[]  # No actions, will default to NORTH
        )
        
        env = self.create_env(map_name="empty_large", opponent=opponent)
        
        # Force snakes to have sufficient length
        env.board.snake_a.start(env.board.snake_a.get_head_loc(), 10)
        env.board.snake_b.start(env.board.snake_b.get_head_loc(), 10)
        
        obs, info = env.reset()
        
        # Move and place a trap
        obs, reward, done, truncated, info = env.step(2)  # FORWARD
        obs, reward, done, truncated, info = env.step(5)  # TRAP
        
        # Get trap position
        pb = PlayerBoard(True, env.board)
        trap_positions = np.where(pb.get_trap_mask(my_traps=True, enemy_traps=False) > 0)
        
        if len(trap_positions[0]) > 0:
            trap_y, trap_x = trap_positions[0][0], trap_positions[1][0]
            self.log(f"Placed trap at ({trap_x}, {trap_y})")
            
            # Get trap lifetime
            trap_life = pb.get_my_trap_life(trap_x, trap_y)
            self.assert_test(
                "trap_initial_lifetime",
                trap_life > 0,
                f"Trap should have positive lifetime, got {trap_life}"
            )
            
            # End turn and let opponent play (potentially hitting trap)
            obs, reward, done, truncated, info = env.step(6)  # END_TURN
            
            # Take another turn
            obs, reward, done, truncated, info = env.step(2)  # FORWARD
            
            # Check if trap is still there
            pb = PlayerBoard(True, env.board)
            new_trap_life = -1
            if pb.has_my_trap(trap_x, trap_y):
                new_trap_life = pb.get_my_trap_life(trap_x, trap_y)
            
            self.assert_test(
                "trap_lifetime_decreases",
                new_trap_life < trap_life and new_trap_life != -1,
                f"Trap lifetime should decrease from {trap_life} to something lower, got {new_trap_life}"
            )
        
    def run_all_tests(self):
        """Run all test cases."""
        self.test_bidding_system()
        self.test_basic_movement()
        self.test_direction_constraints()
        self.test_sacrifice_mechanic()
        self.test_trap_placement()
        self.test_apple_eating()
        #self.test_portal_interaction()
        #self.test_decay_mechanic()
        self.test_invalid_moves()
        #self.test_end_game_conditions()
        self.test_action_masking()
        self.test_opponent_interaction()
        self.test_trap_interaction()
        
        # Print summary of results
        self.print_summary()

def main():
    """Main function to run the tests."""
    parser = argparse.ArgumentParser(description="Comprehensive test suite for ByteFight environment")
    parser.add_argument('--verbose', action='store_true', help='Enable verbose output')
    parser.add_argument('--test', type=str, default='all', 
                      choices=['all', 'bidding', 'movement', 'direction', 'sacrifice', 
                               'trap', 'apple', 'portal', 'decay', 'invalid', 
                               'end_game', 'action_mask', 'opponent', 'trap_interaction'],
                      help='Specific test to run')
    args = parser.parse_args()
    
    tester = ByteFightTester(verbose=args.verbose)
    
    if args.test == 'all':
        tester.run_all_tests()
    elif args.test == 'bidding':
        tester.test_bidding_system()
    elif args.test == 'movement':
        tester.test_basic_movement()
    elif args.test == 'direction':
        tester.test_direction_constraints()
    elif args.test == 'sacrifice':
        tester.test_sacrifice_mechanic()
    elif args.test == 'trap':
        tester.test_trap_placement()
    elif args.test == 'apple':
        tester.test_apple_eating()
    #elif args.test == 'portal':
   #     tester.test_portal_interaction()
  #  elif args.test == 'decay':
  #      tester.test_decay_mechanic()
    elif args.test == 'invalid':
        tester.test_invalid_moves()
  #  elif args.test == 'end_game':
   #     tester.test_end_game_conditions()
    elif args.test == 'action_mask':
        tester.test_action_masking()
    elif args.test == 'opponent':
        tester.test_opponent_interaction()
    elif args.test == 'trap_interaction':
        tester.test_trap_interaction()
    
    # Print summary if only running a single test
    if args.test != 'all':
        tester.print_summary()

if __name__ == "__main__":
    main()