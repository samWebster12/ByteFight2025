class SimplifiedByteFightEnv(gym.Wrapper):
    """
    A wrapper that simplifies the ByteFight environment for early training
    and progressively introduces complexity.
    """
    def __init__(self, env, difficulty=0):
        """
        Initialize the wrapper with a difficulty level:
        0 = Very simplified (single moves only, simplified bidding)
        1 = Basic (allow dual moves with enhanced rewards)
        2 = Intermediate (full moves with standard rewards)
        3 = Full (complete original environment)
        """
        super().__init__(env)
        self.difficulty = difficulty
        self.original_env = env
        
        # Store the original action space
        self.original_action_space = env.action_space
        
        # For simplest difficulty, limit to just bidding + single moves
        if self.difficulty <= 1:
            # We still need to use the MultiDiscrete action space to be compatible
            # but we'll interpret it differently
            self.action_space = spaces.MultiDiscrete([3, 9, 9, 9])
    
    def step(self, action):
        """
        Process the action according to the current difficulty level
        """
        multi_action = np.array(action, dtype=np.int64)
        
        # For the simplest difficulty (0): force single moves and help with bidding
        if self.difficulty == 0:
            # If in bidding phase, use first element as bid
            if hasattr(self.env, '_bid_phase') and self.env._bid_phase:
                # Leave bid as is - first element is used as bid
                pass
            else:
                # Force only single moves by setting first element to 0
                multi_action[0] = 0
                
                # Set second move and third move to safe defaults (no move)
                multi_action[2] = 0  # North (typically safe)
                multi_action[3] = 0  # North (typically safe)
        
        # For difficulty 1: allow up to two moves but with guided selection
        elif self.difficulty == 1:
            # If not in bidding phase
            if not (hasattr(self.env, '_bid_phase') and self.env._bid_phase):
                # Limit to at most 2 moves (0-1 in the action space)
                multi_action[0] = min(1, multi_action[0])
        
        # Call the original environment's step with our processed action
        obs, reward, done, truncated, info = self.env.step(multi_action)
        
        # Modify rewards based on difficulty
        if not done:
            # For the simplest difficulty, boost all rewards to encourage initial learning
            if self.difficulty == 0:
                # Bonus just for surviving a step
                reward += 0.5
                
                # If positive reward (ate apple, etc.), boost it further
                if reward > 0:
                    reward *= 2.0
            
            # For difficulty 1, smaller boost to rewards
            elif self.difficulty == 1:
                # Smaller bonus for surviving
                reward += 0.2
                
                # If positive reward, boost it a bit
                if reward > 0:
                    reward *= 1.5
        
        return obs, reward, done, truncated, info
    
    def reset(self, **kwargs):
        """Reset and apply simplified rules based on difficulty"""
        obs, info = self.env.reset(**kwargs)
        
        # Simplify bidding at the lowest difficulty
        if self.difficulty == 0 and hasattr(self.env, '_bid_phase') and self.env._bid_phase:
            # Auto-bid a safe value
            self.env._board.resolve_bid(1, 0)  # Always bid 1, opponent bids 0
            self.env._bid_phase = False
            
            # Reset observation after auto-bidding
            obs = self.env._make_observation()
        
        return obs, info
    
    def increase_difficulty(self):
        """Increase the difficulty level up to the maximum"""
        if self.difficulty < 3:
            prev_difficulty = self.difficulty
            self.difficulty += 1
            print(f"Difficulty increased from {prev_difficulty} to {self.difficulty}")
            return True
        return False