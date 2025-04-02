import torch
import torch.nn as nn
import numpy as np
import gymnasium as gym
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

class SnakeCNN(BaseFeaturesExtractor):
    """
    Parameter-efficient CNN feature extractor for ByteFight: Snake environment.
    
    This CNN uses more aggressive pooling and smaller feature dimensions
    to reduce the parameter count while maintaining performance.
    """
    def __init__(self, observation_space, features_dim=256):
        super().__init__(observation_space, features_dim)
        
        # Get input dimensions from observation space
        n_input_channels = observation_space.spaces["image"].shape[0]  # 9 channels
        
        # CNN for spatial features with more aggressive pooling
        self.conv_layers = nn.Sequential(
            # First convolutional block
            nn.Conv2d(n_input_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            
            # Max pooling to reduce dimensions early (64x64 -> 32x32)
            nn.MaxPool2d(2),
            
            # Second convolutional block
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            
            # More pooling (32x32 -> 16x16)
            nn.MaxPool2d(2),
            
            # Third convolutional block
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            
            # More pooling (16x16 -> 8x8)
            nn.MaxPool2d(2),
            
            # Fourth convolutional block
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            
            # Final pooling (8x8 -> 4x4)
            nn.MaxPool2d(2),
            
            # Flatten the spatial features
            nn.Flatten()
        )
        
        # Action mask processor (smaller)
        self.action_net = nn.Sequential(
            nn.Linear(observation_space.spaces["action_mask"].shape[0], 32),
            nn.ReLU()
        )
        
        # Calculate CNN output size dynamically
        with torch.no_grad():
            # Sample input - using maximum board dimensions (64x64)
            dummy_input = torch.zeros(1, n_input_channels, 64, 64)
            cnn_out = self.conv_layers(dummy_input)
            cnn_out_size = cnn_out.shape[1]
            print(f"CNN output size: {cnn_out_size}")  # Should be 128 * 4 * 4 = 2048
        
        # Combine CNN features with action mask features (smaller hidden layers)
        self.combined_net = nn.Sequential(
            nn.Linear(cnn_out_size + 32, 512),
            nn.ReLU(),
            nn.Linear(512, features_dim),
            nn.ReLU()
        )
    
    def forward(self, observations):
        # Process the board image through CNN
        cnn_features = self.conv_layers(observations["image"])
        
        # Process the action mask
        mask_features = self.action_net(observations["action_mask"])
        
        # Concatenate the features
        combined = torch.cat([cnn_features, mask_features], dim=1)
        
        # Final processing
        return self.combined_net(combined)


class MaskedActionWrapper(gym.ActionWrapper):
    """
    Wrapper to ensure the agent only selects valid actions based on the action mask.
    """
    def __init__(self, env):
        super().__init__(env)
    
    def action(self, action):
        """
        Apply the action mask to ensure only valid actions are taken.
        """
        observation = self.env.unwrapped._make_observation()
        action_mask = observation["action_mask"]
        
        # If the chosen action is invalid and there are valid actions available
        if action_mask[action] == 0 and np.sum(action_mask) > 0:
            # Choose a random valid action instead
            valid_actions = np.where(action_mask == 1)[0]
            action = np.random.choice(valid_actions)
        
        return action