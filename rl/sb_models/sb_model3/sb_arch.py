import torch
import torch.nn as nn
import numpy as np
import gymnasium as gym
from typing import Dict, List, Tuple, Type, Optional, Union, Callable

from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.policies import ActorCriticPolicy


class SnakeCNN(BaseFeaturesExtractor):
    """CNN feature extractor for ByteFight Snake game."""
    def __init__(self, observation_space: gym.spaces.Dict, features_dim: int = 256):
        super().__init__(observation_space, features_dim)
        
        # CNN for processing the board state image (9 channels, 64x64)
        self.cnn = nn.Sequential(
            nn.Conv2d(9, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),  # Reduce to 32x32
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),  # Reduce to 16x16
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),  # Reduce to 8x8
            nn.Flatten(),
        )
        
        # Calculate CNN output size
        cnn_output_dim = 128 * 8 * 8  # 8192
        
        # Process scalar observations (non-image features)
        self.scalar_extractor = nn.Sequential(
            nn.Linear(11, 64),  # 11 scalar features from observation space
            nn.ReLU(),
        )
        
        # Combine both feature types
        self.combined = nn.Sequential(
            nn.Linear(cnn_output_dim + 64, features_dim),
            nn.ReLU(),
        )
        
    def forward(self, observations: Dict[str, torch.Tensor]) -> torch.Tensor:
        # Process image part
        image = observations["image"].float() / 255.0  # Normalize
        image_features = self.cnn(image)
        
        # Gather and process scalar features
        scalar_features = torch.cat([
            observations["turn_count"].float(),
            observations["my_length"].float(),
            observations["my_queued_length"].float(),
            observations["opponent_length"].float(),
            observations["opponent_queued_length"].float(),
            observations["max_traps_allowed"].float(),
            observations["time_left"].float(),
            observations["opponent_time_left"].float(),
            observations["is_decaying"].float(),
            observations["decay_rate"].float(),
            # Extra feature for bidding phase
            (observations["turn_count"] == 0).float(),
        ], dim=1)
        
        scalar_features = self.scalar_extractor(scalar_features)
        
        # Combine both feature types
        combined_features = torch.cat([image_features, scalar_features], dim=1)
        return self.combined(combined_features)


class MaskedActorCriticPolicy(ActorCriticPolicy):
    """Actor Critic Policy without action masking for now"""
    
    def forward(self, 
                obs: Dict[str, torch.Tensor], 
                deterministic: bool = False
               ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass in all the networks (actor and critic)"""
        features = self.extract_features(obs)
        latent_pi, latent_vf = self.mlp_extractor(features)
        
        # Value function
        values = self.value_net(latent_vf)
        
        # Action distribution
        distribution = self._get_action_dist_from_latent(latent_pi)
        
        # Get actions
        actions = distribution.get_actions(deterministic=deterministic)
        log_prob = distribution.log_prob(actions)
        
        return actions, values, log_prob


# Linear learning rate schedule
def linear_schedule(initial_value: float) -> Callable[[float], float]:
    """Linear learning rate schedule."""
    def func(progress_remaining: float) -> float:
        """Progress will decrease from 1 (beginning) to 0 (end)"""
        return progress_remaining * initial_value
    return func