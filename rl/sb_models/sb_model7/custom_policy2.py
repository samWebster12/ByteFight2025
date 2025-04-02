import torch
from torch.distributions import Categorical
from torch import einsum
import torch.nn as nn
from einops import reduce
from typing import Optional

from stable_baselines3.common.torch_layers import BaseFeaturesExtractor, MlpExtractor
from stable_baselines3.common.policies import ActorCriticPolicy
from gymnasium import spaces

class CategoricalMasked(Categorical):
    """
    A Categorical distribution that allows invalid actions to be masked out.
    Invalid actions get logits set to a very large negative number (finfo.min).
    """
    def __init__(self, logits: torch.Tensor, mask: Optional[torch.Tensor] = None):
        # Convert mask from uint8 -> bool if needed
        if mask is not None:
            mask = mask.bool()
        self.mask = mask
        # Replace invalid-action logits with -inf
        if mask is not None:
            mask_value = torch.finfo(logits.dtype).min
            adjusted_logits = torch.where(mask, logits, mask_value)
            super().__init__(logits=adjusted_logits)
        else:
            super().__init__(logits=logits)

    def get_actions(self, deterministic=False):
        if deterministic:
            # Argmax over the masked logits
            return torch.argmax(self.logits, dim=1)
        else:
            # sample stochastically
            return self.sample()
    
    def sample(self) -> torch.Tensor:
        """
        We override .sample() so invalid actions have zero probability.
        One approach is to do the same "-inf" trick at the time of sampling.
        Or do it in __init__ once. 
        """
        if self.mask is not None:
            # We can apply the mask logic in the constructor 
            # or reapply it here. Let's do it in the constructor for clarity:
            pass
        return super().sample()

    def entropy(self):
        """
        We override the standard entropy computation so it ignores invalid actions.
        """
        if self.mask is None:
            return super().entropy()

        # The standard Categorical uses self.logits and self.probs.
        # We'll do an elementwise multiply (logits * probs) and sum over valid actions only.
        p_log_p = einsum("ij,ij->ij", self.logits, self.probs)
        # Zero out contributions of invalid actions:
        p_log_p = torch.where(self.mask, p_log_p, torch.zeros_like(p_log_p))
        # Sum across actions, then negate
        return -reduce(p_log_p, "b a -> b", "sum", b=self.batch_size, a=self.n_actions)


class ResidualBlock(nn.Module):
    """Residual block with batch normalization for improved gradient flow"""
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)
        
    def forward(self, x):
        residual = x
        out = nn.functional.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        return nn.functional.relu(out)


class SpatialAttention(nn.Module):
    """Spatial attention module to focus on important areas of the board"""
    def __init__(self, in_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, 1, kernel_size=1)
        
    def forward(self, x):
        # x shape: [batch, channels, height, width]
        attention = torch.sigmoid(self.conv(x))  # [batch, 1, height, width]
        return x * attention.expand_as(x)


class FeatureFusion(nn.Module):
    """Feature fusion module for better integration of CNN and MLP features"""
    def __init__(self, cnn_dim, mlp_dim, output_dim):
        super().__init__()
        self.fuse = nn.Sequential(
            nn.Linear(cnn_dim + mlp_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
    def forward(self, cnn_features, mlp_features):
        combined = torch.cat([cnn_features, mlp_features], dim=1)
        return self.fuse(combined)


class ActionSpecificHead(nn.Module):
    """Action-specific processing pathways for different action types"""
    def __init__(self, input_dim, num_actions):
        super().__init__()
        self.common = nn.Linear(input_dim, input_dim)
        self.heads = nn.ModuleList([
            nn.Linear(input_dim, 1) for _ in range(num_actions)
        ])
        
    def forward(self, x):
        common_features = torch.relu(self.common(x))
        return torch.cat([head(common_features) for head in self.heads], dim=1)


class ImprovedByteFightFeaturesExtractor(BaseFeaturesExtractor):
    """
    An improved features extractor for ByteFight that:
    - Uses a CNN with residual blocks and attention for the board state
    - Uses a deeper MLP for scalar features
    - Includes a feature fusion layer for better integration
    """
    def __init__(
        self,
        observation_space: spaces.Dict,
        features_dim=384
    ):
        super().__init__(observation_space, features_dim=features_dim)
        
        # CNN with residual blocks and attention
        self.cnn = nn.Sequential(
            # Initial conv layer
            nn.Conv2d(9, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            
            # First residual block
            ResidualBlock(64),
            nn.MaxPool2d(2),  # Size: 16x16
            
            # Second layer
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            SpatialAttention(128),  # Add attention
            
            # Second residual block
            ResidualBlock(128),
            nn.MaxPool2d(2),  # Size: 8x8
            
            # Third layer
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            SpatialAttention(256),  # Add attention
            
            # Final pooling and flatten
            nn.MaxPool2d(2),  # Size: 4x4
            nn.Flatten(),
            
            # Dense layers
            nn.Linear(256 * 8 * 8, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # Deeper MLP for scalar features
        self.mlp = nn.Sequential(
            nn.Linear(15, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # Feature fusion
        self.fusion = FeatureFusion(256, 128, features_dim)
        
    def forward(self, obs: dict) -> torch.Tensor:
        board_img = obs["board_image"]
        scalar_feats = obs["features"]
        
        cnn_out = self.cnn(board_img)
        mlp_out = self.mlp(scalar_feats)
        
        fused = self.fusion(cnn_out, mlp_out)
        return fused


class ImprovedByteFightMaskedPolicy(ActorCriticPolicy):
    """
    An improved policy that:
    - Uses ImprovedByteFightFeaturesExtractor to encode observations
    - Uses action-specific processing for different action types
    - Uses 'action_mask' to produce a CategoricalMasked distribution
      so invalid actions have probability 0
    """
    def __init__(self, *args, **kwargs):
        # Override features_extractor_class with our improved version
        kwargs["features_extractor_class"] = ImprovedByteFightFeaturesExtractor
        super().__init__(*args, **kwargs)
        
        # Replace the final action_net with action-specific processing
        self.action_net = ActionSpecificHead(self.mlp_extractor.latent_dim_pi, self.action_space.n)

    def _build_mlp_extractor(self) -> None:
        """
        Override the default to use our improved feature extractor.
        """
        self.mlp_extractor = MlpExtractor(
            self.features_dim,
            net_arch=self.net_arch, 
            activation_fn=self.activation_fn,
            device=self.device
        )

    def forward(self, obs, deterministic=False):
        dist = self.get_distribution(obs)

        if deterministic:
            # Argmax over the logits
            actions = torch.argmax(dist.logits, dim=1)
        else:
            # Sample from the masked distribution
            actions = dist.sample()

        log_probs = dist.log_prob(actions)
        values = self.forward_critic(obs)
        return actions, values, log_probs

    def get_distribution(self, obs: dict, **kwargs):
        """
        Build the masked distribution.
        """
        # Extract features
        features = self.extract_features(obs, self.features_extractor)
        
        # MLP extractor => separate policy & value latents
        latent_pi, latent_vf = self.mlp_extractor(features)

        # Final policy logits with action-specific processing
        logits = self.action_net(latent_pi)  # shape (B, n_actions)

        # Retrieve mask from obs
        action_mask = obs["action_mask"]  # expected shape: (B, n_actions)
        
        # Build the masked distribution
        assert action_mask.shape == logits.shape, f"Mask shape {action_mask.shape} doesn't match logits shape {logits.shape}"
        dist = CategoricalMasked(logits=logits, mask=action_mask)

        return dist

    def forward_actor(self, obs: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Used internally by SB3.
        """
        features = self.extract_features(obs, self.features_extractor)
        latent_pi, _ = self.mlp_extractor(features)
        return self.action_net(latent_pi)

    def forward_critic(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Returns the value estimate for the given observation.
        """
        features = self.extract_features(obs, self.features_extractor)
        _, latent_vf = self.mlp_extractor(features)
        values = self.value_net(latent_vf)
        return values