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


# ----------------------------
# 6) A simpler features extractor
# ----------------------------
class ByteFightFeaturesExtractor(BaseFeaturesExtractor):
    """
    Smaller architecture that aims for ~1M parameters total:
      - CNN: Channels [32 -> 64 -> 128] with one residual block and attention
      - MLP for scalar (15 -> 64 -> 64)
      - Fusion: 128 + 64 -> 256
      - Output dimension: 256
    """

    def __init__(self, observation_space: spaces.Dict, features_dim: int = 256):
        super().__init__(observation_space, features_dim=features_dim)

        # CNN with smaller channels
        self.cnn = nn.Sequential(
            nn.Conv2d(9, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            ResidualBlock(32),  # single residual block
            nn.MaxPool2d(2),  # from 64x64 -> 32x32

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            SpatialAttention(64),
            nn.MaxPool2d(2),  # from 32x32 -> 16x16

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            SpatialAttention(128),
            nn.MaxPool2d(2),  # from 16x16 -> 8x8

            nn.Flatten(),
            # 128 channels at 8x8 => 128 * 8 * 8 = 8192
            nn.Linear(128 * 8 * 8, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.1),
        )

        # MLP for scalar features
        self.mlp = nn.Sequential(
            nn.Linear(15, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Dropout(0.1),
        )

        # Fusion => produce final 256-dim features
        self.fusion = FeatureFusion(256, 64, features_dim)

    def forward(self, obs: dict) -> torch.Tensor:
        board_img = obs["board_image"]      # shape: [B,9,64,64]
        scalar_feats = obs["features"]      # shape: [B,15]

        cnn_out = self.cnn(board_img)       # shape: [B,256]
        mlp_out = self.mlp(scalar_feats)    # shape: [B,64]

        fused = self.fusion(cnn_out, mlp_out)  # shape: [B,256]
        return fused

# ----------------------------
# 7) Final simplified policy
# ----------------------------
class ByteFightMaskedPolicy(ActorCriticPolicy):
    """
    A simpler policy ~1M params:
      - uses SimplerByteFightFeaturesExtractor
      - smaller net_arch for the final MLPExtractor
    """
    def __init__(self, *args, **kwargs):
        kwargs["features_extractor_class"] = ByteFightFeaturesExtractor
        super().__init__(*args, **kwargs)

        # Replace final actor with ActionSpecificHead
        self.action_net = ActionSpecificHead(self.mlp_extractor.latent_dim_pi, self.action_space.n)

    def _build_mlp_extractor(self) -> None:
        # smaller net_arch e.g. [128, 64] for pi and vf
        self.mlp_extractor = MlpExtractor(
            self.features_dim,
            net_arch=dict(pi=[128, 64], vf=[128, 64]),
            activation_fn=self.activation_fn,
            device=self.device
        )

    def forward(self, obs, deterministic=False):
        dist = self.get_distribution(obs)
        if deterministic:
            actions = torch.argmax(dist.logits, dim=1)
        else:
            actions = dist.sample()
        log_probs = dist.log_prob(actions)
        values = self.forward_critic(obs)
        return actions, values, log_probs

    def get_distribution(self, obs: dict, **kwargs):
        features = self.extract_features(obs, self.features_extractor)
        latent_pi, latent_vf = self.mlp_extractor(features)
        logits = self.action_net(latent_pi)

        action_mask = obs["action_mask"]
        assert action_mask.shape == logits.shape, f"Mask {action_mask.shape} vs logits {logits.shape}"

        dist = CategoricalMasked(logits=logits, mask=action_mask)
        return dist

    def forward_actor(self, obs: torch.Tensor, **kwargs) -> torch.Tensor:
        features = self.extract_features(obs, self.features_extractor)
        latent_pi, _ = self.mlp_extractor(features)
        return self.action_net(latent_pi)

    def forward_critic(self, obs: torch.Tensor) -> torch.Tensor:
        features = self.extract_features(obs, self.features_extractor)
        _, latent_vf = self.mlp_extractor(features)
        return self.value_net(latent_vf)