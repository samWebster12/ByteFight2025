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
    Invalid actions have their logits set to a very large negative number.
    """
    def __init__(self, logits: torch.Tensor, mask: Optional[torch.Tensor] = None):
        if mask is not None:
            mask = mask.bool()
        self.mask = mask
        if mask is not None:
            mask_value = torch.finfo(logits.dtype).min
            adjusted_logits = torch.where(mask, logits, mask_value)
            super().__init__(logits=adjusted_logits)
        else:
            super().__init__(logits=logits)

    def get_actions(self, deterministic=False):
        if deterministic:
            return torch.argmax(self.logits, dim=1)
        else:
            return self.sample()
    
    def sample(self) -> torch.Tensor:
        # Sampling is handled by the constructor using the masked logits.
        return super().sample()

    def entropy(self):
        if self.mask is None:
            return super().entropy()
        p_log_p = einsum("ij,ij->ij", self.logits, self.probs)
        p_log_p = torch.where(self.mask, p_log_p, torch.zeros_like(p_log_p))
        return -reduce(p_log_p, "b a -> b", "sum", b=self.batch_size, a=self.n_actions)


class ResidualBlock(nn.Module):
    """Residual block with batch normalization."""
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
    """Spatial attention module to focus on important areas."""
    def __init__(self, in_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, 1, kernel_size=1)
        
    def forward(self, x):
        attention = torch.sigmoid(self.conv(x))
        return x * attention.expand_as(x)


class FeatureFusion(nn.Module):
    """Fuses CNN and MLP features."""
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
    """
    Action-specific head that produces a logit for each action.
    With the new absolute action space, num_actions is now 10.
    """
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
# Feature extractor for ByteFight
# ----------------------------
class ByteFightFeaturesExtractor(BaseFeaturesExtractor):
    """
    A features extractor for ByteFight that processes a 9-channel board image and
    a 15-dimensional scalar feature vector. It fuses these into a 256-dim output.
    """
    def __init__(self, observation_space: spaces.Dict, features_dim: int = 256):
        super().__init__(observation_space, features_dim=features_dim)

        # CNN processing the board image (9 channels, 64x64)
        self.cnn = nn.Sequential(
            nn.Conv2d(9, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            ResidualBlock(32),
            nn.MaxPool2d(2),  # 64x64 -> 32x32
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            SpatialAttention(64),
            nn.MaxPool2d(2),  # 32x32 -> 16x16
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            SpatialAttention(128),
            nn.MaxPool2d(2),  # 16x16 -> 8x8
            nn.Flatten(),
            nn.Linear(128 * 8 * 8, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.1),
        )

        # MLP for scalar features (15 dimensions)
        self.mlp = nn.Sequential(
            nn.Linear(15, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Dropout(0.1),
        )

        # Fusion module to combine CNN and MLP outputs
        self.fusion = FeatureFusion(256, 64, features_dim)

    def forward(self, obs: dict) -> torch.Tensor:
        board_img = obs["board_image"]  # [B, 9, 64, 64]
        scalar_feats = obs["features"]  # [B, 15]
        cnn_out = self.cnn(board_img)   # [B, 256]
        mlp_out = self.mlp(scalar_feats)  # [B, 64]
        fused = self.fusion(cnn_out, mlp_out)  # [B, 256]
        return fused


# ----------------------------
# Final policy using a masked categorical distribution
# ----------------------------
class ByteFightMaskedPolicy(ActorCriticPolicy):
    """
    Actor-Critic policy for ByteFight using absolute actions.
    Uses ByteFightFeaturesExtractor and a small MLP.
    The final actor head uses ActionSpecificHead with 10 outputs.
    """
    def __init__(self, *args, **kwargs):
        kwargs["features_extractor_class"] = ByteFightFeaturesExtractor
        super().__init__(*args, **kwargs)
        # Replace final actor head with our ActionSpecificHead.
        # self.action_space.n is now 10.
        self.action_net = ActionSpecificHead(self.mlp_extractor.latent_dim_pi, self.action_space.n)

    def _build_mlp_extractor(self) -> None:
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
        # Ensure the mask shape matches logits shape: [B, 10]
        assert action_mask.shape == logits.shape, f"Mask shape {action_mask.shape} vs logits {logits.shape}"
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
