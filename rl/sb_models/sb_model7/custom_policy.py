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
        One approach is to do the same “-inf” trick at the time of sampling.
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


class ByteFightFeaturesExtractor(BaseFeaturesExtractor):
    """
    A custom SB3 features extractor that:
      - Takes a multi-input Dict obs with 
          "board_image": (9,32,32),
          "features": (15,),
          "action_mask": (7,). 
        (We won't embed the action_mask here, 
         it goes directly to the policy distribution logic.)
      - Applies a CNN to board_image and an MLP to the 15-dim features
      - Concatenates both embeddings into a single output vector
    """
    def __init__(
        self,
        observation_space: spaces.Dict,
        cnn_output_dim=256,
        mlp_hidden_dim=64
    ):
        # The final output of this extractor is cnn_output_dim + mlp_hidden_dim
        super().__init__(observation_space, features_dim=(cnn_output_dim + mlp_hidden_dim))

        # 1) Define your CNN for the board_image:
        # We'll do a small 3-layer example.
        self.cnn = nn.Sequential(
            nn.Conv2d(9, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        # 2) Define an MLP for the 15 scalar features:
        self.mlp = nn.Sequential(
            nn.Linear(15, mlp_hidden_dim),
            nn.ReLU()
        )
        # That’s all. The final output is the concat of [cnn_out, mlp_out].

    def forward(self, obs: dict) -> torch.Tensor:
        """
        obs is a dict with keys: 'board_image', 'features', 'action_mask'
        We'll extract board_image & features for embedding.
        The policy / distribution code can handle the mask separately.
        """
        board_img = obs["board_image"]          # shape (B, 9, 32, 32)
        scalar_feats = obs["features"]          # shape (B, 15)

        cnn_out = self.cnn(board_img)           # (B, cnn_output_dim)
        mlp_out = self.mlp(scalar_feats)        # (B, mlp_hidden_dim)

        # Concat
        fused = torch.cat([cnn_out, mlp_out], dim=1)  # (B, cnn_out_dim + mlp_out_dim)
        return fused
    

class ByteFightMaskedPolicy(ActorCriticPolicy):
    """
    A custom policy that:
      - uses ByteFightFeaturesExtractor to encode (board_image, features)
      - uses 'action_mask' from the observation to produce a CategoricalMasked distribution
        so invalid actions have probability 0.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(
            *args,
            **kwargs,
        )

    def _build_mlp_extractor(self) -> None:
        """
        Override the default to do nothing because we already have 
        a custom feature extractor that merges CNN + MLP. 
        We just store a small MLP to produce policy/value from the features_extractor output.
        """
        # By default, ActorCriticPolicy calls MlpExtractor on top of the features_extractor.
        # We'll replicate that but using net_arch from self.net_arch if needed.
        self.mlp_extractor = MlpExtractor(
            self.features_dim,  # features_dim from our feature extractor
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
        Build the masked distribution. We'll:
         1) Extract features via self.features_extractor
         2) Pass them through self.mlp_extractor -> (latent_pi, latent_vf)
         3) Get raw policy logits = self.action_net(latent_pi)
         4) Get mask from obs["action_mask"]
         5) Return a CategoricalMasked with that mask
        """
        # 1) Extract features
        features = self.extract_features(obs, self.features_extractor)
        # 2) MLP extractor => separate policy & value latents
        latent_pi, latent_vf = self.mlp_extractor(features)

        # 3) Final policy logits
        # ActorCriticPolicy has self.action_net as the final linear layer for policy
        logits = self.action_net(latent_pi)  # shape (B, n_actions)

        # 4) Retrieve mask from obs ( shape (B, n_actions) ), 
        #    but the user might have shape (B,7). Make sure it's bool or byte.
        action_mask = obs["action_mask"]  # expected shape: (B, n_actions), bool
        #action_mask = action_mask.to(self.device)
        # 5) Build the masked distribution
        assert action_mask.shape == logits.shape, f"Mask shape {action_mask.shape} doesn't match logits shape {logits.shape}"
        dist = CategoricalMasked(logits=logits, mask=action_mask)

        return dist

    def forward_actor(self, obs: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Used internally by SB3 in some places.
        We'll replicate the distribution-building steps 
        but only returning the final policy logits (before softmax).
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