import numpy as np
import os, sys

from .base_layers import Conv2D, ReLU, Flatten, Linear, Tanh, Sequential

###############################################
# Composite Module: NatureCNN
###############################################
class NatureCNN:
    """
    Recreates the NatureCNN module, which processes image inputs.
    """
    def __init__(self, in_channels):
        # Define the CNN block
        self.cnn = Sequential([
            Conv2D(in_channels, 32, kernel_size=8, stride=4, padding=0),
            ReLU(),
            Conv2D(32, 64, kernel_size=4, stride=2, padding=0),
            ReLU(),
            Conv2D(64, 64, kernel_size=3, stride=1, padding=0),
            ReLU(),
            Flatten()
        ])
        # The CNN output size depends on the input dimensions
        # For a 84x84 input with the above CNN structure, output size is 3136
        # For a 64x64 input, output size is 1024
        self.linear = Sequential([
            Linear(1024, 256),  # This will be updated when loading weights
            ReLU()
        ])
    
    def load_weights(self, weights, prefix):
        """
        Loads weights from the provided dictionary `weights`.
        """
        # Load weights for the CNN block:
        self.cnn.layers[0].load_weights(weights[f'{prefix}.cnn.0.weight'], weights[f'{prefix}.cnn.0.bias'])
        self.cnn.layers[2].load_weights(weights[f'{prefix}.cnn.2.weight'], weights[f'{prefix}.cnn.2.bias'])
        self.cnn.layers[4].load_weights(weights[f'{prefix}.cnn.4.weight'], weights[f'{prefix}.cnn.4.bias'])
        
        # Determine the correct input size for the linear layer from the weight shape
        linear_weight = weights[f'{prefix}.linear.0.weight']
        input_features = linear_weight.shape[1]
        
        # Update the linear layer if needed
        if self.linear.layers[0].in_features != input_features:
            print(f"Updating {prefix} linear layer input features from {self.linear.layers[0].in_features} to {input_features}")
            self.linear.layers[0].in_features = input_features
        
        # Load weights for the linear block:
        self.linear.layers[0].load_weights(weights[f'{prefix}.linear.0.weight'], weights[f'{prefix}.linear.0.bias'])
    
    def forward(self, x):
        x = self.cnn.forward(x)
        x = self.linear.forward(x)
        return x

###############################################
# Composite Module: CombinedExtractor
###############################################
class CombinedExtractor:
    """
    Combines multiple extractors for different observation modalities.
    """
    def __init__(self, extractors):
        self.extractors = extractors
    
    def forward(self, observations):
        outputs = []
        # Apply each extractor to its corresponding observation.
        for key, extractor in self.extractors.items():
            out = extractor.forward(observations[key])
            outputs.append(out)
        # Concatenate along feature dimension (assumes outputs are (batch_size, features))
        return np.concatenate(outputs, axis=1)

###############################################
# Composite Module: MlpExtractor
###############################################
class MlpExtractor:
    """
    Creates two separate MLPs (one for policy and one for value) from a combined input.
    
    Updated architecture:
      - policy_net: Sequential(
                      Linear(in_features=265, out_features=128, bias=True),
                      Tanh(),
                      Linear(in_features=128, out_features=128, bias=True),
                      Tanh()
                    )
      - value_net: Sequential(
                      Linear(in_features=265, out_features=128, bias=True),
                      Tanh(),
                      Linear(in_features=128, out_features=128, bias=True),
                      Tanh()
                    )
    """
    def __init__(self, in_features=265, hidden_dim=128, out_dim=128):
        # Updated hidden_dim and out_dim to 128 instead of 64
        self.policy_net = Sequential([
            Linear(in_features, hidden_dim),
            Tanh(),
            Linear(hidden_dim, out_dim),
            Tanh()
        ])
        self.value_net = Sequential([
            Linear(in_features, hidden_dim),
            Tanh(),
            Linear(hidden_dim, out_dim),
            Tanh()
        ])
    
    def load_weights(self, weights):
        """
        Loads weights for both the policy and value MLPs.
        """
        # Policy net weights:
        self.policy_net.layers[0].load_weights(weights['mlp_extractor.policy_net.0.weight'],
                                                 weights['mlp_extractor.policy_net.0.bias'])
        self.policy_net.layers[2].load_weights(weights['mlp_extractor.policy_net.2.weight'],
                                                 weights['mlp_extractor.policy_net.2.bias'])
        # Value net weights:
        self.value_net.layers[0].load_weights(weights['mlp_extractor.value_net.0.weight'],
                                                weights['mlp_extractor.value_net.0.bias'])
        self.value_net.layers[2].load_weights(weights['mlp_extractor.value_net.2.weight'],
                                                weights['mlp_extractor.value_net.2.bias'])
    
    def forward(self, x):
        """
        Returns a tuple (policy_features, value_features).
        """
        return self.policy_net.forward(x), self.value_net.forward(x)

###############################################
# MultiInputActorCriticPolicy
###############################################
class MultiInputActorCriticPolicy:
    """
    A simplified composite of the overall policy architecture.
    
    Updated to use:
    - 128-unit hidden layers in the MLP extractor (instead of 64)
    - 128-unit input to the action_net and value_net (instead of 64)
    """
    def __init__(self):
        # Create extractors for policy and value branches.
        self.pi_features_extractor = CombinedExtractor({
            "action_mask": Flatten(),
            "image": NatureCNN(in_channels=9)
        })
        self.vf_features_extractor = CombinedExtractor({
            "action_mask": Flatten(),
            "image": NatureCNN(in_channels=9)
        })
        # MLP extractor with 128-unit hidden layers
        self.mlp_extractor = MlpExtractor(in_features=265, hidden_dim=128, out_dim=128)
        # Final linear layers with updated input size (128 instead of 64)
        self.action_net = Linear(128, 9)   # Produces logits for 9 actions.
        self.value_net = Linear(128, 1)    # Produces a scalar value.
    
    def load_weights(self, weights):
        """
        Loads weights for all composite modules from a weights dictionary.
        """
        # Load weights for policy branch NatureCNN:
        self.pi_features_extractor.extractors["image"].load_weights(weights, "pi_features_extractor.extractors.image")
        # Load weights for value branch NatureCNN:
        self.vf_features_extractor.extractors["image"].load_weights(weights, "vf_features_extractor.extractors.image")
        # Load MLP extractor weights:
        self.mlp_extractor.load_weights(weights)
        # Load final layers:
        self.action_net.load_weights(weights['action_net.weight'], weights['action_net.bias'])
        self.value_net.load_weights(weights['value_net.weight'], weights['value_net.bias'])
    
    def forward(self, observation):
        """
        Forward pass through the policy network.
        """
        # Process the observation through the policy and value extractors:
        pi_features = self.pi_features_extractor.forward(observation)
        vf_features = self.vf_features_extractor.forward(observation)
        
        # Pass the combined features through the MLP extractor:
        pi_mlp, vf_mlp = self.mlp_extractor.forward(pi_features)
        
        # Final outputs:
        logits = self.action_net.forward(pi_mlp)
        
        # Apply action mask if provided
        if 'action_mask' in observation:
            mask = observation['action_mask']
            # Apply mask by setting impossible actions to a large negative value
            logits = np.where(mask > 0, logits, -1e8)
        
        value = self.value_net.forward(vf_mlp)
        return logits, value