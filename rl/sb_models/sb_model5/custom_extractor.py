import torch as th
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

class CNNMaskExtractor(BaseFeaturesExtractor):
    """
    Custom feature extractor that processes both the image and the action_mask.
    The image is passed through a deeper CNN and the mask through a small MLP.
    Their features are then concatenated and passed through a final linear layer
    to produce a feature vector of size `features_dim`.
    """
    def __init__(self, observation_space, features_dim=512):
        super(CNNMaskExtractor, self).__init__(observation_space, features_dim)
        image_space = observation_space["image"]
        mask_space = observation_space["action_mask"]
        n_input_channels = image_space.shape[0]

        # Build a deeper CNN for processing the image
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten()
        )
        # Determine the output size of the CNN
        with th.no_grad():
            sample_img = th.as_tensor(image_space.sample()[None]).float()
            cnn_output_dim = self.cnn(sample_img).shape[1]

        # Process the action mask with a small MLP
        self.mask_mlp = nn.Sequential(
            nn.Linear(mask_space.shape[0], 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU()
        )
        mask_feature_dim = 32

        # Combine the CNN and mask features
        combined_dim = cnn_output_dim + mask_feature_dim

        # Final layer to obtain the desired feature dimension
        self.combined_layer = nn.Sequential(
            nn.Linear(combined_dim, features_dim),
            nn.ReLU()
        )

    def forward(self, observations):
        # Normalize the image input from [0,255] to [0,1]
        img = observations["image"].float() / 255.0
        cnn_features = self.cnn(img)
        
        # Process the action mask (convert to float)
        mask = observations["action_mask"].float()
        mask_features = self.mask_mlp(mask)
        
        # Concatenate features from the CNN and mask MLP
        combined = th.cat((cnn_features, mask_features), dim=1)
        features = self.combined_layer(combined)
        return features
