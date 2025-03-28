import torch
import numpy as np
from stable_baselines3 import PPO

if __name__ == "__main__":
    # Load your SB3 model from the .zip
    model = PPO.load("bytefight_logs/ppo_bytefight_final.zip", device="cpu")
    policy = model.policy

    # Dump everything in policy's state_dict() to a single .npz
    # enumerating keys (like 'features_extractor.extractors.image.cnn.0.weight', etc.)
    sd = policy.state_dict()  # an OrderedDict of PyTorch Tensors
    param_dict = {}
    for k, tensor in sd.items():
        param_dict[k] = tensor.cpu().numpy()

    np.savez("weights.npz", **param_dict)
    print("Exported to weights.npz!")