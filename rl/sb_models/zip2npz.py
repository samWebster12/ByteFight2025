import numpy as np
from stable_baselines3 import PPO

# Load the saved model
model = PPO.load("sb_model2/bytefight_logs/ppo_bytefight_final.zip")
# Get the policy state dictionary
state_dict = model.policy.state_dict()

# Convert tensors to NumPy arrays and save as a .npz file
npz_dict = {key: tensor.cpu().numpy() for key, tensor in state_dict.items()}
np.savez("sb_model2/weights/weights.npz", **npz_dict)