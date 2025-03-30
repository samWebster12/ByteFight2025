import numpy as np

print("Weight Layers:")
weights = np.load("stable_baselines_weights.npz")
print("Keys in the .npz file:", weights.files)

from stable_baselines3 import PPO
print("\nAll Weight Layers Hierarchy")
model = PPO.load("bytefight_logs/ppo_bytefight_final.zip")
print(model.policy)

model_path = "bytefight_logs/ppo_bytefight_final.zip"
    
# Load the model
model = PPO.load(model_path)

# Print the policy architecture
print("Model Policy Architecture:")
print(model.policy)