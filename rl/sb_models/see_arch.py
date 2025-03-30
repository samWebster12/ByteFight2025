import numpy as np
from stable_baselines3 import PPO


def print_npz_keys(npz_path):
    weights = np.load(npz_path)
    print("Keys in the .npz file:", weights.files)

def print_model_policy(zip_path):
    model = PPO.load(zip_path)
    print(model.policy)

if __name__ == '__main__':
    npz_path = "sb_model2/weights/weights.npz"
    model_path = "sb_model2/bytefight_logs/ppo_bytefight_final.zip"

    print("NPZ keys:")
    print_npz_keys(npz_path)
    print("\n\nModel Policy Architecture:")
    print_model_policy(model_path)