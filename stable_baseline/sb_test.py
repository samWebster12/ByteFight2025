import json
from stable_baselines3 import PPO
from rl.sb_model1.bytefight_env_single import SingleProcessByteFightEnv

# Load map from maps.json
with open("maps.json") as f:
    map_dict = json.load(f)
map_string = map_dict["cage"]

# Create the test environment with rendering enabled
env = SingleProcessByteFightEnv(
    map_string=map_string,
    opponent_module="custom2",  # opponent bot folder/module
    submission_dir="./workspace",
    render_mode="human"
)

# Load the trained model (adjust the model path as needed)
model = PPO.load("bytefight_logs/ppo_bytefight_final", env=env)

# Run several test episodes
n_episodes = 5
for episode in range(n_episodes):
    obs, info = env.reset()
    done = False
    total_reward = 0.0
    while not done:
        # Use the trained model to predict an action
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = env.step(action)
        total_reward += reward
    print(f"Episode {episode+1} ended.")
    print("  Winner:", info.get("winner"))
    print("  Turns:", info.get("turn_count"))
    print("  Total reward:", total_reward)
    print("-" * 40)

env.close()
