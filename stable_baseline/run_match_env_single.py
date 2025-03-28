import json
from bytefight_env_single import SingleProcessByteFightEnv

# Load a map string from maps.json
with open("maps.json") as f:
    map_dict = json.load(f)
map_string = map_dict["cage"]

env = SingleProcessByteFightEnv(
    map_string=map_string,
    opponent_module="sample_player",   # specify your opponent bot folder/module
    submission_dir="./workspace",
    render_mode="human"
)

obs, info = env.reset()
done = False
total_reward = 0
while not done:
    # For testing, sample a random action for the agent
    action = env.action_space.sample()
    obs, reward, done, truncated, info = env.step(action)
    total_reward += reward

print("Match ended.")
print("Winner:", info.get("winner"))
print("Turns:", info.get("turn_count"))
print("Total reward:", total_reward)
env.close()
