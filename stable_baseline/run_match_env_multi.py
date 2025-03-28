import json
import sys
import os

# Adjust as needed
from bytefight_env_multi import ByteFightTwoBotsEnv

def main():
    with open("maps.json") as f:
        map_dict = json.load(f)
    map_string = map_dict["cage"]

    a_name = "sample_player"
    b_name = "custom2"
    submission_dir = "./workspace"

    env = ByteFightTwoBotsEnv(
        map_string=map_string,
        a_name=a_name,
        b_name=b_name,
        submission_dir=submission_dir,
        render_mode="human"  # environment will print each step
    )

    obs, info = env.reset()

    done = False
    total_reward = 0
    while not done:
        # No extra env.render() call here => environment does it inside step()
        obs, reward, done, truncated, info = env.step(None)
        total_reward += reward

    print("Match ended.")
    print("Winner:", info.get("winner"))
    print("Turns:", info.get("turn_count"))
    print("Total reward (from perspective of env.step calls):", total_reward)

    env.close()

if __name__ == "__main__":
    main()
