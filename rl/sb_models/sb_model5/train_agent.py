import json
import os
import random
import gymnasium as gym

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback, BaseCallback
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
from stable_baselines3.common.utils import set_random_seed

# Import the custom CNN+Mask feature extractor
from custom_extractor import CNNMaskExtractor

# Import your custom ByteFight environment
from bytefight_env_single import SingleProcessByteFightEnv
from stable_baselines3.common.env_checker import check_env  # optional

class PrintCallback(BaseCallback):
    """
    Callback to print training progress after episodes complete.
    """
    def __init__(self, verbose=0):
        super(PrintCallback, self).__init__(verbose)
        self.episode_count = 0
        self.episode_rewards = []
        self.win_count = 0
        self.lose_count = 0
        self.tie_count = 0

    def _on_step(self):
        for i, done in enumerate(self.locals["dones"]):
            if done:
                info = self.locals["infos"][i]
                reward = self.locals["rewards"][i]
                self.episode_rewards.append(reward)
                self.episode_count += 1
                winner = info.get("winner")
                if winner == "PLAYER_A":
                    self.win_count += 1
                elif winner == "PLAYER_B":
                    self.lose_count += 1
                elif winner == "TIE":
                    self.tie_count += 1
                if self.episode_count % 10 == 0:
                    last_10 = self.episode_rewards[-10:]
                    avg_reward = sum(last_10) / len(last_10)
                    win_rate = (self.win_count / self.episode_count) * 100
                    #print(f"Episode {self.episode_count}, Avg Reward: {avg_reward:.2f}, Win Rate: {win_rate:.1f}%")
        return True

def make_env(map_string, opponent_module, submission_dir, rank=0):
    def _init():
        env = SingleProcessByteFightEnv(
            map_string=map_string,
            opponent_module=opponent_module,
            submission_dir=submission_dir,
            render_mode=None
        )
        return env
    set_random_seed(rank)
    return _init

def main():
    # 1) Load map configuration
    with open("maps.json") as f:
        map_dict = json.load(f)
    map_string = map_dict["empty"]

    # 2) Opponent configuration
    opponent_module = "custom2"
    submission_dir = "../../../workspace"

    # 3) Logging directory
    log_dir = "bytefight_logs_custom2"
    os.makedirs(log_dir, exist_ok=True)

    # 4) Create parallel environments
    n_envs = 4
    env_fns = [make_env(map_string, opponent_module, submission_dir, rank=i) for i in range(n_envs)]
    env = SubprocVecEnv(env_fns)
    env = VecMonitor(env)

    # 5) Create a separate evaluation environment
    eval_env = SingleProcessByteFightEnv(
        map_string=map_string,
        opponent_module=opponent_module,
        submission_dir=submission_dir
    )

    # 6) Setup callbacks for checkpointing, evaluation, and progress printing
    checkpoint_callback = CheckpointCallback(
        save_freq=25000 // n_envs,
        save_path=f"{log_dir}/checkpoints/",
        name_prefix="ppo_bytefight",
        save_vecnormalize=True
    )
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=f"{log_dir}/best_model/",
        log_path=f"{log_dir}/eval_results/",
        eval_freq=25000 // n_envs,
        deterministic=True,
        render=False,
        n_eval_episodes=5
    )
    print_callback = PrintCallback()

    # 7) Set up TensorBoard logging if available
    try:
        import tensorboard
        tensorboard_log = f"{log_dir}/tensorboard/"
    except ImportError:
        print("TensorBoard not installed. Disabling tensorboard_log.")
        tensorboard_log = None

    # 8) Configure the policy to use the custom CNNMaskExtractor
    policy_kwargs = {
        "features_extractor_class": CNNMaskExtractor,
        "features_extractor_kwargs": {"features_dim": 512},
        "net_arch": dict(pi=[128, 128], vf=[128, 128])
    }

    # 9) Initialize PPO with the custom policy network
    model = PPO(
        policy="MultiInputPolicy",
        env=env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        policy_kwargs=policy_kwargs,
        verbose=1,
        device="cuda",
        tensorboard_log=tensorboard_log
    )

    total_timesteps = 1000000
    model.learn(
        total_timesteps=total_timesteps,
        callback=[checkpoint_callback, eval_callback, print_callback],
        progress_bar=True
    )

    # 10) Save the final model
    final_model_path = f"{log_dir}/ppo_bytefight_final"
    model.save(final_model_path)
    print("Final model saved to", final_model_path)

    # 11) Evaluate the final model
    test_env = SingleProcessByteFightEnv(
        map_string=map_string,
        opponent_module=opponent_module,
        submission_dir=submission_dir,
        render_mode="human"
    )
    for episode in range(3):
        obs, _ = test_env.reset()
        done = False
        total_reward = 0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = test_env.step(action)
            total_reward += reward
        print(f"Episode {episode+1}: winner={info.get('winner')}, total_reward={total_reward:.2f}, turn_count={info.get('turn_count')}")
    test_env.close()

if __name__ == "__main__":
    main()
