import os
import time
import numpy as np
import torch
import json
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.monitor import Monitor
import gymnasium as gym

# ------------------------------------------------------------------------------
# Custom imports (modify these paths/names to match your actual files)
from bytefight_env import ByteFightSnakeEnv
from custom_policy3 import ByteFightMaskedPolicy, ByteFightFeaturesExtractor
from normalizer import RunningNormalizer
from opponent_pool import OpponentPool
from opp_controller import OppController
# ------------------------------------------------------------------------------

# Training parameters
RANDOM_SEED = 42
NUM_ENV = 8 #8
TOTAL_TIMESTEPS = 2_500_000
ITS = 55
STEPS_PER_ITER = int(TOTAL_TIMESTEPS / ITS)
SAVE_FREQ = 50_000
LOGS_DIR = "./logs"
MODELS_DIR = "./models"
SNAPSHOT_DIR = os.path.join(MODELS_DIR, "league_snapshots")
#CHECKPOINT_PATH = os.path.join(MODELS_DIR, "bytefight_ppo_600000_steps.zip")  # If you have a pretrained model
CHECKPOINT_PATH = os.path.join(MODELS_DIR, "none")  # If you have a pretrained model

LOG_LEVEL = 1

# Create directories
os.makedirs(LOGS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(SNAPSHOT_DIR, exist_ok=True)
os.makedirs(os.path.join(LOGS_DIR, "eval"), exist_ok=True)

############################################
# Dummy environment for initialization
############################################
class ByteFightDummyEnv(gym.Env):
    """
    A trivial environment used ONLY to let SB3 create or load PPO
    with the correct observation/action shapes for ByteFight. 
    We will run 8 copies of this dummy env in parallel 
    so the model sees n_envs=8 from the start.
    """
    def __init__(self, seed=42):
        super().__init__()
        # 10 discrete actions, as in ByteFight
        self.action_space = gym.spaces.Discrete(10)
        # The same Dict observation space you have in your real env
        self.observation_space = gym.spaces.Dict({
            "board_image": gym.spaces.Box(low=0, high=1, shape=(9, 64, 64), dtype=np.float32),
            "features": gym.spaces.Box(low=-1e6, high=1e6, shape=(15,), dtype=np.float32),
            "action_mask": gym.spaces.Box(low=0, high=1, shape=(10,), dtype=np.uint8)
        })
        self.seed_value = seed

    def reset(self, *, seed=None, options=None):
        obs = {
            "board_image": np.zeros((9, 64, 64), dtype=np.float32),
            "features": np.zeros(15, dtype=np.float32),
            "action_mask": np.ones(10, dtype=np.uint8),
        }
        return obs, {}

    def step(self, action):
        # Return dummy obs, no real transitions
        obs = {
            "board_image": np.zeros((9, 64, 64), dtype=np.float32),
            "features": np.zeros(15, dtype=np.float32),
            "action_mask": np.ones(10, dtype=np.uint8),
        }
        reward = 0.0
        done = True  # immediately end
        info = {}
        return obs, reward, done, False, info

############################################
# Real self-play environment factories
############################################
def make_selfplay_env(rank, seed, opponent_pool, obs_normalizer):
    """
    Creates a ByteFightSnakeEnv that references your OpponentPool and RunningNormalizer.
    On reset, it samples an opponent from the pool.
    We set use_opponent=True to do self-play.
    """
    def _init():
        env = ByteFightSnakeEnv(
            map_names=["empty", "empty_large", "ssspline"],  # example
            opponent_pool=opponent_pool,
            obs_normalizer=obs_normalizer,
            render_mode=None,
            use_opponent=True,
            verbose=False
        )
        env = Monitor(env, os.path.join(LOGS_DIR, f"env_{rank}"))
        env.reset(seed=seed + rank)
        return env
    set_random_seed(seed)
    return _init

############################################
# Main training script
############################################
def main():
    # 1) Create shared normalizer and an empty OpponentPool
    obs_normalizer = RunningNormalizer(dim=15, clip_value=5.0)
    opponent_pool = OpponentPool(snapshot_dir=SNAPSHOT_DIR, initial_policy=None, initial_rating=1000)

    # 2) Create 8 parallel copies of the DummyEnv for model init/loading
    def dummy_env_fn(rank):
        def _init():
            return ByteFightDummyEnv(seed=RANDOM_SEED + rank)
        return _init

    dummy_subproc_env = SubprocVecEnv([dummy_env_fn(i) for i in range(NUM_ENV)])
    # => Now dummy_subproc_env.num_envs == 8

    # 3) Create or load the main PPO model with that 8-env dummy
    print("Creating/Loading PPO agent with dummy 8-env to ensure n_envs=8...")
    policy_kwargs = {
        "net_arch": dict(pi=[256, 128], vf=[256, 128]),
        "activation_fn": torch.nn.ReLU,
        "features_extractor_class": ByteFightFeaturesExtractor,
        "features_extractor_kwargs": {"features_dim": 256},
    }

    if os.path.exists(CHECKPOINT_PATH):
        print(f"Loading model from {CHECKPOINT_PATH}...")
        model = PPO.load(
            CHECKPOINT_PATH,
            env=dummy_subproc_env,
            custom_objects={
                "policy_class": ByteFightMaskedPolicy,
                "features_extractor_class": ByteFightFeaturesExtractor
            }
        )
        print("Pretrained model loaded successfully!")
    else:
        print("No pretrained model found; creating new model from scratch.")
        model = PPO(
            policy=ByteFightMaskedPolicy,
            env=dummy_subproc_env,
            learning_rate=3e-4,
            n_steps=2048,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            normalize_advantage=True,
            ent_coef=0.01,
            vf_coef=0.5,
            max_grad_norm=0.5,
            use_sde=False,
            sde_sample_freq=-1,
            target_kl=None,
            tensorboard_log=LOGS_DIR,
            policy_kwargs=policy_kwargs,
            verbose=LOG_LEVEL,
            seed=RANDOM_SEED,
            device="auto"
        )

    # 4) Add the main policy to the pool so it's not empty
    opponent_pool.add_snapshot(model.policy, rating=1000, step=0)

    # 5) Create the real self-play environment (also 8-env)
    vec_env = SubprocVecEnv([make_selfplay_env(i, RANDOM_SEED, opponent_pool, obs_normalizer)
                             for i in range(NUM_ENV)])
    
    # 6) Switch model from dummy to real env
    #    n_envs=8 => no mismatch
    model.set_env(vec_env)

    # 7) Optional: checkpoint callback
    checkpoint_callback = CheckpointCallback(
        save_freq=SAVE_FREQ // NUM_ENV,
        save_path=MODELS_DIR,
        name_prefix="bytefight_ppo",
        save_replay_buffer=True,
        save_vecnormalize=True,
    )

    # 8) Training loop
    iteration = 0

    while iteration * STEPS_PER_ITER < TOTAL_TIMESTEPS:
        print(f"Iteration {iteration+1}/{ITS}: training for {STEPS_PER_ITER} timesteps...")
        model.learn(
            total_timesteps=STEPS_PER_ITER,
            progress_bar=True,
            callback=[checkpoint_callback],
        )

        iteration += 1
        current_step = iteration * STEPS_PER_ITER

        # Add a new snapshot to the pool
        opponent_pool.add_snapshot(model.policy, rating=1000, step=current_step)
        print(f"[TRAIN] After {current_step} timesteps, pool size = {len(opponent_pool.snapshots)}")

        model_path = f"{MODELS_DIR}/league_its/it{iteration}"
        model.save(model_path)

    # Final save
    final_model_path = os.path.join(MODELS_DIR, "bytefight_ppo_league_final")
    model.save(final_model_path)
    print(f"Training complete. Final model saved: {final_model_path}")

    # Clean up
    vec_env.close()
    dummy_subproc_env.close()

if __name__ == "__main__":
    main()
