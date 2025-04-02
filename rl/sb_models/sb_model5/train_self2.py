import json
import os
import random
import copy
import numpy as np
import gymnasium as gym

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback, EvalCallback
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
from stable_baselines3.common.utils import set_random_seed

# Import your custom environment and feature extractor.
from bytefight_env_single import SingleProcessByteFightEnv
from custom_extractor import CNNMaskExtractor

# ===========================
# Self-Play Opponent Wrapper
# ===========================
class SelfPlayOpponent:
    """
    Wraps a model to use it as an opponent.
    The play() method takes a PlayerBoard and returns a turn (a list of actions).
    """
    def __init__(self, model):
        self.model = model

    def play(self, board, time_left):
        obs = build_observation(board)
        action, _ = self.model.predict(obs, deterministic=True)
        return [action]

# ===========================
# Observation Builder Helper
# ===========================
def build_observation(pb):
    """
    Build an observation dictionary from a PlayerBoard instance.
    Mimics the logic from the environment's _make_observation() method.
    """
    dim_x = pb.game_board.map.dim_x
    dim_y = pb.game_board.map.dim_y
    channels = np.zeros((9, dim_y, dim_x), dtype=np.uint8)

    # Channel 0: Walls (Cell.WALL == 1)
    wall_mask = pb.get_wall_mask()
    channels[0] = np.where(wall_mask == 1, 255, 0)
    # Channel 1: Apples (Cell.APPLE == 2)
    apple_mask = pb.get_apple_mask()
    channels[1] = np.where(apple_mask == 2, 255, 0)
    # Channels 2 & 3: My snake head/body (PLAYER_HEAD==3, PLAYER_BODY==4)
    a_snake_mask = pb.get_snake_mask(my_snake=True, enemy_snake=False)
    channels[2] = np.where(a_snake_mask == 3, 255, 0)
    channels[3] = np.where(a_snake_mask == 4, 255, 0)
    # Channels 4 & 5: Enemy snake head/body (ENEMY_HEAD==5, ENEMY_BODY==6)
    b_snake_mask = pb.get_snake_mask(my_snake=False, enemy_snake=True)
    channels[4] = np.where(b_snake_mask == 5, 255, 0)
    channels[5] = np.where(b_snake_mask == 6, 255, 0)
    # Channel 6: My traps.
    my_trap_mask = pb.get_trap_mask(my_traps=True, enemy_traps=False)
    channels[6] = np.where(my_trap_mask > 0, 255, 0)
    # Channel 7: Enemy traps.
    enemy_trap_mask = pb.get_trap_mask(my_traps=False, enemy_traps=True)
    channels[7] = np.where(enemy_trap_mask > 0, 255, 0)
    # Channel 8: Portals.
    try:
        portal_mask = pb.get_portal_mask(descriptive=False)
        if portal_mask.ndim == 3:
            portal_mask = portal_mask[:, :, 0]
        channels[8] = np.where(portal_mask == 1, 255, 0)
    except Exception as e:
        channels[8] = 0

    image = np.zeros((9, 64, 64), dtype=np.uint8)
    image[:, :dim_y, :dim_x] = channels[:, :dim_y, :dim_x]

    # Build the action mask (9-dimensional)
    mask = np.zeros((9,), dtype=np.uint8)
    valid_moves = []
    for move in range(8):
        if pb.is_valid_move(move):
            valid_moves.append(move)
    for m in valid_moves:
        mask[m] = 1
    if pb.is_valid_trap():
        mask[8] = 1
    if mask.sum() == 0:
        mask[0] = 1

    return {"image": image, "action_mask": mask}

# ========================================================
# League Self-Play Callback for Opponent and League Updates
# ========================================================
class LeagueSelfPlayUpdateCallback(BaseCallback):
    """
    Maintains a league of past best models (capped at max_league_size) and updates
    each environment’s opponent by sampling from this league. Every update_freq timesteps,
    a candidate is added. If the league is full, the candidate replaces the worst performing
    model if its performance is higher.
    """
    def __init__(self, update_freq: int, league_add_freq: int, eval_env, max_league_size: int = 5, n_eval_episodes: int = 3, verbose=0):
        super(LeagueSelfPlayUpdateCallback, self).__init__(verbose)
        self.update_freq = update_freq          # timesteps between opponent updates
        self.league_add_freq = league_add_freq  # timesteps between adding a candidate
        self.eval_env = eval_env                # environment to evaluate candidate performance
        self.max_league_size = max_league_size
        self.n_eval_episodes = n_eval_episodes
        self.league_models = []  # list of tuples: (model, performance)

    def evaluate_model(self, model):
        total_reward = 0.0
        for _ in range(self.n_eval_episodes):
            obs, _ = self.eval_env.reset()
            done = False
            episode_reward = 0.0
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, done, _, _ = self.eval_env.step(action)
                episode_reward += reward
            total_reward += episode_reward
        return total_reward / self.n_eval_episodes

    def _on_training_start(self) -> None:
        from stable_baselines3 import PPO
        # Load the pretrained model as the initial league member.
        initial_model = PPO.load("bytefight_logs/ppo_bytefight_final", env=self.training_env)
        perf = self.evaluate_model(initial_model)
        self.league_models.append((initial_model, perf))
        if self.verbose > 0:
            print(f"Initial league size: {len(self.league_models)} with performance: {perf:.2f}")

    def _on_step(self) -> bool:
        # Update the opponent every update_freq timesteps.
        if self.num_timesteps % self.update_freq < self.training_env.num_envs:
            opp_model, _ = random.choice(self.league_models)
            new_opponent_params = opp_model.get_parameters()
            try:
                self.training_env.env_method("update_opponent_model", new_opponent_params)
                if self.verbose > 0:
                    print(f"[Timestep {self.num_timesteps}] Updated opponent from league (league size: {len(self.league_models)}).")
            except Exception as e:
                print(f"Error updating opponent in workers: {e}")

        # Every league_add_freq timesteps, evaluate the current model and add it as a candidate.
        if self.num_timesteps % self.league_add_freq < self.training_env.num_envs:
            try:
                from stable_baselines3 import PPO
                # Get candidate parameters from the current model.
                params = self.model.get_parameters()
                policy_kwargs = self.model.policy_kwargs  # assume same kwargs were used
                candidate_model = PPO("MultiInputPolicy", self.training_env, policy_kwargs=policy_kwargs, device="cpu")
                candidate_model.set_parameters(params)
                candidate_perf = self.evaluate_model(candidate_model)
                if len(self.league_models) < self.max_league_size:
                    self.league_models.append((candidate_model, candidate_perf))
                    if self.verbose > 0:
                        print(f"[Timestep {self.num_timesteps}] Added candidate to league. League size: {len(self.league_models)}; candidate performance: {candidate_perf:.2f}")
                else:
                    # Find the worst-performing member.
                    worst_index = min(range(len(self.league_models)), key=lambda i: self.league_models[i][1])
                    worst_perf = self.league_models[worst_index][1]
                    if candidate_perf > worst_perf:
                        self.league_models[worst_index] = (candidate_model, candidate_perf)
                        if self.verbose > 0:
                            print(f"[Timestep {self.num_timesteps}] Replaced worst model (perf {worst_perf:.2f}) with candidate (perf {candidate_perf:.2f}).")
            except Exception as e:
                if self.verbose > 0:
                    print(f"Error adding candidate to league: {e}")
        return True



# =======================
# Environment Creation
# =======================
def make_env(map_string, submission_dir, rank=0):
    def _init():
        env = SingleProcessByteFightEnv(
            map_string=map_string,
            opponent_module="sample_player",  # will be replaced via callback
            submission_dir=submission_dir,
            render_mode=None
        )
        return env
    set_random_seed(rank)
    return _init

def main():
    with open("maps.json") as f:
        map_dict = json.load(f)
    map_string = map_dict["empty"]
    submission_dir = "../../../workspace"
    log_dir = "bytefight_league_logs"
    os.makedirs(log_dir, exist_ok=True)
    
    n_envs = 4
    env_fns = [make_env(map_string, submission_dir, rank=i) for i in range(n_envs)]
    env = SubprocVecEnv(env_fns)
    env = VecMonitor(env)
    
    eval_env = SingleProcessByteFightEnv(
        map_string=map_string,
        opponent_module="sample_player",
        submission_dir=submission_dir,
        render_mode="human"
    )
    
    checkpoint_callback = CheckpointCallback(
        save_freq=25000 // n_envs,
        save_path=f"{log_dir}/checkpoints/",
        name_prefix="ppo_bytefight_league",
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
    # Set up the league callback with desired frequencies and a maximum league size.
    league_callback = LeagueSelfPlayUpdateCallback(
        update_freq=10000,         # Update opponent every 5,000 timesteps
        league_add_freq=20000,      # Evaluate and try adding a candidate every 10,000 timesteps
        eval_env=eval_env,
        max_league_size=5,          # Cap the league at 5 members
        n_eval_episodes=3,          # Use 3 episodes for evaluation
        verbose=1
    )
    
    try:
        import tensorboard  # noqa: F401
        tensorboard_log = f"{log_dir}/tensorboard/"
    except ImportError:
        print("TensorBoard not installed. Disabling tensorboard_log.")
        tensorboard_log = None
    
    policy_kwargs = {
        "features_extractor_class": CNNMaskExtractor,
        "features_extractor_kwargs": {"features_dim": 512},
        "net_arch": {"pi": [128, 128], "vf": [128, 128]}
    }
    
    model = PPO.load("bytefight_logs/ppo_bytefight_final", env=env,
                     tensorboard_log=tensorboard_log, policy_kwargs=policy_kwargs)
    
    total_timesteps = 1000000
    model.learn(
        total_timesteps=total_timesteps,
        callback=[checkpoint_callback, eval_callback, league_callback],
        progress_bar=True
    )
    
    final_model_path = f"{log_dir}/ppo_bytefight_league_final2"
    model.save(final_model_path)
    print("Final model saved to", final_model_path)

    # 12) Evaluate the final model.
    test_env = SingleProcessByteFightEnv(
        map_string=map_string,
        opponent_module="sample_player",
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
