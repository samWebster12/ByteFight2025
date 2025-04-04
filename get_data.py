import os
import sys
import ray
import random
import numpy as np
from run_game import run_match
import utils

ray.init()  # You can add resources like: ray.init(num_cpus=8)

players = ["cnn1", "cnn2", "cnn3", "deeprl2"]
num_games = 10000
games_per_save = 250
save_dir = "data"
os.makedirs(save_dir, exist_ok=True)

# Ray remote function
@ray.remote(num_cpus=2)
def play_and_process(a_name, b_name):
    submission_dir = os.path.join(os.getcwd(), "workspace")
    try:
        player_a_moves, player_b_moves, label = run_match("workspace", "workspace", a_name, b_name, "pillars")
    except Exception as e:
        print(f"[ERROR] Match {a_name} vs {b_name} failed: {e}")
        return [], []

    label_a = 1.0 if label == "A" else -1.0 if label == "B" else 0.0
    label_b = -label_a

    # Sample every 5th move
    # player_a_moves = player_a_moves[::5]
    # player_b_moves = player_b_moves[::5]

    def process_player_moves(moves, label):
        T = len(moves)
        if T == 0:
            return [], []
        elif T == 1:
            alpha = 1.0  # fallback if only one move
            state = moves[0]
            features = utils.board_to_features(state)
            eval_score = np.clip(utils.evaluate_position(state), -100, 100) / 100.0
            blended_label = alpha * label + (1 - alpha) * eval_score
            return [features], [blended_label]

        X, Y = [], []
        for t, state in enumerate(moves):
            alpha = (t / (T - 1)) ** 2
            eval_score = np.clip(utils.evaluate_position(state), -100, 100) / 100.0
            hybrid_label = alpha * label + (1 - alpha) * eval_score
            X.append(utils.board_to_features(state))
            Y.append(hybrid_label)
        return X, Y

    Xa, Ya = process_player_moves(player_a_moves, label_a)
    Xb, Yb = process_player_moves(player_b_moves, label_b)

    return Xa + Xb, Ya + Yb


def main():
    X_total, Y_total = [], []
    futures = []

    for game_idx in range(1, num_games + 1):
        a_name = random.choice(players)
        b_name = random.choice(players)
        print(f"[INFO] Submitting game {game_idx} between {a_name} and {b_name}")
        futures.append(play_and_process.remote(a_name, b_name))

        # Every `games_per_save` games, wait and write results
        if len(futures) == games_per_save:
            results = ray.get(futures)
            futures.clear()

            for X_batch, Y_batch in results:
                X_total.extend(X_batch)
                Y_total.extend(Y_batch)

            chunk_id = f"{game_idx:04d}"
            out_path = os.path.join(save_dir, f"chunk_{chunk_id}.npz")
            np.savez_compressed(out_path, x=np.array(X_total), y=np.array(Y_total))
            print(f"[✓] Saved {len(X_total)} examples to {out_path}")

            X_total = []
            Y_total = []

    # Final save
    if X_total:
        chunk_id = f"{num_games:04d}"
        out_path = os.path.join(save_dir, f"chunk_{chunk_id}.npz")
        np.savez_compressed(out_path, x=np.array(X_total), y=np.array(Y_total))
        print(f"[✓] Saved final {len(X_total)} examples to {out_path}")

    print("Finished collecting all games.")

if __name__ == "__main__":
    main()
