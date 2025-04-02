import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
import torch
import os
from run_game import run_match
import sys
import shutil
from game import player_board
from game.enums import Action, Cell, Result
from collections.abc import Callable
import random
import numpy as np
import uuid
import os
import utils


def main():
    sys.path.insert(0, os.path.join(os.getcwd(), "workspace"))

    players = ["cnn", "deeprl2"]
    num_games = 5000
    games_per_save = 50

    save_dir = "data"
    os.makedirs(save_dir, exist_ok=True)

    X = []
    Y = []

    for game_idx in range(1, num_games + 1):
        a_name = random.choice(players)
        b_name = random.choice(players)
        submission_dir = os.path.join(os.getcwd(), "workspace") 
        a_sub = os.path.join(submission_dir, a_name)
        b_sub = os.path.join(submission_dir, b_name)

        player_a_moves, player_b_moves, label = run_match(a_sub, b_sub, a_name, b_name, "pillars")

        # Label for each player
        label_a = 1.0 if label == "A" else -1.0 if label == "B" else 0.0
        label_b = -label_a

        # Use every board state
        for state in player_a_moves:
            X.append(utils.board_to_features(state))
            Y.append(label_a)

        for state in player_b_moves:
            X.append(utils.board_to_features(state))
            Y.append(label_b)
        
        # Save every `games_per_save` games
        if game_idx % games_per_save == 0:
            chunk_id = f"{game_idx:04d}"
            out_path = os.path.join(save_dir, f"chunk_{chunk_id}.npz")
            np.savez_compressed(out_path, x=np.array(X), y=np.array(Y))
            print(f"[✓] Saved {len(X)} examples to {out_path}")
            X = []
            Y = []

    print("Finished collecting all games.")



if __name__ == "__main__":
    main()
    print("Finished collecting data.")