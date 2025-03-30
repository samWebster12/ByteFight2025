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

def main():
    saved_boards_path = "./saved_boards"
    if os.path.exists(saved_boards_path):
        shutil.rmtree(saved_boards_path)
    os.makedirs(saved_boards_path)

    workspace_path = os.path.join(os.getcwd(), "workspace")
    if workspace_path not in sys.path:
        sys.path.insert(0, workspace_path)

    def get_num_samples():
        return len([f for f in os.listdir("./saved_boards") if f.endswith(".npz")])

    players = ["sample_player", "greedy"]

    num_samples = 100

    while get_num_samples() < num_samples:
        a_name = "agent"
        b_name = random.choice(players)
        submission_dir = os.path.join(os.getcwd(), "workspace") 
        a_sub = os.path.join(submission_dir, a_name)
        b_sub = os.path.join(submission_dir, b_name)

        run_match(a_sub, b_sub, a_name, b_name, "pillars")

if __name__ == "__main__":
    main()
    print("Finished collecting data.")