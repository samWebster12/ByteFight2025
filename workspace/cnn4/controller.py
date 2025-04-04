from game import player_board
from game.enums import Action
from collections.abc import Callable
import random
import torch
import utils
import copy
import numpy as np
# -----------------------------
# MCTS Node
# -----------------------------
class Node:
    def __init__(self, board, parent=None, action=None):
        self.board = board
        self.parent = parent
        self.action = action
        self.children = []
        self.visits = 0
        self.value_sum = 0
        self.prior = 0

    def is_terminal(self):
        return self.board.is_game_over()

    def is_fully_expanded(self):
        return len(self.children) == len(self.board.get_possible_directions())

    def value(self):
        return self.value_sum / self.visits if self.visits > 0 else 0

    def ucb_score(self, c_puct=1.0):
        if self.parent is None:
            return 0
        return self.value() + c_puct * self.prior * np.sqrt(self.parent.visits) / (1 + self.visits)

    def best_child(self):
        return max(self.children, key=lambda child: child.visits)

    def select_child_ucb(self):
        return max(self.children, key=lambda child: child.ucb_score())

    def expand(self, model, device):
        unexplored = [m for m in self.board.get_possible_directions() if m not in [c.action for c in self.children]]
        if not unexplored:
            return self
        action = random.choice(unexplored)
        if not self.board.is_valid_move(action):
            return self
        new_board = copy.deepcopy(self.board)
        new_board.apply_move(action)
        child = Node(new_board, parent=self, action=action)
        child.prior = 0.25  # Uniform prior for now unless using policy
        self.children.append(child)
        return child

    def backprop(self, value):
        self.visits += 1
        self.value_sum += value
        if self.parent:
            self.parent.backprop(-value)  # Value is from current player perspective



class PlayerController:
    # for the controller to read
    def __init__(self, time_left: Callable):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.simulations = 200

    def bid(self, board:player_board.PlayerBoard, time_left:Callable):
        return 0

    def play(self, board: player_board.PlayerBoard, time_left: Callable):
        if self.model is None:
            h, w = board.get_dim_y(), board.get_dim_x()
            self.model = utils.TinyCNN(input_shape=(6, h, w))
            self.model.load_state_dict(torch.load("cnn_weights/v4.pt", map_location=self.device))
            self.model.to(self.device)
            self.model.eval()

        root = Node(copy.deepcopy(board))

        for _ in range(self.simulations):
            node = root
            # Selection
            while node.children and not node.is_terminal():
                node = node.select_child_ucb()

            # Expansion
            if not node.is_terminal():
                node = node.expand(self.model, self.device)

            # Evaluation
            value = self.evaluate(node.board)

            # Backpropagation
            node.backprop(value)

        best_child = root.best_child() if root.children else None
        if best_child is None or best_child.action is None:
            return [Action.FF]
        return [best_child.action]




    
    def evaluate(self, board: player_board.PlayerBoard):
        features = utils.board_to_features(board)
        x = torch.tensor(features, dtype=torch.float32).unsqueeze(0).to(self.device)
        with torch.no_grad():
            value = self.model(x)
        return value.item()


