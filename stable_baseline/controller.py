import os
import numpy as np
from collections.abc import Callable
from game.enums import Action, Cell
from game.player_board import PlayerBoard

class NumpyPolicy:
    def __init__(self, path):
        data = np.load(path)
        self.conv1_w = data["pi_features_extractor.extractors.image.cnn.0.weight"]
        self.conv1_b = data["pi_features_extractor.extractors.image.cnn.0.bias"]
        self.conv2_w = data["pi_features_extractor.extractors.image.cnn.2.weight"]
        self.conv2_b = data["pi_features_extractor.extractors.image.cnn.2.bias"]
        self.conv3_w = data["pi_features_extractor.extractors.image.cnn.4.weight"]
        self.conv3_b = data["pi_features_extractor.extractors.image.cnn.4.bias"]
        self.fc_w   = data["pi_features_extractor.extractors.image.linear.0.weight"]
        self.fc_b   = data["pi_features_extractor.extractors.image.linear.0.bias"]
        self.w1     = data["mlp_extractor.policy_net.0.weight"]
        self.b1     = data["mlp_extractor.policy_net.0.bias"]
        self.w2     = data["mlp_extractor.policy_net.2.weight"]
        self.b2     = data["mlp_extractor.policy_net.2.bias"]
        self.wo     = data["action_net.weight"]
        self.bo     = data["action_net.bias"]

    def conv2d(self, x, w, b, stride):
        from numpy.lib.stride_tricks import sliding_window_view
        patches = sliding_window_view(x, (w.shape[2], w.shape[3]), axis=(1,2))
        patches = patches[:, ::stride, ::stride, :, :].transpose(1,2,0,3,4)
        flat = patches.reshape(-1, w.shape[1]*w.shape[2]*w.shape[3])
        out = flat.dot(w.reshape(w.shape[0], -1).T) + b
        H, W = patches.shape[0], patches.shape[1]
        return out.reshape(H, W, w.shape[0]).transpose(2,0,1)

    def select(self, obs):
        img = obs["image"].astype(np.float32)
        mask = obs["action_mask"].astype(np.float32)

        x = np.maximum(self.conv2d(img, self.conv1_w, self.conv1_b, 4), 0)
        x = np.maximum(self.conv2d(x, self.conv2_w, self.conv2_b, 2), 0)
        x = np.maximum(self.conv2d(x, self.conv3_w, self.conv3_b, 1), 0)
        x = x.flatten()
        x = np.maximum(x @ self.fc_w.T + self.fc_b, 0)
        x = np.concatenate([x, mask])

        h = np.tanh(x @ self.w1.T + self.b1)
        h = np.tanh(h @ self.w2.T + self.b2)
        logits = h @ self.wo.T + self.bo

        logits[mask == 0] = -1e9
        idx = int(np.argmax(logits))
        return Action.TRAP if idx == 8 else Action(idx)

class PlayerController:
    def __init__(self, time_left: Callable):
        base = os.path.dirname(os.path.abspath(__file__))
        self.policy = NumpyPolicy(os.path.join(base, "weights_bytefight.npz"))

    def bid(self, board: PlayerBoard, time_left: Callable) -> int:
        return 0

    def play(self, board: PlayerBoard, time_left: Callable) -> list[Action]:
        obs = self._create_observation(board)
        return [self.policy.select(obs)]

    def _create_observation(self, pb: PlayerBoard) -> dict:
        dim_x, dim_y = pb.get_dim_x(), pb.get_dim_y()
        channels = np.zeros((9, dim_y, dim_x), dtype=np.uint8)

        channels[0] = np.where(pb.get_wall_mask() == Cell.WALL, 255, 0)
        channels[1] = np.where(pb.get_apple_mask() == Cell.APPLE, 255, 0)

        snake = pb.get_snake_mask(my_snake=True, enemy_snake=False)
        channels[2] = (snake == Cell.PLAYER_HEAD) * 255
        channels[3] = (snake == Cell.PLAYER_BODY) * 255

        snake = pb.get_snake_mask(my_snake=False, enemy_snake=True)
        channels[4] = (snake == Cell.ENEMY_HEAD) * 255
        channels[5] = (snake == Cell.ENEMY_BODY) * 255

        channels[6] = (pb.get_trap_mask(my_traps=True, enemy_traps=False) > 0) * 255
        channels[7] = (pb.get_trap_mask(my_traps=False, enemy_traps=True) > 0) * 255

        try:
            portal = pb.get_portal_mask(descriptive=False)
            if portal.ndim == 3: portal = portal[:,:,0]
            channels[8] = (portal == 1) * 255
        except:
            pass

        image = np.zeros((9,64,64), dtype=np.uint8)
        image[:, :dim_y, :dim_x] = channels

        mask = np.zeros(9, dtype=np.uint8)
        for a in range(8):
            if pb.is_valid_move(Action(a)): mask[a] = 1
        if pb.is_valid_trap(): mask[8] = 1
        if mask.sum() == 0: mask[0] = 1

        return {"image": image, "action_mask": mask}
