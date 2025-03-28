import numpy as np
from game.enums import Action, Cell

# NumPy implementation of the policy network
def relu(x):
    return np.maximum(x, 0)

def tanh(x):
    return np.tanh(x)

def conv2d_nchw(x, w, b, stride=1, padding=0):
    """2D convolution implementation"""
    in_ch, H, W = x.shape
    out_ch, in_ch2, kH, kW = w.shape
    
    # Apply padding if needed
    if padding > 0:
        x_padded = np.zeros((in_ch, H + 2*padding, W + 2*padding), dtype=np.float32)
        x_padded[:, padding:padding+H, padding:padding+W] = x
        x = x_padded
        H, W = x.shape[1], x.shape[2]
    
    outH = (H - kH) // stride + 1
    outW = (W - kW) // stride + 1
    y = np.zeros((out_ch, outH, outW), dtype=np.float32)
    
    for oc in range(out_ch):
        for oh in range(outH):
            for ow in range(outW):
                h_in = oh * stride
                w_in = ow * stride
                patch = x[:, h_in:h_in + kH, w_in:w_in + kW]
                y[oc, oh, ow] = np.sum(patch * w[oc]) + b[oc]
    return y

def linear(x, w, b):
    """Linear layer implementation"""
    return np.dot(x, w.T) + b

class NumpyPolicy:
    def __init__(self, npz_path):
        try:
            params = np.load(npz_path)
        except Exception as e:
            print(f"ERROR loading weights from {npz_path}: {e}")
            raise e

        self.params = {}
        for k in params.keys():
            self.params[k] = params[k]

    def _run_cnn(self, x, prefix="pi_features_extractor.extractors.image.cnn"):
        """Run CNN feature extractor"""
        # Conv1: (9,64,64) -> (32,15,15)
        w0 = self.params[f"{prefix}.0.weight"]  # (32,9,8,8)
        b0 = self.params[f"{prefix}.0.bias"]    # (32,)
        out0 = conv2d_nchw(x, w0, b0, stride=4, padding=0)
        out0 = relu(out0)

        # Conv2: (32,15,15) -> (64,6,6)
        w2 = self.params[f"{prefix}.2.weight"]  # (64,32,4,4)
        b2 = self.params[f"{prefix}.2.bias"]    # (64,)
        out1 = conv2d_nchw(out0, w2, b2, stride=2, padding=0)
        out1 = relu(out1)

        # Conv3: (64,6,6) -> (64,4,4)
        w4 = self.params[f"{prefix}.4.weight"]  # (64,64,3,3)
        b4 = self.params[f"{prefix}.4.bias"]    # (64,)
        out2 = conv2d_nchw(out1, w4, b4, stride=1, padding=0)
        out2 = relu(out2)

        # Flatten: (64,4,4) -> (1024,)
        out2_flat = out2.reshape(-1)

        # Linear: (1024,) -> (256,)
        wlin = self.params[f"{prefix.replace('.cnn','.linear')}.0.weight"]  # (256,1024)
        blin = self.params[f"{prefix.replace('.cnn','.linear')}.0.bias"]    # (256,)
        out3 = linear(out2_flat, wlin, blin)
        out3 = relu(out3)
        
        return out3

    def _forward_features(self, obs_image, action_mask, for_value=False):
        """Extract features from observation"""
        # Normalize image to [0,1]
        obs_image = obs_image.astype(np.float32) / 255.0
        action_mask = action_mask.astype(np.float32)

        # Use correct feature extractor
        prefix = "vf_features_extractor.extractors.image.cnn" if for_value else "pi_features_extractor.extractors.image.cnn"
        cnn_out = self._run_cnn(obs_image, prefix=prefix)  # (256,)
        
        # Concatenate with action mask: (256,) + (9,) -> (265,)
        combined = np.concatenate([cnn_out, action_mask], axis=0)
        return combined

    def _mlp_policy(self, x):
        """Run policy MLP network"""
        # First layer: 265 -> 128 with Tanh activation
        w0 = self.params["mlp_extractor.policy_net.0.weight"]  # (128, 265)
        b0 = self.params["mlp_extractor.policy_net.0.bias"]    # (128,)
        hidden = linear(x, w0, b0)
        hidden = tanh(hidden)

        # Second layer: 128 -> 128 with Tanh activation
        w2 = self.params["mlp_extractor.policy_net.2.weight"]  # (128, 128)
        b2 = self.params["mlp_extractor.policy_net.2.bias"]    # (128,)
        hidden = linear(hidden, w2, b2)
        hidden = tanh(hidden)

        # Action head: 128 -> 9 (logits)
        wact = self.params["action_net.weight"]  # (9,128)
        bact = self.params["action_net.bias"]    # (9,)
        logits = linear(hidden, wact, bact)
        return logits

    def _mlp_value(self, x):
        """Run value MLP network"""
        # First layer: 265 -> 128 with Tanh activation
        w0 = self.params["mlp_extractor.value_net.0.weight"]  # (128, 265)
        b0 = self.params["mlp_extractor.value_net.0.bias"]    # (128,)
        hidden = linear(x, w0, b0)
        hidden = tanh(hidden)

        # Second layer: 128 -> 128 with Tanh activation
        w2 = self.params["mlp_extractor.value_net.2.weight"]  # (128, 128)
        b2 = self.params["mlp_extractor.value_net.2.bias"]    # (128,)
        hidden = linear(hidden, w2, b2)
        hidden = tanh(hidden)

        # Value head: 128 -> 1
        wv = self.params["value_net.weight"]  # (1,128)
        bv = self.params["value_net.bias"]    # (1,)
        value = linear(hidden, wv, bv)
        return value

    def forward(self, obs_image, action_mask):
        """Full forward pass for a single observation"""
        # Policy path
        feat_pi = self._forward_features(obs_image, action_mask, for_value=False)
        logits = self._mlp_policy(feat_pi)

        # Value path
        feat_vf = self._forward_features(obs_image, action_mask, for_value=True)
        value = self._mlp_value(feat_vf)

        return logits, value[0]

    def predict_action(self, obs_image, action_mask):
        """Predict best action given observation and mask"""
        logits, _ = self.forward(obs_image, action_mask)
        
        # Apply action mask
        masked_logits = logits.copy()
        mask = action_mask.astype(bool)
        if np.any(mask):  # If any valid actions
            masked_logits[~mask] = -float('inf')  # Set invalid actions to -inf
        
        return int(np.argmax(masked_logits))

class PlayerController:
    def __init__(self, get_time_left):
        self.model = NumpyPolicy("weights.npz")
        self.get_time_left = get_time_left
        
    def play(self, player_board, time_left_callable):
        # 1. Build observation
        obs_image = self._get_image_observation(player_board)
        action_mask = self._get_valid_actions(player_board)
        
        # 2. Use model to predict action
        action_idx = self.model.predict_action(obs_image, action_mask)
        
        # 3. Convert to ByteFight action
        if action_idx == 8:
            return Action.TRAP
        else:
            return Action(action_idx)
        
    def _get_image_observation(self, player_board):
        """Build the 9-channel image observation"""
        image = np.zeros((9, 64, 64), dtype=np.uint8)
        
        # Channel 0: Walls
        wall_mask = player_board.get_wall_mask()
        image[0] = np.where(wall_mask == Cell.WALL, 255, 0)
        
        # Channel 1: Apples
        apple_mask = player_board.get_apple_mask()
        image[1] = np.where(apple_mask == Cell.APPLE, 255, 0)
        
        # Channel 2: My snake head
        a_snake_mask = player_board.get_snake_mask(my_snake=True, enemy_snake=False)
        image[2] = np.where(a_snake_mask == Cell.PLAYER_HEAD, 255, 0)
        
        # Channel 3: My snake body
        image[3] = np.where(a_snake_mask == Cell.PLAYER_BODY, 255, 0)
        
        # Channel 4: Enemy snake head
        b_snake_mask = player_board.get_snake_mask(my_snake=False, enemy_snake=True)
        image[4] = np.where(b_snake_mask == Cell.ENEMY_HEAD, 255, 0)
        
        # Channel 5: Enemy snake body
        image[5] = np.where(b_snake_mask == Cell.ENEMY_BODY, 255, 0)
        
        # Channel 6: My traps
        my_trap_mask = player_board.get_trap_mask(my_traps=True, enemy_traps=False)
        image[6] = np.where(my_trap_mask > 0, 255, 0)
        
        # Channel 7: Enemy traps
        enemy_trap_mask = player_board.get_trap_mask(my_traps=False, enemy_traps=True)
        image[7] = np.where(enemy_trap_mask > 0, 255, 0)
        
        # Channel 8: Portals
        try:
            portal_mask = player_board.get_portal_mask(descriptive=False)
            # Sometimes, the mask might have an extra dimension
            if portal_mask.ndim == 3:
                portal_mask = portal_mask[:, :, 0]
            image[8] = np.where(portal_mask == 1, 255, 0)
        except Exception as e:
            # Silent error - portals might not be available
            pass
            
        return image
        
    def _get_valid_actions(self, player_board):
        """Get mask of valid actions"""
        mask = np.zeros(9, dtype=np.uint8)
        
        # Check valid directional moves
        for move in range(8):
            action = Action(move)
            if player_board.is_valid_move(action):
                mask[move] = 1
                
        # Check if trap placement is valid
        if player_board.is_valid_trap():
            mask[8] = 1
            
        # If no valid actions, allow action 0 (will fail, but prevents crashes)
        if np.sum(mask) == 0:
            mask[0] = 1
            
        return mask