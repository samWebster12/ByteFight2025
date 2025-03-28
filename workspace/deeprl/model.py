import numpy as np

def relu(x):
    return np.maximum(x, 0)

def tanh(x):
    return np.tanh(x)

def conv2d_nchw(x, w, b, stride=1):
    """
    Naive 2D convolution for a single input (batch=1), using NCHW shape: (channels, height, width).
    x.shape = (in_ch, H, W)
    w.shape = (out_ch, in_ch, kH, kW)
    b.shape = (out_ch,)
    stride: int
    Returns y with shape (out_ch, outH, outW)
    """
    in_ch, H, W = x.shape
    out_ch, in_ch2, kH, kW = w.shape
    if in_ch != in_ch2:
        raise ValueError(f"Input channels mismatch. x:{in_ch}, w:{in_ch2}")
    outH = (H - kH)//stride + 1
    outW = (W - kW)//stride + 1
    y = np.zeros((out_ch, outH, outW), dtype=x.dtype)
    for oc in range(out_ch):
        for oh in range(outH):
            for ow in range(outW):
                h_in = oh*stride
                w_in = ow*stride
                patch = x[:, h_in:h_in + kH, w_in:w_in + kW]
                y[oc, oh, ow] = np.sum(patch * w[oc]) + b[oc]
    return y

def linear(x, w, b):
    """
    x: shape (in_dim,)
    w: shape (out_dim, in_dim)
    b: shape (out_dim,)
    Returns shape (out_dim,)
    """
    return x @ w.T + b

class NumpyPolicy:
    def __init__(self, npz_path: str):
        try:
            params = np.load(npz_path)
        except Exception as e:
            print("ERROR loading weights:", e)
            raise e

        self.params = {}
        for k in params.keys():
            self.params[k] = params[k]

    def _run_cnn(self, x, prefix="features_extractor.extractors.image.cnn"):
        """
        x: shape (9, 64, 64)  (float32 after normalization)
        prefix: we use either features_extractor or pi_features_extractor or vf_features_extractor
        return shape (256,) after the 'linear' block, i.e. CNN => flatten => linear => ReLU
        """
        # 1) Conv(9->32, kernel=8, stride=4) + ReLU
        w0 = self.params[f"{prefix}.0.weight"]   # (32, 9, 8, 8)
        b0 = self.params[f"{prefix}.0.bias"]     # (32,)
        out0 = conv2d_nchw(x, w0, b0, stride=4)
        out0 = relu(out0)

        # 2) Conv(32->64, kernel=4, stride=2) + ReLU
        w2 = self.params[f"{prefix}.2.weight"]   # (64,32,4,4)
        b2 = self.params[f"{prefix}.2.bias"]     # (64,)
        out1 = conv2d_nchw(out0, w2, b2, stride=2)
        out1 = relu(out1)

        # 3) Conv(64->64, kernel=3, stride=1) + ReLU
        w4 = self.params[f"{prefix}.4.weight"]   # (64,64,3,3)
        b4 = self.params[f"{prefix}.4.bias"]     # (64,)
        out2 = conv2d_nchw(out1, w4, b4, stride=1)
        out2 = relu(out2)

        # Flatten => shape (64*4*4=1024,)
        out2_flat = out2.reshape(-1)

        # Then linear: in_features=1024, out_features=256
        wlin = self.params[f"{prefix.replace('.cnn','.linear')}.0.weight"]  # (256, 1024)
        blin = self.params[f"{prefix.replace('.cnn','.linear')}.0.bias"]    # (256,)
        out3 = linear(out2_flat, wlin, blin)
        out3 = relu(out3)  # final 256
        return out3

    def _forward_features(self, obs_image, action_mask, for_value=False):
        # For SB3, we must pass images in [0..1], so divide by 255
        # Make sure it's float32 so we don't blow up as float64
        obs_image = obs_image.astype(np.float32) / 255.0

        # pi vs vf feature extractor
        prefix = "pi_features_extractor.extractors.image.cnn"
        if for_value:
            prefix = "vf_features_extractor.extractors.image.cnn"

        cnn_out = self._run_cnn(obs_image, prefix=prefix)  # shape (256,)

        # Concat with 9-d action mask => total 265
        combined = np.concatenate([cnn_out, action_mask], axis=0)  # (265,)
        return combined

    def _mlp_policy(self, x):
        # x shape (265,)
        w0 = self.params["mlp_extractor.policy_net.0.weight"]
        b0 = self.params["mlp_extractor.policy_net.0.bias"]
        hidden = linear(x, w0, b0)
        hidden = tanh(hidden)

        w2 = self.params["mlp_extractor.policy_net.2.weight"]
        b2 = self.params["mlp_extractor.policy_net.2.bias"]
        hidden = linear(hidden, w2, b2)
        hidden = tanh(hidden)

        wact = self.params["action_net.weight"]  # (9,128)
        bact = self.params["action_net.bias"]    # (9,)
        logits = linear(hidden, wact, bact)      # shape (9,)
        return logits

    def _mlp_value(self, x):
        # x shape (265,)
        w0 = self.params["mlp_extractor.value_net.0.weight"]
        b0 = self.params["mlp_extractor.value_net.0.bias"]
        hidden = linear(x, w0, b0)
        hidden = tanh(hidden)

        w2 = self.params["mlp_extractor.value_net.2.weight"]
        b2 = self.params["mlp_extractor.value_net.2.bias"]
        hidden = linear(hidden, w2, b2)
        hidden = tanh(hidden)

        wv = self.params["value_net.weight"]  # (1,128)
        bv = self.params["value_net.bias"]    # (1,)
        value = linear(hidden, wv, bv)        # shape (1,)
        return value

    def forward(self, obs_image, action_mask):
        """
        Full forward pass for a single observation:
          obs_image: shape (9,64,64)  (uint8 or float32)
          action_mask: shape (9,)     (float32 or float64)
        Returns (logits, value) tuple
        """
        # 1) Policy path
        feat_pi = self._forward_features(obs_image, action_mask, for_value=False)  # (265,)
        logits = self._mlp_policy(feat_pi)  # (9,)

        # 2) Value path
        feat_vf = self._forward_features(obs_image, action_mask, for_value=True)   # (265,)
        value = self._mlp_value(feat_vf)  # shape (1,)

        return logits, value[0]

    def predict_action(self, obs_image, action_mask):
        logits, _ = self.forward(obs_image, action_mask)
        return int(np.argmax(logits))

    def predict_value(self, obs_image, action_mask):
        _, value = self.forward(obs_image, action_mask)
        return float(value)


if __name__ == "__main__":
    # Quick test
    model = NumpyPolicy("weights.npz")
    dummy_image = np.zeros((9,64,64), dtype=np.uint8)  # typical 8-bit
    dummy_mask  = np.zeros((9,),       dtype=np.float32)
    dummy_mask[0] = 1.0

    logits, value = model.forward(dummy_image, dummy_mask)
    action = model.predict_action(dummy_image, dummy_mask)
    print("Logits:", logits)
    print("Value:", value)
    print("Chosen action (argmax):", action)
