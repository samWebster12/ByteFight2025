import torch
from stable_baselines3 import PPO

# Load your PPO model (assume it's loaded on the proper device)
model_path = "bytefight_logs/ppo_bytefight_final.zip"
model = PPO.load(model_path, device="cuda")

class PolicyWrapper(torch.nn.Module):
    def __init__(self, policy):
        super().__init__()
        self.policy = policy

    def forward(self, image, action_mask):
        # The policy expects a dictionary of observations.
        obs = {"image": image, "action_mask": action_mask}
        out = self.policy.forward(obs, deterministic=True)
        
        # If out is a tuple and the first element has 'logits', use that.
        if isinstance(out, tuple) and hasattr(out[0], "logits"):
            distribution, value, _ = out
            return distribution.logits
        # Otherwise, if out is a tuple but doesn't have logits,
        # assume the first element is the tensor we need.
        elif isinstance(out, tuple):
            return out[0]
        else:
            return out

# Instantiate the wrapper
wrapper = PolicyWrapper(model.policy)

# Create dummy inputs on the same device as the model.
dummy_image = torch.zeros(1, 9, 64, 64, dtype=torch.float32, device=model.device)
dummy_mask  = torch.zeros(1, 9, dtype=torch.float32, device=model.device)

torch.onnx.export(
    wrapper,
    (dummy_image, dummy_mask),
    "ppo_bytefight_final.onnx",
    input_names=["image", "action_mask"],
    output_names=["logits"],
    dynamic_axes={
        "image": {0: "batch_size"},
        "action_mask": {0: "batch_size"},
        "logits": {0: "batch_size"}
    },
    opset_version=12
)

print("Model exported to ppo_bytefight_final.onnx")
