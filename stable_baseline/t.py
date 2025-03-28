import numpy as np
import torch
from stable_baselines3 import PPO
import sys

# Import your NumPy policy implementation
from model import NumpyPolicy

class PyTorchDebugger:
    """Helper class to debug PyTorch model layer by layer"""
    
    def __init__(self, model_path="bytefight_logs/ppo_bytefight_final.zip"):
        self.sb3 = PPO.load(model_path, device="cpu")
        self.policy = self.sb3.policy
        
    def prepare_input(self, obs_image, action_mask):
        """Convert NumPy inputs to PyTorch tensors"""
        # Convert to float32 and normalize
        if obs_image.dtype == np.uint8:
            image_tensor = torch.tensor(obs_image[None], dtype=torch.float32) / 255.0
        else:
            image_tensor = torch.tensor(obs_image[None], dtype=torch.float32)
            
        mask_tensor = torch.tensor(action_mask[None], dtype=torch.float32)
        
        # Create observation dict
        return {
            "image": image_tensor,
            "action_mask": mask_tensor,
        }
    
    def extract_features(self, obs_dict):
        """Run the feature extraction"""
        with torch.no_grad():
            # Use the shared feature extractor
            features = self.policy.extract_features(obs_dict)
            # Use specific feature extractors for policy and value
            pi_features = self.policy.pi_features_extractor(obs_dict)
            vf_features = self.policy.vf_features_extractor(obs_dict)
            
            # Extract the individual CNN steps of pi_features_extractor
            # Get the CNN part
            cnn = self.policy.pi_features_extractor.extractors["image"].cnn
            
            # Process image only
            x = obs_dict["image"]
            conv1 = cnn[0](x)  # Conv2d layer
            relu1 = cnn[1](conv1)  # ReLU
            conv2 = cnn[2](relu1)  # Conv2d layer
            relu2 = cnn[3](conv2)  # ReLU
            conv3 = cnn[4](relu1)  # Conv2d layer
            relu3 = cnn[5](conv3)  # ReLU
            flattened = cnn[6](relu3)  # Flatten
            
            # Process through linear layer
            linear = self.policy.pi_features_extractor.extractors["image"].linear[0]
            linear_out = linear(flattened)
            relu_out = self.policy.pi_features_extractor.extractors["image"].linear[1](linear_out)
            
            # Concatenate with action mask
            action_mask_flat = self.policy.pi_features_extractor.extractors["action_mask"](obs_dict["action_mask"])
            combined = torch.cat([relu_out, action_mask_flat], dim=1)
            
            # Extract weights from each CNN layer
            w1 = cnn[0].weight.data.cpu().numpy()
            b1 = cnn[0].bias.data.cpu().numpy()
            w2 = cnn[2].weight.data.cpu().numpy()
            b2 = cnn[2].bias.data.cpu().numpy()
            w3 = cnn[4].weight.data.cpu().numpy()
            b3 = cnn[4].bias.data.cpu().numpy()
            w_linear = linear.weight.data.cpu().numpy()
            b_linear = linear.bias.data.cpu().numpy()
            
            # Save all intermediate outputs
            result = {
                "features": features.cpu().numpy(),
                "pi_features": pi_features.cpu().numpy(),
                "vf_features": vf_features.cpu().numpy(),
                "conv1": conv1.cpu().numpy(),
                "relu1": relu1.cpu().numpy(),
                "conv2": conv2.cpu().numpy(),
                "relu2": relu2.cpu().numpy(),
                "conv3": conv3.cpu().numpy(),
                "relu3": relu3.cpu().numpy(),
                "flattened": flattened.cpu().numpy(),
                "linear_out": linear_out.cpu().numpy(),
                "relu_out": relu_out.cpu().numpy(),
                "combined": combined.cpu().numpy(),
                "weights": {
                    "w1": w1, "b1": b1, 
                    "w2": w2, "b2": b2, 
                    "w3": w3, "b3": b3,
                    "w_linear": w_linear, "b_linear": b_linear
                }
            }
            
            return result
            
    def process_mlp(self, features):
        """Process through the MLP extractor and action/value networks"""
        with torch.no_grad():
            features_tensor = torch.tensor(features[None], dtype=torch.float32)
            
            # Run through MLP extractor
            latent_pi, latent_vf = self.policy.mlp_extractor(features_tensor)
            
            # Extract intermediate values from policy net
            policy_net = self.policy.mlp_extractor.policy_net
            pi_hidden1 = policy_net[0](features_tensor)  # Linear
            pi_tanh1 = policy_net[1](pi_hidden1)  # Tanh
            pi_hidden2 = policy_net[2](pi_tanh1)  # Linear
            pi_tanh2 = policy_net[3](pi_hidden2)  # Tanh
            
            # Extract intermediate values from value net
            value_net = self.policy.mlp_extractor.value_net
            vf_hidden1 = value_net[0](features_tensor)  # Linear
            vf_tanh1 = value_net[1](vf_hidden1)  # Tanh
            vf_hidden2 = value_net[2](vf_tanh1)  # Linear
            vf_tanh2 = value_net[3](vf_hidden2)  # Tanh
            
            # Final outputs
            logits = self.policy.action_net(latent_pi)
            values = self.policy.value_net(latent_vf)
            
            # Extract weights
            pi_w1 = policy_net[0].weight.data.cpu().numpy()
            pi_b1 = policy_net[0].bias.data.cpu().numpy()
            pi_w2 = policy_net[2].weight.data.cpu().numpy()
            pi_b2 = policy_net[2].bias.data.cpu().numpy()
            
            vf_w1 = value_net[0].weight.data.cpu().numpy()
            vf_b1 = value_net[0].bias.data.cpu().numpy()
            vf_w2 = value_net[2].weight.data.cpu().numpy()
            vf_b2 = value_net[2].bias.data.cpu().numpy()
            
            action_w = self.policy.action_net.weight.data.cpu().numpy()
            action_b = self.policy.action_net.bias.data.cpu().numpy()
            value_w = self.policy.value_net.weight.data.cpu().numpy()
            value_b = self.policy.value_net.bias.data.cpu().numpy()
            
            result = {
                "pi_hidden1": pi_hidden1.cpu().numpy(),
                "pi_tanh1": pi_tanh1.cpu().numpy(),
                "pi_hidden2": pi_hidden2.cpu().numpy(),
                "pi_tanh2": pi_tanh2.cpu().numpy(),
                "vf_hidden1": vf_hidden1.cpu().numpy(),
                "vf_tanh1": vf_tanh1.cpu().numpy(),
                "vf_hidden2": vf_hidden2.cpu().numpy(),
                "vf_tanh2": vf_tanh2.cpu().numpy(),
                "logits": logits.cpu().numpy(),
                "values": values.cpu().numpy(),
                "weights": {
                    "pi_w1": pi_w1, "pi_b1": pi_b1,
                    "pi_w2": pi_w2, "pi_b2": pi_b2,
                    "vf_w1": vf_w1, "vf_b1": vf_b1,
                    "vf_w2": vf_w2, "vf_b2": vf_b2,
                    "action_w": action_w, "action_b": action_b,
                    "value_w": value_w, "value_b": value_b
                }
            }
            
            return result
            
    def predict(self, obs_image, action_mask):
        """Get final action and value predictions"""
        with torch.no_grad():
            obs_dict = self.prepare_input(obs_image, action_mask)
            action, _, _ = self.policy.forward(obs_dict)
            
            # Get all features
            features = self.policy.extract_features(obs_dict)
            
            # Get logits and value
            latent_pi, latent_vf = self.policy.mlp_extractor(features)
            logits = self.policy.action_net(latent_pi)
            value = self.policy.value_net(latent_vf)
            
            return {
                "action": action.item(),
                "logits": logits.cpu().numpy().squeeze(),
                "value": value.cpu().numpy().item()
            }

class NumPyDebugger:
    """Helper class to debug NumPy model layer by layer"""
    
    def __init__(self, model_path="weights.npz"):
        self.policy = NumpyPolicy(model_path)
        
    def extract_features(self, obs_image, action_mask):
        """Run through each layer of the CNN"""
        # Normalize inputs
        x = obs_image.astype(np.float32) / 255.0
        action_mask = action_mask.astype(np.float32)
        
        # Get CNN layers for policy path
        prefix = "pi_features_extractor.extractors.image.cnn"
        
        # Run Conv1
        w0 = self.policy.params[f"{prefix}.0.weight"]
        b0 = self.policy.params[f"{prefix}.0.bias"]
        conv1 = conv2d_nchw(x, w0, b0, stride=4, padding=0)
        relu1 = relu(conv1)
        
        # Run Conv2
        w2 = self.policy.params[f"{prefix}.2.weight"]
        b2 = self.policy.params[f"{prefix}.2.bias"]
        conv2 = conv2d_nchw(relu1, w2, b2, stride=2, padding=0)
        relu2 = relu(conv2)
        
        # Run Conv3
        w4 = self.policy.params[f"{prefix}.4.weight"]
        b4 = self.policy.params[f"{prefix}.4.bias"]
        conv3 = conv2d_nchw(relu2, w4, b4, stride=1, padding=0)
        relu3 = relu(conv3)
        
        # Flatten
        flattened = relu3.reshape(-1)
        
        # Linear
        wlin = self.policy.params[f"{prefix.replace('.cnn','.linear')}.0.weight"]
        blin = self.policy.params[f"{prefix.replace('.cnn','.linear')}.0.bias"]
        linear_out = linear(flattened, wlin, blin)
        relu_out = relu(linear_out)
        
        # Combine with action mask
        combined = np.concatenate([relu_out, action_mask], axis=0)
        
        # Feature extraction for value function (for completeness)
        vf_features = self.policy._forward_features(obs_image, action_mask, for_value=True)
        
        return {
            "features": None,  # Not directly accessible in NumPy implementation
            "pi_features": combined,
            "vf_features": vf_features,
            "conv1": conv1,
            "relu1": relu1,
            "conv2": conv2,
            "relu2": relu2,
            "conv3": conv3,
            "relu3": relu3,
            "flattened": flattened,
            "linear_out": linear_out,
            "relu_out": relu_out,
            "combined": combined,
            "weights": {
                "w1": w0, "b1": b0,
                "w2": w2, "b2": b2,
                "w3": w4, "b3": b4,
                "w_linear": wlin, "b_linear": blin
            }
        }
    
    def process_mlp(self, features):
        """Run through the MLP, replicating the policy's _mlp_policy and _mlp_value methods"""
        # Policy path
        w0 = self.policy.params["mlp_extractor.policy_net.0.weight"]
        b0 = self.policy.params["mlp_extractor.policy_net.0.bias"]
        pi_hidden1 = linear(features, w0, b0)
        pi_tanh1 = tanh(pi_hidden1)
        
        w2 = self.policy.params["mlp_extractor.policy_net.2.weight"]
        b2 = self.policy.params["mlp_extractor.policy_net.2.bias"]
        pi_hidden2 = linear(pi_tanh1, w2, b2)
        pi_tanh2 = tanh(pi_hidden2)
        
        # Action output
        wact = self.policy.params["action_net.weight"]
        bact = self.policy.params["action_net.bias"]
        logits = linear(pi_tanh2, wact, bact)
        
        # Value path
        w0v = self.policy.params["mlp_extractor.value_net.0.weight"]
        b0v = self.policy.params["mlp_extractor.value_net.0.bias"]
        vf_hidden1 = linear(features, w0v, b0v)
        vf_tanh1 = tanh(vf_hidden1)
        
        w2v = self.policy.params["mlp_extractor.value_net.2.weight"]
        b2v = self.policy.params["mlp_extractor.value_net.2.bias"]
        vf_hidden2 = linear(vf_tanh1, w2v, b2v)
        vf_tanh2 = tanh(vf_hidden2)
        
        # Value output
        wv = self.policy.params["value_net.weight"]
        bv = self.policy.params["value_net.bias"]
        values = linear(vf_tanh2, wv, bv)
        
        return {
            "pi_hidden1": pi_hidden1,
            "pi_tanh1": pi_tanh1,
            "pi_hidden2": pi_hidden2,
            "pi_tanh2": pi_tanh2,
            "vf_hidden1": vf_hidden1,
            "vf_tanh1": vf_tanh1,
            "vf_hidden2": vf_hidden2,
            "vf_tanh2": vf_tanh2,
            "logits": logits,
            "values": values,
            "weights": {
                "pi_w1": w0, "pi_b1": b0,
                "pi_w2": w2, "pi_b2": b2,
                "vf_w1": w0v, "vf_b1": b0v,
                "vf_w2": w2v, "vf_b2": b2v,
                "action_w": wact, "action_b": bact,
                "value_w": wv, "value_b": bv
            }
        }
        
    def predict(self, obs_image, action_mask):
        """Get final action and value predictions"""
        logits, value = self.policy.forward(obs_image, action_mask)
        action = self.policy.predict_action(obs_image, action_mask)
        
        return {
            "action": action,
            "logits": logits,
            "value": value
        }

# Import important functions from your model implementation for debugging
# Adjust these imports to match your implementation
from model import conv2d_nchw, linear, relu, tanh

def compare_outputs(pt_output, np_output, name, first_n=5, tolerance=1e-5):
    """Compare outputs between PyTorch and NumPy"""
    if pt_output is None or np_output is None:
        print(f"{name}: Cannot compare (missing output)")
        return False
    
    # Check shape match
    pt_shape = pt_output.shape
    np_shape = np_output.shape
    
    # Remove batch dimension for PyTorch outputs
    if len(pt_shape) > len(np_shape) and pt_shape[0] == 1:
        pt_output = pt_output.squeeze(0)
        pt_shape = pt_output.shape
    
    shape_match = (pt_shape == np_shape)
    
    # For multi-dimensional arrays, only look at first n elements along first dimension
    if len(pt_shape) > 1:
        pt_values = pt_output[0:first_n].flatten()
        np_values = np_output[0:first_n].flatten()
    else:
        pt_values = pt_output[0:first_n]
        np_values = np_output[0:first_n]
    
    # Calculate max absolute difference
    with np.errstate(invalid='ignore'):  # Ignore NaN warnings
        if len(pt_values) > 0 and len(np_values) > 0:
            abs_diff = np.abs(pt_values - np_values)
            max_diff = np.max(abs_diff) if abs_diff.size > 0 else float('inf')
            match = (max_diff <= tolerance)
        else:
            max_diff = float('inf')
            match = False
    
    # Print comparison
    print(f"{name}:")
    print(f"  Shapes - PT: {pt_shape}, NP: {np_shape}, Match: {shape_match}")
    print(f"  First {first_n} values:")
    print(f"    PT: {pt_values}")
    print(f"    NP: {np_values}")
    print(f"  Max Diff: {max_diff}")
    print(f"  Match (tolerance {tolerance}): {match}")
    print()
    
    return match

def compare_weights(pt_weights, np_weights, name, tolerance=1e-5):
    """Compare weight matrices between PyTorch and NumPy"""
    print(f"Comparing {name} weights:")
    pt_shape = pt_weights.shape
    np_shape = np_weights.shape
    
    shape_match = (pt_shape == np_shape)
    print(f"  Shapes - PT: {pt_shape}, NP: {np_shape}, Match: {shape_match}")
    
    if not shape_match:
        print("  WARNING: Shape mismatch, cannot compare values accurately!")
        return False
    
    # Calculate max absolute difference
    abs_diff = np.abs(pt_weights - np_weights)
    max_diff = np.max(abs_diff)
    match = (max_diff <= tolerance)
    
    print(f"  Max Diff: {max_diff}")
    print(f"  Match (tolerance {tolerance}): {match}")
    print()
    
    return match

def debug_single_example():
    """Run a thorough debug on a single random example"""
    # Create a debuggable input
    np.random.seed(42)  # Use fixed seed for reproducibility
    obs_image = np.random.randint(0, 256, size=(9, 64, 64), dtype=np.uint8)
    action_mask = np.random.randint(0, 2, size=(9,), dtype=np.uint8)
    
    # Ensure at least one valid action
    if np.sum(action_mask) == 0:
        action_mask[0] = 1
    
    print("=== THOROUGH DEBUG ON SINGLE EXAMPLE ===")
    print(f"Input Image Shape: {obs_image.shape}")
    print(f"Action Mask: {action_mask}")
    
    # Initialize debuggers
    pt_debugger = PyTorchDebugger()
    np_debugger = NumPyDebugger()
    
    # 1. Compare feature extraction
    print("\n=== FEATURE EXTRACTION COMPARISON ===")
    obs_dict = pt_debugger.prepare_input(obs_image, action_mask)
    pt_features = pt_debugger.extract_features(obs_dict)
    np_features = np_debugger.extract_features(obs_image, action_mask)
    
    # Compare each intermediate output
    compare_outputs(pt_features["conv1"], np_features["conv1"], "Conv1 Output")
    compare_outputs(pt_features["relu1"], np_features["relu1"], "ReLU1 Output")
    compare_outputs(pt_features["conv2"], np_features["conv2"], "Conv2 Output")
    compare_outputs(pt_features["relu2"], np_features["relu2"], "ReLU2 Output")
    compare_outputs(pt_features["conv3"], np_features["conv3"], "Conv3 Output")
    compare_outputs(pt_features["relu3"], np_features["relu3"], "ReLU3 Output")
    compare_outputs(pt_features["flattened"], np_features["flattened"], "Flattened Output")
    compare_outputs(pt_features["linear_out"], np_features["linear_out"], "Linear Output")
    compare_outputs(pt_features["relu_out"], np_features["relu_out"], "ReLU Output (after Linear)")
    compare_outputs(pt_features["combined"], np_features["combined"], "Combined Features")
    
    # 2. Compare weight matrices
    print("\n=== WEIGHT MATRIX COMPARISON ===")
    compare_weights(pt_features["weights"]["w1"], np_features["weights"]["w1"], "Conv1 weights")
    compare_weights(pt_features["weights"]["b1"], np_features["weights"]["b1"], "Conv1 bias")
    compare_weights(pt_features["weights"]["w2"], np_features["weights"]["w2"], "Conv2 weights")
    compare_weights(pt_features["weights"]["b2"], np_features["weights"]["b2"], "Conv2 bias")
    compare_weights(pt_features["weights"]["w3"], np_features["weights"]["w3"], "Conv3 weights")
    compare_weights(pt_features["weights"]["b3"], np_features["weights"]["b3"], "Conv3 bias")
    compare_weights(pt_features["weights"]["w_linear"], np_features["weights"]["w_linear"], "Linear weights")
    compare_weights(pt_features["weights"]["b_linear"], np_features["weights"]["b_linear"], "Linear bias")
    
    # 3. Test MLP processing
    print("\n=== MLP PROCESSING COMPARISON ===")
    pt_mlp = pt_debugger.process_mlp(pt_features["pi_features"])
    np_mlp = np_debugger.process_mlp(np_features["pi_features"])
    
    compare_outputs(pt_mlp["pi_hidden1"], np_mlp["pi_hidden1"], "Policy Hidden1")
    compare_outputs(pt_mlp["pi_tanh1"], np_mlp["pi_tanh1"], "Policy Tanh1")
    compare_outputs(pt_mlp["pi_hidden2"], np_mlp["pi_hidden2"], "Policy Hidden2")
    compare_outputs(pt_mlp["pi_tanh2"], np_mlp["pi_tanh2"], "Policy Tanh2")
    compare_outputs(pt_mlp["logits"], np_mlp["logits"], "Policy Logits")
    
    compare_outputs(pt_mlp["vf_hidden1"], np_mlp["vf_hidden1"], "Value Hidden1")
    compare_outputs(pt_mlp["vf_tanh1"], np_mlp["vf_tanh1"], "Value Tanh1")
    compare_outputs(pt_mlp["vf_hidden2"], np_mlp["vf_hidden2"], "Value Hidden2")
    compare_outputs(pt_mlp["vf_tanh2"], np_mlp["vf_tanh2"], "Value Tanh2")
    compare_outputs(pt_mlp["values"], np_mlp["values"], "Value Output")
    
    # 4. Compare MLP weights
    print("\n=== MLP WEIGHT COMPARISON ===")
    compare_weights(pt_mlp["weights"]["pi_w1"], np_mlp["weights"]["pi_w1"], "Policy Hidden1 weights")
    compare_weights(pt_mlp["weights"]["pi_b1"], np_mlp["weights"]["pi_b1"], "Policy Hidden1 bias")
    compare_weights(pt_mlp["weights"]["pi_w2"], np_mlp["weights"]["pi_w2"], "Policy Hidden2 weights")
    compare_weights(pt_mlp["weights"]["pi_b2"], np_mlp["weights"]["pi_b2"], "Policy Hidden2 bias")
    
    compare_weights(pt_mlp["weights"]["action_w"], np_mlp["weights"]["action_w"], "Action weights")
    compare_weights(pt_mlp["weights"]["action_b"], np_mlp["weights"]["action_b"], "Action bias")
    
    compare_weights(pt_mlp["weights"]["vf_w1"], np_mlp["weights"]["vf_w1"], "Value Hidden1 weights")
    compare_weights(pt_mlp["weights"]["vf_b1"], np_mlp["weights"]["vf_b1"], "Value Hidden1 bias")
    compare_weights(pt_mlp["weights"]["vf_w2"], np_mlp["weights"]["vf_w2"], "Value Hidden2 weights")
    compare_weights(pt_mlp["weights"]["vf_b2"], np_mlp["weights"]["vf_b2"], "Value Hidden2 bias")
    
    compare_weights(pt_mlp["weights"]["value_w"], np_mlp["weights"]["value_w"], "Value weights")
    compare_weights(pt_mlp["weights"]["value_b"], np_mlp["weights"]["value_b"], "Value bias")
    
    # 5. Final prediction comparison
    print("\n=== FINAL PREDICTION COMPARISON ===")
    pt_pred = pt_debugger.predict(obs_image, action_mask)
    np_pred = np_debugger.predict(obs_image, action_mask)
    
    print(f"PyTorch Action: {pt_pred['action']}")
    print(f"NumPy Action: {np_pred['action']}")
    print(f"Action Match: {pt_pred['action'] == np_pred['action']}")
    
    print("\nPyTorch Logits:")
    print(pt_pred['logits'])
    print("NumPy Logits:")
    print(np_pred['logits'])
    print(f"Max Logit Diff: {np.max(np.abs(pt_pred['logits'] - np_pred['logits']))}")
    
    print(f"\nPyTorch Value: {pt_pred['value']}")
    print(f"NumPy Value: {np_pred['value']}")
    print(f"Value Diff: {abs(pt_pred['value'] - np_pred['value'])}")


if __name__ == "__main__":
    debug_single_example()