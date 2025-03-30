import torch
import numpy as np
from stable_baselines3 import PPO
from composite_layers import MultiInputActorCriticPolicy

def verify_no_errors(npz_path):
    """Basic verification that model loads and runs without errors"""
    # Load weights from the .npz file
    weights = np.load(npz_path)
    
    # Instantiate the overall policy
    policy = MultiInputActorCriticPolicy()
    
    # Load the weights into the policy composite
    policy.load_weights(weights)
    
    # Create a dummy observation
    dummy_obs = {
        'image': np.random.rand(1, 9, 64, 64).astype(np.float32),
        'action_mask': np.random.randint(0, 2, (1, 9)).astype(np.float32)
    }
    
    # Forward pass through the policy
    logits, value = policy.forward(dummy_obs)
    
    # Print the outputs
    print("Logits:", logits)
    print("Value:", value)
    
    return policy

def verify_accuracy_shallow(zip_path, npz_path):
    """Compare PyTorch and NumPy model outputs to ensure they match (simplified version)"""
    # Set a fixed seed for reproducibility
    np.random.seed(42)
    torch.manual_seed(42)

    # 1. Load the Stable Baselines PPO model on CPU
    sb_model = PPO.load(zip_path, device="cpu")
    sb_model.policy.eval()  # set policy to evaluation mode

    # Print model architecture for reference
    print("PyTorch Model Structure:")
    for name, param in sb_model.policy.named_parameters():
        print(f"  {name}: {param.shape}")

    # 2. Create dummy observation
    dummy_image = np.random.rand(1, 9, 64, 64).astype(np.float32)
    dummy_mask = np.ones((1, 9)).astype(np.float32)
    
    # Convert to PyTorch tensors
    dummy_obs_torch = {
        'image': torch.tensor(dummy_image),
        'action_mask': torch.tensor(dummy_mask)
    }
    
    # 3. Get outputs from PyTorch model
    with torch.no_grad():
        # Extract features
        pi_features = sb_model.policy.pi_features_extractor(dummy_obs_torch)
        # Get policy and value latent representations
        pi_latent, vf_latent = sb_model.policy.mlp_extractor(pi_features)
        # Get logits and value
        pytorch_logits = sb_model.policy.action_net(pi_latent)
        pytorch_value = sb_model.policy.value_net(vf_latent)
        
        # Convert to NumPy
        pytorch_logits = pytorch_logits.detach().cpu().numpy()
        pytorch_value = pytorch_value.detach().cpu().numpy()
    
    # 4. Load weights into NumPy model
    npz_weights = np.load(npz_path)
    numpy_model = MultiInputActorCriticPolicy()
    numpy_model.load_weights(npz_weights)
    
    # 5. Get outputs from NumPy model
    dummy_obs_np = {
        'image': dummy_image,
        'action_mask': dummy_mask
    }
    
    numpy_logits, numpy_value = numpy_model.forward(dummy_obs_np)
    
    # 6. Compare the outputs with appropriate tolerance
    logits_match = np.allclose(pytorch_logits, numpy_logits, rtol=1e-5, atol=1e-5)
    value_match = np.allclose(pytorch_value, numpy_value, rtol=1e-5, atol=1e-5)
    
    # Print the results
    print("PyTorch logits:", pytorch_logits)
    print("NumPy logits:", numpy_logits)
    print("Logits match:", logits_match)
    print("Maximum logits difference:", np.max(np.abs(pytorch_logits - numpy_logits)))
    
    print("\nPyTorch value:", pytorch_value)
    print("NumPy value:", numpy_value)
    print("Value match:", value_match)
    print("Value difference:", np.abs(pytorch_value - numpy_value)[0][0])
    
    # For competition, we care about the predicted action - see if they match
    pytorch_action = np.argmax(pytorch_logits, axis=1)[0]
    numpy_action = np.argmax(numpy_logits, axis=1)[0]
    print("\nPyTorch predicted action:", pytorch_action)
    print("NumPy predicted action:", numpy_action)
    print("Actions match:", pytorch_action == numpy_action)
    
    return logits_match and value_match


def verify_accuracy_deep(zip_path, npz_path):
    """Perform a detailed comparison of all intermediate values"""
    # Load models
    sb_model = PPO.load(zip_path, device="cpu")
    sb_model.policy.eval()
    
    npz_weights = np.load(npz_path)
    numpy_model = MultiInputActorCriticPolicy()
    numpy_model.load_weights(npz_weights)
    
    # Create fixed input
    np.random.seed(42)
    torch.manual_seed(42)
    dummy_image = np.random.rand(1, 9, 64, 64).astype(np.float32)
    dummy_mask = np.ones((1, 9), dtype=np.float32)
    
    torch_image = torch.tensor(dummy_image)
    torch_mask = torch.tensor(dummy_mask)
    torch_obs = {'image': torch_image, 'action_mask': torch_mask}
    numpy_obs = {'image': dummy_image, 'action_mask': dummy_mask}
    
    # Get intermediate values from PyTorch model
    print("\nDebugging PyTorch model intermediate values:")
    with torch.no_grad():
        # CNN
        torch_cnn_output = sb_model.policy.pi_features_extractor.extractors['image'].cnn(torch_image)
        print(f"  PyTorch CNN output shape: {torch_cnn_output.shape}")
        print(f"  PyTorch CNN output first few values: {torch_cnn_output.flatten()[:5].numpy()}")
        
        # Linear layer
        torch_linear_output = sb_model.policy.pi_features_extractor.extractors['image'].linear(torch_cnn_output)
        print(f"  PyTorch linear output shape: {torch_linear_output.shape}")
        print(f"  PyTorch linear output first few values: {torch_linear_output.flatten()[:5].numpy()}")
        
        # Combined features
        torch_pi_features = sb_model.policy.pi_features_extractor(torch_obs)
        print(f"  PyTorch pi_features shape: {torch_pi_features.shape}")
        print(f"  PyTorch pi_features first few values: {torch_pi_features.flatten()[:5].numpy()}")
        
        # MLP extractor
        torch_pi_latent, torch_vf_latent = sb_model.policy.mlp_extractor(torch_pi_features)
        print(f"  PyTorch pi_latent shape: {torch_pi_latent.shape}")
        print(f"  PyTorch pi_latent first few values: {torch_pi_latent.flatten()[:5].numpy()}")
        print(f"  PyTorch vf_latent shape: {torch_vf_latent.shape}")
        print(f"  PyTorch vf_latent first few values: {torch_vf_latent.flatten()[:5].numpy()}")
        
        # Final values
        torch_logits = sb_model.policy.action_net(torch_pi_latent)
        torch_value = sb_model.policy.value_net(torch_vf_latent)
        print(f"  PyTorch logits shape: {torch_logits.shape}")
        print(f"  PyTorch logits: {torch_logits.numpy()}")
        print(f"  PyTorch value shape: {torch_value.shape}")
        print(f"  PyTorch value: {torch_value.numpy()}")
    
    # Get intermediate values from NumPy model
    print("\nDebugging NumPy model intermediate values:")
    
    # CNN output
    numpy_cnn_output = numpy_model.pi_features_extractor.extractors['image'].cnn.forward(numpy_obs['image'])
    print(f"  NumPy CNN output shape: {numpy_cnn_output.shape}")
    print(f"  NumPy CNN output first few values: {numpy_cnn_output.flatten()[:5]}")
    
    # Linear layer output
    numpy_linear_output = numpy_model.pi_features_extractor.extractors['image'].linear.forward(numpy_cnn_output)
    print(f"  NumPy linear output shape: {numpy_linear_output.shape}")
    print(f"  NumPy linear output first few values: {numpy_linear_output.flatten()[:5]}")
    
    # Combined features
    numpy_pi_features = numpy_model.pi_features_extractor.forward(numpy_obs)
    print(f"  NumPy pi_features shape: {numpy_pi_features.shape}")
    print(f"  NumPy pi_features first few values: {numpy_pi_features.flatten()[:5]}")
    
    # MLP extractor
    numpy_pi_latent, numpy_vf_latent = numpy_model.mlp_extractor.forward(numpy_pi_features)
    print(f"  NumPy pi_latent shape: {numpy_pi_latent.shape}")
    print(f"  NumPy pi_latent first few values: {numpy_pi_latent.flatten()[:5]}")
    print(f"  NumPy vf_latent shape: {numpy_vf_latent.shape}")
    print(f"  NumPy vf_latent first few values: {numpy_vf_latent.flatten()[:5]}")
    
    # Final values
    numpy_logits, numpy_value = numpy_model.forward(numpy_obs)
    print(f"  NumPy logits shape: {numpy_logits.shape}")
    print(f"  NumPy logits: {numpy_logits}")
    print(f"  NumPy value shape: {numpy_value.shape}")
    print(f"  NumPy value: {numpy_value}")
    
    return numpy_model

def main(zip_path, npz_path):
    print("=== RUNNING BASIC VERIFICATION ===")
    verify_no_errors(npz_path)
    
    print("\n=== RUNNING SHALLOW ACCURACY VERIFICATION ===")
    shallow_result = verify_accuracy_shallow(zip_path, npz_path)
    
    if shallow_result:
        print("\n✅ SUCCESS: NumPy model matches PyTorch model!")
    else:
        print("\n❌ FAILURE: Model outputs don't match. Running detailed comparison...")
        print("\n=== RUNNING DEEP COMPARISON ===")
        verify_accuracy_deep(zip_path, npz_path)
    
if __name__ == '__main__':
    zip_path = "bytefight_logs/ppo_bytefight_final.zip"
    npz_path = "weights/weights.npz"
    main(zip_path, npz_path)