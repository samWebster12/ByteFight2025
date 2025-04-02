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

def verify_accuracy(zip_path, npz_path):
    """Compare PyTorch and NumPy model outputs to ensure they match"""
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
        # Get policy and value outputs directly
        pi_features = sb_model.policy.pi_features_extractor(dummy_obs_torch)
        pi_latent, vf_latent = sb_model.policy.mlp_extractor(pi_features)
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
    print("\nModel Output Comparison:")
    print(f"PyTorch logits: {pytorch_logits}")
    print(f"NumPy logits: {numpy_logits}")
    print(f"Logits match: {logits_match}")
    print(f"Maximum logits difference: {np.max(np.abs(pytorch_logits - numpy_logits))}")
    
    print(f"\nPyTorch value: {pytorch_value}")
    print(f"NumPy value: {numpy_value}")
    print(f"Value match: {value_match}")
    print(f"Value difference: {np.abs(pytorch_value - numpy_value)[0][0]}")
    
    # 7. Check if predicted actions match
    pytorch_action = np.argmax(pytorch_logits, axis=1)[0]
    numpy_action = np.argmax(numpy_logits, axis=1)[0]
    print(f"\nPyTorch predicted action: {pytorch_action}")
    print(f"NumPy predicted action: {numpy_action}")
    print(f"Actions match: {pytorch_action == numpy_action}")
    
    return logits_match and value_match

def detailed_comparison(zip_path, npz_path):
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
    print("\nPyTorch Model Intermediate Values:")
    with torch.no_grad():
        # CNN
        torch_cnn_output = sb_model.policy.pi_features_extractor.extractors['image'].cnn(torch_image)
        print(f"  CNN output shape: {torch_cnn_output.shape}")
        print(f"  CNN output first few values: {torch_cnn_output.flatten()[:5].numpy()}")
        
        # Linear layer
        torch_linear_output = sb_model.policy.pi_features_extractor.extractors['image'].linear(torch_cnn_output)
        print(f"  Linear output shape: {torch_linear_output.shape}")
        print(f"  Linear output first few values: {torch_linear_output.flatten()[:5].numpy()}")
        
        # Features
        torch_pi_features = sb_model.policy.pi_features_extractor(torch_obs)
        print(f"  Features shape: {torch_pi_features.shape}")
        print(f"  Features first few values: {torch_pi_features.flatten()[:5].numpy()}")
        
        # MLP
        torch_pi_latent, torch_vf_latent = sb_model.policy.mlp_extractor(torch_pi_features)
        print(f"  Policy latent shape: {torch_pi_latent.shape}")
        print(f"  Policy latent first few values: {torch_pi_latent.flatten()[:5].numpy()}")
        print(f"  Value latent shape: {torch_vf_latent.shape}")
        print(f"  Value latent first few values: {torch_vf_latent.flatten()[:5].numpy()}")
        
        # Final outputs
        torch_logits = sb_model.policy.action_net(torch_pi_latent)
        torch_value = sb_model.policy.value_net(torch_vf_latent)
        print(f"  Logits shape: {torch_logits.shape}")
        print(f"  Logits: {torch_logits.numpy()}")
        print(f"  Value shape: {torch_value.shape}")
        print(f"  Value: {torch_value.numpy()}")
    
    # Get intermediate values from NumPy model
    print("\nNumPy Model Intermediate Values:")
    
    # CNN
    numpy_cnn_output = numpy_model.pi_features_extractor.extractors['image'].cnn.forward(numpy_obs['image'])
    print(f"  CNN output shape: {numpy_cnn_output.shape}")
    print(f"  CNN output first few values: {numpy_cnn_output.flatten()[:5]}")
    
    # Linear
    numpy_linear_output = numpy_model.pi_features_extractor.extractors['image'].linear.forward(numpy_cnn_output)
    print(f"  Linear output shape: {numpy_linear_output.shape}")
    print(f"  Linear output first few values: {numpy_linear_output.flatten()[:5]}")
    
    # Features
    numpy_pi_features = numpy_model.pi_features_extractor.forward(numpy_obs)
    print(f"  Features shape: {numpy_pi_features.shape}")
    print(f"  Features first few values: {numpy_pi_features.flatten()[:5]}")
    
    # MLP
    numpy_pi_latent, numpy_vf_latent = numpy_model.mlp_extractor.forward(numpy_pi_features)
    print(f"  Policy latent shape: {numpy_pi_latent.shape}")
    print(f"  Policy latent first few values: {numpy_pi_latent.flatten()[:5]}")
    print(f"  Value latent shape: {numpy_vf_latent.shape}")
    print(f"  Value latent first few values: {numpy_vf_latent.flatten()[:5]}")
    
    # Final outputs
    numpy_logits, numpy_value = numpy_model.forward(numpy_obs)
    print(f"  Logits shape: {numpy_logits.shape}")
    print(f"  Logits: {numpy_logits}")
    print(f"  Value shape: {numpy_value.shape}")
    print(f"  Value: {numpy_value}")

    # 7. Check if predicted actions match
    torch_action = np.argmax(torch_logits.numpy(), axis=1)[0]
    numpy_action = np.argmax(numpy_logits, axis=1)[0]
    print(f"\nPyTorch predicted action: {torch_action}")
    print(f"NumPy predicted action: {numpy_action}")
    print(f"Actions match: {torch_action == numpy_action}")
    
    return numpy_model

def main(zip_path, npz_path):
    print("=== RUNNING BASIC VERIFICATION ===")
    verify_no_errors(npz_path)
    
    print("\n=== RUNNING ACCURACY VERIFICATION ===")
    accuracy_result = verify_accuracy(zip_path, npz_path)
    
    if accuracy_result:
        print("\n✅ SUCCESS: NumPy model matches PyTorch model!")
    else:
        print("\n❌ FAILURE: Model outputs don't match. Running detailed comparison...")
        print("\n=== RUNNING DETAILED COMPARISON ===")
        detailed_comparison(zip_path, npz_path)
    
if __name__ == '__main__':
    zip_path = "bytefight_logs/ppo_bytefight_final.zip"
    npz_path = "weights/weights.npz"
    main(zip_path, npz_path)