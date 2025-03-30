import numpy as np
import torch
import torch.nn as nn

# Import all the layers
from base_layers import Conv2D, ReLU, MaxPool2D, Flatten, Linear
from base_layers import Softmax, Tanh, LayerNorm, Concatenate, DictObsHandler, Sequential

def verify_layer(name, numpy_layer, torch_layer, input_data, rtol=1e-5, atol=1e-5):
    """Verify that NumPy and PyTorch implementations produce the same output."""
    # Convert input to torch tensor if needed
    if isinstance(input_data, np.ndarray):
        torch_input = torch.tensor(input_data, dtype=torch.float32)
    else:
        torch_input = input_data
        input_data = {k: v.numpy() if isinstance(v, torch.Tensor) else v for k, v in input_data.items()}
    
    # Get outputs
    with torch.no_grad():
        torch_output = torch_layer(torch_input)
        if isinstance(torch_output, torch.Tensor):
            torch_output = torch_output.numpy()
        elif isinstance(torch_output, dict):
            torch_output = {k: v.numpy() if isinstance(v, torch.Tensor) else v for k, v in torch_output.items()}
    
    numpy_output = numpy_layer.forward(input_data)
    
    # If outputs are dictionaries, compare each value
    if isinstance(torch_output, dict) and isinstance(numpy_output, dict):
        all_match = True
        for key in torch_output.keys():
            if key not in numpy_output:
                print(f"! Key '{key}' not found in NumPy output")
                all_match = False
                continue
                
            key_match = np.allclose(numpy_output[key], torch_output[key], rtol=rtol, atol=atol)
            max_diff = np.max(np.abs(numpy_output[key] - torch_output[key]))
            
            if key_match:
                print(f"✓ {name} layer output '{key}' matches! Max difference: {max_diff:.8f}")
            else:
                print(f"✗ {name} layer output '{key}' differs! Max difference: {max_diff:.8f}")
                all_match = False
        
        return all_match
    
    # Otherwise compare tensors directly
    is_close = np.allclose(numpy_output, torch_output, rtol=rtol, atol=atol)
    max_diff = np.max(np.abs(numpy_output - torch_output))
    
    if is_close:
        print(f"✓ {name} layer outputs match! Max difference: {max_diff:.8f}")
    else:
        print(f"✗ {name} layer outputs differ! Max difference: {max_diff:.8f}")
        # Show where the differences are
        diff_mask = ~np.isclose(numpy_output, torch_output, rtol=rtol, atol=atol)
        if np.any(diff_mask):
            print("Sample differences:")
            indices = np.nonzero(diff_mask)
            for i in range(min(5, len(indices[0]))):
                idx = tuple(index[i] for index in indices)
                print(f"  At {idx}: NumPy={numpy_output[idx]}, PyTorch={torch_output[idx]}")
    
    return is_close

def verify_all_layers():
    """Test all implemented layers against PyTorch equivalents"""
    # Test Conv2D
    print("Testing Conv2D layer...")
    in_channels, out_channels, kernel_size = 3, 6, 3
    numpy_conv = Conv2D(in_channels, out_channels, kernel_size, stride=1, padding=1)
    torch_conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=1)
    
    # Copy weights from PyTorch to NumPy
    numpy_conv.load_weights(torch_conv.weight.detach().numpy(), torch_conv.bias.detach().numpy())
    
    # Create random input
    input_data = np.random.randn(2, in_channels, 32, 32).astype(np.float32)
    
    # Verify
    verify_layer("Conv2D", numpy_conv, torch_conv, input_data)
    
    # Test ReLU
    print("\nTesting ReLU layer...")
    numpy_relu = ReLU()
    torch_relu = nn.ReLU()
    
    input_data = np.random.randn(2, 10).astype(np.float32)
    verify_layer("ReLU", numpy_relu, torch_relu, input_data)
    
    # Test MaxPool2D
    print("\nTesting MaxPool2D layer...")
    kernel_size, stride = 2, 2
    numpy_pool = MaxPool2D(kernel_size, stride)
    torch_pool = nn.MaxPool2d(kernel_size, stride)
    
    input_data = np.random.randn(2, 3, 32, 32).astype(np.float32)
    verify_layer("MaxPool2D", numpy_pool, torch_pool, input_data)
    
    # Test Linear
    print("\nTesting Linear layer...")
    in_features, out_features = 10, 5
    numpy_linear = Linear(in_features, out_features)
    torch_linear = nn.Linear(in_features, out_features)
    
    # Copy weights
    numpy_linear.load_weights(torch_linear.weight.detach().numpy(), torch_linear.bias.detach().numpy())
    
    input_data = np.random.randn(2, in_features).astype(np.float32)
    verify_layer("Linear", numpy_linear, torch_linear, input_data)
    
    # Test Softmax
    print("\nTesting Softmax layer...")
    numpy_softmax = Softmax()
    torch_softmax = nn.Softmax(dim=1)
    
    input_data = np.random.randn(2, 5).astype(np.float32)
    verify_layer("Softmax", numpy_softmax, torch_softmax, input_data)
    
    # Test Tanh
    print("\nTesting Tanh layer...")
    numpy_tanh = Tanh()
    torch_tanh = nn.Tanh()
    
    input_data = np.random.randn(2, 10).astype(np.float32)
    verify_layer("Tanh", numpy_tanh, torch_tanh, input_data)
    
    # Test LayerNorm
    print("\nTesting LayerNorm layer...")
    normalized_shape = 10
    numpy_ln = LayerNorm(normalized_shape)
    torch_ln = nn.LayerNorm(normalized_shape)

    # Test Flatten
    print("\nTesting Flatten layer...")
    numpy_flatten = Flatten()
    torch_flatten = nn.Flatten()

    input_data = np.random.randn(2, 3, 4, 5).astype(np.float32)
    verify_layer("Flatten", numpy_flatten, torch_flatten, input_data)

    # Test Concatenate
    print("\nTesting Concatenate layer...")
    numpy_concat = Concatenate()
        
    # Define custom PyTorch concatenate function
    class TorchConcatenate(nn.Module):
        def forward(self, tensors, dim=1):
            return torch.cat(tensors, dim=dim)
        
    torch_concat = TorchConcatenate()

    # Create input tensors
    tensor1 = np.random.randn(2, 3, 4, 4).astype(np.float32)
    tensor2 = np.random.randn(2, 5, 4, 4).astype(np.float32)
    input_tensors = [tensor1, tensor2]
    torch_tensors = [torch.tensor(t) for t in input_tensors]

    # We need to modify the verify function call for concatenate since it takes a list
    # Instead of using verify_layer directly, we'll manually implement the test
    numpy_output = numpy_concat.forward(input_tensors)
    with torch.no_grad():
        torch_output = torch_concat.forward(torch_tensors)
        torch_output = torch_output.numpy()

    is_close = np.allclose(numpy_output, torch_output, rtol=1e-5, atol=1e-5)
    max_diff = np.max(np.abs(numpy_output - torch_output))

    if is_close:
        print(f"✓ Concatenate layer outputs match! Max difference: {max_diff:.8f}")
    else:
        print(f"✗ Concatenate layer outputs differ! Max difference: {max_diff:.8f}")
        
    # Copy weights
    numpy_ln.load_weights(torch_ln.weight.detach().numpy(), torch_ln.bias.detach().numpy())
    
    input_data = np.random.randn(2, normalized_shape).astype(np.float32)
    verify_layer("LayerNorm", numpy_ln, torch_ln, input_data)
    
    # Test Sequential
    print("\nTesting Sequential layer...")
    numpy_seq = Sequential([
        Linear(10, 20),
        ReLU(),
        Linear(20, 5)
    ])
    torch_seq = nn.Sequential(
        nn.Linear(10, 20),
        nn.ReLU(),
        nn.Linear(20, 5)
    )
    
    # Copy weights
    numpy_seq.layers[0].load_weights(
        torch_seq[0].weight.detach().numpy(),
        torch_seq[0].bias.detach().numpy()
    )
    numpy_seq.layers[2].load_weights(
        torch_seq[2].weight.detach().numpy(), 
        torch_seq[2].bias.detach().numpy()
    )
    
    input_data = np.random.randn(2, 10).astype(np.float32)
    verify_layer("Sequential", numpy_seq, torch_seq, input_data)
    

    print("\nTesting DictObsHandler with mock data...")
    class MockLayer:
        def forward(self, x):
            return x

    numpy_dict_handler = DictObsHandler(MockLayer(), MockLayer(), MockLayer())

    class TorchDictHandler(nn.Module):
        def forward(self, x):
            return x

    torch_dict_handler = TorchDictHandler()

    # Create input tensors
    mock_input = {
        'image': np.random.randn(2, 3, 32, 32).astype(np.float32),
        'action_mask': np.random.randn(2, 9).astype(np.float32)
    }
    torch_mock_input = {
        'image': torch.tensor(mock_input['image']),
        'action_mask': torch.tensor(mock_input['action_mask'])
    }
    
    # For dictionaries, we need to compare each key separately
    numpy_output = numpy_dict_handler.forward(mock_input)
    with torch.no_grad():
        torch_output = torch_dict_handler.forward(torch_mock_input)

    # Convert torch tensors to numpy arrays
    torch_output_np = {}
    for key, value in torch_output.items():
        if isinstance(value, torch.Tensor):
            torch_output_np[key] = value.numpy()
        else:
            torch_output_np[key] = value

    # Compare each key
    all_match = True
    for key in numpy_output:
        if key in torch_output_np:
            is_close = np.allclose(numpy_output[key], torch_output_np[key], rtol=1e-5, atol=1e-5)
            max_diff = np.max(np.abs(numpy_output[key] - torch_output_np[key]))
            
            if is_close:
                print(f"✓ DictObsHandler key '{key}' outputs match! Max difference: {max_diff:.8f}")
            else:
                print(f"✗ DictObsHandler key '{key}' outputs differ! Max difference: {max_diff:.8f}")
                all_match = False
        else:
            print(f"! Key '{key}' not found in PyTorch output")
            all_match = False

    if all_match:
        print("✓ DictObsHandler test passed!")
    else:
        print("✗ DictObsHandler test failed!")
    
    print("\nAll layer tests completed!")

# This allows the file to be run as a standalone script
if __name__ == "__main__":
    verify_all_layers()