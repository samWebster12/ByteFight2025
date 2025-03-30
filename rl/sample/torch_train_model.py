import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
from torchvision import datasets, transforms

# Create weights directory if it doesn't exist
os.makedirs('weights', exist_ok=True)

# Define a CNN model with the specified architecture
class EnhancedCNN(nn.Module):
    def __init__(self):
        super(EnhancedCNN, self).__init__()
        # First convolutional block
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Second convolutional block
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Fully connected layers
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(128, 10)
    
    def forward(self, x):
        # First conv block
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        
        # Second conv block
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        
        # Flatten
        x = x.view(-1, 64 * 7 * 7)
        
        # Fully connected layers
        x = self.fc1(x)
        x = self.relu3(x)
        x = self.fc2(x)
        
        return x

def main():
    # Setup device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load MNIST dataset
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_dataset = datasets.MNIST('data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('data', train=False, transform=transform)

    # Create data loaders
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1000, shuffle=False)

    # Initialize model, loss function, and optimizer
    model = EnhancedCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    num_epochs = 5
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            if (i + 1) % 100 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_loader)}], Loss: {running_loss/100:.4f}')
                running_loss = 0.0
        
        # Evaluate on test set
        model.eval()
        with torch.no_grad():
            correct = 0
            total = 0
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
            
            print(f'Epoch [{epoch+1}/{num_epochs}], Test Accuracy: {correct/total:.4f}')

    # Save model weights to the weights folder
    weights = {}
    for name, param in model.named_parameters():
        weights[name] = param.detach().cpu().numpy()

    np.savez('weights/enhanced_cnn_weights.npz', **weights)
    print("Model weights saved to weights/enhanced_cnn_weights.npz")

    # Export some sample data for verification
    sample_inputs, sample_labels = next(iter(test_loader))
    sample_input = sample_inputs[0:5].cpu().numpy()  # Take 5 samples
    sample_label = sample_labels[0:5].cpu().numpy()
    np.savez('weights/sample_data.npz', inputs=sample_input, labels=sample_label)
    print("Sample data saved to weights/sample_data.npz")

    # Also save intermediate outputs from PyTorch model for layer-by-layer verification
    layer_outputs = {}
    def save_intermediate_outputs():
        model.eval()
        with torch.no_grad():
            x = torch.tensor(sample_input, device=device)
            
            # Conv1
            conv1_out = model.conv1(x)
            layer_outputs['conv1'] = conv1_out.detach().cpu().numpy()
            
            # ReLU1
            relu1_out = model.relu1(conv1_out)
            layer_outputs['relu1'] = relu1_out.detach().cpu().numpy()
            
            # Pool1
            pool1_out = model.pool1(relu1_out)
            layer_outputs['pool1'] = pool1_out.detach().cpu().numpy()
            
            # Conv2
            conv2_out = model.conv2(pool1_out)
            layer_outputs['conv2'] = conv2_out.detach().cpu().numpy()
            
            # ReLU2
            relu2_out = model.relu2(conv2_out)
            layer_outputs['relu2'] = relu2_out.detach().cpu().numpy()
            
            # Pool2
            pool2_out = model.pool2(relu2_out)
            layer_outputs['pool2'] = pool2_out.detach().cpu().numpy()
            
            # Flatten
            flatten_out = pool2_out.view(-1, 64 * 7 * 7)
            layer_outputs['flatten'] = flatten_out.detach().cpu().numpy()
            
            # FC1
            fc1_out = model.fc1(flatten_out)
            layer_outputs['fc1'] = fc1_out.detach().cpu().numpy()
            
            # ReLU3
            relu3_out = model.relu3(fc1_out)
            layer_outputs['relu3'] = relu3_out.detach().cpu().numpy()
            
            # FC2
            fc2_out = model.fc2(relu3_out)
            layer_outputs['fc2'] = fc2_out.detach().cpu().numpy()
            
            # Output (same as fc2_out)
            layer_outputs['output'] = fc2_out.detach().cpu().numpy()
            
        np.savez('weights/layer_outputs.npz', **layer_outputs)
        print("Layer outputs saved to weights/layer_outputs.npz")

    save_intermediate_outputs()

if __name__ == "__main__":
    main()