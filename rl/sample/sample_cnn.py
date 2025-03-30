import numpy as np

import os, sys
parent_dir = os.path.abspath(os.path.join(__file__, "../.."))
sys.path.insert(0, parent_dir)

from base_layers import Conv2D, ReLU, MaxPool2D, Flatten, Linear

class NumPyCNN:
    def __init__(self):
        # CNN
        self.conv1 = Conv2D(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.relu1 = ReLU()
        self.pool1 = MaxPool2D(kernel_size=2, stride=2)
        self.conv2 = Conv2D(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.relu2 = ReLU()
        self.pool2 = MaxPool2D(kernel_size=2, stride=2)
        self.flatten = Flatten()
        self.fc1 = Linear(in_features=64*7*7, out_features=128)
        self.relu3 = ReLU()
        self.fc2 = Linear(in_features=128, out_features=10)  # 10 classes for MNIST
    
    def load_weights(self, weights_file):
        """Load weights from a saved NumPy file"""
        weights = np.load(weights_file)
        
        # Load weights for each layer
        self.conv1.load_weights(weights['conv1.weight'], weights['conv1.bias'])
        self.conv2.load_weights(weights['conv2.weight'], weights['conv2.bias'])
        self.fc1.load_weights(weights['fc1.weight'], weights['fc1.bias'])
        self.fc2.load_weights(weights['fc2.weight'], weights['fc2.bias'])
    
    def forward(self, x, save_intermediates=False):
        """Forward pass through the entire model"""
        intermediates = {}
        
        # Apply each layer in sequence
        x = self.conv1.forward(x)
        if save_intermediates: intermediates['conv1'] = x.copy()
        
        x = self.relu1.forward(x)
        if save_intermediates: intermediates['relu1'] = x.copy()
        
        x = self.pool1.forward(x)
        if save_intermediates: intermediates['pool1'] = x.copy()
        
        x = self.conv2.forward(x)
        if save_intermediates: intermediates['conv2'] = x.copy()
        
        x = self.relu2.forward(x)
        if save_intermediates: intermediates['relu2'] = x.copy()
        
        x = self.pool2.forward(x)
        if save_intermediates: intermediates['pool2'] = x.copy()
        
        # Flatten for fully connected layer
        x = self.flatten.forward(x)
        if save_intermediates: intermediates['flatten'] = x.copy()
        
        # First fully connected layer
        x = self.fc1.forward(x)
        if save_intermediates: intermediates['fc1'] = x.copy()
        
        # ReLU after first fully connected layer
        x = self.relu3.forward(x)
        if save_intermediates: intermediates['relu3'] = x.copy()
        
        # Second fully connected layer
        x = self.fc2.forward(x)
        if save_intermediates: intermediates['fc2'] = x.copy()
        
        if save_intermediates:
            intermediates['output'] = x.copy()
            return x, intermediates
        else:
            return x

def predict(model, input_data):
    """Make predictions using the NumPy model"""
    outputs = model.forward(input_data)
    # Apply softmax to get probabilities
    exp_outputs = np.exp(outputs - np.max(outputs, axis=1, keepdims=True))
    probabilities = exp_outputs / np.sum(exp_outputs, axis=1, keepdims=True)
    # Get class with highest probability
    predictions = np.argmax(probabilities, axis=1)
    return predictions, probabilities