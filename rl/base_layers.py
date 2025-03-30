import numpy as np

# -------------------- Convolution Layer --------------------
class Conv2D:
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        
        # Initialize weights and biases
        self.weight = None
        self.bias = None
    
    def load_weights(self, weight, bias):
        """Load pre-trained weights from PyTorch model"""
        self.weight = weight
        self.bias = bias
    
    def forward(self, x):
        """Forward pass of the convolution layer"""
        N, C, H, W = x.shape
        
        # Calculate output dimensions
        H_out = ((H + 2 * self.padding - self.kernel_size) // self.stride) + 1
        W_out = ((W + 2 * self.padding - self.kernel_size) // self.stride) + 1
        
        # Initialize output
        output = np.zeros((N, self.out_channels, H_out, W_out))
        
        # Pad input if needed
        if self.padding > 0:
            x_padded = np.pad(x, ((0, 0), (0, 0), (self.padding, self.padding), (self.padding, self.padding)), 'constant')
        else:
            x_padded = x
        
        # Perform convolution
        for n in range(N):  # Batch dimension
            for c_out in range(self.out_channels):  # Output channels
                for h_out in range(H_out):  # Output height
                    for w_out in range(W_out):  # Output width
                        h_start = h_out * self.stride
                        w_start = w_out * self.stride
                        h_end = h_start + self.kernel_size
                        w_end = w_start + self.kernel_size
                        
                        # Extract the window from padded input
                        window = x_padded[n, :, h_start:h_end, w_start:w_end]
                        
                        # Perform the convolution operation
                        output[n, c_out, h_out, w_out] = np.sum(window * self.weight[c_out]) + self.bias[c_out]
        
        return output

# -------------------- ReLU Activation Layer --------------------
class ReLU:
    def forward(self, x):
        """Forward pass for ReLU activation"""
        return np.maximum(0, x)

# -------------------- MaxPooling Layer --------------------
class MaxPool2D:
    def __init__(self, kernel_size, stride=None):
        self.kernel_size = kernel_size
        self.stride = stride if stride is not None else kernel_size
    
    def forward(self, x):
        """Forward pass for MaxPool operation"""
        N, C, H, W = x.shape
        
        # Calculate output dimensions
        H_out = ((H - self.kernel_size) // self.stride) + 1
        W_out = ((W - self.kernel_size) // self.stride) + 1
        
        # Initialize output
        output = np.zeros((N, C, H_out, W_out))
        
        # Perform max pooling
        for n in range(N):
            for c in range(C):
                for h_out in range(H_out):
                    for w_out in range(W_out):
                        h_start = h_out * self.stride
                        w_start = w_out * self.stride
                        h_end = h_start + self.kernel_size
                        w_end = w_start + self.kernel_size
                        
                        # Extract window and find maximum
                        window = x[n, c, h_start:h_end, w_start:w_end]
                        output[n, c, h_out, w_out] = np.max(window)
        
        return output

# -------------------- Flatten Layer --------------------
class Flatten:
    def forward(self, x):
        """Forward pass for flattening operation"""
        batch_size = x.shape[0]
        return x.reshape(batch_size, -1)

# -------------------- Linear (Fully Connected) Layer --------------------
class Linear:
    def __init__(self, in_features, out_features):
        self.in_features = in_features
        self.out_features = out_features
        
        # Initialize weights and biases
        self.weight = None
        self.bias = None
    
    def load_weights(self, weight, bias):
        """Load pre-trained weights from PyTorch model"""
        self.weight = weight
        self.bias = bias
    
    def forward(self, x):
        """Forward pass for fully connected layer"""
        # Ensure x is 2D (batch_size, features)
        if len(x.shape) > 2:
            # Flatten all dimensions except batch
            x = x.reshape(x.shape[0], -1)
        
        # Matrix multiplication and bias addition
        return np.dot(x, self.weight.T) + self.bias

# -------------------- Softmax Layer --------------------
class Softmax:
    def forward(self, x, axis=-1):
        """Forward pass for softmax activation
        
        Args:
            x: Input tensor
            axis: Axis along which softmax is computed (default: -1)
        """
        # Subtract max for numerical stability
        x_max = np.max(x, axis=axis, keepdims=True)
        exp_x = np.exp(x - x_max)
        return exp_x / np.sum(exp_x, axis=axis, keepdims=True)

# -------------------- Tanh Activation Layer --------------------
class Tanh:
    def forward(self, x):
        """Forward pass for Tanh activation"""
        return np.tanh(x)

# -------------------- LayerNorm (Layer Normalization) --------------------
class LayerNorm:
    def __init__(self, normalized_shape, eps=1e-5):
        """Initialize Layer Normalization
        
        Args:
            normalized_shape: Shape of the tensor to normalize
            eps: Small constant to prevent division by zero
        """
        self.normalized_shape = normalized_shape
        self.eps = eps
        
        # Learnable parameters
        self.weight = None  # gamma
        self.bias = None    # beta
    
    def load_weights(self, weight, bias):
        """Load pre-trained weights from PyTorch model"""
        self.weight = weight
        self.bias = bias
    
    def forward(self, x):
        """Forward pass for LayerNorm
        
        This normalizes the last dimension by default
        """
        # Computing mean and variance along the last dimension
        mean = np.mean(x, axis=-1, keepdims=True)
        var = np.var(x, axis=-1, keepdims=True)
        
        # Normalize
        x_norm = (x - mean) / np.sqrt(var + self.eps)
        
        # Scale and shift with learnable parameters
        if self.weight is not None and self.bias is not None:
            return self.weight * x_norm + self.bias
        else:
            return x_norm

# -------------------- Concatenate Layer --------------------
class Concatenate:
    def forward(self, tensors, axis=1):
        """Concatenate a list of tensors along the specified axis
        
        Args:
            tensors: List of tensors to concatenate
            axis: Axis along which to concatenate (default: 1)
        """
        return np.concatenate(tensors, axis=axis)

# -------------------- Dictionary Observer Handler --------------------
class DictObsHandler:
    def __init__(self, image_extractor, action_mask_extractor, combiner):
        self.image_extractor = image_extractor
        self.action_mask_extractor = action_mask_extractor 
        self.combiner = combiner
    
    def forward(self, observation):
        # Process image and mask separately
        image_features = self.image_extractor.forward(observation['image'])
        mask_features = self.action_mask_extractor.forward(observation['action_mask'])
        
        # Combine features using the combiner
        combined = self.combiner.forward([image_features, mask_features])
        
        # If combined is a list (as when using a simple MockLayer that just returns its input),
        # convert it to a dictionary to match the Torch interface.
        if isinstance(combined, list):
            return {'image': combined[0], 'action_mask': combined[1]}
        return combined

# -------------------- Sequential Layer --------------------
class Sequential:
    def __init__(self, layers):
        """Initialize a sequence of layers
        
        Args:
            layers: List of layer objects with a forward method
        """
        self.layers = layers
    
    def forward(self, x):
        """Forward pass through all layers in sequence
        
        Args:
            x: Input tensor
        """
        for layer in self.layers:
            x = layer.forward(x)
        return x