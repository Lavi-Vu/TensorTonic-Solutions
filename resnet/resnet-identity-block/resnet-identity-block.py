import numpy as np

def relu(x):
    return np.maximum(0, x)

class IdentityBlock:
    """
    Identity Block: F(x) + x
    Used when input and output dimensions match.
    """
    
    def __init__(self, channels: int):
        self.channels = channels
        # Simplified: using dense layers instead of conv for demo
        self.W1 = np.random.randn(channels, channels) * 0.01
        self.W2 = np.random.randn(channels, channels) * 0.01
    
    def forward(self, x):
        # assuming x.shape = (batch, channels)
        residual = x
        x = relu(x @ self.W1)       # prefer data @ weight
        x = x @ self.W2
        return relu(x) + residual
