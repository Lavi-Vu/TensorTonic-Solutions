import numpy as np

def _sigmoid(z):
    """Numerically stable sigmoid implementation."""
    return np.where(z >= 0, 1/(1+np.exp(-z)), np.exp(z)/(1+np.exp(z)))

def loss(y_true, y_pred):
    eps = 1e-15
    y_pred = np.clip(y_pred, eps, 1. - eps)
    loss = -(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
    return np.mean(loss)
    
def train_logistic_regression(X, y, lr=0.1, steps=1000):
    """
    Train logistic regression via gradient descent.
    Return (w, b).
    """
    n_samples, n_features = X.shape

    # init weight and bias
    w = np.zeros(n_features)
    b = 0.0

    for step in range(steps):
        # Forward pass
        z = X @ w + b                  # shape: (n_samples,)
        y_pred = _sigmoid(z)            # shape: (n_samples,)
        
        # Compute gradients
        error = y_pred - y             # shape: (n_samples,)
        grad_w = (X.T @ error) / n_samples   # shape: (n_features,)
        grad_b = np.mean(error)              # scalar
        
        # Update parameters
        w -= lr * grad_w
        b -= lr * grad_b
    
    return w, b
            