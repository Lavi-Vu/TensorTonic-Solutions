import numpy as np

def expected_value_discrete(x, p):
    """
    Returns: float expected value
    """
    if not np.allclose(np.sum(p), 1.0):
        raise ValueError("Probabilities must sum to 1")
    else:
        ex = np.dot(x, p)
        return ex
