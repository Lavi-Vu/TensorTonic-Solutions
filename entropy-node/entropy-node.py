import numpy as np

def entropy_node(y):
    """
    Compute entropy for a single node using stable logarithms.
    """
    H = 0.0
    y = np.array(y)
    p = np.unique(y, return_counts=True)
    for item in p[0]:
        count = np.sum(y == item)
        prob = count / len(y)
        if prob > 0:
            H += -prob * np.log2(prob)
    return H
