import numpy as np

def positional_encoding(seq_len, d_model, base=10000.0):
    """
    Return PE of shape (seq_len, d_model) using sin/cos formulation.
    Odd d_model -> last column is sin.
    """
    positions = np.arange(seq_len, dtype=float).reshape(-1, 1)
    _2i = np.arange(0, d_model, 2).astype(float)  # (⌈d/2⌉,)
    pe = np.zeros((seq_len, d_model))
    pe[:, 0::2] = np.sin(positions / base ** (_2i / d_model))  
    pe[:, 1::2] = np.cos((positions / base ** (_2i / d_model))[:,:d_model//2])  
    return pe
