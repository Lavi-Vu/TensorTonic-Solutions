import numpy as np

def positional_encoding(seq_length: int, d_model: int) -> np.ndarray:
    """
    Generate sinusoidal positional encodings.
    """
    # Create position indices: [0, 1, 2, ..., seq_length-1]
    position = np.arange(seq_length)[:, np.newaxis]          # shape: (seq_length, 1)
    
    # Create dimension indices: [0, 1, 2, ..., d_model-1]
    div_term = np.exp(np.arange(0, d_model, 2) * 
                     (-np.log(10000.0) / d_model))           # shape: (d_model // 2,)
    
    # Compute sinusoidal encodings
    pe = np.zeros((seq_length, d_model))
    
    # Even dimensions: sin(position * div_term)
    pe[:, 0::2] = np.sin(position * div_term)
    
    # Odd dimensions: cos(position * div_term)
    pe[:, 1::2] = np.cos(position * div_term)
    
    return pe
