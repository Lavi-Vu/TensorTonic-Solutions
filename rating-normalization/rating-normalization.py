import numpy as np

def rating_normalization(matrix):
    """
    Mean-center each user's ratings in the user-item matrix.
    """
    # For each row, filter out zeros to get the rated values. Compute their mean. Then iterate over the row: if a value is non-zero, subtract the mean; otherwise keep it as 0.0.
    normalized_matrix = []
    for row in matrix:
        rated_values = [x for x in row if x != 0]
        mean_rating = np.mean(rated_values) if rated_values else 0.0
        normalized_row = [x - mean_rating if x != 0 else 0.0 for x in row]
        normalized_matrix.append(normalized_row)
    return normalized_matrix
