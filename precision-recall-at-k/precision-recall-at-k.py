def precision_recall_at_k(recommended, relevant, k):
    """
    Compute precision@k and recall@k for a recommendation list.
    """
    # Write code here
    recommended_k = recommended[:k]
    relevant_set = set(relevant)
    recommended_set = set(recommended_k)
    
    # Compute intersection of recommended and relevant items
    intersection = recommended_set & relevant_set
    
    # Compute precision@k
    precision = len(intersection) / k if k > 0 else 0.0
    
    # Compute recall@k
    recall = len(intersection) / len(relevant_set) if len(relevant_set) > 0 else 0.0
    
    return [precision, recall]