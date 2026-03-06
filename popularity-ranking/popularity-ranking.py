def popularity_ranking(items, min_votes, global_mean):
    """
    Compute the Bayesian weighted rating for each item.
    """
    # Write code here
    weighted_ratings = []
    for item in items:
        ratings, votes = item
        if votes  == 0:
            weighted_ratings.append(global_mean)
        else:
            weighted_rating = (votes / (votes + min_votes)) * ratings + (min_votes / (votes + min_votes)) * global_mean
            weighted_ratings.append(weighted_rating)
    return weighted_ratings
