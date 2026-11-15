def first_rank(retrieved_articles, correct):
    """Position (1, 2, 3...) where the correct article first appears, or None."""
    for rank, art in enumerate(retrieved_articles, start=1):
        if art == correct:
            return rank
    return None

def hit_at_k(ranks, k):
    """Share of questions where the correct article is in the top k."""
    return sum(1 for r in ranks if r is not None and r <= k) / len(ranks)

def mrr(ranks):
    """Mean Reciprocal Rank: rank 1 -> 1.0, rank 2 -> 0.5, rank 10 -> 0.1, not found -> 0."""
    return sum(1 / r for r in ranks if r is not None) / len(ranks)