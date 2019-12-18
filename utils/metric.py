"""
metric.py
Based on
https://github.com/rshk/elo/blob/master/elo.py
"""

import numpy as np

def elo_expected(A, B):
    """
    Calculate expected score of A in a match against B
    :param A: Elo rating for player A
    :param B: Elo rating for player B
    """
    return 1. / (1 + 10 ** ((B - A) / 400.))


def get_elo_score(old, exp, score, k=32):
    """
    Calculate the new Elo rating for a player
    :param old: The previous Elo rating
    :param exp: The expected score for this match
    :param score: The actual score for this match
    :param k: The k-factor for Elo (default: 32)
    """
    return int(old + k * (score - exp))

def elo(a_score, b_score, results, k=32):
    """
    Return an elo score for players a and b according to results
    a_score : int
        elo score of a
    b_score : int
        elo score of b
    results : x, list of xs, np.ndarray of xs
        where x is in [0, 0.5, 1]
        results from game, where 0 is loss for a,
        0.5 is draw, 1 is win for a. Ignores values that are not above

    Returns
    new_a_score : int
        new elo score for a
    new_b_score : int
        new elo score for b
    """
    # convert to list
    if not isinstance(results, (list, np.ndarray)):
        results = [results]

    new_a_score = a_score
    new_b_score = b_score

    allowed_results = [0, 0.5, 1]
    for result in results:
        if result not in allowed_results: continue

        exp_a = elo_expected(new_a_score, new_b_score)
        exp_b = elo_expected(new_b_score, new_a_score)

        new_a_score = get_elo_score(new_a_score, exp_a, result, k=k)
        new_b_score = get_elo_score(new_b_score, exp_b, 1 - result, k=k)

    return new_a_score, new_b_score
