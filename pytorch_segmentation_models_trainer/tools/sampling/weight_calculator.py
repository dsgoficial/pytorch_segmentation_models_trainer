# -*- coding: utf-8 -*-
"""Combine compositional and uniqueness scores into a sampler_weight column."""

import numpy as np


def combine_weights(
    comp_score: np.ndarray,
    uniq_score: np.ndarray,
    method: str = "rank_max",
) -> np.ndarray:
    """Combine compositional rarity and visual uniqueness scores into sampler weights.

    Both input arrays are first converted to percentile ranks in [0, 1] before
    combination (except for 'multiplicative' which uses raw scores directly).

    Args:
        comp_score: (N,) compositional rarity score per patch (e.g. cluster weight).
        uniq_score: (N,) visual uniqueness score per patch (e.g. mean cosine dist).
        method: one of 'rank_max', 'rank_multiply', 'rank_add', 'multiplicative',
            'composition_only', 'uniqueness_only'.

    Returns:
        (N,) sampler weight array ready for torch.utils.data.WeightedRandomSampler.

    Raises:
        ValueError: if method is not recognised.
    """
    n = len(comp_score)
    rank_comp = np.argsort(np.argsort(comp_score)) / max(n - 1, 1)
    rank_uniq = np.argsort(np.argsort(uniq_score)) / max(n - 1, 1)

    if method == "rank_max":
        return np.maximum(rank_comp, rank_uniq)
    if method == "rank_multiply":
        return rank_comp * rank_uniq
    if method == "rank_add":
        return (rank_comp + rank_uniq) / 2.0
    if method == "multiplicative":
        return comp_score * uniq_score
    if method == "composition_only":
        return rank_comp
    if method == "uniqueness_only":
        return rank_uniq
    raise ValueError(
        f"Unknown sampling_method: {method!r}. "
        "Valid options: rank_max, rank_multiply, rank_add, multiplicative, "
        "composition_only, uniqueness_only."
    )
