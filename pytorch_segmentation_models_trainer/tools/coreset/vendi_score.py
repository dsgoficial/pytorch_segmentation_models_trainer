# -*- coding: utf-8 -*-
"""Vendi score computation for FD K-selection (optional path)."""

import numpy as np


def vendi_score(embeddings: np.ndarray) -> float:
    """Compute Vendi score using a cosine kernel matrix.

    Vendi score = exp(H(K / tr(K))) where K is the cosine kernel matrix and
    H is the Shannon entropy of the normalised eigenvalue spectrum.

    Args:
        embeddings: (N, D) float array of patch embeddings.

    Returns:
        Vendi score ≥ 1.0; higher = more diverse.

    References:
        Friedman & Dieng (2023) "The Vendi Score: A Diversity Evaluation Metric
        for Machine Learning." TMLR.
    """
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    e = embeddings / (norms + 1e-8)

    K = e @ e.T  # cosine similarity matrix (N, N)

    tr = np.trace(K)
    if tr <= 0:
        return 1.0

    K_norm = K / tr
    eigvals = np.linalg.eigvalsh(K_norm)
    eigvals = eigvals[eigvals > 1e-10]

    H = float(-np.sum(eigvals * np.log(eigvals)))
    return float(np.exp(H))


def select_k_vendi(
    embeddings: np.ndarray,
    k_min: int = 2,
    delta: float = 0.005,
    max_subsample: int = 200,
) -> int:
    """Select K for K-Means using per-cluster Vendi score improvement.

    Increments K from ``k_min``, clusters the embeddings, and measures the
    average per-cluster Vendi score.  Stops when improvement is below
    ``delta`` for 3 consecutive steps.

    Args:
        embeddings: (N, D) float array.
        k_min: minimum K to try.
        delta: minimum improvement threshold per step.
        max_subsample: maximum per-cluster sample size for speed.

    Returns:
        Selected K.
    """
    from sklearn.cluster import KMeans

    N = len(embeddings)
    k_min = max(k_min, 2)
    max_k = min(N - 1, 50)

    if k_min >= max_k:
        return k_min

    rng = np.random.RandomState(42)
    prev_score: float = None
    consec_no_improve = 0
    best_k = k_min

    for k in range(k_min, max_k + 1):
        km = KMeans(n_clusters=k, random_state=42, n_init="auto")
        labels = km.fit_predict(embeddings)

        cluster_scores = []
        for cid in range(k):
            mask = labels == cid
            sub = embeddings[mask]
            if len(sub) < 2:
                cluster_scores.append(1.0)
                continue
            if len(sub) > max_subsample:
                idx = rng.choice(len(sub), max_subsample, replace=False)
                sub = sub[idx]
            cluster_scores.append(vendi_score(sub))

        avg = float(np.mean(cluster_scores))

        if prev_score is not None:
            improvement = avg - prev_score
            if improvement < delta:
                consec_no_improve += 1
                if consec_no_improve >= 3:
                    break
            else:
                consec_no_improve = 0
                best_k = k

        prev_score = avg

    return best_k
