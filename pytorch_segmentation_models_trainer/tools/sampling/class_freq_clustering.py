# -*- coding: utf-8 -*-
"""CLR transform + K-Means clustering on patch class frequency vectors."""

from typing import Dict, Optional

import numpy as np
from sklearn.cluster import KMeans


def clr_transform(freq_matrix: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Centered log-ratio transform for compositional data.

    Args:
        freq_matrix: (N, C) array of class proportions (rows need not sum to 1 exactly).
        eps: small constant added before log to avoid log(0).

    Returns:
        (N, C) CLR-transformed array; each row sums to 0.
    """
    x = freq_matrix + eps
    log_x = np.log(x)
    geometric_mean_log = log_x.mean(axis=1, keepdims=True)
    return log_x - geometric_mean_log


def select_k_elbow(inertias: list, k_min: int = 8) -> int:
    """Return the K at maximum curvature (elbow) of the inertia curve.

    Args:
        inertias: list of inertia values for K = k_min, k_min+1, ...
        k_min: first K value corresponding to inertias[0].

    Returns:
        Optimal K selected by the perpendicular-distance heuristic.
    """
    if len(inertias) == 1:
        return k_min

    n = len(inertias)
    x = np.arange(n, dtype=float)
    y = np.array(inertias, dtype=float)

    # Normalize both axes to [0, 1] so curvature is scale-independent
    x_norm = (x - x.min()) / (x.max() - x.min() + 1e-12)
    y_norm = (y - y.min()) / (y.max() - y.min() + 1e-12)

    # Direction vector from first to last point
    dx = x_norm[-1] - x_norm[0]
    dy = y_norm[-1] - y_norm[0]
    line_len = np.sqrt(dx**2 + dy**2) + 1e-12

    # Perpendicular distance from each point to the line
    distances = np.abs(dy * x_norm - dx * y_norm + dx * y_norm[0] - dy * x_norm[0])
    distances /= line_len

    elbow_idx = int(np.argmax(distances))
    return k_min + elbow_idx


def compute_cluster_weights(cluster_ids: np.ndarray) -> np.ndarray:
    """Compute per-patch sampling weight as 1 / cluster_size.

    Args:
        cluster_ids: (N,) integer cluster assignment per patch.

    Returns:
        (N,) float array of weights; patches in smaller clusters get larger weights.
    """
    unique_ids, counts = np.unique(cluster_ids, return_counts=True)
    size_map = {int(cid): int(cnt) for cid, cnt in zip(unique_ids, counts)}
    weights = np.array(
        [1.0 / size_map[int(cid)] for cid in cluster_ids], dtype=np.float64
    )
    return weights


def fit_kmeans_on_clr(
    freq_matrix: np.ndarray,
    k_min: int = 8,
    k_max: int = 20,
    k_selection: str = "elbow",
    k: Optional[int] = None,
    random_state: int = 42,
) -> Dict:
    """Fit K-Means on CLR-transformed class frequency vectors.

    Args:
        freq_matrix: (N, C) array of per-patch class proportions.
        k_min: minimum K for elbow search.
        k_max: maximum K for elbow search.
        k_selection: 'elbow' to auto-select K; 'fixed' to use the k argument.
        k: K to use when k_selection='fixed'.
        random_state: random seed for KMeans reproducibility.

    Returns:
        dict with:
            cluster_ids (np.ndarray, shape N): cluster assignment per patch.
            cluster_sizes (np.ndarray, shape N): size of each patch's cluster.
            cluster_weights (np.ndarray, shape N): 1/cluster_size per patch.
    """
    clr = clr_transform(freq_matrix)

    if k_selection == "fixed":
        assert k is not None, "k must be provided when k_selection='fixed'"
        best_k = k
    else:
        inertias = []
        k_range = range(k_min, k_max + 1)
        for ki in k_range:
            km = KMeans(n_clusters=ki, random_state=random_state, n_init="auto")
            km.fit(clr)
            inertias.append(km.inertia_)
        best_k = select_k_elbow(inertias, k_min=k_min)

    km_final = KMeans(n_clusters=best_k, random_state=random_state, n_init="auto")
    km_final.fit(clr)
    cluster_ids = km_final.labels_

    weights = compute_cluster_weights(cluster_ids)
    unique_ids, counts = np.unique(cluster_ids, return_counts=True)
    size_map = {int(cid): int(cnt) for cid, cnt in zip(unique_ids, counts)}
    cluster_sizes = np.array([size_map[int(cid)] for cid in cluster_ids])

    return {
        "cluster_ids": cluster_ids,
        "cluster_sizes": cluster_sizes,
        "cluster_weights": weights,
    }
