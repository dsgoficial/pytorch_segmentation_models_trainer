# -*- coding: utf-8 -*-
"""Patch uniqueness score via cosine kNN in embedding space."""

import numpy as np
from sklearn.neighbors import NearestNeighbors


def compute_uniqueness_sklearn(embeddings: np.ndarray, k: int = 5) -> np.ndarray:
    """Mean cosine distance to k nearest neighbors using scikit-learn.

    Args:
        embeddings: (N, D) float32 array of patch embeddings.
        k: number of nearest neighbors (excluding self).

    Returns:
        (N,) float array of uniqueness scores; higher = more unique.
    """
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    normalized = embeddings / (norms + 1e-8)

    # k+1 because the query point itself is its nearest neighbor
    nn = NearestNeighbors(
        n_neighbors=k + 1, metric="cosine", algorithm="brute", n_jobs=-1
    )
    nn.fit(normalized)
    distances, _ = nn.kneighbors(normalized)
    # distances[:,0] is distance to self (≈0); skip it
    return distances[:, 1:].mean(axis=1)


def compute_uniqueness_faiss(embeddings: np.ndarray, k: int = 5) -> np.ndarray:
    """Mean cosine distance to k nearest neighbors using faiss (optional).

    Requires: ``pip install faiss-cpu`` or ``faiss-gpu``.

    Args:
        embeddings: (N, D) float32 array of patch embeddings.
        k: number of nearest neighbors (excluding self).

    Returns:
        (N,) float array of uniqueness scores; higher = more unique.
    """
    try:
        import faiss
    except ImportError as exc:
        raise ImportError(
            "faiss is required for the faiss uniqueness backend. "
            "Install with: pip install faiss-cpu"
        ) from exc

    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    normalized = (embeddings / (norms + 1e-8)).astype(np.float32)

    index = faiss.IndexFlatIP(normalized.shape[1])
    index.add(normalized)

    # k+1 because each patch is its own nearest neighbor
    similarities, _ = index.search(normalized, k + 1)
    # similarities are cosine similarities in [-1, 1]; convert to distances
    cosine_dist = 1.0 - similarities[:, 1:]  # exclude self (col 0)
    return cosine_dist.mean(axis=1)


def compute_uniqueness_hnsw(
    embeddings: np.ndarray, k: int = 5, m: int = 32, ef_construction: int = 200
) -> np.ndarray:
    """Mean cosine distance to k nearest neighbors using HNSW (hnswlib).

    O(N log N) build, O(log N) per query — recommended for N > 50k.
    Requires: ``pip install hnswlib``.

    Args:
        embeddings: (N, D) float32 array of patch embeddings.
        k: number of nearest neighbors (excluding self).
        m: HNSW graph branching factor (higher = more accurate, more memory).
        ef_construction: build-time beam width (higher = more accurate, slower build).

    Returns:
        (N,) float array of uniqueness scores; higher = more unique.
    """
    try:
        import hnswlib
    except ImportError as exc:
        raise ImportError(
            "hnswlib is required for the hnsw backend. "
            "Install with: pip install hnswlib"
        ) from exc

    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    normalized = (embeddings / (norms + 1e-8)).astype(np.float32)
    n, d = normalized.shape

    index = hnswlib.Index(space="cosine", dim=d)
    index.init_index(max_elements=n, ef_construction=ef_construction, M=m)
    index.add_items(normalized, num_threads=-1)
    index.set_ef(max(ef_construction, k + 1))

    # Returns (labels, distances) where distance is 1 - cosine_similarity
    _, distances = index.knn_query(normalized, k=k + 1, num_threads=-1)
    # distances[:,0] is distance to self (≈0); skip it
    return distances[:, 1:].mean(axis=1)


def compute_uniqueness(
    embeddings: np.ndarray, k: int = 5, backend: str = "sklearn"
) -> np.ndarray:
    """Mean cosine distance to k nearest neighbors; dispatches to chosen backend.

    Args:
        embeddings: (N, D) float32 array of patch embeddings.
        k: number of nearest neighbors (excluding self).
        backend: 'sklearn' (always available, uses all CPU cores),
                 'faiss' (optional, faster for large N, requires faiss-cpu/faiss-gpu),
                 'hnsw' (optional, O(N log N), recommended for N > 50k, requires hnswlib).

    Returns:
        (N,) float array of uniqueness scores.

    Raises:
        ValueError: if backend is unknown.
    """
    if backend == "sklearn":
        return compute_uniqueness_sklearn(embeddings, k=k)
    if backend == "faiss":
        return compute_uniqueness_faiss(embeddings, k=k)
    if backend == "hnsw":
        return compute_uniqueness_hnsw(embeddings, k=k)
    raise ValueError(
        f"Unknown uniqueness backend: {backend!r}. Choose 'sklearn', 'faiss', or 'hnsw'."
    )
