# -*- coding: utf-8 -*-
"""Core-set selection scoring functions (Nogueira et al. 2026, IEEE Access).

All public functions share the signature:
    score_*(df, config) -> np.ndarray  shape (N,), values in [0, 1]
Higher score = more valuable patch for the core-set.
"""

import json

import numpy as np
import pandas as pd
import torch
from sklearn.cluster import KMeans
from tqdm import tqdm

from pytorch_segmentation_models_trainer.tools.coreset.coreset_config import (
    CoreSetConfig,
)
from pytorch_segmentation_models_trainer.tools.sampling.class_freq_clustering import (
    select_k_elbow,
)

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _get_embeddings(df: pd.DataFrame, column: str) -> np.ndarray:
    """Stack embedding column to (N, D) float32 array."""
    return np.stack([np.asarray(e, dtype=np.float32) for e in df[column].values])


def _parse_class_dist(series: pd.Series):
    """Parse class_dist_json column to a list of dicts with string keys."""
    result = []
    for v in series:
        if isinstance(v, str):
            result.append(json.loads(v))
        elif isinstance(v, dict):
            result.append(v)
        else:
            result.append({})
    return result


def _build_freq_matrix(class_dists: list):
    """Build (N, C) class-proportion matrix from a list of class_dist dicts.

    Args:
        class_dists: list of dicts mapping class id (any type) → proportion.

    Returns:
        freq_matrix: (N, C) float64 ndarray.
        all_classes: sorted list of string class ids (column labels).
    """
    all_classes = sorted(set(str(k) for d in class_dists for k in d.keys()))
    C = len(all_classes)
    N = len(class_dists)
    class_to_idx = {k: i for i, k in enumerate(all_classes)}
    freq_matrix = np.zeros((N, C), dtype=np.float64)
    for i, d in enumerate(class_dists):
        for k, v in d.items():
            idx = class_to_idx.get(str(k))
            if idx is not None:
                freq_matrix[i, idx] = float(v)
    return freq_matrix, all_classes


def _get_device(config: CoreSetConfig) -> torch.device:
    """Resolve compute device from config.

    Args:
        config: CoreSetConfig with optional ``device`` field.

    Returns:
        torch.device — cuda when available and not overridden, else cpu.
    """
    if config.device is None:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(config.device)


def _build_spatial_matrix(df: pd.DataFrame) -> np.ndarray:
    """Build (N, 2) spatial feature matrix [lat_norm, lon_norm] from tile bounds.

    Computes the tile centroid from bounding box columns and min-max normalises
    each coordinate to [0, 1] within the pool so that spatial features are on
    the same scale as class proportions or embedding dimensions.

    Args:
        df: DataFrame with ``tile_minx``, ``tile_maxx``, ``tile_miny``,
            ``tile_maxy`` columns (any coordinate system).

    Returns:
        (N, 2) float64 array: ``[:, 0]`` = lat_norm, ``[:, 1]`` = lon_norm.

    Raises:
        ValueError: if any required tile-bounds column is absent.
    """
    required = {"tile_minx", "tile_maxx", "tile_miny", "tile_maxy"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"spatial augmentation requires columns {sorted(missing)} in the DataFrame."
        )
    lat = (df["tile_miny"].values + df["tile_maxy"].values) / 2.0
    lon = (df["tile_minx"].values + df["tile_maxx"].values) / 2.0

    def _minmax_norm(arr):
        lo, hi = arr.min(), arr.max()
        if hi - lo < 1e-10:
            return np.zeros_like(arr, dtype=np.float64)
        return (arr - lo) / (hi - lo)

    return np.column_stack([_minmax_norm(lat), _minmax_norm(lon)])


def _augment_embedding_with_spatial(F: np.ndarray, df: pd.DataFrame) -> np.ndarray:
    """Append min-max-normalised centroid coords to an embedding matrix.

    Args:
        F: (N, D) embedding array (any dtype).
        df: DataFrame with tile bound columns consumed by ``_build_spatial_matrix``.

    Returns:
        (N, D+2) float64 array.
    """
    spatial = _build_spatial_matrix(df)
    return np.hstack([F, spatial])


def _round_robin_rank(
    cluster_ids: np.ndarray, K: int, N: int, seed: int = 42
) -> np.ndarray:
    """Assign selection rank via round-robin across clusters.

    The first patch selected gets rank N, the last gets rank 1.
    Unselected patches (impossible here; all patches eventually selected) get 0.

    Args:
        cluster_ids: (N,) integer cluster assignment per patch.
        K: number of clusters.
        N: total number of patches.
        seed: random seed for cluster order and within-cluster shuffle.

    Returns:
        (N,) float64 rank array.
    """
    rng = np.random.RandomState(seed)

    cluster_to_patches = {k: [] for k in range(K)}
    for i, cid in enumerate(cluster_ids):
        cluster_to_patches[int(cid)].append(i)

    cluster_order = rng.permutation(K).tolist()
    shuffled_lists = []
    for k in cluster_order:
        lst = list(cluster_to_patches[k])
        rng.shuffle(lst)
        shuffled_lists.append(lst)

    rank = np.zeros(N, dtype=np.float64)
    t = N - 1  # first selected gets rank N-1, last gets rank 0
    while any(shuffled_lists):
        for lst in shuffled_lists:
            if lst and t >= 0:
                patch_idx = lst.pop(0)
                rank[patch_idx] = t
                t -= 1

    return rank


# ---------------------------------------------------------------------------
# Public scorers
# ---------------------------------------------------------------------------


def score_lc(df: pd.DataFrame, config: CoreSetConfig) -> np.ndarray:
    """Label Complexity (LC) scorer — normalized class entropy.

    Normalises each patch's ``class_entropy`` by the maximum entropy observed
    in the dataset (not by a theoretical constant), so patches with the highest
    label complexity receive score 1.0.

    Args:
        df: DataFrame containing a ``class_entropy`` column.
        config: CoreSetConfig (method-independent fields used here).

    Returns:
        (N,) float array in [0, 1]; higher = more label-complex.

    Raises:
        ValueError: if ``class_entropy`` is not present in df.
    """
    if "class_entropy" not in df.columns:
        raise ValueError(
            "LC method requires a 'class_entropy' column in the DataFrame."
        )
    entropy = df["class_entropy"].fillna(0.0).values.astype(np.float64)
    max_e = entropy.max()
    if max_e == 0:
        return np.zeros(len(df))
    return entropy / max_e


def score_cb(df: pd.DataFrame, config: CoreSetConfig) -> np.ndarray:
    """Class Balance (CB) scorer — greedy class-entropy maximization.

    Selects patches in the order that maximally increases global class entropy
    at each greedy step.  Early-stops after ``budget_count`` selections to
    achieve O(b × N) complexity instead of O(N²).  Unselected patches get
    score 0.

    Args:
        df: DataFrame containing a ``class_dist_json`` column.
        config: CoreSetConfig; ``budget`` and ``budget_mode`` control early-stop.

    Returns:
        (N,) float array in [0, 1]; 1 = first selected, 0 = not selected.

    Raises:
        ValueError: if ``class_dist_json`` is not present in df.
    """
    if "class_dist_json" not in df.columns:
        raise ValueError(
            "CB method requires a 'class_dist_json' column in the DataFrame."
        )

    N = len(df)
    if N == 1:
        return np.array([1.0])

    class_dists = _parse_class_dist(df["class_dist_json"])
    freq_matrix, _ = _build_freq_matrix(class_dists)

    if config.cb_use_spatial:
        spatial = _build_spatial_matrix(df)
        freq_matrix = np.hstack([freq_matrix, spatial])

    if config.budget_mode == "fraction":
        budget_count = max(1, int(config.budget * N))
    else:
        budget_count = max(1, int(config.budget))
    budget_count = min(budget_count, N)

    device = _get_device(config)
    desc = "CB spatial" if config.cb_use_spatial else "CB"

    if device.type == "cuda":
        rank = _score_cb_gpu(freq_matrix, budget_count, N, device, desc)
    else:
        rank = _score_cb_cpu(freq_matrix, budget_count, N, desc)

    return rank / (N - 1)


def _score_cb_cpu(
    freq_matrix: np.ndarray, budget_count: int, N: int, desc: str
) -> np.ndarray:
    rank = np.zeros(N, dtype=np.float64)
    selected = np.zeros(N, dtype=bool)
    cumulative = np.zeros(freq_matrix.shape[1], dtype=np.float64)
    eps = 1e-12
    for t in tqdm(range(1, budget_count + 1), desc=desc, unit="patch", leave=True):
        unsel = np.where(~selected)[0]
        candidates = cumulative + freq_matrix[unsel]
        totals = candidates.sum(axis=1, keepdims=True)
        totals = np.where(totals == 0, 1.0, totals)
        props = candidates / totals
        entropies = -np.sum(props * np.log(props + eps), axis=1)
        best_local = int(np.argmax(entropies))
        best_i = unsel[best_local]
        rank[best_i] = N - t
        selected[best_i] = True
        cumulative += freq_matrix[best_i]
    return rank


def _score_cb_gpu(
    freq_matrix: np.ndarray,
    budget_count: int,
    N: int,
    device: torch.device,
    desc: str,
) -> np.ndarray:
    F = torch.tensor(freq_matrix, dtype=torch.float32, device=device)
    cumulative = torch.zeros(F.shape[1], dtype=torch.float32, device=device)
    selected = torch.zeros(N, dtype=torch.bool, device=device)
    rank = np.zeros(N, dtype=np.float64)
    eps = 1e-12
    for t in tqdm(
        range(1, budget_count + 1), desc=f"{desc} [GPU]", unit="patch", leave=True
    ):
        unsel_idx = torch.where(~selected)[0]
        candidates = cumulative + F[unsel_idx]
        totals = candidates.sum(dim=1, keepdim=True)
        totals = torch.where(totals == 0, torch.ones_like(totals), totals)
        props = candidates / totals
        entropies = -(props * torch.log(props + eps)).sum(dim=1)
        best_local = int(torch.argmax(entropies).item())
        best_i = int(unsel_idx[best_local].item())
        rank[best_i] = N - t
        selected[best_i] = True
        cumulative += F[best_i]
    return rank


def score_fa(df: pd.DataFrame, config: CoreSetConfig) -> np.ndarray:
    """Feature Activation (FA) scorer using the paper's gamma formula.

    Computes per-patch embedding mean (``mu``) and standard deviation
    (``sigma``), evaluates ``gamma_i = -(1 - mu_i) * log(sigma_i)``, and
    returns the inverse min-max-normalised gamma score:
    ``s_i^FA = 1 - (gamma_i - min(gamma)) / (max(gamma) - min(gamma))``.

    The implementation mirrors the reference code's ``log(sigma + eps)`` guard
    to avoid undefined ``log(0)`` behaviour.

    Args:
        df: DataFrame containing the embedding column.
        config: CoreSetConfig; ``embedding_column`` must be set.

    Returns:
        (N,) float array in [0, 1].

    Raises:
        ValueError: if ``embedding_column`` is None or absent from df.
    """
    if config.embedding_column is None:
        raise ValueError(
            "FA method requires embedding_column to be set in CoreSetConfig."
        )
    if config.embedding_column not in df.columns:
        raise ValueError(
            f"Embedding column '{config.embedding_column}' not found in DataFrame."
        )

    F = _get_embeddings(df, config.embedding_column)
    if config.fd_use_spatial:
        F = _augment_embedding_with_spatial(F, df)
    mu = F.mean(axis=1)
    sigma = F.std(axis=1)

    eps = 1e-12
    gamma = -(1.0 - mu.astype(np.float64)) * np.log(sigma.astype(np.float64) + eps)
    gamma_min = float(gamma.min())
    gamma_max = float(gamma.max())
    denom = gamma_max - gamma_min
    if denom < 1e-8:
        return np.zeros(len(df), dtype=np.float64)

    scores = 1.0 - (gamma - gamma_min) / denom
    return np.clip(scores, 0.0, 1.0)


def score_fd(df: pd.DataFrame, config: CoreSetConfig) -> np.ndarray:
    """Feature Diversity (FD) scorer — round-robin cluster selection.

    Clusters image embeddings with K-Means (K selected via elbow by default,
    or Vendi score when ``fd_use_vendi=True``) and assigns scores by
    round-robin selection order across clusters.  This guarantees visual
    diversity across semantic clusters; within-cluster ordering is random.

    When ``config.device`` resolves to ``"cuda"``, uses the framework's
    GPU-accelerated ``MiniBatchKMeans`` instead of sklearn; otherwise falls
    back to sklearn ``KMeans``.

    Args:
        df: DataFrame containing the embedding column.
        config: CoreSetConfig; ``embedding_column``, ``fd_*``, and ``device``.

    Returns:
        (N,) float array in [0, 1]; higher = selected earlier in round-robin.

    Raises:
        ValueError: if ``embedding_column`` is None or absent from df.
    """
    if config.embedding_column is None:
        raise ValueError(
            "FD method requires embedding_column to be set in CoreSetConfig."
        )
    if config.embedding_column not in df.columns:
        raise ValueError(
            f"Embedding column '{config.embedding_column}' not found in DataFrame."
        )

    F = _get_embeddings(df, config.embedding_column)
    if config.fd_use_spatial:
        F = _augment_embedding_with_spatial(F, df)
    N = len(F)
    device = _get_device(config)
    use_gpu = device.type == "cuda"

    k_min = config.fd_k_min
    k_max = min(config.fd_k_max, N - 1)
    k_max = max(k_max, k_min)

    if config.fd_use_vendi:
        from pytorch_segmentation_models_trainer.tools.coreset.vendi_score import (
            select_k_vendi,
        )

        K = select_k_vendi(F, k_min=k_min, delta=config.fd_vendi_delta)
    elif use_gpu:
        from pytorch_segmentation_models_trainer.tools.kmeans.kmeans_calculator import (
            MiniBatchKMeans as TorchKMeans,
        )

        F_tensor = torch.tensor(F, dtype=torch.float32)
        inertias = []
        for ki in tqdm(range(k_min, k_max + 1), desc="FD elbow [GPU]", unit="k"):
            km = TorchKMeans(n_clusters=ki, random_state=42, device=device)
            km.fit(F_tensor)
            batch_size = 2048
            inertia = 0.0
            for i in range(0, N, batch_size):
                batch = F_tensor[i : i + batch_size].to(device)
                dist = torch.cdist(batch, km.centroids, p=2)
                min_dist, _ = torch.min(dist, dim=1)
                inertia += min_dist.pow(2).sum().item()
            inertias.append(inertia)
        K = select_k_elbow(inertias, k_min=k_min)
    else:
        inertias = []
        for ki in tqdm(range(k_min, k_max + 1), desc="FD elbow [CPU]", unit="k"):
            km = KMeans(n_clusters=ki, random_state=42, n_init="auto")
            km.fit(F)
            inertias.append(km.inertia_)
        K = select_k_elbow(inertias, k_min=k_min)

    K = max(K, 1)
    if use_gpu:
        from pytorch_segmentation_models_trainer.tools.kmeans.kmeans_calculator import (
            MiniBatchKMeans as TorchKMeans,
        )

        F_tensor = torch.tensor(F, dtype=torch.float32)
        km_final = TorchKMeans(n_clusters=K, random_state=42, device=device)
        km_final.fit(F_tensor)
        cluster_ids = km_final.predict(F_tensor).cpu().numpy()
    else:
        km_final = KMeans(n_clusters=K, random_state=42, n_init="auto")
        km_final.fit(F)
        cluster_ids = km_final.labels_

    rank = _round_robin_rank(cluster_ids, K, N)
    return rank / max(N - 1, 1)


def score_lc_fd(df: pd.DataFrame, config: CoreSetConfig) -> np.ndarray:
    """LC/FD hybrid scorer — top-m FD patches, then remaining by LC.

    Selects the top-m patches by FD (visual diversity) and fills the
    remaining positions by LC (label complexity), matching the paper's
    two-stage selection strategy.  ``m`` defaults to 10% of N when
    ``lc_fd_cutoff_m`` is None.

    Args:
        df: DataFrame with ``class_entropy`` and embedding columns.
        config: CoreSetConfig with ``lc_fd_cutoff_m`` and ``embedding_column``.

    Returns:
        (N,) float array in [0, 1].
    """
    N = len(df)
    m = config.lc_fd_cutoff_m if config.lc_fd_cutoff_m is not None else int(0.1 * N)
    m = max(1, min(m, N))

    fd_scores = score_fd(df, config)
    lc_scores = score_lc(df, config)

    top_m_fd_idx = np.argsort(-fd_scores)[:m]
    top_m_set = set(top_m_fd_idx.tolist())

    remaining_idx = np.array([i for i in range(N) if i not in top_m_set])
    if len(remaining_idx) > 0:
        lc_order = np.argsort(-lc_scores[remaining_idx])
        remaining_lc_idx = remaining_idx[lc_order]
    else:
        remaining_lc_idx = np.array([], dtype=int)

    final_order = np.concatenate([top_m_fd_idx, remaining_lc_idx])

    score = np.zeros(N, dtype=np.float64)
    denom = max(N - 1, 1)
    for position, idx in enumerate(final_order):
        score[int(idx)] = (N - 1 - position) / denom

    return score


def score_fa_cb(df: pd.DataFrame, config: CoreSetConfig) -> np.ndarray:
    """FA/CB hybrid scorer — λ × FA + (1-λ) × CB.

    Args:
        df: DataFrame with embedding and ``class_dist_json`` columns.
        config: CoreSetConfig; ``fa_cb_lambda`` and ``embedding_column``.

    Returns:
        (N,) float array in [0, 1].
    """
    lam = config.fa_cb_lambda
    return lam * score_fa(df, config) + (1.0 - lam) * score_cb(df, config)


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

_SCORER_MAP = {
    "lc": score_lc,
    "cb": score_cb,
    "fa": score_fa,
    "fd": score_fd,
    "lc_fd": score_lc_fd,
    "fa_cb": score_fa_cb,
}


def get_scorer(method: str):
    """Return the scorer function for the given method name.

    Args:
        method: one of ``'lc'``, ``'cb'``, ``'fa'``, ``'fd'``,
            ``'lc_fd'``, ``'fa_cb'``.

    Returns:
        Callable with signature ``(df, config) -> np.ndarray``.

    Raises:
        ValueError: if method is not recognised.
    """
    if method not in _SCORER_MAP:
        raise ValueError(
            f"Unknown coreset method: {method!r}. "
            f"Valid options: {sorted(_SCORER_MAP.keys())}."
        )
    return _SCORER_MAP[method]
