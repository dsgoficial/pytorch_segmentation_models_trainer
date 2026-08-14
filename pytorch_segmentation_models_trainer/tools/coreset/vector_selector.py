# -*- coding: utf-8 -*-
"""Spatial and embedding-based coreset selection primitives.

Functions here are stateless and composable. They are used by
:class:`HybridVectorCoresetSelector` to build multi-step coreset selection
pipelines that combine spatial polygon intersection with embedding diversity
(FD / LC-FD) and entropy sweep.
"""

from typing import Optional, Set

import numpy as np
import pandas as pd


def compute_intersection_areas(
    pool_gdf,
    vec_gdf,
    exclude_idx: Optional[Set] = None,
) -> pd.Series:
    """Compute total intersection area between pool patches and vector polygons.

    Args:
        pool_gdf: GeoDataFrame of pool patches (polygon geometries).
        vec_gdf: GeoDataFrame of reference vector polygons.
        exclude_idx: Optional set of pool index values to exclude before joining.

    Returns:
        Series indexed by pool_gdf index with total intersection area per patch.
        Only patches with non-zero intersection are included.
    """
    import geopandas as gpd

    pool = (
        pool_gdf
        if exclude_idx is None
        else pool_gdf.drop(index=exclude_idx, errors="ignore")
    )
    if pool.empty or vec_gdf.empty:
        return pd.Series(dtype=float)

    joined = gpd.sjoin(pool.reset_index(), vec_gdf, how="inner", predicate="intersects")
    if joined.empty:
        return pd.Series(dtype=float)

    pool_reset = pool.reset_index()
    left_idx_col = "index" if "index" in pool_reset.columns else pool_reset.columns[0]

    areas = []
    for _, row in joined.iterrows():
        orig_idx = row[left_idx_col]
        patch_geom = pool_gdf.loc[orig_idx, "geometry"]
        right_geom_idx = row["index_right"]
        vec_geom = vec_gdf.loc[right_geom_idx, "geometry"]
        intersection = patch_geom.intersection(vec_geom)
        areas.append({"orig_idx": orig_idx, "area": intersection.area})

    area_df = pd.DataFrame(areas)
    return area_df.groupby("orig_idx")["area"].sum()


def select_by_vector_intersection(
    pool_gdf,
    vec_gdf,
    min_area_m2: float,
    exclude_idx: Optional[Set] = None,
) -> pd.Index:
    """Select pool patches whose total intersection with vec_gdf >= min_area_m2.

    Args:
        pool_gdf: GeoDataFrame of pool patches.
        vec_gdf: GeoDataFrame of reference vector polygons.
        min_area_m2: Minimum intersection area threshold.
        exclude_idx: Pool indices to exclude from consideration.

    Returns:
        Index of selected pool patches.
    """
    areas = compute_intersection_areas(pool_gdf, vec_gdf, exclude_idx=exclude_idx)
    return areas[areas >= min_area_m2].index


def fd_embedding_select(
    sub_df: pd.DataFrame,
    k: int,
    budget: int,
    exclude_idx: Optional[Set] = None,
    embedding_col: str = "embedding",
    random_state: int = 42,
) -> pd.Index:
    """Facility-Location / K-Means round-robin diversity selection on embeddings.

    Clusters the pool into ``k`` groups (KMeans) then round-robins through
    clusters picking the nearest-to-centroid sample until ``budget`` is reached.

    Args:
        sub_df: DataFrame with an embedding column (list/array per row).
        k: Number of KMeans clusters.
        budget: Maximum number of items to select.
        exclude_idx: Pool indices to skip.
        embedding_col: Column name for embedding vectors.
        random_state: KMeans random state for reproducibility.

    Returns:
        Index of selected items (length <= budget).
    """
    from sklearn.cluster import KMeans

    pool = (
        sub_df
        if exclude_idx is None
        else sub_df.drop(index=exclude_idx, errors="ignore")
    )
    if pool.empty:
        return pd.Index([])

    X = np.stack(pool[embedding_col].values)
    n = len(pool)
    k_eff = min(k, n)
    budget_eff = min(budget, n)

    km = KMeans(n_clusters=k_eff, random_state=random_state, n_init="auto")
    labels = km.fit_predict(X)
    centroids = km.cluster_centers_

    # Group pool indices by cluster
    clusters: dict = {i: [] for i in range(k_eff)}
    for pos, lbl in enumerate(labels):
        clusters[lbl].append(pos)

    # Sort each cluster by distance to centroid ascending
    for lbl, positions in clusters.items():
        dists = np.linalg.norm(X[positions] - centroids[lbl], axis=1)
        order = np.argsort(dists)
        clusters[lbl] = [positions[o] for o in order]

    # Round-robin pick — interleave one item per cluster until budget reached.
    # Total cluster items == n == budget_eff, so StopIteration never fires here.
    selected_positions = []
    cluster_queues = [list(positions) for positions in clusters.values()]
    round_idx = 0
    while len(selected_positions) < budget_eff:
        cluster_idx = round_idx % k_eff
        queue = cluster_queues[cluster_idx]
        if queue:
            selected_positions.append(queue.pop(0))
        round_idx += 1

    pool_indices = pool.index.tolist()
    return pd.Index([pool_indices[p] for p in selected_positions])


def lc_fd_select(
    sub_df: pd.DataFrame,
    k: int,
    budget: int,
    exclude_idx: Optional[Set] = None,
    lc_fraction: float = 0.40,
    embedding_col: str = "embedding",
    entropy_col: str = "class_entropy",
    random_state: int = 42,
) -> pd.Index:
    """LC/FD: high-entropy (LC) selection followed by FD on the remainder.

    Args:
        sub_df: DataFrame with embedding and entropy columns.
        k: KMeans cluster count for FD phase.
        budget: Total items to select.
        exclude_idx: Pool indices to skip.
        lc_fraction: Fraction of budget allocated to top-entropy (LC) phase.
        embedding_col: Column name for embeddings.
        entropy_col: Column name for entropy values.
        random_state: KMeans random state.

    Returns:
        Index of selected items (length <= budget).
    """
    pool = (
        sub_df
        if exclude_idx is None
        else sub_df.drop(index=exclude_idx, errors="ignore")
    )
    if pool.empty:
        return pd.Index([])

    n = len(pool)
    budget_eff = min(budget, n)
    lc_budget = int(budget_eff * lc_fraction)

    # LC phase: top-entropy
    lc_idx = set(pool[entropy_col].nlargest(lc_budget).index.tolist())

    # FD phase on remainder
    fd_budget = budget_eff - len(lc_idx)
    fd_exclude = (exclude_idx or set()) | lc_idx
    fd_idx = fd_embedding_select(
        sub_df,
        k=k,
        budget=fd_budget,
        exclude_idx=fd_exclude,
        embedding_col=embedding_col,
        random_state=random_state,
    )

    return pd.Index(list(lc_idx) + list(fd_idx))


def entropy_sweep_select(
    df: pd.DataFrame,
    budget: int,
    exclude_idx: Optional[Set] = None,
    entropy_col: str = "class_entropy",
) -> pd.Index:
    """Select the top-N unselected patches ranked by entropy descending.

    Args:
        df: Full pool DataFrame with an entropy column.
        budget: Maximum number of patches to select.
        exclude_idx: Pool indices already selected (excluded from sweep).
        entropy_col: Column name for entropy values.

    Returns:
        Index of selected patches (length <= budget).
    """
    pool = df if exclude_idx is None else df.drop(index=exclude_idx, errors="ignore")
    if pool.empty:
        return pd.Index([])
    top = pool[entropy_col].nlargest(min(budget, len(pool)))
    return top.index
