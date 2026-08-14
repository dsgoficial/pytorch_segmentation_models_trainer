# -*- coding: utf-8 -*-
"""Tests for pytorch_segmentation_models_trainer.tools.coreset.vector_selector."""

import numpy as np
import pandas as pd
import pytest

try:
    import geopandas as gpd
    from shapely.geometry import box

    HAS_GEO = True
except ImportError:
    HAS_GEO = False

pytestmark = pytest.mark.skipif(not HAS_GEO, reason="geopandas not installed")

from pytorch_segmentation_models_trainer.tools.coreset.vector_selector import (  # noqa: E402
    compute_intersection_areas,
    entropy_sweep_select,
    fd_embedding_select,
    lc_fd_select,
    select_by_vector_intersection,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

CRS = "EPSG:4326"


def _make_pool(bboxes):
    """GeoDataFrame from list of (minx, miny, maxx, maxy) tuples."""
    geoms = [box(*b) for b in bboxes]
    return gpd.GeoDataFrame({"geometry": geoms}, crs=CRS)


def _make_vec(bboxes):
    geoms = [box(*b) for b in bboxes]
    return gpd.GeoDataFrame({"geometry": geoms}, crs=CRS)


def _make_emb_df(n=10, dim=4, seed=0):
    rng = np.random.default_rng(seed)
    embeddings = list(rng.random((n, dim)).astype(np.float32))
    entropy = rng.random(n)
    return pd.DataFrame({"embedding": embeddings, "class_entropy": entropy})


# ---------------------------------------------------------------------------
# compute_intersection_areas
# ---------------------------------------------------------------------------


class TestComputeIntersectionAreas:
    def test_full_containment(self):
        pool = _make_pool([(0, 0, 1, 1)])
        vec = _make_vec([(0, 0, 2, 2)])  # polygon contains patch
        areas = compute_intersection_areas(pool, vec)
        assert len(areas) == 1
        assert abs(areas.iloc[0] - 1.0) < 1e-9

    def test_partial_overlap(self):
        pool = _make_pool([(0, 0, 2, 2)])
        vec = _make_vec([(1, 0, 3, 2)])  # half overlap
        areas = compute_intersection_areas(pool, vec)
        assert len(areas) == 1
        assert abs(areas.iloc[0] - 2.0) < 1e-9

    def test_no_intersection_returns_empty(self):
        pool = _make_pool([(0, 0, 1, 1)])
        vec = _make_vec([(5, 5, 6, 6)])
        areas = compute_intersection_areas(pool, vec)
        assert len(areas) == 0

    def test_multiple_polygons_summed_per_patch(self):
        pool = _make_pool([(0, 0, 4, 4)])
        # Two separate polygons each covering 1×1
        vec = _make_vec([(0, 0, 1, 1), (2, 2, 3, 3)])
        areas = compute_intersection_areas(pool, vec)
        assert len(areas) == 1
        assert abs(areas.iloc[0] - 2.0) < 1e-9

    def test_exclude_idx_removes_patches(self):
        pool = _make_pool([(0, 0, 1, 1), (2, 2, 3, 3)])
        vec = _make_vec([(0, 0, 4, 4)])
        areas = compute_intersection_areas(pool, vec, exclude_idx={0})
        assert 0 not in areas.index

    def test_empty_vec_returns_empty(self):
        pool = _make_pool([(0, 0, 1, 1)])
        vec = gpd.GeoDataFrame({"geometry": []}, crs=CRS)
        areas = compute_intersection_areas(pool, vec)
        assert len(areas) == 0


# ---------------------------------------------------------------------------
# select_by_vector_intersection
# ---------------------------------------------------------------------------


class TestSelectByVectorIntersection:
    def test_returns_patches_above_threshold(self):
        pool = _make_pool([(0, 0, 2, 2), (10, 10, 11, 11)])
        vec = _make_vec([(0, 0, 2, 2)])  # covers patch 0 fully (area=4), not patch 1
        selected = select_by_vector_intersection(pool, vec, min_area_m2=3.0)
        assert 0 in selected
        assert 1 not in selected

    def test_sliver_excluded(self):
        pool = _make_pool([(0, 0, 1, 1)])
        vec = _make_vec([(0, 0, 0.01, 1)])  # tiny overlap
        selected = select_by_vector_intersection(pool, vec, min_area_m2=0.5)
        assert len(selected) == 0

    def test_exclude_idx_respected(self):
        pool = _make_pool([(0, 0, 2, 2), (3, 0, 5, 2)])
        vec = _make_vec([(0, 0, 6, 2)])  # covers both
        selected = select_by_vector_intersection(
            pool, vec, min_area_m2=1.0, exclude_idx={0}
        )
        assert 0 not in selected
        assert 1 in selected


# ---------------------------------------------------------------------------
# fd_embedding_select
# ---------------------------------------------------------------------------


class TestFdEmbeddingSelect:
    def test_returns_exact_budget(self):
        df = _make_emb_df(n=20)
        idx = fd_embedding_select(df, k=5, budget=8)
        assert len(idx) == 8

    def test_respects_exclude_idx(self):
        df = _make_emb_df(n=20)
        exclude = {0, 1, 2}
        idx = fd_embedding_select(df, k=5, budget=8, exclude_idx=exclude)
        assert len(set(idx) & exclude) == 0

    def test_k_capped_to_pool_size(self):
        df = _make_emb_df(n=3)
        idx = fd_embedding_select(df, k=100, budget=3)
        assert len(idx) == 3

    def test_budget_larger_than_pool_returns_all(self):
        df = _make_emb_df(n=5)
        idx = fd_embedding_select(df, k=3, budget=100)
        assert len(idx) == 5

    def test_reproducible_with_same_seed(self):
        df = _make_emb_df(n=30)
        idx1 = fd_embedding_select(df, k=5, budget=10, random_state=42)
        idx2 = fd_embedding_select(df, k=5, budget=10, random_state=42)
        assert list(idx1) == list(idx2)

    def test_empty_pool_returns_empty(self):
        df = _make_emb_df(n=5)
        idx = fd_embedding_select(df, k=3, budget=5, exclude_idx={0, 1, 2, 3, 4})
        assert len(idx) == 0


# ---------------------------------------------------------------------------
# lc_fd_select
# ---------------------------------------------------------------------------


class TestLcFdSelect:
    def test_returns_at_most_budget(self):
        df = _make_emb_df(n=20)
        idx = lc_fd_select(df, k=5, budget=8)
        assert len(idx) <= 8

    def test_high_entropy_items_included(self):
        df = _make_emb_df(n=20)
        top_entropy = df["class_entropy"].nlargest(3).index.tolist()
        idx = lc_fd_select(df, k=5, budget=15, lc_fraction=0.5)
        for i in top_entropy:
            assert i in idx

    def test_respects_exclude_idx(self):
        df = _make_emb_df(n=20)
        exclude = {0, 1, 2}
        idx = lc_fd_select(df, k=5, budget=10, exclude_idx=exclude)
        assert len(set(idx) & exclude) == 0

    def test_empty_pool_returns_empty(self):
        df = _make_emb_df(n=5)
        idx = lc_fd_select(df, k=3, budget=5, exclude_idx={0, 1, 2, 3, 4})
        assert len(idx) == 0


# ---------------------------------------------------------------------------
# entropy_sweep_select
# ---------------------------------------------------------------------------


class TestEntropySweepSelect:
    def test_returns_top_n_by_entropy(self):
        df = _make_emb_df(n=10)
        idx = entropy_sweep_select(df, budget=3)
        top3 = set(df["class_entropy"].nlargest(3).index)
        assert set(idx) == top3

    def test_budget_larger_than_pool(self):
        df = _make_emb_df(n=5)
        idx = entropy_sweep_select(df, budget=100)
        assert len(idx) == 5

    def test_exclude_idx_excluded(self):
        df = _make_emb_df(n=10)
        exclude = {0, 1, 2}
        idx = entropy_sweep_select(df, budget=5, exclude_idx=exclude)
        assert len(set(idx) & exclude) == 0

    def test_empty_result_when_all_excluded(self):
        df = _make_emb_df(n=3)
        idx = entropy_sweep_select(df, budget=3, exclude_idx={0, 1, 2})
        assert len(idx) == 0
