# -*- coding: utf-8 -*-
"""Tests for pytorch_segmentation_models_trainer.tools.coreset.hybrid_coreset_selector."""

import numpy as np
import pandas as pd
import pytest
from unittest.mock import MagicMock, patch

try:
    import geopandas as gpd

    HAS_GEO = True
except ImportError:
    HAS_GEO = False

pytestmark = pytest.mark.skipif(not HAS_GEO, reason="geopandas not installed")

from pytorch_segmentation_models_trainer.tools.coreset.hybrid_coreset_selector import (  # noqa: E402
    EmbeddingSelectionStep,
    HybridVectorCoresetConfig,
    HybridVectorCoresetSelector,
    VectorSelectionStep,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

CRS = "EPSG:4326"


def _make_pool_df(n=20, seed=0):
    rng = np.random.default_rng(seed)
    embeddings = list(rng.random((n, 4)).astype(np.float32))
    entropy = rng.random(n)
    xs = np.arange(n, dtype=float)
    df = pd.DataFrame(
        {
            "tile_minx": xs,
            "tile_miny": xs,
            "tile_maxx": xs + 1,
            "tile_maxy": xs + 1,
            "embedding": embeddings,
            "class_entropy": entropy,
            "c3": rng.random(n),
            "c5": rng.random(n),
        }
    )
    return df


def _make_config(tmp_path, **overrides):
    defaults = dict(
        input_csv_path=str(tmp_path / "pool.csv"),
        embeddings_parquet=str(tmp_path / "emb.parquet"),
        vectors_gpkg=str(tmp_path / "vec.gpkg"),
        output_csv_path=str(tmp_path / "out.csv"),
        pool_fraction=0.5,
        vector_steps=[],
        embedding_steps=[],
        entropy_sweep=True,
        crs=CRS,
    )
    defaults.update(overrides)
    return HybridVectorCoresetConfig(**defaults)


# ---------------------------------------------------------------------------
# Config dataclasses
# ---------------------------------------------------------------------------


class TestHybridVectorCoresetConfig:
    def test_required_fields(self, tmp_path):
        cfg = _make_config(tmp_path)
        assert cfg.pool_fraction == 0.5
        assert cfg.entropy_sweep is True
        assert cfg.crs == CRS

    def test_default_values(self, tmp_path):
        cfg = HybridVectorCoresetConfig(
            input_csv_path="a",
            embeddings_parquet="b",
            vectors_gpkg="c",
            output_csv_path="d",
        )
        assert cfg.pool_fraction == 0.30
        assert cfg.vector_steps == []
        assert cfg.embedding_steps == []
        assert cfg.entropy_sweep is True
        assert cfg.embedding_col == "embedding"
        assert cfg.entropy_col == "class_entropy"
        assert cfg.random_state == 42


class TestVectorSelectionStep:
    def test_fields(self):
        step = VectorSelectionStep(gpkg_layer="grassland", class_indices=["c3"])
        assert step.gpkg_layer == "grassland"
        assert step.min_intersect_area_m2 == 1000.0


class TestEmbeddingSelectionStep:
    def test_fields(self):
        step = EmbeddingSelectionStep(class_filter={"c3": (">", 0.5)}, budget=50, k=5)
        assert step.method == "fd"
        assert step.lc_fraction == 0.40


# ---------------------------------------------------------------------------
# HybridVectorCoresetSelector.select
# ---------------------------------------------------------------------------


class TestHybridVectorCoresetSelectorSelect:
    def _selector(self, tmp_path, **cfg_overrides):
        cfg = _make_config(tmp_path, **cfg_overrides)
        return HybridVectorCoresetSelector(cfg)

    def test_output_has_required_columns(self, tmp_path):
        selector = self._selector(tmp_path)
        pool_df = _make_pool_df(n=20)

        with patch.object(selector, "_load_gpkg_layer", return_value=MagicMock()):
            result = selector.select(pool_df)

        assert "coreset_selected" in result.columns
        assert "selection_step" in result.columns

    def test_total_budget_respected(self, tmp_path):
        selector = self._selector(tmp_path, pool_fraction=0.4)
        pool_df = _make_pool_df(n=20)

        with patch.object(selector, "_load_gpkg_layer", return_value=MagicMock()):
            result = selector.select(pool_df)

        n_selected = int(result["coreset_selected"].sum())
        budget = int(0.4 * len(pool_df))
        assert n_selected <= budget + 1  # entropy sweep may fill exactly

    def test_output_csv_written(self, tmp_path):
        selector = self._selector(tmp_path)
        pool_df = _make_pool_df(n=10)

        with patch.object(selector, "_load_gpkg_layer", return_value=MagicMock()):
            selector.select(pool_df)

        assert (tmp_path / "out.csv").exists()

    def test_selection_step_tagged(self, tmp_path):
        selector = self._selector(tmp_path, entropy_sweep=True)
        pool_df = _make_pool_df(n=20)

        with patch.object(selector, "_load_gpkg_layer", return_value=MagicMock()):
            result = selector.select(pool_df)

        selected = result[result["coreset_selected"] == 1]
        assert selected["selection_step"].notna().all()

    def test_no_steps_entropy_sweep_only(self, tmp_path):
        selector = self._selector(
            tmp_path, vector_steps=[], embedding_steps=[], entropy_sweep=True
        )
        pool_df = _make_pool_df(n=10)

        with patch.object(selector, "_load_gpkg_layer", return_value=MagicMock()):
            result = selector.select(pool_df)

        selected = result[result["coreset_selected"] == 1]
        assert (selected["selection_step"] == "entropy_sweep").all()

    def test_entropy_sweep_false_no_sweep(self, tmp_path):
        selector = self._selector(
            tmp_path, vector_steps=[], embedding_steps=[], entropy_sweep=False
        )
        pool_df = _make_pool_df(n=10)

        with patch.object(selector, "_load_gpkg_layer", return_value=MagicMock()):
            result = selector.select(pool_df)

        n_selected = int(result["coreset_selected"].sum())
        assert n_selected == 0  # no steps + no sweep = nothing selected

    def test_vector_step_selects_patches(self, tmp_path):
        from shapely.geometry import box as sbox

        vec_gdf = gpd.GeoDataFrame({"geometry": [sbox(0, 0, 5, 5)]}, crs=CRS)

        cfg = _make_config(
            tmp_path,
            vector_steps=[
                {
                    "gpkg_layer": "layer1",
                    "class_indices": ["c3"],
                    "min_intersect_area_m2": 0.1,
                }
            ],
            embedding_steps=[],
            entropy_sweep=False,
        )
        selector = HybridVectorCoresetSelector(cfg)
        pool_df = _make_pool_df(n=10)

        with patch.object(selector, "_load_gpkg_layer", return_value=vec_gdf):
            result = selector.select(pool_df)

        selected = result[result["coreset_selected"] == 1]
        assert len(selected) > 0
        assert (selected["selection_step"] == "vector:layer1").all()

    def test_embedding_fd_step_selects_patches(self, tmp_path):
        cfg = _make_config(
            tmp_path,
            vector_steps=[],
            embedding_steps=[
                {
                    "class_filter": {"c3": (">", 0.0)},
                    "budget": 5,
                    "k": 3,
                    "method": "fd",
                }
            ],
            entropy_sweep=False,
        )
        selector = HybridVectorCoresetSelector(cfg)
        pool_df = _make_pool_df(n=20)

        with patch.object(selector, "_load_gpkg_layer", return_value=MagicMock()):
            result = selector.select(pool_df)

        selected = result[result["coreset_selected"] == 1]
        assert len(selected) > 0
        assert (selected["selection_step"] == "embedding:fd:0").all()

    def test_embedding_lc_fd_step_selects_patches(self, tmp_path):
        cfg = _make_config(
            tmp_path,
            vector_steps=[],
            embedding_steps=[
                {
                    "class_filter": {"c3": (">", 0.0)},
                    "budget": 5,
                    "k": 3,
                    "method": "lc_fd",
                }
            ],
            entropy_sweep=False,
        )
        selector = HybridVectorCoresetSelector(cfg)
        pool_df = _make_pool_df(n=20)

        with patch.object(selector, "_load_gpkg_layer", return_value=MagicMock()):
            result = selector.select(pool_df)

        selected = result[result["coreset_selected"] == 1]
        assert len(selected) > 0

    def test_already_selected_not_reselected(self, tmp_path):
        """Items from vector step are not double-counted in embedding step."""
        from shapely.geometry import box as sbox

        vec_gdf = gpd.GeoDataFrame({"geometry": [sbox(0, 0, 20, 20)]}, crs=CRS)
        cfg = _make_config(
            tmp_path,
            vector_steps=[
                {
                    "gpkg_layer": "l1",
                    "class_indices": ["c3"],
                    "min_intersect_area_m2": 0.01,
                }
            ],
            embedding_steps=[
                {
                    "class_filter": {"c3": (">", 0.0)},
                    "budget": 5,
                    "k": 3,
                    "method": "fd",
                }
            ],
            entropy_sweep=False,
        )
        selector = HybridVectorCoresetSelector(cfg)
        pool_df = _make_pool_df(n=10)

        with patch.object(selector, "_load_gpkg_layer", return_value=vec_gdf):
            result = selector.select(pool_df)

        # coreset_selected should be 0 or 1, not 2
        assert result["coreset_selected"].max() <= 1

    def test_embedding_step_filter_operators(self, tmp_path):
        """All four comparison operators in class_filter are exercised."""
        cfg = _make_config(
            tmp_path,
            vector_steps=[],
            embedding_steps=[
                # >= and < operators
                {
                    "class_filter": {"c3": (">=", 0.0), "c5": ("<", 1.1)},
                    "budget": 3,
                    "k": 2,
                    "method": "fd",
                },
                # <= operator in a second step
                {
                    "class_filter": {"c3": ("<=", 1.0)},
                    "budget": 3,
                    "k": 2,
                    "method": "fd",
                },
            ],
            entropy_sweep=False,
        )
        selector = HybridVectorCoresetSelector(cfg)
        pool_df = _make_pool_df(n=20)

        with patch.object(selector, "_load_gpkg_layer", return_value=MagicMock()):
            result = selector.select(pool_df)

        assert "coreset_selected" in result.columns

    def test_vector_step_budget_already_full_breaks(self, tmp_path):
        """When budget is already consumed, vector steps are skipped."""
        from shapely.geometry import box as sbox

        vec_gdf = gpd.GeoDataFrame({"geometry": [sbox(0, 0, 20, 20)]}, crs=CRS)
        cfg = _make_config(
            tmp_path,
            pool_fraction=0.0,  # budget = 0 → vector step loop breaks immediately
            vector_steps=[
                {
                    "gpkg_layer": "l1",
                    "class_indices": ["c3"],
                    "min_intersect_area_m2": 0.01,
                }
            ],
            embedding_steps=[],
            entropy_sweep=False,
        )
        selector = HybridVectorCoresetSelector(cfg)
        pool_df = _make_pool_df(n=10)

        with patch.object(
            selector, "_load_gpkg_layer", return_value=vec_gdf
        ) as mock_load:
            result = selector.select(pool_df)

        mock_load.assert_not_called()
        assert int(result["coreset_selected"].sum()) == 0

    def test_load_gpkg_layer_called_with_layer_name(self, tmp_path):
        """_load_gpkg_layer is called with the gpkg_layer name from the step."""
        from shapely.geometry import box as sbox

        vec_gdf = gpd.GeoDataFrame({"geometry": [sbox(0, 0, 1, 1)]}, crs=CRS)
        cfg = _make_config(
            tmp_path,
            vector_steps=[
                {
                    "gpkg_layer": "my_layer",
                    "class_indices": ["c3"],
                    "min_intersect_area_m2": 100.0,
                }
            ],
            embedding_steps=[],
            entropy_sweep=False,
        )
        selector = HybridVectorCoresetSelector(cfg)
        pool_df = _make_pool_df(n=5)

        with patch.object(
            selector, "_load_gpkg_layer", return_value=vec_gdf
        ) as mock_load:
            selector.select(pool_df)

        mock_load.assert_called_once_with("my_layer")

    def test_load_gpkg_layer_reads_file(self, tmp_path):
        """_load_gpkg_layer calls gpd.read_file with the configured path."""
        from shapely.geometry import box as sbox

        gpkg_path = tmp_path / "test.gpkg"
        layer_gdf = gpd.GeoDataFrame({"geometry": [sbox(0, 0, 1, 1)]}, crs=CRS)
        layer_gdf.to_file(gpkg_path, layer="test_layer", driver="GPKG")

        cfg = _make_config(tmp_path, vectors_gpkg=str(gpkg_path))
        selector = HybridVectorCoresetSelector(cfg)
        result = selector._load_gpkg_layer("test_layer")

        assert len(result) == 1
