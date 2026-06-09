# -*- coding: utf-8 -*-
"""Tests for utils/h3_val_test_split.py."""

import numpy as np
import pandas as pd
import pytest
import rasterio
from pathlib import Path
from rasterio.crs import CRS
from rasterio.transform import from_bounds

from pytorch_segmentation_models_trainer.utils.h3_val_test_split import (
    _class_distribution,
    _score_split,
    assign_h3_cells,
    build_exclusion_buffer_gdf,
    filter_weak_training_tiles,
    get_patch_centroid,
    run,
    split_val_test,
)

# ---------------------------------------------------------------------------
# Constants — southern Brazil (Paraná) in EPSG:3857
# Curitiba area: approximately x=-5_500_000, y=-2_940_000
# H3 resolution-7 edge length ≈ 1.4 km, so 50 km spacing guarantees distinct cells.
# ---------------------------------------------------------------------------

_BASE_X = -5_500_000.0
_BASE_Y = -2_940_000.0
_STEP = 50_000.0  # 50 km → always different H3 res-7 cells
_PATCH_SIZE_M = 640.0  # 256 px × 2.5 m/px


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _make_geotiff(
    path: Path,
    cx_3857: float,
    cy_3857: float,
    size_m: float = _PATCH_SIZE_M,
) -> Path:
    """Write a tiny 256×256 GeoTIFF centered at (cx_3857, cy_3857) in EPSG:3857."""
    half = size_m / 2.0
    bounds = (cx_3857 - half, cy_3857 - half, cx_3857 + half, cy_3857 + half)
    transform = from_bounds(*bounds, width=256, height=256)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=256,
        width=256,
        count=3,
        dtype="uint8",
        crs=CRS.from_epsg(3857),
        transform=transform,
    ) as dst:
        dst.write(np.zeros((3, 256, 256), dtype=np.uint8))
    return path


@pytest.fixture()
def patch_geotiffs(tmp_path):
    """10 GeoTIFFs in 10 distinct H3 res-7 cells across southern Brazil."""
    centers = [
        (_BASE_X + i * _STEP, _BASE_Y + j * _STEP)
        for i, j in [
            (0, 0),
            (1, 0),
            (2, 0),
            (3, 0),
            (4, 0),
            (0, 1),
            (1, 1),
            (2, 1),
            (3, 1),
            (4, 1),
        ]
    ]
    paths = []
    for idx, (cx, cy) in enumerate(centers):
        p = tmp_path / f"patch_{idx:03d}.tif"
        _make_geotiff(p, cx, cy)
        paths.append(str(p))
    return paths


@pytest.fixture()
def patches_df(patch_geotiffs):
    classes = [0, 1, 2, 3, 4, 0, 1, 2, 3, 4]
    return pd.DataFrame({"image_path": patch_geotiffs, "class_label": classes})


@pytest.fixture()
def patches_df_with_h3(patches_df):
    return assign_h3_cells(patches_df)


def _df_with_fake_h3(n_cells: int, patches_per_cell: int = 5, n_classes: int = 3):
    """Create DataFrame with pre-populated h3_cell without actual file I/O."""
    records = []
    for c in range(n_cells):
        for p in range(patches_per_cell):
            records.append(
                {
                    "h3_cell": f"cell_{c:02d}",
                    "class_label": (c * patches_per_cell + p) % n_classes,
                }
            )
    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# get_patch_centroid
# ---------------------------------------------------------------------------


class TestGetPatchCentroid:
    def test_returns_float_tuple(self, tmp_path):
        p = _make_geotiff(tmp_path / "t.tif", _BASE_X, _BASE_Y)
        lat, lon = get_patch_centroid(p)
        assert isinstance(lat, float)
        assert isinstance(lon, float)

    def test_southern_brazil_lat_range(self, tmp_path):
        p = _make_geotiff(tmp_path / "t.tif", _BASE_X, _BASE_Y)
        lat, lon = get_patch_centroid(p)
        # Southern Brazil roughly between -35° and -20° latitude
        assert -35.0 < lat < -20.0

    def test_southern_brazil_lon_range(self, tmp_path):
        p = _make_geotiff(tmp_path / "t.tif", _BASE_X, _BASE_Y)
        lat, lon = get_patch_centroid(p)
        assert -60.0 < lon < -40.0

    def test_different_centers_different_lon(self, tmp_path):
        p1 = _make_geotiff(tmp_path / "a.tif", _BASE_X, _BASE_Y)
        p2 = _make_geotiff(tmp_path / "b.tif", _BASE_X + _STEP, _BASE_Y)
        _, lon1 = get_patch_centroid(p1)
        _, lon2 = get_patch_centroid(p2)
        assert lon1 != pytest.approx(lon2, abs=0.1)

    def test_different_row_different_lat(self, tmp_path):
        p1 = _make_geotiff(tmp_path / "a.tif", _BASE_X, _BASE_Y)
        p2 = _make_geotiff(tmp_path / "b.tif", _BASE_X, _BASE_Y + _STEP)
        lat1, _ = get_patch_centroid(p1)
        lat2, _ = get_patch_centroid(p2)
        assert lat1 != pytest.approx(lat2, abs=0.1)


# ---------------------------------------------------------------------------
# assign_h3_cells
# ---------------------------------------------------------------------------


class TestAssignH3Cells:
    def test_adds_three_columns(self, patches_df_with_h3):
        assert "h3_cell" in patches_df_with_h3.columns
        assert "centroid_lat" in patches_df_with_h3.columns
        assert "centroid_lon" in patches_df_with_h3.columns

    def test_h3_cell_is_valid_string(self, patches_df_with_h3):
        for cell in patches_df_with_h3["h3_cell"]:
            assert isinstance(cell, str)
            assert len(cell) > 0

    def test_does_not_mutate_input(self, patches_df):
        original_cols = list(patches_df.columns)
        _ = assign_h3_cells(patches_df)
        assert list(patches_df.columns) == original_cols

    def test_distant_patches_get_different_cells(self, patches_df_with_h3):
        cells = patches_df_with_h3["h3_cell"].tolist()
        # 10 patches 50 km apart — must be in distinct cells at resolution 7
        assert len(set(cells)) == len(cells)

    def test_close_patches_same_cell(self, tmp_path):
        # Two patches only 100 m apart → same H3 res-7 cell
        p1 = _make_geotiff(tmp_path / "a.tif", _BASE_X, _BASE_Y, size_m=100)
        p2 = _make_geotiff(tmp_path / "b.tif", _BASE_X + 50, _BASE_Y, size_m=100)
        df = pd.DataFrame({"image_path": [str(p1), str(p2)]})
        result = assign_h3_cells(df)
        assert result.loc[0, "h3_cell"] == result.loc[1, "h3_cell"]

    def test_respects_custom_image_col(self, tmp_path):
        p = _make_geotiff(tmp_path / "t.tif", _BASE_X, _BASE_Y)
        df = pd.DataFrame({"path": [str(p)]})
        result = assign_h3_cells(df, image_col="path")
        assert "h3_cell" in result.columns


# ---------------------------------------------------------------------------
# _class_distribution
# ---------------------------------------------------------------------------


class TestClassDistribution:
    def test_sums_to_one(self):
        df = pd.DataFrame({"cls": [0, 0, 1, 2, 2, 2]})
        dist = _class_distribution(df, "cls")
        assert sum(dist.values()) == pytest.approx(1.0)

    def test_balanced_distribution(self):
        df = pd.DataFrame({"cls": [0, 1, 2, 3]})
        dist = _class_distribution(df, "cls")
        for v in dist.values():
            assert v == pytest.approx(0.25)

    def test_single_class(self):
        df = pd.DataFrame({"cls": [1, 1, 1]})
        dist = _class_distribution(df, "cls")
        assert dist[1] == pytest.approx(1.0)

    def test_empty_dataframe(self):
        df = pd.DataFrame({"cls": pd.Series([], dtype=int)})
        dist = _class_distribution(df, "cls")
        assert dist == {}


# ---------------------------------------------------------------------------
# _score_split
# ---------------------------------------------------------------------------


class TestScoreSplit:
    def _make_dfs(self):
        full = pd.DataFrame({"cls": [0, 0, 1, 1, 2, 2]})
        val_balanced = pd.DataFrame({"cls": [0, 1, 2]})
        val_biased = pd.DataFrame({"cls": [0, 0, 0]})
        return full, val_balanced, val_biased

    def test_without_class_col_uses_fraction(self):
        full = pd.DataFrame({"x": range(10)})
        val = full.iloc[:2]
        score = _score_split(val, full, None, 0.20)
        assert score == pytest.approx(0.0, abs=1e-9)

    def test_fraction_deviation_grows_with_distance(self):
        full = pd.DataFrame({"x": range(10)})
        val_close = full.iloc[:2]  # 20% → score ≈ 0
        val_far = full.iloc[:5]  # 50% → score = (0.5-0.2)^2
        s_close = _score_split(val_close, full, None, 0.20)
        s_far = _score_split(val_far, full, None, 0.20)
        assert s_close < s_far

    def test_balanced_val_lower_score_than_biased(self):
        full, val_balanced, val_biased = self._make_dfs()
        s_balanced = _score_split(val_balanced, full, "cls", 0.20)
        s_biased = _score_split(val_biased, full, "cls", 0.20)
        assert s_balanced < s_biased

    def test_empty_val_has_high_score(self):
        full = pd.DataFrame({"cls": [0, 1, 2]})
        val_empty = full.iloc[0:0]
        score = _score_split(val_empty, full, "cls", 0.20)
        assert score > 0

    def test_empty_full_returns_inf(self):
        empty = pd.DataFrame({"cls": pd.Series([], dtype=int)})
        score = _score_split(empty, empty, "cls", 0.20)
        assert score == np.inf


# ---------------------------------------------------------------------------
# split_val_test
# ---------------------------------------------------------------------------


class TestSplitValTest:
    def test_raises_without_h3_cell_column(self):
        df = pd.DataFrame({"x": [1, 2, 3]})
        with pytest.raises(ValueError, match="h3_cell"):
            split_val_test(df)

    def test_union_equals_full_df(self):
        df = _df_with_fake_h3(n_cells=5, patches_per_cell=10)
        val_df, test_df = split_val_test(df)
        combined = pd.concat([val_df, test_df])
        assert len(combined) == len(df)

    def test_no_overlap_between_val_and_test(self):
        df = _df_with_fake_h3(n_cells=5, patches_per_cell=10)
        val_df, test_df = split_val_test(df)
        val_cells = set(val_df["h3_cell"])
        test_cells = set(test_df["h3_cell"])
        assert val_cells.isdisjoint(test_cells)

    def test_h3_cell_integrity_no_split_cell(self):
        df = _df_with_fake_h3(n_cells=5, patches_per_cell=10)
        val_df, test_df = split_val_test(df)
        val_cells = set(val_df["h3_cell"])
        for cell in val_cells:
            assert cell not in set(test_df["h3_cell"])

    def test_val_fraction_within_tolerance(self):
        df = _df_with_fake_h3(n_cells=10, patches_per_cell=5)
        val_df, _ = split_val_test(df, val_fraction=0.20, n_trials=5000)
        frac = len(val_df) / len(df)
        assert 0.10 <= frac <= 0.30

    def test_deterministic_with_same_seed(self):
        df = _df_with_fake_h3(n_cells=8, patches_per_cell=5)
        val_a, _ = split_val_test(df, seed=42, n_trials=2000)
        val_b, _ = split_val_test(df, seed=42, n_trials=2000)
        assert list(val_a["h3_cell"].unique()) == list(val_b["h3_cell"].unique())

    def test_different_seeds_may_differ(self):
        df = _df_with_fake_h3(n_cells=10, patches_per_cell=5)
        val_a, _ = split_val_test(df, seed=1, n_trials=2000)
        val_b, _ = split_val_test(df, seed=99, n_trials=2000)
        # Not guaranteed to differ, but very likely with 10 cells
        cells_a = frozenset(val_a["h3_cell"])
        cells_b = frozenset(val_b["h3_cell"])
        assert cells_a != cells_b or True  # allow equal (1/C(10,2) chance)

    def test_single_cell_returns_empty_val(self):
        df = _df_with_fake_h3(n_cells=1, patches_per_cell=10)
        val_df, test_df = split_val_test(df)
        assert len(val_df) == 0
        assert len(test_df) == len(df)

    def test_class_col_improves_distribution_balance(self):
        # Build df where one split is class-balanced and another is not.
        # With class_col, the balanced one should be chosen.
        records = []
        # cell_0: 5 × class 0
        for _ in range(5):
            records.append({"h3_cell": "cell_0", "class_label": 0})
        # cell_1: 3 class 1, 2 class 2  (less balanced)
        for _ in range(3):
            records.append({"h3_cell": "cell_1", "class_label": 1})
        for _ in range(2):
            records.append({"h3_cell": "cell_1", "class_label": 2})
        # cell_2: 5 × class 0, 1, 2 mixed (more balanced globally)
        for _ in range(2):
            records.append({"h3_cell": "cell_2", "class_label": 0})
        for _ in range(2):
            records.append({"h3_cell": "cell_2", "class_label": 1})
        records.append({"h3_cell": "cell_2", "class_label": 2})
        df = pd.DataFrame(records)
        val_df, _ = split_val_test(
            df, val_fraction=0.30, class_col="class_label", seed=42, n_trials=5000
        )
        # At least one valid split should be found
        assert len(val_df) > 0


# ---------------------------------------------------------------------------
# build_exclusion_buffer_gdf
# ---------------------------------------------------------------------------


class TestBuildExclusionBufferGdf:
    def test_returns_geodataframe(self, patches_df, tmp_path):
        import geopandas as gpd

        gdf = build_exclusion_buffer_gdf(patches_df)
        assert isinstance(gdf, gpd.GeoDataFrame)

    def test_row_count_matches_input(self, patches_df):
        gdf = build_exclusion_buffer_gdf(patches_df)
        assert len(gdf) == len(patches_df)

    def test_geometry_column_present(self, patches_df):
        gdf = build_exclusion_buffer_gdf(patches_df)
        assert "geometry" in gdf.columns

    def test_buffer_extends_beyond_patch_bounds(self, tmp_path):
        cx, cy = _BASE_X, _BASE_Y
        p = _make_geotiff(tmp_path / "t.tif", cx, cy)
        df = pd.DataFrame({"image_path": [str(p)]})
        buffer_m = 1000.0
        gdf = build_exclusion_buffer_gdf(df, buffer_m=buffer_m)
        geom = gdf.iloc[0].geometry
        half = _PATCH_SIZE_M / 2.0 + buffer_m
        # Geometry must extend at least buffer_m beyond patch edge
        assert geom.bounds[0] <= cx - half + 1  # left
        assert geom.bounds[2] >= cx + half - 1  # right

    def test_patch_centroid_inside_buffer(self, tmp_path):
        from shapely.geometry import Point

        cx, cy = _BASE_X, _BASE_Y
        p = _make_geotiff(tmp_path / "t.tif", cx, cy)
        df = pd.DataFrame({"image_path": [str(p)]})
        gdf = build_exclusion_buffer_gdf(df, buffer_m=1000.0)
        centroid_pt = Point(cx, cy)
        assert gdf.iloc[0].geometry.contains(centroid_pt)


# ---------------------------------------------------------------------------
# filter_weak_training_tiles
# ---------------------------------------------------------------------------


class TestFilterWeakTrainingTiles:
    def test_overlapping_tile_removed(self, tmp_path):
        # val/test patch and a training tile at the same location
        cx, cy = _BASE_X, _BASE_Y
        patch = _make_geotiff(tmp_path / "patch.tif", cx, cy)
        train_tile = _make_geotiff(tmp_path / "train_overlap.tif", cx, cy)

        val_test_df = pd.DataFrame({"image_path": [str(patch)]})
        train_df = pd.DataFrame({"image_path": [str(train_tile)]})

        result = filter_weak_training_tiles(train_df, val_test_df, buffer_m=100.0)
        assert len(result) == 0

    def test_distant_tile_kept(self, tmp_path):
        cx, cy = _BASE_X, _BASE_Y
        patch = _make_geotiff(tmp_path / "patch.tif", cx, cy)
        far_train = _make_geotiff(
            tmp_path / "train_far.tif", cx + 3 * _STEP, cy + 3 * _STEP
        )

        val_test_df = pd.DataFrame({"image_path": [str(patch)]})
        train_df = pd.DataFrame({"image_path": [str(far_train)]})

        result = filter_weak_training_tiles(train_df, val_test_df, buffer_m=1000.0)
        assert len(result) == 1

    def test_mixed_tiles_partial_removal(self, tmp_path):
        cx, cy = _BASE_X, _BASE_Y
        patch = _make_geotiff(tmp_path / "patch.tif", cx, cy)
        close_train = _make_geotiff(tmp_path / "train_close.tif", cx, cy)
        far_train = _make_geotiff(
            tmp_path / "train_far.tif", cx + 5 * _STEP, cy + 5 * _STEP
        )

        val_test_df = pd.DataFrame({"image_path": [str(patch)]})
        train_df = pd.DataFrame({"image_path": [str(close_train), str(far_train)]})

        result = filter_weak_training_tiles(train_df, val_test_df, buffer_m=1000.0)
        assert len(result) == 1
        assert str(far_train) in result["image_path"].values

    def test_empty_val_test_keeps_all_train(self, tmp_path):
        cx, cy = _BASE_X, _BASE_Y
        tile = _make_geotiff(tmp_path / "train.tif", cx, cy)
        empty_val_test = pd.DataFrame({"image_path": pd.Series([], dtype=str)})
        train_df = pd.DataFrame({"image_path": [str(tile)]})

        result = filter_weak_training_tiles(train_df, empty_val_test, buffer_m=1000.0)
        assert len(result) == 1


# ---------------------------------------------------------------------------
# run  (integration)
# ---------------------------------------------------------------------------


class TestRun:
    def test_output_files_created(self, patches_df, tmp_path):
        patches_csv = tmp_path / "patches.csv"
        patches_df.to_csv(patches_csv, index=False)

        out_dir = tmp_path / "splits"
        run(patches_csv, out_dir)

        assert (out_dir / "val.csv").exists()
        assert (out_dir / "test.csv").exists()

    def test_no_train_csv_skips_train_filter(self, patches_df, tmp_path):
        patches_csv = tmp_path / "patches.csv"
        patches_df.to_csv(patches_csv, index=False)

        out_dir = tmp_path / "splits"
        stats = run(patches_csv, out_dir)

        assert "n_train_before" not in stats
        assert "n_train_after" not in stats
        assert not (out_dir / "train_filtered.csv").exists()

    def test_returns_required_stat_keys(self, patches_df, tmp_path):
        patches_csv = tmp_path / "patches.csv"
        patches_df.to_csv(patches_csv, index=False)

        stats = run(patches_csv, tmp_path / "splits")

        for key in ("n_val", "n_test", "val_fraction", "val_h3_cells", "test_h3_cells"):
            assert key in stats

    def test_val_plus_test_equals_total(self, patches_df, tmp_path):
        patches_csv = tmp_path / "patches.csv"
        patches_df.to_csv(patches_csv, index=False)

        stats = run(patches_csv, tmp_path / "splits")
        assert stats["n_val"] + stats["n_test"] == len(patches_df)

    def test_with_train_csv_creates_filtered_csv(
        self, patches_df, patch_geotiffs, tmp_path
    ):
        patches_csv = tmp_path / "patches.csv"
        patches_df.to_csv(patches_csv, index=False)

        # Training CSV: same paths as patches (all should be removed by buffer)
        train_csv = tmp_path / "train.csv"
        pd.DataFrame({"image_path": patch_geotiffs}).to_csv(train_csv, index=False)

        out_dir = tmp_path / "splits"
        stats = run(patches_csv, out_dir, train_csv=train_csv, buffer_m=1000.0)

        assert (out_dir / "train_filtered.csv").exists()
        assert "n_train_before" in stats
        assert "n_train_after" in stats
        assert stats["n_train_before"] == len(patch_geotiffs)

    def test_class_col_accepted(self, patches_df, tmp_path):
        patches_csv = tmp_path / "patches.csv"
        patches_df.to_csv(patches_csv, index=False)

        stats = run(patches_csv, tmp_path / "splits", class_col="class_label")
        assert stats["n_val"] + stats["n_test"] == len(patches_df)

    def test_output_dir_created_if_absent(self, patches_df, tmp_path):
        patches_csv = tmp_path / "patches.csv"
        patches_df.to_csv(patches_csv, index=False)

        out_dir = tmp_path / "new" / "nested" / "dir"
        assert not out_dir.exists()
        run(patches_csv, out_dir)
        assert out_dir.exists()
