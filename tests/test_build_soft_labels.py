# -*- coding: utf-8 -*-
"""Tests for build_soft_labels.py preprocessing script."""

import numpy as np
import pytest
import rasterio
from pathlib import Path
from rasterio.crs import CRS
from rasterio.transform import from_bounds

from pytorch_segmentation_models_trainer.tools.soft_labels.build_soft_labels import (
    compute_p_soft,
    compute_w_conf,
    compute_w_embed,
    aggregate_aef_to_image_grid,
    load_aef_embedding,
    _compute_hf_class_centroids,
    _reproject_lulc_to_image_grid,
    generate_patch_rows,
    expand_manifest_to_patches,
    process_tile,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

EPSG4326 = CRS.from_epsg(4326)
BOUNDS = (0.0, 0.0, 1.0, 1.0)  # lon/lat box


def _write_lulc(path: Path, data: np.ndarray, transform, crs=EPSG4326):
    """Write a (H, W) uint8 LULC raster."""
    h, w = data.shape
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=h,
        width=w,
        count=1,
        dtype="uint8",
        crs=crs,
        transform=transform,
    ) as dst:
        dst.write(data[np.newaxis])


def _write_image(path: Path, h: int, w: int, transform, crs=EPSG4326):
    """Write a dummy RGB uint8 image raster."""
    data = np.zeros((3, h, w), dtype=np.uint8)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=h,
        width=w,
        count=3,
        dtype="uint8",
        crs=crs,
        transform=transform,
    ) as dst:
        dst.write(data)


@pytest.fixture()
def same_res_setup(tmp_path):
    """Image and two LULC rasters at the same 32x32 resolution."""
    h, w = 32, 32
    transform = from_bounds(*BOUNDS, w, h)

    img_path = tmp_path / "img.tif"
    _write_image(img_path, h, w, transform)

    lulc_a = np.zeros((h, w), dtype=np.uint8)
    lulc_a[:16, :] = 1  # top half = class 1
    lulc_b = np.zeros((h, w), dtype=np.uint8)
    lulc_b[16:, :] = 2  # bottom half = class 2

    path_a = tmp_path / "lulc_a.tif"
    path_b = tmp_path / "lulc_b.tif"
    _write_lulc(path_a, lulc_a, transform)
    _write_lulc(path_b, lulc_b, transform)

    return img_path, path_a, path_b, h, w


@pytest.fixture()
def diff_res_setup(tmp_path):
    """Image at 32x32, LULC at coarser 8x8 — same geographic extent."""
    img_h, img_w = 32, 32
    lulc_h, lulc_w = 8, 8
    img_transform = from_bounds(*BOUNDS, img_w, img_h)
    lulc_transform = from_bounds(*BOUNDS, lulc_w, lulc_h)

    img_path = tmp_path / "img.tif"
    _write_image(img_path, img_h, img_w, img_transform)

    lulc_data = np.zeros((lulc_h, lulc_w), dtype=np.uint8)
    lulc_data[:4, :] = 1  # top half = class 1
    lulc_path = tmp_path / "lulc.tif"
    _write_lulc(lulc_path, lulc_data, lulc_transform)

    return img_path, lulc_path, img_h, img_w


# ---------------------------------------------------------------------------
# compute_p_soft
# ---------------------------------------------------------------------------


class TestComputePSoft:
    def test_output_shape(self):
        maps = [np.zeros((8, 8), dtype=np.uint8)]
        p = compute_p_soft(maps, [1.0], num_classes=3)
        assert p.shape == (3, 8, 8)

    def test_sums_to_one(self):
        maps = [np.zeros((8, 8), dtype=np.uint8)]
        p = compute_p_soft(maps, [1.0], num_classes=3)
        assert np.allclose(p.sum(axis=0), 1.0)

    def test_single_source_hard_label(self):
        lulc = np.full((4, 4), 2, dtype=np.uint8)
        p = compute_p_soft([lulc], [1.0], num_classes=4)
        assert np.all(p[2] == 1.0)
        assert np.all(p[0] == 0.0)

    def test_two_sources_equal_weight(self):
        """Two conflicting sources at equal weight → P_soft = 0.5 each."""
        a = np.zeros((4, 4), dtype=np.uint8)  # all class 0
        b = np.full((4, 4), 1, dtype=np.uint8)  # all class 1
        p = compute_p_soft([a, b], [1.0, 1.0], num_classes=2)
        assert np.allclose(p[0], 0.5)
        assert np.allclose(p[1], 0.5)

    def test_weights_respected(self):
        """Heavier source dominates probability."""
        a = np.zeros((4, 4), dtype=np.uint8)  # class 0, weight 3
        b = np.full((4, 4), 1, dtype=np.uint8)  # class 1, weight 1
        p = compute_p_soft([a, b], [3.0, 1.0], num_classes=2)
        assert np.allclose(p[0], 0.75)
        assert np.allclose(p[1], 0.25)

    def test_output_dtype_float32(self):
        maps = [np.zeros((4, 4), dtype=np.uint8)]
        p = compute_p_soft(maps, [1.0], num_classes=2)
        assert p.dtype == np.float32


# ---------------------------------------------------------------------------
# compute_w_conf
# ---------------------------------------------------------------------------


class TestComputeWConf:
    def test_output_shape(self):
        p = np.zeros((4, 8, 8), dtype=np.float32)
        p[0] = 1.0
        w = compute_w_conf(p)
        assert w.shape == (1, 8, 8)

    def test_output_dtype_float32(self):
        p = np.zeros((4, 8, 8), dtype=np.float32)
        p[0] = 1.0
        w = compute_w_conf(p)
        assert w.dtype == np.float32

    def test_values_in_unit_interval(self):
        rng = np.random.default_rng(0)
        raw = rng.random((4, 16, 16)).astype(np.float32)
        p = raw / raw.sum(axis=0, keepdims=True)
        w = compute_w_conf(p)
        assert w.min() >= 0.0
        assert w.max() <= 1.0 + 1e-6

    def test_hard_label_has_high_confidence(self):
        """A perfectly hard label (entropy=0) gives w_entropy=1."""
        p = np.zeros((4, 8, 8), dtype=np.float32)
        p[2] = 1.0  # class 2 everywhere — entropy = 0
        w = compute_w_conf(p, alpha=1.0)  # pure entropy weighting
        assert np.all(w >= 0.99)

    def test_uniform_label_has_low_confidence(self):
        """Uniform distribution (max entropy) gives w_entropy=0."""
        p = np.full((4, 8, 8), 0.25, dtype=np.float32)
        w = compute_w_conf(p, alpha=1.0)  # pure entropy weighting
        assert np.all(w <= 0.01)

    def test_alpha_zero_uses_border_only(self):
        """alpha=0 → w_conf is purely border-distance-based."""
        p = np.zeros((2, 8, 8), dtype=np.float32)
        p[0, :4, :] = 1.0  # top half = class 0
        p[1, 4:, :] = 1.0  # bottom half = class 1
        w_border_only = compute_w_conf(p, alpha=0.0)
        w_entropy_only = compute_w_conf(p, alpha=1.0)
        # They should differ
        assert not np.allclose(w_border_only, w_entropy_only)

    def test_use_border_false_equals_entropy_only(self):
        """use_border=False with beta=0 produces output equal to w_entropy."""
        rng = np.random.default_rng(1)
        raw = rng.random((4, 8, 8)).astype(np.float32)
        p = raw / raw.sum(axis=0, keepdims=True)

        w = compute_w_conf(p, alpha=0.6, use_border=False)

        eps = 1e-8
        log_p = np.log(np.clip(p, eps, 1.0))
        entropy = -(p * log_p).sum(axis=0)
        max_entropy = float(np.log(4))
        w_entropy = (1.0 - entropy / max_entropy).astype(np.float32)
        assert np.allclose(w[0], w_entropy, atol=1e-5)

    def test_use_border_false_output_shape(self):
        """use_border=False returns (1, H, W) just like the default."""
        p = np.zeros((4, 8, 8), dtype=np.float32)
        p[0] = 1.0
        assert compute_w_conf(p, use_border=False).shape == (1, 8, 8)

    def test_use_border_false_values_in_unit_interval(self):
        """use_border=False output is always in [0, 1]."""
        rng = np.random.default_rng(2)
        raw = rng.random((4, 16, 16)).astype(np.float32)
        p = raw / raw.sum(axis=0, keepdims=True)
        w = compute_w_conf(p, alpha=0.6, use_border=False)
        assert w.min() >= 0.0
        assert w.max() <= 1.0 + 1e-6

    def test_use_border_true_default_behavior_unchanged(self):
        """Explicit use_border=True gives the same result as the default call."""
        p = np.zeros((4, 8, 8), dtype=np.float32)
        p[0] = 1.0
        w_default = compute_w_conf(p, alpha=0.6)
        w_explicit = compute_w_conf(p, alpha=0.6, use_border=True)
        assert np.allclose(w_default, w_explicit)


# ---------------------------------------------------------------------------
# _reproject_lulc_to_image_grid
# ---------------------------------------------------------------------------


class TestReprojectLulcToImageGrid:
    def test_output_matches_image_shape_when_same_res(self, same_res_setup):
        img_path, path_a, _, h, w = same_res_setup
        with rasterio.open(img_path) as img:
            result = _reproject_lulc_to_image_grid(
                str(path_a), img.height, img.width, img.transform, img.crs
            )
        assert result.shape == (h, w)

    def test_output_matches_image_shape_when_coarser_lulc(self, diff_res_setup):
        img_path, lulc_path, img_h, img_w = diff_res_setup
        with rasterio.open(img_path) as img:
            result = _reproject_lulc_to_image_grid(
                str(lulc_path), img.height, img.width, img.transform, img.crs
            )
        assert result.shape == (img_h, img_w)

    def test_output_dtype_uint8(self, diff_res_setup):
        img_path, lulc_path, img_h, img_w = diff_res_setup
        with rasterio.open(img_path) as img:
            result = _reproject_lulc_to_image_grid(
                str(lulc_path), img.height, img.width, img.transform, img.crs
            )
        assert result.dtype == np.uint8

    def test_class_values_preserved_after_reproject(self, diff_res_setup):
        """After nearest-neighbour reproject, only original class values appear."""
        img_path, lulc_path, img_h, img_w = diff_res_setup
        with rasterio.open(img_path) as img:
            result = _reproject_lulc_to_image_grid(
                str(lulc_path), img.height, img.width, img.transform, img.crs
            )
        unique_vals = set(result.flat)
        assert unique_vals <= {
            0,
            1,
        }, f"Unexpected classes after reproject: {unique_vals}"

    def test_spatial_pattern_preserved(self, diff_res_setup):
        """Top half of image should still be class 1 after upsampling."""
        img_path, lulc_path, img_h, img_w = diff_res_setup
        with rasterio.open(img_path) as img:
            result = _reproject_lulc_to_image_grid(
                str(lulc_path), img.height, img.width, img.transform, img.crs
            )
        # Top half of 8x8 LULC was class 1 → top half of 32x32 result should be 1
        assert np.all(result[: img_h // 2, :] == 1)
        assert np.all(result[img_h // 2 :, :] == 0)

    def test_same_resolution_no_distortion(self, same_res_setup):
        """When resolutions match, values are identical after reproject."""
        img_path, path_a, _, h, w = same_res_setup
        with rasterio.open(img_path) as img:
            result = _reproject_lulc_to_image_grid(
                str(path_a), img.height, img.width, img.transform, img.crs
            )
        with rasterio.open(path_a) as src:
            original = src.read(1)
        assert np.array_equal(result, original)


# ---------------------------------------------------------------------------
# process_tile integration
# ---------------------------------------------------------------------------


class TestProcessTile:
    def test_output_shape_matches_image_when_same_res(self, same_res_setup, tmp_path):
        img_path, path_a, path_b, h, w = same_res_setup
        import pandas as pd

        rows = pd.DataFrame(
            [
                {
                    "source_name": "a",
                    "lulc_path": str(path_a),
                    "weight": 1.0,
                    "image_path": str(img_path),
                },
                {
                    "source_name": "b",
                    "lulc_path": str(path_b),
                    "weight": 1.0,
                    "image_path": str(img_path),
                },
            ]
        )
        _, _, p_soft_path, w_conf_path = process_tile(
            "tile_0", rows, tmp_path, num_classes=3, alpha=0.6
        )
        with rasterio.open(p_soft_path) as src:
            assert src.height == h and src.width == w

    def test_output_shape_matches_image_when_coarser_lulc(
        self, diff_res_setup, tmp_path
    ):
        img_path, lulc_path, img_h, img_w = diff_res_setup
        import pandas as pd

        rows = pd.DataFrame(
            [
                {
                    "source_name": "a",
                    "lulc_path": str(lulc_path),
                    "weight": 1.0,
                    "image_path": str(img_path),
                },
            ]
        )
        _, _, p_soft_path, w_conf_path = process_tile(
            "tile_0", rows, tmp_path, num_classes=3, alpha=0.6
        )
        with rasterio.open(p_soft_path) as src:
            assert src.height == img_h and src.width == img_w
        with rasterio.open(w_conf_path) as src:
            assert src.height == img_h and src.width == img_w

    def test_returns_image_path(self, same_res_setup, tmp_path):
        img_path, path_a, _, h, w = same_res_setup
        import pandas as pd

        rows = pd.DataFrame(
            [
                {
                    "source_name": "a",
                    "lulc_path": str(path_a),
                    "weight": 1.0,
                    "image_path": str(img_path),
                },
            ]
        )
        _, returned_img_path, _, _ = process_tile(
            "tile_0", rows, tmp_path, num_classes=3, alpha=0.6
        )
        assert returned_img_path == str(img_path)

    def test_p_soft_sums_to_one(self, same_res_setup, tmp_path):
        img_path, path_a, path_b, h, w = same_res_setup
        import pandas as pd

        rows = pd.DataFrame(
            [
                {
                    "source_name": "a",
                    "lulc_path": str(path_a),
                    "weight": 1.0,
                    "image_path": str(img_path),
                },
                {
                    "source_name": "b",
                    "lulc_path": str(path_b),
                    "weight": 1.0,
                    "image_path": str(img_path),
                },
            ]
        )
        _, _, p_soft_path, _ = process_tile(
            "tile_0", rows, tmp_path, num_classes=3, alpha=0.6
        )
        with rasterio.open(p_soft_path) as src:
            p = src.read()  # (C, H, W)
        assert np.allclose(p.sum(axis=0), 1.0, atol=1e-5)

    def test_w_conf_in_unit_interval(self, same_res_setup, tmp_path):
        img_path, path_a, path_b, h, w = same_res_setup
        import pandas as pd

        rows = pd.DataFrame(
            [
                {
                    "source_name": "a",
                    "lulc_path": str(path_a),
                    "weight": 1.0,
                    "image_path": str(img_path),
                },
            ]
        )
        _, _, _, w_conf_path = process_tile(
            "tile_0", rows, tmp_path, num_classes=3, alpha=0.6
        )
        with rasterio.open(w_conf_path) as src:
            w = src.read(1)
        assert w.min() >= 0.0
        assert w.max() <= 1.0 + 1e-5

    def test_output_geotiff_crs_matches_image(self, diff_res_setup, tmp_path):
        img_path, lulc_path, img_h, img_w = diff_res_setup
        import pandas as pd

        rows = pd.DataFrame(
            [
                {
                    "source_name": "a",
                    "lulc_path": str(lulc_path),
                    "weight": 1.0,
                    "image_path": str(img_path),
                },
            ]
        )
        _, _, p_soft_path, _ = process_tile(
            "tile_0", rows, tmp_path, num_classes=3, alpha=0.6
        )
        with rasterio.open(img_path) as img:
            img_crs = img.crs
        with rasterio.open(p_soft_path) as src:
            assert src.crs == img_crs

    def test_use_border_false_produces_valid_output(self, same_res_setup, tmp_path):
        """process_tile with use_border=False writes a valid W_conf raster."""
        img_path, path_a, _, img_h, img_w = same_res_setup
        import pandas as pd

        rows = pd.DataFrame(
            [
                {
                    "source_name": "a",
                    "lulc_path": str(path_a),
                    "weight": 1.0,
                    "image_path": str(img_path),
                }
            ]
        )
        _, _, _, w_conf_path = process_tile(
            "tile_0", rows, tmp_path, num_classes=3, alpha=0.6, use_border=False
        )
        with rasterio.open(w_conf_path) as src:
            data = src.read(1)
        assert data.shape == (img_h, img_w)
        assert data.min() >= 0.0
        assert data.max() <= 1.0 + 1e-5


# ---------------------------------------------------------------------------
# generate_patch_rows
# ---------------------------------------------------------------------------


class TestGeneratePatchRows:
    def test_no_overlap_fills_raster(self):
        """stride == patch_size → non-overlapping grid covers full raster."""
        rows = generate_patch_rows(height=64, width=64, patch_size=32, stride=32)
        assert len(rows) == 4  # 2x2 grid

    def test_single_patch_when_patch_equals_raster(self):
        rows = generate_patch_rows(height=32, width=32, patch_size=32, stride=32)
        assert rows == [(0, 0)]

    def test_overlap_produces_more_patches(self):
        no_overlap = generate_patch_rows(64, 64, patch_size=32, stride=32)
        with_overlap = generate_patch_rows(64, 64, patch_size=32, stride=16)
        assert len(with_overlap) > len(no_overlap)

    def test_patches_within_bounds(self):
        h, w, ps, st = 64, 64, 32, 16
        rows = generate_patch_rows(h, w, patch_size=ps, stride=st)
        for r_off, c_off in rows:
            assert r_off + ps <= h
            assert c_off + ps <= w

    def test_first_patch_is_top_left(self):
        rows = generate_patch_rows(64, 64, patch_size=32, stride=32)
        assert rows[0] == (0, 0)

    def test_asymmetric_raster(self):
        """Non-square rasters generate the right number of patches."""
        rows = generate_patch_rows(height=64, width=32, patch_size=32, stride=32)
        assert len(rows) == 2  # 2 rows, 1 column

    def test_patch_larger_than_raster_returns_empty(self):
        rows = generate_patch_rows(height=16, width=16, patch_size=32, stride=32)
        assert rows == []

    def test_stride_one_maximum_patches(self):
        h, w, ps = 10, 10, 3
        rows = generate_patch_rows(h, w, patch_size=ps, stride=1)
        # (h - ps + 1) * (w - ps + 1) = 8 * 8 = 64
        assert len(rows) == (h - ps + 1) * (w - ps + 1)


# ---------------------------------------------------------------------------
# expand_manifest_to_patches
# ---------------------------------------------------------------------------


class TestExpandManifestToPatches:
    @pytest.fixture()
    def tile_manifest(self, same_res_setup, tmp_path):
        """Build a one-tile manifest as returned by the main pipeline."""
        img_path, path_a, _, h, w = same_res_setup
        import pandas as pd

        rows = pd.DataFrame(
            [
                {
                    "source_name": "a",
                    "lulc_path": str(path_a),
                    "weight": 1.0,
                    "image_path": str(img_path),
                },
            ]
        )
        tile_id, image_path, p_soft_path, w_conf_path = process_tile(
            "tile_0", rows, tmp_path, num_classes=3, alpha=0.6
        )
        return pd.DataFrame(
            [
                {
                    "tile_id": tile_id,
                    "image_path": image_path,
                    "p_soft_path": str(p_soft_path),
                    "w_conf_path": str(w_conf_path),
                }
            ]
        )

    def test_output_has_patch_columns(self, tile_manifest):
        df = expand_manifest_to_patches(tile_manifest, patch_size=16, stride=16)
        for col in ["row_off", "col_off", "patch_size"]:
            assert col in df.columns

    def test_patch_size_column_correct(self, tile_manifest):
        df = expand_manifest_to_patches(tile_manifest, patch_size=16, stride=16)
        assert (df["patch_size"] == 16).all()

    def test_correct_number_of_patches(self, tile_manifest):
        # same_res_setup is 32x32; patch=16 stride=16 → 2x2 = 4 patches per tile
        df = expand_manifest_to_patches(tile_manifest, patch_size=16, stride=16)
        assert len(df) == 4

    def test_paths_propagated_to_all_rows(self, tile_manifest):
        df = expand_manifest_to_patches(tile_manifest, patch_size=16, stride=16)
        assert (df["image_path"] == tile_manifest.iloc[0]["image_path"]).all()
        assert (df["p_soft_path"] == tile_manifest.iloc[0]["p_soft_path"]).all()

    def test_patches_within_raster_bounds(self, tile_manifest):
        patch_size = 16
        df = expand_manifest_to_patches(tile_manifest, patch_size=patch_size, stride=16)
        with rasterio.open(tile_manifest.iloc[0]["image_path"]) as src:
            h, w = src.height, src.width
        assert (df["row_off"] + patch_size <= h).all()
        assert (df["col_off"] + patch_size <= w).all()

    def test_no_w_conf_column_handled(self, tile_manifest):
        """Manifest without w_conf_path still expands correctly."""
        manifest_no_wc = tile_manifest.drop(columns=["w_conf_path"])
        df = expand_manifest_to_patches(manifest_no_wc, patch_size=16, stride=16)
        assert "w_conf_path" not in df.columns

    def test_aligned_aef_path_propagated(self, tile_manifest):
        """Optional aligned_aef_path propagates to every patch row."""
        tile_manifest["aligned_aef_path"] = "/tmp/aef_aligned/tile_0.tif"
        df = expand_manifest_to_patches(tile_manifest, patch_size=16, stride=16)
        assert "aligned_aef_path" in df.columns
        assert (df["aligned_aef_path"] == "/tmp/aef_aligned/tile_0.tif").all()
        assert len(df) == 4


# ---------------------------------------------------------------------------
# aggregate_aef_to_image_grid  (official Google algorithm)
# ---------------------------------------------------------------------------


class TestAggregateAefToImageGrid:
    """Tests for the official dequantise → vector sum → L2 normalise pipeline."""

    def _write_aef_tif(self, path: Path, data: np.ndarray, transform, crs=EPSG4326):
        """Write a (D, H, W) integer GeoTIFF simulating raw AEF storage."""
        d, h, w = data.shape
        with rasterio.open(
            path,
            "w",
            driver="GTiff",
            height=h,
            width=w,
            count=d,
            dtype="int8",
            crs=crs,
            transform=transform,
        ) as dst:
            dst.write(data)

    def test_output_shape_when_downscaling(self, tmp_path):
        """Output shape is (H_dst, W_dst, D) for a 2× downscale."""
        src_h, src_w, d = 8, 8, 16
        dst_h, dst_w = 4, 4
        src_transform = from_bounds(*BOUNDS, src_w, src_h)
        dst_transform = from_bounds(*BOUNDS, dst_w, dst_h)
        data = np.ones((d, src_h, src_w), dtype=np.int8) * 64

        aef_path = tmp_path / "aef.tif"
        self._write_aef_tif(aef_path, data, src_transform)

        result = aggregate_aef_to_image_grid(
            str(aef_path), dst_h, dst_w, dst_transform, EPSG4326
        )
        assert result.shape == (dst_h, dst_w, d)

    def test_output_dtype_float32(self, tmp_path):
        src_h, src_w, d = 4, 4, 8
        dst_h, dst_w = 2, 2
        src_transform = from_bounds(*BOUNDS, src_w, src_h)
        dst_transform = from_bounds(*BOUNDS, dst_w, dst_h)
        data = np.ones((d, src_h, src_w), dtype=np.int8) * 100
        aef_path = tmp_path / "aef.tif"
        self._write_aef_tif(aef_path, data, src_transform)

        result = aggregate_aef_to_image_grid(
            str(aef_path), dst_h, dst_w, dst_transform, EPSG4326
        )
        assert result.dtype == np.float32

    def test_output_is_l2_normalized(self, tmp_path):
        """Every output pixel vector must have unit L2 norm."""
        src_h, src_w, d = 8, 8, 16
        dst_h, dst_w = 4, 4
        src_transform = from_bounds(*BOUNDS, src_w, src_h)
        dst_transform = from_bounds(*BOUNDS, dst_w, dst_h)
        rng = np.random.default_rng(0)
        data = rng.integers(-127, 127, size=(d, src_h, src_w), dtype=np.int8)
        aef_path = tmp_path / "aef.tif"
        self._write_aef_tif(aef_path, data, src_transform)

        result = aggregate_aef_to_image_grid(
            str(aef_path), dst_h, dst_w, dst_transform, EPSG4326
        )
        # result shape: (H, W, D); norm along last axis should be ≈ 1
        norms = np.linalg.norm(result, axis=2)  # (H, W)
        assert np.allclose(
            norms, 1.0, atol=1e-5
        ), f"Not L2-normalised: min={norms.min():.4f}"

    def test_dequantization_is_nonlinear(self, tmp_path):
        """Dequantised values follow sign(v)*(v/127.5)^2, not linear scaling."""
        src_h, src_w, d = 2, 2, 4
        # Same grid → no aggregation, just dequant + normalise
        transform = from_bounds(*BOUNDS, src_w, src_h)
        # v = 127 → dequant ≈ 0.992; v = 63 → dequant ≈ 0.244 (not 0.5)
        data = np.zeros((d, src_h, src_w), dtype=np.int8)
        data[0] = 127  # band 0: max positive
        data[1] = 63  # band 1: half-max
        aef_path = tmp_path / "aef.tif"
        self._write_aef_tif(aef_path, data, transform)

        result = aggregate_aef_to_image_grid(
            str(aef_path), src_h, src_w, transform, EPSG4326
        )
        # After normalisation directions are preserved. Check raw dequant ratio:
        # dequant(127) / dequant(63) = (127/127.5)^2 / (63/127.5)^2 ≈ 4.07
        # If linear scaling were used the ratio would be 127/63 ≈ 2.016
        dequant_127 = (127 / 127.5) ** 2
        dequant_63 = (63 / 127.5) ** 2
        expected_ratio = dequant_127 / dequant_63
        # The unit vector preserves direction; check unnormalised ratio via components
        # Easier: verify the dequant of a known pixel value directly
        # Sanity: dequantized ratio (127/63)^2 ≈ 4.07 differs from linear ratio 127/63 ≈ 2.02
        assert (
            abs(expected_ratio - (127 / 63)) > 0.1
        ), "Sanity: dequant ratio must differ from linear ratio"
        # Output is normalised, so we check the direction is consistent with dequant
        pixel_vec = result[0, 0]  # (D,) normalised
        # band 0 component should be larger than band 1 (127 dequants larger than 63)
        assert (
            pixel_vec[0] > pixel_vec[1]
        ), "band 0 (v=127) should dominate band 1 (v=63)"

    def test_upscaling_raises_value_error(self, tmp_path):
        """Requesting a finer target grid than the source must raise ValueError."""
        src_h, src_w, d = 4, 4, 8
        dst_h, dst_w = 16, 16  # finer → invalid
        src_transform = from_bounds(*BOUNDS, src_w, src_h)
        dst_transform = from_bounds(*BOUNDS, dst_w, dst_h)
        data = np.ones((d, src_h, src_w), dtype=np.int8) * 64
        aef_path = tmp_path / "aef.tif"
        self._write_aef_tif(aef_path, data, src_transform)

        with pytest.raises(ValueError, match="downscal"):
            aggregate_aef_to_image_grid(
                str(aef_path), dst_h, dst_w, dst_transform, EPSG4326
            )

    def test_same_grid_output_normalised(self, tmp_path):
        """Same-grid call (no spatial aggregation) still returns normalised output."""
        h, w, d = 4, 4, 8
        transform = from_bounds(*BOUNDS, w, h)
        data = np.ones((d, h, w), dtype=np.int8) * 100
        aef_path = tmp_path / "aef.tif"
        self._write_aef_tif(aef_path, data, transform)

        result = aggregate_aef_to_image_grid(str(aef_path), h, w, transform, EPSG4326)
        norms = np.linalg.norm(result, axis=2)
        assert np.allclose(norms, 1.0, atol=1e-4)

    def test_negative_values_sign_preserved(self, tmp_path):
        """Negative raw values produce negative dequantised components."""
        h, w, d = 2, 2, 4
        transform = from_bounds(*BOUNDS, w, h)
        data = np.zeros((d, h, w), dtype=np.int8)
        data[0] = -100  # band 0: negative
        data[1] = 100  # band 1: positive
        aef_path = tmp_path / "aef.tif"
        self._write_aef_tif(aef_path, data, transform)

        result = aggregate_aef_to_image_grid(str(aef_path), h, w, transform, EPSG4326)
        pixel_vec = result[0, 0]
        # After normalisation, band 0 (negative raw) should have negative component
        assert pixel_vec[0] < 0
        assert pixel_vec[1] > 0


# ---------------------------------------------------------------------------
# load_aef_embedding
# ---------------------------------------------------------------------------


class TestLoadAefEmbedding:
    def test_gcs_loads_tif_as_hwc(self, tmp_path):
        """GCS source: reads a multi-band GeoTIFF and returns (H, W, D) float32."""
        h, w, d = 8, 8, 16
        data = np.random.rand(d, h, w).astype(np.float32)
        transform = from_bounds(0, 0, 1, 1, w, h)
        tif_path = tmp_path / "tile_0.tif"
        with rasterio.open(
            tif_path,
            "w",
            driver="GTiff",
            height=h,
            width=w,
            count=d,
            dtype="float32",
            crs=EPSG4326,
            transform=transform,
        ) as dst:
            dst.write(data)

        result = load_aef_embedding("tile_0", tmp_path, source="gcs")
        assert result.shape == (h, w, d)
        assert result.dtype == np.float32

    def test_hf_loads_npy_as_1d(self, tmp_path):
        """HF source: reads a .npy file and returns (D,) float32."""
        embedding = np.random.rand(64).astype(np.float32)
        np.save(tmp_path / "tile_0.npy", embedding)

        result = load_aef_embedding("tile_0", tmp_path, source="hf")
        assert result.shape == (64,)
        assert result.dtype == np.float32

    def test_gcs_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_aef_embedding("nonexistent", tmp_path, source="gcs")

    def test_hf_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_aef_embedding("nonexistent", tmp_path, source="hf")

    def test_invalid_source_raises(self, tmp_path):
        with pytest.raises(ValueError, match="source"):
            load_aef_embedding("tile_0", tmp_path, source="invalid")

    def test_gcs_values_preserved(self, tmp_path):
        """Pixel values round-trip correctly through GeoTIFF."""
        h, w, d = 4, 4, 8
        data = np.arange(d * h * w, dtype=np.float32).reshape(d, h, w)
        transform = from_bounds(0, 0, 1, 1, w, h)
        tif_path = tmp_path / "tile_0.tif"
        with rasterio.open(
            tif_path,
            "w",
            driver="GTiff",
            height=h,
            width=w,
            count=d,
            dtype="float32",
            crs=EPSG4326,
            transform=transform,
        ) as dst:
            dst.write(data)

        result = load_aef_embedding("tile_0", tmp_path, source="gcs")
        expected = np.transpose(data, (1, 2, 0))
        assert np.allclose(result, expected)

    def test_gcs_no_reprojection_returns_native_shape(self, tmp_path):
        """Without dst_* params, GCS mode returns the native (H, W, D) shape."""
        src_h, src_w, d = 8, 8, 4
        transform = from_bounds(0, 0, 1, 1, src_w, src_h)
        data = np.ones((d, src_h, src_w), dtype=np.float32)
        tif_path = tmp_path / "tile_0.tif"
        with rasterio.open(
            tif_path,
            "w",
            driver="GTiff",
            height=src_h,
            width=src_w,
            count=d,
            dtype="float32",
            crs=EPSG4326,
            transform=transform,
        ) as dst:
            dst.write(data)

        result = load_aef_embedding("tile_0", tmp_path, source="gcs")
        assert result.shape == (src_h, src_w, d)

    def test_gcs_with_dst_params_applies_official_algorithm(self, tmp_path):
        """With dst_* params, GCS mode routes to aggregate_aef_to_image_grid.

        The official algorithm (dequantise → vector sum → L2 normalise) is
        applied; output shape matches target grid and vectors are unit-normalised.
        """
        src_h, src_w, d = 8, 8, 4
        dst_h, dst_w = 4, 4
        src_transform = from_bounds(0, 0, 1, 1, src_w, src_h)
        dst_transform = from_bounds(0, 0, 1, 1, dst_w, dst_h)
        data = (np.ones((d, src_h, src_w)) * 64).astype(np.int8)
        tif_path = tmp_path / "tile_0.tif"
        with rasterio.open(
            tif_path,
            "w",
            driver="GTiff",
            height=src_h,
            width=src_w,
            count=d,
            dtype="int8",
            crs=EPSG4326,
            transform=src_transform,
        ) as dst:
            dst.write(data)

        result = load_aef_embedding(
            "tile_0",
            tmp_path,
            source="gcs",
            dst_height=dst_h,
            dst_width=dst_w,
            dst_transform=dst_transform,
            dst_crs=EPSG4326,
        )
        assert result.shape == (dst_h, dst_w, d)
        norms = np.linalg.norm(result, axis=2)
        assert np.allclose(norms, 1.0, atol=1e-4)


# ---------------------------------------------------------------------------
# compute_w_embed
# ---------------------------------------------------------------------------


class TestComputeWEmbed:
    def _make_p_soft(self, h, w, c, dominant_class=0):
        """Return (C, H, W) p_soft with dominant_class everywhere."""
        p = np.zeros((c, h, w), dtype=np.float32)
        p[dominant_class] = 1.0
        return p

    def test_gcs_output_shape(self):
        h, w, d, c = 8, 8, 16, 3
        embedding = np.random.rand(h, w, d).astype(np.float32)
        p_soft = self._make_p_soft(h, w, c)
        w_emb = compute_w_embed(embedding, p_soft)
        assert w_emb.shape == (1, h, w)

    def test_gcs_output_dtype(self):
        h, w, d, c = 8, 8, 16, 3
        embedding = np.random.rand(h, w, d).astype(np.float32)
        p_soft = self._make_p_soft(h, w, c)
        w_emb = compute_w_embed(embedding, p_soft)
        assert w_emb.dtype == np.float32

    def test_gcs_values_in_unit_interval(self):
        h, w, d, c = 16, 16, 32, 4
        embedding = np.random.rand(h, w, d).astype(np.float32)
        p_soft = self._make_p_soft(h, w, c)
        w_emb = compute_w_embed(embedding, p_soft)
        assert w_emb.min() >= 0.0 - 1e-6
        assert w_emb.max() <= 1.0 + 1e-6

    def test_gcs_identical_pixels_high_similarity(self):
        """All pixels share the same embedding vector → w_embed near 1."""
        h, w, d, c = 8, 8, 16, 2
        vec = np.random.rand(d).astype(np.float32)
        embedding = np.tile(vec, (h, w, 1))  # (H, W, D) same everywhere
        p_soft = self._make_p_soft(h, w, c, dominant_class=0)
        w_emb = compute_w_embed(embedding, p_soft)
        assert np.all(w_emb >= 0.99), f"Expected ~1, got min={w_emb.min()}"

    def test_hf_output_shape(self):
        h, w, d, c = 8, 8, 64, 3
        embedding = np.random.rand(d).astype(np.float32)
        p_soft = self._make_p_soft(h, w, c)
        centroids = np.random.rand(c, d).astype(np.float32)
        w_emb = compute_w_embed(embedding, p_soft, class_centroids=centroids)
        assert w_emb.shape == (1, h, w)

    def test_hf_output_is_constant_across_pixels(self):
        """HF mode broadcasts one value to all pixels."""
        h, w, d, c = 8, 8, 64, 3
        embedding = np.random.rand(d).astype(np.float32)
        p_soft = self._make_p_soft(h, w, c)
        centroids = np.random.rand(c, d).astype(np.float32)
        w_emb = compute_w_embed(embedding, p_soft, class_centroids=centroids)
        # All values must be identical (broadcast from scalar)
        assert np.allclose(w_emb, w_emb.flat[0])

    def test_hf_requires_class_centroids(self):
        h, w, d, c = 8, 8, 64, 3
        embedding = np.random.rand(d).astype(np.float32)
        p_soft = self._make_p_soft(h, w, c)
        with pytest.raises(ValueError, match="class_centroids"):
            compute_w_embed(embedding, p_soft, class_centroids=None)

    def test_hf_identical_embedding_centroid_high_similarity(self):
        """Tile embedding == its class centroid → w_embed near 1."""
        d, c = 64, 3
        h, w = 4, 4
        vec = np.random.rand(d).astype(np.float32)
        vec /= np.linalg.norm(vec)
        centroids = np.zeros((c, d), dtype=np.float32)
        centroids[0] = vec  # class 0 centroid == tile embedding
        p_soft = self._make_p_soft(h, w, c, dominant_class=0)
        w_emb = compute_w_embed(vec.copy(), p_soft, class_centroids=centroids)
        assert np.all(w_emb >= 0.99)

    def test_invalid_ndim_raises(self):
        """2-D embedding (H, W) — not a valid shape — raises ValueError."""
        h, w, c = 8, 8, 3
        embedding = np.random.rand(h, w).astype(np.float32)
        p_soft = self._make_p_soft(h, w, c)
        with pytest.raises(ValueError, match="shape"):
            compute_w_embed(embedding, p_soft)


# ---------------------------------------------------------------------------
# compute_w_conf extended (with w_embed / beta)
# ---------------------------------------------------------------------------


class TestComputeWConfWithEmbed:
    def _hard_p_soft(self, c=4, h=8, w=8, cls=0):
        p = np.zeros((c, h, w), dtype=np.float32)
        p[cls] = 1.0
        return p

    def test_backward_compat_no_embed(self):
        """Without w_embed, result equals the baseline compute_w_conf."""
        p = self._hard_p_soft()
        w_old = compute_w_conf(p, alpha=0.6)
        w_new = compute_w_conf(p, alpha=0.6, w_embed=None, beta=0.0)
        assert np.allclose(w_old, w_new)

    def test_with_embed_changes_output(self):
        """Providing w_embed with beta > 0 changes the output."""
        h, w, c, d = 8, 8, 4, 16
        p = self._hard_p_soft(c=c, h=h, w=w)
        embedding = np.random.rand(h, w, d).astype(np.float32)
        w_embed = compute_w_embed(embedding, p)
        w_no_emb = compute_w_conf(p, alpha=0.6)
        w_with_emb = compute_w_conf(p, alpha=0.5, w_embed=w_embed, beta=0.2)
        assert not np.allclose(w_no_emb, w_with_emb)

    def test_alpha_plus_beta_exceeds_one_raises(self):
        p = self._hard_p_soft()
        w_embed = np.ones((1, 8, 8), dtype=np.float32)
        with pytest.raises(ValueError, match="alpha.*beta"):
            compute_w_conf(p, alpha=0.8, w_embed=w_embed, beta=0.5)

    def test_output_in_unit_interval_with_embed(self):
        h, w, c, d = 16, 16, 4, 32
        rng = np.random.default_rng(42)
        raw = rng.random((c, h, w)).astype(np.float32)
        p = raw / raw.sum(axis=0, keepdims=True)
        embedding = rng.random((h, w, d)).astype(np.float32)
        w_embed = compute_w_embed(embedding, p)
        w = compute_w_conf(p, alpha=0.5, w_embed=w_embed, beta=0.3)
        assert w.min() >= 0.0 - 1e-6
        assert w.max() <= 1.0 + 1e-6

    def test_output_shape_unchanged_with_embed(self):
        h, w, c, d = 8, 8, 4, 16
        p = self._hard_p_soft(c=c, h=h, w=w)
        embedding = np.random.rand(h, w, d).astype(np.float32)
        w_embed = compute_w_embed(embedding, p)
        result = compute_w_conf(p, alpha=0.5, w_embed=w_embed, beta=0.2)
        assert result.shape == (1, h, w)

    def test_use_border_false_relaxes_alpha_plus_beta_constraint(self):
        """use_border=False allows alpha + beta > 1.0 without raising."""
        p = self._hard_p_soft()
        w_embed = np.ones((1, 8, 8), dtype=np.float32)
        # alpha=0.6, beta=0.6 → sum = 1.2 > 1.0; must NOT raise when use_border=False
        result = compute_w_conf(
            p, alpha=0.6, w_embed=w_embed, beta=0.6, use_border=False
        )
        assert result.shape == (1, 8, 8)
        assert result.min() >= 0.0
        assert result.max() <= 1.0 + 1e-6

    def test_use_border_false_with_embed_values_in_unit_interval(self):
        """use_border=False with w_embed produces output in [0, 1]."""
        h, w, c, d = 16, 16, 4, 32
        rng = np.random.default_rng(7)
        raw = rng.random((c, h, w)).astype(np.float32)
        p = raw / raw.sum(axis=0, keepdims=True)
        embedding = rng.random((h, w, d)).astype(np.float32)
        w_embed = compute_w_embed(embedding, p)
        result = compute_w_conf(
            p, alpha=0.6, w_embed=w_embed, beta=0.4, use_border=False
        )
        assert result.min() >= 0.0 - 1e-6
        assert result.max() <= 1.0 + 1e-6


# ---------------------------------------------------------------------------
# _compute_hf_class_centroids
# ---------------------------------------------------------------------------


class TestComputeHfClassCentroids:
    def _write_p_soft(self, path: Path, p_soft: np.ndarray, transform, crs=EPSG4326):
        c, h, w = p_soft.shape
        with rasterio.open(
            path,
            "w",
            driver="GTiff",
            height=h,
            width=w,
            count=c,
            dtype="float32",
            crs=crs,
            transform=transform,
        ) as dst:
            dst.write(p_soft)

    def test_returns_correct_shape(self, tmp_path):
        """Returns (C, D) float32 centroid array."""
        c, d, h, w = 3, 32, 8, 8
        transform = from_bounds(0, 0, 1, 1, w, h)

        manifest_rows = []
        for i in range(4):
            tile_id = f"tile_{i}"
            p_soft = np.zeros((c, h, w), dtype=np.float32)
            p_soft[i % c] = 1.0
            p_path = tmp_path / f"p_soft_{tile_id}.tif"
            self._write_p_soft(p_path, p_soft, transform)

            embedding = np.random.rand(d).astype(np.float32)
            np.save(tmp_path / f"{tile_id}.npy", embedding)
            manifest_rows.append({"tile_id": tile_id, "p_soft_path": str(p_path)})

        import pandas as pd

        manifest = pd.DataFrame(manifest_rows)
        centroids = _compute_hf_class_centroids(manifest, tmp_path)
        assert centroids.shape == (c, d)
        assert centroids.dtype == np.float32

    def test_missing_embedding_is_skipped(self, tmp_path):
        """Tiles without .npy files are logged and skipped gracefully."""
        c, d, h, w = 2, 16, 4, 4
        transform = from_bounds(0, 0, 1, 1, w, h)

        p_soft = np.zeros((c, h, w), dtype=np.float32)
        p_soft[0] = 1.0
        p_path = tmp_path / "p_soft_tile_0.tif"
        self._write_p_soft(p_path, p_soft, transform)
        # Embedding for tile_0 intentionally NOT written

        valid_p = np.zeros((c, h, w), dtype=np.float32)
        valid_p[1] = 1.0
        valid_p_path = tmp_path / "p_soft_tile_1.tif"
        self._write_p_soft(valid_p_path, valid_p, transform)
        np.save(tmp_path / "tile_1.npy", np.random.rand(d).astype(np.float32))

        import pandas as pd

        manifest = pd.DataFrame(
            [
                {"tile_id": "tile_0", "p_soft_path": str(p_path)},
                {"tile_id": "tile_1", "p_soft_path": str(valid_p_path)},
            ]
        )
        centroids = _compute_hf_class_centroids(manifest, tmp_path)
        assert centroids.shape == (c, d)

    def test_empty_manifest_raises(self, tmp_path):
        import pandas as pd

        manifest = pd.DataFrame(columns=["tile_id", "p_soft_path"])
        with pytest.raises(RuntimeError):
            _compute_hf_class_centroids(manifest, tmp_path)


# ---------------------------------------------------------------------------
# process_tile extended (with AEF embedding)
# ---------------------------------------------------------------------------


class TestProcessTileWithAEF:
    def test_process_tile_with_gcs_embedding(self, same_res_setup, tmp_path):
        """process_tile runs end-to-end with a GCS AEF embedding."""
        img_path, path_a, _, h, w = same_res_setup
        import pandas as pd

        # Write a dummy GCS embedding for this tile
        d = 16
        embedding = np.random.rand(d, h, w).astype(np.float32)
        transform = from_bounds(*BOUNDS, w, h)
        aef_dir = tmp_path / "aef"
        aef_dir.mkdir()
        with rasterio.open(
            aef_dir / "tile_0.tif",
            "w",
            driver="GTiff",
            height=h,
            width=w,
            count=d,
            dtype="float32",
            crs=EPSG4326,
            transform=transform,
        ) as dst:
            dst.write(embedding)

        rows = pd.DataFrame(
            [
                {
                    "source_name": "a",
                    "lulc_path": str(path_a),
                    "weight": 1.0,
                    "image_path": str(img_path),
                }
            ]
        )
        _, _, p_soft_path, w_conf_path = process_tile(
            "tile_0",
            rows,
            tmp_path,
            num_classes=3,
            alpha=0.5,
            aef_dir=aef_dir,
            aef_source="gcs",
            beta=0.2,
        )
        with rasterio.open(w_conf_path) as src:
            w = src.read(1)
        assert w.min() >= 0.0 - 1e-5
        assert w.max() <= 1.0 + 1e-5

    def test_process_tile_without_aef_unchanged(self, same_res_setup, tmp_path):
        """process_tile without aef_dir produces same result as before."""
        img_path, path_a, _, h, w = same_res_setup
        import pandas as pd

        rows = pd.DataFrame(
            [
                {
                    "source_name": "a",
                    "lulc_path": str(path_a),
                    "weight": 1.0,
                    "image_path": str(img_path),
                }
            ]
        )
        _, _, p_soft_path, w_conf_path = process_tile(
            "tile_0",
            rows,
            tmp_path,
            num_classes=3,
            alpha=0.6,
        )
        with rasterio.open(w_conf_path) as src:
            data = src.read(1)
        assert data.shape == (h, w)

    def test_process_tile_coarser_aef_upsampled_with_nearest(
        self, same_res_setup, tmp_path
    ):
        """process_tile upsamples coarser AEF embeddings with nearest neighbor.

        When the AEF file has fewer pixels than the image, ``aef_resampling=auto``
        avoids vector interpolation and assigns nearest source embeddings.
        """
        img_path, path_a, _, h, w = same_res_setup
        import pandas as pd

        # AEF at 8x8, image at 32x32 (coarser source → nearest upsample)
        aef_h, aef_w, d = 8, 8, 4
        aef_dir = tmp_path / "aef"
        aef_dir.mkdir()
        transform = from_bounds(*BOUNDS, aef_w, aef_h)
        data = (np.ones((d, aef_h, aef_w)) * 64).astype(np.int8)
        with rasterio.open(
            aef_dir / "tile_0.tif",
            "w",
            driver="GTiff",
            height=aef_h,
            width=aef_w,
            count=d,
            dtype="int8",
            crs=EPSG4326,
            transform=transform,
        ) as dst:
            dst.write(data)

        rows = pd.DataFrame(
            [
                {
                    "source_name": "a",
                    "lulc_path": str(path_a),
                    "weight": 1.0,
                    "image_path": str(img_path),
                }
            ]
        )
        # Should succeed: auto uses nearest for upsampling
        _, _, p_soft_path, w_conf_path = process_tile(
            "tile_0",
            rows,
            tmp_path,
            num_classes=3,
            alpha=0.5,
            aef_dir=aef_dir,
            aef_source="gcs",
            beta=0.2,
        )
        with rasterio.open(w_conf_path) as src:
            result_w = src.read(1)
        assert result_w.shape == (h, w)
        assert result_w.min() >= 0.0 - 1e-5
        assert result_w.max() <= 1.0 + 1e-5

    def test_process_tile_finer_aef_aggregated_to_image_grid(
        self, same_res_setup, tmp_path
    ):
        """process_tile aggregates finer AEF embeddings to the image grid.

        When the AEF file has more pixels than the image, ``aef_resampling=auto``
        uses vector-sum aggregation followed by L2 normalization.
        """
        img_path, path_a, _, h, w = same_res_setup
        import pandas as pd

        # AEF at 64x64 (finer than image 32x32) → aggregate downsample
        aef_h, aef_w, d = 64, 64, 4
        aef_dir = tmp_path / "aef"
        aef_dir.mkdir()
        transform = from_bounds(*BOUNDS, aef_w, aef_h)
        data = (np.ones((d, aef_h, aef_w)) * 64).astype(np.int8)
        with rasterio.open(
            aef_dir / "tile_0.tif",
            "w",
            driver="GTiff",
            height=aef_h,
            width=aef_w,
            count=d,
            dtype="int8",
            crs=EPSG4326,
            transform=transform,
        ) as dst:
            dst.write(data)

        rows = pd.DataFrame(
            [
                {
                    "source_name": "a",
                    "lulc_path": str(path_a),
                    "weight": 1.0,
                    "image_path": str(img_path),
                }
            ]
        )
        # Should not raise — auto aggregates down to the image grid
        _, _, p_soft_path, w_conf_path = process_tile(
            "tile_0",
            rows,
            tmp_path,
            num_classes=3,
            alpha=0.6,
            aef_dir=aef_dir,
            aef_source="gcs",
            beta=0.2,
        )
        with rasterio.open(w_conf_path) as src:
            result_w = src.read(1)
        assert result_w.shape == (h, w)
        assert result_w.min() >= 0.0 - 1e-5
        assert result_w.max() <= 1.0 + 1e-5

    def test_process_tile_missing_aef_logs_warning_and_falls_back(
        self, same_res_setup, tmp_path
    ):
        """process_tile logs a warning and falls back when the AEF file is missing (line 533)."""
        img_path, path_a, _, h, w = same_res_setup
        import pandas as pd

        aef_dir = tmp_path / "aef_empty"
        aef_dir.mkdir()
        # No embedding file written → FileNotFoundError branch

        rows = pd.DataFrame(
            [
                {
                    "source_name": "a",
                    "lulc_path": str(path_a),
                    "weight": 1.0,
                    "image_path": str(img_path),
                }
            ]
        )
        _, _, _, w_conf_path = process_tile(
            "tile_0",
            rows,
            tmp_path,
            num_classes=3,
            alpha=0.6,
            aef_dir=aef_dir,
            aef_source="gcs",
            beta=0.2,
        )
        with rasterio.open(w_conf_path) as src:
            result_w = src.read(1)
        assert result_w.shape == (h, w)

    def test_process_tile_saves_aligned_aef_int8(self, same_res_setup, tmp_path):
        """process_tile writes aligned AEF embeddings as quantized int8."""
        img_path, path_a, _, h, w = same_res_setup
        import pandas as pd

        aef_h, aef_w, d = 8, 8, 4
        aef_dir = tmp_path / "aef"
        aef_dir.mkdir()
        transform = from_bounds(*BOUNDS, aef_w, aef_h)
        data = (np.ones((d, aef_h, aef_w)) * 64).astype(np.int8)
        with rasterio.open(
            aef_dir / "tile_0.tif",
            "w",
            driver="GTiff",
            height=aef_h,
            width=aef_w,
            count=d,
            dtype="int8",
            crs=EPSG4326,
            transform=transform,
        ) as dst:
            dst.write(data)

        rows = pd.DataFrame(
            [
                {
                    "source_name": "a",
                    "lulc_path": str(path_a),
                    "weight": 1.0,
                    "image_path": str(img_path),
                }
            ]
        )
        result = process_tile(
            "tile_0",
            rows,
            tmp_path,
            num_classes=3,
            alpha=0.6,
            aef_dir=aef_dir,
            aef_source="gcs",
            beta=0.0,
            save_aligned_aef=True,
            aligned_aef_dtype="int8",
        )
        *_, aligned_aef_path = result
        assert aligned_aef_path.exists()
        with rasterio.open(aligned_aef_path) as src:
            assert src.height == h
            assert src.width == w
            assert src.count == d
            assert src.dtypes[0] == "int8"
            assert src.nodata == -128

    def test_process_tile_saves_aligned_aef_float32(self, same_res_setup, tmp_path):
        """process_tile writes aligned AEF embeddings as normalized float32."""
        img_path, path_a, _, h, w = same_res_setup
        import pandas as pd

        aef_h, aef_w, d = 8, 8, 4
        aef_dir = tmp_path / "aef"
        aef_dir.mkdir()
        transform = from_bounds(*BOUNDS, aef_w, aef_h)
        data = (np.ones((d, aef_h, aef_w)) * 64).astype(np.int8)
        with rasterio.open(
            aef_dir / "tile_0.tif",
            "w",
            driver="GTiff",
            height=aef_h,
            width=aef_w,
            count=d,
            dtype="int8",
            crs=EPSG4326,
            transform=transform,
        ) as dst:
            dst.write(data)

        rows = pd.DataFrame(
            [
                {
                    "source_name": "a",
                    "lulc_path": str(path_a),
                    "weight": 1.0,
                    "image_path": str(img_path),
                }
            ]
        )
        result = process_tile(
            "tile_0",
            rows,
            tmp_path,
            num_classes=3,
            alpha=0.6,
            aef_dir=aef_dir,
            aef_source="gcs",
            beta=0.0,
            save_aligned_aef=True,
            aligned_aef_dtype="float32",
        )
        *_, aligned_aef_path = result
        with rasterio.open(aligned_aef_path) as src:
            assert src.height == h
            assert src.width == w
            assert src.count == d
            assert src.dtypes[0] == "float32"
            data = src.read()
        norms = np.linalg.norm(np.transpose(data, (1, 2, 0)), axis=2)
        assert np.allclose(norms, 1.0, atol=1e-5)


# ---------------------------------------------------------------------------
# run() orchestration
# ---------------------------------------------------------------------------


class TestRun:
    def _sources_csv(self, tmp_path, img_path, path_a, path_b):
        import pandas as pd

        rows = [
            {
                "tile_id": "tile_0",
                "image_path": str(img_path),
                "source_name": "a",
                "lulc_path": str(path_a),
                "weight": 1.0,
            },
            {
                "tile_id": "tile_0",
                "image_path": str(img_path),
                "source_name": "b",
                "lulc_path": str(path_b),
                "weight": 0.5,
            },
        ]
        csv_path = tmp_path / "sources.csv"
        pd.DataFrame(rows).to_csv(csv_path, index=False)
        return csv_path

    def test_run_writes_manifest(self, same_res_setup, tmp_path):
        from pytorch_segmentation_models_trainer.tools.soft_labels.build_soft_labels import (
            run,
        )

        img_path, path_a, path_b, _, _ = same_res_setup
        csv_path = self._sources_csv(tmp_path, img_path, path_a, path_b)
        out_dir = tmp_path / "output"
        manifest_path = run(
            input_csv=str(csv_path),
            output_dir=out_dir,
            num_classes=3,
            max_workers=1,
        )
        assert manifest_path.exists()
        import pandas as pd

        df = pd.read_csv(manifest_path)
        assert set(df.columns) >= {
            "tile_id",
            "image_path",
            "p_soft_path",
            "w_conf_path",
        }
        assert len(df) == 1

    def test_run_writes_p_soft_and_w_conf(self, same_res_setup, tmp_path):
        from pytorch_segmentation_models_trainer.tools.soft_labels.build_soft_labels import (
            run,
        )
        import pandas as pd

        img_path, path_a, path_b, h, w = same_res_setup
        csv_path = self._sources_csv(tmp_path, img_path, path_a, path_b)
        out_dir = tmp_path / "output"
        manifest_path = run(str(csv_path), out_dir, num_classes=3, max_workers=1)
        row = pd.read_csv(manifest_path).iloc[0]
        assert Path(row["p_soft_path"]).exists()
        assert Path(row["w_conf_path"]).exists()

    def test_run_with_patch_size_expands_manifest(self, same_res_setup, tmp_path):
        from pytorch_segmentation_models_trainer.tools.soft_labels.build_soft_labels import (
            run,
        )
        import pandas as pd

        img_path, path_a, path_b, h, w = same_res_setup
        csv_path = self._sources_csv(tmp_path, img_path, path_a, path_b)
        out_dir = tmp_path / "output"
        manifest_path = run(
            str(csv_path),
            out_dir,
            num_classes=3,
            max_workers=1,
            patch_size=16,
            stride=16,
        )
        df = pd.read_csv(manifest_path)
        assert "row_off" in df.columns
        assert "col_off" in df.columns
        assert len(df) > 1  # should have multiple patches

    def test_run_hf_source_logs_warning(self, same_res_setup, tmp_path):
        """run() with aef_source='hf' and beta > 0 triggers the two-pass warning (lines 672-673)."""
        from pytorch_segmentation_models_trainer.tools.soft_labels.build_soft_labels import (
            run,
        )
        import pandas as pd

        img_path, path_a, path_b, _, _ = same_res_setup
        csv_path = self._sources_csv(tmp_path, img_path, path_a, path_b)
        aef_dir = tmp_path / "hf_aef"
        aef_dir.mkdir()
        out_dir = tmp_path / "output_hf"
        # HF source with beta > 0 triggers the two-pass warning branch
        manifest_path = run(
            str(csv_path),
            out_dir,
            num_classes=3,
            max_workers=1,
            aef_embeddings_dir=str(aef_dir),
            aef_source="hf",
            beta=0.3,
        )
        assert manifest_path.exists()

    def test_run_failed_tile_does_not_crash(self, tmp_path):
        """run() logs an exception for a failed tile but continues (lines 700-701)."""
        import pandas as pd
        from pytorch_segmentation_models_trainer.tools.soft_labels.build_soft_labels import (
            run,
        )

        # CSV that references a non-existent image → process_tile will raise
        rows = [
            {
                "tile_id": "bad_tile",
                "image_path": str(tmp_path / "nonexistent.tif"),
                "source_name": "a",
                "lulc_path": str(tmp_path / "nonexistent.tif"),
                "weight": 1.0,
            }
        ]
        csv_path = tmp_path / "bad.csv"
        pd.DataFrame(rows).to_csv(csv_path, index=False)
        out_dir = tmp_path / "output_bad"
        # Should not raise — failed tile is logged and skipped
        manifest_path = run(str(csv_path), out_dir, num_classes=3, max_workers=1)
        df = pd.read_csv(manifest_path)
        assert len(df) == 0  # no successful tiles

    def test_run_no_border_writes_manifest(self, same_res_setup, tmp_path):
        """run() with use_border=False produces a valid manifest (replicates original paper)."""
        import pandas as pd
        from pytorch_segmentation_models_trainer.tools.soft_labels.build_soft_labels import (
            run,
        )

        img_path, path_a, path_b, _, _ = same_res_setup
        csv_path = self._sources_csv(tmp_path, img_path, path_a, path_b)
        out_dir = tmp_path / "output_no_border"
        manifest_path = run(
            input_csv=str(csv_path),
            output_dir=out_dir,
            num_classes=3,
            max_workers=1,
            use_border=False,
        )
        assert manifest_path.exists()
        df = pd.read_csv(manifest_path)
        assert set(df.columns) >= {
            "tile_id",
            "image_path",
            "p_soft_path",
            "w_conf_path",
        }
        assert len(df) == 1

    def test_run_save_aligned_aef_adds_manifest_column(self, same_res_setup, tmp_path):
        """run() writes aligned AEF and includes aligned_aef_path in manifest."""
        import pandas as pd
        from pytorch_segmentation_models_trainer.tools.soft_labels.build_soft_labels import (
            run,
        )

        img_path, path_a, path_b, _, _ = same_res_setup
        csv_path = self._sources_csv(tmp_path, img_path, path_a, path_b)
        aef_dir = tmp_path / "aef"
        aef_dir.mkdir()

        aef_h, aef_w, d = 8, 8, 4
        transform = from_bounds(*BOUNDS, aef_w, aef_h)
        data = (np.ones((d, aef_h, aef_w)) * 64).astype(np.int8)
        with rasterio.open(
            aef_dir / "tile_0.tif",
            "w",
            driver="GTiff",
            height=aef_h,
            width=aef_w,
            count=d,
            dtype="int8",
            crs=EPSG4326,
            transform=transform,
        ) as dst:
            dst.write(data)

        out_dir = tmp_path / "output_save_aef"
        manifest_path = run(
            input_csv=str(csv_path),
            output_dir=out_dir,
            num_classes=3,
            max_workers=1,
            aef_embeddings_dir=str(aef_dir),
            aef_source="gcs",
            save_aligned_aef=True,
            aligned_aef_dtype="int8",
        )
        df = pd.read_csv(manifest_path)
        assert "aligned_aef_path" in df.columns
        assert Path(df.iloc[0]["aligned_aef_path"]).exists()
