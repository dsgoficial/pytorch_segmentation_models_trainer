# -*- coding: utf-8 -*-
"""Tests for SoftLabelCachedDataset."""

import numpy as np
import pandas as pd
import pytest
import rasterio
import torch
from pathlib import Path
from rasterio.crs import CRS
from rasterio.transform import from_bounds

from pytorch_segmentation_models_trainer.dataset_loader.soft_label_cached_dataset import (
    SoftLabelCachedDataset,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

EPSG4326 = CRS.from_epsg(4326)
BOUNDS = (0.0, 0.0, 1.0, 1.0)
IMG_H, IMG_W = 32, 32
NUM_CLASSES = 3
TRANSFORM = from_bounds(*BOUNDS, IMG_W, IMG_H)


def _write_image(path: Path) -> None:
    data = np.zeros((3, IMG_H, IMG_W), dtype=np.uint8)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=IMG_H,
        width=IMG_W,
        count=3,
        dtype="uint8",
        crs=EPSG4326,
        transform=TRANSFORM,
    ) as dst:
        dst.write(data)


def _write_lulc(path: Path, data: np.ndarray) -> None:
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=IMG_H,
        width=IMG_W,
        count=1,
        dtype="uint8",
        crs=EPSG4326,
        transform=TRANSFORM,
    ) as dst:
        dst.write(data[np.newaxis])


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def sources_setup(tmp_path):
    """Two tiles, each with two LULC sources — writes all rasters and sources.csv."""
    rows = []
    for i in range(2):
        tile_id = f"tile_{i}"
        tile_dir = tmp_path / tile_id
        tile_dir.mkdir()

        img_path = tile_dir / "image.tif"
        _write_image(img_path)

        lulc_a = np.zeros((IMG_H, IMG_W), dtype=np.uint8)
        lulc_a[: IMG_H // 2, :] = 1
        lulc_a_path = tile_dir / "lulc_a.tif"
        _write_lulc(lulc_a_path, lulc_a)

        lulc_b = np.zeros((IMG_H, IMG_W), dtype=np.uint8)
        lulc_b[IMG_H // 2 :, :] = 2
        lulc_b_path = tile_dir / "lulc_b.tif"
        _write_lulc(lulc_b_path, lulc_b)

        rows.append(
            {
                "tile_id": tile_id,
                "image_path": str(img_path),
                "source_name": "a",
                "lulc_path": str(lulc_a_path),
                "weight": 1.0,
            }
        )
        rows.append(
            {
                "tile_id": tile_id,
                "image_path": str(img_path),
                "source_name": "b",
                "lulc_path": str(lulc_b_path),
                "weight": 1.0,
            }
        )

    csv_path = tmp_path / "sources.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    return csv_path


def _make_ds(sources_setup, tmp_path, **kwargs) -> SoftLabelCachedDataset:
    cache_dir = tmp_path / "cache"
    defaults = dict(
        sources_csv=str(sources_setup),
        cache_dir=str(cache_dir),
        num_classes=NUM_CLASSES,
        alpha=0.6,
        use_border=False,
    )
    defaults.update(kwargs)
    return SoftLabelCachedDataset(**defaults)


# ---------------------------------------------------------------------------
# __len__ and indexing
# ---------------------------------------------------------------------------


class TestSoftLabelCachedDatasetLen:
    def test_len_equals_num_unique_tiles(self, sources_setup, tmp_path):
        ds = _make_ds(sources_setup, tmp_path)
        assert len(ds) == 2

    def test_n_first_rows_to_read_limits_tiles(self, sources_setup, tmp_path):
        ds = _make_ds(sources_setup, tmp_path, n_first_rows_to_read=1)
        assert len(ds) == 1

    def test_index_wraps_modulo(self, sources_setup, tmp_path):
        ds = _make_ds(sources_setup, tmp_path)
        sample_0 = ds[0]
        sample_len = ds[len(ds)]
        assert sample_0["path"] == sample_len["path"]


# ---------------------------------------------------------------------------
# Output structure
# ---------------------------------------------------------------------------


class TestSoftLabelCachedDatasetOutput:
    def test_returns_image_mask_path_keys(self, sources_setup, tmp_path):
        ds = _make_ds(sources_setup, tmp_path)
        sample = ds[0]
        assert "image" in sample
        assert "mask" in sample
        assert "path" in sample

    def test_mask_dict_has_p_soft_and_w_conf(self, sources_setup, tmp_path):
        ds = _make_ds(sources_setup, tmp_path)
        sample = ds[0]
        assert "mask" in sample["mask"]
        assert "w_conf" in sample["mask"]

    def test_image_shape_and_dtype(self, sources_setup, tmp_path):
        ds = _make_ds(sources_setup, tmp_path)
        img = ds[0]["image"]
        assert img.shape == (3, IMG_H, IMG_W)
        assert img.dtype == torch.float32

    def test_image_values_in_unit_interval(self, sources_setup, tmp_path):
        ds = _make_ds(sources_setup, tmp_path)
        img = ds[0]["image"]
        assert img.min() >= 0.0
        assert img.max() <= 1.0 + 1e-5

    def test_p_soft_shape(self, sources_setup, tmp_path):
        ds = _make_ds(sources_setup, tmp_path)
        p_soft = ds[0]["mask"]["mask"]
        assert p_soft.shape == (NUM_CLASSES, IMG_H, IMG_W)

    def test_p_soft_dtype_float32(self, sources_setup, tmp_path):
        ds = _make_ds(sources_setup, tmp_path)
        assert ds[0]["mask"]["mask"].dtype == torch.float32

    def test_p_soft_sums_to_one(self, sources_setup, tmp_path):
        ds = _make_ds(sources_setup, tmp_path)
        p_soft = ds[0]["mask"]["mask"]
        assert torch.allclose(p_soft.sum(dim=0), torch.ones(IMG_H, IMG_W), atol=1e-5)

    def test_w_conf_shape(self, sources_setup, tmp_path):
        ds = _make_ds(sources_setup, tmp_path)
        w_conf = ds[0]["mask"]["w_conf"]
        assert w_conf.shape == (1, IMG_H, IMG_W)

    def test_w_conf_dtype_float32(self, sources_setup, tmp_path):
        ds = _make_ds(sources_setup, tmp_path)
        assert ds[0]["mask"]["w_conf"].dtype == torch.float32

    def test_w_conf_values_in_unit_interval(self, sources_setup, tmp_path):
        ds = _make_ds(sources_setup, tmp_path)
        w_conf = ds[0]["mask"]["w_conf"]
        assert w_conf.min() >= 0.0 - 1e-5
        assert w_conf.max() <= 1.0 + 1e-5

    def test_path_is_string(self, sources_setup, tmp_path):
        ds = _make_ds(sources_setup, tmp_path)
        assert isinstance(ds[0]["path"], str)

    def test_all_tiles_accessible(self, sources_setup, tmp_path):
        ds = _make_ds(sources_setup, tmp_path)
        for i in range(len(ds)):
            sample = ds[i]
            assert sample["mask"]["mask"].shape == (NUM_CLASSES, IMG_H, IMG_W)


# ---------------------------------------------------------------------------
# Caching behaviour
# ---------------------------------------------------------------------------


class TestSoftLabelCachedDatasetCache:
    def test_cache_file_created_after_first_access(self, sources_setup, tmp_path):
        cache_dir = tmp_path / "cache"
        ds = _make_ds(sources_setup, tmp_path, cache_dir=str(cache_dir))
        ds[0]
        assert (cache_dir / "tile_0.tif").exists()

    def test_cache_file_is_valid_geotiff(self, sources_setup, tmp_path):
        cache_dir = tmp_path / "cache"
        ds = _make_ds(sources_setup, tmp_path, cache_dir=str(cache_dir))
        ds[0]
        with rasterio.open(cache_dir / "tile_0.tif") as src:
            data = src.read()
        assert data.shape == (NUM_CLASSES, IMG_H, IMG_W)
        assert data.dtype == np.float32

    def test_cache_file_not_rewritten_on_second_access(self, sources_setup, tmp_path):
        cache_dir = tmp_path / "cache"
        ds = _make_ds(sources_setup, tmp_path, cache_dir=str(cache_dir))
        ds[0]
        cache_file = cache_dir / "tile_0.tif"
        mtime_first = cache_file.stat().st_mtime_ns
        ds[0]
        mtime_second = cache_file.stat().st_mtime_ns
        assert mtime_first == mtime_second

    def test_tmp_file_not_left_after_successful_write(self, sources_setup, tmp_path):
        cache_dir = tmp_path / "cache"
        ds = _make_ds(sources_setup, tmp_path, cache_dir=str(cache_dir))
        ds[0]
        assert not (cache_dir / "tile_0.tif.tmp").exists()

    def test_stale_tmp_file_overwritten(self, sources_setup, tmp_path):
        """A leftover .tmp from a crashed write does not prevent caching."""
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()
        stale_tmp = cache_dir / "tile_0.tif.tmp"
        stale_tmp.write_bytes(b"corrupted")

        ds = _make_ds(sources_setup, tmp_path, cache_dir=str(cache_dir))
        sample = ds[0]
        assert (cache_dir / "tile_0.tif").exists()
        assert sample["mask"]["mask"].shape == (NUM_CLASSES, IMG_H, IMG_W)

    def test_different_alpha_reuses_same_p_soft_cache(self, sources_setup, tmp_path):
        """Changing alpha does not invalidate or re-write the P_soft cache."""
        cache_dir = tmp_path / "cache"
        ds1 = _make_ds(
            sources_setup,
            tmp_path,
            cache_dir=str(cache_dir),
            alpha=0.6,
            use_border=False,
        )
        ds2 = _make_ds(
            sources_setup,
            tmp_path,
            cache_dir=str(cache_dir),
            alpha=1.0,
            use_border=False,
        )
        ds1[0]
        mtime_first = (cache_dir / "tile_0.tif").stat().st_mtime_ns
        ds2[0]
        mtime_second = (cache_dir / "tile_0.tif").stat().st_mtime_ns
        assert mtime_first == mtime_second

    def test_different_alpha_produces_different_w_conf(self, sources_setup, tmp_path):
        """Same cached P_soft, different alpha → different W_conf (use_border=True mixes entropy + border)."""
        cache_dir = tmp_path / "cache"
        # use_border=True: w_conf = alpha*w_entropy + (1-alpha)*w_border
        # alpha=0.6 and alpha=1.0 give different blends
        ds06 = _make_ds(
            sources_setup,
            tmp_path,
            cache_dir=str(cache_dir),
            alpha=0.6,
            use_border=True,
        )
        ds10 = _make_ds(
            sources_setup,
            tmp_path,
            cache_dir=str(cache_dir),
            alpha=1.0,
            use_border=True,
        )
        ds06[0]  # populate cache
        w06 = ds06[0]["mask"]["w_conf"]
        w10 = ds10[0]["mask"]["w_conf"]
        assert not torch.allclose(w06, w10)

    def test_cache_dir_created_if_missing(self, sources_setup, tmp_path):
        """cache_dir is created automatically if it does not exist."""
        cache_dir = tmp_path / "nonexistent" / "deep" / "cache"
        assert not cache_dir.exists()
        ds = _make_ds(sources_setup, tmp_path, cache_dir=str(cache_dir))
        ds[0]
        assert cache_dir.exists()


# ---------------------------------------------------------------------------
# Parameter variants
# ---------------------------------------------------------------------------


class TestSoftLabelCachedDatasetParams:
    def test_entropy_norm_minmax_produces_valid_output(self, sources_setup, tmp_path):
        ds = _make_ds(sources_setup, tmp_path, entropy_norm="minmax", use_border=False)
        w_conf = ds[0]["mask"]["w_conf"]
        assert w_conf.min() >= 0.0 - 1e-5
        assert w_conf.max() <= 1.0 + 1e-5

    def test_use_border_true_produces_valid_output(self, sources_setup, tmp_path):
        ds = _make_ds(sources_setup, tmp_path, use_border=True, alpha=0.6)
        w_conf = ds[0]["mask"]["w_conf"]
        assert w_conf.min() >= 0.0 - 1e-5
        assert w_conf.max() <= 1.0 + 1e-5

    def test_entropy_norm_max_entropy_matches_explicit_default(
        self, sources_setup, tmp_path
    ):
        """entropy_norm='max_entropy' gives same result as not passing the param."""
        cache_a = tmp_path / "cache_a"
        cache_b = tmp_path / "cache_b"
        ds_default = _make_ds(
            sources_setup, tmp_path, cache_dir=str(cache_a), use_border=False
        )
        ds_explicit = _make_ds(
            sources_setup,
            tmp_path,
            cache_dir=str(cache_b),
            entropy_norm="max_entropy",
            use_border=False,
        )
        w_default = ds_default[0]["mask"]["w_conf"]
        w_explicit = ds_explicit[0]["mask"]["w_conf"]
        assert torch.allclose(w_default, w_explicit, atol=1e-5)

    def test_minmax_and_max_entropy_produce_different_w_conf(
        self, sources_setup, tmp_path
    ):
        """Different entropy_norm modes produce different W_conf on mixed input."""
        cache_me = tmp_path / "cache_me"
        cache_mm = tmp_path / "cache_mm"
        ds_me = _make_ds(
            sources_setup,
            tmp_path,
            cache_dir=str(cache_me),
            entropy_norm="max_entropy",
            use_border=False,
        )
        ds_mm = _make_ds(
            sources_setup,
            tmp_path,
            cache_dir=str(cache_mm),
            entropy_norm="minmax",
            use_border=False,
        )
        w_me = ds_me[0]["mask"]["w_conf"]
        w_mm = ds_mm[0]["mask"]["w_conf"]
        assert not torch.allclose(w_me, w_mm)


# ---------------------------------------------------------------------------
# Windowed read
# ---------------------------------------------------------------------------


class TestSoftLabelCachedDatasetWindowed:
    def test_no_patch_size_keeps_tile_level_len(self, sources_setup, tmp_path):
        ds = _make_ds(sources_setup, tmp_path)
        assert len(ds) == 2

    def test_windowed_len_is_tiles_times_patches(self, sources_setup, tmp_path):
        # 2 tiles, patch 16×16 on 32×32 → (32/16)^2 = 4 per tile → 8 total
        ds = _make_ds(sources_setup, tmp_path, patch_size=(16, 16))
        assert len(ds) == 8

    def test_windowed_output_image_shape(self, sources_setup, tmp_path):
        ds = _make_ds(sources_setup, tmp_path, patch_size=(16, 16))
        assert ds[0]["image"].shape == (3, 16, 16)

    def test_windowed_output_p_soft_shape(self, sources_setup, tmp_path):
        ds = _make_ds(sources_setup, tmp_path, patch_size=(16, 16))
        assert ds[0]["mask"]["mask"].shape == (NUM_CLASSES, 16, 16)

    def test_windowed_output_w_conf_shape(self, sources_setup, tmp_path):
        ds = _make_ds(sources_setup, tmp_path, patch_size=(16, 16))
        assert ds[0]["mask"]["w_conf"].shape == (1, 16, 16)

    def test_windowed_w_conf_values_in_unit_interval(self, sources_setup, tmp_path):
        ds = _make_ds(sources_setup, tmp_path, patch_size=(16, 16))
        w_conf = ds[0]["mask"]["w_conf"]
        assert w_conf.min() >= -1e-5
        assert w_conf.max() <= 1.0 + 1e-5

    def test_windowed_p_soft_sums_to_one(self, sources_setup, tmp_path):
        ds = _make_ds(sources_setup, tmp_path, patch_size=(16, 16))
        p_soft = ds[0]["mask"]["mask"]
        assert torch.allclose(p_soft.sum(dim=0), torch.ones(16, 16), atol=1e-5)

    def test_windowed_all_patches_accessible(self, sources_setup, tmp_path):
        ds = _make_ds(sources_setup, tmp_path, patch_size=(16, 16))
        for i in range(len(ds)):
            s = ds[i]
            assert s["mask"]["mask"].shape == (NUM_CLASSES, 16, 16)

    def test_windowed_cache_is_full_tile(self, sources_setup, tmp_path):
        cache_dir = tmp_path / "cache"
        ds = _make_ds(
            sources_setup, tmp_path, cache_dir=str(cache_dir), patch_size=(16, 16)
        )
        ds[0]
        with rasterio.open(cache_dir / "tile_0.tif") as src:
            assert src.read().shape == (NUM_CLASSES, IMG_H, IMG_W)

    def test_windowed_stride_changes_count(self, sources_setup, tmp_path):
        # stride (8,8): range(0,17,8)=[0,8,16] → 3×3=9 per tile, 18 total
        ds = _make_ds(sources_setup, tmp_path, patch_size=(16, 16), patch_stride=(8, 8))
        assert len(ds) == 18

    def test_windowed_index_modulo(self, sources_setup, tmp_path):
        ds = _make_ds(sources_setup, tmp_path, patch_size=(16, 16))
        s0 = ds[0]
        s_mod = ds[len(ds)]
        assert s0["path"] == s_mod["path"]

    def test_windowed_path_is_string(self, sources_setup, tmp_path):
        ds = _make_ds(sources_setup, tmp_path, patch_size=(16, 16))
        assert isinstance(ds[0]["path"], str)


# ---------------------------------------------------------------------------
# Windows CSV — image coordinates
# ---------------------------------------------------------------------------


class TestSoftLabelCachedDatasetWindowsCsvImage:
    """windows_csv in image pixel coordinates (row_off, col_off, patch_h, patch_w)."""

    def _wcsv(self, path, rows):
        pd.DataFrame(rows).to_csv(path, index=False)

    def test_image_coords_len(self, sources_setup, tmp_path):
        wcsv = tmp_path / "windows.csv"
        self._wcsv(
            wcsv,
            [
                {
                    "tile_id": "tile_0",
                    "row_off": 0,
                    "col_off": 0,
                    "patch_h": 16,
                    "patch_w": 16,
                },
                {
                    "tile_id": "tile_0",
                    "row_off": 16,
                    "col_off": 16,
                    "patch_h": 16,
                    "patch_w": 16,
                },
                {
                    "tile_id": "tile_1",
                    "row_off": 0,
                    "col_off": 0,
                    "patch_h": 16,
                    "patch_w": 16,
                },
            ],
        )
        ds = _make_ds(
            sources_setup, tmp_path, windows_csv=str(wcsv), windows_coord_type="image"
        )
        assert len(ds) == 3

    def test_image_coords_output_shapes(self, sources_setup, tmp_path):
        wcsv = tmp_path / "windows.csv"
        self._wcsv(
            wcsv,
            [
                {
                    "tile_id": "tile_0",
                    "row_off": 0,
                    "col_off": 0,
                    "patch_h": 16,
                    "patch_w": 16,
                }
            ],
        )
        ds = _make_ds(
            sources_setup, tmp_path, windows_csv=str(wcsv), windows_coord_type="image"
        )
        s = ds[0]
        assert s["image"].shape == (3, 16, 16)
        assert s["mask"]["mask"].shape == (NUM_CLASSES, 16, 16)
        assert s["mask"]["w_conf"].shape == (1, 16, 16)

    def test_image_coords_p_soft_sums_to_one(self, sources_setup, tmp_path):
        wcsv = tmp_path / "windows.csv"
        self._wcsv(
            wcsv,
            [
                {
                    "tile_id": "tile_0",
                    "row_off": 0,
                    "col_off": 0,
                    "patch_h": 16,
                    "patch_w": 16,
                }
            ],
        )
        ds = _make_ds(
            sources_setup, tmp_path, windows_csv=str(wcsv), windows_coord_type="image"
        )
        p_soft = ds[0]["mask"]["mask"]
        assert torch.allclose(p_soft.sum(dim=0), torch.ones(16, 16), atol=1e-5)

    def test_image_coords_unknown_tile_id_skipped(self, sources_setup, tmp_path):
        wcsv = tmp_path / "windows.csv"
        self._wcsv(
            wcsv,
            [
                {
                    "tile_id": "tile_0",
                    "row_off": 0,
                    "col_off": 0,
                    "patch_h": 16,
                    "patch_w": 16,
                },
                {
                    "tile_id": "no_such_tile",
                    "row_off": 0,
                    "col_off": 0,
                    "patch_h": 16,
                    "patch_w": 16,
                },
            ],
        )
        ds = _make_ds(
            sources_setup, tmp_path, windows_csv=str(wcsv), windows_coord_type="image"
        )
        assert len(ds) == 1

    def test_windows_csv_overrides_patch_size(self, sources_setup, tmp_path):
        """windows_csv wins over patch_size when both are given."""
        wcsv = tmp_path / "windows.csv"
        self._wcsv(
            wcsv,
            [
                {
                    "tile_id": "tile_0",
                    "row_off": 0,
                    "col_off": 0,
                    "patch_h": 8,
                    "patch_w": 8,
                }
            ],
        )
        ds = _make_ds(
            sources_setup,
            tmp_path,
            windows_csv=str(wcsv),
            windows_coord_type="image",
            patch_size=(16, 16),
        )
        assert len(ds) == 1
        assert ds[0]["image"].shape == (3, 8, 8)

    def test_image_coords_path_is_string(self, sources_setup, tmp_path):
        wcsv = tmp_path / "windows.csv"
        self._wcsv(
            wcsv,
            [
                {
                    "tile_id": "tile_0",
                    "row_off": 0,
                    "col_off": 0,
                    "patch_h": 16,
                    "patch_w": 16,
                }
            ],
        )
        ds = _make_ds(
            sources_setup, tmp_path, windows_csv=str(wcsv), windows_coord_type="image"
        )
        assert isinstance(ds[0]["path"], str)


# ---------------------------------------------------------------------------
# Windows CSV — world coordinates
# ---------------------------------------------------------------------------


class TestSoftLabelCachedDatasetWindowsCsvWorld:
    """windows_csv in geographic world coordinates (minx, miny, maxx, maxy [, crs])."""

    # tile fixture: BOUNDS=(0,0,1,1) EPSG:4326, 32×32 px
    # top-left 16×16 pixels ↔ world bbox (0.0, 0.5, 0.5, 1.0) in EPSG:4326

    def test_world_same_crs_output_shape(self, sources_setup, tmp_path):
        wcsv = tmp_path / "windows.csv"
        pd.DataFrame(
            [
                {
                    "tile_id": "tile_0",
                    "minx": 0.0,
                    "miny": 0.5,
                    "maxx": 0.5,
                    "maxy": 1.0,
                    "crs": "EPSG:4326",
                }
            ]
        ).to_csv(wcsv, index=False)
        ds = _make_ds(
            sources_setup, tmp_path, windows_csv=str(wcsv), windows_coord_type="world"
        )
        s = ds[0]
        assert s["image"].shape == (3, 16, 16)
        assert s["mask"]["mask"].shape == (NUM_CLASSES, 16, 16)
        assert s["mask"]["w_conf"].shape == (1, 16, 16)

    def test_world_no_crs_column_uses_tile_crs(self, sources_setup, tmp_path):
        """No crs column → world coords treated as tile's CRS (no reprojection)."""
        wcsv = tmp_path / "windows.csv"
        pd.DataFrame(
            [{"tile_id": "tile_0", "minx": 0.0, "miny": 0.5, "maxx": 0.5, "maxy": 1.0}]
        ).to_csv(wcsv, index=False)
        ds = _make_ds(
            sources_setup, tmp_path, windows_csv=str(wcsv), windows_coord_type="world"
        )
        assert ds[0]["image"].shape == (3, 16, 16)

    def test_world_crs_reprojection(self, sources_setup, tmp_path):
        """Same bbox in EPSG:3857 reprojected to tile EPSG:4326 → same 16×16 region."""
        import rasterio.warp

        minx, miny, maxx, maxy = rasterio.warp.transform_bounds(
            "EPSG:4326", "EPSG:3857", 0.0, 0.5, 0.5, 1.0
        )
        wcsv = tmp_path / "windows_3857.csv"
        pd.DataFrame(
            [
                {
                    "tile_id": "tile_0",
                    "minx": minx,
                    "miny": miny,
                    "maxx": maxx,
                    "maxy": maxy,
                    "crs": "EPSG:3857",
                }
            ]
        ).to_csv(wcsv, index=False)
        ds = _make_ds(
            sources_setup, tmp_path, windows_csv=str(wcsv), windows_coord_type="world"
        )
        ph, pw = ds[0]["image"].shape[1], ds[0]["image"].shape[2]
        # reprojection introduces sub-pixel error; allow ±1 pixel
        assert abs(ph - 16) <= 1
        assert abs(pw - 16) <= 1

    def test_world_multiple_windows_len(self, sources_setup, tmp_path):
        wcsv = tmp_path / "windows.csv"
        pd.DataFrame(
            [
                {
                    "tile_id": "tile_0",
                    "minx": 0.0,
                    "miny": 0.5,
                    "maxx": 0.5,
                    "maxy": 1.0,
                    "crs": "EPSG:4326",
                },
                {
                    "tile_id": "tile_1",
                    "minx": 0.0,
                    "miny": 0.5,
                    "maxx": 0.5,
                    "maxy": 1.0,
                    "crs": "EPSG:4326",
                },
            ]
        ).to_csv(wcsv, index=False)
        ds = _make_ds(
            sources_setup, tmp_path, windows_csv=str(wcsv), windows_coord_type="world"
        )
        assert len(ds) == 2
