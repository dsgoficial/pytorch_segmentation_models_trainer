# -*- coding: utf-8 -*-
"""Tests for LulcInputDataset and LulcInputWindowedDataset."""

import os
import tempfile

import numpy as np
import pandas as pd
import pytest
import rasterio
import torch
from rasterio.transform import from_bounds

from pytorch_segmentation_models_trainer.dataset_loader.lulc_input_dataset import (
    LulcInputDataset,
    LulcInputWindowedDataset,
    _one_hot_chw,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

H, W, C, N_LULC = 32, 32, 6, 3  # tile size, num_classes, num_lulc_sources


def _write_geotiff(path: str, data: np.ndarray, dtype=None):
    """Write a (bands, H, W) or (H, W) array to a minimal GeoTIFF."""
    if data.ndim == 2:
        data = data[np.newaxis]
    bands, h, w = data.shape
    dtype = dtype or data.dtype
    transform = from_bounds(0, 0, 1, 1, w, h)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=h,
        width=w,
        count=bands,
        dtype=dtype,
        crs="EPSG:4326",
        transform=transform,
    ) as dst:
        dst.write(data.astype(dtype))


@pytest.fixture()
def tile_dir(tmp_path):
    """Create one set of tile GeoTIFFs and return paths dict."""
    img = np.random.randint(0, 256, (3, H, W), dtype=np.uint8)
    mask = np.random.randint(0, C, (H, W), dtype=np.uint8)
    lulc = [np.random.randint(0, C, (H, W), dtype=np.uint8) for _ in range(N_LULC)]

    paths = {
        "image": str(tmp_path / "img.tif"),
        "mask": str(tmp_path / "mask.tif"),
        **{f"lulc_{i}": str(tmp_path / f"lulc_{i}.tif") for i in range(N_LULC)},
    }
    _write_geotiff(paths["image"], img, dtype="uint8")
    _write_geotiff(paths["mask"], mask, dtype="uint8")
    for i in range(N_LULC):
        _write_geotiff(paths[f"lulc_{i}"], lulc[i], dtype="uint8")

    return paths


@pytest.fixture()
def tile_csv(tile_dir, tmp_path):
    csv_path = str(tmp_path / "tiles.csv")
    row = {
        "image_path": tile_dir["image"],
        "mask_path": tile_dir["mask"],
        **{f"lulc_{i}_path": tile_dir[f"lulc_{i}"] for i in range(N_LULC)},
    }
    pd.DataFrame([row, row]).to_csv(csv_path, index=False)
    return csv_path


@pytest.fixture()
def lulc_keys():
    return [f"lulc_{i}_path" for i in range(N_LULC)]


@pytest.fixture()
def scene_dir(tmp_path):
    """Create full-scene GeoTIFFs (2×H × 2×W) and windowed CSV."""
    sh, sw = H * 2, W * 2
    img = np.random.randint(0, 256, (3, sh, sw), dtype=np.uint8)
    mask = np.random.randint(0, C, (sh, sw), dtype=np.uint8)
    lulc = [np.random.randint(0, C, (sh, sw), dtype=np.uint8) for _ in range(N_LULC)]

    paths = {
        "image": str(tmp_path / "scene_img.tif"),
        "mask": str(tmp_path / "scene_mask.tif"),
        **{f"lulc_{i}": str(tmp_path / f"scene_lulc_{i}.tif") for i in range(N_LULC)},
    }
    _write_geotiff(paths["image"], img, dtype="uint8")
    _write_geotiff(paths["mask"], mask, dtype="uint8")
    for i in range(N_LULC):
        _write_geotiff(paths[f"lulc_{i}"], lulc[i], dtype="uint8")

    return paths


@pytest.fixture()
def windowed_csv(scene_dir, tmp_path):
    csv_path = str(tmp_path / "windowed.csv")
    rows = [
        {
            "image_path": scene_dir["image"],
            "mask_path": scene_dir["mask"],
            **{f"lulc_{i}_path": scene_dir[f"lulc_{i}"] for i in range(N_LULC)},
            "row_off": 0,
            "col_off": 0,
            "patch_size": H,
        },
        {
            "image_path": scene_dir["image"],
            "mask_path": scene_dir["mask"],
            **{f"lulc_{i}_path": scene_dir[f"lulc_{i}"] for i in range(N_LULC)},
            "row_off": H,
            "col_off": W,
            "patch_size": H,
        },
    ]
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    return csv_path


# ---------------------------------------------------------------------------
# _one_hot_chw unit tests
# ---------------------------------------------------------------------------


class TestOneHotChw:
    def test_basic_shape(self):
        arr = np.array([[0, 1], [2, 3]])
        out = _one_hot_chw(arr, num_classes=4)
        assert out.shape == (4, 2, 2)

    def test_values_correct(self):
        arr = np.array([[0, 1], [2, 3]])
        out = _one_hot_chw(arr, num_classes=4)
        assert out[0, 0, 0] == 1.0  # class 0 at (0,0)
        assert out[1, 0, 1] == 1.0  # class 1 at (0,1)
        assert out[2, 1, 0] == 1.0  # class 2 at (1,0)
        assert out[3, 1, 1] == 1.0  # class 3 at (1,1)

    def test_ignore_index_produces_all_zeros(self):
        arr = np.array([[255, 0], [0, 255]])
        out = _one_hot_chw(arr, num_classes=4, ignore_index=255)
        assert out[:, 0, 0].sum() == 0.0
        assert out[:, 1, 1].sum() == 0.0
        assert out[0, 0, 1] == 1.0  # valid pixel

    def test_out_of_range_produces_all_zeros(self):
        arr = np.array([[10, 1]])  # 10 >= num_classes=4
        out = _one_hot_chw(arr, num_classes=4)
        assert out[:, 0, 0].sum() == 0.0

    def test_float32_dtype(self):
        arr = np.zeros((4, 4), dtype=np.uint8)
        out = _one_hot_chw(arr, num_classes=3)
        assert out.dtype == np.float32

    def test_sums_to_one_for_valid_pixels(self):
        arr = np.random.randint(0, 6, (8, 8), dtype=np.uint8)
        out = _one_hot_chw(arr, num_classes=6)
        assert np.allclose(out.sum(axis=0), 1.0)


# ---------------------------------------------------------------------------
# LulcInputDataset — tile-based
# ---------------------------------------------------------------------------


class TestLulcInputDataset:
    def test_output_shape(self, tile_csv, lulc_keys):
        ds = LulcInputDataset(
            input_csv_path=tile_csv,
            lulc_keys=lulc_keys,
            num_classes=C,
        )
        sample = ds[0]
        expected_channels = 3 + N_LULC * C
        assert sample["image"].shape == (expected_channels, H, W)
        assert sample["mask"].shape == (H, W)

    def test_image_dtype_float32(self, tile_csv, lulc_keys):
        ds = LulcInputDataset(
            input_csv_path=tile_csv, lulc_keys=lulc_keys, num_classes=C
        )
        assert ds[0]["image"].dtype == torch.float32

    def test_mask_dtype_int64(self, tile_csv, lulc_keys):
        ds = LulcInputDataset(
            input_csv_path=tile_csv, lulc_keys=lulc_keys, num_classes=C
        )
        assert ds[0]["mask"].dtype == torch.int64

    def test_lulc_channels_binary(self, tile_csv, lulc_keys):
        ds = LulcInputDataset(
            input_csv_path=tile_csv, lulc_keys=lulc_keys, num_classes=C
        )
        lulc_part = ds[0]["image"][3:]  # drop RGB
        unique = torch.unique(lulc_part)
        assert set(unique.tolist()).issubset({0.0, 1.0})

    def test_lulc_onehot_sums_to_one_per_source(self, tile_csv, lulc_keys):
        ds = LulcInputDataset(
            input_csv_path=tile_csv, lulc_keys=lulc_keys, num_classes=C
        )
        img = ds[0]["image"]
        for i in range(N_LULC):
            start = 3 + i * C
            one_hot_block = img[start : start + C]  # (C, H, W)
            sums = one_hot_block.sum(dim=0)  # (H, W)
            assert torch.all(sums == 1.0)

    def test_path_returned(self, tile_csv, lulc_keys):
        ds = LulcInputDataset(
            input_csv_path=tile_csv, lulc_keys=lulc_keys, num_classes=C
        )
        assert isinstance(ds[0]["path"], str)

    def test_no_lulc_keys_returns_rgb_only(self, tile_csv):
        ds = LulcInputDataset(input_csv_path=tile_csv, lulc_keys=[], num_classes=C)
        assert ds[0]["image"].shape == (3, H, W)

    def test_missing_column_raises(self, tile_csv):
        with pytest.raises(ValueError, match="columns not found"):
            LulcInputDataset(
                input_csv_path=tile_csv,
                lulc_keys=["nonexistent_col"],
                num_classes=C,
            )

    def test_len(self, tile_csv, lulc_keys):
        ds = LulcInputDataset(
            input_csv_path=tile_csv, lulc_keys=lulc_keys, num_classes=C
        )
        assert len(ds) == 2

    def test_augmentation_applied(self, tile_csv, lulc_keys):
        aug = [
            {"_target_": "albumentations.HorizontalFlip", "p": 1.0},
        ]
        ds = LulcInputDataset(
            input_csv_path=tile_csv,
            lulc_keys=lulc_keys,
            num_classes=C,
            augmentation_list=aug,
        )
        # Just ensure it doesn't crash and returns correct shapes
        sample = ds[0]
        assert sample["image"].shape == (3 + N_LULC * C, H, W)

    def test_dynamic_number_of_sources(self, tile_dir, tmp_path):
        """Adding a fourth LULC source increases output channels by num_classes."""
        extra_path = str(tmp_path / "extra_lulc.tif")
        _write_geotiff(extra_path, np.zeros((H, W), dtype=np.uint8))
        csv_path = str(tmp_path / "extra.csv")
        row = {
            "image_path": tile_dir["image"],
            "mask_path": tile_dir["mask"],
            **{f"lulc_{i}_path": tile_dir[f"lulc_{i}"] for i in range(N_LULC)},
            "extra_lulc_path": extra_path,
        }
        pd.DataFrame([row]).to_csv(csv_path, index=False)

        keys_4 = [f"lulc_{i}_path" for i in range(N_LULC)] + ["extra_lulc_path"]
        ds = LulcInputDataset(input_csv_path=csv_path, lulc_keys=keys_4, num_classes=C)
        assert ds[0]["image"].shape == (3 + 4 * C, H, W)


# ---------------------------------------------------------------------------
# LulcInputWindowedDataset
# ---------------------------------------------------------------------------


class TestLulcInputWindowedDataset:
    def test_output_shape(self, windowed_csv, lulc_keys):
        ds = LulcInputWindowedDataset(
            input_csv_path=windowed_csv,
            lulc_keys=lulc_keys,
            num_classes=C,
        )
        sample = ds[0]
        expected_channels = 3 + N_LULC * C
        assert sample["image"].shape == (expected_channels, H, W)
        assert sample["mask"].shape == (H, W)

    def test_different_windows_produce_different_patches(self, windowed_csv, lulc_keys):
        ds = LulcInputWindowedDataset(
            input_csv_path=windowed_csv,
            lulc_keys=lulc_keys,
            num_classes=C,
        )
        s0 = ds[0]["image"]
        s1 = ds[1]["image"]
        # Two patches from different offsets should not be identical
        assert not torch.equal(s0, s1)

    def test_missing_window_column_raises(self, windowed_csv, lulc_keys, tmp_path):
        # Remove patch_size column
        df = pd.read_csv(windowed_csv).drop(columns=["patch_size"])
        bad_csv = str(tmp_path / "no_patch_size.csv")
        df.to_csv(bad_csv, index=False)
        with pytest.raises(ValueError, match="patch_size"):
            LulcInputWindowedDataset(
                input_csv_path=bad_csv,
                lulc_keys=lulc_keys,
                num_classes=C,
            )

    def test_lulc_channels_binary(self, windowed_csv, lulc_keys):
        ds = LulcInputWindowedDataset(
            input_csv_path=windowed_csv, lulc_keys=lulc_keys, num_classes=C
        )
        lulc_part = ds[0]["image"][3:]
        unique = torch.unique(lulc_part)
        assert set(unique.tolist()).issubset({0.0, 1.0})

    def test_mask_dtype_int64(self, windowed_csv, lulc_keys):
        ds = LulcInputWindowedDataset(
            input_csv_path=windowed_csv, lulc_keys=lulc_keys, num_classes=C
        )
        assert ds[0]["mask"].dtype == torch.int64

    def test_load_image_np_without_window(self, windowed_csv, lulc_keys):
        """_load_image_np with window=None computes the window internally."""
        ds = LulcInputWindowedDataset(
            input_csv_path=windowed_csv, lulc_keys=lulc_keys, num_classes=C
        )
        arr = ds._load_image_np(0)
        assert arr.shape == (H, W, 3)

    def test_load_mask_np_without_window(self, windowed_csv, lulc_keys):
        """_load_mask_np with window=None computes the window internally."""
        ds = LulcInputWindowedDataset(
            input_csv_path=windowed_csv, lulc_keys=lulc_keys, num_classes=C
        )
        arr = ds._load_mask_np(0)
        assert arr.shape == (H, W)

    def test_load_lulc_np_without_window(self, windowed_csv, lulc_keys):
        """_load_lulc_np with window=None computes the window internally."""
        ds = LulcInputWindowedDataset(
            input_csv_path=windowed_csv, lulc_keys=lulc_keys, num_classes=C
        )
        arr = ds._load_lulc_np(0, key=lulc_keys[0])
        assert arr.shape == (H, W)


# ---------------------------------------------------------------------------
# Static conversion helpers
# ---------------------------------------------------------------------------


class TestConversionHelpers:
    def test_image_to_chw_float_from_chw_tensor(self):
        t = torch.randint(0, 256, (3, H, W), dtype=torch.uint8)
        out = LulcInputDataset._image_to_chw_float(t)
        assert out.shape == (3, H, W)
        assert out.dtype == torch.float32

    def test_image_to_chw_float_from_hwc_tensor(self):
        """HWC tensor (not CHW) should be permuted to CHW."""
        t = torch.randint(0, 256, (H, W, 3), dtype=torch.uint8)
        out = LulcInputDataset._image_to_chw_float(t)
        assert out.shape == (3, H, W)
        assert out.dtype == torch.float32

    def test_mask_to_hw_long_from_tensor(self):
        t = torch.zeros(H, W, dtype=torch.uint8)
        out = LulcInputDataset._mask_to_hw_long(t)
        assert out.dtype == torch.int64
        assert out.shape == (H, W)

    def test_lulc_to_hw_uint8_from_tensor(self):
        t = torch.zeros(H, W, dtype=torch.int32)
        out = LulcInputDataset._lulc_to_hw_uint8(t)
        assert out.dtype == np.uint8
        assert out.shape == (H, W)
