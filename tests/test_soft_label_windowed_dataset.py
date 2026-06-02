# -*- coding: utf-8 -*-
"""Tests for SoftLabelWindowedDataset."""

import numpy as np
import pandas as pd
import pytest
import rasterio
import torch
from rasterio.transform import from_bounds

from pytorch_segmentation_models_trainer.dataset_loader.soft_label_windowed_dataset import (
    SoftLabelWindowedDataset,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Full raster dimensions
FULL_H, FULL_W = 64, 64
C = 4  # number of classes
PATCH = 32  # patch size


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _write_tif(path, data: np.ndarray, dtype=None):
    if data.ndim == 2:
        data = data[np.newaxis]
    c, h, w = data.shape
    dtype = dtype or data.dtype.name
    transform = from_bounds(0, 0, 1, 1, w, h)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=h,
        width=w,
        count=c,
        dtype=dtype,
        crs="EPSG:4326",
        transform=transform,
    ) as dst:
        dst.write(data)


@pytest.fixture()
def full_rasters(tmp_path):
    """Full-scene rasters: image (RGB), p_soft (C bands), w_conf (1 band)."""
    rng = np.random.default_rng(0)

    img = rng.integers(0, 255, (3, FULL_H, FULL_W), dtype=np.uint8)
    img_path = tmp_path / "image.tif"
    _write_tif(img_path, img, dtype="uint8")

    raw = rng.random((C, FULL_H, FULL_W)).astype(np.float32)
    raw /= raw.sum(axis=0, keepdims=True)
    ps_path = tmp_path / "p_soft.tif"
    _write_tif(ps_path, raw, dtype="float32")

    wc = rng.random((1, FULL_H, FULL_W)).astype(np.float32)
    wc_path = tmp_path / "w_conf.tif"
    _write_tif(wc_path, wc, dtype="float32")

    # Two non-overlapping patches
    df = pd.DataFrame(
        [
            {
                "image_path": str(img_path),
                "p_soft_path": str(ps_path),
                "w_conf_path": str(wc_path),
                "row_off": 0,
                "col_off": 0,
                "patch_size": PATCH,
            },
            {
                "image_path": str(img_path),
                "p_soft_path": str(ps_path),
                "w_conf_path": str(wc_path),
                "row_off": PATCH,
                "col_off": PATCH,
                "patch_size": PATCH,
            },
        ]
    )
    return df, img, raw, wc


@pytest.fixture()
def dataset(full_rasters):
    df, *_ = full_rasters
    return SoftLabelWindowedDataset(df=df)


@pytest.fixture()
def dataset_no_wconf(full_rasters, tmp_path):
    df, *_ = full_rasters
    return SoftLabelWindowedDataset(df=df.drop(columns=["w_conf_path"]))


# ---------------------------------------------------------------------------
# Shape / type contract
# ---------------------------------------------------------------------------


class TestSoftLabelWindowedDatasetShapes:
    def test_len(self, dataset):
        assert len(dataset) == 2

    def test_image_shape(self, dataset):
        assert dataset[0]["image"].shape == (3, PATCH, PATCH)

    def test_image_dtype_float32(self, dataset):
        assert dataset[0]["image"].dtype == torch.float32

    def test_mask_is_dict(self, dataset):
        assert isinstance(dataset[0]["mask"], dict)

    def test_p_soft_shape(self, dataset):
        assert dataset[0]["mask"]["mask"].shape == (C, PATCH, PATCH)

    def test_p_soft_dtype_float32(self, dataset):
        assert dataset[0]["mask"]["mask"].dtype == torch.float32

    def test_w_conf_shape(self, dataset):
        assert dataset[0]["mask"]["w_conf"].shape == (1, PATCH, PATCH)

    def test_w_conf_dtype_float32(self, dataset):
        assert dataset[0]["mask"]["w_conf"].dtype == torch.float32

    def test_path_key_present(self, dataset):
        assert isinstance(dataset[0]["path"], str)


# ---------------------------------------------------------------------------
# Probability invariants
# ---------------------------------------------------------------------------


class TestSoftLabelWindowedDatasetProbabilities:
    def test_p_soft_sums_to_one(self, dataset):
        p = dataset[0]["mask"]["mask"]
        assert torch.allclose(p.sum(dim=0), torch.ones(PATCH, PATCH), atol=1e-5)

    def test_p_soft_non_negative(self, dataset):
        assert (dataset[0]["mask"]["mask"] >= 0).all()

    def test_w_conf_in_unit_interval(self, dataset):
        w = dataset[0]["mask"]["w_conf"]
        assert (w >= 0).all() and (w <= 1).all()

    def test_image_in_unit_interval(self, dataset):
        img = dataset[0]["image"]
        assert img.min().item() >= 0.0
        assert img.max().item() <= 1.0 + 1e-5


# ---------------------------------------------------------------------------
# Windowing correctness — pixel values match the full raster at the right offsets
# ---------------------------------------------------------------------------


class TestSoftLabelWindowedDatasetWindowing:
    def test_patch0_image_pixels_match_full_raster(self, full_rasters):
        df, img_chw, p_soft_chw, wc_chw = full_rasters
        ds = SoftLabelWindowedDataset(df=df)
        patch_img = ds[0]["image"]  # (3, PATCH, PATCH), float [0,1]

        expected = torch.from_numpy(
            img_chw[:, 0:PATCH, 0:PATCH].astype(np.float32) / 255.0
        )
        assert torch.allclose(patch_img, expected, atol=1e-5)

    def test_patch1_uses_correct_offset(self, full_rasters):
        df, img_chw, p_soft_chw, wc_chw = full_rasters
        ds = SoftLabelWindowedDataset(df=df)
        patch_img = ds[1]["image"]  # row_off=PATCH, col_off=PATCH

        expected = torch.from_numpy(
            img_chw[:, PATCH : 2 * PATCH, PATCH : 2 * PATCH].astype(np.float32) / 255.0
        )
        assert torch.allclose(patch_img, expected, atol=1e-5)

    def test_patch0_p_soft_matches_full_raster(self, full_rasters):
        df, img_chw, p_soft_chw, wc_chw = full_rasters
        ds = SoftLabelWindowedDataset(df=df)
        patch_p = ds[0]["mask"]["mask"]

        expected = torch.from_numpy(p_soft_chw[:, 0:PATCH, 0:PATCH])
        assert torch.allclose(patch_p, expected, atol=1e-5)

    def test_patch0_w_conf_matches_full_raster(self, full_rasters):
        df, img_chw, p_soft_chw, wc_chw = full_rasters
        ds = SoftLabelWindowedDataset(df=df)
        patch_w = ds[0]["mask"]["w_conf"]

        expected = torch.from_numpy(wc_chw[:, 0:PATCH, 0:PATCH])
        assert torch.allclose(patch_w, expected, atol=1e-5)


# ---------------------------------------------------------------------------
# Without w_conf column
# ---------------------------------------------------------------------------


class TestSoftLabelWindowedDatasetNoWConf:
    def test_w_conf_absent(self, dataset_no_wconf):
        assert "w_conf" not in dataset_no_wconf[0]["mask"]

    def test_mask_key_present(self, dataset_no_wconf):
        assert "mask" in dataset_no_wconf[0]["mask"]

    def test_p_soft_shape(self, dataset_no_wconf):
        assert dataset_no_wconf[0]["mask"]["mask"].shape == (C, PATCH, PATCH)


# ---------------------------------------------------------------------------
# CSV validation
# ---------------------------------------------------------------------------


class TestSoftLabelWindowedDatasetValidation:
    def test_missing_row_off_raises(self, full_rasters):
        df, *_ = full_rasters
        with pytest.raises(ValueError, match="row_off"):
            SoftLabelWindowedDataset(df=df.drop(columns=["row_off"]))

    def test_missing_col_off_raises(self, full_rasters):
        df, *_ = full_rasters
        with pytest.raises(ValueError, match="col_off"):
            SoftLabelWindowedDataset(df=df.drop(columns=["col_off"]))

    def test_missing_patch_size_raises(self, full_rasters):
        df, *_ = full_rasters
        with pytest.raises(ValueError, match="patch_size"):
            SoftLabelWindowedDataset(df=df.drop(columns=["patch_size"]))

    def test_requires_csv_or_df(self):
        with pytest.raises(ValueError):
            SoftLabelWindowedDataset()


# ---------------------------------------------------------------------------
# Custom CSV column keys
# ---------------------------------------------------------------------------


class TestSoftLabelWindowedDatasetCustomKeys:
    def test_custom_offset_keys(self, full_rasters, tmp_path):
        df, *_ = full_rasters
        df2 = df.rename(columns={"row_off": "r", "col_off": "c", "patch_size": "sz"})
        ds = SoftLabelWindowedDataset(
            df=df2, row_off_key="r", col_off_key="c", patch_size_key="sz"
        )
        assert ds[0]["image"].shape == (3, PATCH, PATCH)


# ---------------------------------------------------------------------------
# Augmentation consistency
# ---------------------------------------------------------------------------


class TestSoftLabelWindowedDatasetAugmentation:
    def test_horizontal_flip_applied_consistently(self, full_rasters):
        df, *_ = full_rasters
        aug = [{"_target_": "albumentations.HorizontalFlip", "p": 1.0}]
        ds_aug = SoftLabelWindowedDataset(df=df, augmentation_list=aug)
        ds_base = SoftLabelWindowedDataset(df=df)

        img_aug = ds_aug[0]["image"]
        img_base = ds_base[0]["image"]
        p_aug = ds_aug[0]["mask"]["mask"]
        p_base = ds_base[0]["mask"]["mask"]

        assert torch.allclose(img_aug, img_base.flip(-1), atol=1e-5)
        assert torch.allclose(p_aug, p_base.flip(-1), atol=1e-5)

    def test_w_conf_flipped_consistently(self, full_rasters):
        df, *_ = full_rasters
        aug = [{"_target_": "albumentations.HorizontalFlip", "p": 1.0}]
        ds_aug = SoftLabelWindowedDataset(df=df, augmentation_list=aug)
        ds_base = SoftLabelWindowedDataset(df=df)

        w_aug = ds_aug[0]["mask"]["w_conf"]
        w_base = ds_base[0]["mask"]["w_conf"]
        assert torch.allclose(w_aug, w_base.flip(-1), atol=1e-5)
