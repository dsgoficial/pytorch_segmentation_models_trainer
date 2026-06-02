# -*- coding: utf-8 -*-
"""Tests for SoftLabelDataset."""

import numpy as np
import pandas as pd
import pytest
import rasterio
import torch
from rasterio.transform import from_bounds

from pytorch_segmentation_models_trainer.dataset_loader.soft_label_dataset import (
    SoftLabelDataset,
)

H, W, C = 32, 32, 4
N = 3  # number of tiles in the test CSV


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _write_tif(path, data: np.ndarray, dtype=None):
    """Write a (C, H, W) numpy array to a GeoTIFF."""
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
def tile_dir(tmp_path):
    """Create N test tiles: image (RGB uint8), p_soft (C float32), w_conf (1 float32)."""
    rng = np.random.default_rng(42)
    rows = []
    for i in range(N):
        img_path = tmp_path / f"img_{i}.tif"
        ps_path = tmp_path / f"p_soft_{i}.tif"
        wc_path = tmp_path / f"w_conf_{i}.tif"

        # Planet RGB uint8
        _write_tif(
            img_path, rng.integers(0, 255, (3, H, W), dtype=np.uint8), dtype="uint8"
        )

        # P_soft float32 (C channels, valid probability distribution)
        raw = rng.random((C, H, W)).astype(np.float32)
        raw /= raw.sum(axis=0, keepdims=True)
        _write_tif(ps_path, raw, dtype="float32")

        # W_conf float32 (1 channel, [0,1])
        _write_tif(wc_path, rng.random((1, H, W)).astype(np.float32), dtype="float32")

        rows.append(
            {
                "image_path": str(img_path),
                "p_soft_path": str(ps_path),
                "w_conf_path": str(wc_path),
            }
        )

    df = pd.DataFrame(rows)
    csv_path = tmp_path / "tiles.csv"
    df.to_csv(csv_path, index=False)
    return tmp_path, csv_path, df


@pytest.fixture()
def dataset(tile_dir):
    _, csv_path, _ = tile_dir
    return SoftLabelDataset(input_csv_path=str(csv_path))


@pytest.fixture()
def dataset_no_wconf(tile_dir):
    """Dataset without w_conf_path column."""
    _, _, df = tile_dir
    df_no_wc = df.drop(columns=["w_conf_path"])
    return SoftLabelDataset(df=df_no_wc)


# ---------------------------------------------------------------------------
# Basic shape / type contract
# ---------------------------------------------------------------------------


class TestSoftLabelDatasetShapes:
    def test_len(self, dataset):
        assert len(dataset) == N

    def test_image_is_float_tensor(self, dataset):
        item = dataset[0]
        assert isinstance(item["image"], torch.Tensor)
        assert item["image"].dtype == torch.float32

    def test_image_shape_chw(self, dataset):
        item = dataset[0]
        assert item["image"].shape == (3, H, W)

    def test_mask_is_dict(self, dataset):
        item = dataset[0]
        assert isinstance(item["mask"], dict)

    def test_mask_dict_has_mask_key(self, dataset):
        item = dataset[0]
        assert "mask" in item["mask"]

    def test_mask_dict_has_w_conf_key(self, dataset):
        item = dataset[0]
        assert "w_conf" in item["mask"]

    def test_p_soft_shape(self, dataset):
        item = dataset[0]
        assert item["mask"]["mask"].shape == (C, H, W)

    def test_p_soft_dtype_float32(self, dataset):
        item = dataset[0]
        assert item["mask"]["mask"].dtype == torch.float32

    def test_w_conf_shape(self, dataset):
        item = dataset[0]
        assert item["mask"]["w_conf"].shape == (1, H, W)

    def test_w_conf_dtype_float32(self, dataset):
        item = dataset[0]
        assert item["mask"]["w_conf"].dtype == torch.float32

    def test_path_key_present(self, dataset):
        item = dataset[0]
        assert "path" in item
        assert isinstance(item["path"], str)

    def test_all_indices_accessible(self, dataset):
        for i in range(N):
            item = dataset[i]
            assert "image" in item
            assert "mask" in item


# ---------------------------------------------------------------------------
# Probability invariants
# ---------------------------------------------------------------------------


class TestSoftLabelDatasetProbabilities:
    def test_p_soft_sums_to_one_per_pixel(self, dataset):
        p = dataset[0]["mask"]["mask"]  # (C, H, W)
        sums = p.sum(dim=0)  # (H, W)
        assert torch.allclose(sums, torch.ones(H, W), atol=1e-5)

    def test_p_soft_non_negative(self, dataset):
        p = dataset[0]["mask"]["mask"]
        assert (p >= 0).all()

    def test_w_conf_in_unit_interval(self, dataset):
        w = dataset[0]["mask"]["w_conf"]
        assert (w >= 0).all()
        assert (w <= 1).all()

    def test_image_in_unit_interval_after_normalization(self, dataset):
        img = dataset[0]["image"]
        assert img.min().item() >= 0.0
        assert img.max().item() <= 1.0 + 1e-5


# ---------------------------------------------------------------------------
# Without w_conf column
# ---------------------------------------------------------------------------


class TestSoftLabelDatasetNoWConf:
    def test_mask_dict_no_w_conf_key(self, dataset_no_wconf):
        item = dataset_no_wconf[0]
        assert "w_conf" not in item["mask"]

    def test_mask_dict_has_mask_key(self, dataset_no_wconf):
        item = dataset_no_wconf[0]
        assert "mask" in item["mask"]

    def test_p_soft_shape(self, dataset_no_wconf):
        item = dataset_no_wconf[0]
        assert item["mask"]["mask"].shape == (C, H, W)


# ---------------------------------------------------------------------------
# From in-memory DataFrame
# ---------------------------------------------------------------------------


class TestSoftLabelDatasetFromDF:
    def test_from_df(self, tile_dir):
        _, _, df = tile_dir
        ds = SoftLabelDataset(df=df)
        assert len(ds) == N
        item = ds[0]
        assert item["image"].shape == (3, H, W)

    def test_requires_csv_or_df(self):
        with pytest.raises(ValueError):
            SoftLabelDataset()


# ---------------------------------------------------------------------------
# Augmentation consistency
# ---------------------------------------------------------------------------


class TestSoftLabelDatasetAugmentation:
    def test_geometric_aug_applied_consistently(self, tile_dir):
        """HorizontalFlip applied to image, p_soft and w_conf identically."""
        _, _, df = tile_dir
        aug = [
            {"_target_": "albumentations.HorizontalFlip", "p": 1.0},
        ]
        ds_aug = SoftLabelDataset(df=df, augmentation_list=aug)
        ds_base = SoftLabelDataset(df=df)

        img_aug = ds_aug[0]["image"]
        img_base = ds_base[0]["image"]

        p_aug = ds_aug[0]["mask"]["mask"]
        p_base = ds_base[0]["mask"]["mask"]

        # Flipped image should equal base flipped along W axis
        assert torch.allclose(img_aug, img_base.flip(-1), atol=1e-5)
        assert torch.allclose(p_aug, p_base.flip(-1), atol=1e-5)

    def test_w_conf_augmented_consistently(self, tile_dir):
        _, _, df = tile_dir
        aug = [{"_target_": "albumentations.HorizontalFlip", "p": 1.0}]
        ds_aug = SoftLabelDataset(df=df, augmentation_list=aug)
        ds_base = SoftLabelDataset(df=df)

        w_aug = ds_aug[0]["mask"]["w_conf"]
        w_base = ds_base[0]["mask"]["w_conf"]
        assert torch.allclose(w_aug, w_base.flip(-1), atol=1e-5)


# ---------------------------------------------------------------------------
# Static helper coverage: _to_chw_float and _image_to_chw_float
# ---------------------------------------------------------------------------


class TestSoftLabelDatasetStaticHelpers:
    def test_to_chw_float_3d_tensor_hwc(self):
        """3D torch.Tensor (H, W, C) is permuted to (C, H, W)."""
        x = torch.randint(0, 255, (H, W, C)).float()
        out = SoftLabelDataset._to_chw_float(x)
        assert out.shape == (C, H, W)
        assert out.dtype == torch.float32

    def test_to_chw_float_2d_tensor(self):
        """2D torch.Tensor (H, W) gets a channel dim → (1, H, W)."""
        x = torch.rand(H, W)
        out = SoftLabelDataset._to_chw_float(x)
        assert out.shape == (1, H, W)
        assert out.dtype == torch.float32

    def test_to_chw_float_2d_numpy(self):
        """2D numpy array (H, W) gets a channel dim → (1, H, W)."""
        x = np.random.rand(H, W).astype(np.float32)
        out = SoftLabelDataset._to_chw_float(x)
        assert out.shape == (1, H, W)
        assert out.dtype == torch.float32

    def test_image_to_chw_float_tensor_uint8_range(self):
        """torch.Tensor with values > 1 (uint8 range) is divided by 255."""
        x = torch.randint(2, 255, (3, H, W)).float()
        out = SoftLabelDataset._image_to_chw_float(x)
        assert out.dtype == torch.float32
        assert out.max().item() <= 1.0 + 1e-3

    def test_image_to_chw_float_tensor_already_normalized(self):
        """torch.Tensor with values in [0, 1] is returned unchanged."""
        x = torch.rand(3, H, W)
        out = SoftLabelDataset._image_to_chw_float(x)
        assert out.dtype == torch.float32
        assert out.max().item() <= 1.0 + 1e-3
