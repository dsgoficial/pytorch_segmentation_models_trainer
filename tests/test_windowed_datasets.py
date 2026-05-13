# -*- coding: utf-8 -*-
import numpy as np
import pytest
import torch
import rasterio
from pathlib import Path
from rasterio.transform import from_bounds
from pytorch_segmentation_models_trainer.dataset_loader.image_dataset import (
    WindowedImageDataset,
    WindowedImageAutoencoderDataset,
)
from tests.utils import BasicTestCase


def _write_tif(path: Path, width: int, height: int, bands: int = 3, dtype="uint8"):
    path.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed=42)
    if dtype == "uint8":
        data = rng.integers(0, 255, (bands, height, width), dtype=np.uint8)
    else:
        data = rng.random((bands, height, width)).astype(np.float32)

    transform = from_bounds(0, 0, 1, 1, width, height)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=height,
        width=width,
        count=bands,
        dtype=dtype,
        crs="EPSG:4326",
        transform=transform,
    ) as dst:
        dst.write(data)


class TestWindowedDatasets(BasicTestCase):
    def setUp(self):
        super().setUp()
        self.tmp = Path(self.make_temp_dir())
        self.img_dir = self.tmp / "images"

        # Create two images:
        # img1: 100x100 -> With patch 50x50 and stride 50, should have 2x2 = 4 patches
        # img2: 150x100 -> With patch 50x50 and stride 50, should have 3x2 = 6 patches
        # Total patches: 10
        _write_tif(self.img_dir / "img1.tif", width=100, height=100)
        _write_tif(self.img_dir / "img2.tif", width=150, height=100)

    def test_windowed_image_dataset_indexing(self):
        ds = WindowedImageDataset(image_dir=self.img_dir, crop_size=[50, 50], stride=50)

        # 100/50 * 100/50 = 4
        # 150/50 * 100/50 = 6
        assert len(ds) == 10

        # Test first patch of first image
        sample = ds[0]
        assert sample["image"].shape == (3, 50, 50)
        assert "path" in sample
        assert "img1.tif" in sample["path"]

        # Test first patch of second image (index 4)
        sample = ds[4]
        assert "img2.tif" in sample["path"]
        assert sample["image"].shape == (3, 50, 50)

        # Test last patch
        sample = ds[9]
        assert "img2.tif" in sample["path"]

    def test_windowed_image_autoencoder_dataset(self):
        ds = WindowedImageAutoencoderDataset(
            image_dir=self.img_dir, crop_size=[50, 50], stride=50
        )

        assert len(ds) == 10
        sample = ds[0]
        assert "image" in sample
        assert "target" in sample
        assert "path" in sample

        # Without corruption, image and target should be identical
        assert torch.allclose(sample["image"], sample["target"])

    def test_windowed_image_autoencoder_corruption(self):
        # Using a simple corruption (Blur) to verify it only affects "image"
        corruption = [
            {"_target_": "albumentations.Blur", "blur_limit": [11, 11], "p": 1.0}
        ]

        ds = WindowedImageAutoencoderDataset(
            image_dir=self.img_dir,
            crop_size=[50, 50],
            stride=50,
            corruption_augmentation_list=corruption,
        )

        sample = ds[0]
        # Blur should make image different from target
        assert not torch.allclose(sample["image"], sample["target"])
        # Target should remain sharp (relatively)

    def test_windowed_image_dataset_small_images(self):
        # Image smaller than 100x100
        _write_tif(self.img_dir / "too_small.tif", width=50, height=50)

        ds = WindowedImageDataset(
            image_dir=self.img_dir, crop_size=[100, 100], stride=100
        )

        # img1 (100x100) -> 1 patch
        # img2 (150x100) -> 1 patch (if stride is 100, (150-100)//100 + 1 = 1)
        # too_small (50x50) -> 0 patches
        assert len(ds) == 2
        for i in range(len(ds)):
            assert "too_small.tif" not in ds[i]["path"]
