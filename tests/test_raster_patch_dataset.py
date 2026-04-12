# -*- coding: utf-8 -*-
"""
/***************************************************************************
 pytorch_segmentation_models_trainer
                              -------------------
        begin                : 2021-02-25
        git sha              : $Format:%H$
        copyright            : (C) 2021 by Philipe Borba - Cartographic Engineer
                                                            @ Brazilian Army
        email                : philipeborba at gmail dot com
 ***************************************************************************/
/***************************************************************************
 *                                                                         *
 *   This program is free software; you can redistribute it and/or modify  *
 *   it under the terms of the GNU General Public License as published by  *
 *   the Free Software Foundation; either version 2 of the License, or     *
 *   (at your option) any later version.                                   *
 *                                                                         *
 ****
"""

import os
import warnings
from pathlib import Path

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from pytorch_segmentation_models_trainer.dataset_loader.raster_patch_dataset import (
    RasterPatchDataset,
)
from tests.utils import BasicTestCase

try:
    import rasterio
    from rasterio.transform import from_bounds

    HAS_RASTERIO = True
except ImportError:
    HAS_RASTERIO = False

pytestmark = pytest.mark.skipif(not HAS_RASTERIO, reason="rasterio not installed")


# ── Fixtures ──────────────────────────────────────────────────────────────────


def _write_tif(path: Path, width: int, height: int, bands: int = 3, dtype="uint8"):
    """Create a synthetic single-band or multi-band GeoTIFF for testing."""
    path.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed=42)
    if dtype == "uint8":
        data = rng.integers(0, 255, (bands, height, width), dtype=np.uint8)
    elif dtype == "uint16":
        data = rng.integers(0, 65535, (bands, height, width), dtype=np.uint16)
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


def _write_mask(path: Path, width: int, height: int):
    """Create a single-band binary mask GeoTIFF for testing."""
    _write_tif(path, width=width, height=height, bands=1, dtype="uint8")


class TestRasterPatchDataset(BasicTestCase):
    """Unit tests for RasterPatchDataset."""

    # ── Setup ─────────────────────────────────────────────────────────────────

    def setUp(self):
        super().setUp()
        self.tmp = Path(self.make_temp_dir())
        self.img_dir = self.tmp / "images"
        self.mask_dir = self.tmp / "masks"

        # Two images: 200×200 (3 bands) each
        _write_tif(self.img_dir / "img_a.tif", width=200, height=200, bands=3)
        _write_mask(self.mask_dir / "img_a.tif", width=200, height=200)

        _write_tif(self.img_dir / "img_b.tif", width=200, height=200, bands=3)
        _write_mask(self.mask_dir / "img_b.tif", width=200, height=200)

    # ── Instantiation ─────────────────────────────────────────────────────────

    def test_init_valid(self):
        """Dataset is created and __len__ is positive."""
        ds = RasterPatchDataset(
            image_dir=self.img_dir,
            mask_dir=self.mask_dir,
            extension=".tif",
            patch_size=64,
            stride=32,
        )
        self.assertGreater(len(ds), 0)

    def test_image_info_length_matches_valid_pairs(self):
        """image_info contains one entry per valid image/mask pair."""
        ds = RasterPatchDataset(
            image_dir=self.img_dir,
            mask_dir=self.mask_dir,
            extension=".tif",
            patch_size=64,
            stride=64,
        )
        self.assertEqual(len(ds.image_info), 2)

    def test_invalid_image_dtype_raises(self):
        """Passing an unknown image_dtype must raise ValueError."""
        with self.assertRaises(ValueError):
            RasterPatchDataset(
                image_dir=self.img_dir,
                mask_dir=self.mask_dir,
                extension=".tif",
                patch_size=64,
                stride=32,
                image_dtype="int32",
            )

    def test_invalid_selected_bands_raises(self):
        """selected_bands with non-positive integer must raise ValueError."""
        with self.assertRaises(ValueError):
            RasterPatchDataset(
                image_dir=self.img_dir,
                mask_dir=self.mask_dir,
                extension=".tif",
                patch_size=64,
                stride=32,
                selected_bands=[0],  # base-1: must be >= 1
            )

    def test_no_valid_pairs_raises(self):
        """Empty or fully unmatched directories must raise ValueError."""
        empty_dir = self.tmp / "empty"
        empty_dir.mkdir()
        with self.assertRaises(ValueError):
            RasterPatchDataset(
                image_dir=empty_dir,
                mask_dir=self.mask_dir,
                extension=".tif",
                patch_size=64,
                stride=32,
            )

    # ── Patch count arithmetic ────────────────────────────────────────────────

    def test_total_patches_formula(self):
        """Total patches matches ((W-k)//s+1) * ((H-k)//s+1) summed over images."""
        patch_size, stride = 64, 32
        ds = RasterPatchDataset(
            image_dir=self.img_dir,
            mask_dir=self.mask_dir,
            extension=".tif",
            patch_size=patch_size,
            stride=stride,
        )
        # Each 200×200 image: ((200-64)//32+1)^2 = (4+1)^2 = 25 patches
        patches_per_image = ((200 - patch_size) // stride + 1) ** 2
        expected = patches_per_image * 2  # two images
        self.assertEqual(len(ds), expected)

    def test_stride_equal_to_patch_no_overlap(self):
        """stride == patch_size → no overlap, minimal patch count."""
        k = 64
        ds = RasterPatchDataset(
            image_dir=self.img_dir,
            mask_dir=self.mask_dir,
            extension=".tif",
            patch_size=k,
            stride=k,
        )
        # 200 // 64 = 3 → 3×3 = 9 patches per image
        patches_per_image = (200 // k) ** 2
        self.assertEqual(len(ds), patches_per_image * 2)

    def test_patch_larger_than_image_ignored_with_warning(self):
        """Images smaller than patch_size are ignored with a warning."""
        # Write a tiny image that is too small for the patch
        _write_tif(self.img_dir / "tiny.tif", width=32, height=32, bands=3)
        _write_mask(self.mask_dir / "tiny.tif", width=32, height=32)

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            ds = RasterPatchDataset(
                image_dir=self.img_dir,
                mask_dir=self.mask_dir,
                extension=".tif",
                patch_size=64,
                stride=32,
            )
        # tiny.tif must be excluded
        self.assertEqual(len(ds.image_info), 2)
        warning_messages = [str(w.message) for w in caught]
        self.assertTrue(any("tiny.tif" in m for m in warning_messages))

    # ── __getitem__ ───────────────────────────────────────────────────────────

    def test_getitem_returns_dict_with_image_and_mask(self):
        """__getitem__ returns a dict with 'image' and 'mask' keys."""
        ds = RasterPatchDataset(
            image_dir=self.img_dir,
            mask_dir=self.mask_dir,
            extension=".tif",
            patch_size=64,
            stride=32,
        )
        item = ds[0]
        self.assertIn("image", item)
        self.assertIn("mask", item)

    def test_getitem_tensor_types(self):
        """'image' is float32 tensor and 'mask' is int64 tensor (no transform)."""
        ds = RasterPatchDataset(
            image_dir=self.img_dir,
            mask_dir=self.mask_dir,
            extension=".tif",
            patch_size=64,
            stride=32,
        )
        item = ds[0]
        self.assertEqual(item["image"].dtype, torch.float32)
        self.assertEqual(item["mask"].dtype, torch.int64)

    def test_getitem_patch_shape(self):
        """Patch tensors have the expected spatial dimensions."""
        k = 64
        ds = RasterPatchDataset(
            image_dir=self.img_dir,
            mask_dir=self.mask_dir,
            extension=".tif",
            patch_size=k,
            stride=32,
        )
        item = ds[0]
        # image: (C, k, k); mask: (k, k)
        C, H, W = item["image"].shape
        self.assertEqual(H, k)
        self.assertEqual(W, k)
        self.assertEqual(item["mask"].shape, (k, k))

    def test_getitem_image_normalized_uint8(self):
        """uint8 images without transform are normalized to [0, 1]."""
        ds = RasterPatchDataset(
            image_dir=self.img_dir,
            mask_dir=self.mask_dir,
            extension=".tif",
            patch_size=64,
            stride=32,
            image_dtype="uint8",
        )
        item = ds[0]
        self.assertLessEqual(item["image"].max().item(), 1.0)
        self.assertGreaterEqual(item["image"].min().item(), 0.0)

    def test_getitem_index_out_of_bounds(self):
        """IndexError is raised for out-of-range indices."""
        ds = RasterPatchDataset(
            image_dir=self.img_dir,
            mask_dir=self.mask_dir,
            extension=".tif",
            patch_size=64,
            stride=32,
        )
        with self.assertRaises(IndexError):
            _ = ds[len(ds)]
        with self.assertRaises(IndexError):
            _ = ds[-1]

    # ── Bisect mapping correctness ────────────────────────────────────────────

    def test_bisect_mapping_first_patch_is_image_zero(self):
        """Global index 0 must resolve to the first image."""
        ds = RasterPatchDataset(
            image_dir=self.img_dir,
            mask_dir=self.mask_dir,
            extension=".tif",
            patch_size=64,
            stride=64,
        )
        import bisect

        img_idx = bisect.bisect_right(ds._cumulative, 0) - 1
        self.assertEqual(img_idx, 0)

    def test_bisect_mapping_boundary_is_second_image(self):
        """The index at the cumulative boundary must resolve to the second image."""
        ds = RasterPatchDataset(
            image_dir=self.img_dir,
            mask_dir=self.mask_dir,
            extension=".tif",
            patch_size=64,
            stride=64,
        )
        import bisect

        # ds._cumulative[1] is the first global index of the second image
        boundary_idx = ds._cumulative[1]
        img_idx = bisect.bisect_right(ds._cumulative, boundary_idx) - 1
        self.assertEqual(img_idx, 1)

    # ── selected_bands ────────────────────────────────────────────────────────

    def test_selected_bands_reduces_channels(self):
        """Selecting one band produces a single-channel image tensor."""
        ds = RasterPatchDataset(
            image_dir=self.img_dir,
            mask_dir=self.mask_dir,
            extension=".tif",
            patch_size=64,
            stride=32,
            selected_bands=[1],
        )
        item = ds[0]
        self.assertEqual(item["image"].shape[0], 1)

    # ── Missing mask ──────────────────────────────────────────────────────────

    def test_missing_mask_emits_warning_and_skips(self):
        """Images whose mask file is absent produce a warning and are skipped."""
        _write_tif(self.img_dir / "orphan.tif", width=200, height=200, bands=3)
        # No corresponding mask created

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            ds = RasterPatchDataset(
                image_dir=self.img_dir,
                mask_dir=self.mask_dir,
                extension=".tif",
                patch_size=64,
                stride=32,
            )
        self.assertEqual(len(ds.image_info), 2)  # orphan excluded
        warning_messages = [str(w.message) for w in caught]
        self.assertTrue(any("orphan.tif" in m for m in warning_messages))

    # ── Extension normalisation ───────────────────────────────────────────────

    def test_extension_without_dot_accepted(self):
        """Extension 'tif' (no leading dot) works identically to '.tif'."""
        ds_dot = RasterPatchDataset(
            image_dir=self.img_dir,
            mask_dir=self.mask_dir,
            extension=".tif",
            patch_size=64,
            stride=32,
        )
        ds_no_dot = RasterPatchDataset(
            image_dir=self.img_dir,
            mask_dir=self.mask_dir,
            extension="tif",
            patch_size=64,
            stride=32,
        )
        self.assertEqual(len(ds_dot), len(ds_no_dot))

    # ── DataLoader integration ────────────────────────────────────────────────

    def test_dataloader_batch_shape(self):
        """DataLoader delivers correctly shaped batches without errors."""
        ds = RasterPatchDataset(
            image_dir=self.img_dir,
            mask_dir=self.mask_dir,
            extension=".tif",
            patch_size=64,
            stride=32,
        )
        loader = DataLoader(ds, batch_size=4, shuffle=False, num_workers=0)
        batch = next(iter(loader))
        imgs = batch["image"]
        masks = batch["mask"]
        self.assertEqual(imgs.shape, (4, 3, 64, 64))
        self.assertEqual(masks.shape, (4, 64, 64))

    # ── Hydra instantiation ───────────────────────────────────────────────────

    def test_hydra_instantiation(self):
        """Dataset can be instantiated via hydra.utils.instantiate from YAML config."""
        import hydra
        from hydra import compose, initialize

        with initialize(config_path="./test_configs"):
            cfg = compose(
                config_name="raster_patch_dataset.yaml",
                overrides=[
                    f"image_dir={self.img_dir}",
                    f"mask_dir={self.mask_dir}",
                ],
            )
            ds = hydra.utils.instantiate(cfg, _recursive_=False)
        self.assertIsInstance(ds, RasterPatchDataset)
        self.assertGreater(len(ds), 0)

    # ── repr ──────────────────────────────────────────────────────────────────

    def test_repr(self):
        """__repr__ includes key attributes."""
        ds = RasterPatchDataset(
            image_dir=self.img_dir,
            mask_dir=self.mask_dir,
            extension=".tif",
            patch_size=64,
            stride=32,
        )
        r = repr(ds)
        self.assertIn("RasterPatchDataset", r)
        self.assertIn("64", r)
        self.assertIn("32", r)

    # ── uint16 dtype ──────────────────────────────────────────────────────────

    def test_uint16_dtype_normalization(self):
        """uint16 images without transform are normalized to [0, 1]."""
        img_dir16 = self.tmp / "images16"
        mask_dir16 = self.tmp / "masks16"
        _write_tif(img_dir16 / "img.tif", width=200, height=200, bands=3, dtype="uint16")
        _write_mask(mask_dir16 / "img.tif", width=200, height=200)

        ds = RasterPatchDataset(
            image_dir=img_dir16,
            mask_dir=mask_dir16,
            extension=".tif",
            patch_size=64,
            stride=32,
            image_dtype="uint16",
        )
        item = ds[0]
        self.assertLessEqual(item["image"].max().item(), 1.0)
        self.assertGreaterEqual(item["image"].min().item(), 0.0)

    # ── Subdirectory matching ─────────────────────────────────────────────────

    def test_recursive_subdirectory_matching(self):
        """Images in subdirectories are matched to masks in the same relative path."""
        sub_img = self.img_dir / "subdir"
        sub_mask = self.mask_dir / "subdir"
        _write_tif(sub_img / "sub_img.tif", width=200, height=200, bands=3)
        _write_mask(sub_mask / "sub_img.tif", width=200, height=200)

        ds = RasterPatchDataset(
            image_dir=self.img_dir,
            mask_dir=self.mask_dir,
            extension=".tif",
            patch_size=64,
            stride=64,
        )
        self.assertEqual(len(ds.image_info), 3)  # 2 root + 1 subdir
