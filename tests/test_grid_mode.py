# -*- coding: utf-8 -*-
"""
Tests for grid_mode in RandomCropSegmentationDataset.

Covers:
  - Deterministic grid position computation with various overlaps
  - Pre-filtering of invalid tiles (min_valid_ratio)
  - Flush-edge tile generation
  - CutMix with grid mode (second crop is random)
  - Backward compatibility (grid_mode=False)
  - Multi-source labels with grid mode
  - Soft labels with grid mode
  - Class-balanced sampling integration
"""

import csv
import os
import shutil
import tempfile
import unittest
import warnings

import numpy as np

# IMPORTANT: import pytorch_segmentation_models_trainer BEFORE torch
# to resolve fbgemm.dll vs GDAL DLL conflict on Windows.
import pytorch_segmentation_models_trainer  # noqa: F401

import rasterio
from rasterio.transform import from_bounds

from pytorch_segmentation_models_trainer.dataset_loader.dataset import (
    RandomCropSegmentationDataset,
)


def _create_geotiff(path, data, count=None, dtype="uint8"):
    """Write a numpy array as a GeoTIFF."""
    if data.ndim == 2:
        h, w = data.shape
        bands = 1
        data = data[np.newaxis, ...]
    else:
        bands, h, w = data.shape

    if count is not None:
        bands = count

    transform = from_bounds(0, 0, w, h, w, h)
    with rasterio.open(
        path, "w", driver="GTiff", height=h, width=w,
        count=bands, dtype=dtype, transform=transform,
    ) as dst:
        for b in range(bands):
            dst.write(data[b], b + 1)


class GridModeTestMixin:
    """Creates a temporary directory with synthetic GeoTIFF images/masks."""

    N_CLASSES = 4
    IMG_W = 100
    IMG_H = 80
    N_IMAGES = 2
    CROP_SIZE = [32, 32]

    def _create_synthetic_dataset(self):
        self.tmpdir = tempfile.mkdtemp(prefix="test_grid_")
        img_dir = os.path.join(self.tmpdir, "images")
        msk_dir = os.path.join(self.tmpdir, "masks")
        os.makedirs(img_dir)
        os.makedirs(msk_dir)

        self.image_paths = []
        self.mask_paths = []
        h, w = self.IMG_H, self.IMG_W
        half_h, half_w = h // 2, w // 2

        for i in range(self.N_IMAGES):
            img = np.full((3, h, w), fill_value=100 + i * 20, dtype=np.uint8)

            mask = np.zeros((h, w), dtype=np.uint8)
            mask[:half_h, :half_w] = 0
            mask[:half_h, half_w:] = 1
            mask[half_h:, :half_w] = 2
            mask[half_h:, half_w:] = 3

            img_path = os.path.join(img_dir, f"img_{i:02d}.tif")
            msk_path = os.path.join(msk_dir, f"msk_{i:02d}.tif")
            _create_geotiff(img_path, img)
            _create_geotiff(msk_path, mask)
            self.image_paths.append(img_path)
            self.mask_paths.append(msk_path)

        self.csv_path = os.path.join(self.tmpdir, "dataset.csv")
        with open(self.csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["image", "mask"])
            for img_p, msk_p in zip(self.image_paths, self.mask_paths):
                writer.writerow([img_p, msk_p])

    def _cleanup(self):
        if hasattr(self, "tmpdir") and os.path.exists(self.tmpdir):
            shutil.rmtree(self.tmpdir)


class TestGridPositionCount(unittest.TestCase, GridModeTestMixin):
    """Verify that __len__ returns the expected number of grid tiles."""

    def setUp(self):
        warnings.simplefilter("ignore")
        self._create_synthetic_dataset()

    def tearDown(self):
        self._cleanup()

    def test_overlap_50_percent(self):
        """50% overlap: stride = 16 for crop 32x32."""
        ds = RandomCropSegmentationDataset(
            input_csv_path=self.csv_path,
            crop_size=self.CROP_SIZE,
            n_classes=self.N_CLASSES,
            grid_mode=True,
            overlap_x=0.5,
            overlap_y=0.5,
            min_valid_ratio=0.0,
        )
        crop_h, crop_w = self.CROP_SIZE
        stride = 16  # crop * (1 - 0.5)
        # Columns: range(0, 100-32+1, 16) = [0,16,32,48,64] → 5, last=64 < 68 → +68 = 6
        # Rows: range(0, 80-32+1, 16) = [0,16,32,48] → 4, last=48 == 48 → no flush = 4
        # Per image: 6 * 4 = 24, total: 24 * 2 = 48
        cols = list(range(0, self.IMG_W - crop_w + 1, stride))
        if cols[-1] < self.IMG_W - crop_w:
            cols.append(self.IMG_W - crop_w)
        rows = list(range(0, self.IMG_H - crop_h + 1, stride))
        if rows[-1] < self.IMG_H - crop_h:
            rows.append(self.IMG_H - crop_h)
        expected = len(cols) * len(rows) * self.N_IMAGES
        self.assertEqual(len(ds), expected)

    def test_overlap_zero(self):
        """0% overlap: stride = crop_size (non-overlapping tiles)."""
        ds = RandomCropSegmentationDataset(
            input_csv_path=self.csv_path,
            crop_size=self.CROP_SIZE,
            n_classes=self.N_CLASSES,
            grid_mode=True,
            overlap_x=0.0,
            overlap_y=0.0,
            min_valid_ratio=0.0,
        )
        crop_h, crop_w = self.CROP_SIZE
        stride_x, stride_y = crop_w, crop_h
        cols = list(range(0, self.IMG_W - crop_w + 1, stride_x))
        if cols[-1] < self.IMG_W - crop_w:
            cols.append(self.IMG_W - crop_w)
        rows = list(range(0, self.IMG_H - crop_h + 1, stride_y))
        if rows[-1] < self.IMG_H - crop_h:
            rows.append(self.IMG_H - crop_h)
        expected = len(cols) * len(rows) * self.N_IMAGES
        self.assertEqual(len(ds), expected)

    def test_overlap_75_percent(self):
        """75% overlap: stride = 8 for crop 32."""
        ds = RandomCropSegmentationDataset(
            input_csv_path=self.csv_path,
            crop_size=self.CROP_SIZE,
            n_classes=self.N_CLASSES,
            grid_mode=True,
            overlap_x=0.75,
            overlap_y=0.75,
            min_valid_ratio=0.0,
        )
        crop_h, crop_w = self.CROP_SIZE
        stride = 8
        cols = list(range(0, self.IMG_W - crop_w + 1, stride))
        if cols[-1] < self.IMG_W - crop_w:
            cols.append(self.IMG_W - crop_w)
        rows = list(range(0, self.IMG_H - crop_h + 1, stride))
        if rows[-1] < self.IMG_H - crop_h:
            rows.append(self.IMG_H - crop_h)
        expected = len(cols) * len(rows) * self.N_IMAGES
        self.assertEqual(len(ds), expected)


class TestGridDeterminism(unittest.TestCase, GridModeTestMixin):
    """Same index always returns the same crop."""

    def setUp(self):
        warnings.simplefilter("ignore")
        self._create_synthetic_dataset()

    def tearDown(self):
        self._cleanup()

    def test_same_idx_same_crop(self):
        ds = RandomCropSegmentationDataset(
            input_csv_path=self.csv_path,
            crop_size=self.CROP_SIZE,
            n_classes=self.N_CLASSES,
            grid_mode=True,
            overlap_x=0.5,
            overlap_y=0.5,
            min_valid_ratio=0.0,
        )
        sample1 = ds[5]
        sample2 = ds[5]
        np.testing.assert_array_equal(
            sample1["image"] if isinstance(sample1["image"], np.ndarray)
            else sample1["image"].numpy(),
            sample2["image"] if isinstance(sample2["image"], np.ndarray)
            else sample2["image"].numpy(),
        )
        np.testing.assert_array_equal(
            sample1["mask"] if isinstance(sample1["mask"], np.ndarray)
            else sample1["mask"].numpy(),
            sample2["mask"] if isinstance(sample2["mask"], np.ndarray)
            else sample2["mask"].numpy(),
        )


class TestGridInvalidTileFiltering(unittest.TestCase):
    """Tiles with insufficient valid pixels are excluded."""

    def setUp(self):
        warnings.simplefilter("ignore")
        self.tmpdir = tempfile.mkdtemp(prefix="test_grid_filter_")
        img_dir = os.path.join(self.tmpdir, "images")
        msk_dir = os.path.join(self.tmpdir, "masks")
        os.makedirs(img_dir)
        os.makedirs(msk_dir)

        # Image 64x64, mask has right half as ignore (255)
        h = w = 64
        img = np.full((3, h, w), 128, dtype=np.uint8)
        mask = np.zeros((h, w), dtype=np.uint8)
        mask[:, w // 2:] = 255  # right half is ignore

        img_path = os.path.join(img_dir, "img.tif")
        msk_path = os.path.join(msk_dir, "msk.tif")
        _create_geotiff(img_path, img)
        _create_geotiff(msk_path, mask)

        self.csv_path = os.path.join(self.tmpdir, "dataset.csv")
        with open(self.csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["image", "mask"])
            writer.writerow([img_path, msk_path])

    def tearDown(self):
        shutil.rmtree(self.tmpdir)

    def test_invalid_tiles_filtered(self):
        """With min_valid_ratio=0.8, tiles in the ignore region are excluded."""
        ds = RandomCropSegmentationDataset(
            input_csv_path=self.csv_path,
            crop_size=[32, 32],
            n_classes=2,
            grid_mode=True,
            overlap_x=0.0,
            overlap_y=0.0,
            min_valid_ratio=0.8,
        )
        # Only tiles fully in the left half (cols 0) should survive.
        # With crop=32, stride=32, image=64: cols=[0, 32], rows=[0, 32]
        # col=0: fully valid (class 0). col=32: fully ignore (255) → rejected.
        # Expected: 2 tiles (col=0, row=0 and col=0, row=32)
        self.assertEqual(len(ds), 2)

    def test_no_filtering_with_low_ratio(self):
        """With min_valid_ratio=0.0, all tiles pass."""
        ds = RandomCropSegmentationDataset(
            input_csv_path=self.csv_path,
            crop_size=[32, 32],
            n_classes=2,
            grid_mode=True,
            overlap_x=0.0,
            overlap_y=0.0,
            min_valid_ratio=0.0,
        )
        # 4 tiles: (0,0), (0,32), (32,0), (32,32)
        self.assertEqual(len(ds), 4)


class TestGridFlushEdges(unittest.TestCase, GridModeTestMixin):
    """Last row/column tiles are clamped to image edge."""

    def setUp(self):
        warnings.simplefilter("ignore")
        self._create_synthetic_dataset()

    def tearDown(self):
        self._cleanup()

    def test_last_tile_reaches_edge(self):
        ds = RandomCropSegmentationDataset(
            input_csv_path=self.csv_path,
            crop_size=self.CROP_SIZE,
            n_classes=self.N_CLASSES,
            grid_mode=True,
            overlap_x=0.5,
            overlap_y=0.5,
            min_valid_ratio=0.0,
        )
        crop_h, crop_w = self.CROP_SIZE
        # Check that the last grid position reaches the image edge
        positions = ds._grid_positions
        # Filter positions for image 0
        img0_positions = positions[positions[:, 0] == 0]
        max_col = img0_positions[:, 1].max()
        max_row = img0_positions[:, 2].max()
        self.assertEqual(max_col + crop_w, self.IMG_W)
        self.assertEqual(max_row + crop_h, self.IMG_H)


class TestGridWithCutMix(unittest.TestCase, GridModeTestMixin):
    """CutMix still works: main crop from grid, second crop random."""

    def setUp(self):
        warnings.simplefilter("ignore")
        self._create_synthetic_dataset()

    def tearDown(self):
        self._cleanup()

    def test_cutmix_with_grid(self):
        ds = RandomCropSegmentationDataset(
            input_csv_path=self.csv_path,
            crop_size=self.CROP_SIZE,
            n_classes=self.N_CLASSES,
            grid_mode=True,
            overlap_x=0.5,
            overlap_y=0.5,
            min_valid_ratio=0.0,
            cutmix_prob=1.0,
            cutmix_alpha=1.0,
        )
        # Should not raise — CutMix uses _get_random_crop for the second crop
        sample = ds[0]
        self.assertIn("image", sample)
        self.assertIn("mask", sample)

    def test_classmix_with_grid(self):
        ds = RandomCropSegmentationDataset(
            input_csv_path=self.csv_path,
            crop_size=self.CROP_SIZE,
            n_classes=self.N_CLASSES,
            grid_mode=True,
            overlap_x=0.5,
            overlap_y=0.5,
            min_valid_ratio=0.0,
            cutmix_prob=1.0,
            classmix=True,
        )
        sample = ds[0]
        self.assertIn("image", sample)
        self.assertIn("mask", sample)


class TestGridBackwardCompat(unittest.TestCase, GridModeTestMixin):
    """grid_mode=False preserves original random behavior."""

    def setUp(self):
        warnings.simplefilter("ignore")
        self._create_synthetic_dataset()

    def tearDown(self):
        self._cleanup()

    def test_random_mode_unchanged(self):
        ds = RandomCropSegmentationDataset(
            input_csv_path=self.csv_path,
            crop_size=self.CROP_SIZE,
            n_classes=self.N_CLASSES,
            grid_mode=False,
            samples_per_epoch=50,
        )
        self.assertEqual(len(ds), 50)
        self.assertFalse(hasattr(ds, '_grid_positions') and ds._grid_positions is not None
                         and len(ds._grid_positions) > 0)
        sample = ds[0]
        self.assertIn("image", sample)
        self.assertIn("mask", sample)


class TestGridValidation(unittest.TestCase, GridModeTestMixin):
    """Parameter validation for grid mode."""

    def setUp(self):
        warnings.simplefilter("ignore")
        self._create_synthetic_dataset()

    def tearDown(self):
        self._cleanup()

    def test_invalid_overlap_raises(self):
        with self.assertRaises(ValueError):
            RandomCropSegmentationDataset(
                input_csv_path=self.csv_path,
                crop_size=self.CROP_SIZE,
                n_classes=self.N_CLASSES,
                grid_mode=True,
                overlap_x=1.0,
            )
        with self.assertRaises(ValueError):
            RandomCropSegmentationDataset(
                input_csv_path=self.csv_path,
                crop_size=self.CROP_SIZE,
                n_classes=self.N_CLASSES,
                grid_mode=True,
                overlap_y=-0.1,
            )


class TestGridWithClassBalancing(unittest.TestCase, GridModeTestMixin):
    """Grid mode with class_balanced_sampling builds class indices in single pass."""

    def setUp(self):
        warnings.simplefilter("ignore")
        self._create_synthetic_dataset()

    def tearDown(self):
        self._cleanup()

    def test_class_positions_built(self):
        ds = RandomCropSegmentationDataset(
            input_csv_path=self.csv_path,
            crop_size=self.CROP_SIZE,
            n_classes=self.N_CLASSES,
            grid_mode=True,
            overlap_x=0.5,
            overlap_y=0.5,
            min_valid_ratio=0.0,
            class_balanced_sampling=True,
        )
        # Class positions should be populated from single-pass
        self.assertTrue(len(ds._class_positions) > 0)
        self.assertEqual(len(ds._class_sampling_weights), self.N_CLASSES)
        # All 4 classes should be present somewhere
        for c in range(self.N_CLASSES):
            self.assertIn(c, ds._class_positions)

    def test_cutmix_rare_with_grid(self):
        ds = RandomCropSegmentationDataset(
            input_csv_path=self.csv_path,
            crop_size=self.CROP_SIZE,
            n_classes=self.N_CLASSES,
            grid_mode=True,
            overlap_x=0.5,
            overlap_y=0.5,
            min_valid_ratio=0.0,
            cutmix_prob=1.0,
            cutmix_rare_classes=[3],
            cutmix_rare_prob=1.0,
        )
        # Class-to-tiles index should be built
        self.assertTrue(len(ds._class_to_tiles) > 0)
        sample = ds[0]
        self.assertIn("image", sample)


class TestGridSoftLabels(unittest.TestCase):
    """Grid mode with soft_labels (multi-band float32 probability masks)."""

    N_CLASSES = 3
    CROP_SIZE = [32, 32]

    def setUp(self):
        warnings.simplefilter("ignore")
        self.tmpdir = tempfile.mkdtemp(prefix="test_grid_soft_")
        img_dir = os.path.join(self.tmpdir, "images")
        msk_dir = os.path.join(self.tmpdir, "masks")
        os.makedirs(img_dir)
        os.makedirs(msk_dir)

        h = w = 64
        self.image_paths = []
        self.mask_paths = []

        for i in range(2):
            img = np.full((3, h, w), 100 + i * 30, dtype=np.uint8)

            # Soft mask: N_CLASSES bands, float32, probabilities
            soft_mask = np.zeros((self.N_CLASSES, h, w), dtype=np.float32)
            soft_mask[0, :h // 2, :] = 1.0  # class 0 top half
            soft_mask[1, h // 2:, :] = 1.0  # class 1 bottom half
            # Leave some pixels with all-zero (ignore)
            soft_mask[:, -4:, -4:] = 0.0  # bottom-right corner is ignore

            img_path = os.path.join(img_dir, f"img_{i:02d}.tif")
            msk_path = os.path.join(msk_dir, f"msk_{i:02d}.tif")
            _create_geotiff(img_path, img)
            _create_geotiff(msk_path, soft_mask, dtype="float32")
            self.image_paths.append(img_path)
            self.mask_paths.append(msk_path)

        self.csv_path = os.path.join(self.tmpdir, "dataset.csv")
        with open(self.csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["image", "mask"])
            for img_p, msk_p in zip(self.image_paths, self.mask_paths):
                writer.writerow([img_p, msk_p])

    def tearDown(self):
        shutil.rmtree(self.tmpdir)

    def test_grid_with_soft_labels(self):
        ds = RandomCropSegmentationDataset(
            input_csv_path=self.csv_path,
            crop_size=self.CROP_SIZE,
            n_classes=self.N_CLASSES,
            grid_mode=True,
            overlap_x=0.0,
            overlap_y=0.0,
            min_valid_ratio=0.0,
            soft_labels=True,
        )
        self.assertGreater(len(ds), 0)
        sample = ds[0]
        mask = sample["mask"]
        if hasattr(mask, 'numpy'):
            mask = mask.numpy()
        # Soft mask should be (C, H, W) float
        self.assertEqual(mask.ndim, 3)
        self.assertEqual(mask.shape[0], self.N_CLASSES)


class TestGridBinaryClassMask(unittest.TestCase, GridModeTestMixin):
    """Grid mode with n_classes=2 binarizes mask in __getitem__."""

    def setUp(self):
        warnings.simplefilter("ignore")
        # Create a simple 2-class dataset
        self.tmpdir = tempfile.mkdtemp(prefix="test_grid_binary_")
        img_dir = os.path.join(self.tmpdir, "images")
        msk_dir = os.path.join(self.tmpdir, "masks")
        os.makedirs(img_dir)
        os.makedirs(msk_dir)

        h = w = 64
        img = np.full((3, h, w), 128, dtype=np.uint8)
        mask = np.zeros((h, w), dtype=np.uint8)
        mask[h // 2:, :] = 1

        img_path = os.path.join(img_dir, "img.tif")
        msk_path = os.path.join(msk_dir, "msk.tif")
        _create_geotiff(img_path, img)
        _create_geotiff(msk_path, mask)

        self.csv_path = os.path.join(self.tmpdir, "dataset.csv")
        with open(self.csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["image", "mask"])
            writer.writerow([img_path, msk_path])

    def tearDown(self):
        shutil.rmtree(self.tmpdir)

    def test_binary_mask(self):
        ds = RandomCropSegmentationDataset(
            input_csv_path=self.csv_path,
            crop_size=[32, 32],
            n_classes=2,
            grid_mode=True,
            overlap_x=0.0,
            overlap_y=0.0,
            min_valid_ratio=0.0,
        )
        sample = ds[0]
        mask = sample["mask"]
        if hasattr(mask, 'numpy'):
            mask = mask.numpy()
        unique_values = np.unique(mask)
        # Should only contain 0 and/or 1
        self.assertTrue(all(v in [0, 1] for v in unique_values))


if __name__ == "__main__":
    unittest.main()
