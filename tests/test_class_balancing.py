# -*- coding: utf-8 -*-
"""
Tests for the integrated class balancing system:
  - RandomCropSegmentationDataset (class-balanced sampling, CutMix)
  - WeightedDiceSCELoss
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
import torch

from pytorch_segmentation_models_trainer.custom_losses.loss import (
    WeightedDiceSCELoss,
)
from pytorch_segmentation_models_trainer.dataset_loader.dataset import (
    RandomCropSegmentationDataset,
    _RasterioLRUCache,
)


def _create_geotiff(path, data, count=None, dtype="uint8"):
    """Write a numpy array as a GeoTIFF with rasterio.

    Args:
        path: Output file path.
        data: numpy array. (H, W) for single band, (C, H, W) for multi-band.
        count: Number of bands. Inferred from data shape if None.
        dtype: Data type string.
    """
    if data.ndim == 2:
        h, w = data.shape
        bands = 1
        data = data[np.newaxis, ...]  # (1, H, W)
    else:
        bands, h, w = data.shape

    if count is not None:
        bands = count

    transform = from_bounds(0, 0, w, h, w, h)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=h,
        width=w,
        count=bands,
        dtype=dtype,
        transform=transform,
    ) as dst:
        for b in range(bands):
            dst.write(data[b], b + 1)


class SyntheticDatasetMixin:
    """Creates a temporary directory with synthetic GeoTIFF images/masks and CSV."""

    N_CLASSES = 4
    IMG_SIZE = 64  # small for fast tests
    N_IMAGES = 3
    CROP_SIZE = [32, 32]

    def _create_synthetic_dataset(self):
        """Build a temp dir with N_IMAGES synthetic image/mask pairs.

        Mask layout (for a 64x64 image, 4 classes):
          - top-left quadrant:  class 0 (background)
          - top-right quadrant: class 1
          - bottom-left:        class 2
          - bottom-right:       class 3

        This guarantees known spatial distribution for testing
        class-balanced sampling and diversity checks.
        """
        self.tmpdir = tempfile.mkdtemp(prefix="test_cb_")
        img_dir = os.path.join(self.tmpdir, "images")
        msk_dir = os.path.join(self.tmpdir, "masks")
        os.makedirs(img_dir)
        os.makedirs(msk_dir)

        self.image_paths = []
        self.mask_paths = []
        h = w = self.IMG_SIZE
        half = h // 2

        for i in range(self.N_IMAGES):
            # RGB image with distinct quadrant colors
            img = np.zeros((3, h, w), dtype=np.uint8)
            img[:, :half, :half] = 30 + i * 10      # dark (class 0 region)
            img[:, :half, half:] = 100 + i * 10      # mid  (class 1 region)
            img[:, half:, :half] = 170 + i * 5       # bright (class 2 region)
            img[:, half:, half:] = 220 + i * 3       # very bright (class 3 region)

            mask = np.zeros((h, w), dtype=np.uint8)
            mask[:half, :half] = 0
            mask[:half, half:] = 1
            mask[half:, :half] = 2
            mask[half:, half:] = 3

            img_path = os.path.join(img_dir, f"img_{i:02d}.tif")
            msk_path = os.path.join(msk_dir, f"msk_{i:02d}.tif")
            _create_geotiff(img_path, img)
            _create_geotiff(msk_path, mask)
            self.image_paths.append(img_path)
            self.mask_paths.append(msk_path)

        # Write CSV
        self.csv_path = os.path.join(self.tmpdir, "dataset.csv")
        with open(self.csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["image", "mask"])
            for img_p, msk_p in zip(self.image_paths, self.mask_paths):
                writer.writerow([img_p, msk_p])

    def _cleanup(self):
        if hasattr(self, "tmpdir") and os.path.exists(self.tmpdir):
            shutil.rmtree(self.tmpdir)


# ----------------------------------------------------------------------
# 1. _RasterioLRUCache
# ----------------------------------------------------------------------

class TestRasterioLRUCache(unittest.TestCase, SyntheticDatasetMixin):

    def setUp(self):
        warnings.simplefilter("ignore")
        self._create_synthetic_dataset()

    def tearDown(self):
        self._cleanup()

    def test_get_returns_open_dataset(self):
        cache = _RasterioLRUCache(maxsize=4)
        src = cache.get(self.image_paths[0])
        self.assertFalse(src.closed)
        self.assertEqual(src.width, self.IMG_SIZE)
        self.assertEqual(src.height, self.IMG_SIZE)
        cache.close_all()

    def test_cache_returns_same_handle(self):
        cache = _RasterioLRUCache(maxsize=4)
        src1 = cache.get(self.image_paths[0])
        src2 = cache.get(self.image_paths[0])
        self.assertIs(src1, src2)
        cache.close_all()

    def test_eviction_on_maxsize(self):
        cache = _RasterioLRUCache(maxsize=2)
        src0 = cache.get(self.image_paths[0])
        _ = cache.get(self.image_paths[1])
        # Adding a 3rd should evict the first (LRU)
        _ = cache.get(self.image_paths[2])
        self.assertEqual(len(cache), 2)
        self.assertTrue(src0.closed)
        cache.close_all()

    def test_close_all(self):
        cache = _RasterioLRUCache(maxsize=4)
        handles = [cache.get(p) for p in self.image_paths]
        cache.close_all()
        for h in handles:
            self.assertTrue(h.closed)
        self.assertEqual(len(cache), 0)


# ----------------------------------------------------------------------
# 2. _build_class_position_index
# ----------------------------------------------------------------------

class TestBuildClassPositionIndex(unittest.TestCase, SyntheticDatasetMixin):

    def setUp(self):
        warnings.simplefilter("ignore")
        self._create_synthetic_dataset()

    def tearDown(self):
        if hasattr(self, "ds"):
            self.ds._close_cache()
        self._cleanup()

    def _make_dataset(self, **kwargs):
        defaults = dict(
            input_csv_path=self.csv_path,
            crop_size=self.CROP_SIZE,
            samples_per_epoch=100,
            n_classes=self.N_CLASSES,
            use_rasterio=True,
            class_balanced_sampling=True,
            class_sampling_stride=32,  # same as crop -> non-overlapping grid
        )
        defaults.update(kwargs)
        self.ds = RandomCropSegmentationDataset(**defaults)
        return self.ds

    def test_all_classes_have_positions(self):
        ds = self._make_dataset()
        for c in range(self.N_CLASSES):
            self.assertIn(c, ds._class_positions)
            self.assertGreater(len(ds._class_positions[c]), 0)

    def test_positions_cover_all_images(self):
        ds = self._make_dataset()
        # Each class should have positions in all N_IMAGES images
        for c in range(self.N_CLASSES):
            img_indices = set(ds._class_positions[c][:, 0])
            for idx in ds._valid_indices:
                self.assertIn(idx, img_indices,
                              f"Class {c} missing from image {idx}")

    def test_sampling_weights_are_normalized(self):
        ds = self._make_dataset()
        self.assertAlmostEqual(
            ds._class_sampling_weights.sum(), 1.0, places=6
        )

    def test_uniform_distribution_yields_equal_weights(self):
        """With equal-area quadrants, all classes should get equal weight."""
        ds = self._make_dataset()
        # Classes occupy equal area -> equal position count -> equal weights
        weights = ds._class_sampling_weights
        for c in range(self.N_CLASSES):
            self.assertAlmostEqual(
                weights[c], 1.0 / self.N_CLASSES, places=2,
                msg=f"Weight for class {c} should be ~{1.0/self.N_CLASSES:.3f}"
            )

    def test_rare_class_gets_higher_weight(self):
        """A class with fewer positions should get higher sampling weight."""
        # Create a dataset with unbalanced classes
        tmpdir2 = tempfile.mkdtemp(prefix="test_cb_unbal_")
        h = w = 64
        img = np.full((3, h, w), 128, dtype=np.uint8)
        mask = np.zeros((h, w), dtype=np.uint8)
        # Class 0: 90% of area, Class 1: 10% of area
        mask[:, :] = 0
        mask[:6, :] = 1  # ~10% rows

        img_path = os.path.join(tmpdir2, "img.tif")
        msk_path = os.path.join(tmpdir2, "msk.tif")
        _create_geotiff(img_path, img)
        _create_geotiff(msk_path, mask)

        csv_path = os.path.join(tmpdir2, "dataset.csv")
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["image", "mask"])
            writer.writerow([img_path, msk_path])

        ds = RandomCropSegmentationDataset(
            input_csv_path=csv_path,
            crop_size=[32, 32],
            samples_per_epoch=50,
            n_classes=2,
            use_rasterio=True,
            class_balanced_sampling=True,
            class_sampling_stride=16,
        )

        # Class 1 (rare) should have higher weight than class 0 (dominant)
        self.assertGreater(
            ds._class_sampling_weights[1],
            ds._class_sampling_weights[0],
            "Rare class should have higher sampling weight"
        )

        ds._close_cache()
        shutil.rmtree(tmpdir2)


# ----------------------------------------------------------------------
# 3. _sample_class_aware_position
# ----------------------------------------------------------------------

class TestSampleClassAwarePosition(unittest.TestCase, SyntheticDatasetMixin):

    def setUp(self):
        warnings.simplefilter("ignore")
        self._create_synthetic_dataset()
        self.ds = RandomCropSegmentationDataset(
            input_csv_path=self.csv_path,
            crop_size=self.CROP_SIZE,
            samples_per_epoch=100,
            n_classes=self.N_CLASSES,
            use_rasterio=True,
            class_balanced_sampling=True,
            class_sampling_stride=32,
        )

    def tearDown(self):
        self.ds._close_cache()
        self._cleanup()

    def test_returns_valid_position(self):
        np.random.seed(42)
        img_idx, x, y = self.ds._sample_class_aware_position()
        self.assertIn(img_idx, self.ds._valid_indices)
        w, h = self.ds.image_dims[img_idx]
        crop_h, crop_w = self.ds.crop_size
        self.assertGreaterEqual(x, 0)
        self.assertGreaterEqual(y, 0)
        self.assertLessEqual(x + crop_w, w)
        self.assertLessEqual(y + crop_h, h)

    def test_exclude_class_avoids_target(self):
        """When excluding a class, the sampled position should not target it."""
        np.random.seed(0)
        n_samples = 200
        exclude = 2
        targeted_classes = []
        for _ in range(n_samples):
            img_idx, x, y = self.ds._sample_class_aware_position(
                exclude_class=exclude
            )
            # Read the mask at that position to see what class is dominant
            mask_path = self.ds.get_path(img_idx, key=self.ds.mask_key)
            mask_crop = self.ds._read_crop(mask_path, x, y, is_mask=True)
            dominant = self.ds._get_dominant_class(mask_crop)
            targeted_classes.append(dominant)

        # The excluded class should NOT be the majority of targets.
        # Due to jitter it may occasionally appear, but shouldn't dominate.
        from collections import Counter
        counts = Counter(targeted_classes)
        total = sum(counts.values())
        excluded_ratio = counts.get(exclude, 0) / total
        self.assertLess(
            excluded_ratio, 0.4,
            f"Excluded class {exclude} appeared as dominant in "
            f"{excluded_ratio:.0%} of samples (should be rare)"
        )

    def test_jitter_varies_positions(self):
        """Multiple calls should return different positions (not grid-locked)."""
        np.random.seed(42)
        positions = set()
        for _ in range(50):
            _, x, y = self.ds._sample_class_aware_position()
            positions.add((x, y))
        # With jitter, we should get more than just the grid positions
        self.assertGreater(len(positions), 5)


# ----------------------------------------------------------------------
# 4. _get_dominant_class
# ----------------------------------------------------------------------

class TestGetDominantClass(unittest.TestCase):

    def setUp(self):
        warnings.simplefilter("ignore")

    def _make_dummy_ds(self, n_classes=4):
        """Create a minimal object with _get_dominant_class accessible."""
        # We only need n_classes to call _get_dominant_class
        obj = object.__new__(RandomCropSegmentationDataset)
        obj.n_classes = n_classes
        obj.soft_labels = False
        return obj

    def test_single_class(self):
        ds = self._make_dummy_ds()
        mask = np.full((32, 32), 2, dtype=np.uint8)
        self.assertEqual(ds._get_dominant_class(mask), 2)

    def test_majority_class(self):
        ds = self._make_dummy_ds()
        mask = np.zeros((32, 32), dtype=np.uint8)
        mask[:20, :] = 1  # 20/32 rows = 62.5%
        mask[20:, :] = 3  # 12/32 rows = 37.5%
        self.assertEqual(ds._get_dominant_class(mask), 1)

    def test_ignores_values_above_n_classes(self):
        ds = self._make_dummy_ds(n_classes=4)
        mask = np.full((32, 32), 255, dtype=np.uint8)
        mask[0, 0] = 2
        self.assertEqual(ds._get_dominant_class(mask), 2)

    def test_all_ignore_returns_minus_one(self):
        ds = self._make_dummy_ds(n_classes=4)
        mask = np.full((32, 32), 255, dtype=np.uint8)
        self.assertEqual(ds._get_dominant_class(mask), -1)

    def test_empty_mask(self):
        ds = self._make_dummy_ds(n_classes=4)
        mask = np.array([], dtype=np.uint8).reshape(0, 0)
        self.assertEqual(ds._get_dominant_class(mask), -1)


# ----------------------------------------------------------------------
# 5. _apply_cutmix
# ----------------------------------------------------------------------

class TestApplyCutMix(unittest.TestCase):

    def setUp(self):
        warnings.simplefilter("ignore")
        self.ds = object.__new__(RandomCropSegmentationDataset)
        self.ds.cutmix_alpha = 1.0

    def test_output_shape_matches_input(self):
        np.random.seed(42)
        img1 = np.full((32, 32, 3), 100, dtype=np.uint8)
        mask1 = np.full((32, 32), 0, dtype=np.uint8)
        img2 = np.full((32, 32, 3), 200, dtype=np.uint8)
        mask2 = np.full((32, 32), 1, dtype=np.uint8)

        out_img, out_mask = self.ds._apply_cutmix(img1, mask1, img2, mask2)
        self.assertEqual(out_img.shape, img1.shape)
        self.assertEqual(out_mask.shape, mask1.shape)

    def test_image_and_mask_mixed_consistently(self):
        """Where image has img2 values, mask should have mask2 values."""
        np.random.seed(42)
        img1 = np.full((32, 32, 3), 100, dtype=np.uint8)
        mask1 = np.full((32, 32), 0, dtype=np.uint8)
        img2 = np.full((32, 32, 3), 200, dtype=np.uint8)
        mask2 = np.full((32, 32), 1, dtype=np.uint8)

        out_img, out_mask = self.ds._apply_cutmix(img1, mask1, img2, mask2)

        # Pixels from img2 (value 200) should correspond to mask2 (value 1)
        from_img2 = np.all(out_img == 200, axis=2)
        from_mask2 = out_mask == 1
        np.testing.assert_array_equal(from_img2, from_mask2)

    def test_does_not_modify_inputs(self):
        np.random.seed(42)
        img1 = np.full((32, 32, 3), 100, dtype=np.uint8)
        mask1 = np.full((32, 32), 0, dtype=np.uint8)
        img2 = np.full((32, 32, 3), 200, dtype=np.uint8)
        mask2 = np.full((32, 32), 1, dtype=np.uint8)

        img1_copy = img1.copy()
        mask1_copy = mask1.copy()
        self.ds._apply_cutmix(img1, mask1, img2, mask2)
        np.testing.assert_array_equal(img1, img1_copy)
        np.testing.assert_array_equal(mask1, mask1_copy)

    def test_cutmix_produces_mixed_content(self):
        """Over many seeds, CutMix should sometimes mix both sources."""
        found_mixed = False
        for seed in range(50):
            np.random.seed(seed)
            img1 = np.full((32, 32, 3), 100, dtype=np.uint8)
            mask1 = np.full((32, 32), 0, dtype=np.uint8)
            img2 = np.full((32, 32, 3), 200, dtype=np.uint8)
            mask2 = np.full((32, 32), 1, dtype=np.uint8)
            out_img, _ = self.ds._apply_cutmix(img1, mask1, img2, mask2)
            has_100 = np.any(out_img == 100)
            has_200 = np.any(out_img == 200)
            if has_100 and has_200:
                found_mixed = True
                break
        self.assertTrue(found_mixed, "CutMix never produced mixed output")


# ----------------------------------------------------------------------
# 6. _is_crop_valid
# ----------------------------------------------------------------------

class TestIsCropValid(unittest.TestCase):

    def setUp(self):
        warnings.simplefilter("ignore")

    def _make_dummy_ds(self, min_valid_ratio=0.1, n_classes=4):
        ds = object.__new__(RandomCropSegmentationDataset)
        ds.min_valid_ratio = min_valid_ratio
        ds.n_classes = n_classes
        ds.soft_labels = False
        return ds

    def test_all_black_below_ratio(self):
        ds = self._make_dummy_ds(min_valid_ratio=0.1)
        img = np.zeros((32, 32, 3), dtype=np.uint8)
        self.assertFalse(ds._is_crop_valid(img))

    def test_all_nonzero_above_ratio(self):
        ds = self._make_dummy_ds(min_valid_ratio=0.1)
        img = np.full((32, 32, 3), 128, dtype=np.uint8)
        self.assertTrue(ds._is_crop_valid(img))

    def test_partial_valid(self):
        ds = self._make_dummy_ds(min_valid_ratio=0.5)
        img = np.zeros((32, 32, 3), dtype=np.uint8)
        img[:20, :, :] = 128  # 20/32 = 62.5% valid
        self.assertTrue(ds._is_crop_valid(img))

        img2 = np.zeros((32, 32, 3), dtype=np.uint8)
        img2[:10, :, :] = 128  # 10/32 = 31.25% valid
        self.assertFalse(ds._is_crop_valid(img2))

    def test_empty_image_invalid(self):
        ds = self._make_dummy_ds()
        img = np.array([], dtype=np.uint8).reshape(0, 0, 3)
        self.assertFalse(ds._is_crop_valid(img))



# ----------------------------------------------------------------------
# 7. _get_random_crop (integration)
# ----------------------------------------------------------------------

class TestGetRandomCrop(unittest.TestCase, SyntheticDatasetMixin):

    def setUp(self):
        warnings.simplefilter("ignore")
        self._create_synthetic_dataset()

    def tearDown(self):
        if hasattr(self, "ds"):
            self.ds._close_cache()
        self._cleanup()

    def test_crop_shape(self):
        self.ds = RandomCropSegmentationDataset(
            input_csv_path=self.csv_path,
            crop_size=self.CROP_SIZE,
            samples_per_epoch=50,
            n_classes=self.N_CLASSES,
            use_rasterio=True,
        )
        np.random.seed(42)
        img, mask = self.ds._get_random_crop()
        self.assertEqual(img.shape, (32, 32, 3))
        self.assertEqual(mask.shape, (32, 32))

    def test_crop_with_balanced_sampling(self):
        self.ds = RandomCropSegmentationDataset(
            input_csv_path=self.csv_path,
            crop_size=self.CROP_SIZE,
            samples_per_epoch=50,
            n_classes=self.N_CLASSES,
            use_rasterio=True,
            class_balanced_sampling=True,
            class_sampling_stride=32,
        )
        np.random.seed(42)
        img, mask = self.ds._get_random_crop()
        self.assertEqual(img.shape, (32, 32, 3))
        self.assertEqual(mask.shape, (32, 32))
        self.assertTrue(img.dtype == np.uint8)
        self.assertTrue(mask.dtype == np.uint8)

    def test_exclude_class_in_balanced_crop(self):
        """_get_random_crop with exclude_class should bias away from it."""
        self.ds = RandomCropSegmentationDataset(
            input_csv_path=self.csv_path,
            crop_size=self.CROP_SIZE,
            samples_per_epoch=200,
            n_classes=self.N_CLASSES,
            use_rasterio=True,
            class_balanced_sampling=True,
            class_sampling_stride=32,
        )
        np.random.seed(42)
        exclude = 0
        dominant_classes = []
        for _ in range(100):
            _, mask = self.ds._get_random_crop(exclude_class=exclude)
            dominant_classes.append(self.ds._get_dominant_class(mask))

        from collections import Counter
        counts = Counter(dominant_classes)
        total = sum(counts.values())
        excluded_ratio = counts.get(exclude, 0) / total
        self.assertLess(excluded_ratio, 0.4)


# ----------------------------------------------------------------------
# 8. __getitem__ with CutMix
# ----------------------------------------------------------------------

class TestGetItemWithCutMix(unittest.TestCase, SyntheticDatasetMixin):

    def setUp(self):
        warnings.simplefilter("ignore")
        self._create_synthetic_dataset()

    def tearDown(self):
        if hasattr(self, "ds"):
            self.ds._close_cache()
        self._cleanup()

    def test_getitem_without_cutmix(self):
        self.ds = RandomCropSegmentationDataset(
            input_csv_path=self.csv_path,
            crop_size=self.CROP_SIZE,
            samples_per_epoch=10,
            n_classes=self.N_CLASSES,
            use_rasterio=True,
            cutmix_prob=0.0,
        )
        np.random.seed(42)
        result = self.ds[0]
        self.assertIn("image", result)
        self.assertIn("mask", result)
        self.assertEqual(result["image"].shape, (3, 32, 32))
        self.assertEqual(result["mask"].shape, (32, 32))

    def test_getitem_with_cutmix(self):
        self.ds = RandomCropSegmentationDataset(
            input_csv_path=self.csv_path,
            crop_size=self.CROP_SIZE,
            samples_per_epoch=10,
            n_classes=self.N_CLASSES,
            use_rasterio=True,
            cutmix_prob=1.0,  # always apply
            cutmix_alpha=1.0,
            class_balanced_sampling=True,
            class_sampling_stride=32,
        )
        np.random.seed(42)
        result = self.ds[0]
        self.assertIn("image", result)
        self.assertIn("mask", result)
        self.assertEqual(result["image"].shape, (3, 32, 32))
        self.assertEqual(result["mask"].shape, (32, 32))

    def test_len_matches_samples_per_epoch(self):
        self.ds = RandomCropSegmentationDataset(
            input_csv_path=self.csv_path,
            crop_size=self.CROP_SIZE,
            samples_per_epoch=42,
            n_classes=self.N_CLASSES,
            use_rasterio=True,
        )
        self.assertEqual(len(self.ds), 42)

    def test_auto_samples_per_epoch(self):
        self.ds = RandomCropSegmentationDataset(
            input_csv_path=self.csv_path,
            crop_size=self.CROP_SIZE,
            samples_per_epoch=0,  # auto
            n_classes=self.N_CLASSES,
            use_rasterio=True,
        )
        # auto = 3 * total_area / crop_area
        total_area = self.N_IMAGES * self.IMG_SIZE * self.IMG_SIZE
        crop_area = 32 * 32
        expected = int(3 * total_area / crop_area)
        self.assertEqual(self.ds.samples_per_epoch, expected)


# ----------------------------------------------------------------------
# 9. WeightedDiceSCELoss
# ----------------------------------------------------------------------

class TestWeightedDiceSCELoss(unittest.TestCase):

    def setUp(self):
        warnings.simplefilter("ignore")

    def test_basic_forward(self):
        loss_fn = WeightedDiceSCELoss(
            num_classes=4,
            dice_weight=0.25,
            sce_weight=0.75,
            smooth_factor=0.0,
            ignore_index=255,
        )
        # (B, C, H, W) predictions, (B, H, W) targets
        pred = torch.randn(2, 4, 32, 32, requires_grad=True)
        target = torch.randint(0, 4, (2, 32, 32))

        loss = loss_fn(pred, target)
        self.assertEqual(loss.dim(), 0)  # scalar
        self.assertTrue(loss.requires_grad)
        self.assertGreater(loss.item(), 0)

    def test_perfect_prediction_low_loss(self):
        loss_fn = WeightedDiceSCELoss(
            num_classes=4,
            dice_weight=0.25,
            sce_weight=0.75,
            smooth_factor=0.0,
            ignore_index=255,
        )
        target = torch.randint(0, 4, (2, 32, 32))
        # Create "perfect" one-hot logits
        pred = torch.zeros(2, 4, 32, 32)
        for b in range(2):
            for c in range(4):
                pred[b, c, target[b] == c] = 10.0  # high logit for correct class

        loss = loss_fn(pred, target)
        self.assertLess(loss.item(), 0.5, "Perfect prediction should have low loss")

    def test_ignore_index_excluded(self):
        loss_fn = WeightedDiceSCELoss(
            num_classes=4,
            dice_weight=0.25,
            sce_weight=0.75,
            smooth_factor=0.0,
            ignore_index=255,
        )
        pred = torch.randn(2, 4, 32, 32, requires_grad=True)
        target = torch.full((2, 32, 32), 255, dtype=torch.long)
        # All pixels ignored -> loss should be 0
        loss = loss_fn(pred, target)
        self.assertAlmostEqual(loss.item(), 0.0, places=5)

    def test_gradient_flows(self):
        loss_fn = WeightedDiceSCELoss(
            num_classes=4,
            dice_weight=0.25,
            sce_weight=0.75,
            ignore_index=255,
        )
        pred = torch.randn(2, 4, 32, 32, requires_grad=True)
        target = torch.randint(0, 4, (2, 32, 32))

        loss = loss_fn(pred, target)
        loss.backward()
        self.assertIsNotNone(pred.grad)
        self.assertFalse(torch.all(pred.grad == 0))

    def test_sce_components(self):
        """SCE alpha/beta should scale CE and RCE components."""
        pred = torch.randn(2, 4, 32, 32)
        target = torch.randint(0, 4, (2, 32, 32))

        loss_high_rce = WeightedDiceSCELoss(
            num_classes=4, dice_weight=0.0, sce_weight=1.0,
            sce_alpha=0.0, sce_beta=1.0, ignore_index=255,
        )
        loss_high_ce = WeightedDiceSCELoss(
            num_classes=4, dice_weight=0.0, sce_weight=1.0,
            sce_alpha=1.0, sce_beta=0.0, ignore_index=255,
        )
        val_rce = loss_high_rce(pred, target).item()
        val_ce = loss_high_ce(pred, target).item()
        # Both should be positive, and generally different
        self.assertGreater(val_rce, 0)
        self.assertGreater(val_ce, 0)

    def test_mixed_ignore_and_valid(self):
        loss_fn = WeightedDiceSCELoss(
            num_classes=4,
            dice_weight=0.25,
            sce_weight=0.75,
            ignore_index=255,
        )
        pred = torch.randn(2, 4, 32, 32, requires_grad=True)
        target = torch.randint(0, 4, (2, 32, 32))
        target[:, :16, :] = 255  # half ignored

        loss = loss_fn(pred, target)
        self.assertGreater(loss.item(), 0)
        self.assertTrue(loss.requires_grad)


# ----------------------------------------------------------------------
# 12. Pickle / Serialization (for DataLoader workers)
# ----------------------------------------------------------------------

class TestDatasetPickle(unittest.TestCase, SyntheticDatasetMixin):

    def setUp(self):
        warnings.simplefilter("ignore")
        self._create_synthetic_dataset()

    def tearDown(self):
        if hasattr(self, "ds"):
            self.ds._close_cache()
        self._cleanup()

    def test_getstate_clears_cache(self):
        self.ds = RandomCropSegmentationDataset(
            input_csv_path=self.csv_path,
            crop_size=self.CROP_SIZE,
            samples_per_epoch=10,
            n_classes=self.N_CLASSES,
            use_rasterio=True,
            class_balanced_sampling=True,
            class_sampling_stride=32,
        )
        # Populate cache
        np.random.seed(42)
        _ = self.ds[0]
        self.assertGreater(len(self.ds._file_cache), 0)

        # Pickle should clear cache
        state = self.ds.__getstate__()
        self.assertEqual(len(state["_file_cache"]), 0)

    def test_setstate_restores_functionality(self):
        import pickle
        self.ds = RandomCropSegmentationDataset(
            input_csv_path=self.csv_path,
            crop_size=self.CROP_SIZE,
            samples_per_epoch=10,
            n_classes=self.N_CLASSES,
            use_rasterio=True,
            class_balanced_sampling=True,
            class_sampling_stride=32,
        )
        # Pickle round-trip
        data = pickle.dumps(self.ds)
        ds2 = pickle.loads(data)

        np.random.seed(42)
        result = ds2[0]
        self.assertIn("image", result)
        self.assertIn("mask", result)
        ds2._close_cache()


# ----------------------------------------------------------------------
# 13. Edge cases
# ----------------------------------------------------------------------

class TestEdgeCases(unittest.TestCase):

    def setUp(self):
        warnings.simplefilter("ignore")

    def test_dataset_rejects_images_too_small(self):
        """Images smaller than crop_size should be excluded from _valid_indices."""
        tmpdir = tempfile.mkdtemp(prefix="test_cb_small_")
        # One image smaller than crop, one larger
        img_small = np.full((3, 16, 16), 128, dtype=np.uint8)
        img_large = np.full((3, 64, 64), 128, dtype=np.uint8)
        mask_small = np.zeros((16, 16), dtype=np.uint8)
        mask_large = np.zeros((64, 64), dtype=np.uint8)
        mask_large[:32, :] = 1

        paths = []
        for name, img, mask in [
            ("small", img_small, mask_small),
            ("large", img_large, mask_large),
        ]:
            ip = os.path.join(tmpdir, f"{name}_img.tif")
            mp = os.path.join(tmpdir, f"{name}_msk.tif")
            _create_geotiff(ip, img)
            _create_geotiff(mp, mask)
            paths.append((ip, mp))

        csv_path = os.path.join(tmpdir, "dataset.csv")
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["image", "mask"])
            for ip, mp in paths:
                writer.writerow([ip, mp])

        ds = RandomCropSegmentationDataset(
            input_csv_path=csv_path,
            crop_size=[32, 32],
            samples_per_epoch=10,
            n_classes=2,
            use_rasterio=True,
        )
        # Only the large image should be in _valid_indices
        self.assertEqual(len(ds._valid_indices), 1)
        self.assertEqual(ds._valid_indices[0], 1)  # index of large image

        ds._close_cache()
        shutil.rmtree(tmpdir)

    def test_all_images_too_small_raises(self):
        tmpdir = tempfile.mkdtemp(prefix="test_cb_allsmall_")
        img = np.full((3, 16, 16), 128, dtype=np.uint8)
        mask = np.zeros((16, 16), dtype=np.uint8)

        ip = os.path.join(tmpdir, "img.tif")
        mp = os.path.join(tmpdir, "msk.tif")
        _create_geotiff(ip, img)
        _create_geotiff(mp, mask)

        csv_path = os.path.join(tmpdir, "dataset.csv")
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["image", "mask"])
            writer.writerow([ip, mp])

        with self.assertRaises(ValueError):
            RandomCropSegmentationDataset(
                input_csv_path=csv_path,
                crop_size=[32, 32],
                samples_per_epoch=10,
                n_classes=2,
                use_rasterio=True,
            )
        shutil.rmtree(tmpdir)

    def test_binary_classification(self):
        """With n_classes=2, mask should be binarized."""
        tmpdir = tempfile.mkdtemp(prefix="test_cb_binary_")
        img = np.full((3, 64, 64), 128, dtype=np.uint8)
        mask = np.zeros((64, 64), dtype=np.uint8)
        mask[:32, :] = 3  # any nonzero value

        ip = os.path.join(tmpdir, "img.tif")
        mp = os.path.join(tmpdir, "msk.tif")
        _create_geotiff(ip, img)
        _create_geotiff(mp, mask)

        csv_path = os.path.join(tmpdir, "dataset.csv")
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["image", "mask"])
            writer.writerow([ip, mp])

        ds = RandomCropSegmentationDataset(
            input_csv_path=csv_path,
            crop_size=[32, 32],
            samples_per_epoch=10,
            n_classes=2,
            use_rasterio=True,
        )
        np.random.seed(42)
        _, mask_out = ds._get_random_crop()
        unique = set(np.unique(mask_out))
        self.assertTrue(
            unique.issubset({0, 1}),
            f"Binary mask should only have 0 and 1, got {unique}"
        )

        ds._close_cache()
        shutil.rmtree(tmpdir)

    def test_selected_bands(self):
        """selected_bands should restrict which bands are read."""
        tmpdir = tempfile.mkdtemp(prefix="test_cb_bands_")
        # 4-band image (RGBI)
        img = np.zeros((4, 64, 64), dtype=np.uint8)
        img[0] = 10   # R
        img[1] = 20   # G
        img[2] = 30   # B
        img[3] = 40   # I
        mask = np.zeros((64, 64), dtype=np.uint8)

        ip = os.path.join(tmpdir, "img.tif")
        mp = os.path.join(tmpdir, "msk.tif")
        _create_geotiff(ip, img, count=4)
        _create_geotiff(mp, mask)

        csv_path = os.path.join(tmpdir, "dataset.csv")
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["image", "mask"])
            writer.writerow([ip, mp])

        ds = RandomCropSegmentationDataset(
            input_csv_path=csv_path,
            crop_size=[32, 32],
            samples_per_epoch=10,
            n_classes=2,
            use_rasterio=True,
            selected_bands=[1, 3],  # R and I only
        )
        np.random.seed(42)
        img_out, _ = ds._get_random_crop()
        # Should have 2 channels
        self.assertEqual(img_out.shape[2], 2)
        # selected_bands is 1-based: [1, 3] reads rasterio bands 1 and 3.
        # Band 1 = img[0] = R = 10, Band 3 = img[2] = B = 30.
        self.assertTrue(np.all(img_out[:, :, 0] == 10))
        self.assertTrue(np.all(img_out[:, :, 1] == 30))

        ds._close_cache()
        shutil.rmtree(tmpdir)


# ----------------------------------------------------------------------
# 14. Statistical validation of class-balanced sampling
# ----------------------------------------------------------------------

class TestBalancedSamplingDistribution(unittest.TestCase):

    def setUp(self):
        warnings.simplefilter("ignore")

    def test_balanced_sampling_reduces_class_imbalance(self):
        """Class-balanced sampling should produce more uniform class distribution
        than area-based sampling on an unbalanced dataset."""
        tmpdir = tempfile.mkdtemp(prefix="test_cb_stat_")
        h = w = 128
        img = np.full((3, h, w), 128, dtype=np.uint8)
        mask = np.zeros((h, w), dtype=np.uint8)
        # Class 0: 75% of area, Class 1: 25% (rows 0-31).
        # Class 1 must span at least crop_h=32 rows so a crop can land
        # entirely within the class-1 region and be dominant.
        mask[:, :] = 0
        mask[:32, :] = 1

        ip = os.path.join(tmpdir, "img.tif")
        mp = os.path.join(tmpdir, "msk.tif")
        _create_geotiff(ip, img)
        _create_geotiff(mp, mask)

        csv_path = os.path.join(tmpdir, "dataset.csv")
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["image", "mask"])
            writer.writerow([ip, mp])

        n_samples = 300

        # Without balancing
        ds_unbal = RandomCropSegmentationDataset(
            input_csv_path=csv_path,
            crop_size=[32, 32],
            samples_per_epoch=n_samples,
            n_classes=2,
            use_rasterio=True,
            class_balanced_sampling=False,
        )
        np.random.seed(42)
        dominant_unbal = []
        for i in range(n_samples):
            _, m = ds_unbal._get_random_crop()
            dominant_unbal.append(ds_unbal._get_dominant_class(m))

        # With balancing
        ds_bal = RandomCropSegmentationDataset(
            input_csv_path=csv_path,
            crop_size=[32, 32],
            samples_per_epoch=n_samples,
            n_classes=2,
            use_rasterio=True,
            class_balanced_sampling=True,
            class_sampling_stride=16,
        )
        np.random.seed(42)
        dominant_bal = []
        for i in range(n_samples):
            _, m = ds_bal._get_random_crop()
            dominant_bal.append(ds_bal._get_dominant_class(m))

        from collections import Counter
        counts_unbal = Counter(dominant_unbal)
        counts_bal = Counter(dominant_bal)

        # Unbalanced: class 1 should be very rare (close to 12.5%)
        ratio_unbal = counts_unbal.get(1, 0) / n_samples
        # Balanced: class 1 should be much more frequent
        ratio_bal = counts_bal.get(1, 0) / n_samples

        self.assertGreater(
            ratio_bal, ratio_unbal,
            f"Balanced sampling ({ratio_bal:.1%}) should have more class-1 "
            f"crops than unbalanced ({ratio_unbal:.1%})"
        )
        # Class 1 with balanced sampling should be at least 30%
        self.assertGreater(
            ratio_bal, 0.3,
            f"Balanced sampling should give class 1 at least 30%, got {ratio_bal:.1%}"
        )

        ds_unbal._close_cache()
        ds_bal._close_cache()
        shutil.rmtree(tmpdir)


if __name__ == "__main__":
    unittest.main()
