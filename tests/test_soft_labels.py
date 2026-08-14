# -*- coding: utf-8 -*-
"""
Tests for soft label support in the training pipeline:
  - WeightedJMLSCELoss with soft targets (B, C, H, W) float
  - RandomCropSegmentationDataset with soft_labels=True
  - Model training_step/validation_step soft label handling
"""

import os
import tempfile
import unittest

# IMPORTANT: import pytorch_segmentation_models_trainer BEFORE torch
# to resolve fbgemm.dll vs GDAL DLL conflict on Windows.
import pytorch_segmentation_models_trainer  # noqa: F401

import numpy as np
import torch
import torch.nn.functional as F

from pytorch_segmentation_models_trainer.custom_losses.loss import (
    WeightedJMLSCELoss,
)


class TestJMLSCELossSoftTargets(unittest.TestCase):
    """Tests for WeightedJMLSCELoss with soft label distributions."""

    def setUp(self):
        self.num_classes = 7
        self.loss_fn = WeightedJMLSCELoss(
            num_classes=self.num_classes,
            jml_weight=0.25,
            sce_weight=0.75,
            sce_alpha=1.0,
            sce_beta=0.5,
        )
        self.B, self.H, self.W = 2, 16, 16

    def _make_soft_targets(self, dominant_class=1, confidence=0.95):
        """Create soft targets with a dominant class and small uniform noise."""
        targets = torch.zeros(self.B, self.num_classes, self.H, self.W)
        noise = (1 - confidence) / (self.num_classes - 1)
        targets.fill_(noise)
        targets[:, dominant_class, :, :] = confidence
        return targets

    def _make_predictions_matching(self, targets, temperature=0.1):
        """Create logits whose softmax closely matches targets."""
        # Use log of targets as logits (inverse of softmax approximation)
        log_targets = torch.log(torch.clamp(targets, min=1e-6))
        return log_targets / temperature

    def test_autodetect_soft_targets(self):
        """Float (B,C,H,W) targets should be detected as soft."""
        targets_soft = self._make_soft_targets()
        logits = torch.randn(
            self.B, self.num_classes, self.H, self.W, requires_grad=True
        )
        loss = self.loss_fn(logits, targets_soft)
        self.assertFalse(torch.isnan(loss))
        self.assertFalse(torch.isinf(loss))
        self.assertGreater(loss.item(), 0.0)

    def test_autodetect_hard_targets(self):
        """Long (B,H,W) targets should be detected as hard (backward compat)."""
        targets_hard = torch.randint(0, self.num_classes, (self.B, self.H, self.W))
        logits = torch.randn(
            self.B, self.num_classes, self.H, self.W, requires_grad=True
        )
        loss = self.loss_fn(logits, targets_hard)
        self.assertFalse(torch.isnan(loss))
        self.assertGreater(loss.item(), 0.0)

    def test_soft_matching_vs_wrong_prediction(self):
        """Matching predictions should produce lower loss than wrong ones."""
        targets = self._make_soft_targets(dominant_class=3, confidence=0.9)

        # Matching logits
        logits_match = self._make_predictions_matching(targets)
        loss_match = self.loss_fn(logits_match, targets)

        # Wrong logits (strongly favor a different class)
        wrong_targets = self._make_soft_targets(dominant_class=0, confidence=0.95)
        logits_wrong = self._make_predictions_matching(wrong_targets)
        loss_wrong = self.loss_fn(logits_wrong, targets)

        self.assertLess(loss_match.item(), loss_wrong.item())

    def test_soft_ignore_pixels(self):
        """Pixels with all-zero channels should be ignored."""
        targets = self._make_soft_targets()
        # Set half the pixels to all-zero (ignore)
        targets[:, :, : self.H // 2, :] = 0.0
        logits = torch.randn(self.B, self.num_classes, self.H, self.W)
        loss = self.loss_fn(logits, targets)
        self.assertFalse(torch.isnan(loss))
        self.assertFalse(torch.isinf(loss))

    def test_soft_all_ignore_returns_zero(self):
        """All pixels ignored should return zero loss."""
        targets = torch.zeros(self.B, self.num_classes, self.H, self.W)
        logits = torch.randn(self.B, self.num_classes, self.H, self.W)
        loss = self.loss_fn(logits, targets)
        self.assertAlmostEqual(loss.item(), 0.0, places=5)

    def test_soft_gradient_flow(self):
        """Verify gradients flow through soft loss path."""
        targets = self._make_soft_targets()
        logits = torch.randn(
            self.B, self.num_classes, self.H, self.W, requires_grad=True
        )
        loss = self.loss_fn(logits, targets)
        loss.backward()
        self.assertIsNotNone(logits.grad)
        self.assertFalse(torch.all(logits.grad == 0))

    def test_soft_vs_hard_onehot_equivalence(self):
        """Soft one-hot targets should produce similar loss as hard targets."""
        hard = torch.randint(0, self.num_classes, (self.B, self.H, self.W))
        # Convert to one-hot soft
        soft = F.one_hot(hard, self.num_classes).permute(0, 3, 1, 2).float()

        logits = torch.randn(self.B, self.num_classes, self.H, self.W)
        loss_hard = self.loss_fn(logits, hard)
        loss_soft = self.loss_fn(logits, soft)

        # Should be very close (not exact due to RCE clamping differences)
        self.assertAlmostEqual(loss_hard.item(), loss_soft.item(), delta=0.15)

    def test_jml_loss_soft_method(self):
        """Test _jml_loss_soft directly."""
        targets = self._make_soft_targets()
        valid_mask = targets.sum(dim=1) > 0
        logits = torch.randn(self.B, self.num_classes, self.H, self.W)
        probas = F.softmax(logits, dim=1)
        jml = self.loss_fn._jml_loss_soft(probas, targets, valid_mask)
        self.assertFalse(torch.isnan(jml))
        self.assertGreaterEqual(jml.item(), 0.0)
        self.assertLessEqual(jml.item(), 1.0)

    def test_sce_soft_method(self):
        """Test _sce_soft directly."""
        targets = self._make_soft_targets()
        valid_mask = targets.sum(dim=1) > 0
        logits = torch.randn(self.B, self.num_classes, self.H, self.W)
        probas = F.softmax(logits, dim=1)
        sce = self.loss_fn._sce_soft(logits, probas, targets, valid_mask)
        self.assertFalse(torch.isnan(sce))
        self.assertFalse(torch.isinf(sce))
        self.assertGreater(sce.item(), 0.0)

    def test_sce_soft_no_nan_with_small_targets(self):
        """RCE with small target values should not produce NaN."""
        targets = torch.full((self.B, self.num_classes, self.H, self.W), 1e-5)
        # Make it sum to ~1
        targets[:, 0, :, :] = 1.0 - 1e-5 * (self.num_classes - 1)
        valid_mask = torch.ones(self.B, self.H, self.W, dtype=torch.bool)
        logits = torch.randn(self.B, self.num_classes, self.H, self.W)
        probas = F.softmax(logits, dim=1)
        sce = self.loss_fn._sce_soft(logits, probas, targets, valid_mask)
        self.assertFalse(torch.isnan(sce))


class TestSoftLabelDatasetHelpers(unittest.TestCase):
    """Tests for dataset helper methods with soft labels."""

    def test_dominant_class_soft(self):
        """_get_dominant_class should use argmax for soft masks."""
        from pytorch_segmentation_models_trainer.dataset_loader.dataset import (
            RandomCropSegmentationDataset,
        )

        # Create a minimal mock to test the method
        ds = object.__new__(RandomCropSegmentationDataset)
        ds.soft_labels = True
        ds.n_classes = 7

        # Soft mask (H, W, C) with class 3 dominant
        mask = np.zeros((16, 16, 7), dtype=np.float32)
        mask[:, :, 3] = 0.9
        mask[:, :, 1] = 0.1
        result = ds._get_dominant_class(mask)
        self.assertEqual(result, 3)

    def test_dominant_class_soft_all_ignore(self):
        """All-zero soft mask should return -1."""
        from pytorch_segmentation_models_trainer.dataset_loader.dataset import (
            RandomCropSegmentationDataset,
        )

        ds = object.__new__(RandomCropSegmentationDataset)
        ds.soft_labels = True
        ds.n_classes = 7

        mask = np.zeros((16, 16, 7), dtype=np.float32)
        result = ds._get_dominant_class(mask)
        self.assertEqual(result, -1)

    def test_cutmix_soft_labels_shape(self):
        """CutMix should work with (H,W,C) soft masks."""
        from pytorch_segmentation_models_trainer.dataset_loader.dataset import (
            RandomCropSegmentationDataset,
        )

        ds = object.__new__(RandomCropSegmentationDataset)
        ds.cutmix_alpha = 1.0

        H, W, C = 64, 64, 7
        image1 = np.random.randint(0, 255, (H, W, 3), dtype=np.uint8)
        mask1 = np.random.rand(H, W, C).astype(np.float32)
        image2 = np.random.randint(0, 255, (H, W, 3), dtype=np.uint8)
        mask2 = np.random.rand(H, W, C).astype(np.float32)

        img_out, mask_out = ds._apply_cutmix(image1, mask1, image2, mask2)
        self.assertEqual(mask_out.shape, (H, W, C))
        self.assertEqual(mask_out.dtype, np.float32)

    def test_is_crop_valid_soft(self):
        """_is_crop_valid should check soft mask validity."""
        from pytorch_segmentation_models_trainer.dataset_loader.dataset import (
            RandomCropSegmentationDataset,
        )

        ds = object.__new__(RandomCropSegmentationDataset)
        ds.soft_labels = True
        ds.min_valid_ratio = 0.1

        image = np.ones((16, 16, 3), dtype=np.uint8) * 128

        # All-zero mask = invalid
        mask_invalid = np.zeros((16, 16, 7), dtype=np.float32)
        self.assertFalse(ds._is_crop_valid(image, mask_data=mask_invalid))

        # Mostly valid mask
        mask_valid = np.zeros((16, 16, 7), dtype=np.float32)
        mask_valid[:, :, 2] = 1.0
        self.assertTrue(ds._is_crop_valid(image, mask_data=mask_valid))

    def test_ensure_soft_mask_tensor_numpy(self):
        """_ensure_soft_mask_tensor converts numpy (H,W,C) to tensor (C,H,W)."""
        from pytorch_segmentation_models_trainer.dataset_loader.dataset import (
            RandomCropSegmentationDataset,
        )

        ds = object.__new__(RandomCropSegmentationDataset)
        ds.n_classes = 7

        mask_np = np.random.rand(32, 32, 7).astype(np.float32)
        result = ds._ensure_soft_mask_tensor(mask_np)
        self.assertIsInstance(result, torch.Tensor)
        self.assertEqual(result.shape, (7, 32, 32))
        self.assertEqual(result.dtype, torch.float32)

    def test_ensure_soft_mask_tensor_hwc(self):
        """_ensure_soft_mask_tensor converts (H,W,C) tensor to (C,H,W)."""
        from pytorch_segmentation_models_trainer.dataset_loader.dataset import (
            RandomCropSegmentationDataset,
        )

        ds = object.__new__(RandomCropSegmentationDataset)
        ds.n_classes = 7

        mask_tensor = torch.rand(32, 32, 7)
        result = ds._ensure_soft_mask_tensor(mask_tensor)
        self.assertEqual(result.shape, (7, 32, 32))

    def test_ensure_soft_mask_tensor_chw(self):
        """_ensure_soft_mask_tensor keeps (C,H,W) tensor as-is."""
        from pytorch_segmentation_models_trainer.dataset_loader.dataset import (
            RandomCropSegmentationDataset,
        )

        ds = object.__new__(RandomCropSegmentationDataset)
        ds.n_classes = 7

        mask_tensor = torch.rand(7, 32, 32)
        result = ds._ensure_soft_mask_tensor(mask_tensor)
        self.assertEqual(result.shape, (7, 32, 32))


class TestSoftLabelIntegration(unittest.TestCase):
    """Integration tests with temporary GeoTIFF files."""

    def setUp(self):
        self.num_classes = 7
        self.H, self.W = 64, 64
        self.tmpdir = tempfile.mkdtemp()

    def _create_geotiff(self, path, data, dtype="float32"):
        """Create a minimal GeoTIFF for testing."""
        import rasterio
        from rasterio.transform import from_bounds

        if data.ndim == 2:
            count = 1
            h, w = data.shape
        else:
            count, h, w = data.shape

        transform = from_bounds(0, 0, w, h, w, h)
        profile = {
            "driver": "GTiff",
            "dtype": dtype,
            "width": w,
            "height": h,
            "count": count,
            "crs": "EPSG:3857",
            "transform": transform,
        }
        with rasterio.open(path, "w", **profile) as dst:
            if data.ndim == 2:
                dst.write(data, 1)
            else:
                dst.write(data)

    def test_read_crop_soft_labels(self):
        """_read_crop should read 7-band float32 mask as (H,W,C)."""
        import rasterio
        from pytorch_segmentation_models_trainer.dataset_loader.dataset import (
            RandomCropSegmentationDataset,
        )

        # Create 7-band soft mask
        soft_data = np.random.rand(self.num_classes, self.H, self.W).astype(np.float32)
        # Normalize to sum to 1
        soft_data /= soft_data.sum(axis=0, keepdims=True)
        mask_path = os.path.join(self.tmpdir, "soft_mask.tif")
        self._create_geotiff(mask_path, soft_data, dtype="float32")

        # Create minimal dataset object
        ds = object.__new__(RandomCropSegmentationDataset)
        ds.soft_labels = True
        ds.crop_size = [32, 32]
        ds.file_cache_maxsize = 10
        ds.selected_bands = None

        # Use a simple dict cache for testing
        ds._file_cache = {}
        ds._file_cache[mask_path] = rasterio.open(mask_path)

        crop = ds._read_crop(mask_path, x=0, y=0, is_mask=True)
        self.assertEqual(crop.shape, (32, 32, self.num_classes))
        self.assertEqual(crop.dtype, np.float32)

        # Verify values are preserved
        np.testing.assert_allclose(crop[:, :, 0], soft_data[0, :32, :32], atol=1e-6)

        # Cleanup
        ds._file_cache[mask_path].close()

    def test_read_crop_hard_labels_unchanged(self):
        """_read_crop with soft_labels=False should still return (H,W) uint8."""
        import rasterio
        from pytorch_segmentation_models_trainer.dataset_loader.dataset import (
            RandomCropSegmentationDataset,
        )

        hard_data = np.random.randint(0, 7, (self.H, self.W)).astype(np.uint8)
        mask_path = os.path.join(self.tmpdir, "hard_mask.tif")
        self._create_geotiff(mask_path, hard_data, dtype="uint8")

        ds = object.__new__(RandomCropSegmentationDataset)
        ds.soft_labels = False
        ds.crop_size = [32, 32]
        ds.selected_bands = None
        ds._file_cache = {}
        ds._file_cache[mask_path] = rasterio.open(mask_path)

        crop = ds._read_crop(mask_path, x=0, y=0, is_mask=True)
        self.assertEqual(crop.shape, (32, 32))
        self.assertEqual(crop.dtype, np.uint8)

        ds._file_cache[mask_path].close()

    def tearDown(self):
        import shutil

        shutil.rmtree(self.tmpdir, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
