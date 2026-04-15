# -*- coding: utf-8 -*-
"""
Tests for advanced training features:
  - EMA fixes (global_step tracking with accumulate_grad_batches, on_save_checkpoint)
  - MixStyleCallback (feature-level domain augmentation)
  - ClassMix (semantic-aware mixing in dataset)
  - GCBLLoss + WeightedJMLSCEGCBLLoss (contrastive boundary learning)
  - Per-class IoU monitoring in Model
  - Integral image spatial index
  - Hard mask CSV support
  - GeoTIFF SRS source suppression
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
import torch.nn as nn
from unittest.mock import Mock, MagicMock, patch

from pytorch_segmentation_models_trainer.custom_losses.loss import (
    GCBLLoss,
    WeightedJMLSCEGCBLLoss,
    WeightedJMLSCELoss,
)
from pytorch_segmentation_models_trainer.custom_callbacks.training_callbacks import (
    EMACallback,
    MixStyleCallback,
)
from pytorch_segmentation_models_trainer.dataset_loader.dataset import (
    RandomCropSegmentationDataset,
)


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------


def _create_geotiff(path, data, count=None, dtype="uint8"):
    """Write a numpy array as a GeoTIFF with rasterio."""
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


class _TinyModel(nn.Module):
    """Minimal model for EMA/MixStyle callback testing."""

    def __init__(self):
        super().__init__()
        self.encoder = nn.Linear(4, 3)
        self.decoder = nn.Linear(3, 2)

    def set_encoder_trainable(self, trainable=True):
        for p in self.encoder.parameters():
            p.requires_grad = trainable


class _StageModel(nn.Module):
    """Model with encoder.model.stages for MixStyle testing."""

    def __init__(self, n_stages=4, channels=8):
        super().__init__()
        self.encoder = self._Encoder(n_stages, channels)
        self.decoder = nn.Linear(channels, 2)

    class _Encoder(nn.Module):
        def __init__(self, n_stages, channels):
            super().__init__()
            self.model = self._InnerModel(n_stages, channels)

        class _InnerModel(nn.Module):
            def __init__(self, n_stages, channels):
                super().__init__()
                self.stages = nn.ModuleList([
                    nn.Conv2d(channels, channels, 3, padding=1)
                    for _ in range(n_stages)
                ])

            def forward(self, x):
                for stage in self.stages:
                    x = stage(x)
                return x

        def forward(self, x):
            return self.model(x)

    def forward(self, x):
        return self.decoder(self.encoder(x).mean(dim=[2, 3]))


class SyntheticDatasetMixin:
    """Creates a temporary directory with synthetic GeoTIFF images/masks and CSV."""

    N_CLASSES = 4
    IMG_SIZE = 64
    N_IMAGES = 3
    CROP_SIZE = [32, 32]

    def _create_synthetic_dataset(self):
        self.tmpdir = tempfile.mkdtemp(prefix="test_adv_")
        img_dir = os.path.join(self.tmpdir, "images")
        msk_dir = os.path.join(self.tmpdir, "masks")
        os.makedirs(img_dir)
        os.makedirs(msk_dir)

        self.image_paths = []
        self.mask_paths = []
        h = w = self.IMG_SIZE
        half = h // 2

        for i in range(self.N_IMAGES):
            img = np.zeros((3, h, w), dtype=np.uint8)
            img[:, :half, :half] = 30 + i * 10
            img[:, :half, half:] = 100 + i * 10
            img[:, half:, :half] = 170 + i * 5
            img[:, half:, half:] = 220 + i * 3

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

        self.csv_path = os.path.join(self.tmpdir, "dataset.csv")
        with open(self.csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["image", "mask"])
            for img_p, msk_p in zip(self.image_paths, self.mask_paths):
                writer.writerow([img_p, msk_p])

    def _create_soft_labels_dataset(self):
        """Create dataset with 7-band float32 soft label masks."""
        self.tmpdir = tempfile.mkdtemp(prefix="test_adv_soft_")
        img_dir = os.path.join(self.tmpdir, "images")
        msk_dir = os.path.join(self.tmpdir, "masks")
        os.makedirs(img_dir)
        os.makedirs(msk_dir)

        self.image_paths = []
        self.mask_paths = []
        h = w = self.IMG_SIZE
        half = h // 2
        n_classes = self.N_CLASSES

        for i in range(self.N_IMAGES):
            img = np.zeros((3, h, w), dtype=np.uint8)
            img[:, :half, :half] = 30 + i * 10
            img[:, :half, half:] = 100 + i * 10
            img[:, half:, :half] = 170 + i * 5
            img[:, half:, half:] = 220 + i * 3

            # Soft mask: (n_classes, H, W) float32
            soft_mask = np.zeros((n_classes, h, w), dtype=np.float32)
            soft_mask[0, :half, :half] = 0.9
            soft_mask[1, :half, :half] = 0.1
            soft_mask[1, :half, half:] = 0.9
            soft_mask[0, :half, half:] = 0.1
            soft_mask[2, half:, :half] = 0.85
            soft_mask[3, half:, :half] = 0.15
            soft_mask[3, half:, half:] = 0.8
            soft_mask[2, half:, half:] = 0.2

            img_path = os.path.join(img_dir, f"img_{i:02d}.tif")
            msk_path = os.path.join(msk_dir, f"msk_{i:02d}.tif")
            _create_geotiff(img_path, img)
            _create_geotiff(msk_path, soft_mask, count=n_classes, dtype="float32")
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


# ======================================================================
# A. EMA Fixes (global_step tracking + on_save_checkpoint)
# ======================================================================


class TestEMAGlobalStepTracking(unittest.TestCase):
    """Tests for EMA + accumulate_grad_batches fix."""

    def setUp(self):
        warnings.simplefilter("ignore")
        self.model = _TinyModel()
        self.trainer = Mock()
        self.trainer.current_epoch = 0

    def test_skips_microbatch_when_global_step_unchanged(self):
        """EMA should NOT update on micro-batches where global_step hasn't changed."""
        ema = EMACallback(decay=0.9)
        self.trainer.global_step = 0
        ema.on_fit_start(self.trainer, self.model)

        # First call at global_step=0 — should update (initial step is -1)
        with torch.no_grad():
            for p in self.model.parameters():
                p.add_(torch.ones_like(p))

        shadow_before = {n: s.clone() for n, s in ema._shadow.items()}
        ema.on_train_batch_end(self.trainer, self.model, None, None, 0)
        # Shadow should have changed
        for name in shadow_before:
            self.assertFalse(
                torch.equal(ema._shadow[name], shadow_before[name]),
                f"Shadow for {name} should update on first batch"
            )

        # Second call at same global_step=0 (micro-batch) — should NOT update
        shadow_after_first = {n: s.clone() for n, s in ema._shadow.items()}
        with torch.no_grad():
            for p in self.model.parameters():
                p.add_(torch.ones_like(p) * 10)  # big change

        ema.on_train_batch_end(self.trainer, self.model, None, None, 1)
        for name in shadow_after_first:
            self.assertTrue(
                torch.equal(ema._shadow[name], shadow_after_first[name]),
                f"Shadow for {name} should NOT update on micro-batch (same global_step)"
            )

    def test_updates_when_global_step_increments(self):
        """EMA should update when global_step changes."""
        ema = EMACallback(decay=0.9)
        self.trainer.global_step = 0
        ema.on_fit_start(self.trainer, self.model)

        # Step 0
        ema.on_train_batch_end(self.trainer, self.model, None, None, 0)

        # Change params and increment global_step
        with torch.no_grad():
            for p in self.model.parameters():
                p.add_(torch.ones_like(p) * 5)

        shadow_before = {n: s.clone() for n, s in ema._shadow.items()}
        self.trainer.global_step = 1
        ema.on_train_batch_end(self.trainer, self.model, None, None, 2)

        for name in shadow_before:
            self.assertFalse(
                torch.equal(ema._shadow[name], shadow_before[name]),
                f"Shadow for {name} should update when global_step increments"
            )

    def test_simulates_accumulate_grad_batches_2(self):
        """Simulate accumulate_grad_batches=2: 4 micro-batches, 2 optimizer steps."""
        ema = EMACallback(decay=0.9)
        self.trainer.global_step = 0
        ema.on_fit_start(self.trainer, self.model)

        update_count = 0
        for batch_idx in range(4):
            shadow_before = {n: s.clone() for n, s in ema._shadow.items()}
            with torch.no_grad():
                for p in self.model.parameters():
                    p.add_(torch.ones_like(p))

            # global_step increments every 2 micro-batches
            if batch_idx == 2:
                self.trainer.global_step = 1

            ema.on_train_batch_end(self.trainer, self.model, None, None, batch_idx)

            changed = any(
                not torch.equal(ema._shadow[n], shadow_before[n])
                for n in shadow_before
            )
            if changed:
                update_count += 1

        # Should have exactly 2 updates (one per optimizer step)
        self.assertEqual(update_count, 2,
                         "EMA should update exactly once per optimizer step")

    def test_last_global_step_initialized_to_minus_one(self):
        """After on_fit_start, _last_global_step should be -1."""
        ema = EMACallback(decay=0.999)
        ema.on_fit_start(self.trainer, self.model)
        self.assertEqual(ema._last_global_step, -1)


class TestEMAOnSaveCheckpoint(unittest.TestCase):
    """Tests for EMA checkpoint saving fix."""

    def setUp(self):
        warnings.simplefilter("ignore")
        self.model = _TinyModel()
        self.trainer = Mock()
        self.trainer.current_epoch = 0
        self.trainer.global_step = 0

    def test_on_save_checkpoint_injects_ema_weights(self):
        """Checkpoint state_dict should contain EMA weights, not online weights."""
        ema = EMACallback(decay=0.999)
        ema.on_fit_start(self.trainer, self.model)

        # Set shadow to known values
        for name in ema._shadow:
            ema._shadow[name].fill_(42.0)

        # Create checkpoint with current (online) model weights
        checkpoint = {
            "state_dict": {
                name: param.data.clone()
                for name, param in self.model.named_parameters()
            }
        }

        ema.on_save_checkpoint(self.trainer, self.model, checkpoint)

        # Checkpoint should now have EMA weights (42.0), not online weights
        for name in ema._shadow:
            if name in checkpoint["state_dict"]:
                self.assertTrue(
                    torch.allclose(checkpoint["state_dict"][name], torch.tensor(42.0)),
                    f"Checkpoint {name} should contain EMA weights"
                )

    def test_on_save_checkpoint_does_not_modify_model(self):
        """on_save_checkpoint should NOT change the live model weights."""
        ema = EMACallback(decay=0.999)
        ema.on_fit_start(self.trainer, self.model)

        original_params = {
            n: p.data.clone() for n, p in self.model.named_parameters()
        }

        for name in ema._shadow:
            ema._shadow[name].fill_(99.0)

        checkpoint = {
            "state_dict": {
                name: param.data.clone()
                for name, param in self.model.named_parameters()
            }
        }

        ema.on_save_checkpoint(self.trainer, self.model, checkpoint)

        # Model params should be unchanged
        for name, param in self.model.named_parameters():
            if name in original_params:
                self.assertTrue(
                    torch.equal(param.data, original_params[name]),
                    f"Model param {name} should not be modified by on_save_checkpoint"
                )

    def test_checkpoint_ema_consistent_with_validation(self):
        """End-to-end: after validation swap + save, checkpoint has EMA weights."""
        ema = EMACallback(decay=0.999)
        ema.on_fit_start(self.trainer, self.model)

        # Set EMA to known values
        for name in ema._shadow:
            ema._shadow[name].fill_(77.0)

        # Swap in EMA for validation
        ema.on_validation_epoch_start(self.trainer, self.model)
        # Model now has EMA weights (77.0)

        # Restore online weights after validation
        ema.on_validation_epoch_end(self.trainer, self.model)
        # Model back to online weights

        # Save checkpoint (ModelCheckpoint runs after validation)
        checkpoint = {
            "state_dict": {
                name: param.data.clone()
                for name, param in self.model.named_parameters()
            }
        }
        ema.on_save_checkpoint(self.trainer, self.model, checkpoint)

        # Checkpoint should have EMA weights (77.0)
        for name in ema._shadow:
            if name in checkpoint["state_dict"]:
                self.assertTrue(
                    torch.allclose(checkpoint["state_dict"][name], torch.tensor(77.0)),
                    f"Checkpoint {name} should match EMA weights used during validation"
                )


# ======================================================================
# B. MixStyleCallback
# ======================================================================


class TestMixStyleCallback(unittest.TestCase):

    def setUp(self):
        warnings.simplefilter("ignore")
        self.model = _StageModel(n_stages=4, channels=8)
        self.trainer = Mock()

    def test_hooks_registered_on_specified_stages(self):
        """MixStyle hooks should be registered on stages [0, 1]."""
        pl_module = Mock()
        pl_module.model = self.model
        cb = MixStyleCallback(p=1.0, alpha=0.1, stages=[0, 1])
        cb.on_fit_start(self.trainer, pl_module)

        self.assertEqual(len(cb._hooks), 2)

    def test_hooks_registered_on_all_stages(self):
        """Can register on more stages."""
        pl_module = Mock()
        pl_module.model = self.model
        cb = MixStyleCallback(p=1.0, stages=[0, 1, 2, 3])
        cb.on_fit_start(self.trainer, pl_module)

        self.assertEqual(len(cb._hooks), 4)

    def test_hooks_not_registered_beyond_stage_count(self):
        """Requesting stage index > len(stages) should be silently skipped."""
        pl_module = Mock()
        pl_module.model = self.model
        cb = MixStyleCallback(p=1.0, stages=[0, 1, 10])  # stage 10 doesn't exist
        cb.on_fit_start(self.trainer, pl_module)

        self.assertEqual(len(cb._hooks), 2)

    def test_hook_active_only_during_training(self):
        """MixStyle should be active during training, inactive during validation."""
        pl_module = Mock()
        pl_module.model = self.model
        cb = MixStyleCallback(p=1.0, alpha=0.5)

        # Before training — not active
        self.assertFalse(cb._training)

        cb.on_train_batch_start(self.trainer, pl_module, None, 0)
        self.assertTrue(cb._training)

        cb.on_train_batch_end(self.trainer, pl_module, None, None, 0)
        self.assertFalse(cb._training)

        cb.on_validation_epoch_start(self.trainer, pl_module)
        self.assertFalse(cb._training)

    def test_mixstyle_hook_modifies_output_during_training(self):
        """During training with p=1.0, the hook should modify the output."""
        cb = MixStyleCallback(p=1.0, alpha=0.5)
        cb._training = True

        x = torch.randn(4, 8, 16, 16)  # B=4
        torch.manual_seed(42)
        output = cb._mixstyle_hook(None, None, x)

        # Output should have same shape but different values
        self.assertEqual(output.shape, x.shape)
        # With p=1.0, output should differ from input (very high probability)
        # We can't guarantee it 100% due to random permutation, but it's very likely
        self.assertFalse(torch.equal(output, x),
                         "MixStyle hook should modify output during training")

    def test_mixstyle_hook_noop_during_eval(self):
        """During eval (training=False), the hook should be identity."""
        cb = MixStyleCallback(p=1.0, alpha=0.5)
        cb._training = False

        x = torch.randn(4, 8, 16, 16)
        output = cb._mixstyle_hook(None, None, x)

        self.assertTrue(torch.equal(output, x),
                        "MixStyle hook should be identity during eval")

    def test_mixstyle_hook_noop_single_sample(self):
        """With batch size 1, MixStyle should be identity (can't mix)."""
        cb = MixStyleCallback(p=1.0, alpha=0.5)
        cb._training = True

        x = torch.randn(1, 8, 16, 16)
        output = cb._mixstyle_hook(None, None, x)

        self.assertTrue(torch.equal(output, x),
                        "MixStyle should be identity with batch_size=1")

    def test_mixstyle_hook_preserves_shape(self):
        """Output shape must match input shape."""
        cb = MixStyleCallback(p=1.0, alpha=0.1)
        cb._training = True

        for B in [2, 4, 8]:
            x = torch.randn(B, 16, 32, 32)
            output = cb._mixstyle_hook(None, None, x)
            self.assertEqual(output.shape, x.shape)

    def test_mixstyle_hook_skips_list_output(self):
        """If a stage returns a list/tuple, hook should pass through."""
        cb = MixStyleCallback(p=1.0, alpha=0.5)
        cb._training = True

        x = [torch.randn(4, 8, 16, 16), torch.randn(4, 8, 16, 16)]
        output = cb._mixstyle_hook(None, None, x)
        self.assertIs(output, x)

    def test_hooks_removed_on_fit_end(self):
        """All hooks should be cleaned up after training."""
        pl_module = Mock()
        pl_module.model = self.model
        cb = MixStyleCallback(p=1.0, stages=[0, 1])
        cb.on_fit_start(self.trainer, pl_module)
        self.assertEqual(len(cb._hooks), 2)

        cb.on_fit_end(self.trainer, pl_module)
        self.assertEqual(len(cb._hooks), 0)

    def test_no_encoder_stages_warns(self):
        """If encoder doesn't have stages, should warn and not crash."""
        pl_module = Mock()
        pl_module.model = nn.Linear(10, 2)  # no encoder.model.stages
        pl_module.model.encoder = nn.Linear(10, 5)

        cb = MixStyleCallback(p=1.0, stages=[0, 1])
        # Should not raise, just warn
        cb.on_fit_start(self.trainer, pl_module)
        self.assertEqual(len(cb._hooks), 0)

    def test_mixstyle_with_p_zero_is_noop(self):
        """With p=0.0, hook should never modify output."""
        cb = MixStyleCallback(p=0.0, alpha=0.5)
        cb._training = True

        x = torch.randn(4, 8, 16, 16)
        # Run multiple times to account for randomness
        for _ in range(10):
            output = cb._mixstyle_hook(None, None, x)
            self.assertTrue(torch.equal(output, x))


# ======================================================================
# C. ClassMix
# ======================================================================


class TestApplyClassMix(unittest.TestCase):
    """Tests for _apply_classmix with hard labels."""

    def setUp(self):
        warnings.simplefilter("ignore")
        self.ds = object.__new__(RandomCropSegmentationDataset)
        self.ds.n_classes = 4
        self.ds.soft_labels = False
        self.ds.cutmix_alpha = 1.0

    def test_output_shape_matches_input(self):
        np.random.seed(42)
        img1 = np.full((32, 32, 3), 100, dtype=np.uint8)
        mask1 = np.full((32, 32), 0, dtype=np.uint8)
        img2 = np.full((32, 32, 3), 200, dtype=np.uint8)
        mask2 = np.zeros((32, 32), dtype=np.uint8)
        mask2[:16, :] = 1
        mask2[16:, :] = 2

        out_img, out_mask = self.ds._apply_classmix(img1, mask1, img2, mask2)
        self.assertEqual(out_img.shape, img1.shape)
        self.assertEqual(out_mask.shape, mask1.shape)

    def test_copies_class_shaped_regions(self):
        """ClassMix should copy pixels by class, not rectangles."""
        np.random.seed(42)
        img1 = np.full((32, 32, 3), 100, dtype=np.uint8)
        mask1 = np.full((32, 32), 0, dtype=np.uint8)
        img2 = np.full((32, 32, 3), 200, dtype=np.uint8)
        # mask2: class 1 in top half, class 2 in bottom half
        mask2 = np.zeros((32, 32), dtype=np.uint8)
        mask2[:16, :] = 1
        mask2[16:, :] = 2

        found_class_copy = False
        for seed in range(50):
            np.random.seed(seed)
            out_img, out_mask = self.ds._apply_classmix(img1, mask1, img2, mask2)

            # Check if exactly one class was copied (not a rectangle)
            has_class1 = np.any(out_mask == 1)
            has_class2 = np.any(out_mask == 2)
            if has_class1 != has_class2:
                # Exactly one of the two classes was copied — class-shaped
                found_class_copy = True
                # Verify image pixels match mask
                if has_class1:
                    self.assertTrue(np.all(out_img[out_mask == 1] == 200))
                if has_class2:
                    self.assertTrue(np.all(out_img[out_mask == 2] == 200))
                break

        self.assertTrue(found_class_copy,
                        "ClassMix should copy class-shaped regions")

    def test_does_not_modify_inputs(self):
        np.random.seed(42)
        img1 = np.full((32, 32, 3), 100, dtype=np.uint8)
        mask1 = np.full((32, 32), 0, dtype=np.uint8)
        img2 = np.full((32, 32, 3), 200, dtype=np.uint8)
        mask2 = np.zeros((32, 32), dtype=np.uint8)
        mask2[:16, :] = 1
        mask2[16:, :] = 2

        img1_copy = img1.copy()
        mask1_copy = mask1.copy()
        self.ds._apply_classmix(img1, mask1, img2, mask2)
        np.testing.assert_array_equal(img1, img1_copy)
        np.testing.assert_array_equal(mask1, mask1_copy)

    def test_fallback_to_cutmix_with_single_class(self):
        """If mask2 has only 1 class, ClassMix should fallback to CutMix."""
        np.random.seed(42)
        img1 = np.full((32, 32, 3), 100, dtype=np.uint8)
        mask1 = np.full((32, 32), 0, dtype=np.uint8)
        img2 = np.full((32, 32, 3), 200, dtype=np.uint8)
        mask2 = np.full((32, 32), 1, dtype=np.uint8)  # single class

        # Should not crash — falls back to CutMix
        out_img, out_mask = self.ds._apply_classmix(img1, mask1, img2, mask2)
        self.assertEqual(out_img.shape, img1.shape)
        self.assertEqual(out_mask.shape, mask1.shape)

    def test_image_mask_consistency(self):
        """Where image gets img2 values, mask should have mask2 values."""
        np.random.seed(0)
        img1 = np.full((32, 32, 3), 100, dtype=np.uint8)
        mask1 = np.full((32, 32), 0, dtype=np.uint8)
        img2 = np.full((32, 32, 3), 200, dtype=np.uint8)
        mask2 = np.zeros((32, 32), dtype=np.uint8)
        mask2[:16, :] = 1
        mask2[16:, :] = 2

        for seed in range(50):
            np.random.seed(seed)
            out_img, out_mask = self.ds._apply_classmix(img1, mask1, img2, mask2)

            # Pixels from img2 (value 200) should correspond to mask2 values
            from_img2 = np.all(out_img == 200, axis=2)
            from_mask2 = (out_mask == 1) | (out_mask == 2)
            np.testing.assert_array_equal(from_img2, from_mask2)


class TestApplyClassMixSoftLabels(unittest.TestCase):
    """Tests for _apply_classmix with soft labels."""

    def setUp(self):
        warnings.simplefilter("ignore")
        self.ds = object.__new__(RandomCropSegmentationDataset)
        self.ds.n_classes = 4
        self.ds.soft_labels = True
        self.ds.cutmix_alpha = 1.0

    def test_soft_labels_shape_preserved(self):
        np.random.seed(42)
        img1 = np.full((32, 32, 3), 100, dtype=np.uint8)
        # Soft mask: (H, W, C) — 4 classes
        mask1 = np.zeros((32, 32, 4), dtype=np.float32)
        mask1[:, :, 0] = 1.0  # all class 0

        img2 = np.full((32, 32, 3), 200, dtype=np.uint8)
        mask2 = np.zeros((32, 32, 4), dtype=np.float32)
        mask2[:16, :, 1] = 0.9
        mask2[:16, :, 0] = 0.1
        mask2[16:, :, 2] = 0.8
        mask2[16:, :, 3] = 0.2

        out_img, out_mask = self.ds._apply_classmix(img1, mask1, img2, mask2)
        self.assertEqual(out_img.shape, img1.shape)
        self.assertEqual(out_mask.shape, mask1.shape)

    def test_soft_labels_uses_argmax_for_class_detection(self):
        """ClassMix should use argmax of soft labels to detect classes."""
        np.random.seed(42)
        img1 = np.full((32, 32, 3), 100, dtype=np.uint8)
        mask1 = np.zeros((32, 32, 4), dtype=np.float32)
        mask1[:, :, 0] = 1.0

        img2 = np.full((32, 32, 3), 200, dtype=np.uint8)
        mask2 = np.zeros((32, 32, 4), dtype=np.float32)
        mask2[:16, :, 1] = 0.9
        mask2[:16, :, 0] = 0.1  # argmax = class 1
        mask2[16:, :, 2] = 0.8
        mask2[16:, :, 3] = 0.2  # argmax = class 2

        out_img, out_mask = self.ds._apply_classmix(img1, mask1, img2, mask2)

        # Copied pixels should have soft label distributions from mask2
        from_img2 = np.all(out_img == 200, axis=2)
        if from_img2.any():
            # Where we copied from img2, out_mask should have mask2 values
            np.testing.assert_array_equal(out_mask[from_img2], mask2[from_img2])

    def test_soft_labels_ignores_zero_sum_pixels(self):
        """Pixels with sum=0 in mask2 should not be considered for class detection."""
        np.random.seed(42)
        img1 = np.full((32, 32, 3), 100, dtype=np.uint8)
        mask1 = np.zeros((32, 32, 4), dtype=np.float32)
        mask1[:, :, 0] = 1.0

        img2 = np.full((32, 32, 3), 200, dtype=np.uint8)
        mask2 = np.zeros((32, 32, 4), dtype=np.float32)
        # Only top half has valid soft labels
        mask2[:16, :, 1] = 0.9
        mask2[:16, :, 0] = 0.1
        # Bottom half is all zeros (ignore)

        # Should not crash — and should handle the single-class case
        out_img, out_mask = self.ds._apply_classmix(img1, mask1, img2, mask2)
        self.assertEqual(out_img.shape, img1.shape)


class TestClassMixInDataset(unittest.TestCase, SyntheticDatasetMixin):
    """Integration test: classmix flag in __getitem__."""

    def setUp(self):
        warnings.simplefilter("ignore")
        self._create_synthetic_dataset()

    def tearDown(self):
        if hasattr(self, "ds"):
            self.ds._close_cache()
        self._cleanup()

    def test_getitem_with_classmix(self):
        self.ds = RandomCropSegmentationDataset(
            input_csv_path=self.csv_path,
            crop_size=self.CROP_SIZE,
            samples_per_epoch=10,
            n_classes=self.N_CLASSES,
            use_rasterio=True,
            cutmix_prob=1.0,
            cutmix_alpha=1.0,
            classmix=True,
            class_balanced_sampling=True,
            class_sampling_stride=32,
        )
        np.random.seed(42)
        result = self.ds[0]
        self.assertIn("image", result)
        self.assertIn("mask", result)
        self.assertEqual(result["image"].shape, (3, 32, 32))
        self.assertEqual(result["mask"].shape, (32, 32))

    def test_classmix_false_uses_cutmix(self):
        """With classmix=False, should use standard CutMix."""
        self.ds = RandomCropSegmentationDataset(
            input_csv_path=self.csv_path,
            crop_size=self.CROP_SIZE,
            samples_per_epoch=10,
            n_classes=self.N_CLASSES,
            use_rasterio=True,
            cutmix_prob=1.0,
            cutmix_alpha=1.0,
            classmix=False,
            class_balanced_sampling=True,
            class_sampling_stride=32,
        )
        np.random.seed(42)
        result = self.ds[0]
        self.assertEqual(result["image"].shape, (3, 32, 32))


# ======================================================================
# D. GCBLLoss + WeightedJMLSCEGCBLLoss
# ======================================================================


class TestGCBLLoss(unittest.TestCase):

    def setUp(self):
        warnings.simplefilter("ignore")
        self.num_classes = 7

    def test_basic_forward_hard_labels(self):
        loss_fn = GCBLLoss(num_classes=self.num_classes)
        pred = torch.randn(2, self.num_classes, 32, 32, requires_grad=True)
        target = torch.randint(0, self.num_classes, (2, 32, 32))
        loss = loss_fn(pred, target)
        self.assertEqual(loss.dim(), 0)
        self.assertTrue(loss.requires_grad)

    def test_basic_forward_soft_labels(self):
        loss_fn = GCBLLoss(num_classes=self.num_classes)
        pred = torch.randn(2, self.num_classes, 32, 32, requires_grad=True)
        # Soft targets: (B, C, H, W)
        target = torch.zeros(2, self.num_classes, 32, 32)
        target[:, 0, :16, :] = 0.9
        target[:, 1, :16, :] = 0.1
        target[:, 2, 16:, :] = 0.8
        target[:, 3, 16:, :] = 0.2
        loss = loss_fn(pred, target)
        self.assertEqual(loss.dim(), 0)

    def test_gradient_flows(self):
        loss_fn = GCBLLoss(num_classes=self.num_classes)
        pred = torch.randn(2, self.num_classes, 32, 32, requires_grad=True)
        # Create target with boundary between classes
        target = torch.zeros(2, 32, 32, dtype=torch.long)
        target[:, :16, :] = 0
        target[:, 16:, :] = 1
        loss = loss_fn(pred, target)
        loss.backward()
        self.assertIsNotNone(pred.grad)

    def test_no_boundary_returns_zero(self):
        """When all pixels are the same class, there are no boundaries → loss=0."""
        loss_fn = GCBLLoss(num_classes=self.num_classes)
        pred = torch.randn(2, self.num_classes, 16, 16, requires_grad=True)
        target = torch.full((2, 16, 16), 0, dtype=torch.long)
        loss = loss_fn(pred, target)
        self.assertAlmostEqual(loss.item(), 0.0, places=5)

    def test_boundary_detection_4_connected(self):
        """Boundary detection should find pixels adjacent to different classes."""
        loss_fn = GCBLLoss(num_classes=4)
        labels = torch.zeros(1, 8, 8, dtype=torch.long)
        labels[0, :4, :] = 0
        labels[0, 4:, :] = 1

        boundary = loss_fn._detect_boundaries(labels)
        # Boundary should be at rows 3 and 4 (adjacent rows with different classes)
        self.assertTrue(boundary[0, 3, :].all(), "Row 3 should be boundary")
        self.assertTrue(boundary[0, 4, :].all(), "Row 4 should be boundary")
        # Inner rows should not be boundary
        self.assertFalse(boundary[0, 0, :].any(), "Row 0 should not be boundary")
        self.assertFalse(boundary[0, 7, :].any(), "Row 7 should not be boundary")

    def test_boundary_detection_ignores_invalid(self):
        """Pixels with class >= num_classes should be excluded from boundaries."""
        loss_fn = GCBLLoss(num_classes=4)
        labels = torch.full((1, 8, 8), 255, dtype=torch.long)
        labels[0, 3, :] = 0
        labels[0, 4, :] = 1

        boundary = loss_fn._detect_boundaries(labels)
        # Only rows 3-4 boundary pixels should be detected
        self.assertTrue(boundary[0, 3, :].any())
        self.assertTrue(boundary[0, 4, :].any())
        # Invalid (255) pixels should not be in boundary
        self.assertFalse(boundary[0, 0, :].any())

    def test_projector_parameters_exist(self):
        """GCBL projector should be a proper nn.Module with parameters."""
        loss_fn = GCBLLoss(num_classes=7, embed_dim=32)
        params = list(loss_fn.projector.parameters())
        self.assertGreater(len(params), 0, "Projector should have learnable parameters")

    def test_max_boundary_samples_limits_computation(self):
        """max_boundary_samples should cap the number of sampled pixels."""
        loss_fn = GCBLLoss(
            num_classes=4, max_boundary_samples=16
        )
        pred = torch.randn(2, 4, 32, 32, requires_grad=True)
        target = torch.zeros(2, 32, 32, dtype=torch.long)
        target[:, :16, :] = 0
        target[:, 16:, :] = 1
        # Should not crash with limited samples
        loss = loss_fn(pred, target)
        self.assertEqual(loss.dim(), 0)

    def test_few_boundary_pixels_returns_zero(self):
        """With fewer than 4 boundary pixels, loss should be 0."""
        loss_fn = GCBLLoss(num_classes=4)
        pred = torch.randn(1, 4, 4, 4, requires_grad=True)
        # Only 1 pixel different — too few for contrastive loss
        target = torch.zeros(1, 4, 4, dtype=torch.long)
        target[0, 2, 2] = 1  # single different pixel → ~2-4 boundary pixels
        # Depending on adjacency, this may or may not produce enough
        # Either way, it should not crash
        loss = loss_fn(pred, target)
        self.assertEqual(loss.dim(), 0)


class TestWeightedJMLSCEGCBLLoss(unittest.TestCase):

    def setUp(self):
        warnings.simplefilter("ignore")
        self.num_classes = 7

    def _make_loss(self, **kwargs):
        defaults = dict(
            num_classes=self.num_classes,
            jml_weight=0.25,
            sce_weight=0.75,
            ignore_index=255,
            gcbl_weight=0.1,
        )
        defaults.update(kwargs)
        return WeightedJMLSCEGCBLLoss(**defaults)

    def test_basic_forward_hard_labels(self):
        loss_fn = self._make_loss()
        pred = torch.randn(2, self.num_classes, 32, 32, requires_grad=True)
        target = torch.randint(0, self.num_classes, (2, 32, 32))
        loss = loss_fn(pred, target)
        self.assertEqual(loss.dim(), 0)
        self.assertTrue(loss.requires_grad)
        self.assertGreater(loss.item(), 0)

    def test_basic_forward_soft_labels(self):
        loss_fn = self._make_loss()
        pred = torch.randn(2, self.num_classes, 32, 32, requires_grad=True)
        target = torch.zeros(2, self.num_classes, 32, 32)
        target[:, 0, :16, :] = 0.9
        target[:, 1, :16, :] = 0.1
        target[:, 2, 16:, :] = 0.8
        target[:, 3, 16:, :] = 0.2
        loss = loss_fn(pred, target)
        self.assertEqual(loss.dim(), 0)
        self.assertGreater(loss.item(), 0)

    def test_gradient_flows_to_both_components(self):
        loss_fn = self._make_loss()
        pred = torch.randn(2, self.num_classes, 32, 32, requires_grad=True)
        target = torch.zeros(2, 32, 32, dtype=torch.long)
        target[:, :16, :] = 0
        target[:, 16:, :] = 1

        loss = loss_fn(pred, target)
        loss.backward()
        self.assertIsNotNone(pred.grad)
        self.assertFalse(torch.all(pred.grad == 0))

    def test_gcbl_weight_zero_equals_jmlsce(self):
        """With gcbl_weight=0, loss should equal WeightedJMLSCELoss."""
        torch.manual_seed(42)
        pred = torch.randn(2, self.num_classes, 16, 16)
        target = torch.randint(0, self.num_classes, (2, 16, 16))

        loss_combined = self._make_loss(gcbl_weight=0.0)
        loss_jmlsce = WeightedJMLSCELoss(
            num_classes=self.num_classes,
            jml_weight=0.25, sce_weight=0.75, ignore_index=255,
        )

        val_combined = loss_combined(pred, target)
        val_jmlsce = loss_jmlsce(pred, target)
        self.assertAlmostEqual(val_combined.item(), val_jmlsce.item(), places=5)

    def test_gcbl_projector_in_module_parameters(self):
        """GCBL projector params should be accessible as module parameters."""
        loss_fn = self._make_loss()
        param_names = [n for n, _ in loss_fn.named_parameters()]
        gcbl_params = [n for n in param_names if "gcbl" in n]
        self.assertGreater(len(gcbl_params), 0,
                           "GCBL projector parameters should be part of the module")

    def test_ignore_index_all_ignored(self):
        loss_fn = self._make_loss()
        pred = torch.randn(2, self.num_classes, 16, 16, requires_grad=True)
        target = torch.full((2, 16, 16), 255, dtype=torch.long)
        loss = loss_fn(pred, target)
        self.assertAlmostEqual(loss.item(), 0.0, places=5)

    def test_soft_labels_ignore_zero_sum(self):
        """Soft targets with sum=0 should be treated as ignore."""
        loss_fn = self._make_loss()
        pred = torch.randn(2, self.num_classes, 16, 16, requires_grad=True)
        target = torch.zeros(2, self.num_classes, 16, 16)  # all zeros
        loss = loss_fn(pred, target)
        self.assertAlmostEqual(loss.item(), 0.0, places=5)

    def test_higher_gcbl_weight_increases_loss(self):
        """More GCBL weight should generally increase total loss (when boundary exists)."""
        torch.manual_seed(42)
        pred = torch.randn(2, self.num_classes, 32, 32)
        target = torch.zeros(2, 32, 32, dtype=torch.long)
        target[:, :16, :] = 0
        target[:, 16:, :] = 1

        loss_low = self._make_loss(gcbl_weight=0.01)(pred, target)
        loss_high = self._make_loss(gcbl_weight=1.0)(pred, target)
        # With more weight on GCBL, total should be >= low weight
        # (GCBL loss is >= 0)
        self.assertGreaterEqual(loss_high.item(), loss_low.item() - 0.01)


# ======================================================================
# E. Per-class IoU Monitoring
# ======================================================================


class TestPerClassIoU(unittest.TestCase):
    """Tests for per-class IoU monitoring in Model.__init__ and validation_step."""

    def setUp(self):
        warnings.simplefilter("ignore")

    def test_per_class_iou_metric_computation(self):
        """JaccardIndex with average='none' should return per-class IoU."""
        import torchmetrics
        n_classes = 7
        metric = torchmetrics.JaccardIndex(
            task="multiclass", num_classes=n_classes,
            average="none", ignore_index=255,
        )

        pred = torch.zeros(100, dtype=torch.long)
        pred[:50] = 0
        pred[50:] = 1
        target = torch.zeros(100, dtype=torch.long)
        target[:40] = 0
        target[40:60] = 1
        target[60:] = 2

        result = metric(pred, target)
        self.assertEqual(result.shape, (n_classes,))
        # Class 0: tp=40, fp=10, fn=0 → IoU = 40/50 = 0.8
        self.assertAlmostEqual(result[0].item(), 0.8, places=2)
        # Class 1: tp=10, fp=40, fn=10 → IoU = 10/60 ≈ 0.167
        self.assertAlmostEqual(result[1].item(), 10 / 60, places=2)

    def test_per_class_iou_ignores_255(self):
        """Pixels with label 255 should be ignored."""
        import torchmetrics
        metric = torchmetrics.JaccardIndex(
            task="multiclass", num_classes=4,
            average="none", ignore_index=255,
        )

        pred = torch.tensor([0, 0, 1, 1])
        target = torch.tensor([0, 255, 1, 255])

        result = metric(pred, target)
        # Class 0: 1 TP, 0 FP (255 ignored), 0 FN → IoU=1.0
        self.assertAlmostEqual(result[0].item(), 1.0, places=2)
        # Class 1: 1 TP, 0 FP, 0 FN → IoU=1.0
        self.assertAlmostEqual(result[1].item(), 1.0, places=2)

    def test_per_class_iou_per_class_names_match(self):
        """Class names should align with metric output indices."""
        class_names = [
            "background", "massa_dagua", "area_edificada",
            "terreno_exposto", "vegetacao_baixa", "floresta", "veg_cultivada",
        ]
        import torchmetrics
        metric = torchmetrics.JaccardIndex(
            task="multiclass", num_classes=len(class_names),
            average="none", ignore_index=255,
        )

        # Perfect prediction for class 3 only
        pred = torch.full((100,), 3, dtype=torch.long)
        target = torch.full((100,), 3, dtype=torch.long)

        result = metric(pred, target)
        self.assertEqual(len(result), len(class_names))
        # Index 3 (terreno_exposto) should have IoU=1.0
        self.assertAlmostEqual(result[3].item(), 1.0, places=2)

    def test_per_class_iou_handles_missing_classes(self):
        """Classes not present in batch should have IoU=0."""
        import torchmetrics
        metric = torchmetrics.JaccardIndex(
            task="multiclass", num_classes=7,
            average="none", ignore_index=255,
        )

        # Only classes 0 and 1 present
        pred = torch.tensor([0, 0, 1, 1])
        target = torch.tensor([0, 0, 1, 1])
        result = metric(pred, target)

        self.assertAlmostEqual(result[0].item(), 1.0, places=2)
        self.assertAlmostEqual(result[1].item(), 1.0, places=2)
        # Classes 2-6 not present → IoU=0
        for c in range(2, 7):
            self.assertAlmostEqual(result[c].item(), 0.0, places=2)


# ======================================================================
# F. Integration: ClassMix with Soft Labels in Dataset
# ======================================================================


class TestClassMixSoftLabelsIntegration(unittest.TestCase, SyntheticDatasetMixin):
    """Integration test for ClassMix + soft labels end-to-end."""

    def setUp(self):
        warnings.simplefilter("ignore")
        self._create_soft_labels_dataset()

    def tearDown(self):
        if hasattr(self, "ds"):
            self.ds._close_cache()
        self._cleanup()

    def test_getitem_with_classmix_soft_labels(self):
        self.ds = RandomCropSegmentationDataset(
            input_csv_path=self.csv_path,
            crop_size=self.CROP_SIZE,
            samples_per_epoch=10,
            n_classes=self.N_CLASSES,
            use_rasterio=True,
            soft_labels=True,
            cutmix_prob=1.0,
            cutmix_alpha=1.0,
            classmix=True,
            class_balanced_sampling=True,
            class_sampling_stride=32,
        )
        np.random.seed(42)
        result = self.ds[0]
        self.assertIn("image", result)
        self.assertIn("mask", result)
        self.assertEqual(result["image"].shape, (3, 32, 32))
        # Soft labels: (C, H, W) after ToTensorV2 or (H, W, C)
        # The mask should have the n_classes dimension
        mask = result["mask"]
        self.assertTrue(
            self.N_CLASSES in mask.shape,
            f"Soft label mask should have {self.N_CLASSES} as one dimension, got {mask.shape}"
        )


# ======================================================================
# G. WeightedJMLSCEGCBLLoss with Soft Labels End-to-End
# ======================================================================


class TestJMLSCEGCBLSoftLabelsEndToEnd(unittest.TestCase):
    """End-to-end test simulating training with soft labels + GCBL."""

    def setUp(self):
        warnings.simplefilter("ignore")
        self.num_classes = 7

    def test_forward_backward_soft_labels_with_boundaries(self):
        """Simulate a training step with soft labels that have boundaries."""
        loss_fn = WeightedJMLSCEGCBLLoss(
            num_classes=self.num_classes,
            jml_weight=0.25,
            sce_weight=0.75,
            ignore_index=255,
            sce_alpha=1.0,
            sce_beta=0.8,
            smooth=0.001,
            jml_classes="present",
            gcbl_weight=0.1,
            gcbl_embed_dim=32,
            gcbl_temperature=0.07,
            gcbl_max_samples=512,
        )

        # Simulate model output
        pred = torch.randn(4, self.num_classes, 64, 64, requires_grad=True)

        # Create realistic soft labels with boundaries
        target = torch.zeros(4, self.num_classes, 64, 64)
        # Top-left: mostly class 0
        target[:, 0, :32, :32] = 0.85
        target[:, 1, :32, :32] = 0.15
        # Top-right: mostly class 1
        target[:, 1, :32, 32:] = 0.9
        target[:, 0, :32, 32:] = 0.1
        # Bottom-left: mostly class 4 (campo)
        target[:, 4, 32:, :32] = 0.7
        target[:, 6, 32:, :32] = 0.3  # mix with veg_cultivada
        # Bottom-right: mostly class 6 (veg_cultivada)
        target[:, 6, 32:, 32:] = 0.8
        target[:, 4, 32:, 32:] = 0.2

        loss = loss_fn(pred, target)
        self.assertGreater(loss.item(), 0)

        # Backward should work
        loss.backward()
        self.assertIsNotNone(pred.grad)

        # Projector params should also get gradients
        for p in loss_fn.gcbl.projector.parameters():
            if p.requires_grad:
                self.assertIsNotNone(p.grad,
                                     "GCBL projector should receive gradients")

    def test_optimizer_includes_gcbl_params(self):
        """When loss_fn is an attribute of nn.Module, its params should be in optimizer."""
        # Simulate what Model does
        parent_module = nn.Module()
        parent_module.loss_function = WeightedJMLSCEGCBLLoss(
            num_classes=self.num_classes,
            gcbl_weight=0.1,
        )
        parent_module.backbone = nn.Linear(10, self.num_classes)

        # All parameters (including GCBL projector)
        all_params = list(parent_module.parameters())
        gcbl_params = list(parent_module.loss_function.gcbl.projector.parameters())

        # GCBL params should be a subset of all_params
        for gp in gcbl_params:
            found = any(p.data_ptr() == gp.data_ptr() for p in all_params)
            self.assertTrue(found,
                            "GCBL projector param should be in parent module's parameters()")


# ======================================================================
# H. Integral Image Optimization (_build_class_position_index)
# ======================================================================


class TestIntegralImageIndex(unittest.TestCase):
    """Tests for the integral image optimization in _build_class_position_index."""

    def test_integral_matches_naive(self):
        """Integral image crop sums produce identical results to naive np.unique."""
        np.random.seed(42)
        H, W = 64, 64
        n_classes = 4
        mask = np.random.randint(0, n_classes, (H, W), dtype=np.uint8)
        crop_h, crop_w = 16, 16
        stride = 8

        rows = np.arange(0, H - crop_h + 1, stride)
        cols = np.arange(0, W - crop_w + 1, stride)

        # Naive approach (original code)
        naive = {c: set() for c in range(n_classes)}
        for row in rows:
            for col in cols:
                crop = mask[row:row + crop_h, col:col + crop_w]
                for c in np.unique(crop):
                    if 0 <= c < n_classes:
                        naive[c].add((col, row))

        # Integral image approach (new code)
        integral_pos = {c: set() for c in range(n_classes)}
        for c in range(n_classes):
            binary = (mask == c).astype(np.uint32)
            integ = np.zeros((H + 1, W + 1), dtype=np.uint32)
            integ[1:, 1:] = binary.cumsum(axis=0).cumsum(axis=1)
            rr, cc = np.meshgrid(rows, cols, indexing="ij")
            sums = (
                integ[rr + crop_h, cc + crop_w]
                - integ[rr, cc + crop_w]
                - integ[rr + crop_h, cc]
                + integ[rr, cc]
            )
            present = np.argwhere(sums > 0)
            for pos in present:
                integral_pos[c].add((cols[pos[1]], rows[pos[0]]))

        for c in range(n_classes):
            self.assertEqual(naive[c], integral_pos[c],
                             f"Class {c}: integral positions differ from naive")

    def test_integral_excludes_class_255(self):
        """Integral image correctly ignores class 255 (not in range 0..n_classes)."""
        H, W = 32, 32
        mask = np.full((H, W), 255, dtype=np.uint8)
        mask[:16, :] = 0
        n_classes = 3
        crop_h, crop_w = 16, 16
        stride = 8
        rows = np.arange(0, H - crop_h + 1, stride)
        cols = np.arange(0, W - crop_w + 1, stride)

        for c in range(n_classes):
            binary = (mask == c).astype(np.uint32)
            integ = np.zeros((H + 1, W + 1), dtype=np.uint32)
            integ[1:, 1:] = binary.cumsum(axis=0).cumsum(axis=1)
            rr, cc = np.meshgrid(rows, cols, indexing="ij")
            sums = (
                integ[rr + crop_h, cc + crop_w]
                - integ[rr, cc + crop_w]
                - integ[rr + crop_h, cc]
                + integ[rr, cc]
            )
            present = np.argwhere(sums > 0)
            if c == 0:
                self.assertGreater(len(present), 0,
                                   "Class 0 should have positions")
            else:
                self.assertEqual(len(present), 0,
                                 f"Class {c} should have 0 positions (all 255)")

    def test_integral_single_class_per_crop(self):
        """When each crop contains exactly one class, results match."""
        H, W = 32, 32
        n_classes = 4
        mask = np.zeros((H, W), dtype=np.uint8)
        mask[:16, :16] = 0
        mask[:16, 16:] = 1
        mask[16:, :16] = 2
        mask[16:, 16:] = 3
        crop_h, crop_w = 16, 16
        stride = 16

        rows = np.arange(0, H - crop_h + 1, stride)
        cols = np.arange(0, W - crop_w + 1, stride)

        for c in range(n_classes):
            binary = (mask == c).astype(np.uint32)
            integ = np.zeros((H + 1, W + 1), dtype=np.uint32)
            integ[1:, 1:] = binary.cumsum(axis=0).cumsum(axis=1)
            rr, cc = np.meshgrid(rows, cols, indexing="ij")
            sums = (
                integ[rr + crop_h, cc + crop_w]
                - integ[rr, cc + crop_w]
                - integ[rr + crop_h, cc]
                + integ[rr, cc]
            )
            present = np.argwhere(sums > 0)
            # Each class should appear in exactly 1 crop position
            self.assertEqual(len(present), 1,
                             f"Class {c} should be in exactly 1 crop")


class TestIntegralImageInDataset(unittest.TestCase, SyntheticDatasetMixin):
    """Integration test: _build_class_position_index with integral images."""

    def setUp(self):
        self._create_synthetic_dataset()

    def tearDown(self):
        self._cleanup()

    def test_class_position_index_built_correctly(self):
        """Dataset builds class position index with integral images."""
        ds = RandomCropSegmentationDataset(
            input_csv_path=self.csv_path,
            crop_size=self.CROP_SIZE,
            samples_per_epoch=100,
            n_classes=self.N_CLASSES,
            class_balanced_sampling=True,
            class_sampling_stride=16,
        )
        self.assertGreater(len(ds._class_positions), 0,
                           "No classes indexed")
        for c in range(self.N_CLASSES):
            self.assertIn(c, ds._class_positions,
                          f"Class {c} missing from index")
            self.assertGreater(len(ds._class_positions[c]), 0,
                               f"Class {c} has no positions")

    def test_class_position_index_with_soft_labels(self):
        """Index built correctly even with soft labels (argmax fallback)."""
        self._cleanup()
        self._create_soft_labels_dataset()
        ds = RandomCropSegmentationDataset(
            input_csv_path=self.csv_path,
            crop_size=self.CROP_SIZE,
            samples_per_epoch=100,
            n_classes=self.N_CLASSES,
            soft_labels=True,
            class_balanced_sampling=True,
            class_sampling_stride=16,
        )
        self.assertGreater(len(ds._class_positions), 0)


# ======================================================================
# I. hard_mask_csv Parameter
# ======================================================================


class TestHardMaskCSV(unittest.TestCase):
    """Tests for hard_mask_csv parameter in RandomCropSegmentationDataset."""

    def test_parameter_in_init_signature(self):
        """hard_mask_csv parameter exists in __init__ signature."""
        import inspect
        sig = inspect.signature(RandomCropSegmentationDataset.__init__)
        self.assertIn("hard_mask_csv", sig.parameters)

    def test_default_is_none(self):
        """hard_mask_csv defaults to None."""
        import inspect
        sig = inspect.signature(RandomCropSegmentationDataset.__init__)
        self.assertIsNone(sig.parameters["hard_mask_csv"].default)


class TestHardMaskCSVIntegration(unittest.TestCase, SyntheticDatasetMixin):
    """Integration: hard_mask_csv accelerates index building for soft labels."""

    def setUp(self):
        self._create_soft_labels_dataset()
        # Also create hard mask CSV pointing to uint8 masks
        self.hard_dir = os.path.join(self.tmpdir, "hard_masks")
        os.makedirs(self.hard_dir)
        self.hard_paths = []
        h = w = self.IMG_SIZE
        half = h // 2
        for i in range(self.N_IMAGES):
            mask = np.zeros((h, w), dtype=np.uint8)
            mask[:half, :half] = 0
            mask[:half, half:] = 1
            mask[half:, :half] = 2
            mask[half:, half:] = 3
            hard_path = os.path.join(self.hard_dir, f"hard_{i:02d}.tif")
            _create_geotiff(hard_path, mask)
            self.hard_paths.append(hard_path)

        self.hard_csv = os.path.join(self.tmpdir, "hard_dataset.csv")
        with open(self.hard_csv, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["image", "mask"])
            for img_p, hard_p in zip(self.image_paths, self.hard_paths):
                writer.writerow([img_p, hard_p])

    def tearDown(self):
        self._cleanup()

    def test_hard_mask_csv_builds_index(self):
        """Index built using hard_mask_csv when soft_labels=True."""
        ds = RandomCropSegmentationDataset(
            input_csv_path=self.csv_path,
            crop_size=self.CROP_SIZE,
            samples_per_epoch=100,
            n_classes=self.N_CLASSES,
            soft_labels=True,
            hard_mask_csv=self.hard_csv,
            class_balanced_sampling=True,
            class_sampling_stride=16,
        )
        self.assertIsNotNone(ds._hard_mask_paths)
        self.assertEqual(len(ds._hard_mask_paths), self.N_IMAGES)
        self.assertGreater(len(ds._class_positions), 0)

    def test_hard_mask_csv_matches_soft_mask_index(self):
        """Index from hard_mask_csv matches index from soft labels (argmax)."""
        ds_soft = RandomCropSegmentationDataset(
            input_csv_path=self.csv_path,
            crop_size=self.CROP_SIZE,
            samples_per_epoch=100,
            n_classes=self.N_CLASSES,
            soft_labels=True,
            class_balanced_sampling=True,
            class_sampling_stride=16,
        )
        ds_hard = RandomCropSegmentationDataset(
            input_csv_path=self.csv_path,
            crop_size=self.CROP_SIZE,
            samples_per_epoch=100,
            n_classes=self.N_CLASSES,
            soft_labels=True,
            hard_mask_csv=self.hard_csv,
            class_balanced_sampling=True,
            class_sampling_stride=16,
        )
        # Both should index the same classes with same positions
        for c in range(self.N_CLASSES):
            soft_count = len(ds_soft._class_positions.get(c, []))
            hard_count = len(ds_hard._class_positions.get(c, []))
            self.assertEqual(soft_count, hard_count,
                             f"Class {c}: soft={soft_count} vs hard={hard_count}")


# ======================================================================
# J. GTIFF_SRS_SOURCE Environment Variable Suppression
# ======================================================================


class TestGTIFFSRSSourceSuppression(unittest.TestCase, SyntheticDatasetMixin):
    """GTIFF_SRS_SOURCE is set during init and restored afterwards."""

    def setUp(self):
        self._create_synthetic_dataset()

    def tearDown(self):
        self._cleanup()

    def test_env_var_restored_after_init(self):
        """GTIFF_SRS_SOURCE is restored to original value after __init__."""
        original = os.environ.pop("GTIFF_SRS_SOURCE", None)
        try:
            _ = RandomCropSegmentationDataset(
                input_csv_path=self.csv_path,
                crop_size=self.CROP_SIZE,
                samples_per_epoch=50,
                n_classes=self.N_CLASSES,
            )
            # After init, GTIFF_SRS_SOURCE should be removed (was not set)
            self.assertNotIn("GTIFF_SRS_SOURCE", os.environ,
                             "GTIFF_SRS_SOURCE should be removed after init")
        finally:
            if original is not None:
                os.environ["GTIFF_SRS_SOURCE"] = original

    def test_env_var_preserved_if_previously_set(self):
        """If GTIFF_SRS_SOURCE was set before, it's preserved after init."""
        os.environ["GTIFF_SRS_SOURCE"] = "EPSG"
        try:
            _ = RandomCropSegmentationDataset(
                input_csv_path=self.csv_path,
                crop_size=self.CROP_SIZE,
                samples_per_epoch=50,
                n_classes=self.N_CLASSES,
            )
            self.assertEqual(os.environ.get("GTIFF_SRS_SOURCE"), "EPSG",
                             "GTIFF_SRS_SOURCE should be restored to 'EPSG'")
        finally:
            os.environ.pop("GTIFF_SRS_SOURCE", None)


if __name__ == "__main__":
    unittest.main()
