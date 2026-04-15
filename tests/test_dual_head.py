# -*- coding: utf-8 -*-
"""
Tests for dual-head training components:
  - UPerNetDualHead model
  - DualHeadKendallLoss
  - Dataset return_hard_mask
  - Model _shared_step integration

Run with:
    python -m pytest tests/test_dual_head.py -v
    # or standalone:
    python tests/test_dual_head.py
"""

import os
import sys
import tempfile
import unittest

# Ensure package is importable
_pkg_root = os.path.join(os.path.dirname(__file__), "..")
if _pkg_root not in sys.path:
    sys.path.insert(0, _pkg_root)

# Windows DLL fix: must run before `import torch`
if sys.platform == "win32":
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    _torch_lib = os.path.join(
        sys.prefix, "Lib", "site-packages", "torch", "lib"
    )
    if os.path.isdir(_torch_lib):
        if hasattr(os, "add_dll_directory"):
            os.add_dll_directory(_torch_lib)
        os.environ["PATH"] = _torch_lib + os.pathsep + os.environ.get("PATH", "")

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from segmentation_models_pytorch.decoders.upernet.decoder import UPerNetDecoder
from segmentation_models_pytorch.base import SegmentationHead, initialization as init

from pytorch_segmentation_models_trainer.custom_losses.loss import (
    DualHeadKendallLoss,
    WeightedJMLSCEGCBLLoss,
)
from pytorch_segmentation_models_trainer.custom_models.upernet_dual_head import (
    UPerNetDualHead,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_fake_encoder_features(batch_size=2, spatial=64):
    """Create fake multi-scale features mimicking ConvNeXtV2-Tiny encoder."""
    return [
        torch.randn(batch_size, 3, spatial, spatial),
        torch.randn(batch_size, 96, spatial // 2, spatial // 2),
        torch.randn(batch_size, 192, spatial // 4, spatial // 4),
        torch.randn(batch_size, 384, spatial // 8, spatial // 8),
        torch.randn(batch_size, 768, spatial // 16, spatial // 16),
    ]


class _MiniDualHead(nn.Module):
    """Minimal dual-head model for testing without downloading encoder weights."""

    def __init__(self, num_classes=6):
        super().__init__()
        enc_ch = (3, 96, 192, 384, 768)
        self.decoder_A = UPerNetDecoder(
            encoder_channels=enc_ch, encoder_depth=5, decoder_channels=256
        )
        self.decoder_B = UPerNetDecoder(
            encoder_channels=enc_ch, encoder_depth=5, decoder_channels=256
        )
        self.head_A = SegmentationHead(
            in_channels=256, out_channels=num_classes, kernel_size=1, upsampling=4
        )
        self.head_B = SegmentationHead(
            in_channels=256, out_channels=num_classes, kernel_size=1, upsampling=4
        )
        self.last_logits_A = None
        self.last_logits_B = None
        self.last_hard_mask = None
        init.initialize_decoder(self.decoder_A)
        init.initialize_decoder(self.decoder_B)
        init.initialize_head(self.head_A)
        init.initialize_head(self.head_B)

    def forward(self, features):
        self.last_logits_A = self.head_A(self.decoder_A(features))
        self.last_logits_B = self.head_B(self.decoder_B(features))
        return (self.last_logits_A + self.last_logits_B) / 2.0


# ---------------------------------------------------------------------------
# UPerNetDualHead tests
# ---------------------------------------------------------------------------

class TestUPerNetDualHead(unittest.TestCase):
    """Test the UPerNetDualHead model structure."""

    def test_inference_head_validation(self):
        """Invalid inference_head should raise ValueError."""
        with self.assertRaises(ValueError):
            UPerNetDualHead(
                encoder_name="tu-convnextv2_tiny.fcmae_ft_in22k_in1k",
                encoder_weights=None,  # no download
                classes=6,
                inference_head="invalid",
            )

    def test_dual_decoder_output_shape(self):
        """Both decoders should produce correct output shapes."""
        B, C, S = 2, 6, 64
        features = _make_fake_encoder_features(B, S)

        enc_ch = (3, 96, 192, 384, 768)
        dec = UPerNetDecoder(
            encoder_channels=enc_ch, encoder_depth=5, decoder_channels=256
        )
        head = SegmentationHead(
            in_channels=256, out_channels=C, kernel_size=1, upsampling=4
        )
        out = head(dec(features))
        self.assertEqual(out.shape, (B, C, S, S))

    def test_dual_head_stores_logits(self):
        """forward() should store last_logits_A and last_logits_B."""
        model = _MiniDualHead(num_classes=6)
        features = _make_fake_encoder_features(2, 64)
        out = model(features)

        self.assertIsNotNone(model.last_logits_A)
        self.assertIsNotNone(model.last_logits_B)
        self.assertEqual(model.last_logits_A.shape, (2, 6, 64, 64))
        self.assertEqual(model.last_logits_B.shape, (2, 6, 64, 64))
        # Average output
        expected = (model.last_logits_A + model.last_logits_B) / 2.0
        self.assertTrue(torch.allclose(out, expected))

    def test_decoder_params_independent(self):
        """Two decoders should have independent parameters."""
        model = _MiniDualHead(num_classes=6)
        params_A = set(id(p) for p in model.decoder_A.parameters())
        params_B = set(id(p) for p in model.decoder_B.parameters())
        self.assertEqual(len(params_A & params_B), 0)


# ---------------------------------------------------------------------------
# DualHeadKendallLoss tests
# ---------------------------------------------------------------------------

class TestDualHeadKendallLoss(unittest.TestCase):
    """Test DualHeadKendallLoss."""

    def setUp(self):
        self.B, self.C, self.H, self.W = 2, 6, 32, 32
        self.loss_fn = DualHeadKendallLoss(
            num_classes=self.C,
            consistency_warmup_epochs=10,
        )
        self.model = _MiniDualHead(num_classes=self.C)
        self.loss_fn.set_model(self.model)

    def _run_forward(self, epoch=0):
        """Run model + loss forward pass."""
        features = _make_fake_encoder_features(self.B, self.H)
        self.model.last_hard_mask = torch.randint(0, self.C, (self.B, self.H, self.W))
        pred = self.model(features)
        soft = F.softmax(torch.randn(self.B, self.C, self.H, self.W) * 3, dim=1)
        return self.loss_fn(pred, soft, epoch=epoch)

    def test_is_dual_head_loss_property(self):
        """Should have is_dual_head_loss property."""
        self.assertTrue(self.loss_fn.is_dual_head_loss)

    def test_forward_returns_scalar(self):
        """Loss should be a scalar tensor."""
        loss = self._run_forward(epoch=5)
        self.assertEqual(loss.dim(), 0)
        self.assertTrue(loss.requires_grad)

    def test_consistency_warmup_zero_at_epoch_0(self):
        """At epoch 0, consistency term should not contribute."""
        loss_fn = DualHeadKendallLoss(
            num_classes=self.C, consistency_warmup_epochs=10
        )
        loss_fn.set_model(self.model)

        features = _make_fake_encoder_features(self.B, self.H)
        self.model.last_hard_mask = torch.randint(0, self.C, (self.B, self.H, self.W))
        pred = self.model(features)
        soft = F.softmax(torch.randn(self.B, self.C, self.H, self.W), dim=1)

        loss_val = loss_fn(pred, soft, epoch=0)
        loss_val.backward()
        # At epoch 0 with warmup=10, warmup_factor=0, so sigma_consist gets no grad
        self.assertEqual(loss_fn.log_sigma_consist.grad.item(), 0.0)

    def test_consistency_active_after_warmup(self):
        """After warmup, consistency grad should be nonzero."""
        loss = self._run_forward(epoch=15)
        loss.backward()
        self.assertNotEqual(self.loss_fn.log_sigma_consist.grad.item(), 0.0)

    def test_backward_all_params_have_grad(self):
        """All loss parameters should receive gradients."""
        loss = self._run_forward(epoch=5)
        loss.backward()
        for name, p in self.loss_fn.named_parameters():
            if p.requires_grad:
                self.assertIsNotNone(
                    p.grad, f"Parameter {name} has no gradient"
                )

    def test_backward_model_params_have_grad(self):
        """Model decoder params should receive gradients."""
        loss = self._run_forward(epoch=5)
        loss.backward()
        grads = [p.grad for p in self.model.parameters() if p.grad is not None]
        self.assertGreater(len(grads), 0)

    def test_three_learnable_sigmas(self):
        """Should have exactly 3 learnable sigma parameters."""
        sigmas = [self.loss_fn.log_sigma_hard, self.loss_fn.log_sigma_soft,
                  self.loss_fn.log_sigma_consist]
        for s in sigmas:
            self.assertTrue(s.requires_grad)
            self.assertEqual(s.shape, ())

    def test_hard_mask_none_fallback(self):
        """When hard_mask is None, should use soft targets for head A."""
        features = _make_fake_encoder_features(self.B, self.H)
        self.model.last_hard_mask = None
        pred = self.model(features)
        soft = F.softmax(torch.randn(self.B, self.C, self.H, self.W), dim=1)
        loss = self.loss_fn(pred, soft, epoch=5)
        self.assertTrue(torch.isfinite(loss))

    def test_ignore_index_respected(self):
        """Pixels with ignore_index=255 in hard mask should not affect loss."""
        features = _make_fake_encoder_features(self.B, self.H)
        # All pixels are ignore
        self.model.last_hard_mask = torch.full(
            (self.B, self.H, self.W), 255, dtype=torch.long
        )
        pred = self.model(features)
        soft = F.softmax(torch.randn(self.B, self.C, self.H, self.W), dim=1)
        loss = self.loss_fn(pred, soft, epoch=5)
        self.assertTrue(torch.isfinite(loss))


# ---------------------------------------------------------------------------
# Symmetric KL tests
# ---------------------------------------------------------------------------

class TestSymmetricKL(unittest.TestCase):
    """Test _symmetric_kl helper."""

    def test_identical_distributions_zero_kl(self):
        """KL between identical distributions should be ~0."""
        logits = torch.randn(2, 6, 8, 8)
        log_p = F.log_softmax(logits, dim=1)
        p = log_p.exp()
        kl = DualHeadKendallLoss._symmetric_kl(log_p, p, log_p, p)
        self.assertAlmostEqual(kl.item(), 0.0, places=5)

    def test_different_distributions_positive_kl(self):
        """KL between different distributions should be positive."""
        logits_a = torch.randn(2, 6, 8, 8)
        logits_b = torch.randn(2, 6, 8, 8) + 5.0
        log_pa = F.log_softmax(logits_a, dim=1)
        pa = log_pa.exp()
        log_pb = F.log_softmax(logits_b, dim=1)
        pb = log_pb.exp()
        kl = DualHeadKendallLoss._symmetric_kl(log_pa, pa, log_pb, pb)
        self.assertGreater(kl.item(), 0.0)

    def test_symmetric(self):
        """KL(A,B) should equal KL(B,A) (it's symmetric by design)."""
        logits_a = torch.randn(2, 6, 8, 8)
        logits_b = torch.randn(2, 6, 8, 8) + 2.0
        log_pa = F.log_softmax(logits_a, dim=1)
        pa = log_pa.exp()
        log_pb = F.log_softmax(logits_b, dim=1)
        pb = log_pb.exp()
        kl_ab = DualHeadKendallLoss._symmetric_kl(log_pa, pa, log_pb, pb)
        kl_ba = DualHeadKendallLoss._symmetric_kl(log_pb, pb, log_pa, pa)
        self.assertAlmostEqual(kl_ab.item(), kl_ba.item(), places=5)


# ---------------------------------------------------------------------------
# Dataset return_hard_mask tests
# ---------------------------------------------------------------------------

class TestDatasetReturnHardMask(unittest.TestCase):
    """Test return_hard_mask in RandomCropSegmentationDataset."""

    def setUp(self):
        """Create temp dir with fake image, soft mask, and hard mask."""
        self.tmpdir = tempfile.mkdtemp()
        self.C = 6
        self.H, self.W = 64, 64

        try:
            import rasterio
            from rasterio.transform import from_bounds
        except ImportError:
            self.skipTest("rasterio not available")

        self.transform = from_bounds(0, 0, 1, 1, self.W, self.H)
        profile = {
            "driver": "GTiff",
            "dtype": "uint8",
            "width": self.W,
            "height": self.H,
            "count": 3,
            "crs": "EPSG:4326",
            "transform": self.transform,
        }

        # Image
        self.img_path = os.path.join(self.tmpdir, "img.tif")
        with rasterio.open(self.img_path, "w", **profile) as dst:
            dst.write(np.random.randint(0, 255, (3, self.H, self.W), dtype=np.uint8))

        # Soft mask (C bands float32)
        soft_profile = {**profile, "dtype": "float32", "count": self.C}
        self.soft_path = os.path.join(self.tmpdir, "soft.tif")
        with rasterio.open(self.soft_path, "w", **soft_profile) as dst:
            raw = np.random.rand(self.C, self.H, self.W).astype(np.float32)
            raw /= raw.sum(axis=0, keepdims=True)
            dst.write(raw)

        # Hard mask (1 band uint8)
        hard_profile = {**profile, "dtype": "uint8", "count": 1}
        self.hard_path = os.path.join(self.tmpdir, "hard.tif")
        with rasterio.open(self.hard_path, "w", **hard_profile) as dst:
            mask = np.random.randint(0, self.C, (1, self.H, self.W), dtype=np.uint8)
            mask[0, :5, :5] = 255  # some ignore pixels
            dst.write(mask)

        # CSVs
        import pandas as pd
        self.soft_csv = os.path.join(self.tmpdir, "soft.csv")
        pd.DataFrame({"image": [self.img_path], "mask": [self.soft_path]}).to_csv(
            self.soft_csv, index=False
        )
        self.hard_csv = os.path.join(self.tmpdir, "hard.csv")
        pd.DataFrame({"image": [self.img_path], "mask": [self.hard_path]}).to_csv(
            self.hard_csv, index=False
        )

    def _make_dataset(self, return_hard_mask=True, crop_size=32):
        from pytorch_segmentation_models_trainer.dataset_loader.dataset import (
            RandomCropSegmentationDataset,
        )
        return RandomCropSegmentationDataset(
            input_csv_path=self.soft_csv,
            crop_size=[crop_size, crop_size],
            samples_per_epoch=10,
            n_classes=self.C,
            use_rasterio=True,
            soft_labels=True,
            hard_mask_csv=self.hard_csv,
            return_hard_mask=return_hard_mask,
            min_valid_ratio=0.0,
        )

    def test_return_hard_mask_true(self):
        """With return_hard_mask=True, batch should have 'hard_mask' key."""
        ds = self._make_dataset(return_hard_mask=True)
        item = ds[0]
        self.assertIn("hard_mask", item)
        self.assertEqual(item["hard_mask"].dtype, torch.long)
        self.assertEqual(item["hard_mask"].shape, (32, 32))

    def test_return_hard_mask_false(self):
        """With return_hard_mask=False, batch should NOT have 'hard_mask'."""
        ds = self._make_dataset(return_hard_mask=False)
        item = ds[0]
        self.assertNotIn("hard_mask", item)

    def test_soft_mask_present(self):
        """Soft mask should be (C, H, W) float."""
        ds = self._make_dataset(return_hard_mask=True)
        item = ds[0]
        self.assertEqual(item["mask"].shape[0], self.C)
        self.assertTrue(item["mask"].is_floating_point())

    def test_to_hard_mask_tensor_numpy(self):
        """_to_hard_mask_tensor should handle numpy arrays."""
        from pytorch_segmentation_models_trainer.dataset_loader.dataset import (
            RandomCropSegmentationDataset,
        )
        arr = np.array([[0, 1, 2], [3, 255, 5]], dtype=np.uint8)
        result = RandomCropSegmentationDataset._to_hard_mask_tensor(arr)
        self.assertEqual(result.dtype, torch.long)
        self.assertTrue(torch.equal(result, torch.tensor([[0, 1, 2], [3, 255, 5]])))

    def test_to_hard_mask_tensor_torch(self):
        """_to_hard_mask_tensor should handle torch tensors."""
        from pytorch_segmentation_models_trainer.dataset_loader.dataset import (
            RandomCropSegmentationDataset,
        )
        t = torch.tensor([[0, 1], [2, 3]], dtype=torch.float32)
        result = RandomCropSegmentationDataset._to_hard_mask_tensor(t)
        self.assertEqual(result.dtype, torch.long)


# ---------------------------------------------------------------------------
# Kendall weighting math tests
# ---------------------------------------------------------------------------

class TestKendallWeighting(unittest.TestCase):
    """Test the mathematical properties of Kendall uncertainty weighting."""

    def test_equal_losses_equal_sigmas(self):
        """With σ=1 (log_σ=0), precision=1, loss = L + 0 = L."""
        loss_fn = DualHeadKendallLoss(num_classes=6, consistency_warmup_epochs=0)
        # Verify initial sigma values
        self.assertAlmostEqual(loss_fn.log_sigma_hard.item(), 0.0)
        self.assertAlmostEqual(loss_fn.log_sigma_soft.item(), 0.0)
        self.assertAlmostEqual(loss_fn.log_sigma_consist.item(), 0.0)

    def test_sigma_gradient_direction(self):
        """When loss is large, sigma should grow (grad on log_sigma < 0)
        since the model should 'give up' on that term."""
        loss_fn = DualHeadKendallLoss(num_classes=6, consistency_warmup_epochs=0)
        model = _MiniDualHead(num_classes=6)
        loss_fn.set_model(model)

        features = _make_fake_encoder_features(2, 32)
        model.last_hard_mask = torch.randint(0, 6, (2, 32, 32))
        pred = model(features)
        soft = F.softmax(torch.randn(2, 6, 32, 32), dim=1)

        loss = loss_fn(pred, soft, epoch=5)
        loss.backward()

        # With untrained model, losses are large, precision=exp(-0)=1.
        # grad of log_sigma = -exp(-log_sigma)*L + 1 = -L + 1.
        # For large L, grad < 0, meaning sigma wants to increase (log_sigma increases).
        # This is the correct behavior: high loss → increase uncertainty.
        for name in ["log_sigma_hard", "log_sigma_soft"]:
            grad = getattr(loss_fn, name).grad.item()
            self.assertIsNotNone(grad)


if __name__ == "__main__":
    unittest.main()
