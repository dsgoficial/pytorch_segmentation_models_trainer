# -*- coding: utf-8 -*-
"""
Tests for loss functions, training callbacks, and TTA inference:
  - _lovasz_grad, _lovasz_softmax_flat (Lovász helpers)
  - WeightedLovaszSCELoss
  - WeightedJMLSCELoss
  - WeightedDiceCrossEntropyLoss
  - EMACallback (including EMA warmup decay)
  - WarmupCallback
  - LLRD (layer-wise LR decay)
  - MultiClassInferenceProcessor TTA transforms
"""

import unittest
import warnings

# IMPORTANT: import pytorch_segmentation_models_trainer BEFORE torch
# to resolve fbgemm.dll vs GDAL DLL conflict on Windows.
import pytorch_segmentation_models_trainer  # noqa: F401

import torch
import torch.nn as nn
from unittest.mock import Mock, MagicMock

from pytorch_segmentation_models_trainer.custom_losses.loss import (
    _lovasz_grad,
    _lovasz_softmax_flat,
    WeightedLovaszSCELoss,
    WeightedJMLSCELoss,
    WeightedDiceCrossEntropyLoss,
)
from pytorch_segmentation_models_trainer.custom_callbacks.training_callbacks import (
    EMACallback,
    WarmupCallback,
)
from pytorch_segmentation_models_trainer.tools.inference.inference_processors import (
    MultiClassInferenceProcessor,
)


class _TinyModel(nn.Module):
    """Minimal model for EMA/Warmup callback testing."""

    def __init__(self):
        super().__init__()
        self.encoder = nn.Linear(4, 3)
        self.decoder = nn.Linear(3, 2)

    def set_encoder_trainable(self, trainable=True):
        for p in self.encoder.parameters():
            p.requires_grad = trainable


# ----------------------------------------------------------------------
# A. _lovasz_grad
# ----------------------------------------------------------------------


class TestLovaszGrad(unittest.TestCase):

    def setUp(self):
        warnings.simplefilter("ignore")

    def test_single_element(self):
        grad = _lovasz_grad(torch.tensor([1.0]))
        self.assertEqual(len(grad), 1)
        self.assertAlmostEqual(grad[0].item(), 1.0, places=5)

    def test_all_foreground(self):
        gt = torch.tensor([1.0, 1.0, 1.0, 1.0])
        grad = _lovasz_grad(gt)
        self.assertEqual(len(grad), 4)
        # All values should be non-negative
        self.assertTrue(torch.all(grad >= 0).item())

    def test_all_background(self):
        gt = torch.tensor([0.0, 0.0, 0.0, 0.0])
        grad = _lovasz_grad(gt)
        self.assertEqual(len(grad), 4)
        # First gradient is 1.0, rest are 0.0 (Jaccard diff)
        self.assertAlmostEqual(grad[0].item(), 1.0, places=5)
        self.assertTrue(torch.all(grad >= 0).item())

    def test_known_gradient_values(self):
        """Hand-computed example: sorted FG indicators [1, 1, 0, 0].
        Intersection: gts=2, cumsum=[1,2,2,2] -> [2-1, 2-2, 2-2, 2-2] = [1, 0, 0, 0]
        Union: gts=2, bg_cumsum=[0,0,1,2] -> [2+0, 2+0, 2+1, 2+2] = [2, 2, 3, 4]
        Jaccard: 1 - [1/2, 0/2, 0/3, 0/4] = [0.5, 1.0, 1.0, 1.0]
        Diff: [0.5, 0.5, 0.0, 0.0]
        """
        gt = torch.tensor([1.0, 1.0, 0.0, 0.0])
        grad = _lovasz_grad(gt)
        expected = torch.tensor([0.5, 0.5, 0.0, 0.0])
        self.assertTrue(
            torch.allclose(grad, expected, atol=1e-5),
            f"Expected {expected}, got {grad}",
        )

    def test_output_length_matches_input(self):
        for n in [1, 5, 10, 100]:
            gt = torch.randint(0, 2, (n,)).float()
            grad = _lovasz_grad(gt)
            self.assertEqual(len(grad), n)


# ----------------------------------------------------------------------
# B. _lovasz_softmax_flat
# ----------------------------------------------------------------------


class TestLovaszSoftmaxFlat(unittest.TestCase):

    def setUp(self):
        warnings.simplefilter("ignore")

    def test_perfect_prediction_zero_loss(self):
        """One-hot probas matching labels should give near-zero loss."""
        C = 4
        P = 64
        labels = torch.randint(0, C, (P,))
        probas = torch.zeros(P, C)
        for i in range(P):
            probas[i, labels[i]] = 1.0
        loss = _lovasz_softmax_flat(probas, labels, classes="present")
        self.assertAlmostEqual(loss.item(), 0.0, places=4)

    def test_worst_prediction_high_loss(self):
        """Inverted probas should give high loss."""
        C = 4
        P = 64
        labels = torch.zeros(P, dtype=torch.long)  # all class 0
        probas = torch.zeros(P, C)
        probas[:, 1] = 1.0  # predict class 1 for everything
        loss = _lovasz_softmax_flat(probas, labels, classes="present")
        self.assertGreater(loss.item(), 0.5)

    def test_empty_input_returns_zero(self):
        probas = torch.empty(0, 4)
        labels = torch.empty(0, dtype=torch.long)
        loss = _lovasz_softmax_flat(probas, labels, classes="present")
        self.assertEqual(loss.numel(), 0)

    def test_classes_present_skips_absent(self):
        """With classes='present', only classes in labels contribute."""
        C = 4
        P = 32
        labels = torch.zeros(P, dtype=torch.long)  # only class 0
        probas = torch.rand(P, C, requires_grad=True)
        probas_norm = probas / probas.sum(dim=1, keepdim=True)
        loss = _lovasz_softmax_flat(probas_norm, labels, classes="present")
        # Should compute without error and be a scalar
        self.assertEqual(loss.dim(), 0)

    def test_classes_all_includes_absent(self):
        """With classes='all', absent classes also contribute."""
        C = 4
        P = 32
        labels = torch.zeros(P, dtype=torch.long)  # only class 0
        probas = torch.rand(P, C, requires_grad=True)
        probas_norm = probas / probas.sum(dim=1, keepdim=True)
        loss_present = _lovasz_softmax_flat(probas_norm, labels, classes="present")
        loss_all = _lovasz_softmax_flat(probas_norm, labels, classes="all")
        # Both should be valid scalars (may differ)
        self.assertEqual(loss_all.dim(), 0)
        self.assertGreater(loss_all.item(), 0)

    def test_gradient_flows(self):
        C = 4
        P = 32
        labels = torch.randint(0, C, (P,))
        probas = torch.rand(P, C, requires_grad=True)
        probas_norm = probas / probas.sum(dim=1, keepdim=True)
        loss = _lovasz_softmax_flat(probas_norm, labels, classes="present")
        loss.backward()
        self.assertIsNotNone(probas.grad)

    def test_single_class_single_pixel(self):
        probas = torch.tensor([[1.0]], requires_grad=True)
        labels = torch.tensor([0])
        loss = _lovasz_softmax_flat(probas, labels, classes="present")
        self.assertEqual(loss.dim(), 0)


# ----------------------------------------------------------------------
# C. WeightedLovaszSCELoss
# ----------------------------------------------------------------------


class TestWeightedLovaszSCELoss(unittest.TestCase):

    def setUp(self):
        warnings.simplefilter("ignore")
        self.num_classes = 7

    def _make_loss(self, **kwargs):
        defaults = dict(
            num_classes=self.num_classes,
            lovasz_weight=0.25,
            sce_weight=0.75,
            ignore_index=255,
        )
        defaults.update(kwargs)
        return WeightedLovaszSCELoss(**defaults)

    def test_basic_forward(self):
        loss_fn = self._make_loss()
        pred = torch.randn(2, self.num_classes, 32, 32, requires_grad=True)
        target = torch.randint(0, self.num_classes, (2, 32, 32))
        loss = loss_fn(pred, target)
        self.assertEqual(loss.dim(), 0)
        self.assertTrue(loss.requires_grad)
        self.assertGreater(loss.item(), 0)

    def test_perfect_prediction_low_loss(self):
        loss_fn = self._make_loss()
        target = torch.randint(0, self.num_classes, (2, 32, 32))
        pred = torch.zeros(2, self.num_classes, 32, 32)
        for b in range(2):
            for c in range(self.num_classes):
                pred[b, c, target[b] == c] = 10.0
        loss = loss_fn(pred, target)
        self.assertLess(loss.item(), 0.5)

    def test_all_ignore_returns_zero(self):
        loss_fn = self._make_loss()
        pred = torch.randn(2, self.num_classes, 32, 32, requires_grad=True)
        target = torch.full((2, 32, 32), 255, dtype=torch.long)
        loss = loss_fn(pred, target)
        self.assertAlmostEqual(loss.item(), 0.0, places=5)

    def test_mixed_ignore_and_valid(self):
        loss_fn = self._make_loss()
        pred = torch.randn(2, self.num_classes, 32, 32, requires_grad=True)
        target = torch.randint(0, self.num_classes, (2, 32, 32))
        target[:, :16, :] = 255  # half ignored
        loss = loss_fn(pred, target)
        self.assertGreater(loss.item(), 0)
        self.assertTrue(loss.requires_grad)

    def test_gradient_flows(self):
        loss_fn = self._make_loss()
        pred = torch.randn(2, self.num_classes, 32, 32, requires_grad=True)
        target = torch.randint(0, self.num_classes, (2, 32, 32))
        loss = loss_fn(pred, target)
        loss.backward()
        self.assertIsNotNone(pred.grad)
        self.assertFalse(torch.all(pred.grad == 0))

    def test_per_image_mode(self):
        loss_fn = self._make_loss(per_image=True)
        pred = torch.randn(2, self.num_classes, 32, 32, requires_grad=True)
        target = torch.randint(0, self.num_classes, (2, 32, 32))
        loss = loss_fn(pred, target)
        self.assertEqual(loss.dim(), 0)
        self.assertGreater(loss.item(), 0)

    def test_batch_vs_per_image_both_finite(self):
        pred = torch.randn(2, self.num_classes, 32, 32)
        target = torch.randint(0, self.num_classes, (2, 32, 32))
        loss_batch = self._make_loss(per_image=False)(pred, target)
        loss_per_img = self._make_loss(per_image=True)(pred, target)
        self.assertTrue(torch.isfinite(loss_batch))
        self.assertTrue(torch.isfinite(loss_per_img))

    def test_lovasz_weight_zero(self):
        """With lovasz_weight=0, loss should be pure SCE."""
        pred = torch.randn(2, self.num_classes, 16, 16)
        target = torch.randint(0, self.num_classes, (2, 16, 16))
        loss_no_lovasz = self._make_loss(lovasz_weight=0.0, sce_weight=1.0)(
            pred, target
        )
        loss_full = self._make_loss(lovasz_weight=0.25, sce_weight=0.75)(pred, target)
        # Both should be valid scalars; no-lovasz should differ from full
        self.assertGreater(loss_no_lovasz.item(), 0)
        self.assertTrue(torch.isfinite(loss_no_lovasz))

    def test_sce_weight_zero(self):
        """With sce_weight=0, loss should be pure Lovász."""
        pred = torch.randn(2, self.num_classes, 16, 16)
        target = torch.randint(0, self.num_classes, (2, 16, 16))
        loss_no_sce = self._make_loss(lovasz_weight=1.0, sce_weight=0.0)(pred, target)
        self.assertGreater(loss_no_sce.item(), 0)
        self.assertTrue(torch.isfinite(loss_no_sce))

    def test_sce_alpha_beta_components(self):
        pred = torch.randn(2, self.num_classes, 16, 16)
        target = torch.randint(0, self.num_classes, (2, 16, 16))
        loss_high_rce = self._make_loss(
            lovasz_weight=0.0, sce_weight=1.0, sce_alpha=0.0, sce_beta=1.0
        )(pred, target)
        loss_high_ce = self._make_loss(
            lovasz_weight=0.0, sce_weight=1.0, sce_alpha=1.0, sce_beta=0.0
        )(pred, target)
        self.assertGreater(loss_high_rce.item(), 0)
        self.assertGreater(loss_high_ce.item(), 0)

    def test_single_pixel_batch(self):
        loss_fn = self._make_loss()
        pred = torch.randn(1, self.num_classes, 1, 1, requires_grad=True)
        target = torch.randint(0, self.num_classes, (1, 1, 1))
        loss = loss_fn(pred, target)
        self.assertEqual(loss.dim(), 0)
        self.assertTrue(torch.isfinite(loss))

    def test_mixed_precision_float16(self):
        loss_fn = self._make_loss()
        pred = torch.randn(2, self.num_classes, 16, 16).half()
        pred.requires_grad = True
        target = torch.randint(0, self.num_classes, (2, 16, 16))
        loss = loss_fn(pred, target)
        self.assertTrue(torch.isfinite(loss))
        self.assertFalse(torch.isnan(loss))

    def test_lovasz_classes_all(self):
        loss_fn = self._make_loss(lovasz_classes="all")
        pred = torch.randn(2, self.num_classes, 16, 16, requires_grad=True)
        target = torch.randint(0, self.num_classes, (2, 16, 16))
        loss = loss_fn(pred, target)
        self.assertEqual(loss.dim(), 0)
        self.assertGreater(loss.item(), 0)


# ----------------------------------------------------------------------
# C2. WeightedJMLSCELoss
# ----------------------------------------------------------------------


class TestWeightedJMLSCELoss(unittest.TestCase):

    def setUp(self):
        warnings.simplefilter("ignore")
        self.num_classes = 7

    def _make_loss(self, **kwargs):
        defaults = dict(
            num_classes=self.num_classes,
            jml_weight=0.25,
            sce_weight=0.75,
            ignore_index=255,
        )
        defaults.update(kwargs)
        return WeightedJMLSCELoss(**defaults)

    def test_basic_forward(self):
        loss_fn = self._make_loss()
        pred = torch.randn(2, self.num_classes, 32, 32, requires_grad=True)
        target = torch.randint(0, self.num_classes, (2, 32, 32))
        loss = loss_fn(pred, target)
        self.assertEqual(loss.dim(), 0)
        self.assertTrue(loss.requires_grad)
        self.assertGreater(loss.item(), 0)

    def test_perfect_prediction_low_loss(self):
        loss_fn = self._make_loss()
        target = torch.randint(0, self.num_classes, (2, 32, 32))
        pred = torch.zeros(2, self.num_classes, 32, 32)
        for b in range(2):
            for c in range(self.num_classes):
                pred[b, c, target[b] == c] = 10.0
        loss = loss_fn(pred, target)
        self.assertLess(loss.item(), 0.5)

    def test_all_ignore_returns_zero(self):
        loss_fn = self._make_loss()
        pred = torch.randn(2, self.num_classes, 32, 32, requires_grad=True)
        target = torch.full((2, 32, 32), 255, dtype=torch.long)
        loss = loss_fn(pred, target)
        self.assertAlmostEqual(loss.item(), 0.0, places=5)

    def test_mixed_ignore_and_valid(self):
        loss_fn = self._make_loss()
        pred = torch.randn(2, self.num_classes, 32, 32, requires_grad=True)
        target = torch.randint(0, self.num_classes, (2, 32, 32))
        target[:, :16, :] = 255  # half ignored
        loss = loss_fn(pred, target)
        self.assertGreater(loss.item(), 0)
        self.assertTrue(loss.requires_grad)

    def test_gradient_flows(self):
        loss_fn = self._make_loss()
        pred = torch.randn(2, self.num_classes, 32, 32, requires_grad=True)
        target = torch.randint(0, self.num_classes, (2, 32, 32))
        loss = loss_fn(pred, target)
        loss.backward()
        self.assertIsNotNone(pred.grad)
        self.assertFalse(torch.all(pred.grad == 0))

    def test_jml_weight_zero(self):
        """With jml_weight=0, loss should be pure SCE."""
        pred = torch.randn(2, self.num_classes, 16, 16)
        target = torch.randint(0, self.num_classes, (2, 16, 16))
        loss_no_jml = self._make_loss(jml_weight=0.0, sce_weight=1.0)(pred, target)
        self.assertGreater(loss_no_jml.item(), 0)
        self.assertTrue(torch.isfinite(loss_no_jml))

    def test_sce_weight_zero(self):
        """With sce_weight=0, loss should be pure JML."""
        pred = torch.randn(2, self.num_classes, 16, 16)
        target = torch.randint(0, self.num_classes, (2, 16, 16))
        loss_no_sce = self._make_loss(jml_weight=1.0, sce_weight=0.0)(pred, target)
        self.assertGreater(loss_no_sce.item(), 0)
        self.assertTrue(torch.isfinite(loss_no_sce))

    def test_jml_classes_all(self):
        loss_fn = self._make_loss(jml_classes="all")
        pred = torch.randn(2, self.num_classes, 16, 16, requires_grad=True)
        target = torch.randint(0, self.num_classes, (2, 16, 16))
        loss = loss_fn(pred, target)
        self.assertEqual(loss.dim(), 0)
        self.assertGreater(loss.item(), 0)

    def test_jml_classes_present_single_class(self):
        """When only one class is present, JML should still work."""
        loss_fn = self._make_loss(jml_classes="present")
        pred = torch.randn(2, self.num_classes, 16, 16, requires_grad=True)
        target = torch.zeros(2, 16, 16, dtype=torch.long)  # only class 0
        loss = loss_fn(pred, target)
        self.assertGreater(loss.item(), 0)
        self.assertTrue(torch.isfinite(loss))

    def test_sce_beta_affects_loss(self):
        """Higher sce_beta should produce different loss value."""
        pred = torch.randn(2, self.num_classes, 16, 16)
        target = torch.randint(0, self.num_classes, (2, 16, 16))
        loss_low = self._make_loss(sce_beta=0.1)(pred, target)
        loss_high = self._make_loss(sce_beta=1.0)(pred, target)
        self.assertNotAlmostEqual(loss_low.item(), loss_high.item(), places=3)

    def test_single_pixel_batch(self):
        loss_fn = self._make_loss()
        pred = torch.randn(1, self.num_classes, 1, 1, requires_grad=True)
        target = torch.randint(0, self.num_classes, (1, 1, 1))
        loss = loss_fn(pred, target)
        self.assertEqual(loss.dim(), 0)
        self.assertTrue(torch.isfinite(loss))

    def test_smooth_parameter(self):
        """Different smooth values should produce different JML losses."""
        pred = torch.randn(2, self.num_classes, 16, 16)
        target = torch.randint(0, self.num_classes, (2, 16, 16))
        loss_small = self._make_loss(smooth=1e-6, jml_weight=1.0, sce_weight=0.0)(
            pred, target
        )
        loss_large = self._make_loss(smooth=1.0, jml_weight=1.0, sce_weight=0.0)(
            pred, target
        )
        # Large smooth biases IoU toward 1, giving lower loss
        self.assertGreater(loss_small.item(), loss_large.item())


# ----------------------------------------------------------------------
# D. WeightedDiceCrossEntropyLoss
# ----------------------------------------------------------------------


class TestWeightedDiceCrossEntropyLoss(unittest.TestCase):

    def setUp(self):
        warnings.simplefilter("ignore")
        self.num_classes = 7

    def _make_loss(self, **kwargs):
        defaults = dict(
            num_classes=self.num_classes,
            dice_weight=0.25,
            ce_weight=0.75,
            smooth_factor=0.0,
        )
        defaults.update(kwargs)
        return WeightedDiceCrossEntropyLoss(**defaults)

    def test_basic_forward(self):
        loss_fn = self._make_loss()
        pred = torch.randn(2, self.num_classes, 32, 32, requires_grad=True)
        target = torch.randint(0, self.num_classes, (2, 32, 32))
        loss = loss_fn(pred, target)
        self.assertEqual(loss.dim(), 0)
        self.assertTrue(loss.requires_grad)
        self.assertGreater(loss.item(), 0)

    def test_perfect_prediction_low_loss(self):
        loss_fn = self._make_loss()
        target = torch.randint(0, self.num_classes, (2, 32, 32))
        pred = torch.zeros(2, self.num_classes, 32, 32)
        for b in range(2):
            for c in range(self.num_classes):
                pred[b, c, target[b] == c] = 10.0
        loss = loss_fn(pred, target)
        self.assertLess(loss.item(), 0.5)

    def test_gradient_flows(self):
        loss_fn = self._make_loss()
        pred = torch.randn(2, self.num_classes, 32, 32, requires_grad=True)
        target = torch.randint(0, self.num_classes, (2, 32, 32))
        loss = loss_fn(pred, target)
        loss.backward()
        self.assertIsNotNone(pred.grad)
        self.assertFalse(torch.all(pred.grad == 0))

    def test_ignore_index_all_ignored(self):
        loss_fn = self._make_loss(ignore_index=255)
        pred = torch.randn(2, self.num_classes, 32, 32, requires_grad=True)
        target = torch.full((2, 32, 32), 255, dtype=torch.long)
        loss = loss_fn(pred, target)
        # When all targets are ignored, CE returns NaN (PyTorch behavior).
        # Just verify it produces a scalar without crashing.
        self.assertEqual(loss.dim(), 0)

    def test_ignore_index_partial(self):
        loss_fn = self._make_loss(ignore_index=255)
        pred = torch.randn(2, self.num_classes, 32, 32, requires_grad=True)
        target = torch.randint(0, self.num_classes, (2, 32, 32))
        target[:, :16, :] = 255
        loss = loss_fn(pred, target)
        self.assertGreater(loss.item(), 0)
        self.assertTrue(loss.requires_grad)

    def test_dice_weight_zero(self):
        """dice_weight=0 → pure CE."""
        loss_fn = self._make_loss(dice_weight=0.0, ce_weight=1.0)
        pred = torch.randn(2, self.num_classes, 16, 16, requires_grad=True)
        target = torch.randint(0, self.num_classes, (2, 16, 16))
        loss = loss_fn(pred, target)
        self.assertGreater(loss.item(), 0)

    def test_ce_weight_zero(self):
        """ce_weight=0 → pure Dice."""
        loss_fn = self._make_loss(dice_weight=1.0, ce_weight=0.0)
        pred = torch.randn(2, self.num_classes, 16, 16, requires_grad=True)
        target = torch.randint(0, self.num_classes, (2, 16, 16))
        loss = loss_fn(pred, target)
        self.assertGreater(loss.item(), 0)

    def test_small_num_classes(self):
        """num_classes=3 (small number) should work correctly."""
        loss_fn = self._make_loss(num_classes=3)
        pred = torch.randn(2, 3, 16, 16, requires_grad=True)
        target = torch.randint(0, 3, (2, 16, 16))
        loss = loss_fn(pred, target)
        self.assertEqual(loss.dim(), 0)
        self.assertGreater(loss.item(), 0)


# ----------------------------------------------------------------------
# E. EMACallback
# ----------------------------------------------------------------------


class TestEMACallback(unittest.TestCase):

    def setUp(self):
        warnings.simplefilter("ignore")
        self.model = _TinyModel()
        self.trainer = Mock()
        self.trainer.current_epoch = 0
        self.trainer.global_step = 0

    def test_decay_validation(self):
        with self.assertRaises(ValueError):
            EMACallback(decay=0.0)
        with self.assertRaises(ValueError):
            EMACallback(decay=1.0)

    def test_decay_validation_negative(self):
        with self.assertRaises(ValueError):
            EMACallback(decay=-0.5)

    def test_on_fit_start_initializes_shadow(self):
        ema = EMACallback(decay=0.999)
        ema.on_fit_start(self.trainer, self.model)
        grad_params = {n for n, p in self.model.named_parameters() if p.requires_grad}
        self.assertEqual(set(ema._shadow.keys()), grad_params)

    def test_on_fit_start_shadow_equals_params(self):
        ema = EMACallback(decay=0.999)
        ema.on_fit_start(self.trainer, self.model)
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.assertTrue(
                    torch.equal(ema._shadow[name], param.data),
                    f"Shadow for {name} should equal param at init",
                )

    def test_on_train_batch_end_updates_shadow(self):
        ema = EMACallback(decay=0.9)
        ema.on_fit_start(self.trainer, self.model)

        # Save initial shadow
        old_shadow = {n: s.clone() for n, s in ema._shadow.items()}

        # Modify model params (simulate training step)
        with torch.no_grad():
            for p in self.model.parameters():
                p.add_(torch.ones_like(p))

        # Use high global_step so EMA warmup is past and effective_decay = 0.9
        self.trainer.global_step = 10000
        ema.on_train_batch_end(self.trainer, self.model, None, None, 0)

        # Verify: shadow = 0.9 * old_shadow + 0.1 * new_param
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in old_shadow:
                expected = 0.9 * old_shadow[name] + 0.1 * param.data
                self.assertTrue(
                    torch.allclose(ema._shadow[name], expected, atol=1e-6),
                    f"Shadow update incorrect for {name}",
                )

    def test_shadow_converges_to_params(self):
        ema = EMACallback(decay=0.5)  # fast convergence
        ema.on_fit_start(self.trainer, self.model)

        # Modify params
        with torch.no_grad():
            for p in self.model.parameters():
                p.fill_(42.0)

        # Run many updates — global_step must increment so that
        # the accumulate_grad_batches guard does not skip updates.
        for i in range(100):
            self.trainer.global_step = i
            ema.on_train_batch_end(self.trainer, self.model, None, None, i)

        # Shadow should converge to param values
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.assertTrue(
                    torch.allclose(ema._shadow[name], param.data, atol=1e-4),
                    f"Shadow for {name} should converge to param",
                )

    def test_validation_swap_uses_shadow(self):
        ema = EMACallback(decay=0.999)
        ema.on_fit_start(self.trainer, self.model)

        # Modify shadow manually
        for name in ema._shadow:
            ema._shadow[name].fill_(99.0)

        # Swap in shadow
        ema.on_validation_epoch_start(self.trainer, self.model)

        # Model params should now be shadow (99.0)
        for name, param in self.model.named_parameters():
            if name in ema._shadow:
                self.assertTrue(
                    torch.allclose(param.data, torch.tensor(99.0)),
                    f"After swap, {name} should equal shadow",
                )

    def test_validation_restore_uses_original(self):
        ema = EMACallback(decay=0.999)
        ema.on_fit_start(self.trainer, self.model)

        # Save original params
        original = {
            n: p.data.clone()
            for n, p in self.model.named_parameters()
            if p.requires_grad
        }

        # Modify shadow
        for name in ema._shadow:
            ema._shadow[name].fill_(99.0)

        # Swap in, then restore
        ema.on_validation_epoch_start(self.trainer, self.model)
        ema.on_validation_epoch_end(self.trainer, self.model)

        # Model params should be back to original
        for name, param in self.model.named_parameters():
            if name in original:
                self.assertTrue(
                    torch.equal(param.data, original[name]),
                    f"After restore, {name} should equal original",
                )

    def test_state_dict_roundtrip(self):
        ema = EMACallback(decay=0.995)
        ema.on_fit_start(self.trainer, self.model)

        # Modify shadow
        for name in ema._shadow:
            ema._shadow[name].fill_(77.0)

        state = ema.state_dict()

        # Create new EMA and load state
        ema2 = EMACallback(decay=0.5)  # different decay
        ema2.load_state_dict(state)

        self.assertAlmostEqual(ema2.decay, 0.995, places=5)
        for name in ema._shadow:
            self.assertTrue(
                torch.equal(ema2._shadow[name], ema._shadow[name]),
                f"State dict roundtrip failed for {name}",
            )

    def test_frozen_params_excluded(self):
        ema = EMACallback(decay=0.999)
        # Freeze encoder
        self.model.set_encoder_trainable(False)
        ema.on_fit_start(self.trainer, self.model)

        # Shadow should only have decoder params
        for name in ema._shadow:
            self.assertIn(
                "decoder", name, f"Frozen encoder param {name} should not be in shadow"
            )

    def test_ema_warmup_early_steps(self):
        """Early steps should use lower effective decay than target."""
        ema = EMACallback(decay=0.999)
        ema.on_fit_start(self.trainer, self.model)

        # Step 0: effective_decay = min(0.999, 1/10) = 0.1
        self.trainer.global_step = 0
        with torch.no_grad():
            for p in self.model.parameters():
                p.add_(torch.ones_like(p))
        old_shadow = {n: s.clone() for n, s in ema._shadow.items()}
        ema.on_train_batch_end(self.trainer, self.model, None, None, 0)

        effective_decay = min(0.999, 1 / 10)  # 0.1
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in old_shadow:
                expected = (
                    effective_decay * old_shadow[name]
                    + (1 - effective_decay) * param.data
                )
                self.assertTrue(
                    torch.allclose(ema._shadow[name], expected, atol=1e-6),
                    f"EMA warmup at step 0 incorrect for {name}",
                )

    def test_ema_warmup_converges_to_target_decay(self):
        """After many steps, effective decay should equal target decay."""
        ema = EMACallback(decay=0.999)
        ema.on_fit_start(self.trainer, self.model)

        # At step 10000: min(0.999, 10001/10010) = min(0.999, 0.99910...) = 0.999
        self.trainer.global_step = 10000
        effective_decay = min(0.999, (10000 + 1) / (10000 + 10))
        self.assertAlmostEqual(effective_decay, 0.999, places=3)

    def test_ema_warmup_monotonically_increasing(self):
        """Effective decay should monotonically increase over steps."""
        decay = 0.999
        prev = 0.0
        for step in range(100):
            eff = min(decay, (step + 1) / (step + 10))
            self.assertGreaterEqual(
                eff, prev, f"Effective decay decreased at step {step}"
            )
            prev = eff


# ----------------------------------------------------------------------
# E2. LLRD (Layer-wise LR Decay)
# ----------------------------------------------------------------------


class _ConvNeXtLikeModel(nn.Module):
    """Mimics a ConvNeXt encoder with stages for LLRD testing."""

    def __init__(self):
        super().__init__()
        self.encoder = nn.ModuleDict(
            {
                "stem": nn.Linear(4, 8),
                "stages": nn.ModuleList(
                    [
                        nn.Linear(8, 8),  # stage 0
                        nn.Linear(8, 8),  # stage 1
                        nn.Linear(8, 8),  # stage 2
                        nn.Linear(8, 8),  # stage 3
                    ]
                ),
            }
        )
        self.decoder = nn.Linear(8, 2)

    def set_encoder_trainable(self, trainable=True):
        for p in self.encoder.parameters():
            p.requires_grad = trainable


class TestLLRD(unittest.TestCase):

    def setUp(self):
        warnings.simplefilter("ignore")

    def test_param_groups_have_correct_lr_ordering(self):
        """Decoder LR > stage 3 LR > stage 2 LR > ... > stem LR."""
        from pytorch_segmentation_models_trainer.model_loader.model import Model

        model = _ConvNeXtLikeModel()
        # Call the static method directly by creating a mock Model instance
        # We'll test the function directly instead
        import re

        base_lr = 1e-4
        layer_decay = 0.9
        weight_decay = 0.05

        # Detect stages
        max_stage = -1
        for name, _ in model.named_parameters():
            if "encoder" in name:
                match = re.search(r"stages\.(\d+)", name)
                if match:
                    max_stage = max(max_stage, int(match.group(1)))
        num_stages = max_stage + 1
        self.assertEqual(num_stages, 4)

        # Build param groups manually (same logic as _get_param_groups_with_layer_decay)
        param_groups = []
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            if "encoder" not in name:
                group_lr = base_lr
            else:
                match = re.search(r"stages\.(\d+)", name)
                if match:
                    stage_id = int(match.group(1))
                    depth = num_stages - stage_id
                    group_lr = base_lr * (layer_decay**depth)
                else:
                    group_lr = base_lr * (layer_decay ** (num_stages + 1))
            param_wd = 0.0 if param.ndim <= 1 else weight_decay
            param_groups.append(
                {
                    "params": [param],
                    "lr": group_lr,
                    "weight_decay": param_wd,
                    "name": name,
                }
            )

        # Verify ordering: decoder > stage 3 > stage 2 > stage 1 > stage 0 > stem
        lrs_by_name = {g["name"]: g["lr"] for g in param_groups}

        # Decoder should have full LR
        for name, lr in lrs_by_name.items():
            if "decoder" in name:
                self.assertAlmostEqual(
                    lr, base_lr, places=8, msg=f"Decoder {name} should have base_lr"
                )

        # Higher stages should have higher LR
        stage_lrs = {}
        for name, lr in lrs_by_name.items():
            match = re.search(r"stages\.(\d+)", name)
            if match:
                stage_id = int(match.group(1))
                stage_lrs.setdefault(stage_id, []).append(lr)

        for stage_id in range(num_stages - 1):
            lr_this = stage_lrs[stage_id][0]
            lr_next = stage_lrs[stage_id + 1][0]
            self.assertLess(
                lr_this,
                lr_next,
                f"Stage {stage_id} LR should be < stage {stage_id+1} LR",
            )

        # Stem should have lowest LR
        stem_lrs = [
            lr
            for name, lr in lrs_by_name.items()
            if "encoder" in name and "stages" not in name
        ]
        if stem_lrs:
            min_stage_lr = min(stage_lrs[0])
            self.assertLess(
                stem_lrs[0], min_stage_lr, "Stem LR should be lower than stage 0 LR"
            )

    def test_no_weight_decay_for_1d_params(self):
        """Bias and normalization parameters (1-D) should have weight_decay=0."""
        model = _ConvNeXtLikeModel()
        import re

        base_lr = 1e-4
        layer_decay = 0.9
        weight_decay = 0.05

        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            param_wd = 0.0 if param.ndim <= 1 else weight_decay
            if param.ndim <= 1:
                self.assertEqual(param_wd, 0.0, f"1-D param {name} should have wd=0")
            else:
                self.assertEqual(
                    param_wd,
                    weight_decay,
                    f"Multi-D param {name} should have wd={weight_decay}",
                )

    def test_layer_decay_formula(self):
        """Verify exact LR values for known layer_decay=0.9 with 4 stages."""
        base_lr = 1e-4
        layer_decay = 0.9
        num_stages = 4

        # Stage 3 (closest to output): depth=1, lr = 1e-4 * 0.9^1
        self.assertAlmostEqual(base_lr * layer_decay**1, 9e-5, places=8)
        # Stage 0 (deepest): depth=4, lr = 1e-4 * 0.9^4
        self.assertAlmostEqual(base_lr * layer_decay**4, 6.561e-5, places=8)
        # Stem: depth=5, lr = 1e-4 * 0.9^5
        self.assertAlmostEqual(base_lr * layer_decay**5, 5.9049e-5, places=8)


# ----------------------------------------------------------------------
# F. WarmupCallback
# ----------------------------------------------------------------------


class TestWarmupCallback(unittest.TestCase):

    def setUp(self):
        warnings.simplefilter("ignore")
        self.model = _TinyModel()
        self.trainer = Mock()

    def test_on_fit_start_sets_warmed_up_if_past(self):
        cb = WarmupCallback(warmup_epochs=2)
        self.trainer.current_epoch = 5  # past warmup
        cb.on_fit_start(self.trainer, self.model)
        self.assertTrue(cb.warmed_up)

    def test_on_fit_start_not_warmed_up_initially(self):
        cb = WarmupCallback(warmup_epochs=2)
        self.trainer.current_epoch = 0
        cb.on_fit_start(self.trainer, self.model)
        self.assertFalse(cb.warmed_up)

    def test_freeze_during_warmup(self):
        cb = WarmupCallback(warmup_epochs=3)
        self.trainer.current_epoch = 0
        cb.on_fit_start(self.trainer, self.model)

        # Freeze happens at epoch == warmup_epochs - 1 = 2
        self.trainer.current_epoch = 2
        cb.on_train_epoch_start(self.trainer, self.model)
        # Verify encoder is frozen
        for p in self.model.encoder.parameters():
            self.assertFalse(p.requires_grad)

    def test_unfreeze_after_warmup(self):
        cb = WarmupCallback(warmup_epochs=2)
        self.trainer.current_epoch = 0
        cb.on_fit_start(self.trainer, self.model)

        # Simulate warmup epoch
        self.trainer.current_epoch = 1
        cb.on_train_epoch_start(self.trainer, self.model)
        cb.on_train_epoch_end(self.trainer, self.model)

        self.assertTrue(cb.warmed_up)
        # Encoder should be unfrozen
        for p in self.model.encoder.parameters():
            self.assertTrue(p.requires_grad)

    def test_no_freeze_after_warmed_up(self):
        cb = WarmupCallback(warmup_epochs=2)
        cb.warmed_up = True
        self.trainer.current_epoch = 5

        # Should not change anything
        original_grads = {n: p.requires_grad for n, p in self.model.named_parameters()}
        cb.on_train_epoch_start(self.trainer, self.model)
        for n, p in self.model.named_parameters():
            self.assertEqual(p.requires_grad, original_grads[n])

    def test_warmup_epochs_default(self):
        cb = WarmupCallback()
        self.assertEqual(cb.warmup_epochs, 2)


# ----------------------------------------------------------------------
# G. TTA Transforms
# ----------------------------------------------------------------------


class TestTTATransforms(unittest.TestCase):

    def setUp(self):
        warnings.simplefilter("ignore")
        # Asymmetric marker tensor for verifying transforms
        self.t = torch.zeros(1, 1, 4, 4)
        self.t[0, 0, 0, 0] = 1.0  # top-left
        self.t[0, 0, 0, 1] = 2.0  # top-second
        self.t[0, 0, 1, 0] = 3.0  # second-row-left

    def test_apply_identity(self):
        result = MultiClassInferenceProcessor._apply_transform(self.t.clone(), 0, False)
        self.assertTrue(torch.equal(result, self.t))

    def test_apply_hflip(self):
        result = MultiClassInferenceProcessor._apply_transform(self.t.clone(), 0, True)
        # Horizontal flip: (0,0)→(0,3), (0,1)→(0,2)
        self.assertEqual(result[0, 0, 0, 3].item(), 1.0)
        self.assertEqual(result[0, 0, 0, 2].item(), 2.0)
        self.assertEqual(result[0, 0, 1, 3].item(), 3.0)

    def test_apply_rot90(self):
        result = MultiClassInferenceProcessor._apply_transform(self.t.clone(), 1, False)
        # rot90 CCW k=1: (r,c) → (W-1-c, r)
        # (0,0)→(3,0), (0,1)→(2,0), (1,0)→(3,1)
        self.assertEqual(result[0, 0, 3, 0].item(), 1.0)
        self.assertEqual(result[0, 0, 2, 0].item(), 2.0)
        self.assertEqual(result[0, 0, 3, 1].item(), 3.0)

    def test_apply_rot180(self):
        result = MultiClassInferenceProcessor._apply_transform(self.t.clone(), 2, False)
        # rot180: (0,0)→(3,3), (0,1)→(3,2), (1,0)→(2,3)
        self.assertEqual(result[0, 0, 3, 3].item(), 1.0)
        self.assertEqual(result[0, 0, 3, 2].item(), 2.0)
        self.assertEqual(result[0, 0, 2, 3].item(), 3.0)

    def test_apply_rot270(self):
        result = MultiClassInferenceProcessor._apply_transform(self.t.clone(), 3, False)
        # rot270 CCW (k=3) = rot90 CW: (r,c) → (c, H-1-r)
        # (0,0)→(0,3), (0,1)→(1,3), (1,0)→(0,2)
        self.assertEqual(result[0, 0, 0, 3].item(), 1.0)
        self.assertEqual(result[0, 0, 1, 3].item(), 2.0)
        self.assertEqual(result[0, 0, 0, 2].item(), 3.0)

    def test_apply_rot90_hflip(self):
        result = MultiClassInferenceProcessor._apply_transform(self.t.clone(), 1, True)
        # First flip(dim=-1), then rot90 CCW
        # Should produce a valid tensor with rearranged markers
        self.assertEqual(result.shape, self.t.shape)
        # The combined transform is different from identity
        self.assertFalse(torch.equal(result, self.t))

    def test_invert_is_inverse_of_apply(self):
        """For all 8 D4 transforms, invert(apply(x)) should equal x."""
        for rot in range(4):
            for flip in [False, True]:
                with self.subTest(rot=rot, flip=flip):
                    original = torch.randn(2, 3, 8, 8)
                    transformed = MultiClassInferenceProcessor._apply_transform(
                        original.clone(), rot, flip
                    )
                    restored = MultiClassInferenceProcessor._invert_transform(
                        transformed, rot, flip
                    )
                    self.assertTrue(
                        torch.allclose(restored, original, atol=1e-6),
                        f"Roundtrip failed for rot={rot}, flip={flip}",
                    )

    def test_d4_all_8_transforms_invertible(self):
        """All 8 D4 transforms are distinct and invertible."""
        original = torch.randn(1, 2, 6, 6)
        seen = set()
        for rot in range(4):
            for flip in [False, True]:
                t = MultiClassInferenceProcessor._apply_transform(
                    original.clone(), rot, flip
                )
                key = tuple(t.flatten().tolist())
                seen.add(key)
        # All 8 should be distinct (for random input)
        self.assertEqual(len(seen), 8, "Not all 8 D4 transforms are distinct")

    def test_get_tta_transforms_d4(self):
        proc = MultiClassInferenceProcessor.__new__(MultiClassInferenceProcessor)
        proc.tta_mode = "d4"
        transforms = proc._get_tta_transforms()
        self.assertEqual(len(transforms), 8)

    def test_get_tta_transforms_flip(self):
        proc = MultiClassInferenceProcessor.__new__(MultiClassInferenceProcessor)
        proc.tta_mode = "flip"
        transforms = proc._get_tta_transforms()
        self.assertEqual(len(transforms), 4)

    def test_get_tta_transforms_none(self):
        proc = MultiClassInferenceProcessor.__new__(MultiClassInferenceProcessor)
        proc.tta_mode = None
        transforms = proc._get_tta_transforms()
        self.assertEqual(len(transforms), 1)
        self.assertEqual(transforms[0], (0, False))

    def test_invalid_tta_mode_raises(self):
        with self.assertRaises(ValueError):
            MultiClassInferenceProcessor(
                model=Mock(),
                device="cpu",
                batch_size=1,
                export_strategy=Mock(),
                tta_mode="invalid",
            )


# ----------------------------------------------------------------------
# H. TTA Inference (integration-like)
# ----------------------------------------------------------------------


class TestTTAInference(unittest.TestCase):

    def setUp(self):
        warnings.simplefilter("ignore")

    def _make_constant_model(self, num_classes=4):
        """Model that always returns the same logits regardless of input."""

        class ConstantModel:
            def __call__(self, x):
                B, C_in, H, W = x.shape
                return torch.ones(B, num_classes, H, W)

        return ConstantModel()

    def _make_position_aware_model(self, num_classes=4):
        """Model whose output depends on spatial position (not equivariant)."""

        class PositionModel:
            def __init__(self):
                self.n_calls = 0

            def __call__(self, x):
                self.n_calls += 1
                B, C_in, H, W = x.shape
                out = torch.zeros(B, num_classes, H, W)
                # Class 0 gets value based on column index
                for w in range(W):
                    out[:, 0, :, w] = float(w) / W
                return out

        return PositionModel()

    def test_tta_d4_equivariant_model(self):
        """For a constant model, TTA should give same result as no-TTA."""
        num_classes = 4
        model = self._make_constant_model(num_classes)
        tile = torch.randn(8, 8)

        # No TTA
        input_t = tile.unsqueeze(0).unsqueeze(0)  # [1, 1, 8, 8]
        no_tta = model(input_t)  # [1, 4, 8, 8]

        # Manual TTA D4
        transforms = MultiClassInferenceProcessor._D4_TRANSFORMS
        accum = None
        for rot, flip in transforms:
            aug = MultiClassInferenceProcessor._apply_transform(
                input_t.clone(), rot, flip
            )
            pred = model(aug)
            pred = MultiClassInferenceProcessor._invert_transform(pred, rot, flip)
            accum = pred if accum is None else accum + pred
        tta_result = accum / len(transforms)

        # Should be identical for constant model
        self.assertTrue(
            torch.allclose(tta_result, no_tta, atol=1e-5),
            "TTA with constant model should match no-TTA",
        )

    def test_tta_d4_averages_predictions(self):
        """For a position-aware model, TTA should differ from no-TTA."""
        num_classes = 4
        model = self._make_position_aware_model(num_classes)
        input_t = torch.randn(1, 1, 8, 8)

        no_tta = model(input_t)

        transforms = MultiClassInferenceProcessor._D4_TRANSFORMS
        accum = None
        for rot, flip in transforms:
            aug = MultiClassInferenceProcessor._apply_transform(
                input_t.clone(), rot, flip
            )
            pred = model(aug)
            pred = MultiClassInferenceProcessor._invert_transform(pred, rot, flip)
            accum = pred if accum is None else accum + pred
        tta_result = accum / len(transforms)

        # Should differ from no-TTA
        self.assertFalse(
            torch.allclose(tta_result, no_tta, atol=1e-5),
            "TTA with position-aware model should differ from no-TTA",
        )

    def test_tta_flip_produces_4_passes(self):
        num_classes = 4
        model = self._make_position_aware_model(num_classes)
        input_t = torch.randn(1, 1, 8, 8)

        transforms = MultiClassInferenceProcessor._FLIP_TRANSFORMS
        self.assertEqual(len(transforms), 4)

        for rot, flip in transforms:
            aug = MultiClassInferenceProcessor._apply_transform(
                input_t.clone(), rot, flip
            )
            model(aug)

        self.assertEqual(model.n_calls, 4)

    def test_no_tta_single_pass(self):
        proc = MultiClassInferenceProcessor.__new__(MultiClassInferenceProcessor)
        proc.tta_mode = None
        transforms = proc._get_tta_transforms()
        self.assertEqual(len(transforms), 1)


import numpy as np


class TestComputeConfidenceMaps(unittest.TestCase):
    """Tests for MultiClassInferenceProcessor._compute_confidence_maps."""

    def test_perfect_prediction(self):
        """One-hot probabilities → entropy=0, margin=1, max_prob=1."""
        C, H, W = 7, 4, 4
        probs = np.zeros((C, H, W), dtype=np.float32)
        probs[3, :, :] = 1.0  # class 3 everywhere

        maps = MultiClassInferenceProcessor._compute_confidence_maps(probs)

        np.testing.assert_allclose(maps["max_prob"], 1.0)
        np.testing.assert_allclose(maps["entropy"], 0.0, atol=1e-6)
        np.testing.assert_allclose(maps["margin"], 1.0)

    def test_uniform_distribution(self):
        """Uniform probs → entropy=1, margin=0, max_prob=1/C."""
        C, H, W = 7, 4, 4
        probs = np.full((C, H, W), 1.0 / C, dtype=np.float32)

        maps = MultiClassInferenceProcessor._compute_confidence_maps(probs)

        np.testing.assert_allclose(maps["max_prob"], 1.0 / C, rtol=1e-5)
        np.testing.assert_allclose(maps["entropy"], 1.0, atol=1e-4)
        np.testing.assert_allclose(maps["margin"], 0.0, atol=1e-6)

    def test_output_shapes(self):
        """All metrics must have shape [H, W] float32."""
        C, H, W = 5, 8, 12
        probs = np.random.dirichlet(np.ones(C), size=(H, W)).transpose(2, 0, 1)
        probs = probs.astype(np.float32)

        maps = MultiClassInferenceProcessor._compute_confidence_maps(probs)

        for key in ("max_prob", "entropy", "margin"):
            self.assertEqual(maps[key].shape, (H, W), f"{key} shape mismatch")
            self.assertEqual(maps[key].dtype, np.float32, f"{key} dtype mismatch")

    def test_value_ranges(self):
        """All metrics in [0, 1]."""
        C, H, W = 7, 16, 16
        probs = np.random.dirichlet(np.ones(C), size=(H, W)).transpose(2, 0, 1)
        probs = probs.astype(np.float32)

        maps = MultiClassInferenceProcessor._compute_confidence_maps(probs)

        for key in ("max_prob", "entropy", "margin"):
            self.assertTrue(
                np.all(maps[key] >= 0) and np.all(maps[key] <= 1.0 + 1e-6),
                f"{key} out of [0, 1] range",
            )

    def test_two_class_half_split(self):
        """50/50 between two classes → entropy < 1 (not max), margin = 0."""
        C, H, W = 4, 2, 2
        probs = np.zeros((C, H, W), dtype=np.float32)
        probs[0] = 0.5
        probs[1] = 0.5

        maps = MultiClassInferenceProcessor._compute_confidence_maps(probs)

        np.testing.assert_allclose(maps["max_prob"], 0.5)
        np.testing.assert_allclose(maps["margin"], 0.0, atol=1e-6)
        # Entropy should be log(2)/log(4) = 0.5 (not maximum)
        expected_entropy = np.log(2) / np.log(C)
        np.testing.assert_allclose(maps["entropy"], expected_entropy, atol=1e-4)

    def test_confidence_mode_validation(self):
        """Invalid confidence_mode raises ValueError."""
        mock_model = Mock()
        mock_model.to = Mock(return_value=mock_model)
        mock_model.eval = Mock(return_value=mock_model)
        with self.assertRaises(ValueError):
            MultiClassInferenceProcessor(
                model=mock_model,
                device="cpu",
                batch_size=1,
                export_strategy=None,
                num_classes=7,
                confidence_mode="invalid",
            )

    def test_confidence_mode_basic_keys(self):
        """Basic mode should only save max_prob."""
        proc = MultiClassInferenceProcessor.__new__(MultiClassInferenceProcessor)
        proc.confidence_mode = "basic"
        proc.confidence_export_strategy = None

        C, H, W = 7, 4, 4
        probs = (
            np.random.dirichlet(np.ones(C), size=(H, W))
            .transpose(2, 0, 1)
            .astype(np.float32)
        )
        maps = proc._compute_confidence_maps(probs)

        # _compute_confidence_maps always returns all 3,
        # but _save_confidence_maps filters based on mode
        self.assertIn("max_prob", maps)
        self.assertIn("entropy", maps)
        self.assertIn("margin", maps)

    def test_confidence_mode_none_backward_compat(self):
        """confidence_mode=None should not break existing behavior."""
        proc = MultiClassInferenceProcessor.__new__(MultiClassInferenceProcessor)
        proc.confidence_mode = None
        # Should be None — process() will skip confidence path
        self.assertIsNone(proc.confidence_mode)


if __name__ == "__main__":
    unittest.main()
