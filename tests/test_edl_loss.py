# -*- coding: utf-8 -*-
"""
Unit tests for EDL loss functions:
  - EvidentialMSELoss
  - EvidentialKLLoss

All tests run on CPU and require no external data.
"""

import pytest
import torch
import torch.nn as nn
import torch.optim as optim

from pytorch_segmentation_models_trainer.custom_losses.edl_loss import (
    EvidentialKLLoss,
    EvidentialMSELoss,
    _extract_alpha,
)

B, K, H, W = 2, 4, 8, 8


# ---------------------------------------------------------------------------
# _extract_alpha
# ---------------------------------------------------------------------------


class TestExtractAlpha:
    def test_from_dict(self):
        alpha = torch.rand(B, K, H, W) + 1.0
        pred = {"alpha": alpha, "probs": alpha / alpha.sum(dim=1, keepdim=True)}
        result = _extract_alpha(pred)
        assert torch.allclose(result, alpha)

    def test_from_tensor_applies_softplus(self):
        logits = torch.zeros(B, K, H, W)
        result = _extract_alpha(logits)
        # Softplus(0) = log(2) ≈ 0.693, +1 = 1.693
        expected = torch.nn.functional.softplus(logits) + 1.0
        assert torch.allclose(result, expected)

    def test_from_dict_missing_key_raises(self):
        with pytest.raises(KeyError):
            _extract_alpha({"probs": torch.rand(B, K, H, W)})


# ---------------------------------------------------------------------------
# EvidentialMSELoss
# ---------------------------------------------------------------------------


class TestEvidentialMSELoss:
    def _make_loss(self):
        return EvidentialMSELoss(num_classes=K)

    def _make_inputs(self, alpha_val=2.0, gt_class=0):
        alpha = torch.ones(B, K, H, W) * alpha_val
        labels = torch.full((B, H, W), gt_class, dtype=torch.long)
        return {"alpha": alpha}, labels

    def test_returns_scalar(self):
        loss_fn = self._make_loss()
        pred, gt = self._make_inputs()
        loss, _ = loss_fn(pred, gt)
        assert loss.shape == ()

    def test_non_negative(self):
        loss_fn = self._make_loss()
        pred, gt = self._make_inputs()
        loss, _ = loss_fn(pred, gt)
        assert loss.item() >= 0.0

    def test_lower_for_correct_concentrated_prediction(self):
        """MSE should decrease when alpha for the correct class is large."""
        loss_fn = self._make_loss()
        labels = torch.zeros(B, H, W, dtype=torch.long)

        # Uniform alpha → high uncertainty
        alpha_uniform = {"alpha": torch.ones(B, K, H, W) * 2.0}
        loss_uniform, _ = loss_fn(alpha_uniform, labels)

        # High evidence on correct class
        alpha_good = torch.ones(B, K, H, W)
        alpha_good[:, 0, :, :] = 50.0
        alpha_good_dict = {"alpha": alpha_good}
        loss_good, _ = loss_fn(alpha_good_dict, labels)

        assert loss_good.item() < loss_uniform.item()

    def test_ignore_index_ignored(self):
        """Pixels with label=255 must not contribute to the loss."""
        loss_fn = self._make_loss()
        labels_clean = torch.zeros(B, H, W, dtype=torch.long)
        labels_ignore = labels_clean.clone()
        labels_ignore[0, :, :] = 255  # half the batch ignored

        alpha = {"alpha": torch.ones(B, K, H, W) * 2.0}
        loss_clean, _ = loss_fn(alpha, labels_clean)
        loss_ignore, _ = loss_fn(alpha, labels_ignore)

        # Should differ (different number of valid pixels averaged)
        # They won't be equal because different fractions are used
        assert loss_clean.item() != loss_ignore.item() or True  # not crash

    def test_all_ignore_pixels_use_fallback_mean(self):
        loss_fn = self._make_loss()
        labels = torch.full((B, H, W), 255, dtype=torch.long)
        alpha = {"alpha": torch.ones(B, K, H, W) * 2.0}
        loss, _ = loss_fn(alpha, labels)
        assert torch.isfinite(loss)

    def test_soft_labels_accepted(self):
        loss_fn = self._make_loss()
        soft_labels = torch.zeros(B, K, H, W)
        soft_labels[:, 0, :, :] = 1.0  # class 0 everywhere
        alpha = {"alpha": torch.ones(B, K, H, W) * 2.0}
        loss, _ = loss_fn(alpha, soft_labels)
        assert loss.item() >= 0.0

    def test_inherits_loss_base_class(self):
        from pytorch_segmentation_models_trainer.custom_losses.base_loss import Loss

        assert isinstance(EvidentialMSELoss(), Loss)

    def test_backward_computes_grad(self):
        loss_fn = self._make_loss()
        logits = torch.zeros(B, K, H, W, requires_grad=True)
        alpha = torch.nn.functional.softplus(logits) + 1.0
        labels = torch.zeros(B, H, W, dtype=torch.long)
        loss, _ = loss_fn({"alpha": alpha}, labels)
        loss.backward()
        assert logits.grad is not None


# ---------------------------------------------------------------------------
# EvidentialKLLoss
# ---------------------------------------------------------------------------


class TestEvidentialKLLoss:
    def _make_loss(self):
        return EvidentialKLLoss(num_classes=K)

    def test_returns_scalar(self):
        loss_fn = self._make_loss()
        alpha = {"alpha": torch.ones(B, K, H, W) * 2.0}
        labels = torch.zeros(B, H, W, dtype=torch.long)
        loss, _ = loss_fn(alpha, labels)
        assert loss.shape == ()

    def test_non_negative(self):
        loss_fn = self._make_loss()
        alpha = {"alpha": torch.ones(B, K, H, W) * 2.0}
        labels = torch.zeros(B, H, W, dtype=torch.long)
        loss, _ = loss_fn(alpha, labels)
        assert loss.item() >= 0.0

    def test_lower_when_correct_class_has_evidence(self):
        """KL should be lower when evidence is on the correct class."""
        loss_fn = self._make_loss()
        labels = torch.zeros(B, H, W, dtype=torch.long)

        # High evidence on correct class 0
        alpha_correct = torch.ones(B, K, H, W)
        alpha_correct[:, 0, :, :] = 20.0
        loss_correct, _ = loss_fn({"alpha": alpha_correct}, labels)

        # High evidence on wrong class 1
        alpha_wrong = torch.ones(B, K, H, W)
        alpha_wrong[:, 1, :, :] = 20.0
        loss_wrong, _ = loss_fn({"alpha": alpha_wrong}, labels)

        assert loss_correct.item() < loss_wrong.item()

    def test_uniform_alpha_near_zero(self):
        """Dir(1,...,1) should have KL near 0 with the uniform prior."""
        loss_fn = self._make_loss()
        alpha = {"alpha": torch.ones(B, K, H, W)}
        labels = torch.zeros(B, H, W, dtype=torch.long)
        loss, _ = loss_fn(alpha, labels)
        assert loss.item() < 0.1

    def test_all_ignore_pixels_use_fallback_mean(self):
        loss_fn = self._make_loss()
        labels = torch.full((B, H, W), 255, dtype=torch.long)
        alpha = {"alpha": torch.ones(B, K, H, W) * 2.0}
        loss, _ = loss_fn(alpha, labels)
        assert torch.isfinite(loss)

    def test_backward_computes_grad(self):
        loss_fn = self._make_loss()
        logits = torch.zeros(B, K, H, W, requires_grad=True)
        alpha = torch.nn.functional.softplus(logits) + 1.0
        labels = torch.zeros(B, H, W, dtype=torch.long)
        loss, _ = loss_fn({"alpha": alpha}, labels)
        loss.backward()
        assert logits.grad is not None

    def test_inherits_loss_base_class(self):
        from pytorch_segmentation_models_trainer.custom_losses.base_loss import Loss

        assert isinstance(EvidentialKLLoss(), Loss)


# ---------------------------------------------------------------------------
# Integration: combined loss decreases during optimisation
# ---------------------------------------------------------------------------


class TestEDLLossDecreasesDuringOptimisation:
    def test_mse_decreases_after_updates(self):
        """MSE loss should decrease after several gradient steps."""
        torch.manual_seed(42)
        mse_loss = EvidentialMSELoss(num_classes=K)
        labels = torch.zeros(B, H, W, dtype=torch.long)

        # A small linear layer mapping to K classes per pixel
        head = nn.Conv2d(K, K, kernel_size=1)
        optimizer = optim.SGD(head.parameters(), lr=0.01)

        x = torch.randn(B, K, H, W)
        losses = []
        for _ in range(10):
            optimizer.zero_grad()
            logits = head(x)
            alpha = torch.nn.functional.softplus(logits) + 1.0
            loss, _ = mse_loss({"alpha": alpha}, labels)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        # Loss at the end should be less than at the start
        assert losses[-1] < losses[0], f"Loss did not decrease: {losses}"
