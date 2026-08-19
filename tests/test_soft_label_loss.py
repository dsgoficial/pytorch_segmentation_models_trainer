# -*- coding: utf-8 -*-
"""Tests for SoftLabelWeightedCELoss."""

import torch

from pytorch_segmentation_models_trainer.custom_losses.soft_label_loss import (
    SoftLabelWeightedCELoss,
)

B, C, H, W = 2, 4, 8, 8


def _logits():
    return torch.randn(B, C, H, W, requires_grad=True)


def _p_soft():
    raw = torch.rand(B, C, H, W)
    return (raw / raw.sum(dim=1, keepdim=True)).detach()


def _w_conf():
    return torch.rand(B, 1, H, W).detach()


class TestSoftLabelWeightedCELossCompute:
    def setup_method(self):
        self.fn = SoftLabelWeightedCELoss(name="soft_ce", num_classes=C)

    # --- output contract ---

    def test_returns_scalar_with_w_conf(self):
        loss = self.fn.compute(_logits(), {"mask": _p_soft(), "w_conf": _w_conf()})
        assert loss.shape == torch.Size([])

    def test_returns_scalar_without_w_conf(self):
        loss = self.fn.compute(_logits(), {"mask": _p_soft()})
        assert loss.shape == torch.Size([])

    def test_returns_scalar_plain_tensor_gt(self):
        loss = self.fn.compute(_logits(), _p_soft())
        assert loss.shape == torch.Size([])

    def test_output_dtype_float(self):
        loss = self.fn.compute(_logits(), {"mask": _p_soft(), "w_conf": _w_conf()})
        assert loss.dtype == torch.float32

    # --- gradient flows ---

    def test_gradient_flows_through_pred(self):
        logits = _logits()
        loss = self.fn.compute(logits, {"mask": _p_soft(), "w_conf": _w_conf()})
        loss.backward()
        assert logits.grad is not None
        assert not torch.all(logits.grad == 0)

    def test_gradient_flows_no_w_conf(self):
        logits = _logits()
        loss = self.fn.compute(logits, {"mask": _p_soft()})
        loss.backward()
        assert logits.grad is not None

    # --- weight semantics ---

    def test_uniform_weight_equals_unweighted(self):
        logits = torch.randn(B, C, H, W)
        p = _p_soft()
        w_ones = torch.ones(B, 1, H, W)
        loss_weighted = self.fn.compute(logits, {"mask": p, "w_conf": w_ones})
        loss_plain = self.fn.compute(logits, {"mask": p})
        assert torch.allclose(loss_weighted, loss_plain, atol=1e-5)

    def test_zero_weight_gives_zero_loss(self):
        w_zero = torch.zeros(B, 1, H, W)
        loss = self.fn.compute(_logits(), {"mask": _p_soft(), "w_conf": w_zero})
        assert torch.isclose(loss, torch.tensor(0.0), atol=1e-6)

    def test_partial_mask_uses_only_unmasked_pixels(self):
        logits = torch.randn(B, C, H, W)
        p = _p_soft()
        w_half = torch.zeros(B, 1, H, W)
        w_half[:, :, : H // 2, :] = 1.0
        loss_half = self.fn.compute(logits, {"mask": p, "w_conf": w_half})
        assert loss_half.item() >= 0.0

    # --- correctness direction ---

    def test_loss_decreases_with_better_prediction(self):
        # Use a peaked (non-uniform) p_soft so the better-logits gain is unambiguous.
        # One class gets 90 % of probability mass, ensuring argmax-based logits
        # clearly outperform uniform logits.
        hard = torch.randint(0, C, (B, H, W))  # class indices
        p = torch.zeros(B, C, H, W)
        p.scatter_(1, hard.unsqueeze(1), 0.9)
        remainder = (1.0 - 0.9) / (C - 1)
        p = p + remainder
        p[p == remainder] = remainder  # already set, but explicit
        p[p > remainder] = 0.9
        # Recompute cleanly
        p = torch.full((B, C, H, W), remainder)
        p.scatter_(1, hard.unsqueeze(1), 0.9)
        p = p / p.sum(dim=1, keepdim=True)  # normalise numerically

        logits_bad = torch.zeros(B, C, H, W)  # uniform → CE = log(C)
        # Good logits: high value at argmax class
        logits_good = torch.zeros(B, C, H, W)
        logits_good.scatter_(1, hard.unsqueeze(1), 10.0)

        loss_bad = self.fn.compute(logits_bad, {"mask": p})
        loss_good = self.fn.compute(logits_good, {"mask": p})
        assert loss_good.item() < loss_bad.item()

    def test_loss_non_negative(self):
        for _ in range(10):
            loss = self.fn.compute(_logits(), {"mask": _p_soft(), "w_conf": _w_conf()})
            assert loss.item() >= -1e-6

    # --- dict pred_batch (FrameField path) ---

    def test_dict_pred_batch_seg_key(self):
        logits = _logits()
        loss = self.fn.compute(
            {"seg": logits}, {"mask": _p_soft(), "w_conf": _w_conf()}
        )
        assert loss.shape == torch.Size([])

    def test_dict_pred_and_dict_gt_gradient(self):
        logits = _logits()
        loss = self.fn.compute(
            {"seg": logits}, {"mask": _p_soft(), "w_conf": _w_conf()}
        )
        loss.backward()
        assert logits.grad is not None

    # --- custom keys ---

    def test_custom_mask_key(self):
        fn = SoftLabelWeightedCELoss(name="x", num_classes=C, mask_key="p_soft")
        loss = fn.compute(_logits(), {"p_soft": _p_soft()})
        assert loss.shape == torch.Size([])

    def test_custom_weight_key(self):
        fn = SoftLabelWeightedCELoss(name="x", num_classes=C, weight_key="confidence")
        loss = fn.compute(_logits(), {"mask": _p_soft(), "confidence": _w_conf()})
        assert loss.shape == torch.Size([])


class TestSoftLabelWeightedCELossForward:
    """Test the inherited Loss.forward() wrapper."""

    def setup_method(self):
        self.fn = SoftLabelWeightedCELoss(name="soft_ce", num_classes=C)

    def test_forward_returns_tuple_of_two(self):
        result = self.fn(_logits(), {"mask": _p_soft(), "w_conf": _w_conf()})
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_forward_first_element_is_scalar(self):
        loss_val, _ = self.fn(_logits(), {"mask": _p_soft(), "w_conf": _w_conf()})
        assert loss_val.shape == torch.Size([])

    def test_forward_second_element_is_dict(self):
        _, extra = self.fn(_logits(), {"mask": _p_soft(), "w_conf": _w_conf()})
        assert isinstance(extra, dict)

    def test_forward_normalize_false_equals_compute(self):
        logits = torch.randn(B, C, H, W)
        gt = {"mask": _p_soft(), "w_conf": _w_conf()}
        computed = self.fn.compute(logits, gt)
        loss_val, _ = self.fn(logits, gt, normalize=False)
        assert torch.allclose(computed, loss_val, atol=1e-6)
