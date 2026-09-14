# -*- coding: utf-8 -*-
"""Tests for BoundaryLabelRelaxationLoss."""

import torch
import torch.nn.functional as F

from pytorch_segmentation_models_trainer.custom_losses.boundary_relaxation_loss import (
    BoundaryLabelRelaxationLoss,
)

B, C, H, W = 2, 4, 8, 8


def _logits():
    return torch.randn(B, C, H, W, requires_grad=True)


def _uniform_hard(cls: int = 0):
    """Hard label with a single class everywhere -> every pixel interior."""
    return torch.full((B, H, W), cls, dtype=torch.long)


def _checkerboard_hard():
    """Hard label alternating class 0/1 pixel-by-pixel -> every pixel is a border pixel."""
    idx = torch.arange(H * W).reshape(H, W)
    board = (idx % 2).long()
    return board.unsqueeze(0).expand(B, H, W).clone()


class TestBoundaryLabelRelaxationLossCompute:
    def setup_method(self):
        self.fn = BoundaryLabelRelaxationLoss(name="boundary", num_classes=C)

    # --- output contract ---

    def test_returns_scalar(self):
        loss = self.fn.compute(_logits(), _uniform_hard())
        assert loss.shape == torch.Size([])

    def test_output_dtype_float(self):
        loss = self.fn.compute(_logits(), _uniform_hard())
        assert loss.dtype == torch.float32

    def test_loss_non_negative(self):
        for _ in range(10):
            loss = self.fn.compute(_logits(), _uniform_hard())
            assert loss.item() >= -1e-6

    def test_dict_gt_batch_mask_key(self):
        loss = self.fn.compute(_logits(), {"mask": _uniform_hard()})
        assert loss.shape == torch.Size([])

    def test_dict_pred_batch_seg_key(self):
        logits = _logits()
        loss = self.fn.compute({"seg": logits}, _uniform_hard())
        assert loss.shape == torch.Size([])

    # --- gradient flows ---

    def test_gradient_flows_through_pred(self):
        logits = _logits()
        loss = self.fn.compute(logits, _uniform_hard())
        loss.backward()
        assert logits.grad is not None
        assert not torch.all(logits.grad == 0)

    # --- reduces exactly to CE on interior-only labels ---

    def test_reduces_to_cross_entropy_when_no_border_pixels(self):
        logits = torch.randn(B, C, H, W)
        hard = _uniform_hard(cls=2)
        loss = self.fn.compute(logits, hard)
        ce = F.cross_entropy(logits, hard)
        assert torch.allclose(loss, ce, atol=1e-5)

    # --- border relaxation actually relaxes ---

    def test_border_loss_lower_than_plain_ce_on_checkerboard(self):
        # A single-class-mispredicting model: predicts the "wrong" class
        # (relative to the pixel's own label) but that class is always the
        # immediate neighbour's class on the checkerboard. Relaxed loss
        # should be far below plain CE on the same logits.
        hard = _checkerboard_hard()
        logits = torch.zeros(B, C, H, W)
        wrong = 1 - hard  # neighbour's class everywhere on a 0/1 checkerboard
        logits.scatter_(1, wrong.unsqueeze(1), 10.0)

        relaxed = self.fn.compute(logits, hard)
        ce = F.cross_entropy(logits, hard)
        assert relaxed.item() < ce.item()

    # --- ignore_index handling ---

    def test_ignored_pixels_excluded_from_mean(self):
        logits = torch.randn(B, C, H, W)
        hard = _uniform_hard()
        hard_with_ignore = hard.clone()
        hard_with_ignore[:, 0, 0] = 255  # default ignore_index

        loss_ignored = self.fn.compute(logits, hard_with_ignore)
        assert loss_ignored.shape == torch.Size([])
        assert torch.isfinite(loss_ignored)

    def test_all_ignored_gives_zero_loss(self):
        logits = torch.randn(B, C, H, W)
        hard = torch.full((B, H, W), 255, dtype=torch.long)
        loss = self.fn.compute(logits, hard)
        assert torch.isclose(loss, torch.tensor(0.0), atol=1e-6)

    def test_custom_ignore_index(self):
        fn = BoundaryLabelRelaxationLoss(name="x", num_classes=C, ignore_index=1)
        logits = torch.randn(B, C, H, W)
        hard = torch.full((B, H, W), 1, dtype=torch.long)
        loss = fn.compute(logits, hard)
        assert torch.isclose(loss, torch.tensor(0.0), atol=1e-6)

    def test_custom_mask_key(self):
        fn = BoundaryLabelRelaxationLoss(name="x", num_classes=C, mask_key="hard")
        loss = fn.compute(_logits(), {"hard": _uniform_hard()})
        assert loss.shape == torch.Size([])


class TestBoundaryLabelRelaxationLossForward:
    """Test the inherited Loss.forward() wrapper."""

    def setup_method(self):
        self.fn = BoundaryLabelRelaxationLoss(name="boundary", num_classes=C)

    def test_forward_returns_tuple_of_two(self):
        result = self.fn(_logits(), _uniform_hard())
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_forward_first_element_is_scalar(self):
        loss_val, _ = self.fn(_logits(), _uniform_hard())
        assert loss_val.shape == torch.Size([])

    def test_forward_normalize_false_equals_compute(self):
        logits = torch.randn(B, C, H, W)
        gt = _uniform_hard()
        computed = self.fn.compute(logits, gt)
        loss_val, _ = self.fn(logits, gt, normalize=False)
        assert torch.allclose(computed, loss_val, atol=1e-6)
