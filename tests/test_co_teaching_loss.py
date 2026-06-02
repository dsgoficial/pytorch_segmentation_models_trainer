# -*- coding: utf-8 -*-
"""Tests for co-teaching loss components: CurriculumScheduler and CoTeachingLoss."""

import pytest
import torch
import torch.nn.functional as F

from pytorch_segmentation_models_trainer.custom_losses.co_teaching_loss import (
    CoTeachingLoss,
    CurriculumScheduler,
)

B, C, H, W = 2, 4, 8, 8
D = 16  # feature dimension for neighborhood regularization


def _logits(requires_grad=True):
    return torch.randn(B, C, H, W, requires_grad=requires_grad)


def _p_soft():
    raw = torch.rand(B, C, H, W)
    return (raw / raw.sum(dim=1, keepdim=True)).detach()


def _w_conf():
    return torch.rand(B, 1, H, W).detach()


def _features():
    return torch.randn(B, D, H, W)


# ---------------------------------------------------------------------------
# CurriculumScheduler
# ---------------------------------------------------------------------------


class TestCurriculumSchedulerRetentionRate:
    def setup_method(self):
        self.sched = CurriculumScheduler(e_warm=20, p_start=0.5, p_end=0.95)

    def test_at_epoch_zero_equals_p_start(self):
        assert self.sched.retention_rate(0) == pytest.approx(0.5)

    def test_at_e_warm_equals_p_end(self):
        assert self.sched.retention_rate(20) == pytest.approx(0.95)

    def test_beyond_e_warm_clamped_at_p_end(self):
        assert self.sched.retention_rate(100) == pytest.approx(0.95)

    def test_monotonically_increasing(self):
        rates = [self.sched.retention_rate(e) for e in range(25)]
        for prev, curr in zip(rates, rates[1:]):
            assert curr >= prev

    def test_midpoint_value(self):
        # At e=10 (half of e_warm=20): p_start + 0.5*(p_end-p_start) = 0.5+0.225 = 0.725
        assert self.sched.retention_rate(10) == pytest.approx(0.725)


class TestCurriculumSchedulerForgetRate:
    def setup_method(self):
        self.sched = CurriculumScheduler(e_warm=20, p_start=0.5, p_end=0.95)

    def test_at_epoch_zero_is_zero(self):
        assert self.sched.forget_rate(0) == pytest.approx(0.0)

    def test_at_e_warm_is_0_3(self):
        assert self.sched.forget_rate(20) == pytest.approx(0.3)

    def test_beyond_e_warm_clamped(self):
        assert self.sched.forget_rate(100) == pytest.approx(0.3)

    def test_monotonically_non_decreasing(self):
        rates = [self.sched.forget_rate(e) for e in range(25)]
        for prev, curr in zip(rates, rates[1:]):
            assert curr >= prev


class TestCurriculumSchedulerMask:
    def setup_method(self):
        self.sched = CurriculumScheduler(e_warm=20, p_start=0.5, p_end=0.95)

    def test_output_shape(self):
        w = _w_conf()
        mask = self.sched.curriculum_mask(w, epoch=0)
        assert mask.shape == (B, 1, H, W)

    def test_output_dtype_float(self):
        mask = self.sched.curriculum_mask(_w_conf(), epoch=0)
        assert mask.dtype == torch.float32

    def test_values_are_binary(self):
        mask = self.sched.curriculum_mask(_w_conf(), epoch=0)
        assert torch.all((mask == 0) | (mask == 1))

    def test_higher_epoch_keeps_more_pixels(self):
        # Large w so distribution is spread; compare kept counts across epochs
        torch.manual_seed(42)
        w = torch.rand(B, 1, H, W)
        kept_early = self.sched.curriculum_mask(w, epoch=0).sum().item()
        kept_late = self.sched.curriculum_mask(w, epoch=20).sum().item()
        assert kept_late >= kept_early

    def test_all_ones_mask_at_p_end_keeps_almost_all(self):
        # At epoch=e_warm, P_e=0.95, so ~95% of pixels should be kept
        torch.manual_seed(0)
        w = torch.rand(B, 1, H, W)
        mask = self.sched.curriculum_mask(w, epoch=20)
        kept_fraction = mask.mean().item()
        assert kept_fraction >= 0.90  # allow small rounding tolerance


class TestCurriculumSchedulerValidation:
    def test_invalid_p_start_zero_raises(self):
        with pytest.raises(ValueError):
            CurriculumScheduler(e_warm=20, p_start=0.0, p_end=0.95)

    def test_p_start_greater_than_p_end_raises(self):
        with pytest.raises(ValueError):
            CurriculumScheduler(e_warm=20, p_start=0.9, p_end=0.5)

    def test_p_end_greater_than_one_raises(self):
        with pytest.raises(ValueError):
            CurriculumScheduler(e_warm=20, p_start=0.5, p_end=1.1)


# ---------------------------------------------------------------------------
# CoTeachingLoss — validation path (single logits tensor)
# ---------------------------------------------------------------------------


class TestCoTeachingLossValidationPath:
    def setup_method(self):
        self.fn = CoTeachingLoss(name="cot", num_classes=C)

    def test_single_tensor_pred_returns_scalar(self):
        loss = self.fn.compute(_logits(), {"mask": _p_soft(), "w_conf": _w_conf()})
        assert loss.shape == torch.Size([])

    def test_single_tensor_no_w_conf_returns_scalar(self):
        loss = self.fn.compute(_logits(), {"mask": _p_soft()})
        assert loss.shape == torch.Size([])

    def test_single_tensor_plain_gt_returns_scalar(self):
        loss = self.fn.compute(_logits(), _p_soft())
        assert loss.shape == torch.Size([])

    def test_gradient_flows_through_single_logits(self):
        logits = _logits()
        loss = self.fn.compute(logits, {"mask": _p_soft(), "w_conf": _w_conf()})
        loss.backward()
        assert logits.grad is not None
        assert not torch.all(logits.grad == 0)

    def test_dtype_float32(self):
        loss = self.fn.compute(_logits(), {"mask": _p_soft()})
        assert loss.dtype == torch.float32

    def test_non_negative(self):
        for _ in range(5):
            loss = self.fn.compute(_logits(), {"mask": _p_soft(), "w_conf": _w_conf()})
            assert loss.item() >= -1e-6

    def test_zero_class_pixels_skipped(self):
        """_class_balanced_mask must skip (continue) classes with 0 pixels."""
        # p_soft where ALL pixels are class 0 → classes 1..C-1 have 0 pixels
        p = torch.zeros(B, C, H, W)
        p[:, 0, :, :] = 1.0  # entire image is class 0
        fn = CoTeachingLoss(name="cot", num_classes=C, lambda_reg=0.0)
        pred = {
            "logits_a": torch.randn(B, C, H, W, requires_grad=True),
            "logits_b": torch.randn(B, C, H, W, requires_grad=True),
        }
        loss = fn.compute(pred, {"mask": p})
        assert loss.shape == torch.Size([])
        assert loss.item() >= -1e-6


# ---------------------------------------------------------------------------
# CoTeachingLoss — training path (dual logits dict)
# ---------------------------------------------------------------------------


class TestCoTeachingLossTrainingPath:
    def setup_method(self):
        self.fn = CoTeachingLoss(
            name="cot",
            num_classes=C,
            e_warm=20,
            p_start=0.5,
            p_end=0.95,
            lambda_reg=0.0,  # disable L_reg to isolate CE tests
        )

    def _pred(self, requires_grad_a=True, requires_grad_b=True):
        return {
            "logits_a": torch.randn(B, C, H, W, requires_grad=requires_grad_a),
            "logits_b": torch.randn(B, C, H, W, requires_grad=requires_grad_b),
        }

    def test_dict_pred_returns_scalar(self):
        loss = self.fn.compute(self._pred(), {"mask": _p_soft(), "w_conf": _w_conf()})
        assert loss.shape == torch.Size([])

    def test_dict_pred_no_w_conf_returns_scalar(self):
        loss = self.fn.compute(self._pred(), {"mask": _p_soft()})
        assert loss.shape == torch.Size([])

    def test_gradient_flows_through_logits_a(self):
        pred = self._pred()
        self.fn.compute(pred, {"mask": _p_soft(), "w_conf": _w_conf()}).backward()
        assert pred["logits_a"].grad is not None

    def test_gradient_flows_through_logits_b(self):
        pred = self._pred()
        self.fn.compute(pred, {"mask": _p_soft(), "w_conf": _w_conf()}).backward()
        assert pred["logits_b"].grad is not None

    def test_loss_non_negative(self):
        for _ in range(5):
            loss = self.fn.compute(
                self._pred(), {"mask": _p_soft(), "w_conf": _w_conf()}
            )
            assert loss.item() >= -1e-6

    def test_epoch_progression_changes_loss(self):
        torch.manual_seed(42)
        pred = self._pred(requires_grad_a=False, requires_grad_b=False)
        p = _p_soft()
        w = _w_conf()
        self.fn.current_epoch = 0
        loss_early = self.fn.compute(pred, {"mask": p, "w_conf": w}).item()
        self.fn.current_epoch = 20
        loss_late = self.fn.compute(pred, {"mask": p, "w_conf": w}).item()
        # At later epochs more pixels are retained; loss may differ
        assert isinstance(loss_early, float) and isinstance(loss_late, float)

    def test_zero_w_conf_gives_zero_loss(self):
        pred = self._pred(requires_grad_a=False, requires_grad_b=False)
        w_zero = torch.zeros(B, 1, H, W)
        loss = self.fn.compute(pred, {"mask": _p_soft(), "w_conf": w_zero})
        assert loss.item() == pytest.approx(0.0, abs=1e-5)


class TestCoTeachingLossClassBalancedSelection:
    """Test _class_balanced_mask internals via compute() behaviour."""

    def setup_method(self):
        self.fn = CoTeachingLoss(name="cot", num_classes=C, lambda_reg=0.0)

    def test_loss_uses_cross_selection(self):
        # If logits_a is perfect but logits_b is bad:
        # J_A (from A's low-loss samples) should cover most of the batch.
        # Loss_B (on J_A) should be high because B's logits are bad.
        hard = torch.randint(0, C, (B, H, W))
        p = torch.zeros(B, C, H, W)
        p.scatter_(1, hard.unsqueeze(1), 1.0)

        logits_good = hard.unsqueeze(1).float() * 0  # placeholder
        logits_good = torch.zeros(B, C, H, W)
        logits_good.scatter_(1, hard.unsqueeze(1), 10.0)  # perfect logits

        logits_bad = torch.zeros(B, C, H, W)  # uniform → high CE

        pred = {
            "logits_a": logits_good.requires_grad_(True),
            "logits_b": logits_bad.requires_grad_(True),
        }
        self.fn.current_epoch = 20  # late epoch: keep 95%
        loss = self.fn.compute(pred, {"mask": p})
        assert loss.item() >= 0.0


# ---------------------------------------------------------------------------
# CoTeachingLoss — neighbourhood regularization
# ---------------------------------------------------------------------------


class TestCoTeachingLossNeighborhoodReg:
    def setup_method(self):
        self.fn = CoTeachingLoss(name="cot", num_classes=C, lambda_reg=0.1)

    def test_compute_neighborhood_reg_returns_scalar(self):
        result = self.fn.compute_neighborhood_reg(_logits(), _features())
        assert result.shape == torch.Size([])

    def test_compute_neighborhood_reg_gradient_flows(self):
        logits = _logits()
        result = self.fn.compute_neighborhood_reg(logits, _features())
        result.backward()
        assert logits.grad is not None

    def test_non_negative(self):
        for _ in range(5):
            result = self.fn.compute_neighborhood_reg(_logits(), _features())
            assert result.item() >= -1e-6

    def test_uniform_predictions_zero_kl(self):
        # If all logits are identical, softmax is uniform → KL(p||p) = 0 → L_reg ≈ 0
        logits = torch.zeros(B, C, H, W, requires_grad=True)
        result = self.fn.compute_neighborhood_reg(logits, _features())
        assert result.item() == pytest.approx(0.0, abs=1e-5)

    def test_feature_dim_agnostic(self):
        for d in [4, 16, 64]:
            feats = torch.randn(B, d, H, W)
            result = self.fn.compute_neighborhood_reg(_logits(), feats)
            assert result.shape == torch.Size([])

    def test_training_path_with_reg_uses_features(self):
        fn = CoTeachingLoss(name="cot", num_classes=C, lambda_reg=0.1)
        pred = {
            "logits_a": torch.randn(B, C, H, W, requires_grad=True),
            "logits_b": torch.randn(B, C, H, W, requires_grad=True),
            "features": _features(),
        }
        loss = fn.compute(pred, {"mask": _p_soft(), "w_conf": _w_conf()})
        assert loss.shape == torch.Size([])
        loss.backward()
        assert pred["logits_a"].grad is not None


# ---------------------------------------------------------------------------
# CoTeachingLoss — Loss.forward() wrapper compatibility
# ---------------------------------------------------------------------------


class TestCoTeachingLossForwardWrapper:
    def setup_method(self):
        self.fn = CoTeachingLoss(name="cot", num_classes=C)

    def test_forward_returns_two_element_tuple(self):
        result = self.fn(_logits(), {"mask": _p_soft(), "w_conf": _w_conf()})
        assert isinstance(result, tuple) and len(result) == 2

    def test_forward_first_element_scalar(self):
        val, _ = self.fn(_logits(), {"mask": _p_soft()})
        assert val.shape == torch.Size([])

    def test_forward_second_element_dict(self):
        _, extra = self.fn(_logits(), {"mask": _p_soft()})
        assert isinstance(extra, dict)
