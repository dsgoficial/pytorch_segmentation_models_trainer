# -*- coding: utf-8 -*-
"""Tests for CoTeachingModel: dual-branch architecture and training mechanics."""

import pytest
import torch
import torch.nn as nn
from unittest.mock import MagicMock, PropertyMock, patch

from pytorch_segmentation_models_trainer.model_loader.co_teaching_model import (
    CoTeachingModel,
)
from pytorch_segmentation_models_trainer.model_loader.soft_label_model import (
    SoftLabelModel,
)
from pytorch_segmentation_models_trainer.custom_losses.co_teaching_loss import (
    CoTeachingLoss,
    CurriculumScheduler,
)

B, C, H, W = 2, 4, 8, 8


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _p_soft():
    raw = torch.rand(B, C, H, W)
    return (raw / raw.sum(dim=1, keepdim=True)).detach()


def _w_conf():
    return torch.rand(B, 1, H, W).detach()


def _minimal_model_instance():
    """Build a minimal CoTeachingModel without running Model.__init__.

    Uses object.__new__ to bypass dataset/loss instantiation, then wires
    up only what the methods under test require.
    """
    obj = object.__new__(CoTeachingModel)
    torch.nn.Module.__init__(obj)

    # Minimal backbone: 3-channel input → C-channel output
    backbone = nn.Sequential(
        nn.Conv2d(3, C, kernel_size=1),
    )
    obj.model = backbone
    obj.model_b = nn.Sequential(nn.Conv2d(3, C, kernel_size=1))

    obj.cfg = MagicMock()
    obj.cfg.get.side_effect = lambda key, default=None: default

    obj._w_conf = None
    obj.loss_function = CoTeachingLoss(name="cot", num_classes=C, lambda_reg=0.0)
    obj._curriculum = CurriculumScheduler()
    obj.use_compound_loss = False
    obj._is_dual_head = False
    obj._per_class_iou = None
    obj._class_names = None
    obj.automatic_optimization = False

    # Lightning's trainer / current_epoch properties access _fabric and _trainer;
    # bypass nn.Module.__setattr__ using object.__setattr__ to set them directly.
    mock_trainer = MagicMock()
    mock_trainer.current_epoch = 0
    object.__setattr__(obj, "_fabric", None)
    object.__setattr__(obj, "_trainer", mock_trainer)
    object.__setattr__(obj, "_jit_is_scripting", False)

    return obj


# ---------------------------------------------------------------------------
# Structure tests
# ---------------------------------------------------------------------------


class TestCoTeachingModelStructure:
    def test_has_model_a_as_model(self):
        obj = _minimal_model_instance()
        assert hasattr(obj, "model")

    def test_has_model_b(self):
        obj = _minimal_model_instance()
        assert hasattr(obj, "model_b")

    def test_model_and_model_b_are_independent_modules(self):
        obj = _minimal_model_instance()
        assert obj.model is not obj.model_b

    def test_automatic_optimization_disabled(self):
        obj = _minimal_model_instance()
        assert obj.automatic_optimization is False

    def test_loss_function_is_co_teaching_loss(self):
        obj = _minimal_model_instance()
        assert isinstance(obj.loss_function, CoTeachingLoss)

    def test_has_curriculum_scheduler(self):
        obj = _minimal_model_instance()
        assert hasattr(obj, "_curriculum")
        assert isinstance(obj._curriculum, CurriculumScheduler)


# ---------------------------------------------------------------------------
# forward() uses model_a only
# ---------------------------------------------------------------------------


class TestCoTeachingModelForward:
    def test_forward_returns_tensor(self):
        obj = _minimal_model_instance()
        images = torch.randn(B, 3, H, W)
        out = obj.forward(images)
        assert isinstance(out, torch.Tensor)

    def test_forward_output_shape(self):
        obj = _minimal_model_instance()
        images = torch.randn(B, 3, H, W)
        out = obj.forward(images)
        assert out.shape == (B, C, H, W)

    def test_forward_uses_model_a(self):
        obj = _minimal_model_instance()
        images = torch.randn(B, 3, H, W)
        with patch.object(obj.model, "forward", wraps=obj.model.forward) as mock_a:
            obj.forward(images)
        mock_a.assert_called_once()

    def test_forward_does_not_call_model_b_outside_training(self):
        obj = _minimal_model_instance()
        obj.training = False
        images = torch.randn(B, 3, H, W)
        with patch.object(obj.model_b, "forward", wraps=obj.model_b.forward) as mock_b:
            obj.forward(images)
        mock_b.assert_not_called()


# ---------------------------------------------------------------------------
# training_step mechanics
# ---------------------------------------------------------------------------


class TestCoTeachingModelTrainingStep:
    def _batch(self):
        return {
            "image": torch.randn(B, 3, H, W),
            "mask": {"mask": _p_soft(), "w_conf": _w_conf()},
        }

    def test_training_step_returns_scalar(self):
        obj = _minimal_model_instance()
        # Mock optimizers for manual optimization
        opt_a = MagicMock()
        opt_b = MagicMock()
        with patch.object(obj, "optimizers", return_value=[opt_a, opt_b]):
            with patch.object(obj, "log"):
                with patch.object(obj, "manual_backward"):
                    result = obj.training_step(self._batch(), 0)
        assert isinstance(result, torch.Tensor)
        assert result.shape == torch.Size([])

    def test_training_step_calls_both_models(self):
        obj = _minimal_model_instance()
        opt_a, opt_b = MagicMock(), MagicMock()
        with patch.object(obj, "optimizers", return_value=[opt_a, opt_b]):
            with patch.object(obj, "log"):
                with patch.object(obj, "manual_backward"):
                    with patch.object(
                        obj.model, "forward", wraps=obj.model.forward
                    ) as ma:
                        with patch.object(
                            obj.model_b, "forward", wraps=obj.model_b.forward
                        ) as mb:
                            obj.training_step(self._batch(), 0)
        ma.assert_called_once()
        mb.assert_called_once()

    def test_training_step_steps_both_optimizers(self):
        obj = _minimal_model_instance()
        opt_a, opt_b = MagicMock(), MagicMock()
        with patch.object(obj, "optimizers", return_value=[opt_a, opt_b]):
            with patch.object(obj, "log"):
                with patch.object(obj, "manual_backward"):
                    obj.training_step(self._batch(), 0)
        opt_a.step.assert_called()
        opt_b.step.assert_called()

    def test_training_step_calls_manual_backward_twice(self):
        obj = _minimal_model_instance()
        opt_a, opt_b = MagicMock(), MagicMock()
        backward_calls = []
        with patch.object(obj, "optimizers", return_value=[opt_a, opt_b]):
            with patch.object(obj, "log"):
                with patch.object(
                    obj,
                    "manual_backward",
                    side_effect=lambda t: backward_calls.append(t),
                ):
                    obj.training_step(self._batch(), 0)
        assert len(backward_calls) == 2

    def test_training_step_without_w_conf(self):
        """Batch without w_conf should still run without errors."""
        obj = _minimal_model_instance()
        batch = {"image": torch.randn(B, 3, H, W), "mask": {"mask": _p_soft()}}
        opt_a, opt_b = MagicMock(), MagicMock()
        with patch.object(obj, "optimizers", return_value=[opt_a, opt_b]):
            with patch.object(obj, "log"):
                with patch.object(obj, "manual_backward"):
                    result = obj.training_step(batch, 0)
        assert result.shape == torch.Size([])

    def test_training_step_plain_tensor_mask(self):
        """Batch with plain tensor mask (no w_conf wrapper) runs without errors."""
        obj = _minimal_model_instance()
        batch = {"image": torch.randn(B, 3, H, W), "mask": _p_soft()}
        opt_a, opt_b = MagicMock(), MagicMock()
        with patch.object(obj, "optimizers", return_value=[opt_a, opt_b]):
            with patch.object(obj, "log"):
                with patch.object(obj, "manual_backward"):
                    result = obj.training_step(batch, 0)
        assert result.shape == torch.Size([])

    def test_current_epoch_passed_to_loss(self):
        obj = _minimal_model_instance()
        opt_a, opt_b = MagicMock(), MagicMock()

        with patch.object(obj, "optimizers", return_value=[opt_a, opt_b]):
            with patch.object(obj, "log"):
                with patch.object(obj, "manual_backward"):
                    with patch.object(
                        type(obj),
                        "current_epoch",
                        new_callable=PropertyMock,
                        return_value=10,
                    ):
                        obj.training_step(self._batch(), 0)

        assert obj.loss_function.current_epoch == 10


# ---------------------------------------------------------------------------
# configure_optimizers returns two optimizers
# ---------------------------------------------------------------------------


class TestCoTeachingModelConfigureOptimizers:
    def test_configure_optimizers_returns_two(self):
        obj = _minimal_model_instance()
        obj.cfg.optimizer = MagicMock()
        obj.cfg.optimizer._target_ = "torch.optim.Adam"
        obj.cfg.get.side_effect = lambda key, default=None: {
            "scheduler_list": None,
        }.get(key, default)

        with patch(
            "pytorch_segmentation_models_trainer.model_loader.co_teaching_model.instantiate"
        ) as mock_inst:
            mock_opt_a = MagicMock()
            mock_opt_b = MagicMock()
            mock_inst.side_effect = [mock_opt_a, mock_opt_b]
            result = obj.configure_optimizers()

        optimizers, schedulers = result
        assert len(optimizers) == 2
        assert schedulers == []


# ---------------------------------------------------------------------------
# __init__ via inference_mode=True (covers lines 57-78)
# ---------------------------------------------------------------------------


class TestCoTeachingModelInit:
    def test_init_inference_mode_creates_model_b(self):
        backbone = nn.Sequential(nn.Conv2d(3, C, kernel_size=1))
        cfg = MagicMock()
        cfg.get.side_effect = lambda key, default=None: {
            "curriculum": {},
            "seed": None,
        }.get(key, default)

        with patch(
            "pytorch_segmentation_models_trainer.model_loader.model.instantiate",
            return_value=backbone,
        ):
            model = CoTeachingModel(cfg, inference_mode=True)

        assert hasattr(model, "model_b")
        assert model.automatic_optimization is False
        assert isinstance(model._curriculum, CurriculumScheduler)

    def test_init_curriculum_cfg_dict_branch(self):
        """cfg.curriculum as a plain dict uses the dict branch (lines 65-68)."""
        backbone = nn.Sequential(nn.Conv2d(3, C, kernel_size=1))
        cfg = MagicMock()
        cfg.get.side_effect = lambda key, default=None: {
            "curriculum": {"e_warm": 10, "p_start": 0.3, "p_end": 0.9},
            "seed": None,
        }.get(key, default)

        with patch(
            "pytorch_segmentation_models_trainer.model_loader.model.instantiate",
            return_value=backbone,
        ):
            model = CoTeachingModel(cfg, inference_mode=True)

        assert model._curriculum.e_warm == 10
        assert model._curriculum.p_start == pytest.approx(0.3)

    def test_init_curriculum_cfg_object_branch(self):
        """cfg.curriculum as an object uses getattr branch (lines 70-72)."""
        backbone = nn.Sequential(nn.Conv2d(3, C, kernel_size=1))
        curr_cfg = MagicMock()
        curr_cfg.e_warm = 30
        curr_cfg.p_start = 0.4
        curr_cfg.p_end = 0.85

        cfg = MagicMock()
        cfg.get.side_effect = lambda key, default=None: {
            "curriculum": curr_cfg,
            "seed": None,
        }.get(key, default)

        with patch(
            "pytorch_segmentation_models_trainer.model_loader.model.instantiate",
            return_value=backbone,
        ):
            model = CoTeachingModel(cfg, inference_mode=True)

        assert model._curriculum.e_warm == 30

    def test_init_syncs_curriculum_to_loss_function(self):
        """__init__ syncs _curriculum to loss_function when it is CoTeachingLoss (line 78)."""
        loss = CoTeachingLoss(name="cot", num_classes=C)
        backbone = nn.Sequential(nn.Conv2d(3, C, kernel_size=1))
        cfg = MagicMock()
        cfg.get.side_effect = lambda key, default=None: {
            "curriculum": {},
            "seed": None,
        }.get(key, default)

        def fake_parent_init(self_obj, cfg, inference_mode=False):
            torch.nn.Module.__init__(self_obj)
            self_obj.cfg = cfg
            self_obj.model = backbone
            self_obj.loss_function = loss
            self_obj._is_dual_head = False
            self_obj.use_compound_loss = False

        with patch.object(SoftLabelModel, "__init__", fake_parent_init):
            with patch.object(
                CoTeachingModel,
                "get_model",
                return_value=nn.Sequential(nn.Conv2d(3, C, kernel_size=1)),
            ):
                model = CoTeachingModel(cfg)

        assert model.loss_function._curriculum is model._curriculum


# ---------------------------------------------------------------------------
# training_step with lambda_reg > 0 (covers line 181)
# ---------------------------------------------------------------------------


class TestCoTeachingModelNeighborhoodReg:
    def _batch(self):
        return {
            "image": torch.randn(B, 3, H, W),
            "mask": {"mask": _p_soft(), "w_conf": _w_conf()},
        }

    def test_training_step_with_lambda_reg(self):
        obj = _minimal_model_instance()
        # Replace loss with one that has lambda_reg > 0
        obj.loss_function = CoTeachingLoss(name="cot", num_classes=C, lambda_reg=0.1)
        obj._curriculum = obj.loss_function._curriculum

        opt_a, opt_b = MagicMock(), MagicMock()
        with patch.object(obj, "optimizers", return_value=[opt_a, opt_b]):
            with patch.object(obj, "log"):
                with patch.object(obj, "manual_backward"):
                    result = obj.training_step(self._batch(), 0)

        assert result.shape == torch.Size([])
        opt_a.step.assert_called()
        opt_b.step.assert_called()
