# -*- coding: utf-8 -*-
"""Tests for MeanTeacherModel (AIO2 baseline) and find_nearest_checkpoint."""

from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn as nn
from omegaconf import OmegaConf

from pytorch_segmentation_models_trainer.custom_models.mean_teacher_wrapper import (
    MeanTeacherWrapper,
)
from pytorch_segmentation_models_trainer.model_loader.mean_teacher_model import (
    MeanTeacherModel,
    find_nearest_checkpoint,
)
from pytorch_segmentation_models_trainer.utils.act_trigger import ACTTracker

B, H, W, CLASSES = 2, 64, 64, 3


# ---------------------------------------------------------------------------
# Real-construction helpers (small enough to instantiate a real Model)
# ---------------------------------------------------------------------------


def _make_cfg(**extra):
    base = {
        "model": {
            "_target_": "segmentation_models_pytorch.Unet",
            "encoder_name": "resnet18",
            "encoder_weights": None,
            "in_channels": 3,
            "classes": CLASSES,
        },
        "loss": {
            "_target_": "pytorch_segmentation_models_trainer.custom_losses.loss.WeightedDiceCrossEntropyLoss",
            "num_classes": CLASSES,
        },
        "image_key": "image",
        "mask_key": "mask",
        "hyperparameters": {"batch_size": B, "devices": 1, "accelerator": "cpu"},
        "classes_to_correct": [0, 1],
        "alpha": 0.9,
    }
    base.update(extra)
    return OmegaConf.create(base)


def _make_batch():
    return {
        "image": torch.randn(B, 3, H, W),
        "mask": torch.randint(0, CLASSES, (B, H, W), dtype=torch.long),
    }


# ---------------------------------------------------------------------------
# Minimal-instance helper (object.__new__ bypass + MagicMock trainer), for
# on_train_epoch_end / checkpoint / stop mechanics — mirrors
# tests/test_co_teaching_model.py's `_minimal_model_instance`.
# ---------------------------------------------------------------------------


def _minimal_instance(**overrides):
    obj = object.__new__(MeanTeacherModel)
    torch.nn.Module.__init__(obj)

    backbone = nn.Sequential(nn.Conv2d(3, CLASSES, kernel_size=1))
    obj.model = MeanTeacherWrapper(backbone, alpha=0.9)

    obj.cfg = MagicMock()
    obj.cfg.get.side_effect = lambda key, default=None: default

    obj.correct_model = "teacher"
    obj.correct_base = "iter"
    obj._epoch_snapshot = None
    obj.classes_to_correct = [0, 1]
    obj.conflict_resolution = "priority_order"
    obj.o2c_filter_size = -1
    obj.o2c_active = False
    obj.resume_epoch = None
    obj.act_tracker = ACTTracker(window_sizes=[2, 3])
    obj.checkpoint_every_n_epochs = 5
    obj._act_checkpoint_dir = None
    obj._teacher_train_iou = None

    for key, value in overrides.items():
        setattr(obj, key, value)

    mock_trainer = MagicMock()
    mock_trainer.current_epoch = 0
    mock_trainer.should_stop = False
    object.__setattr__(obj, "_fabric", None)
    object.__setattr__(obj, "_trainer", mock_trainer)
    object.__setattr__(obj, "_jit_is_scripting", False)
    return obj


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


class TestConstruction:
    def test_model_is_wrapped_in_mean_teacher_wrapper(self):
        model = MeanTeacherModel(_make_cfg())
        assert isinstance(model.model, MeanTeacherWrapper)

    def test_o2c_starts_inactive(self):
        model = MeanTeacherModel(_make_cfg())
        assert model.o2c_active is False

    def test_resume_epoch_starts_none(self):
        model = MeanTeacherModel(_make_cfg())
        assert model.resume_epoch is None

    def test_classes_to_correct_from_cfg(self):
        model = MeanTeacherModel(_make_cfg(classes_to_correct=[1, 2]))
        assert model.classes_to_correct == [1, 2]

    def test_default_correct_model_is_teacher(self):
        model = MeanTeacherModel(_make_cfg())
        assert model.correct_model == "teacher"

    def test_invalid_correct_model_raises(self):
        with pytest.raises(ValueError):
            MeanTeacherModel(_make_cfg(correct_model="bogus"))

    def test_act_tracker_is_act_tracker_instance(self):
        model = MeanTeacherModel(_make_cfg())
        assert isinstance(model.act_tracker, ACTTracker)

    def test_default_correct_base_is_iter(self):
        model = MeanTeacherModel(_make_cfg())
        assert model.correct_base == "iter"

    def test_invalid_correct_base_raises(self):
        with pytest.raises(ValueError):
            MeanTeacherModel(_make_cfg(correct_base="bogus"))

    def test_epoch_snapshot_starts_none(self):
        model = MeanTeacherModel(_make_cfg())
        assert model._epoch_snapshot is None

    def test_inference_mode_still_wraps_model_but_skips_train_only_state(self):
        model = MeanTeacherModel(_make_cfg(), inference_mode=True)
        assert isinstance(model.model, MeanTeacherWrapper)
        assert not hasattr(model, "o2c_active")
        assert not hasattr(model, "act_tracker")


class TestForward:
    def test_forward_uses_student(self):
        model = MeanTeacherModel(_make_cfg())
        x = torch.randn(B, 3, H, W)
        model.eval()
        with torch.no_grad():
            out = model(x)
            expected = model.model.student(x)
        assert torch.equal(out, expected)


# ---------------------------------------------------------------------------
# training_step — warm-up path (o2c_active=False)
# ---------------------------------------------------------------------------


class TestTrainingStepWarmup:
    def test_returns_scalar_loss(self):
        model = MeanTeacherModel(_make_cfg())
        loss = model.training_step(_make_batch(), 0)
        assert loss.shape == torch.Size([])
        assert loss.requires_grad

    def test_teacher_train_iou_metric_created_after_one_batch(self):
        model = MeanTeacherModel(_make_cfg())
        assert model._teacher_train_iou is None
        model.training_step(_make_batch(), 0)
        assert model._teacher_train_iou is not None

    def test_soft_label_batch_skips_teacher_iou_tracking(self):
        model = MeanTeacherModel(_make_cfg())
        batch = _make_batch()
        batch["mask"] = torch.rand(B, CLASSES, H, W)  # float -> soft label
        # Should not crash even though the loss itself won't accept this
        # shape; only the ACT bookkeeping's early-return is under test here.
        with pytest.raises(Exception):
            model.training_step(batch, 0)
        assert model._teacher_train_iou is None

    def test_o2c_not_applied_during_warmup(self):
        model = MeanTeacherModel(_make_cfg())
        with patch(
            "pytorch_segmentation_models_trainer.model_loader.mean_teacher_model.online_object_correction"
        ) as mock_o2c:
            model.training_step(_make_batch(), 0)
        mock_o2c.assert_not_called()


# ---------------------------------------------------------------------------
# training_step — O2C path (o2c_active=True)
# ---------------------------------------------------------------------------


class TestTrainingStepO2C:
    def test_o2c_applied_when_active(self):
        model = MeanTeacherModel(_make_cfg())
        model.o2c_active = True
        with patch(
            "pytorch_segmentation_models_trainer.model_loader.mean_teacher_model.online_object_correction",
            wraps=None,
            side_effect=lambda noisy_hard, *a, **kw: noisy_hard,
        ) as mock_o2c:
            loss = model.training_step(_make_batch(), 0)
        assert mock_o2c.call_count == B  # once per image in the batch
        assert loss.shape == torch.Size([])

    def test_float_mask_raises_type_error_when_o2c_active(self):
        model = MeanTeacherModel(_make_cfg())
        model.o2c_active = True
        batch = _make_batch()
        batch["mask"] = torch.rand(B, CLASSES, H, W)
        with pytest.raises(TypeError):
            model.training_step(batch, 0)

    def test_correct_model_teacher_uses_teacher_forward(self):
        model = MeanTeacherModel(_make_cfg(correct_model="teacher"))
        model.o2c_active = True
        with (
            patch.object(
                model.model, "teacher_forward", wraps=model.model.teacher_forward
            ) as spy_teacher,
            patch(
                "pytorch_segmentation_models_trainer.model_loader.mean_teacher_model.online_object_correction",
                side_effect=lambda noisy_hard, *a, **kw: noisy_hard,
            ),
        ):
            model.training_step(_make_batch(), 0)
        spy_teacher.assert_called_once()

    def test_correct_model_student_does_not_use_teacher_forward(self):
        model = MeanTeacherModel(_make_cfg(correct_model="student"))
        model.o2c_active = True
        with (
            patch.object(
                model.model, "teacher_forward", wraps=model.model.teacher_forward
            ) as spy_teacher,
            patch(
                "pytorch_segmentation_models_trainer.model_loader.mean_teacher_model.online_object_correction",
                side_effect=lambda noisy_hard, *a, **kw: noisy_hard,
            ),
        ):
            model.training_step(_make_batch(), 0)
        spy_teacher.assert_not_called()


# ---------------------------------------------------------------------------
# correct_base: "iter" (live, default) vs "epoch" (frozen snapshot)
# ---------------------------------------------------------------------------


class TestCorrectBaseEpochSnapshot:
    def test_on_train_epoch_start_noop_when_correct_base_iter(self):
        obj = _minimal_instance(correct_base="iter", o2c_active=True)
        obj.on_train_epoch_start()
        assert obj._epoch_snapshot is None

    def test_on_train_epoch_start_noop_when_o2c_inactive(self):
        obj = _minimal_instance(correct_base="epoch", o2c_active=False)
        obj.on_train_epoch_start()
        assert obj._epoch_snapshot is None

    def test_on_train_epoch_start_creates_snapshot_when_epoch_and_active(self):
        obj = _minimal_instance(correct_base="epoch", o2c_active=True)
        obj.on_train_epoch_start()
        assert obj._epoch_snapshot is not None

    def test_snapshot_is_independent_copy_not_same_object(self):
        obj = _minimal_instance(correct_base="epoch", o2c_active=True)
        obj.on_train_epoch_start()
        assert obj._epoch_snapshot is not obj.model.teacher

    def test_snapshot_params_require_no_grad(self):
        obj = _minimal_instance(correct_base="epoch", o2c_active=True)
        obj.on_train_epoch_start()
        assert all(not p.requires_grad for p in obj._epoch_snapshot.parameters())

    def test_snapshot_taken_from_teacher_when_correct_model_teacher(self):
        obj = _minimal_instance(
            correct_base="epoch", o2c_active=True, correct_model="teacher"
        )
        with torch.no_grad():
            for p in obj.model.teacher.parameters():
                p.add_(1.0)  # diverge teacher from student
        obj.on_train_epoch_start()
        x = torch.randn(1, 3, 8, 8)
        with torch.no_grad():
            assert torch.equal(obj._epoch_snapshot(x), obj.model.teacher_forward(x))
            assert not torch.equal(obj._epoch_snapshot(x), obj.model(x))

    def test_snapshot_taken_from_student_when_correct_model_student(self):
        obj = _minimal_instance(
            correct_base="epoch", o2c_active=True, correct_model="student"
        )
        with torch.no_grad():
            for p in obj.model.teacher.parameters():
                p.add_(1.0)  # diverge teacher from student
        obj.on_train_epoch_start()
        x = torch.randn(1, 3, 8, 8)
        with torch.no_grad():
            assert torch.equal(obj._epoch_snapshot(x), obj.model(x))
            assert not torch.equal(obj._epoch_snapshot(x), obj.model.teacher_forward(x))

    def test_snapshot_stays_frozen_after_later_ema_updates(self):
        obj = _minimal_instance(correct_base="epoch", o2c_active=True)
        obj.on_train_epoch_start()
        x = torch.randn(1, 3, 8, 8)
        with torch.no_grad():
            before = obj._epoch_snapshot(x).clone()
        obj.model.ema_update(global_step=0)
        with torch.no_grad():
            for p in obj.model.student.parameters():
                p.add_(5.0)
        obj.model.ema_update(global_step=1)  # teacher moves after the snapshot
        with torch.no_grad():
            after = obj._epoch_snapshot(x)
        assert torch.equal(before, after)  # snapshot didn't move with it

    def test_correction_source_logits_raises_if_epoch_snapshot_missing(self):
        obj = _minimal_instance(correct_base="epoch", o2c_active=True)
        # on_train_epoch_start deliberately not called
        x = torch.randn(1, 3, 8, 8)
        with pytest.raises(RuntimeError):
            obj._correction_source_logits(x)

    def test_training_step_uses_snapshot_not_live_teacher_forward_in_epoch_mode(self):
        model = MeanTeacherModel(_make_cfg(correct_base="epoch"))
        model.o2c_active = True
        model.on_train_epoch_start()
        with (
            patch.object(
                model.model, "teacher_forward", wraps=model.model.teacher_forward
            ) as spy_teacher,
            patch(
                "pytorch_segmentation_models_trainer.model_loader.mean_teacher_model.online_object_correction",
                side_effect=lambda noisy_hard, *a, **kw: noisy_hard,
            ),
        ):
            model.training_step(_make_batch(), 0)
        spy_teacher.assert_not_called()

    def test_training_step_uses_live_teacher_forward_in_iter_mode(self):
        model = MeanTeacherModel(_make_cfg(correct_base="iter"))
        model.o2c_active = True
        with (
            patch.object(
                model.model, "teacher_forward", wraps=model.model.teacher_forward
            ) as spy_teacher,
            patch(
                "pytorch_segmentation_models_trainer.model_loader.mean_teacher_model.online_object_correction",
                side_effect=lambda noisy_hard, *a, **kw: noisy_hard,
            ),
        ):
            model.training_step(_make_batch(), 0)
        spy_teacher.assert_called_once()


# ---------------------------------------------------------------------------
# EMA hook wiring
# ---------------------------------------------------------------------------


class TestOnTrainBatchEnd:
    def test_calls_ema_update_with_global_step(self):
        obj = _minimal_instance()
        obj._trainer.global_step = 7
        with patch.object(obj.model, "ema_update") as mock_ema:
            obj.on_train_batch_end(outputs=None, batch=_make_batch(), batch_idx=0)
        mock_ema.assert_called_once_with(7)


# ---------------------------------------------------------------------------
# ACT: epoch-end tracking, checkpoint, trigger/stop
# ---------------------------------------------------------------------------


class TestOnTrainEpochEnd:
    def test_noop_when_no_teacher_iou_tracked_yet(self):
        obj = _minimal_instance()
        obj.on_train_epoch_end()  # must not raise
        assert obj.resume_epoch is None
        assert obj._trainer.should_stop is False

    def test_noop_when_o2c_already_active(self):
        obj = _minimal_instance(o2c_active=True)
        obj._teacher_train_iou = MagicMock()
        obj.on_train_epoch_end()
        obj._teacher_train_iou.compute.assert_not_called()

    def test_logs_teacher_iou(self):
        obj = _minimal_instance()
        obj._teacher_train_iou = MagicMock()
        obj._teacher_train_iou.compute.return_value = torch.tensor(0.5)
        with patch.object(obj, "log") as mock_log:
            obj.on_train_epoch_end()
        mock_log.assert_called_once()
        assert mock_log.call_args[0][0] == "act/teacher_train_iou"
        assert mock_log.call_args[0][1] == pytest.approx(0.5)

    def test_teacher_iou_metric_reset_each_epoch(self):
        obj = _minimal_instance()
        obj._teacher_train_iou = MagicMock()
        obj._teacher_train_iou.compute.return_value = torch.tensor(0.5)
        obj.on_train_epoch_end()
        obj._teacher_train_iou.reset.assert_called_once()

    def test_checkpoint_saved_on_interval(self):
        obj = _minimal_instance(checkpoint_every_n_epochs=5)
        obj._act_checkpoint_dir = MagicMock()
        obj._teacher_train_iou = MagicMock()
        obj._teacher_train_iou.compute.return_value = torch.tensor(0.5)
        obj._trainer.current_epoch = 4  # (4+1) % 5 == 0
        obj.on_train_epoch_end()
        obj._trainer.save_checkpoint.assert_called_once()

    def test_checkpoint_not_saved_off_interval(self):
        obj = _minimal_instance(checkpoint_every_n_epochs=5)
        obj._act_checkpoint_dir = MagicMock()
        obj._teacher_train_iou = MagicMock()
        obj._teacher_train_iou.compute.return_value = torch.tensor(0.5)
        obj._trainer.current_epoch = 2  # (2+1) % 5 != 0
        obj.on_train_epoch_end()
        obj._trainer.save_checkpoint.assert_not_called()

    def test_missing_checkpoint_dir_warns_and_does_not_crash(self):
        obj = _minimal_instance(checkpoint_every_n_epochs=1, _act_checkpoint_dir=None)
        obj._teacher_train_iou = MagicMock()
        obj._teacher_train_iou.compute.return_value = torch.tensor(0.5)
        obj._trainer.current_epoch = 0
        obj.on_train_epoch_end()  # must not raise
        obj._trainer.save_checkpoint.assert_not_called()

    def test_trigger_sets_resume_epoch_and_stops_trainer(self):
        obj = _minimal_instance()
        obj._teacher_train_iou = MagicMock()
        obj._teacher_train_iou.compute.return_value = torch.tensor(0.5)
        with patch.object(obj.act_tracker, "update", return_value=17):
            obj.on_train_epoch_end()
        assert obj.resume_epoch == 17
        assert obj._trainer.should_stop is True

    def test_no_trigger_leaves_trainer_running(self):
        obj = _minimal_instance()
        obj._teacher_train_iou = MagicMock()
        obj._teacher_train_iou.compute.return_value = torch.tensor(0.5)
        with patch.object(obj.act_tracker, "update", return_value=None):
            obj.on_train_epoch_end()
        assert obj.resume_epoch is None
        assert obj._trainer.should_stop is False


# ---------------------------------------------------------------------------
# find_nearest_checkpoint
# ---------------------------------------------------------------------------


class TestFindNearestCheckpoint:
    def test_picks_largest_epoch_leq_target(self, tmp_path):
        for e in [5, 10, 15, 20]:
            (tmp_path / f"epoch_{e}.ckpt").touch()
        result = find_nearest_checkpoint(tmp_path, target_epoch=17)
        assert result.name == "epoch_15.ckpt"

    def test_exact_match_preferred(self, tmp_path):
        for e in [5, 10, 15]:
            (tmp_path / f"epoch_{e}.ckpt").touch()
        result = find_nearest_checkpoint(tmp_path, target_epoch=10)
        assert result.name == "epoch_10.ckpt"

    def test_raises_when_none_qualify(self, tmp_path):
        (tmp_path / "epoch_20.ckpt").touch()
        with pytest.raises(FileNotFoundError):
            find_nearest_checkpoint(tmp_path, target_epoch=5)

    def test_raises_on_empty_dir(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            find_nearest_checkpoint(tmp_path, target_epoch=5)

    def test_ignores_malformed_filenames(self, tmp_path):
        (tmp_path / "epoch_10.ckpt").touch()
        (tmp_path / "epoch_bogus.ckpt").touch()
        (tmp_path / "not_a_checkpoint.ckpt").touch()
        result = find_nearest_checkpoint(tmp_path, target_epoch=100)
        assert result.name == "epoch_10.ckpt"

    def test_accepts_string_path(self, tmp_path):
        (tmp_path / "epoch_5.ckpt").touch()
        result = find_nearest_checkpoint(str(tmp_path), target_epoch=5)
        assert result.name == "epoch_5.ckpt"
