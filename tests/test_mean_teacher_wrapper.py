# -*- coding: utf-8 -*-
"""Tests for MeanTeacherWrapper (AIO2 mean-teacher student/teacher pair)."""

import torch
import torch.nn as nn
import pytest

from pytorch_segmentation_models_trainer.custom_models.mean_teacher_wrapper import (
    MeanTeacherWrapper,
)


def _tiny_backbone():
    return nn.Sequential(nn.Conv2d(3, 4, kernel_size=1))


def _backbone_with_bn():
    return nn.Sequential(nn.Conv2d(3, 4, kernel_size=1), nn.BatchNorm2d(4))


B, IN_C, H, W = 2, 3, 5, 5


class TestInit:
    def test_alpha_out_of_range_raises(self):
        with pytest.raises(ValueError):
            MeanTeacherWrapper(_tiny_backbone(), alpha=0.0)
        with pytest.raises(ValueError):
            MeanTeacherWrapper(_tiny_backbone(), alpha=1.0)
        with pytest.raises(ValueError):
            MeanTeacherWrapper(_tiny_backbone(), alpha=-0.1)

    def test_accepts_kwargs_for_hydra_compat(self):
        MeanTeacherWrapper(_tiny_backbone(), alpha=0.9, name="mt", extra_field=1)

    def test_teacher_starts_as_exact_copy_of_student(self):
        student = _tiny_backbone()
        wrapper = MeanTeacherWrapper(student, alpha=0.9)
        for t_param, s_param in zip(
            wrapper.teacher.parameters(), wrapper.student.parameters()
        ):
            assert torch.equal(t_param, s_param)

    def test_teacher_is_independent_module_not_same_object(self):
        wrapper = MeanTeacherWrapper(_tiny_backbone(), alpha=0.9)
        assert wrapper.teacher is not wrapper.student
        for t_param, s_param in zip(
            wrapper.teacher.parameters(), wrapper.student.parameters()
        ):
            assert t_param.data_ptr() != s_param.data_ptr()

    def test_teacher_params_require_no_grad(self):
        wrapper = MeanTeacherWrapper(_tiny_backbone(), alpha=0.9)
        assert all(not p.requires_grad for p in wrapper.teacher.parameters())

    def test_student_params_keep_requiring_grad(self):
        wrapper = MeanTeacherWrapper(_tiny_backbone(), alpha=0.9)
        assert all(p.requires_grad for p in wrapper.student.parameters())


class TestForward:
    def test_forward_uses_student(self):
        wrapper = MeanTeacherWrapper(_tiny_backbone(), alpha=0.9)
        x = torch.randn(B, IN_C, H, W)
        out = wrapper(x)
        assert torch.equal(out, wrapper.student(x))

    def test_forward_output_requires_grad(self):
        wrapper = MeanTeacherWrapper(_tiny_backbone(), alpha=0.9)
        x = torch.randn(B, IN_C, H, W)
        out = wrapper(x)
        assert out.requires_grad

    def test_teacher_forward_output_does_not_require_grad(self):
        wrapper = MeanTeacherWrapper(_tiny_backbone(), alpha=0.9)
        x = torch.randn(B, IN_C, H, W)
        out = wrapper.teacher_forward(x)
        assert not out.requires_grad

    def test_forward_and_teacher_forward_diverge_after_student_changes(self):
        wrapper = MeanTeacherWrapper(_tiny_backbone(), alpha=0.9)
        x = torch.randn(B, IN_C, H, W)
        with torch.no_grad():
            for p in wrapper.student.parameters():
                p.add_(1.0)
        assert not torch.equal(wrapper(x), wrapper.teacher_forward(x))


class TestEffectiveDecay:
    def test_decay_ramps_up_from_low_value(self):
        d0 = MeanTeacherWrapper._effective_decay(0, alpha=0.999)
        assert d0 == pytest.approx(1 / 10)

    def test_decay_approaches_alpha_at_large_steps(self):
        d_large = MeanTeacherWrapper._effective_decay(100_000, alpha=0.999)
        assert d_large == pytest.approx(0.999, abs=1e-4)

    def test_decay_never_exceeds_alpha(self):
        for step in [0, 1, 10, 1000, 1_000_000]:
            assert MeanTeacherWrapper._effective_decay(step, alpha=0.9) <= 0.9


class TestEmaUpdate:
    def test_returns_true_on_first_call(self):
        wrapper = MeanTeacherWrapper(_tiny_backbone(), alpha=0.9)
        assert wrapper.ema_update(global_step=0) is True

    def test_returns_false_on_duplicate_global_step(self):
        wrapper = MeanTeacherWrapper(_tiny_backbone(), alpha=0.9)
        wrapper.ema_update(global_step=5)
        assert wrapper.ema_update(global_step=5) is False

    def test_teacher_unchanged_on_duplicate_global_step(self):
        wrapper = MeanTeacherWrapper(_tiny_backbone(), alpha=0.9)
        with torch.no_grad():
            for p in wrapper.student.parameters():
                p.add_(1.0)
        wrapper.ema_update(global_step=5)
        teacher_after_first = [p.clone() for p in wrapper.teacher.parameters()]
        wrapper.ema_update(global_step=5)  # duplicate, should be a no-op
        for before, after in zip(teacher_after_first, wrapper.teacher.parameters()):
            assert torch.equal(before, after)

    def test_teacher_moves_toward_student_but_not_all_the_way(self):
        wrapper = MeanTeacherWrapper(_tiny_backbone(), alpha=0.9)
        teacher_before = [p.clone() for p in wrapper.teacher.parameters()]
        with torch.no_grad():
            for p in wrapper.student.parameters():
                p.add_(1.0)
        wrapper.ema_update(global_step=0)  # decay = 1/10 at step 0
        for before, t_after, s_after in zip(
            teacher_before, wrapper.teacher.parameters(), wrapper.student.parameters()
        ):
            assert not torch.equal(t_after, before)  # moved
            assert not torch.equal(t_after, s_after)  # didn't jump all the way

    def test_student_untouched_by_ema_update(self):
        wrapper = MeanTeacherWrapper(_tiny_backbone(), alpha=0.9)
        student_before = [p.clone() for p in wrapper.student.parameters()]
        wrapper.ema_update(global_step=0)
        for before, after in zip(student_before, wrapper.student.parameters()):
            assert torch.equal(before, after)

    def test_bn_running_stats_not_touched_by_ema_update(self):
        # Matches the official reference implementation's property: EMA only
        # updates .parameters(), never .buffers() (BatchNorm running stats).
        wrapper = MeanTeacherWrapper(_backbone_with_bn(), alpha=0.9)
        bn_teacher = wrapper.teacher[1]
        running_mean_before = bn_teacher.running_mean.clone()
        running_var_before = bn_teacher.running_var.clone()

        # Drive the student's BN running stats away from the teacher's by
        # running a forward pass in train mode.
        wrapper.student.train()
        wrapper.student(torch.randn(B, IN_C, H, W) * 5 + 3)
        wrapper.ema_update(global_step=0)

        assert torch.equal(bn_teacher.running_mean, running_mean_before)
        assert torch.equal(bn_teacher.running_var, running_var_before)
