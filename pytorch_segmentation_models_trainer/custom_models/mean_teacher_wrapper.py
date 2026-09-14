# -*- coding: utf-8 -*-
"""Mean-teacher student/teacher pair — Liu et al., TGRS 2024 (AIO2), §2.1.

Plain ``nn.Module`` domain logic, owned by ``MeanTeacherModel`` (the
``LightningModule`` responsible for the training loop) — mirrors the
canonical split used by ``BaseDomainAdaptationMethod`` /
``DomainAdaptationModel`` in this project.
"""

import copy
import logging

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class MeanTeacherWrapper(nn.Module):
    """Student network + an EMA "teacher" copy of it.

    The student (``self.student``) is trained normally via backprop. The
    teacher (``self.teacher``) starts as an exact copy of the student and is
    updated only by exponential moving average (EMA) of the student's
    weights — never by gradient. It supplies (a) a smoother training-signal
    curve for ``ACTTracker`` and (b) the pseudo-labels consumed by
    ``online_object_correction`` (O2C).

    EMA decay warmup formula reused verbatim from
    ``custom_callbacks.training_callbacks.EMACallback`` (not reinvented):
    ``effective_decay = min(alpha, (step+1)/(step+10))``, so early updates
    track the student closely instead of being contaminated by the
    teacher's (== student's, at construction) random-init weights.

    Note on BatchNorm buffers: ``ema_update`` only iterates
    ``.parameters()`` (weights/biases), never ``.buffers()``
    (BatchNorm ``running_mean``/``running_var``) — this matches the official
    AIO2 reference implementation's ``update_ema_variables``
    (https://github.com/zhu-xlab/AIO2, ``utils/self_ensembling.py``), which
    has the same property. The official training script keeps its EMA model
    in ``.train()`` mode for the whole run so its BatchNorm buffers still
    evolve, driven by the teacher's *own* forward passes on its own input
    distribution — decoupled from the EMA weight update, not skipped
    outright. This wrapper doesn't force any particular train()/eval() mode
    on the teacher; it inherits whatever mode the owning ``LightningModule``
    is in (train during ``training_step``, eval during
    ``validation_step``/``test_step``), since both ``student`` and
    ``teacher`` are registered submodules.

    Args:
        student: The backbone network to wrap (already instantiated, e.g.
            via ``Model.get_model()``).
        alpha: Target EMA decay factor, in ``(0, 1)``. Default ``0.999``,
            matching both ``EMACallback`` and the official AIO2 example.
        **kwargs: Accepted for Hydra / ConfigStore compatibility.

    Example:
        wrapper = MeanTeacherWrapper(student=backbone, alpha=0.999)
        student_logits = wrapper(images)                  # == wrapper.student(images)
        with torch.no_grad():
            teacher_probs = wrapper.teacher_forward(images).softmax(dim=1)
        wrapper.ema_update(global_step=trainer.global_step)
    """

    def __init__(self, student: nn.Module, alpha: float = 0.999, **kwargs) -> None:
        super().__init__()
        if not 0.0 < alpha < 1.0:
            raise ValueError(f"alpha must be in (0, 1), got {alpha}")

        self.student = student
        self.teacher = copy.deepcopy(student)
        for param in self.teacher.parameters():
            param.requires_grad_(False)

        self.alpha = alpha
        self._last_global_step: int = -1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Student forward pass — used for training and (by default) metrics."""
        return self.student(x)

    def teacher_forward(self, x: torch.Tensor) -> torch.Tensor:
        """Teacher forward pass, no gradient. Used for ACT curves and O2C pseudo-labels."""
        with torch.no_grad():
            return self.teacher(x)

    @staticmethod
    def _effective_decay(global_step: int, alpha: float) -> float:
        """EMA warmup decay at ``global_step`` (0-indexed)."""
        return min(alpha, (global_step + 1) / (global_step + 10))

    def ema_update(self, global_step: int) -> bool:
        """Update the teacher's weights toward the student's via EMA.

        Safe to call once per ``on_train_batch_end`` even under
        ``accumulate_grad_batches > 1`` (where that hook fires once per
        micro-batch but the optimizer — and thus ``global_step`` — only
        advances once per accumulated step): a repeated ``global_step`` is a
        no-op, same guard as ``EMACallback``.

        Args:
            global_step: ``trainer.global_step`` at the time of the call.

        Returns:
            ``True`` if the teacher was actually updated, ``False`` if this
            call was skipped as a duplicate ``global_step``.
        """
        if global_step == self._last_global_step:
            return False
        self._last_global_step = global_step

        decay = self._effective_decay(global_step, self.alpha)
        with torch.no_grad():
            for t_param, s_param in zip(
                self.teacher.parameters(), self.student.parameters()
            ):
                t_param.data.mul_(decay).add_(s_param.data, alpha=1.0 - decay)
        return True
