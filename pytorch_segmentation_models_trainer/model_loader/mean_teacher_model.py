# -*- coding: utf-8 -*-
"""Mean-teacher LightningModule — AIO2 baseline (Liu et al., TGRS 2024).

Simplification vs. the plan first presented for this baseline: no separate
``ACTCheckpointCallback``. ACT needs the *teacher's* training accuracy, which
only this LightningModule can compute (the base ``Model``'s own ``train/*``
metrics reflect the student — see ``forward()`` — and a generic callback
would need to reach into model internals to get the teacher's instead,
which isn't a real separation of concerns, just callback ceremony). ACT
tracking, periodic checkpointing, and the stop signal all live in
``on_train_epoch_end`` below instead.
"""

import copy
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import torch
import torchmetrics

from pytorch_segmentation_models_trainer.custom_models.mean_teacher_wrapper import (
    MeanTeacherWrapper,
)
from pytorch_segmentation_models_trainer.model_loader.model import Model
from pytorch_segmentation_models_trainer.utils.act_trigger import ACTTracker
from pytorch_segmentation_models_trainer.utils.o2c_correction import (
    online_object_correction,
)

logger = logging.getLogger(__name__)

_VALID_CORRECT_MODEL = ("teacher", "student")
_VALID_CORRECT_BASE = ("iter", "epoch")


class MeanTeacherModel(Model):
    """Mean-teacher + ACT + O2C noisy-label baseline (AIO2).

    Two phases, run as **two separate ``Trainer.fit()`` calls in one Python
    process** — mirroring the official reference implementation's two CLI
    invocations (``--resume``/``--resume_from_detection``,
    https://github.com/zhu-xlab/AIO2) rather than an in-run weight rewind:

    **Phase 1 (warm-up).** ``o2c_active=False`` (the default). Trains on the
    noisy label as-is. Each epoch, tracks the *teacher's* training IoU
    (against the noisy label) via ``ACTTracker`` (§2.3). Saves a checkpoint
    every ``checkpoint_every_n_epochs`` epochs. The first time ``ACTTracker``
    detects the trigger point ``I_r``, this module sets
    ``self.resume_epoch = I_r`` and ``self.trainer.should_stop = True`` —
    the driver script's ``Trainer.fit()`` call returns at that point.

    **Phase 2 (correction).** The driver script calls
    ``find_nearest_checkpoint(checkpoint_dir, model.resume_epoch)``, reloads
    it (Lightning's own ``Trainer.fit(model, ckpt_path=...)``, restoring
    optimizer state natively), sets ``model.o2c_active = True``, and calls
    ``Trainer.fit()`` again. From here, every ``training_step`` corrects the
    noisy label per-image via ``online_object_correction`` (O2C, §2.4)
    before computing the loss, using ``self.correct_model``'s predictions.

    ``correct_base`` controls *when* those predictions are taken, independent
    of *which* network (``correct_model``) provides them:

    - ``"iter"`` (default): every ``training_step`` reads the source
      network's **current, live** state — true to the "online" in O2C's
      name (§2.4: no historical correction is saved, everything is redone
      from scratch every iteration).
    - ``"epoch"``: a frozen copy of the source network is taken once, in
      ``on_train_epoch_start``, and reused for every batch that epoch — the
      official code's other supported mode. Amortizes the forward pass +
      connected-component cost from "per batch" to "per epoch × unique
      images," at the cost of a slightly stale correction (the snapshot
      doesn't see that epoch's EMA updates until the next epoch's snapshot).
      Revisit this only if a pilot shows the per-batch cost of ``"iter"`` is
      the actual bottleneck — don't default to it speculatively.

    Example driver script::

        model = MeanTeacherModel(cfg)
        trainer1 = Trainer(max_epochs=cfg.max_warmup_epochs, ...)
        trainer1.fit(model)
        assert model.resume_epoch is not None  # ACT triggered

        ckpt = find_nearest_checkpoint(cfg.act_checkpoint_dir, model.resume_epoch)
        model.o2c_active = True
        trainer2 = Trainer(max_epochs=cfg.max_epochs, ...)
        trainer2.fit(model, ckpt_path=str(ckpt))

    Loss: reuses ``WeightedDiceCrossEntropyLoss`` (CE + Dice) — set it via
    ``cfg.loss`` like any other simple-loss model; no new loss class needed
    for this baseline (unlike ``BoundaryLabelRelaxationLoss`` for Zhu2019).

    ``forward()`` (used for the base class's own training/val/test metrics
    and by ``trainer.predict``/inference) is inherited unchanged from
    ``Model``, i.e. it delegates to ``self.model()`` = ``MeanTeacherWrapper.
    forward()`` = the **student**. Decision, not fixed by the paper: keeps
    evaluation on the same footing as every other condition in the
    experiment matrix (a single deployed model, not an EMA ensemble) —
    revisit if the pilot suggests otherwise.

    Config (all optional, defaults shown):

    .. code-block:: yaml

        _target_: pytorch_segmentation_models_trainer.model_loader.mean_teacher_model.MeanTeacherModel
        alpha: 0.999                        # EMA decay
        correct_model: teacher              # or: student
        correct_base: iter                  # or: epoch — see docstring above
        classes_to_correct: [1, 2]          # ordered; see online_object_correction
        conflict_resolution: priority_order # or: teacher_confidence
        o2c_filter_size: -1                 # odd int > 0 to enable boundary erosion
        act_window_sizes: [10, 20, 30, 40]
        checkpoint_every_n_epochs: 5
        act_checkpoint_dir: /path/to/warmup_checkpoints

        loss:
          _target_: pytorch_segmentation_models_trainer.custom_losses.loss.WeightedDiceCrossEntropyLoss
          num_classes: 6

    Args:
        cfg: Hydra DictConfig (same format as ``Model``).
        inference_mode: When ``True``, skips dataset/loss/ACT setup (matches
            ``Model``'s contract) — the mean-teacher wrapping still happens,
            so a saved checkpoint loads correctly for inference.
    """

    def __init__(self, cfg, inference_mode: bool = False) -> None:
        super().__init__(cfg, inference_mode)
        self.model = MeanTeacherWrapper(self.model, alpha=cfg.get("alpha", 0.999))
        if inference_mode:
            return

        self.correct_model: str = cfg.get("correct_model", "teacher")
        if self.correct_model not in _VALID_CORRECT_MODEL:
            raise ValueError(
                f"correct_model must be one of {_VALID_CORRECT_MODEL}, "
                f"got {self.correct_model!r}"
            )
        self.correct_base: str = cfg.get("correct_base", "iter")
        if self.correct_base not in _VALID_CORRECT_BASE:
            raise ValueError(
                f"correct_base must be one of {_VALID_CORRECT_BASE}, "
                f"got {self.correct_base!r}"
            )
        self._epoch_snapshot: Optional[torch.nn.Module] = None
        self.classes_to_correct: List[int] = list(cfg.get("classes_to_correct", []))
        self.conflict_resolution: str = cfg.get("conflict_resolution", "priority_order")
        self.o2c_filter_size: int = cfg.get("o2c_filter_size", -1)

        # Phase-1/phase-2 switch — see class docstring, "Two-phase orchestration".
        self.o2c_active: bool = False
        self.resume_epoch: Optional[int] = None  # I_r, set once ACT triggers

        self.act_tracker = ACTTracker(window_sizes=cfg.get("act_window_sizes", None))
        self.checkpoint_every_n_epochs: int = cfg.get("checkpoint_every_n_epochs", 5)
        checkpoint_dir = cfg.get("act_checkpoint_dir", None)
        self._act_checkpoint_dir: Optional[Path] = (
            Path(checkpoint_dir) if checkpoint_dir else None
        )

        self._teacher_train_iou: Optional[torchmetrics.JaccardIndex] = None

    # ------------------------------------------------------------------
    # EMA update — once per optimizer step, not per micro-batch.
    # ------------------------------------------------------------------

    def on_train_batch_end(self, outputs, batch, batch_idx) -> None:
        self.model.ema_update(self.trainer.global_step)

    # ------------------------------------------------------------------
    # correct_base="epoch" snapshot — refreshed once per epoch, only while
    # O2C is active (irrelevant, so skipped, during warm-up).
    # ------------------------------------------------------------------

    def on_train_epoch_start(self) -> None:
        if self.correct_base == "epoch" and self.o2c_active:
            self._refresh_epoch_snapshot()

    def _refresh_epoch_snapshot(self) -> None:
        source = (
            self.model.teacher
            if self.correct_model == "teacher"
            else self.model.student
        )
        snapshot = copy.deepcopy(source)
        snapshot.eval()
        for param in snapshot.parameters():
            param.requires_grad_(False)
        self._epoch_snapshot = snapshot
        logger.info(
            "correct_base='epoch': refreshed correction snapshot from %s at epoch %d",
            self.correct_model,
            self.current_epoch,
        )

    # ------------------------------------------------------------------
    # O2C-aware training step
    # ------------------------------------------------------------------

    def training_step(self, batch: Dict[str, Any], batch_idx: int):
        if not self.o2c_active:
            self._update_teacher_train_iou(batch)
            return super().training_step(batch, batch_idx)
        return super().training_step(self._apply_o2c(batch), batch_idx)

    def _correction_source_logits(self, images: torch.Tensor) -> torch.Tensor:
        """Return the logits O2C corrects against — see ``correct_base`` docstring."""
        if self.correct_base == "epoch":
            if self._epoch_snapshot is None:
                raise RuntimeError(
                    "correct_base='epoch' but no snapshot exists yet. "
                    "on_train_epoch_start should have created one since "
                    "o2c_active is True — was o2c_active flipped on mid-epoch "
                    "instead of before calling Trainer.fit()?"
                )
            with torch.no_grad():
                return self._epoch_snapshot(images)
        # correct_base == "iter": always the live, current-step state.
        if self.correct_model == "teacher":
            return self.model.teacher_forward(images)
        with torch.no_grad():
            return self.model(images)

    def _apply_o2c(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        image_key = self.cfg.get("image_key", "image")
        mask_key = self.cfg.get("mask_key", "mask")
        images = batch[image_key]
        noisy_hard = batch[mask_key]
        if noisy_hard.is_floating_point():
            raise TypeError(
                "MeanTeacherModel/O2C requires a hard-label (long) mask; got "
                f"floating-point dtype {noisy_hard.dtype}. O2C corrects "
                "TVS-style hard labels, not soft-label distributions."
            )

        with torch.no_grad():
            source_logits = self._correction_source_logits(images)
            probs = torch.softmax(source_logits, dim=1)

        corrected = torch.stack(
            [
                online_object_correction(
                    noisy_hard[i],
                    probs[i],
                    self.classes_to_correct,
                    filter_size=self.o2c_filter_size,
                    conflict_resolution=self.conflict_resolution,
                )
                for i in range(images.shape[0])
            ]
        )

        new_batch = dict(batch)
        new_batch[mask_key] = corrected
        return new_batch

    # ------------------------------------------------------------------
    # ACT: teacher train-IoU tracking, trigger detection, checkpoint + stop
    # (warm-up / phase-1 only — no-ops once o2c_active is True)
    # ------------------------------------------------------------------

    def _update_teacher_train_iou(self, batch: Dict[str, Any]) -> None:
        """Accumulate this batch's teacher-vs-noisy-label IoU.

        Deliberately the *teacher's* accuracy, not the student's: §2.3 relies
        on the teacher's smoother curve for ACT's local linear regression to
        be numerically stable. Skipped for soft-label batches (ACT/O2C need
        a hard-label target).
        """
        image_key = self.cfg.get("image_key", "image")
        mask_key = self.cfg.get("mask_key", "mask")
        images = batch[image_key]
        hard_mask = batch[mask_key]
        if hard_mask.is_floating_point():
            return
        with torch.no_grad():
            teacher_logits = self.model.teacher_forward(images)
            if self._teacher_train_iou is None:
                self._teacher_train_iou = torchmetrics.JaccardIndex(
                    task="multiclass",
                    num_classes=teacher_logits.shape[1],
                    ignore_index=255,
                ).to(teacher_logits.device)
            self._teacher_train_iou.update(teacher_logits, hard_mask.long())

    def on_train_epoch_end(self) -> None:
        if self.o2c_active or self._teacher_train_iou is None:
            return

        epoch_iou = self._teacher_train_iou.compute().item()
        self._teacher_train_iou.reset()
        self.log("act/teacher_train_iou", epoch_iou, on_epoch=True, sync_dist=True)

        if (self.current_epoch + 1) % self.checkpoint_every_n_epochs == 0:
            self._save_act_checkpoint()

        i_r = self.act_tracker.update(epoch_iou)
        if i_r is not None:
            self.resume_epoch = i_r
            logger.info(
                "ACT triggered at epoch %d: resume_epoch (I_r) = %d. Stopping "
                "warm-up — see MeanTeacherModel docstring, 'Two-phase orchestration'.",
                self.current_epoch,
                i_r,
            )
            self.trainer.should_stop = True

    def _save_act_checkpoint(self) -> None:
        if self._act_checkpoint_dir is None:
            logger.warning(
                "checkpoint_every_n_epochs reached but act_checkpoint_dir is not "
                "set — skipping periodic ACT checkpoint."
            )
            return
        self._act_checkpoint_dir.mkdir(parents=True, exist_ok=True)
        path = self._act_checkpoint_dir / f"epoch_{self.current_epoch}.ckpt"
        self.trainer.save_checkpoint(str(path))
        logger.info("Saved ACT warm-up checkpoint: %s", path)


def find_nearest_checkpoint(
    checkpoint_dir: Union[str, Path], target_epoch: int
) -> Path:
    """Return the saved ``epoch_{N}.ckpt`` with the largest ``N <= target_epoch``.

    For the phase-1 -> phase-2 handoff: ``target_epoch`` is usually
    ``model.resume_epoch`` (``I_r``) after phase 1's ``Trainer.fit()``
    returns, and checkpoints only exist at multiples of
    ``checkpoint_every_n_epochs`` — this picks the closest one at or before
    ``I_r``.

    Args:
        checkpoint_dir: Directory of ``epoch_{N}.ckpt`` files saved by
            ``MeanTeacherModel._save_act_checkpoint``.
        target_epoch: Upper bound (inclusive) on the checkpoint's epoch.

    Returns:
        Path to the matching checkpoint file.

    Raises:
        FileNotFoundError: No checkpoint with epoch ``<= target_epoch`` exists
            in ``checkpoint_dir``.
    """
    checkpoint_dir = Path(checkpoint_dir)
    candidates = []
    for p in checkpoint_dir.glob("epoch_*.ckpt"):
        try:
            epoch = int(p.stem.split("_")[1])
        except (IndexError, ValueError):
            continue
        if epoch <= target_epoch:
            candidates.append((epoch, p))
    if not candidates:
        raise FileNotFoundError(
            f"No ACT checkpoint with epoch <= {target_epoch} found in {checkpoint_dir}"
        )
    candidates.sort(key=lambda t: t[0])
    return candidates[-1][1]
