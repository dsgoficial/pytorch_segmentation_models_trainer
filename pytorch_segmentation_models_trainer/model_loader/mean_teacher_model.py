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
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

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
from pytorch_segmentation_models_trainer.utils.resume_utils import (
    load_resume_checkpoint,
    read_aio2_state,
    write_aio2_state,
)

logger = logging.getLogger(__name__)

_VALID_CORRECT_MODEL = ("teacher", "student")
_VALID_CORRECT_BASE = ("iter", "epoch")
_VALID_PHASE2_TRIGGER = ("act", "val_best")


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

        # When phase 2 starts — "act" (paper, §2.3) or "val_best" (adaptation:
        # warm-up's best-validation checkpoint). See fit_two_phase.
        self.phase2_trigger: str = cfg.get("phase2_trigger", "act")
        if self.phase2_trigger not in _VALID_PHASE2_TRIGGER:
            raise ValueError(
                f"phase2_trigger must be one of {_VALID_PHASE2_TRIGGER}, "
                f"got {self.phase2_trigger!r}"
            )

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
        if self.phase2_trigger != "act":
            return  # val_best: EarlyStopping ends the warm-up, ACT is not consulted

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

    @property
    def act_checkpoint_dir(self) -> Optional[Path]:
        """Where phase-1 warm-up checkpoints go.

        ``cfg.act_checkpoint_dir`` when set; otherwise
        ``<trainer.default_root_dir>/act_warmup_checkpoints``. The fallback is
        what makes multi-seed runs safe: ``ExperimentsRunner`` gives every seed
        its own ``pl_trainer.default_root_dir``, so per-seed checkpoints never
        collide (a single fixed ``act_checkpoint_dir`` shared by all seeds
        would let seed B resume from seed A's warm-up).
        """
        if self._act_checkpoint_dir is not None:
            return self._act_checkpoint_dir
        trainer = getattr(self, "_trainer", None)
        root = getattr(trainer, "default_root_dir", None) if trainer else None
        return Path(root) / "act_warmup_checkpoints" if root else None

    def _save_act_checkpoint(self) -> None:
        ckpt_dir = self.act_checkpoint_dir
        if ckpt_dir is None:
            logger.warning(
                "checkpoint_every_n_epochs reached but neither act_checkpoint_dir "
                "nor trainer.default_root_dir is set — skipping periodic ACT checkpoint."
            )
            return
        # Pin the resolved dir so phase 2 (a different Trainer) reads the same one.
        self._act_checkpoint_dir = ckpt_dir
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


def prepare_phase2_checkpoint(ckpt_path: Union[str, Path]) -> Path:
    """Copy a warm-up checkpoint without its callback state, for phase 2.

    ``Trainer.fit(model, ckpt_path=...)`` restores *everything* in the
    checkpoint, including callback state. For the phase-1 -> phase-2 handoff
    that is wrong for two callbacks:

    - ``ModelCheckpoint`` would carry phase 1's ``best_model_score``/
      ``best_model_path``. Phase 2's checkpoints would then have to beat a
      warm-up score, and the "best" model handed to ``trainer.test()`` could
      be a warm-up (pre-correction) model — i.e. not AIO2's output at all.
    - ``EarlyStopping`` would carry phase 1's ``wait_count``.

    Model weights (student + EMA teacher), optimizer, LR scheduler and loop
    progress (epoch / global step) are kept, so phase 2 resumes at ``I_r``
    exactly as the official two-invocation script does. The copy is written
    next to the original as ``<stem>_phase2_init.ckpt``.

    Args:
        ckpt_path: Warm-up checkpoint chosen by ``find_nearest_checkpoint``.

    Returns:
        Path to the stripped copy.
    """
    ckpt_path = Path(ckpt_path)
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state.pop("callbacks", None)
    out = ckpt_path.with_name(f"{ckpt_path.stem}_phase2_init.ckpt")
    torch.save(state, out)
    return out


def fit_two_phase(
    model: "MeanTeacherModel",
    make_trainer: Callable[..., Any],
    resume_root: Optional[Union[str, Path]] = None,
) -> Tuple[Any, Dict[str, float]]:
    """Run AIO2's warm-up (phase 1) and online-correction (phase 2) fits.

    This is the "two-phase driver" referred to in the class docstring.

    Phase 1 runs until ``ACTTracker`` triggers (``model.resume_epoch`` set,
    ``trainer.should_stop``) or ``max_epochs`` is reached. If ACT triggered,
    phase 2 restarts from the warm-up checkpoint nearest to (and not after)
    ``I_r`` — callback state stripped, see ``prepare_phase2_checkpoint`` —
    with ``model.o2c_active = True``, and trains until its own stopping
    criteria (``max_epochs`` / early stopping).

    If ACT never triggers, phase 2 is skipped and the phase-1 trainer is
    returned: the run degrades to a plain mean-teacher model and
    ``aio2/act_triggered`` / ``aio2/phase2_started`` are 0 in the returned
    info (and in the trainer's ``callback_metrics``), so it is visible in
    every results table instead of silently passing as an AIO2 result.

    ``model.phase2_trigger = "val_best"`` (``cfg.phase2_trigger``) replaces
    ACT — an adaptation, not the paper: the warm-up keeps ``EarlyStopping``
    and phase 2 restarts from the warm-up ``ModelCheckpoint``'s best model.
    Meant for label noise where ACT's curve shape never appears (gradual
    memorization without a plateau-then-acceleration transition). It uses
    the validation labels, which AIO2's premise assumes unavailable.

    Resuming (``resume_root`` set — ``train()`` does it for ``resumable: true``
    with ``phase2_trigger="val_best"`` only): a ``correction_last.ckpt`` plus
    the recorded hand-off resumes phase 2 without touching the warm-up; the
    hand-off alone (warm-up done, phase 2 saved no epoch yet) restarts phase 2
    from its init checkpoint; otherwise a ``warmup_last.ckpt`` resumes the
    warm-up. ``make_trainer`` must then save ``<phase>_last.ckpt`` under
    ``resume_root`` (``utils.resume_utils.resume_checkpoint_callback``).

    Args:
        model: A freshly built ``MeanTeacherModel`` (``o2c_active=False``).
        make_trainer: ``make_trainer(phase, point)`` with ``phase`` in
            ``{"warmup", "correction"}`` returns a new ``Trainer``;
            ``point`` is the ``ResumePoint`` it will resume from (or
            ``None``), to pin its logger to the interrupted version dir. The
            caller decides the callbacks per phase — in particular
            ``train()`` leaves ``EarlyStopping`` out of the warm-up trainer
            when ``phase2_trigger="act"``, because then when to stop warming
            up is ACT's job (``"val_best"`` keeps it).
        resume_root: The run's ``default_root_dir`` to resume from, or
            ``None`` (never resume, write no resume state).

    Returns:
        ``(trainer, info)``: the trainer of the last phase that ran (its
        checkpoint callback is what ``train()`` uses for ``trainer.test()``)
        and ``aio2/*`` diagnostics.
    """
    handoff = read_aio2_state(resume_root) if resume_root is not None else None
    if handoff is not None:
        init_ckpt, info = Path(handoff["init_ckpt"]), handoff["info"]
        point = load_resume_checkpoint(resume_root, "correction")
        logger.info(
            "AIO2: warm-up already done — %s phase 2.",
            "resuming" if point is not None else "restarting",
        )
    else:
        warmup = _fit_warmup(model, make_trainer, resume_root)
        val_best = model.phase2_trigger == "val_best"
        info: Dict[str, float] = {
            "aio2/phase2_started": 0.0,
            "aio2/phase2_trigger_val_best": float(val_best),
            "aio2/warmup_epochs": float(warmup.current_epoch),  # epochs completed after fit()
        }
        ckpt = (
            _val_best_checkpoint(warmup, info)
            if val_best
            else _act_checkpoint(model, warmup, info)
        )
        if ckpt is None:
            stamp_aio2_info(warmup, info)
            return warmup, info

        init_ckpt = prepare_phase2_checkpoint(ckpt)
        info["aio2/phase2_started"] = 1.0
        logger.info(
            "AIO2: phase 2 (%s trigger) resumes from %s (after %d epochs) with O2C on.",
            model.phase2_trigger,
            ckpt,
            int(info["aio2/resume_checkpoint_epoch"]),
        )
        if resume_root is not None:
            write_aio2_state(resume_root, {"init_ckpt": str(init_ckpt), "info": info})
        point = None

    model.o2c_active = True
    correction = make_trainer("correction", point)
    # weights_only=False: the checkpoint is this run's own output and carries
    # the Hydra config (OmegaConf objects) in hyper_parameters, which torch's
    # weights-only unpickler rejects.
    start = init_ckpt if point is None else point.path
    correction.fit(model, ckpt_path=str(start), weights_only=False)
    stamp_aio2_info(correction, info)
    return correction, info


def _fit_warmup(
    model: "MeanTeacherModel",
    make_trainer: Callable[..., Any],
    resume_root: Optional[Union[str, Path]],
) -> Any:
    """Phase 1, resumed from ``warmup_last.ckpt`` when one exists."""
    point = (
        load_resume_checkpoint(resume_root, "warmup") if resume_root is not None else None
    )
    warmup = make_trainer("warmup", point)
    if point is None:
        warmup.fit(model)
    else:
        logger.info("AIO2: resuming the warm-up from %s", point.path)
        warmup.fit(model, ckpt_path=str(point.path), weights_only=False)
    return warmup


def _act_checkpoint(
    model: "MeanTeacherModel", warmup: Any, info: Dict[str, float]
) -> Optional[Path]:
    """``phase2_trigger="act"``: warm-up checkpoint nearest to (not after) I_r."""
    info["aio2/act_triggered"] = 0.0
    if model.resume_epoch is None:
        logger.warning(
            "AIO2: ACT did not trigger within the warm-up (%d epochs) — phase 2 "
            "(O2C) skipped; this run is a plain mean-teacher model. Raise "
            "max_epochs, revisit act_window_sizes, or use phase2_trigger=val_best.",
            warmup.current_epoch,
        )
        return None
    # Index conventions differ: ACTTracker's I_r is a 1-indexed count of
    # completed epochs, while warm-up files are named by Lightning's 0-indexed
    # current_epoch (epoch_9.ckpt = state after 10 epochs). The state after
    # I_r epochs is therefore epoch_{I_r - 1}.ckpt; searching with I_r itself
    # could pick a checkpoint one epoch *past* I_r.
    ckpt = find_nearest_checkpoint(model.act_checkpoint_dir, model.resume_epoch - 1)
    info["aio2/act_triggered"] = 1.0
    info["aio2/resume_epoch"] = float(model.resume_epoch)
    # same convention as resume_epoch: epochs completed at that checkpoint
    info["aio2/resume_checkpoint_epoch"] = float(int(ckpt.stem.split("_")[1]) + 1)
    return ckpt


def _val_best_checkpoint(warmup: Any, info: Dict[str, float]) -> Optional[Path]:
    """``phase2_trigger="val_best"``: the warm-up ``ModelCheckpoint``'s best model.

    Adaptation, not the paper: replaces ACT (which assumes no clean labels)
    with the validation set every other condition already uses for early
    stopping and checkpoint selection.
    """
    callback = warmup.checkpoint_callback
    best = callback.best_model_path if callback is not None else ""
    if not best:
        logger.warning(
            "AIO2: phase2_trigger=val_best but the warm-up saved no best "
            "checkpoint (no monitored ModelCheckpoint?) — phase 2 (O2C) skipped."
        )
        return None
    epoch = torch.load(best, map_location="cpu", weights_only=False)["epoch"]
    info["aio2/resume_checkpoint_epoch"] = float(epoch + 1)  # completed epochs
    return Path(best)


def stamp_aio2_info(trainer: Any, info: Dict[str, float]) -> None:
    """Write ``aio2/*`` diagnostics into ``trainer.callback_metrics``.

    ``ExperimentsRunner``'s seed worker reads ``callback_metrics`` into the
    per-seed JSON / ``summary.csv``, so this is how ``act_triggered`` and the
    resume epochs reach the results table. Called by ``fit_two_phase`` and
    again by ``train()`` after ``trainer.test()``, which resets the metrics.

    Args:
        trainer: Lightning ``Trainer`` (``callback_metrics`` is a plain dict).
        info: ``aio2/*`` name -> value, as returned by ``fit_two_phase``.
    """
    for key, value in info.items():
        trainer.callback_metrics[key] = torch.tensor(value)
