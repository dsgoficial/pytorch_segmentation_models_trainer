# -*- coding: utf-8 -*-
"""Resuming interrupted runs (``resumable: true``).

A run killed mid-training (SIGTERM from a GPU-sharing guard, OOM, reboot)
used to restart from epoch 0: ``ExperimentsRunner`` deleted the run's
directory before retrying it and only the best-validation checkpoint was
saved, which cannot resume a fit (it lags behind and early stopping would
lose its patience counter).

With ``resumable: true``:

- ``train()`` adds a :func:`resume_checkpoint_callback` that writes
  ``<default_root_dir>/resume_ckpt/<phase>_last.ckpt`` after every
  validation (full Lightning state: weights, optimizer, scheduler, loops,
  callback state);
- on the next attempt ``train()`` resumes with ``trainer.fit(ckpt_path=...)``
  and pins the default logger to the interrupted run's ``version_N``
  directory (:func:`resume_logger`), so the config's ``ModelCheckpoint``
  sees the same ``dirpath`` and restores ``best_model_score`` — otherwise
  Lightning drops it and any later epoch would overwrite the true best;
- ``ExperimentsRunner`` keeps the directory of an interrupted run that has
  resume state (:func:`has_resume_state`);
- ``train()`` removes the resume state once fit + test completed
  (:func:`clear_resume_state`).

AIO2 (``MeanTeacherModel`` with ``phase2_trigger: val_best``) resumes per
phase: ``warmup_last.ckpt`` / ``correction_last.ckpt`` plus the phase-2
hand-off recorded by :func:`write_aio2_state` once the warm-up is done.

Example::

    resumable: true   # top-level key, next to seed / pl_trainer
"""

import json
import logging
import shutil
from pathlib import Path
from typing import Any, Dict, NamedTuple, Optional, Union

import torch
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

logger = logging.getLogger(__name__)

RESUME_DIR = "resume_ckpt"
AIO2_STATE_FILE = "aio2_state.json"


class ResumePoint(NamedTuple):
    """A readable ``<phase>_last.ckpt``.

    Attributes:
        path: Checkpoint file, passed to ``trainer.fit(ckpt_path=...)``.
        state: ``{"callbacks": ...}`` — only the callback states are kept in
            memory (enough for :func:`logger_from_checkpoint`).
    """

    path: Path
    state: Dict[str, Any]


def resume_dir(root: Union[str, Path]) -> Path:
    """``<root>/resume_ckpt`` — where resume state lives for a run rooted at ``root``."""
    return Path(root) / RESUME_DIR


def resume_checkpoint_callback(root: Union[str, Path], phase: str) -> ModelCheckpoint:
    """``ModelCheckpoint`` writing only ``<root>/resume_ckpt/<phase>_last.ckpt``.

    ``save_top_k=1`` with no monitor keeps only the latest full training state,
    overwritten after every validation under a fixed name. (``save_last`` is
    not used: Lightning only writes ``last`` in a step where a top-k
    checkpoint was also saved, i.e. never with ``save_top_k=0``.)

    Args:
        root: The run's ``pl_trainer.default_root_dir``.
        phase: ``"fit"`` for single-phase models; ``"warmup"`` /
            ``"correction"`` for AIO2.

    Returns:
        The callback (append it after the config's callbacks so the config's
        ``ModelCheckpoint`` stays ``trainer.checkpoint_callback``).
    """
    return ModelCheckpoint(
        dirpath=str(resume_dir(root)),
        filename=f"{phase}_last",
        monitor=None,
        save_top_k=1,
        enable_version_counter=False,
    )


def load_resume_checkpoint(root: Union[str, Path], phase: str) -> Optional[ResumePoint]:
    """Return the run's ``<phase>_last.ckpt`` if present and readable.

    A file truncated by a kill during ``torch.save`` is reported and ignored
    (the phase then starts from scratch) instead of crashing the retry.

    Args:
        root: The run's ``pl_trainer.default_root_dir``.
        phase: See :func:`resume_checkpoint_callback`.

    Returns:
        :class:`ResumePoint`, or ``None`` when there is nothing to resume.
    """
    path = resume_dir(root) / f"{phase}_last.ckpt"
    if not path.exists():
        return None
    try:
        full = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as exc:
        logger.warning(
            "Resume checkpoint %s is unreadable (%r) — starting %s from scratch.",
            path,
            exc,
            phase,
        )
        return None
    return ResumePoint(path, {"callbacks": full.get("callbacks", {})})


def logger_from_checkpoint(
    state: Dict[str, Any], root: Union[str, Path]
) -> Optional[TensorBoardLogger]:
    """TensorBoard logger pinned to the interrupted run's ``version_N`` dir.

    Finds the config's ``ModelCheckpoint`` state (any whose ``dirpath`` is not
    the resume dir) and, when it is the logger-derived
    ``<save_dir>/<name>/<version>/checkpoints``, rebuilds that logger.

    Args:
        state: :attr:`ResumePoint.state`.
        root: The run's ``pl_trainer.default_root_dir``.

    Returns:
        The pinned logger, or ``None`` if no such checkpoint dir is recorded.
    """
    own = resume_dir(root).resolve()
    for key, cb_state in state.get("callbacks", {}).items():
        if not key.startswith("ModelCheckpoint"):
            continue
        dirpath = cb_state.get("dirpath")
        if not dirpath or Path(dirpath).resolve() == own:
            continue
        dirpath = Path(dirpath)
        if dirpath.name != "checkpoints":
            continue
        version_dir = dirpath.parent
        return TensorBoardLogger(
            save_dir=str(version_dir.parent.parent),
            name=version_dir.parent.name,
            version=version_dir.name,
        )
    return None


def resume_logger(trainer_logger: Any, point: Optional[ResumePoint], root: Union[str, Path]) -> Any:
    """Logger for a ``Trainer`` that may resume from ``point``.

    Only Lightning's default logger (``True``) is replaced; a configured
    logger keeps its own (runner-injected) version.

    Args:
        trainer_logger: What ``train()`` would pass as ``Trainer(logger=...)``.
        point: The checkpoint being resumed, or ``None``.
        root: The run's ``pl_trainer.default_root_dir``.

    Returns:
        A pinned logger, or ``trainer_logger`` unchanged.
    """
    if point is None or trainer_logger is not True:
        return trainer_logger
    return logger_from_checkpoint(point.state, root) or trainer_logger


def write_aio2_state(root: Union[str, Path], state: Dict[str, Any]) -> None:
    """Record AIO2's phase-2 hand-off (``init_ckpt`` + ``aio2/*`` info).

    Written once the warm-up is done, so a run interrupted before phase 2
    saved its first epoch restarts phase 2 instead of redoing the warm-up.
    """
    resume_dir(root).mkdir(parents=True, exist_ok=True)
    (resume_dir(root) / AIO2_STATE_FILE).write_text(json.dumps(state))


def read_aio2_state(root: Union[str, Path]) -> Optional[Dict[str, Any]]:
    """The state written by :func:`write_aio2_state`, or ``None``."""
    path = resume_dir(root) / AIO2_STATE_FILE
    return json.loads(path.read_text()) if path.exists() else None


def has_resume_state(root: Union[str, Path]) -> bool:
    """Whether the run rooted at ``root`` was interrupted with resume state."""
    rdir = resume_dir(root)
    return rdir.is_dir() and (
        any(rdir.glob("*_last.ckpt")) or (rdir / AIO2_STATE_FILE).exists()
    )


def clear_resume_state(root: Union[str, Path]) -> None:
    """Delete ``<root>/resume_ckpt`` (no-op if absent)."""
    shutil.rmtree(resume_dir(root), ignore_errors=True)
