# -*- coding: utf-8 -*-
"""
/***************************************************************************
 pytorch_segmentation_models_trainer
                              -------------------
        begin                : 2021-03-01
        git sha              : $Format:%H$
        copyright            : (C) 2021 by Philipe Borba - Cartographic Engineer
                                                            @ Brazilian Army
        email                : philipeborba at gmail dot com
 ***************************************************************************/
/***************************************************************************
 *                                                                         *
 *   This program is free software; you can redistribute it and/or modify  *
 *   it under the terms of the GNU General Public License as published by  *
 *   the Free Software Foundation; either version 2 of the License, or     *
 *   (at your option) any later version.                                   *
 *                                                                         *
 ****
"""

import logging
import os
import re
from pathlib import Path


from pytorch_segmentation_models_trainer.custom_callbacks.training_callbacks import (
    FinalMetricsCallback,
    FrameFieldComputeWeightNormLossesCallback,
)
from pytorch_segmentation_models_trainer.model_loader.model import Model
from pytorch_segmentation_models_trainer.model_loader.frame_field_model import (
    FrameFieldSegmentationPLModel,
)
from pytorch_segmentation_models_trainer.model_loader.mean_teacher_model import (
    MeanTeacherModel,
    fit_two_phase,
    stamp_aio2_info,
)

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping

from pytorch_segmentation_models_trainer.utils.os_utils import import_module_from_cfg
from pytorch_segmentation_models_trainer.utils.resume_utils import (
    clear_resume_state,
    load_resume_checkpoint,
    resume_checkpoint_callback,
    resume_logger,
)
from pytorch_segmentation_models_trainer.utils.seed_utils import set_training_seed

logger = logging.getLogger(__name__)

_OLD_MONITOR_RE = re.compile(r"^(train|val|test)/(.+)$")
_OLD_FILENAME_RE = re.compile(r"\{(train|val|test)/[^}]+\}")


def fix_callback_metric_convention(cfg: DictConfig) -> DictConfig:
    """Detect and auto-correct {prefix}/{name} monitor keys to {name}/{prefix}.

    Scans all callback entries in ``cfg.callbacks`` for a ``monitor`` key
    using the deprecated ``{train|val|test}/metric`` convention and rewrites
    it to ``metric/{train|val|test}``, which groups train and val curves on
    the same TensorBoard chart.

    Also warns when a ``filename`` template contains ``{val/...}`` or similar
    patterns that the OS interprets as path separators.

    Args:
        cfg: Hydra DictConfig for the training run.

    Returns:
        Updated DictConfig (same object when no changes are needed).
    """
    if "callbacks" not in cfg:
        return cfg

    for i, cb in enumerate(cfg.callbacks):
        monitor = cb.get("monitor", None)
        if isinstance(monitor, str):
            m = _OLD_MONITOR_RE.match(monitor)
            if m:
                prefix, name = m.group(1), m.group(2)
                corrected = f"{name}/{prefix}"
                logger.warning(
                    "callbacks[%d].monitor='%s' uses deprecated {prefix}/{name} "
                    "convention — auto-correcting to '%s'.",
                    i,
                    monitor,
                    corrected,
                )
                OmegaConf.update(cfg, f"callbacks.{i}.monitor", corrected)

        filename = cb.get("filename", None)
        if isinstance(filename, str) and _OLD_FILENAME_RE.search(filename):
            logger.warning(
                "callbacks[%d].filename='%s' contains a metric key with '/' "
                "(e.g. {val/loss:.4f}). The OS will interpret this as a directory "
                "separator and save_top_k will not work correctly. "
                "Remove the metric from the filename or rename the metric.",
                i,
                filename,
            )

    return cfg


@hydra.main(config_path=None, version_base="1.2")
def train(cfg: DictConfig):
    """Trains the model.

    Args:
        cfg (DictConfig): hydra yaml config.  When ``cfg.seed`` is set, all
            randomness sources (PyTorch, NumPy, Python ``random``, CUDA, and
            the DataLoader shuffle sampler) are seeded before any model or
            dataset is created.  Set ``cfg.deterministic_cudnn: true`` to
            also force deterministic CuDNN algorithms (slower but fully
            reproducible on GPU). Set ``cfg.resumable: true`` to save the full
            training state every epoch under
            ``<default_root_dir>/resume_ckpt`` and resume from it when the
            run is retried after an interruption (see
            ``utils.resume_utils``); the state is removed once fit + test
            complete.

    Returns:
        Trainer: trainer monitoring object
    """
    seed = cfg.get("seed", None)
    deterministic_cudnn = cfg.get("deterministic_cudnn", False)
    if seed is not None:
        set_training_seed(seed, deterministic_cudnn=deterministic_cudnn)

    cfg = fix_callback_metric_convention(cfg)

    logger.info(
        "Starting the training of a model with the following configuration: \n%s",
        OmegaConf.to_yaml(cfg),
    )
    if "resume_from_checkpoint" in cfg.hyperparameters:
        logger.info(
            f"Resuming from checkpoint: {cfg.hyperparameters.resume_from_checkpoint}"
        )
        model = import_module_from_cfg(cfg.pl_model).load_from_checkpoint(
            cfg.hyperparameters.resume_from_checkpoint,
            cfg=cfg,
            weights_only=False,
        )
    else:
        model = (
            Model(cfg)
            if "pl_model" not in cfg
            else import_module_from_cfg(cfg.pl_model)(cfg)
        )
    trainer_logger = (
        instantiate(cfg.logger, _recursive_=False) if "logger" in cfg else True
    )
    callback_list = (
        [instantiate(i, _recursive_=False) for i in cfg.callbacks]
        if "callbacks" in cfg
        else []
    )
    if isinstance(model, FrameFieldSegmentationPLModel):
        is_norm_loss_added = False
        for callback in callback_list:
            if isinstance(callback, FrameFieldComputeWeightNormLossesCallback):
                is_norm_loss_added = True
                break
        if not is_norm_loss_added:
            callback_list.append(FrameFieldComputeWeightNormLossesCallback())
    if cfg.get("add_final_metrics_callback", True):
        if not any(isinstance(cb, FinalMetricsCallback) for cb in callback_list):
            callback_list.append(FinalMetricsCallback())
    model.setup("fit")
    pl_trainer_cfg = OmegaConf.to_container(cfg.pl_trainer, resolve=True)
    if deterministic_cudnn:
        pl_trainer_cfg.setdefault("deterministic", True)
    aio2_info = None
    if isinstance(model, MeanTeacherModel):
        trainer, aio2_info = _fit_aio2(cfg, model, pl_trainer_cfg, trainer_logger)
    else:
        trainer = _fit_single(cfg, model, pl_trainer_cfg, trainer_logger, callback_list)
    if "test_dataset" in cfg:
        logger.info("test_dataset found in config — running trainer.test()")
        checkpoint_callbacks = getattr(trainer, "checkpoint_callbacks", None)
        if checkpoint_callbacks is None:
            checkpoint_callbacks = [
                cb for cb in getattr(trainer, "callbacks", []) if hasattr(cb, "best_model_path")
            ]
        ckpt_cb = next(
            (
                cb
                for cb in checkpoint_callbacks
                if isinstance(getattr(cb, "best_model_path", None), str)
                and getattr(cb, "best_model_path")
            ),
            checkpoint_callbacks[0] if checkpoint_callbacks else None,
        )
        best_ckpt = getattr(ckpt_cb, "best_model_path", None) if ckpt_cb else None
        if isinstance(best_ckpt, str) and best_ckpt:
            logger.info("Using best checkpoint for test: %s", best_ckpt)
            model_cls = (
                Model if "pl_model" not in cfg else import_module_from_cfg(cfg.pl_model)
            )
            model = model_cls.load_from_checkpoint(
                best_ckpt, cfg=cfg, weights_only=False
            )
        trainer.test(model)
    if aio2_info:
        stamp_aio2_info(trainer, aio2_info)  # trainer.test() reset callback_metrics
    if cfg.get("resumable", False):
        clear_resume_state(_resume_root(pl_trainer_cfg))
    return trainer


def _resume_root(pl_trainer_cfg: dict) -> Path:
    """Run root holding ``resume_ckpt/``: ``default_root_dir`` (Lightning's default: cwd)."""
    return Path(pl_trainer_cfg.get("default_root_dir") or os.getcwd())


def _fit_single(
    cfg: DictConfig, model, pl_trainer_cfg: dict, trainer_logger, callback_list: list
) -> Trainer:
    """Fit a single-phase model, resuming an interrupted run when ``cfg.resumable``.

    With ``resumable: true`` a ``resume_checkpoint_callback("fit")`` is added
    and, if ``resume_ckpt/fit_last.ckpt`` exists, the fit resumes from it in
    the interrupted run's ``version_N`` directory (default logger only).
    """
    point = None
    if cfg.get("resumable", False):
        root = _resume_root(pl_trainer_cfg)
        callback_list = callback_list + [resume_checkpoint_callback(root, "fit")]
        point = load_resume_checkpoint(root, "fit")
        trainer_logger = resume_logger(trainer_logger, point, root)
    trainer = Trainer(**pl_trainer_cfg, logger=trainer_logger, callbacks=callback_list)
    if point is None:
        trainer.fit(model)
    else:
        logger.info("Resuming interrupted run from %s", point.path)
        # weights_only=False: our own checkpoint, with the Hydra config pickled in.
        trainer.fit(model, ckpt_path=str(point.path), weights_only=False)
    return trainer


def _build_callbacks(cfg: DictConfig, drop_early_stopping: bool = False) -> list:
    """Instantiate ``cfg.callbacks`` (+ ``FinalMetricsCallback``), fresh objects."""
    callbacks = (
        [instantiate(i, _recursive_=False) for i in cfg.callbacks]
        if "callbacks" in cfg
        else []
    )
    if drop_early_stopping:
        callbacks = [cb for cb in callbacks if not isinstance(cb, EarlyStopping)]
    if cfg.get("add_final_metrics_callback", True):
        if not any(isinstance(cb, FinalMetricsCallback) for cb in callbacks):
            callbacks.append(FinalMetricsCallback())
    return callbacks


def _fit_aio2(cfg: DictConfig, model, pl_trainer_cfg: dict, trainer_logger):
    """AIO2 (``MeanTeacherModel``): warm-up fit, then O2C fit from ``I_r``.

    Each phase gets its own ``Trainer`` and freshly instantiated callbacks.
    With ``phase2_trigger="act"`` (default) the warm-up trainer has no
    ``EarlyStopping`` — ACT decides when warm-up ends; with ``"val_best"`` it
    keeps it (see ``mean_teacher_model.fit_two_phase``). The correction phase uses
    the config's callbacks unchanged, so it stops like every other condition.
    ``trainer_logger`` is shared: with the default (``True``) each phase gets
    its own ``lightning_logs/version_N`` (0 = warm-up, 1 = correction), which
    keeps phase-1 checkpoints out of phase 2's ``ModelCheckpoint`` directory.

    ``cfg.resumable`` with ``phase2_trigger="val_best"``: each phase also
    saves ``resume_ckpt/<phase>_last.ckpt`` and ``fit_two_phase`` resumes the
    interrupted phase. ACT mode is never resumed (``ACTTracker``'s history is
    not checkpointed), so it saves no resume state and the runner retries it
    from scratch as before.
    """
    resume_root = None
    if cfg.get("resumable", False) and model.phase2_trigger == "val_best":
        resume_root = _resume_root(pl_trainer_cfg)

    def make_trainer(phase: str, point=None) -> Trainer:
        callbacks = _build_callbacks(
            cfg,
            drop_early_stopping=(phase == "warmup" and model.phase2_trigger == "act"),
        )
        if resume_root is not None:
            callbacks.append(resume_checkpoint_callback(resume_root, phase))
        return Trainer(
            **pl_trainer_cfg,
            logger=resume_logger(trainer_logger, point, resume_root),
            callbacks=callbacks,
        )

    return fit_two_phase(model, make_trainer, resume_root=resume_root)


if __name__ == "__main__":
    train()
