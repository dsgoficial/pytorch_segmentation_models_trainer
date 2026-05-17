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
import re

from pytorch_segmentation_models_trainer.custom_callbacks.training_callbacks import (
    FinalMetricsCallback,
    FrameFieldComputeWeightNormLossesCallback,
)
from pytorch_segmentation_models_trainer.model_loader.model import Model
from pytorch_segmentation_models_trainer.model_loader.frame_field_model import (
    FrameFieldSegmentationPLModel,
)

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import Trainer

from pytorch_segmentation_models_trainer.model_loader.model import Model
from pytorch_segmentation_models_trainer.utils.os_utils import import_module_from_cfg
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
            reproducible on GPU).

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
    pl_trainer_cfg = dict(cfg.pl_trainer)
    if deterministic_cudnn:
        pl_trainer_cfg.setdefault("deterministic", True)
    trainer = Trainer(**pl_trainer_cfg, logger=trainer_logger, callbacks=callback_list)
    trainer.fit(model)
    if "test_dataset" in cfg:
        logger.info("test_dataset found in config — running trainer.test()")
        trainer.test(model)
    return trainer


if __name__ == "__main__":
    train()
