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
from pytorch_segmentation_models_trainer.custom_callbacks.training_callbacks import (
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
    if seed is not None:
        set_training_seed(seed, deterministic_cudnn=cfg.get("deterministic_cudnn", False))

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
    model.setup("fit")
    trainer = Trainer(**cfg.pl_trainer, logger=trainer_logger, callbacks=callback_list)
    trainer.fit(model)
    if "test_dataset" in cfg:
        logger.info("test_dataset found in config — running trainer.test()")
        trainer.test(model)
    return trainer


if __name__ == "__main__":
    train()
