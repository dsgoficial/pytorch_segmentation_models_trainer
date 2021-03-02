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

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import Trainer

from pytorch_segmentation_models_trainer.model_loader.model import Model

logger = logging.getLogger(__name__)

@hydra.main(config_path="conf", config_name="config")
def train(cfg: DictConfig) -> Trainer:
    """Trains the model.
    Args:
        cfg (DictConfig): hydra yaml config

    Returns:
        Trainer: trainer monitoring object
    """
    logger.info(
        "Starting the training of a model with the following configuration: \n%s",
        OmegaConf.to_yaml(cfg)
    )
    model = Model(cfg)
    trainer_logger = instantiate(cfg.logger) if "logger" in cfg else True
    callback_list = [instantiate(i) for i in cfg.callbacks] if "callbacks" in cfg else []
    trainer = Trainer(**cfg.pl_trainer, logger=trainer_logger, callbacks=callback_list)
    trainer.fit(model)
    return trainer

if __name__=="__main__":
    train()
