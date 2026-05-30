# -*- coding: utf-8 -*-
"""
/***************************************************************************
 pytorch_segmentation_models_trainer
                              -------------------
        begin                : 2021-04-22
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
from pytorch_segmentation_models_trainer.tools.dataset_handlers.convert_dataset import (
    ConversionProcessor,
)

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

logger = logging.getLogger(__name__)


@hydra.main(config_path=None)
def convert_dataset(cfg: DictConfig) -> str:
    logger.info(
        "Starting the dataset conversion with the following configuration: \n%s",
        OmegaConf.to_yaml(cfg),
    )
    conversion_processor = ConversionProcessor(
        input_dataset=instantiate(cfg.input_dataset),
        conversion_strategy=instantiate(cfg.conversion_strategy),
    )
    conversion_processor.process()
