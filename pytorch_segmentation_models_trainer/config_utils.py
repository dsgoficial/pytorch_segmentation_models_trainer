# -*- coding: utf-8 -*-
"""
/***************************************************************************
 pytorch_segmentation_models_trainer
                              -------------------
        begin                : 2021-03-04
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
import hydra
import logging

from omegaconf import DictConfig, OmegaConf

logger = logging.getLogger(__name__)

@hydra.main()
def validate_config(cfg: DictConfig) -> None:
    logger.info(
        "Input configuration: \n%s",
        OmegaConf.to_yaml(cfg)
    )

if __name__=="__main__":
    validate_config()