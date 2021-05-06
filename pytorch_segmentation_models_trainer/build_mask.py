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
import os
from dataclasses import dataclass, field
from typing import Any, List

import hydra
from hydra.core.config_store import ConfigStore
from hydra.utils import instantiate
from omegaconf import MISSING, DictConfig, OmegaConf

from pytorch_segmentation_models_trainer.tools.mask_building.mask_builder import (
    BatchFileGeoDFConfig,
    PostgisConfig,
    FileGeoDFConfig,
    MaskBuilder,
    replicate_image_structure,
)

logger = logging.getLogger(__name__)

cs = ConfigStore.instance()
cs.store(name="config", node=MaskBuilder)
cs.store(group="geo_df", name='batch_file', node=BatchFileGeoDFConfig)
cs.store(group="geo_df", name='file', node=FileGeoDFConfig)
cs.store(group="geo_df", name='postgis', node=PostgisConfig)

@hydra.main(config_name="config")
def build_masks(cfg: DictConfig):
    logger.info(
        "Starting the training of a model with the following configuration: \n%s",
        OmegaConf.to_yaml(cfg)
    )
    if 'root_dir' in cfg.geo_df and cfg.geo_df.root_dir.startswith(".."):
        cfg.geo_df.root_dir = str(
            os.path.join(
                os.path.dirname( __file__ ),
                cfg.geo_df.root_dir
            )
        )
    geo_df = instantiate(cfg.geo_df)
    if cfg.replicate_image_folder_structure:
        replicate_image_structure(cfg)

if __name__=="__main__":
    build_masks()
