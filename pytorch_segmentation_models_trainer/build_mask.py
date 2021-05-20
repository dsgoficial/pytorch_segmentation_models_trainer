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
from pytorch_segmentation_models_trainer.tools.data_readers.raster_reader import MaskOutputTypeEnum
from pytorch_segmentation_models_trainer.tools.parallel_processing.process_executor import Executor
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
    build_csv_file_from_concurrent_futures_output,
    build_dir_dict,
    build_generator,
    build_mask_func,
    build_mask_type_list
)

logger = logging.getLogger(__name__)

cs = ConfigStore.instance()
cs.store(name="mask_config", node=MaskBuilder)
cs.store(group="geo_df", name='batch_file', node=BatchFileGeoDFConfig)
cs.store(group="geo_df", name='file', node=FileGeoDFConfig)
cs.store(group="geo_df", name='postgis', node=PostgisConfig)

@hydra.main(config_name="mask_config")
def build_masks(cfg: DictConfig) -> str:
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
    output_dir_dict = build_dir_dict(cfg)
    mask_type_list = build_mask_type_list(cfg)
    mask_func = lambda x: build_mask_func(
        cfg=cfg,
        input_raster_path=x[0],
        input_vector=geo_df,
        output_dir=x[1],
        output_dir_dict=output_dir_dict,
        mask_type_list=mask_type_list,
        output_extension=cfg.mask_output_extension
    )
    executor = Executor(mask_func)
    generator = build_generator(cfg)
    output_list = executor.execute_tasks(
        generator
    )
    csv_file = build_csv_file_from_concurrent_futures_output(
        cfg,
        futures=output_list
    )
    print(f"Dataset saved at {csv_file}")
    return csv_file

if __name__=="__main__":
    build_masks()
