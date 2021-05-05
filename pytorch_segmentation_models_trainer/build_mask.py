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

logger = logging.getLogger(__name__)

@dataclass
class VectorReaderConfig:
    pass

@dataclass
class FileGeoDFConfig:
    _target_: str = "pytorch_segmentation_models_trainer.tools.data_readers.vector_reader.FileGeoDF"
    file_name: str = "../tests/testing_data/data/vectors/test_polygons.geojson"

@dataclass
class PostgisConfig(VectorReaderConfig):
    _target_: str = "pytorch_segmentation_models_trainer.tools.data_readers.vector_reader.PostgisGeoDF"
    database: str = "dataset_mestrado"
    sql: str = "select id, geom from buildings"
    user: str = "postgres"
    password: str = "postgres"
    host: str = "localhost"
    port: int = 5432

@dataclass
class BatchFileGeoDFConfig(VectorReaderConfig):
    _target_: str = "pytorch_segmentation_models_trainer.tools.data_readers.vector_reader.BatchFileGeoDF"
    root_dir: str = "../tests/testing_data/data/vectors"
    file_extension: str = "geojson"

@dataclass
class MaskBuilder:
    defaults: List[Any] = field(default_factory=lambda: [{"geo_df": "batch_file"}])
    geo_df: VectorReaderConfig = MISSING
    root_dir: str = '/data'
    output_csv_path: str = '/data'
    image_root_dir: str = 'images'
    replicate_image_folder_structure: bool = True
    relative_path_on_csv: bool = True
    build_polygon_mask: bool = True
    polygon_mask_folder_name: str = 'polygon_masks'
    build_boundary_mask: bool = True
    boundary_mask_folder_name: str = 'boundary_masks'
    build_vertex_mask: bool = True
    vertex_mask_folder_name: str = 'vertex_masks'

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
    if cfg.geo_df.root_dir.startswith(".."):
        cfg.geo_df.root_dir = str(
            os.path.join(
                os.path.dirname( __file__ ),
                cfg.geo_df.root_dir
            )
        )
    geo_df = instantiate(cfg.geo_df)
    print(geo_df)

if __name__=="__main__":
    build_masks()
