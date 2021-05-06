# -*- coding: utf-8 -*-
"""
/***************************************************************************
 pytorch_segmentation_models_trainer
                              -------------------
        begin                : 2021-04-02
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
import os
from dataclasses import dataclass, field
from pathlib import Path
from omegaconf import MISSING, DictConfig, OmegaConf

import rasterio
from rasterio.plot import reshape_as_image
from typing import Any, List
from pytorch_segmentation_models_trainer.tools.data_readers.raster_reader import (
    RasterFile, MaskOutputType
)
from pytorch_segmentation_models_trainer.tools.data_readers.vector_reader import (
    GeoDF, FileGeoDF, GeomTypeEnum, GeomType
)

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
    image_dir_is_relative_to_root_dir: str = True
    replicate_image_folder_structure: bool = True
    relative_path_on_csv: bool = True
    build_polygon_mask: bool = True
    polygon_mask_folder_name: str = 'polygon_masks'
    build_boundary_mask: bool = True
    boundary_mask_folder_name: str = 'boundary_masks'
    build_vertex_mask: bool = True
    vertex_mask_folder_name: str = 'vertex_masks'
        
def replicate_image_structure(cfg):
    input_base_path = str(
        os.path.join(cfg.root_dir, cfg.image_root_dir)
    ) if cfg.image_dir_is_relative_to_root_dir else cfg.image_root_dir
    for mask_type in ['polygon_mask', 'boundary_mask', 'vertex_mask']:
        if getattr(cfg, f"build_{mask_type}"):
            mask_folder_name = getattr(cfg, f"{mask_type}_folder_name")
            output_base_path = str(
                os.path.join(cfg.root_dir, mask_folder_name)
            )
            build_destination_dirs(
                input_base_path=input_base_path,
                output_base_path=output_base_path
            )

def build_mask_type_list(cfg: MaskBuilder):
    """[summary]

    Args:
        cfg (MaskBuilder): [description]

    Returns:
        [type]: [description]
    """
    mask_type_list = []
    if cfg.build_polygon_mask:
        mask_type_list.append(GeomTypeEnum.POLYGON)
    if cfg.build_boundary_mask:
        mask_type_list.append(GeomTypeEnum.LINE)
    if cfg.build_vertex_mask:
        mask_type_list.append(GeomTypeEnum.POINT)
    return mask_type_list

def build_destination_dirs(input_base_path: str, output_base_path: str):
    if input_base_path == os.path.commonpath([input_base_path, output_base_path]) \
        and not output_base_path.split(
            os.path.commonpath([input_base_path, output_base_path])
        )[-1].startswith("/.."):
        raise Exception("input path must not be in output_path")
    return [
        Path(
            dirpath.replace(input_base_path, output_base_path)    
        ).mkdir(parents=True, exist_ok=True) for dirpath, _, __ in os.walk(input_base_path)
    ]

def build_mask_func(input_raster_path: str, input_vector: GeoDF, \
    output_dir: str, mask_type_list: List[GeomType], mask_output_type: MaskOutputType):
    """[summary]

    Args:
        input_raster_path (str): [description]
        input_vector (GeoDF): [description]
        mask_type_list (List[GeomType]): [description]
        mask_output_type (MaskOutputType): [description]
    """
    raster_df = RasterFile(file_name=input_raster_path)
    return raster_df.build_mask(
        input_vector_layer=input_vector,
        output_dir=output_dir,
        mask_types=mask_type_list,
        mask_output_type=mask_output_type
    )

