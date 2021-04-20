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
from dataclasses import dataclass
from pathlib import Path

import rasterio
from rasterio.plot import reshape_as_image
from typing import List
from pytorch_segmentation_models_trainer.tools.data_readers.raster_reader import (
    RasterFile, MaskOutputType
)
from pytorch_segmentation_models_trainer.tools.data_readers.vector_reader import (
    GeoDF, GeomTypeEnum, GeomType
)

@dataclass
class MaskBuilder:
    root_dir: str
    output_csv_path: str
    image_root_dir: str = 'images'
    replicate_image_folder_structure: bool = True
    relative_path_on_csv: bool = True
    build_polygon_mask: bool = True
    polygon_mask_folder_name: str = 'polygon_masks'
    build_boundary_mask: bool = True
    boundary_mask_folder_name: str = 'boundary_masks'
    build_vertex_mask: bool = True
    vertex_mask_folder_name: str = 'vertex_masks'

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



def build_mask_func(input_raster_path: str, input_vector: GeoDF, \
    output_dir: str, mask_type_list: List[GeomType], mask_output_type: MaskOutputType):
    """[summary]

    Args:
        input_raster_path (str): [description]
        input_vector (GeoDF): [description]
        mask_type_list (List[GeomType]): [description]
        mask_output_type (MaskOutputType): [description]
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    raster_df = RasterFile(file_name=input_raster_path)
    return raster_df.build_mask(
        input_vector_layer=input_vector,
        output_dir=output_dir,
        mask_types=mask_type_list,
        mask_output_type=mask_output_type
    )

