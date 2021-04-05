# -*- coding: utf-8 -*-
"""
/***************************************************************************
 pytorch_segmentation_models_trainer
                              -------------------
        begin                : 2021-04-01
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
from enum import Enum
from pathlib import Path

import numpy as np
import rasterio
from geopandas.geoseries import GeoSeries
from pytorch_segmentation_models_trainer.tools.data_readers.vector_reader import (
    GeoDF, GeomType, GeomTypeEnum, handle_features)
from pytorch_segmentation_models_trainer.utils.os_utils import create_folder
from rasterio.features import rasterize
from rasterio.plot import reshape_as_image, reshape_as_raster

suffix_dict = {
    "PNG": ".png",
    "GTiff": ".tif",
    "JPEG": ".jpg"
}

class MaskOutputType(Enum):
    SINGLE_FILE_MULTIPLE_BAND, MULTIPLE_FILES_SINGLE_BAND = range(2)

MaskOutputTypeEnum = MaskOutputType

@dataclass
class RasterFile:
    file_name: str
    channels_last: bool = False

    def read_as_numpy_array(self):
        with rasterio.open(self.file_name) as src:
            numpy_array = src.read()
        return numpy_array if not self.channels_last \
            else reshape_as_image(numpy_array)
    
    def export_to(self, output_dir: Path, output_format: str, output_basename=None):
        with rasterio.open(self.file_name) as src:
            profile = src.profile
            profile['driver'] = output_format
            adjusted_filename = Path(self.file_name).with_suffix(
                suffix_dict[output_format]
            ).name if output_basename is None else output_basename+suffix_dict[output_format]
            output_dir = create_folder(output_dir)
            output_filename = os.path.join(output_dir, adjusted_filename)
            input_raster = src.read()
            with rasterio.open(output_filename, 'w', **profile) as out:
                out.write(input_raster)
        return output_filename
    
    def build_mask(self, input_vector_layer: GeoDF,\
            output_dir: Path, output_filename: str =None,\
            mask_types: list[GeomType] = None,\
            mask_output_type:MaskOutputType = MaskOutputType.SINGLE_FILE_MULTIPLE_BAND
        ) -> str:
        if mask_types is not None and not isinstance(mask_types, list):
            raise Exception('Invalid parameter for mask_types')
        # input handling
        mask_types = [GeomTypeEnum.POLYGON] if mask_types is None else mask_types
        input_name, extension = os.path.basename(self.file_name).split('.')
        output_filename = output_filename if output_filename is not None else input_name+'_polygon_mask'
        output = os.path.join(output_dir, output_filename+'.'+extension)
        #read the raster
        raster_ds = rasterio.open(self.file_name)
        profile = raster_ds.profile.copy()
        profile['count'] = len(mask_types)
        mask_feats = input_vector_layer.get_features_from_bbox(
            raster_ds.bounds.left, raster_ds.bounds.right, raster_ds.bounds.bottom, raster_ds.bounds.top
        )
        raster_iter = (
            self.build_numpy_mask_from_vector_layer(
                mask_feats=mask_feats,
                output_shape=raster_ds.shape,
                transform=profile['transform'],
                mask_type=mask_type
            ) for mask_type in mask_types
        )
        
        save_with_rasterio(output, profile, raster_iter, mask_types)
        return output
    
    def build_numpy_mask_from_vector_layer(self, mask_feats: GeoSeries, output_shape:tuple,\
        transform, mask_type: GeomType = None) -> np.ndarray:
        """Builds numpy mask from vector layer using rasterio.

        Args:
            mask_feats (GeoSeries): Features to be used in mask building.
            output_shape (tuple): Shape of the output numpy array.

        Raises:
            Exception: Invalid mask type. Allowed types: POLYGON_MASK, BOUNDARY_MASK and
            VERTEX_MASK

        Returns:
            np.ndarray: Numpy array
        """
        raster_array = np.zeros(output_shape, dtype=rasterio.uint8)
        rasterize(
            handle_features(mask_feats, mask_type, return_list=True),
            out=raster_array,
            out_shape=output_shape,
            fill=0,
            default_value=1,
            all_touched=True,
            transform=transform,
            dtype=rasterio.uint8
        )
        return raster_array

def save_with_rasterio(output, profile, raster_iter, mask_types):
    raster_array = list(raster_iter)[0] if len(mask_types) == 1 \
        else np.dstack(raster_iter)
    with rasterio.open(output, 'w', **profile) as out:
        if len(raster_array.shape) == 2:
            out.write(raster_array, 1)
        else:
            out.write(reshape_as_raster(raster_array))
