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
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Iterator, List

import numpy as np
import rasterio
from geopandas.geoseries import GeoSeries
from pytorch_segmentation_models_trainer.tools.data_readers.vector_reader import (
    GeoDF, GeomType, GeomTypeEnum, handle_features)
from pytorch_segmentation_models_trainer.utils.os_utils import create_folder
from rasterio.features import rasterize
from rasterio.plot import reshape_as_image, reshape_as_raster
from bidict import bidict

suffix_dict = bidict({
    "PNG": ".png",
    "GTiff": ".tif",
    "JPEG": ".jpg"
})

class MaskOutputType(Enum):
    SINGLE_FILE_MULTIPLE_BAND, MULTIPLE_FILES_SINGLE_BAND = range(2)

MaskOutputTypeEnum = MaskOutputType

@dataclass(frozen=True)
class DatasetEntry:
    image: str
    polygon_mask: str
    width: int
    height: int
    bands_means: list = field(default_factory=list)
    bands_stds: list = field(default_factory=list)
    boundary_mask: str = None
    vertex_mask: str = None
    mask_is_single_band: bool = False

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
            mask_types: List[GeomType] = None,\
            mask_output_type: MaskOutputType = MaskOutputType.SINGLE_FILE_MULTIPLE_BAND,\
            mask_output_folders: List[str] = None, filter_area: float = None, output_extension: str = None
        ) -> List[str]:
        output_extension = os.path.basename(self.file_name).split('.')[-1] \
            if output_extension is None else output_extension
        if mask_types is not None and not isinstance(mask_types, list):
            raise Exception('Invalid parameter for mask_types')
        # input handling
        mask_types = [GeomTypeEnum.POLYGON] if mask_types is None else mask_types
        #read the raster
        raster_ds = rasterio.open(self.file_name)
        profile = raster_ds.profile.copy()
        profile['count'] = len(mask_types) \
            if mask_output_type == MaskOutputType.SINGLE_FILE_MULTIPLE_BAND else 1
        profile['compress'] = None
        profile['width'] = raster_ds.width
        profile['height'] = raster_ds.height
        if output_extension is not None:
            profile['driver'] = suffix_dict.inverse[
                f'.{output_extension}' if not output_extension.startswith('.') else output_extension
            ]
        mask_feats = input_vector_layer.get_features_from_bbox(
            raster_ds.bounds.left, raster_ds.bounds.right,
            raster_ds.bounds.bottom, raster_ds.bounds.top,
            filter_area=filter_area
        )  
        return self.build_single_file_multiple_band_mask(
            mask_feats=mask_feats,
            raster_ds=raster_ds,
            profile=profile,
            mask_types=mask_types,
            output_dir=output_dir,
            output_filename=output_filename,
            output_extension=output_extension
        ) if mask_output_type == MaskOutputType.SINGLE_FILE_MULTIPLE_BAND \
        else self.build_multiple_file_single_band_mask(
            mask_feats=mask_feats,
            raster_ds=raster_ds,
            profile=profile,
            mask_types=mask_types,
            output_dir=output_dir,
            output_filename=output_filename,
            mask_output_folders=mask_output_folders,
            output_extension=output_extension
        )
    
    def build_single_file_multiple_band_mask(self, mask_feats, raster_ds, profile,\
        mask_types, output_dir, output_filename, output_extension) -> List[str]:
        input_name, _ = os.path.basename(self.file_name).split('.')
        output_filename = output_filename if output_filename is not None else input_name
        raster_iter = (
            self.build_numpy_mask_from_vector_layer(
                mask_feats=mask_feats,
                output_shape=raster_ds.shape,
                transform=profile['transform'],
                mask_type=mask_type
            ) for mask_type in mask_types
        )
        output = os.path.join(output_dir, output_filename+'.'+output_extension)
        save_with_rasterio(output, profile, raster_iter, mask_types)
        return [output]
    
    def build_multiple_file_single_band_mask(self, mask_feats, raster_ds, profile,\
        mask_types, output_dir, output_filename, mask_output_folders, output_extension) -> List[str]:
        input_name, _ = os.path.basename(self.file_name).split('.')
        output_filename = output_filename if output_filename is not None else input_name
        def compute(args):
            idx, mask_output_folder = args
            raster_array = self.build_numpy_mask_from_vector_layer(
                mask_feats=mask_feats,
                output_shape=raster_ds.shape,
                transform=profile['transform'],
                mask_type=mask_types[idx]
            )
            output = os.path.join(output_dir, mask_output_folder, output_filename+'.'+output_extension)
            with rasterio.open(output, 'w', **profile) as out:
                out.write(raster_array, 1)
            return output
        return list(
            map(
                compute, enumerate(mask_output_folders)
            )
        )

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
        # raster_array = np.zeros(output_shape, dtype=rasterio.uint8)
        raster_array = rasterize(
            handle_features(mask_feats, mask_type, return_list=True),
            out_shape=output_shape,
            fill=0,
            default_value=1,
            all_touched=True,
            transform=transform,
            dtype=rasterio.uint8
        )
        return raster_array
    
    def get_image_stats(self):
        raster_ds = rasterio.open(self.file_name)
        raster_np = self.read_as_numpy_array()
        return {
            'width': raster_ds.width,
            'height': raster_ds.height,
            'bands_means': np.mean(raster_np, axis=(1, 2)).tolist(),
            'bands_stds': np.std(raster_np, axis=(1, 2)).tolist()
        }

def save_with_rasterio(output, profile, raster_iter, mask_types):
    raster_array = list(raster_iter)[0] if len(mask_types) == 1 \
        else np.dstack(raster_iter)
    with rasterio.open(output, 'w', **profile) as out:
        if len(raster_array.shape) == 2:
            out.write(raster_array, 1)
        else:
            out.write(reshape_as_raster(raster_array))
