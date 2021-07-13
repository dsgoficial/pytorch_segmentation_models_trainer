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
from collections import OrderedDict
import os
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import List

import numpy as np
import rasterio
from bidict import bidict
from geopandas.geoseries import GeoSeries
from pytorch_segmentation_models_trainer.tools.data_readers.vector_reader import (
    GeoDF, GeomType, GeomTypeEnum, handle_features)
from pytorch_segmentation_models_trainer.utils.os_utils import create_folder
from pytorch_segmentation_models_trainer.utils.polygon_utils import (
    build_crossfield, compute_raster_masks)
from rasterio.plot import reshape_as_image, reshape_as_raster

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
    class_freq: list = field(default_factory=list)
    boundary_mask: str = None
    vertex_mask: str = None
    crossfield_mask: str = None
    distance_mask: str = None
    size_mask: str = None
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
            output_dir: Path, output_dir_dict: OrderedDict, output_filename: str =None,\
            mask_types: List[GeomType] = None, filter_area: float = None, output_extension: str = None,\
            compute_crossfield: bool = False, compute_distances: bool = False, compute_sizes: bool = False
        ) -> List[str]:
        """Build segmentation mask according to parameters.

        Args:
            input_vector_layer (GeoDF): [description]
            output_dir_dict (OrderedDict): [description]
            output_filename (str, optional): [description]. Defaults to None.
            filter_area (float, optional): [description]. Defaults to None.
            output_extension (str, optional): [description]. Defaults to None.
            compute_distances (bool, optional): [description]. Defaults to False.
            compute_sizes (bool, optional): [description]. Defaults to False.

        Returns:
            List[str]: [description]
        """
        mask_types = [GeomTypeEnum.POLYGON] if mask_types is None else mask_types
        output_extension = os.path.basename(self.file_name).split('.')[-1] \
            if output_extension is None else output_extension
        #read the raster
        raster_ds = rasterio.open(self.file_name)
        profile = self._prepare_profile(raster_ds, output_extension)
        mask_feats = input_vector_layer.get_features_from_bbox(
            raster_ds.bounds.left, raster_ds.bounds.right,
            raster_ds.bounds.bottom, raster_ds.bounds.top,
            filter_area=filter_area
        )
        raster_dict = self.build_numpy_mask_from_vector_layer(
            mask_feats=mask_feats,
            output_shape=raster_ds.shape,
            transform=profile['transform'],
            mask_types=mask_types,
            compute_distances=compute_distances,
            compute_sizes=compute_sizes,
            compute_crossfield=compute_crossfield
        )
        raster_ds.close()
        return self.write_masks_to_disk(
            raster_dict=raster_dict,
            profile=profile,
            output_dir_dict=output_dir_dict,
            output_filename=output_filename,
            output_extension=output_extension
        )
    
    def _prepare_profile(self, raster_ds, output_extension: str):
        """Prepares the rasterio profile according to an input raster.

        Args:
            raster_ds ([type]): rasterio dataset
            output_extension (str): extension of the output

        Returns:
            [type]: rasterio profile
        """
        profile = raster_ds.profile.copy()
        profile['compress'] = None
        profile['width'] = raster_ds.width
        profile['height'] = raster_ds.height
        if output_extension is not None:
            profile['driver'] = suffix_dict.inverse[
                f'.{output_extension}' if not output_extension.startswith('.') \
                    else output_extension
            ]
        return profile
    
    def write_masks_to_disk(self, raster_dict, profile,\
            output_dir_dict, output_filename, output_extension,\
            folder_basename=None, replicate_input_structure=True
        ) -> List[str]:
        path_dict = OrderedDict()
        input_name, _ = os.path.basename(self.file_name).split('.')
        output_filename = output_filename if output_filename is not None else input_name
        folder_basename = 'images' if folder_basename is None else folder_basename
        subfolders = str(self.file_name.parents[0]).split(folder_basename)[-1] if replicate_input_structure else ''
        subfolders = subfolders[1::] if subfolders.startswith(os.sep) else subfolders
        for key, path in output_dir_dict.items():
            profile['count'] = 1 if len(raster_dict[key].shape) == 2\
                else min(raster_dict[key].shape)
            output = os.path.join(path, subfolders, output_filename+'.'+output_extension)
            profile['dtype'] = raster_dict[key].dtype
            if 'photometric' in profile:
                profile.pop('photometric')
            with rasterio.open(output, 'w', **profile) as out:
                if len(raster_dict[key].shape) == 2:
                    out.write(raster_dict[key], 1)
                else:
                    out.write(reshape_as_raster(raster_dict[key]))
            path_dict[key] = output
        return path_dict

    def build_numpy_mask_from_vector_layer(self, mask_feats: GeoSeries, output_shape:tuple,\
        transform, mask_types: List[GeomType], compute_distances=True, compute_sizes=True,\
        compute_crossfield=False
    ) -> np.ndarray:
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
        polygons = handle_features(mask_feats, GeomTypeEnum.POLYGON, return_list=True)
        return_dict = compute_raster_masks(
            polygons,
            shape=output_shape,
            transform=transform,
            edges=GeomType.LINE in mask_types,
            vertices=GeomType.POINT in mask_types,
            line_width=3,
            compute_distances=compute_distances,
            compute_sizes=compute_sizes
        )
        if compute_crossfield:
            return_dict['crossfield_masks'] = build_crossfield(
                polygons,
                output_shape,
                transform,
                line_width=4
            )
        return return_dict
    
    def get_image_stats(self):
        raster_ds = rasterio.open(self.file_name)
        raster_np = self.read_as_numpy_array()
        return_dict = {
            'width': raster_ds.width,
            'height': raster_ds.height,
            'bands_means': np.mean(raster_np, axis=(1, 2)).tolist(),
            'bands_stds': np.std(raster_np, axis=(1, 2)).tolist(),
            'class_freq': np.mean(raster_np, axis=(1, 2)) / 255
        }
        raster_ds.close()
        return return_dict

def save_with_rasterio(output, profile, raster_iter, mask_types):
    raster_array = list(raster_iter)[0] if len(mask_types) == 1 \
        else np.dstack(raster_iter)
    with rasterio.open(output, 'w', **profile) as out:
        if len(raster_array.shape) == 2:
            out.write(raster_array, 1)
        else:
            out.write(reshape_as_raster(raster_array))
