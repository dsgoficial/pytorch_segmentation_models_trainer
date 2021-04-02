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
from pathlib import Path

import rasterio
from rasterio.plot import reshape_as_image

from pytorch_segmentation_models_trainer.utils.os_utils import create_folder

suffix_dict = {
    "PNG": ".png",
    "GTiff": ".tif",
    "JPEG": ".jpg"
}

@dataclass
class RasterFile:
    file_name: str
    channels_last: bool = False

    def read_as_numpy_array(self):
        with rasterio.open(self.file_name) as src:
            numpy_array = src.read()
        return numpy_array if not self.channels_last \
            else reshape_as_image(numpy_array)
    
    def export_to(self, output_dir, output_format, output_basename=None):
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
