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

from pytorch_segmentation_models_trainer.tools.data_readers.raster_reader import RasterFile
from pytorch_segmentation_models_trainer.tools.data_readers.vector_reader import (
    GeoDF
)

def build_mask(input_raster: RasterFile, input_vector: GeoDF, output_file_path: str):
    pass