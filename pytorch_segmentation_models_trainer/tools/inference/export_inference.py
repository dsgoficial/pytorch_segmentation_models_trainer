# -*- coding: utf-8 -*-
"""
/***************************************************************************
 pytorch_segmentation_models_trainer
                              -------------------
        begin                : 2021-07-15
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

from abc import ABC
import os
from typing import Dict, Union

import albumentations as A
import numpy as np
import rasterio
from pytorch_segmentation_models_trainer.tools.data_handlers.data_writer import (
    RasterDataWriter, VectorDatabaseDataWriter, VectorFileDataWriter)

class ExportInferenceTemplate(ABC):
    def __init__(self, input_raster_path) -> None:
        super().__init__()
        self.input_raster_path = input_raster_path
        with rasterio.open(input_raster_path) as src:
            self.profile = src.profile
            self.crs = src.crs
    def save_inference(self, inference: Union[np.array, Dict[str, np.array]]) -> None:
        self.writer.write_data(inference)

class RasterExportInferenceStrategy(ExportInferenceTemplate):
    def __init__(self, input_raster_path, output_file_path):
        super(RasterExportInferenceStrategy, self).__init__(input_raster_path)
        self.output_file_path = output_file_path
        self.writer = RasterDataWriter(
            output_file_path=output_file_path,
            profile=self.profile
        )
    def save_inference(self, inference: Union[np.array, Dict[str, np.array]]) -> None:
        inference = inference['seg'] if isinstance(inference, dict) else inference
        super(RasterExportInferenceStrategy, self).save_inference(inference)

class MultipleRasterExportInferenceStrategy(ExportInferenceTemplate):
    def __init__(self, input_raster_path, output_folder, output_basename):
        super(MultipleRasterExportInferenceStrategy, self).__init__(input_raster_path)
        self.output_folder = output_folder
        self.output_basename = output_basename
    
    def save_inference(self, inference_dict: Dict[str, np.array]) -> None:
        for key, value in inference_dict.items():
            output_file_path = os.path.join(
                self.output_folder,
                f'{key}_{self.output_basename}'
            )
            writer = RasterDataWriter(
                output_file_path=output_file_path,
                profile=self.profile
            )
            writer.write_data(value)


class VectorFileExportInferenceStrategy(ExportInferenceTemplate):
    def __init__(self, input_raster_path, output_file_path, driver="GeoJSON"):
        super(VectorFileExportInferenceStrategy, self).__init__(input_raster_path)
        self.output_file_path = output_file_path
        self.driver = driver
        self.writer = VectorFileDataWriter(
            output_file_path=output_file_path,
            crs=f"EPSG:{self.crs.to_epsg()}",
            driver=driver
        )

class VectorDatabaseExportInferenceStrategy(ExportInferenceTemplate):
    def __init__(self, input_raster_path, user: str, database: str, password: str, sql: str,\
        host: str, port: int, table_name: str = "buildings", geometry_column: str = "geom"):
        super(VectorDatabaseExportInferenceStrategy, self).__init__(input_raster_path)
        self.writer = VectorDatabaseDataWriter(
            user=user,
            password=password,
            database=database,
            sql=sql,
            crs=f"EPSG:{self.crs.to_epsg()}",
            host=host,
            port=port,
            table_name=table_name,
            geometry_column=geometry_column
        )