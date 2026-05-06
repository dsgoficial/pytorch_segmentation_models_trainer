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

import numpy as np
from pytorch_segmentation_models_trainer.tools.data_handlers.data_writer import (
    ObjectDetectionDataWriter,
    RasterDataWriter,
    VectorDatabaseDataWriter,
    VectorFileDataWriter,
)


class ExportInferenceTemplate(ABC):
    def __init__(self) -> None:
        super().__init__()

    def save_inference(
        self, inference: Union[np.array, Dict[str, np.array]], profile: dict
    ) -> None:
        self.writer.write_data(inference, profile)


class RasterExportInferenceStrategy(ExportInferenceTemplate):
    def __init__(self, output_file_path, output_profile=None):
        super(RasterExportInferenceStrategy, self).__init__()
        self.output_file_path = output_file_path
        self.writer = RasterDataWriter(
            output_file_path=output_file_path, output_profile=output_profile
        )

    def save_inference(
        self, inference: Union[np.array, Dict[str, np.array]], profile: dict
    ) -> None:
        inference = inference["seg"] if isinstance(inference, dict) else inference
        super(RasterExportInferenceStrategy, self).save_inference(inference, profile)


class MultipleRasterExportInferenceStrategy(ExportInferenceTemplate):
    def __init__(self, output_folder, output_basename, output_profile=None):
        super(MultipleRasterExportInferenceStrategy, self).__init__()
        self.output_folder = output_folder
        self.output_basename = output_basename
        self.output_profile = output_profile

    def save_inference(
        self, inference_dict: Dict[str, np.array], profile: dict
    ) -> None:
        name = "" if "input_name" not in profile else f'_{profile.pop("input_name")}'
        for key, value in inference_dict.items():
            file_name = f"{key}_{self.output_basename}"
            output_file_path = os.path.join(
                self.output_folder, f"{key}{name}_{self.output_basename}"
            )
            writer = RasterDataWriter(
                output_file_path=output_file_path, output_profile=self.output_profile
            )
            writer.write_data(value, profile)


class ConfidenceExportStrategy:
    """Export per-pixel confidence metrics as individual GeoTIFF files.

    Each metric (e.g. ``max_prob``, ``entropy``, ``margin``) is saved in its
    own subfolder under ``output_file_path``::

        output_file_path/
        ├── confidence_max_prob/
        │   └── <image_name>.tif    (float32, 1 band)
        ├── confidence_entropy/
        │   └── <image_name>.tif    (float32, 1 band)
        └── confidence_margin/
            └── <image_name>.tif    (float32, 1 band)
    """

    def __init__(self, output_file_path):
        self.output_file_path = output_file_path

    def save_confidence(self, metric_name, data, profile):
        """Save a single confidence metric as a GeoTIFF.

        Args:
            metric_name: Name used for the subfolder (e.g. ``"entropy"``).
            data: np.ndarray [H, W, 1] float32 with values in [0, 1].
            profile: rasterio-compatible profile dict (must contain
                ``input_name`` and ``suffix`` keys).
        """
        subfolder = os.path.join(self.output_file_path, f"confidence_{metric_name}")
        input_name = profile.get("input_name", "")
        suffix = profile.get("suffix", ".tif")
        output_file_path = os.path.join(subfolder, f"{input_name}{suffix}")
        writer = RasterDataWriter(output_file_path=output_file_path)
        writer.write_data(data, profile)


class VectorFileExportInferenceStrategy(ExportInferenceTemplate):
    def __init__(self, output_file_path, driver="GeoJSON"):
        super(VectorFileExportInferenceStrategy, self).__init__()
        self.output_file_path = output_file_path
        self.driver = driver
        self.writer = VectorFileDataWriter(
            output_file_name=output_file_path, driver=driver
        )


class VectorDatabaseExportInferenceStrategy(ExportInferenceTemplate):
    def __init__(
        self,
        user: str,
        database: str,
        password: str,
        sql: str,
        host: str,
        port: int,
        table_name: str = "buildings",
        geometry_column: str = "geom",
    ):
        super(VectorDatabaseExportInferenceStrategy, self).__init__()
        self.writer = VectorDatabaseDataWriter(
            user=user,
            password=password,
            database=database,
            sql=sql,
            host=host,
            port=port,
            table_name=table_name,
            geometry_column=geometry_column,
        )


class ObjectDetectionExportInferenceStrategy(ExportInferenceTemplate):
    def __init__(self, output_file_path):
        super(ObjectDetectionExportInferenceStrategy, self).__init__()
        self.output_file_path = output_file_path
        self.writer = ObjectDetectionDataWriter(output_file_path=output_file_path)
