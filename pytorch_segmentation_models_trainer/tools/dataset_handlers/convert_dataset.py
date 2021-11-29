# -*- coding: utf-8 -*-
"""
/***************************************************************************
 pytorch_segmentation_models_trainer
                              -------------------
        begin                : 2021-10-05
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
import csv
import itertools
import json
import os
from typing import Dict, List
import numpy as np
from copy import deepcopy
from pathlib import Path
from PIL import Image
from dataclasses import dataclass
from abc import ABC, abstractmethod
import dataclasses
from omegaconf import MISSING, DictConfig, OmegaConf
from shapely.geometry.polygon import Polygon

from pytorch_segmentation_models_trainer.dataset_loader.dataset import (
    AbstractDataset,
    InstanceSegmentationDataset,
)
from pytorch_segmentation_models_trainer.tools.parallel_processing.process_executor import (
    Executor,
    ProcessPoolExecutor,
)
from pytorch_segmentation_models_trainer.utils.os_utils import create_folder
from pytorch_segmentation_models_trainer.utils import polygonrnn_utils


@dataclass
class AbstractConversionStrategy(ABC):
    @abstractmethod
    def convert(self, input_dataset: AbstractDataset):
        pass


@dataclass
class PolygonRNNDatasetConversionStrategy(AbstractConversionStrategy):
    output_dir: str = MISSING
    output_file_name: str = MISSING
    output_images_folder: str = "images_croped"
    output_polygons_folder: str = "polygons_croped"
    write_output_files: bool = True
    original_images_folder_name: str = "images"
    simultaneous_tasks: int = 1
    image_size: int = 224

    def __post_init__(self):
        self.output_dir = create_folder(self.output_dir)
        self.output_images_folder = create_folder(
            os.path.join(self.output_dir, self.output_images_folder)
        )
        self.output_polygons_folder = create_folder(
            os.path.join(self.output_dir, self.output_polygons_folder)
        )
        self.output_file_name = os.path.join(
            self.output_dir, f"{self.output_file_name}.csv"
        )

    def convert(self, input_dataset: AbstractDataset):
        if not isinstance(input_dataset, InstanceSegmentationDataset):
            raise TypeError(
                "input_dataset must be an instance of InstanceSegmentationDataset"
            )

        lambda_func = lambda x: self._convert_single(x[0], x[1])
        executor = Executor(
            compute_func=lambda_func, simultaneous_tasks=self.simultaneous_tasks
        )
        ds_len = len(input_dataset)
        output = (
            executor.execute_tasks(self._build_generator(input_dataset), ds_len)
            if self.simultaneous_tasks > 1
            else [self._convert_single(i) for i in self._build_generator(input_dataset)]
        )
        # output is a list of lists, so we have to flatten it using itertools.chain
        self._write_output_ds(itertools.chain(*output))

    def _build_generator(self, input_dataset: AbstractDataset):
        return (
            (
                input_dataset.get_path(idx),
                input_dataset.get_path(idx, key=input_dataset.keypoint_key),
            )
            for idx in range(len(input_dataset))
        )

    def _convert_single(self, image_path: str, json_path: str) -> List[Dict]:
        """[summary]

        Args:
            image_path (str): [description]
            json_path (str): [description]

        Returns:
            List[Dict]: [description]
        """
        image = Image.open(image_path)
        with open(json_path) as f:
            json_object = json.load(f)
        image_name = Path(image_path).stem
        output_image_folder = create_folder(
            os.path.join(self.output_images_folder, image_name)
        )
        output_polygon_folder = create_folder(
            os.path.join(self.output_polygons_folder, image_name)
        )
        csv_entries_list = []
        for i, item in enumerate(json_object["objects"]):
            min_row, min_col, max_row, max_col = self._get_bounds(json_object, item)
            if max_row - min_row == 0 or max_col - min_col == 0:
                continue
            scale_h, scale_w = polygonrnn_utils.get_scales(
                min_row, min_col, max_row, max_col
            )
            csv_entries_list.append(
                {
                    "image": os.path.join(
                        Path(self.output_images_folder).stem, image_name, f"{i}.png"
                    ),
                    "mask": os.path.join(
                        Path(self.output_polygons_folder).stem, image_name, f"{i}.json"
                    ),
                    "scale_h": scale_h,
                    "scale_w": scale_w,
                    "min_col": min_col,
                    "min_row": min_row,
                    "original_image_path": os.path.join(
                        self.original_images_folder_name,
                        image_path.split(self.original_images_folder_name)[1::][0][1::],
                    ),
                    "original_polygon_wkt": Polygon(item["polygon"]).wkt,
                }
            )
            if not self.write_output_files:
                continue
            self._crop_image(
                image,
                os.path.join(output_image_folder, f"{i}.png"),
                min_row,
                min_col,
                max_row,
                max_col,
            )
            self._crop_polygon(
                item["polygon"],
                os.path.join(output_polygon_folder, f"{i}.json"),
                min_row,
                min_col,
                scale_h,
                scale_w,
            )
        return csv_entries_list

    def _crop_polygon(self, polygon, output_path, min_row, min_col, scale_h, scale_w):
        polygon_dict = {
            "polygon": [
                [
                    np.maximum(0, np.minimum(223, (points[0] - min_col) * scale_w)),
                    np.maximum(0, np.minimum(223, (points[1] - min_row) * scale_h)),
                ]
                for points in polygon
            ]
        }
        with open(output_path, "w") as f:
            json.dump(polygon_dict, f)

    def _crop_image(
        self,
        image: Image,
        output_image_name: str,
        min_row: int,
        min_col: int,
        max_row: int,
        max_col: int,
    ) -> None:
        I_obj = image.crop(box=(min_col, min_row, max_col, max_row))
        I_obj_new = I_obj.resize((self.image_size, self.image_size), Image.BILINEAR)
        I_obj_new.save(output_image_name, "PNG")

    def _get_bounds(self, json_object: dict, obj: dict) -> tuple:
        """
        Calculates min_row, min_col, max_row, max_col.

        Args:
            json_object (dict): json object
            obj (dict): item

        Returns:
            tuple: min_row, min_col, max_row, max_col
        """
        min_c = np.min(np.array(obj["polygon"]), axis=0)
        max_c = np.max(np.array(obj["polygon"]), axis=0)
        h_extend = int(round(0.1 * (max_c[1] - min_c[1])))
        w_extend = int(round(0.1 * (max_c[0] - min_c[0])))
        min_row = np.maximum(0, min_c[1] - h_extend)
        min_col = np.maximum(0, min_c[0] - w_extend)
        max_row = np.minimum(json_object["imgHeight"], max_c[1] + h_extend)
        max_col = np.minimum(json_object["imgWidth"], max_c[0] + w_extend)
        return min_row, min_col, max_row, max_col

    def _write_output_ds(self, output_list: list):
        with open(self.output_file_name, "w") as data_file:
            csv_writer = csv.writer(data_file)
            for i, data in enumerate(output_list):
                if i == 0:
                    # writes header
                    csv_writer.writerow(data.keys())
                csv_writer.writerow(data.values())


@dataclass
class ConversionProcessor(ABC):
    input_dataset: AbstractDataset = MISSING
    conversion_strategy: AbstractConversionStrategy = MISSING

    def process(self):
        self.conversion_strategy.convert(self.input_dataset)
