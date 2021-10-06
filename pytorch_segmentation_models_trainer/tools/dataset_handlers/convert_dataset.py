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
from pathlib import Path
from PIL import Image
from dataclasses import dataclass
from abc import ABC, abstractmethod
import dataclasses
from omegaconf import MISSING, DictConfig, OmegaConf

from pytorch_segmentation_models_trainer.dataset_loader.dataset import (
    AbstractDataset,
    InstanceSegmentationDataset,
)
from pytorch_segmentation_models_trainer.tools.parallel_processing.process_executor import (
    ProcessPoolExecutor,
)
from pytorch_segmentation_models_trainer.utils.os_utils import create_folder


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

    def convert(self, input_dataset: AbstractDataset):
        if not isinstance(input_dataset, InstanceSegmentationDataset):
            raise TypeError(
                "input_dataset must be an instance of InstanceSegmentationDataset"
            )
        lambda_func = lambda x: self._convert_single(x, input_dataset)
        executor = ProcessPoolExecutor(
            compute_func=lambda_func, simultaneous_tasks=self.simultaneous_tasks
        )
        ds_len = len(input_dataset)
        output = (
            executor.execute_tasks(range(ds_len), ds_len)
            if self.simultaneous_tasks > 1
            else [lambda_func(i) for i in range(ds_len)]
        )
        # output is a list of lists, so we have to flatten it using itertools.chain
        self._write_output_ds(itertools.chain(*output))

    def _convert_single(self, idx: int, input_dataset: AbstractDataset) -> List[Dict]:
        """[summary]

        Args:
            idx (int): index of the dataset entry
            input_dataset (AbstractDataset): input dataset to convert

        Returns:
            List[Dict]: List of dictionaries with the converted dataset entry, in the format
            {"image": image_path, "mask": mask_path}
        """
        image = Image.open(input_dataset.get_path(idx))
        json_object = json.load(
            open(input_dataset.get_path(idx, key=input_dataset.keypoint_key))
        )
        image_name = Path(input_dataset.get_path(idx)).stem
        output_image_folder = create_folder(
            os.path.join(self.output_images_folder, image_name)
        )
        output_polygon_folder = create_folder(
            os.path.join(self.output_polygons_folder, image_name)
        )
        csv_entries_list = []
        for i, item in enumerate(json_object["objects"]):
            min_row, min_col, max_row, max_col = self._get_bounds(json_object, item)
            scale_h, scale_w = self._get_scales(min_row, min_col, max_row, max_col)
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
            csv_entries_list.append(
                {
                    "image": os.path.join(
                        Path(self.output_images_folder).stem, image_name, f"{i}.png"
                    ),
                    "mask": os.path.join(
                        Path(self.output_polygons_folder).stem, image_name, f"{i}.json"
                    ),
                }
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

    def _get_scales(
        self, min_row: int, min_col: int, max_row: int, max_col: int
    ) -> tuple:
        """
        Gets scales for the image.

        Args:
            min_row (int): min row
            min_col (int): min col
            max_row (int): max row
            max_col (int): max col

        Returns:
            tuple: scale_h, scale_w
        """
        object_h = max_row - min_row
        object_w = max_col - min_col
        scale_h = 224.0 / object_h
        scale_w = 224.0 / object_w
        return scale_h, scale_w

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
            for i, result in enumerate(output_list):
                data = dataclasses.asdict(result)
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
