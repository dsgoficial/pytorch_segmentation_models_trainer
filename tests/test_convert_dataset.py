# -*- coding: utf-8 -*-
"""
/***************************************************************************
 segmentation_models_trainer
                              -------------------
        begin                : 2021-10-06
        git sha              : $Format:%H$
        copyright            : (C) 2021 by Philipe Borba -
                                    Cartographic Engineer @ Brazilian Army
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
import json
import unittest
import warnings
from hydra import compose, initialize
import hydra

import pandas as pd
from PIL import Image
from parameterized import parameterized
from pytorch_segmentation_models_trainer.dataset_loader.dataset import (
    InstanceSegmentationDataset,
)
from pytorch_segmentation_models_trainer.tools.dataset_handlers.convert_dataset import (
    AbstractConversionStrategy,
    ConversionProcessor,
    PolygonRNNDatasetConversionStrategy,
)
from pytorch_segmentation_models_trainer.utils.os_utils import (
    create_folder,
    remove_folder,
)
from pytorch_segmentation_models_trainer.convert_ds import convert_dataset

current_dir = os.path.dirname(__file__)
root_dir = os.path.join(current_dir, "testing_data")
detection_root_dir = os.path.join(root_dir, "data", "detection_data")
convert_dataset_dir = os.path.join(
    current_dir, "testing_data", "expected_outputs", "convert_dataset"
)
from tests.utils import BasicTestCase

current_dir = os.path.dirname(__file__)
...


class Test_ConvertDataset(BasicTestCase):
    def setUp(self) -> None:
        super().setUp()
        self.output_dir = self.make_temp_dir()

    @parameterized.expand(
        [(None,), ([f"+conversion_strategy.write_output_files=False"],)]
    )
    def test_convert_dataset(self, extra_overrides):
        """
        Tests the convert_dataset function
        """
        extra_overrides = extra_overrides if extra_overrides is not None else []
        csv_path = os.path.join(
            detection_root_dir, "geo", "dsg_dataset_with_polygons.csv"
        )
        with initialize(config_path="./test_configs"):
            cfg = compose(
                config_name="convert_dataset.yaml",
                overrides=[
                    f"input_dataset.input_csv_path={csv_path}",
                    f"input_dataset.root_dir={os.path.dirname(csv_path)}",
                    f"conversion_strategy.output_dir={self.output_dir}",
                    f"conversion_strategy.simultaneous_tasks={os.cpu_count()}",
                ]
                + extra_overrides,
            )
            convert_dataset(cfg)
        expected_df = pd.read_csv(
            os.path.join(convert_dataset_dir, "polygonrnn_dataset.csv")
        ).sort_values("image")
        output_df = pd.read_csv(
            os.path.join(self.output_dir, "polygonrnn_dataset.csv")
        ).sort_values("image")
        pd.testing.assert_frame_equal(
            expected_df.reset_index(drop=True), output_df.reset_index(drop=True)
        )

    def test_abstract_conversion_strategy_method_body_is_noop(self):
        class ConcreteStrategy(AbstractConversionStrategy):
            def convert(self, input_dataset):
                return super().convert(input_dataset)

        self.assertIsNone(ConcreteStrategy().convert(object()))

    def test_polygonrnn_convert_rejects_non_instance_dataset(self):
        strategy = PolygonRNNDatasetConversionStrategy(
            output_dir=self.output_dir,
            output_file_name="out",
            write_output_files=False,
        )

        with self.assertRaises(TypeError):
            strategy.convert(object())

    def test_polygonrnn_convert_single_skips_degenerate_polygon(self):
        images_dir = create_folder(os.path.join(self.output_dir, "images"))
        image_path = os.path.join(images_dir, "sample.png")
        json_path = os.path.join(self.output_dir, "sample.json")
        Image.new("RGB", (8, 8), color="white").save(image_path)
        with open(json_path, "w") as f:
            json.dump(
                {
                    "imgHeight": 8,
                    "imgWidth": 8,
                    "objects": [{"polygon": [[1, 1], [1, 1], [1, 1]]}],
                },
                f,
            )
        strategy = PolygonRNNDatasetConversionStrategy(
            output_dir=self.output_dir,
            output_file_name="out",
            write_output_files=False,
        )

        output = strategy._convert_single(image_path, json_path)

        self.assertEqual(output, [])
