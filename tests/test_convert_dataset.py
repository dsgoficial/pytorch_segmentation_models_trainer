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
import unittest
import warnings
from hydra import compose, initialize
import hydra

import pandas as pd
from parameterized import parameterized
from pytorch_segmentation_models_trainer.dataset_loader.dataset import (
    InstanceSegmentationDataset,
)
from pytorch_segmentation_models_trainer.tools.dataset_handlers.convert_dataset import (
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


class Test_ConvertDataset(unittest.TestCase):
    def setUp(self) -> None:
        warnings.simplefilter("ignore", category=ImportWarning)
        warnings.simplefilter("ignore", category=DeprecationWarning)
        warnings.simplefilter("ignore", category=FutureWarning)
        warnings.simplefilter("ignore", category=UserWarning)
        self.output_dir = create_folder(os.path.join(root_dir, "test_output"))

    def tearDown(self):
        remove_folder(self.output_dir)

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
