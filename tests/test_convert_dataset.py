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

current_dir = os.path.dirname(__file__)
root_dir = os.path.join(current_dir, "testing_data")
detection_root_dir = os.path.join(root_dir, "data", "detection_data")
convert_dataset_dir = os.path.join(
    current_dir, "testing_data", "expected_outputs", "dataset", "convert_dataset"
)


class Test_TestConvertDataset(unittest.TestCase):
    def setUp(self) -> None:
        warnings.simplefilter("ignore", category=ImportWarning)
        warnings.simplefilter("ignore", category=DeprecationWarning)
        warnings.simplefilter("ignore", category=FutureWarning)
        warnings.simplefilter("ignore", category=UserWarning)
        self.output_dir = create_folder(os.path.join(root_dir, "test_output"))

    def tearDown(self):
        remove_folder(self.output_dir)

    def test_convert_dataset(self):
        """
        Tests the convert_dataset function
        """
        csv_path = os.path.join(
            detection_root_dir, "geo", "dsg_dataset_with_polygons.csv"
        )
        conversion_processor = ConversionProcessor(
            input_dataset=InstanceSegmentationDataset(
                input_csv_path=csv_path,
                root_dir=os.path.dirname(csv_path),
                keypoint_key="polygon_lists",
            ),
            conversion_strategy=PolygonRNNDatasetConversionStrategy(
                output_dir=self.output_dir,
                output_file_name="polygonrnn_dataset",
                simultaneous_tasks=1,
            ),
        )
        conversion_processor.process()
        expected_df = pd.read_csv(
            os.path.join(convert_dataset_dir, "polygonrnn_dataset.csv")
        ).sort_values("image")
        output_df = pd.read_csv(
            os.path.join(self.output_dir, "polygonrnn_dataset.csv")
        ).sort_values("image")
        pd.testing.assert_frame_equal(
            expected_df.reset_index(drop=True), output_df.reset_index(drop=True)
        )
