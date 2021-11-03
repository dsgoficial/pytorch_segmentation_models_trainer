# -*- coding: utf-8 -*-
"""
/***************************************************************************
 segmentation_models_trainer
                              -------------------
        begin                : 2021-07-15
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
from pathlib import Path

import albumentations as A
import numpy as np
from dataset_loader.dataset import ObjectDetectionDataset
from numpy.testing import assert_array_equal
from parameterized import parameterized
from pytorch_segmentation_models_trainer.utils.os_utils import (
    create_folder,
    remove_folder,
)
from tools.visualization.base_plot_tools import visualize_image_with_bboxes

current_dir = os.path.dirname(__file__)
root_dir = os.path.join(current_dir, "testing_data")
detection_root_dir = os.path.join(current_dir, "testing_data", "data", "detection_data")


class Test_TestBasePlotTools(unittest.TestCase):
    def setUp(self):
        warnings.simplefilter("ignore", category=ImportWarning)
        warnings.simplefilter("ignore", category=DeprecationWarning)
        warnings.simplefilter("ignore", category=FutureWarning)
        warnings.simplefilter("ignore", category=UserWarning)
        self.output_dir = create_folder(os.path.join(root_dir, "test_output"))

    def tearDown(self):
        remove_folder(self.output_dir)

    def test_visualize_image_with_bboxes(self):
        csv_path = os.path.join(detection_root_dir, "geo", "dsg_dataset.csv")
        obj_det_ds = ObjectDetectionDataset(
            input_csv_path=csv_path,
            root_dir=os.path.dirname(csv_path),
            augmentation_list=A.Compose(
                [A.pytorch.ToTensorV2()],
                bbox_params=A.BboxParams(format="coco", label_fields=["labels"]),
            ),
        )
        image, target = obj_det_ds[0]
        output = visualize_image_with_bboxes(
            image.unsqueeeze(0), target["boxes"].unsqueeze(0)
        )
        assert True
