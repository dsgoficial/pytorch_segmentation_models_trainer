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
import torch
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
from pytorch_segmentation_models_trainer.tools.detection import bbox_handler

current_dir = os.path.dirname(__file__)
root_dir = os.path.join(current_dir, "testing_data")


class Test_TestBoxHandler(unittest.TestCase):
    def setUp(self) -> None:
        warnings.simplefilter("ignore", category=ImportWarning)
        warnings.simplefilter("ignore", category=DeprecationWarning)
        warnings.simplefilter("ignore", category=FutureWarning)
        warnings.simplefilter("ignore", category=UserWarning)
        self.output_dir = create_folder(os.path.join(root_dir, "test_output"))

    def tearDown(self):
        remove_folder(self.output_dir)

    def test_shift_bbox(self) -> None:
        """
        Tests the shift_bbox function
        """
        bbox = [0, 0, 10, 10]
        shift = [5, 5]
        expected_bbox = [5, 5, 15, 15]
        result = bbox_handler.shift_bbox(bbox, *shift)
        self.assertEqual(result, expected_bbox)

    def test_union_merge_postprocess(self) -> None:
        """
        Tests the union_merge_postprocess function
        """
        bboxes = torch.tensor([[0, 0, 10, 10], [0.5, 0.5, 10, 10]])
        scores = torch.tensor([0.9, 0.5])
        labels = torch.tensor([1, 1])
        expected_bbox = torch.tensor([[0, 0, 10, 10]])
        (
            result_boxes,
            result_scores,
            result_labels,
        ) = bbox_handler.union_merge_postprocess(bboxes, scores, labels)
        torch.testing.assert_close(result_boxes, expected_bbox)

    def test_nms_postprocess(self) -> None:
        """
        Tests the union_merge_postprocess function
        """
        bboxes = torch.tensor([[0, 0, 10, 10], [1, 1, 5, 5], [5, 5, 12, 12]])
        scores = torch.tensor([0.9, 0.5, 0.5])
        labels = torch.tensor([1, 1, 1])
        expected_bbox1 = torch.tensor([[0, 0, 10, 10], [1, 1, 5, 5], [5, 5, 12, 12]])
        expected_bbox2 = torch.tensor([[0, 0, 10, 10]])
        (result_boxes1, _, __) = bbox_handler.nms_postprocess(
            bboxes, scores, labels, threshold=0.5, match_metric="IOU"
        )
        (result_boxes2, _, __) = bbox_handler.nms_postprocess(
            bboxes, scores, labels, threshold=0.5, match_metric="IOS"
        )
        torch.testing.assert_close(result_boxes1, expected_bbox1)
        torch.testing.assert_close(result_boxes2, expected_bbox2)
