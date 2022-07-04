# -*- coding: utf-8 -*-
"""
/***************************************************************************
 segmentation_models_trainer
                              -------------------
        begin                : 2021-08-02
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
import subprocess
import unittest
from importlib import import_module

import albumentations as A
import hydra
import numpy as np
import segmentation_models_pytorch as smp
import shapely
import shapely.geometry
import torch
from hydra.experimental import compose, initialize
from parameterized import parameterized
from pytorch_segmentation_models_trainer.custom_models import models as pytorch_smt_cm
from pytorch_segmentation_models_trainer.dataset_loader.dataset import PolygonRNNDataset
from pytorch_segmentation_models_trainer.model_loader.frame_field_model import (
    FrameFieldModel,
    FrameFieldSegmentationPLModel,
)
from pytorch_segmentation_models_trainer.model_loader.polygon_rnn_model import (
    PolygonRNN,
)
from pytorch_segmentation_models_trainer.train import train
from pytorch_segmentation_models_trainer.utils import polygonrnn_utils

current_dir = os.path.dirname(__file__)
polygon_rnn_root_dir = os.path.join(
    current_dir, "testing_data", "data", "polygon_rnn_data"
)


class Test_PolygonRNNUtils(unittest.TestCase):
    @parameterized.expand(
        [
            ([[100, 100], [100, 204], [204, 204], [204, 100]],),
            (
                [
                    (196.0, 92.0),
                    (204.0, 220.0),
                    (36.0, 220.0),
                    (28.0, 140.0),
                    (20.0, 44.0),
                    (84.0, 20.0),
                ],
            ),
        ]
    )
    def test_encode_polygon(self, coordinates) -> None:
        """
        Tests the function that encodes a polygon into a tensor
        """
        polygon = np.array(coordinates)
        label_array, label_index_array = polygonrnn_utils.build_arrays(
            polygon, len(coordinates), 60
        )
        output_vertex_list = polygonrnn_utils.get_vertex_list(label_index_array[2::])
        np.testing.assert_array_almost_equal(polygon, output_vertex_list)

    def test_encode_polygons_on_batch(self) -> None:
        polygon1 = np.array([[100, 100], [100, 204], [204, 204], [204, 100]])
        polygon2 = np.array([[204, 204], [220, 204], [220, 220], [220, 204]])
        _, label_index_array1 = polygonrnn_utils.build_arrays(polygon1, 4, 60)
        _, label_index_array2 = polygonrnn_utils.build_arrays(polygon2, 4, 60)
        batch = np.stack([label_index_array1[2::], label_index_array2[2::]], axis=0)
        output_batch = polygonrnn_utils.get_vertex_list_from_batch(batch)
        self.assertEqual(output_batch.shape, (2, 4, 2))
        np.testing.assert_array_almost_equal(
            output_batch, np.stack([polygon1, polygon2])
        )

    def test_encode_polygons_on_tensor_batch(self) -> None:
        polygon1 = np.array([[100, 100], [100, 204], [204, 204], [204, 100]])
        polygon2 = np.array([[204, 204], [220, 204], [220, 220]])
        _, label_index_array1 = polygonrnn_utils.build_arrays(polygon1, 4, 60)
        _, label_index_array2 = polygonrnn_utils.build_arrays(polygon2, 3, 60)
        batch = np.stack([label_index_array1[2::], label_index_array2[2::]], axis=0)
        batch_tensor = torch.from_numpy(batch).float()
        output_tensor_batch = polygonrnn_utils.get_vertex_list_from_batch_tensors(
            batch_tensor,
            scale_h=torch.ones([2, 1]),
            scale_w=torch.ones([2, 1]),
            min_col=torch.zeros([2, 1]),
            min_row=torch.zeros([2, 1]),
        )
        for idx, polygon in enumerate([polygon1, polygon2]):
            np.testing.assert_array_almost_equal(
                output_tensor_batch[idx], polygon.astype(np.float32)
            )

    @parameterized.expand(
        [
            ([[100, 100], [100, 204], [204, 204], [204, 100]],),
            (
                [
                    (196.0, 92.0),
                    (204.0, 220.0),
                    (36.0, 220.0),
                    (28.0, 140.0),
                    (20.0, 44.0),
                    (84.0, 20.0),
                ],
            ),
        ]
    )
    def test_get_vertex_list_from_numpy(self, coordinates) -> None:
        polygon = np.array(coordinates)
        label_array, label_index_array = polygonrnn_utils.build_arrays(
            polygon, len(coordinates), 60
        )
        output_vertex_list = polygonrnn_utils.get_vertex_list_from_numpy(
            label_index_array[2::]
        )
        np.testing.assert_array_almost_equal(polygon, output_vertex_list)

    def test_crop_and_rescale_polygons_to_bounding_boxes(self) -> None:
        polygon1 = shapely.geometry.Polygon(
            [(100, 100), (100, 604), (604, 604), (604, 100)]
        )
        polygon2 = shapely.geometry.Polygon(
            [[100, 100], [100, 204], [204, 204], [204, 100]]
        )
        bounding_boxes = [[100, 100, 512, 512], [100, 100, 204, 204]]
        image_bounds_list = [(512, 512), (512, 512)]
        output_polygon_list = (
            polygonrnn_utils.crop_and_rescale_polygons_to_bounding_boxes(
                [polygon1, polygon2],
                bounding_boxes,
                image_bounds_list,
                extend_factor=0.1,
            )
        )
        self.assertEqual(len(output_polygon_list), 2)
        expected_outputs = [
            shapely.wkt.loads(
                "POLYGON ((100 509.9776785714286, 509.9776785714286 509.9776785714286, 509.9776785714286 100, 100 100, 100 509.9776785714286))"
            ),
            polygon2,
        ]
        for idx, output in enumerate(output_polygon_list):
            polygon = shapely.geometry.Polygon(output.pop("polygon"))
            output.pop("bbox")
            scaled_polygon = polygonrnn_utils.scale_shapely_polygon(polygon, **output)
            self.assertTrue(
                scaled_polygon.almost_equals(expected_outputs[idx])
                or scaled_polygon.equals(expected_outputs[idx])
            )
