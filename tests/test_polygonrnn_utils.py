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
from unittest.mock import patch

import albumentations as A
import hydra
import numpy as np
import segmentation_models_pytorch as smp
import shapely
import shapely.geometry
import torch
from hydra import compose, initialize
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
                scaled_polygon.equals_exact(expected_outputs[idx], tolerance=1e-6)
                or scaled_polygon.equals(expected_outputs[idx])
            )

    def test_small_polygonrnn_helpers_and_edge_cases(self):
        self.assertEqual(
            polygonrnn_utils.label2vertex([0, 29, 784, 1]),
            [(0, 0.0), (8, 8.285714285714286)],
        )
        vertices = polygonrnn_utils.get_vertex_list(
            [0, 1, 4],
            scale_h=2.0,
            scale_w=4.0,
            min_col=10,
            min_row=20,
            return_cast_func=tuple,
            grid_size=2,
        )
        self.assertIsInstance(vertices, tuple)
        self.assertEqual(len(vertices), 2)

        empty = polygonrnn_utils.get_vertex_list_from_numpy(np.array([28 * 28]))
        self.assertEqual(empty.shape[0], 0)
        tensor_vertices = polygonrnn_utils.get_vertex_list_from_numpy(
            torch.tensor([0, 1, 28 * 28]),
            scale_h=torch.tensor(1.0),
            scale_w=torch.tensor(1.0),
            min_col=torch.tensor(0.0),
            min_row=torch.tensor(0.0),
            return_cast_func=lambda x: x,
        )
        self.assertIsInstance(tensor_vertices, np.ndarray)
        self.assertEqual(
            polygonrnn_utils.get_vertex_list_from_batch_tensors([], [], [], [], []), []
        )

        bbox = polygonrnn_utils.getbboxfromkps([(10, 10), (20, 30)], h=100, w=100)
        self.assertEqual(bbox, (8, 9, 32, 21))

        image = np.ones((2, 3, 1), dtype=np.float32)
        tensor = polygonrnn_utils.img2tensor(image)
        self.assertEqual(tensor.shape, (1, 2, 3))
        restored = polygonrnn_utils.tensor2img(tensor)
        self.assertEqual(restored.dtype, np.uint8)
        self.assertEqual(restored.shape, (2, 3, 1))

    def test_build_arrays_long_polygon_branch(self):
        polygon = np.array([[0, 0], [8, 0], [16, 0], [24, 0], [32, 0]])
        label_array, label_index_array = polygonrnn_utils.build_arrays(
            polygon, num_vertexes=5, sequence_length=6, grid_size=28
        )
        self.assertEqual(label_array.shape, (6, 28 * 28 + 3))
        self.assertTrue(np.all(label_index_array[5:] == 28 * 28))

    def test_geometry_validation_and_cropping_helpers(self):
        self.assertIsInstance(
            polygonrnn_utils.handle_vertices(shapely.geometry.Point(0, 0)),
            shapely.geometry.Point,
        )
        self.assertIsInstance(
            polygonrnn_utils.handle_vertices([]), shapely.geometry.Point
        )
        self.assertIsInstance(
            polygonrnn_utils.handle_vertices([[1, 2]]), shapely.geometry.Point
        )
        self.assertIsInstance(
            polygonrnn_utils.handle_vertices([[0, 0], [1, 1]]),
            shapely.geometry.LineString,
        )
        self.assertIsInstance(
            polygonrnn_utils.handle_vertices([[0, 0], [1, 0], [1, 1], [0, 0]]),
            shapely.geometry.Polygon,
        )

        valid = shapely.geometry.Polygon([(0, 0), (1, 0), (1, 1), (0, 0)])
        invalid = shapely.geometry.Polygon([(0, 0), (1, 1), (1, 0), (0, 1), (0, 0)])
        self.assertEqual(polygonrnn_utils.validate_polygon(valid), [valid])
        self.assertTrue(polygonrnn_utils.validate_polygon(invalid))
        with patch.object(
            polygonrnn_utils,
            "make_valid",
            return_value=shapely.geometry.GeometryCollection(
                [valid, shapely.geometry.Point(0, 0)]
            ),
        ):
            self.assertEqual(polygonrnn_utils.validate_polygon(invalid), [valid])
        with patch.object(
            polygonrnn_utils,
            "make_valid",
            return_value=[valid, shapely.geometry.Point(0, 0)],
        ):
            self.assertEqual(polygonrnn_utils.validate_polygon(invalid), [valid])
        with patch.object(polygonrnn_utils, "make_valid", return_value=object()):
            self.assertEqual(polygonrnn_utils.validate_polygon(invalid), [])

        crop = polygonrnn_utils.crop_polygons_to_bounding_boxes(
            [valid],
            [shapely.geometry.box(0, 0, 0.5, 0.5)],
        )
        self.assertEqual(len(crop), 1)
        duplicate_crop = polygonrnn_utils.crop_polygons_to_bounding_boxes(
            [valid, valid],
            [shapely.geometry.box(0, 0, 2, 2)],
        )
        self.assertEqual(len(duplicate_crop), 1)

        multipolygon = shapely.geometry.MultiPolygon(
            [
                shapely.geometry.Polygon([(0, 0), (1, 0), (1, 1), (0, 0)]),
                shapely.geometry.Polygon([(2, 0), (3, 0), (3, 1), (2, 0)]),
            ]
        )
        multi_crop = polygonrnn_utils.crop_polygons_to_bounding_boxes(
            [multipolygon],
            [shapely.geometry.box(-1, -1, 4, 4)],
        )
        self.assertEqual(len(multi_crop), 2)

    def test_bounds_targets_and_extra_info_helpers(self):
        polygon = shapely.geometry.Polygon([(10, 10), (20, 10), (20, 30), (10, 10)])
        self.assertEqual(polygonrnn_utils.get_scales(10, 10, 20, 30), (22.4, 11.2))
        bounds = polygonrnn_utils.get_extended_bounds(
            polygon, (100, 100), extend_factor=0.1
        )
        self.assertEqual(bounds, (8, 9, 32, 21))
        bounds_np = polygonrnn_utils.get_extended_bounds_from_np_array_polygon(
            np.array(polygon.exterior.coords), (100, 100), extend_factor=0.1
        )
        self.assertEqual(bounds, bounds_np)
        self.assertEqual(
            polygonrnn_utils.get_bboxes_from_polygons([polygon]), [polygon.bounds]
        )
        self.assertEqual(
            len(
                polygonrnn_utils.scale_polygon_list(
                    [polygon],
                    [2.0],
                    [4.0],
                    [1],
                    [2],
                )
            ),
            1,
        )

        targets = polygonrnn_utils.target_list_to_dict(
            [
                {"labels": [1, 2], "boxes": [1.0, 2.0], "score": torch.tensor(0.5)},
                {"labels": [3], "boxes": [3.0, 4.0], "score": torch.tensor(0.75)},
            ]
        )
        self.assertTrue(torch.equal(targets["labels"], torch.tensor([1, 2, 3])))
        self.assertEqual(targets["boxes"].shape, (4,))
        self.assertTrue(torch.allclose(targets["score"], torch.tensor([0.5, 0.75])))
        self.assertTrue(
            torch.equal(
                polygonrnn_utils.target_list_to_dict([{"empty": []}])["empty"],
                torch.tensor([]),
            )
        )

        bboxes = torch.tensor([[0.0, 10.0, 20.0, 30.0]])
        extra = polygonrnn_utils.build_polygonrnn_extra_info_from_bboxes(bboxes)
        self.assertEqual(set(extra), {"scale_h", "scale_w", "min_row", "min_col"})
        extended = polygonrnn_utils.get_extended_bounds_from_tensor_bbox(
            torch.tensor([[10, 10, 20, 30]]),
            image_bounds=(25, 35),
            extend_factor=0.1,
        )
        self.assertTrue(
            torch.equal(extended, torch.tensor([[9, 8, 21, 32]], dtype=torch.int32))
        )

    def test_crop_and_rescale_with_tensor_bboxes_and_extra_crops(self):
        polygon1 = shapely.geometry.Polygon([(0, 0), (2, 0), (2, 2), (0, 0)])
        polygon2 = shapely.geometry.Polygon([(3, 0), (5, 0), (5, 2), (3, 0)])
        bboxes = torch.tensor([[0.0, 0.0, 10.0, 10.0]])
        result = polygonrnn_utils.crop_and_rescale_polygons_to_bounding_boxes(
            [polygon1, polygon2],
            bboxes,
            [(10, 10), (10, 10)],
            extend_factor=0.0,
        )
        self.assertEqual(len(result), 2)
        self.assertIsInstance(result[0]["bbox"], torch.Tensor)
