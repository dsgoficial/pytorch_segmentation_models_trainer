# -*- coding: utf-8 -*-
"""
/***************************************************************************
 pytorch_segmentation_models_trainer
                              -------------------
        begin                : 2021-04-08
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
import warnings

import numpy as np
import skan
import torch
import matplotlib.pyplot as plt
from matplotlib.testing.compare import compare_images
from parameterized.parameterized import parameterized
from pytorch_segmentation_models_trainer.utils.tensor_utils import (
    polygons_to_tensorpoly,
    tensorpoly_pad,
    SpatialGradient,
    get_scharr_kernel2d,
    batch_to_cuda,
    tensor_dict_to_device,
)
from tests.utils import BasicTestCase

current_dir = os.path.dirname(__file__)
root_dir = os.path.join(current_dir, "testing_data")


class Test_TensorUtils(BasicTestCase):
    def setUp(self):
        super().setUp()
        self.output_dir = self.make_temp_dir()

    def test_tensor_utils(self) -> None:
        device = "cpu"
        np.random.seed(0)
        torch.manual_seed(0)
        padding = (0, 1)

        batch_size = 2
        poly_count = 3
        vertex_min_count = 4
        vertex_max_count = 5

        polygons_batch = []
        for batch_i in range(batch_size):
            polygons = []
            for poly_i in range(poly_count):
                vertex_count = np.random.randint(vertex_min_count, vertex_max_count)
                polygon = np.random.uniform(0, 1, (vertex_count, 2))
                polygons.append(polygon)
            polygons_batch.append(polygons)
        tensorpoly = polygons_to_tensorpoly(polygons_batch)
        assert torch.equal(
            tensorpoly.batch,
            torch.tensor(
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
            ),
        )
        self.assertEqual(tensorpoly.pos.shape, torch.Size([24, 2]))
        self.assertEqual(tensorpoly.poly_slice.shape, torch.Size([6, 2]))
        assert torch.equal(
            tensorpoly.poly_slice,
            torch.tensor([[0, 4], [4, 8], [8, 12], [12, 16], [16, 20], [20, 24]]),
        )

        tensorpoly.to(device)

        tensorpoly = tensorpoly_pad(tensorpoly, padding)
        to_padded_index = tensorpoly.to_padded_index
        self.assertEqual(to_padded_index.shape, torch.Size([30]))
        assert torch.equal(
            to_padded_index,
            torch.tensor(
                [
                    0,
                    1,
                    2,
                    3,
                    0,
                    4,
                    5,
                    6,
                    7,
                    4,
                    8,
                    9,
                    10,
                    11,
                    8,
                    12,
                    13,
                    14,
                    15,
                    12,
                    16,
                    17,
                    18,
                    19,
                    16,
                    20,
                    21,
                    22,
                    23,
                    20,
                ]
            ),
        )

    def test_spatial_gradient(self):
        input_tensor = torch.rand(1, 1, 10, 10)
        grad_layer = SpatialGradient(mode="sobel", order=1)
        output = grad_layer(input_tensor)
        # Expected shape: (B, C, 2, H, W) for order 1
        self.assertEqual(output.shape, (1, 1, 2, 10, 10))

        grad_layer_2 = SpatialGradient(mode="sobel", order=2)
        output_2 = grad_layer_2(input_tensor)
        # Expected shape: (B, C, 3, H, W) for order 2
        self.assertEqual(output_2.shape, (1, 1, 3, 10, 10))

    def test_scharr_kernel(self):
        kernel = get_scharr_kernel2d()
        self.assertEqual(kernel.shape, (2, 3, 3))

    def test_batch_to_cuda(self):
        # Even if we don't have CUDA, we can test that it tries to call .cuda()
        from unittest.mock import MagicMock

        mock_tensor = MagicMock()
        mock_tensor.cuda.return_value = "cuda_tensor"
        batch = {"data": mock_tensor}
        res = batch_to_cuda(batch)
        self.assertEqual(res["data"], "cuda_tensor")

    def test_tensor_dict_to_device(self):
        from unittest.mock import MagicMock

        mock_tensor = MagicMock()
        tensor_dict = {"data": mock_tensor}
        tensor_dict_to_device(tensor_dict, "cpu")
        mock_tensor.to.assert_called_with("cpu")
