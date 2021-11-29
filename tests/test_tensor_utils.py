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
import unittest
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
)
from pytorch_segmentation_models_trainer.utils.os_utils import (
    create_folder,
    remove_folder,
)

current_dir = os.path.dirname(__file__)
root_dir = os.path.join(current_dir, "testing_data")


class Test_TensorUtils(unittest.TestCase):
    def setUp(self):
        warnings.simplefilter("ignore", category=ImportWarning)
        warnings.simplefilter("ignore", category=DeprecationWarning)
        warnings.simplefilter("ignore", category=FutureWarning)
        warnings.simplefilter("ignore", category=UserWarning)
        self.output_dir = create_folder(os.path.join(root_dir, "test_output"))
        return super().setUp()

    def tearDown(self) -> None:
        remove_folder(self.output_dir)

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
