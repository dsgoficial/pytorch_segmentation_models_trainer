# -*- coding: utf-8 -*-
"""
/***************************************************************************
 pytorch_segmentation_models_trainer
                              -------------------
        begin                : 2021-04-07
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
from pytorch_segmentation_models_trainer.tools.polygonization.skeletonize_tensor_tools import (
    Paths,
    Skeleton,
    plot_skeleton,
    skeletons_to_tensorskeleton,
    tensorskeleton_to_skeletons,
)
from pytorch_segmentation_models_trainer.utils.os_utils import (
    create_folder,
    remove_folder,
)


def build_skeleton1():
    spatial_shape = (10, 10)
    image = np.zeros(spatial_shape, dtype=np.bool)
    image[2, :] = True
    image[:, 2] = True
    image[7, :] = True
    image[:, 7] = True
    return skan.Skeleton(image, keep_images=False)


def build_skeleton2():
    spatial_shape = (10, 10)
    image = np.zeros(spatial_shape, dtype=np.bool)
    image[5, :] = True
    image[:, 5] = True
    return skan.Skeleton(image, keep_images=False)


current_dir = os.path.dirname(__file__)
root_dir = os.path.join(current_dir, "testing_data")


class Test_Skeletonize(unittest.TestCase):
    def setUp(self):
        warnings.simplefilter("ignore", category=ImportWarning)
        warnings.simplefilter("ignore", category=DeprecationWarning)
        warnings.simplefilter("ignore", category=FutureWarning)
        warnings.simplefilter("ignore", category=UserWarning)
        self.skan_skeletons_list = [build_skeleton1(), build_skeleton2()]
        self.output_dir = create_folder(os.path.join(root_dir, "test_output"))
        return super().setUp()

    def tearDown(self) -> None:
        remove_folder(self.output_dir)

    def test_skeletonize(self) -> None:
        device = "cpu"
        np.random.seed(0)
        torch.manual_seed(0)
        skeletons_batch = [
            Skeleton(
                skan_skeleton.coordinates,
                Paths(skan_skeleton.paths.indices, skan_skeleton.paths.indptr),
            )
            for skan_skeleton in self.skan_skeletons_list
        ]
        tensorskeleton = skeletons_to_tensorskeleton(skeletons_batch, device=device)
        self.assertEqual(tensorskeleton.path_index.shape, torch.Size([70]))
        assert torch.equal(
            tensorskeleton.path_index,
            torch.tensor(
                [
                    1,
                    3,
                    7,
                    2,
                    4,
                    12,
                    5,
                    6,
                    7,
                    7,
                    8,
                    9,
                    10,
                    11,
                    12,
                    7,
                    15,
                    17,
                    19,
                    21,
                    25,
                    12,
                    13,
                    14,
                    12,
                    16,
                    18,
                    20,
                    22,
                    30,
                    23,
                    24,
                    25,
                    25,
                    26,
                    27,
                    28,
                    29,
                    30,
                    25,
                    33,
                    35,
                    30,
                    31,
                    32,
                    30,
                    34,
                    36,
                    38,
                    39,
                    40,
                    41,
                    42,
                    48,
                    43,
                    44,
                    45,
                    46,
                    47,
                    48,
                    48,
                    49,
                    50,
                    51,
                    52,
                    48,
                    53,
                    54,
                    55,
                    56,
                ]
            ),
        )
        self.assertEqual(tensorskeleton.path_delim.shape, torch.Size([17]))
        assert torch.equal(
            tensorskeleton.path_delim,
            torch.tensor(
                [0, 3, 6, 9, 15, 21, 24, 30, 33, 39, 42, 45, 48, 54, 60, 65, 70]
            ),
        )
        self.assertEqual(tensorskeleton.batch_delim.shape, torch.Size([3]))
        assert torch.equal(tensorskeleton.batch_delim, torch.tensor([0, 12, 16]))
