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
import numpy as np
import skan
import torch
from pytorch_segmentation_models_trainer.tools.polygonization.skeletonize_tensor_tools import (
    Paths,
    Skeleton,
    TensorSkeleton,
    plot_skeleton,
    skeletons_to_tensorskeleton,
    tensorskeleton_to_skeletons,
)
from tests.utils import BasicTestCase


def build_skeleton1():
    spatial_shape = (10, 10)
    image = np.zeros(spatial_shape, dtype=bool)
    image[2, :] = True
    image[:, 2] = True
    image[7, :] = True
    image[:, 7] = True
    return skan.Skeleton(image, keep_images=False)


def build_skeleton2():
    spatial_shape = (10, 10)
    image = np.zeros(spatial_shape, dtype=bool)
    image[5, :] = True
    image[:, 5] = True
    return skan.Skeleton(image, keep_images=False)


current_dir = os.path.dirname(__file__)
root_dir = os.path.join(current_dir, "testing_data")


class Test_Skeletonize(BasicTestCase):
    def setUp(self):
        super().setUp()
        self.skan_skeletons_list = [build_skeleton1(), build_skeleton2()]
        self.output_dir = self.make_temp_dir()

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
                    0,
                    2,
                    6,
                    1,
                    3,
                    11,
                    4,
                    5,
                    6,
                    6,
                    7,
                    8,
                    9,
                    10,
                    11,
                    6,
                    14,
                    16,
                    18,
                    20,
                    24,
                    11,
                    12,
                    13,
                    11,
                    15,
                    17,
                    19,
                    21,
                    29,
                    22,
                    23,
                    24,
                    24,
                    25,
                    26,
                    27,
                    28,
                    29,
                    24,
                    32,
                    34,
                    29,
                    30,
                    31,
                    29,
                    33,
                    35,
                    36,
                    37,
                    38,
                    39,
                    40,
                    46,
                    41,
                    42,
                    43,
                    44,
                    45,
                    46,
                    46,
                    47,
                    48,
                    49,
                    50,
                    46,
                    51,
                    52,
                    53,
                    54,
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

    def test_tensorskeleton_properties_to_roundtrip_and_plot(self):
        skeleton = Skeleton(
            coordinates=np.array([[0, 0], [0, 1], [1, 1]], dtype=float),
            paths=Paths(
                indices=np.array([0, 1, 2], dtype=np.int64),
                indptr=np.array([0, 3], dtype=np.int64),
            ),
            degrees=np.array([1, 2, 1], dtype=np.int64),
        )
        tensorskeleton = skeletons_to_tensorskeleton([skeleton], device="cpu")

        self.assertEqual(tensorskeleton.num_nodes, 3)
        self.assertEqual(tensorskeleton.num_paths, 1)
        self.assertIsNone(tensorskeleton.to("cpu"))

        output = tensorskeleton_to_skeletons(tensorskeleton)
        self.assertEqual(len(output), 1)
        self.assertEqual(output[0].coordinates.shape, (3, 2))

        plot_skeleton(output[0])

        empty_batch = TensorSkeleton(
            pos=torch.zeros((0, 2)),
            degrees=torch.zeros(0, dtype=torch.long),
            path_index=torch.zeros(0, dtype=torch.long),
            path_delim=torch.zeros(1, dtype=torch.long),
            batch=torch.zeros(0, dtype=torch.long),
            batch_delim=torch.tensor([0, 0], dtype=torch.long),
            batch_size=1,
        )
        self.assertEqual(
            tensorskeleton_to_skeletons(empty_batch)[0].coordinates.shape, (0, 2)
        )

    def test_tensorskeleton_asserts_pos_and_batch_lengths(self):
        with self.assertRaises(AssertionError):
            TensorSkeleton(
                pos=torch.zeros((2, 2)),
                degrees=torch.zeros(2),
                path_index=torch.zeros(0),
                path_delim=torch.zeros(0),
                batch=torch.zeros(1),
                batch_delim=torch.zeros(0),
                batch_size=1,
            )
