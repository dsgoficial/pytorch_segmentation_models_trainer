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
from pytorch_segmentation_models_trainer.utils import polygonrnn_utils
from pytorch_segmentation_models_trainer.train import train

from tests.utils import CustomTestCase

current_dir = os.path.dirname(__file__)
polygon_rnn_root_dir = os.path.join(
    current_dir, "testing_data", "data", "polygon_rnn_data"
)


class Test_TestPolygonRNNUtils(CustomTestCase):
    def test_encode_polygon(self) -> None:
        """
        Tests the function that encodes a polygon into a tensor
        """
        polygon = np.array([[100, 100], [100, 204], [204, 204], [204, 100]])
        label_array, label_index_array = polygonrnn_utils.build_arrays(polygon, 4, 60)
        output_vertex_list = polygonrnn_utils.get_vertex_list(label_index_array[2::])
        np.testing.assert_array_almost_equal(polygon, output_vertex_list)
