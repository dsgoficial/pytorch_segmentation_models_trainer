# -*- coding: utf-8 -*-
"""
/***************************************************************************
 segmentation_models_trainer
                              -------------------
        begin                : 2021-02-25
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

import unittest

import hydra
import numpy as np
import segmentation_models_pytorch as smp
import torch
from hydra.experimental import compose, initialize
from hydra.utils import instantiate
from parameterized import parameterized
from pytorch_segmentation_models_trainer.custom_models import models

input_model_list = [
    (models.DeepLab101, torch.Size([2, 128, 256, 256])),
    (models.DeepLab50, torch.Size([2, 128, 256, 256])),
    (models.FCN101, torch.Size([2, 256, 256, 256])),
    (models.FCN50, torch.Size([2, 256, 256, 256])),
    (models.UNetResNet, torch.Size([2, 32, 256, 256])),
]

class Test_TestCustomModels(unittest.TestCase):
    

    @parameterized.expand(input_model_list)
    def test_create_inference_from_model(self, input_model, expected_output_shape) -> None:
        model = input_model()
        sample = torch.ones([2, 3, 256, 256])
        with torch.no_grad():
            out = model(sample)
        self.assertEqual(
            out.shape,
            expected_output_shape
        )
