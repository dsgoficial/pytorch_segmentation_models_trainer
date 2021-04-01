# -*- coding: utf-8 -*-
"""
/***************************************************************************
 segmentation_models_trainer
                              -------------------
        begin                : 2021-03-25
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
import numpy as np
import hydra
import segmentation_models_pytorch as smp
import torch
from parameterized import parameterized
from hydra.experimental import initialize, compose
from pytorch_segmentation_models_trainer.model_loader.frame_field_model import (
    FrameFieldModel
)

from tests.utils import CustomTestCase

input_model_list = [
    (smp.Unet,)
]

class Test_TestFrameFieldModel(CustomTestCase):

    def test_create_instance(self) -> None:
        model = smp.Unet()
        frame_field_model = FrameFieldModel(
            model
        )
        print(frame_field_model)
        return True

    @parameterized.expand(input_model_list)
    def test_create_inference_from_model(self, input_model) -> None:
        model = input_model()
        frame_field_model = FrameFieldModel(
            model
        )
        sample = torch.ones([1, 3, 64, 64])
        with torch.no_grad():
            out = frame_field_model(sample)
        print(out['crossfield'].shape)
        return True

    def test_train_one_epoch(self) -> None:
        return True