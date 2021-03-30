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
from hydra.experimental import initialize, compose
from pytorch_segmentation_models_trainer.model_loader.frame_field_model import (
    FrameFieldModel
)

class Test_TestFrameFieldModel(unittest.TestCase):

    def test_create_instance(self) -> None:
        model = smp.Unet()
        frame_field_model = FrameFieldModel(
            model
        )
        sample = torch.ones([1, 3, 64, 64])
        with torch.no_grad():
            out = frame_field_model(sample)
        print(frame_field_model)
        print(out['crossfield'].shape)
        return True

    def test_create_inference_from_model(self) -> None:
        return True

    def test_train_one_epoch(self) -> None:
        return True
