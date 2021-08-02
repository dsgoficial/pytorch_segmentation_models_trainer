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
from pytorch_segmentation_models_trainer.model_loader.polygon_rnn_model import PolygonRNN
import subprocess
import unittest
from importlib import import_module

import hydra
import numpy as np
import segmentation_models_pytorch as smp
import torch
from hydra.experimental import compose, initialize
from parameterized import parameterized
from pytorch_segmentation_models_trainer.model_loader.frame_field_model import (
    FrameFieldModel, FrameFieldSegmentationPLModel)
from pytorch_segmentation_models_trainer.train import train
from pytorch_segmentation_models_trainer.custom_models import models as pytorch_smt_cm

from tests.utils import CustomTestCase


current_dir = os.path.dirname(__file__)
frame_field_root_dir = os.path.join(
    current_dir, 'testing_data', 'data', 'polygonrnn_data')

class Test_TestPolygonRNNModel(CustomTestCase):

    def make_inference(self, sample, model):
        with torch.no_grad():
            out = model(sample)
        self.assertEqual(
            out.shape,
            torch.Size([sample.shape[0], 3, sample.shape[-2], sample.shape[-1]])
        )

    def test_create_instance(self) -> None:
        polygonrnn = PolygonRNN()
        print(polygonrnn)
        return True

    def test_create_inference_from_model(self) -> None:
        # polygonrnn = PolygonRNN()
        # self.make_inference(
        #     torch.ones([2, 3, 256, 256]),
        #     polygonrnn
        # )
        # TODO
        return True
    
    def test_create_model_from_cfg(self) -> None:
        """
            #TODO
        """
        return True
    
    