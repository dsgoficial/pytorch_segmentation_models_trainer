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
import warnings

import hydra
import numpy as np
import segmentation_models_pytorch as smp
import torch
from hydra import compose, initialize
from hydra.utils import instantiate
from parameterized import parameterized
from pytorch_segmentation_models_trainer.custom_models import models

input_model_list = [
    (models.DeepLab101, {}, torch.Size([2, 128, 256, 256])),
    (models.DeepLab50, {}, torch.Size([2, 128, 256, 256])),
    (models.FCN101, {}, torch.Size([2, 256, 256, 256])),
    (models.FCN50, {}, torch.Size([2, 256, 256, 256])),
    (models.UNetResNet, {}, torch.Size([2, 32, 256, 256])),
    (models.HRNetOCRW48, {}, torch.Size([2, 1, 64, 64])),
    (models.HRNetOCRW48, {"pretrained": "cityscapes"}, torch.Size([2, 19, 64, 64])),
    (models.HRNetOCRW48, {"pretrained": "lip"}, torch.Size([2, 20, 64, 64])),
    (models.HRNetOCRW48, {"pretrained": "pascal"}, torch.Size([2, 59, 64, 64])),
]


class Test_CustomModels(unittest.TestCase):
    def setUp(self) -> None:
        warnings.simplefilter("ignore", category=ImportWarning)
        warnings.simplefilter("ignore", category=DeprecationWarning)
        warnings.simplefilter("ignore", category=FutureWarning)
        warnings.simplefilter("ignore", category=UserWarning)

    @parameterized.expand(input_model_list)
    def test_create_inference_from_model(
        self, input_model, model_args, expected_output_shape
    ) -> None:
        model = input_model(**model_args)
        sample = torch.ones([2, 3, 256, 256])
        with torch.no_grad():
            out = model(sample)
        self.assertEqual(out.shape, expected_output_shape)

    def test_object_detection_model(self) -> None:
        with initialize(config_path="./test_configs"):
            cfg = compose(config_name="object_detection_model.yaml")
            model = hydra.utils.instantiate(cfg, _recursive_=False)
        sample = torch.ones([2, 3, 256, 256])
        model.eval()
        with torch.no_grad():
            out = model(sample)
        self.assertEqual(len(out), 2)

    def test_instance_segmentation_model(self) -> None:
        with initialize(config_path="./test_configs"):
            cfg = compose(config_name="instance_segmentation_model.yaml")
            model = hydra.utils.instantiate(cfg, _recursive_=False)
        sample = torch.ones([2, 3, 256, 256])
        model.eval()
        with torch.no_grad():
            out = model(sample)
        self.assertEqual(len(out), 2)
