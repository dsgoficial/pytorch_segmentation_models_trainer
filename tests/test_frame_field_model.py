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
from hydra.experimental import initialize, compose

class Test_TestFrameFieldModel(unittest.TestCase):

    def test_create_instance(self) -> None:
        return True

    def test_create_instance_from_config(self) -> None:
        return True
        # with initialize(config_path="./test_configs"):
        #     cfg = compose(config_name="model.yaml")
        #     model_obj = hydra.utils.instantiate(cfg)
        #     assert isinstance(model_obj, smp.Unet)

    def test_create_inference_from_model(self) -> None:
        return True

    def test_train_one_epoch(self) -> None:
        return True
