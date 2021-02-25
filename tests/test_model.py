# -*- coding: utf-8 -*-
"""
/***************************************************************************
 segmentation_models_trainer
                              -------------------
        begin                : 2020-09-14
        git sha              : $Format:%H$
        copyright            : (C) 2020 by Philipe Borba - Cartographic Engineer @ Brazilian Army
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
import json
from segmentation_models_trainer.model_builder.segmentation_model import SegmentationModel
import segmentation_models as sm
import numpy as np


class Test_TestSegmentationModel(unittest.TestCase):
    
    model = SegmentationModel(
        description='test case',
        backbone='resnet18',
        architecture='Unet'
    )
    json_dict = json.loads('{"description": "test case", "backbone": "resnet18", "architecture": "Unet", "activation": "sigmoid", "use_imagenet_weights": true, "input_bands": 3}')

    def test_create_instance(self):
        """[summary]
        Tests instance creation
        """          
        self.assertEqual(
            self.model.description, 'test case'
        )
        self.assertEqual(
            self.model.backbone, 'resnet18'
        )
        self.assertEqual(
            self.model.architecture, 'Unet'
        )
    
    def test_export_instance(self):
        self.assertEqual(
            self.model.to_dict(),
            self.json_dict
        )
    
    def test_import_instance(self):
        new_model = SegmentationModel.from_dict(
            self.json_dict
        )
        self.assertEqual(
            self.model,
            new_model
        )
    
    def test_get_model(self):
        input_shape = (256, 256, 3)
        x = np.ones((1, *input_shape))
        sm_model = sm.Unet(
            'resnet18',
            encoder_weights='imagenet',
            encoder_freeze=False
        )
        self.assertEqual(
            self.model.get_model(
                1,
                False,
                input_shape=input_shape
            ).predict(x).shape[:-1],
            sm_model.predict(x).shape[:-1]
        )
    
    def test_invalid_values(self):
        with self.assertRaises(ValueError):
            SegmentationModel(
                description='test case',
                backbone='lalala',
                architecture='Unet'
            )

        with self.assertRaises(ValueError):
            SegmentationModel(
                description='test case',
                backbone='resnet18',
                architecture='lalala'
            )
