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
import os
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

from tests.utils import CustomTestCase

input_model_list = [
    (smp.Unet,),
    #(smp.DeepLabV3Plus,)
]

current_dir = os.path.dirname(__file__)
frame_field_root_dir = os.path.join(
    current_dir, 'testing_data', 'data', 'frame_field_data')

class Test_TestFrameFieldModel(CustomTestCase):

    def make_inference(self, sample, frame_field_model):
        with torch.no_grad():
            out = frame_field_model(sample)
        self.assertEqual(
            out['seg'].shape,
            torch.Size([sample.shape[0], 3, sample.shape[-2], sample.shape[-1]])
        )
        self.assertEqual(
            out['crossfield'].shape,
            torch.Size([sample.shape[0], 4, sample.shape[-2], sample.shape[-1]])
        )

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
        self.make_inference(
            torch.ones([2, 3, 256, 256]),
            frame_field_model
        )
    
    def test_create_model_from_cfg(self) -> None:
        with initialize(config_path="./test_configs"):
            cfg = compose(
                config_name="frame_field_pl_model.yaml"
            )
            frame_field_model = hydra.utils.instantiate(cfg)
        self.make_inference(
            torch.ones([2, 3, 256, 256]),
            frame_field_model
        )
    
    def test_create_pl_model(self) -> None:
        csv_path = os.path.join(frame_field_root_dir, 'dsg_dataset.csv')
        with initialize(config_path="./test_configs"):
            cfg = compose(
                config_name="experiment_frame_field.yaml",
                overrides=[
                    'train_dataset.input_csv_path='+csv_path,
                    'train_dataset.root_dir='+frame_field_root_dir,
                    'val_dataset.input_csv_path='+csv_path,
                    'val_dataset.root_dir='+frame_field_root_dir,
                ]
            )
            module_path, class_name = cfg.pl_model._target_.rsplit('.', 1)
            module = import_module(module_path)
            frame_field_model = getattr(module, class_name)(cfg)
        self.assertIsInstance(
            frame_field_model,
            FrameFieldSegmentationPLModel
        )

    def test_train_frame_field_model(self) -> None:
        csv_path = os.path.join(frame_field_root_dir, 'dsg_dataset.csv')
        config_path = os.path.join(os.path.abspath(current_dir), 'test_configs')
        with initialize(config_path="./test_configs"):
            cfg = compose(
                config_name="experiment_frame_field.yaml",
                overrides=[
                    'train_dataset.input_csv_path='+csv_path,
                    'train_dataset.root_dir='+frame_field_root_dir,
                    'val_dataset.input_csv_path='+csv_path,
                    'val_dataset.root_dir='+frame_field_root_dir,
                    # 'pl_trainer.gpus=1',
                    # 'device=cuda',
                    # 'optimizer.lr=0.00001',
                    # 'hyperparameters.batch_size=4',
                    # 'hyperparameters.epochs=100'
                ]
            )
            trainer = train(cfg)
    
    def test_train_frame_field_model_with_callback(self) -> None:
        csv_path = os.path.join(frame_field_root_dir, 'dsg_dataset.csv')
        config_path = os.path.join(os.path.abspath(current_dir), 'test_configs')
        with initialize(config_path="./test_configs"):
            cfg = compose(
                config_name="experiment_frame_field_with_callback.yaml",
                overrides=[
                    'train_dataset.input_csv_path='+csv_path,
                    'train_dataset.root_dir='+frame_field_root_dir,
                    'val_dataset.input_csv_path='+csv_path,
                    'val_dataset.root_dir='+frame_field_root_dir,
                    # 'pl_trainer.gpus=1',
                    # 'device=cuda',
                    # 'optimizer.lr=0.00001',
                    # 'hyperparameters.batch_size=4',
                    # 'hyperparameters.epochs=10'
                ]
            )
            trainer = train(cfg)
