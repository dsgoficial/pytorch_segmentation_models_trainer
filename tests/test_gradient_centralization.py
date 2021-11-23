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
from hydra import compose, initialize
from parameterized import parameterized
from pytorch_segmentation_models_trainer.model_loader.frame_field_model import (
    FrameFieldModel,
    FrameFieldSegmentationPLModel,
)
from pytorch_segmentation_models_trainer.train import train
from pytorch_segmentation_models_trainer.custom_models import models as pytorch_smt_cm

from tests.utils import CustomTestCase

gradient_centralization_list = [
    (
        [
            "optimizer._target_=pytorch_segmentation_models_trainer.optimizers.gradient_centralization.Adam",
            "+optimizer.use_gc=True",
        ],
    ),
    (
        [
            "optimizer._target_=pytorch_segmentation_models_trainer.optimizers.gradient_centralization.AdamW",
            "+optimizer.use_gc=True",
        ],
    ),
    (
        [
            "optimizer._target_=pytorch_segmentation_models_trainer.optimizers.gradient_centralization.RAdam",
            "+optimizer.use_gc=True",
        ],
    ),
    (
        [
            "optimizer._target_=pytorch_segmentation_models_trainer.optimizers.gradient_centralization.PlainRAdam",
            "+optimizer.use_gc=True",
        ],
    ),
    (
        [
            "optimizer._target_=pytorch_segmentation_models_trainer.optimizers.gradient_centralization.SGD",
            "+optimizer.use_gc=True",
        ],
    ),
]

current_dir = os.path.dirname(__file__)
frame_field_root_dir = os.path.join(
    current_dir, "testing_data", "data", "frame_field_data"
)


class Test_GradientCentralization(CustomTestCase):
    @parameterized.expand(gradient_centralization_list)
    def test_train_with_gradient_centralization(self, overrides_list):
        csv_path = os.path.join(frame_field_root_dir, "dsg_dataset.csv")
        config_path = os.path.join(os.path.abspath(current_dir), "test_configs")
        with initialize(config_path="./test_configs"):
            cfg = compose(
                config_name="experiment_frame_field.yaml",
                overrides=overrides_list
                + [
                    "train_dataset.input_csv_path=" + csv_path,
                    "train_dataset.root_dir=" + frame_field_root_dir,
                    "val_dataset.input_csv_path=" + csv_path,
                    "val_dataset.root_dir=" + frame_field_root_dir,
                    "+pl_trainer.fast_dev_run=true",
                ],
            )
            trainer = train(cfg)
