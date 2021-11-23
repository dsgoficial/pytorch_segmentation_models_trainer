# -*- coding: utf-8 -*-
"""
/***************************************************************************
 segmentation_models_trainer
                              -------------------
        begin                : 2021-08-19
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

import pytorch_lightning as pl
from hydra import compose, initialize
from parameterized import parameterized
from pytorch_segmentation_models_trainer.train import train

from tests.utils import CustomTestCase

optimizer_name_list = ["Adam", "AdamW", "RAdam", "PlainRAdam", "SGD"]


class Test_Optimizers(CustomTestCase):
    @parameterized.expand(optimizer_name_list)
    def test_run_train_using_optimizer(self, optimizer_class_name: str) -> None:
        with initialize(config_path="./test_configs"):
            cfg = compose(
                config_name="experiment_optimizer_gradient_centralization.yaml",
                overrides=[
                    "train_dataset.input_csv_path=" + self.csv_ds_file,
                    "val_dataset.input_csv_path=" + self.csv_ds_file,
                    f"optimizer._target_=pytorch_segmentation_models_trainer.optimizers.gradient_centralization.{optimizer_class_name}",
                    "+pl_trainer.fast_dev_run=true",
                ],
            )
            train_obj = train(cfg)
            assert isinstance(train_obj, pl.Trainer)
