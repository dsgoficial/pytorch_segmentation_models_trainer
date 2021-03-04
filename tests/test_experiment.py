# -*- coding: utf-8 -*-
"""
/***************************************************************************
 segmentation_models_trainer
                              -------------------
        begin                : 2021-03-01
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
from hydra.experimental import compose, initialize
from pytorch_segmentation_models_trainer.train import train

from tests.utils import CustomTestCase


class Test_TestExperiment(CustomTestCase):
    def test_run_experiment_from_object(self) -> None:
        with initialize(config_path="./test_configs"):
            cfg = compose(
                config_name="experiment.yaml",
                overrides=[
                    'train_dataset.input_csv_path='+self.csv_ds_file,
                    'val_dataset.input_csv_path='+self.csv_ds_file,
                ]
            )
            train_obj = train(cfg)
            assert isinstance(train_obj, pl.Trainer)
