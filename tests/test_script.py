# -*- coding: utf-8 -*-
"""
/***************************************************************************
 segmentation_models_trainer
                              -------------------
        begin                : 2021-03-04
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
from typing import List

import pytorch_lightning as pl
from parameterized import parameterized
from hydra import compose, initialize
from pytorch_segmentation_models_trainer.train import train

from tests.utils import CustomTestCase

config_path_list = [
    ("./tests/test_configs"),
    (str(os.path.abspath(os.path.join(os.path.dirname(__file__), "test_configs")))),
]


class Test_Script(CustomTestCase):
    def test_run_train_from_python_script(self) -> None:
        script_path = os.path.join(
            os.path.dirname(__file__),
            "..",
            "pytorch_segmentation_models_trainer",
            "train.py",
        )
        return_from_process = subprocess.run(
            [
                "python3",
                script_path,
                "--config-path",
                "../tests/test_configs",
                "--config-name",
                "experiment",
                "train_dataset.input_csv_path=" + self.csv_ds_file,
                "val_dataset.input_csv_path=" + self.csv_ds_file,
            ],
            check=True,
        )
        self.assertEqual(return_from_process.returncode, 0)

    @parameterized.expand(config_path_list)
    def test_run_train_from_command_line(self, config_path: str) -> None:
        return_from_process = subprocess.run(
            [
                "pytorch-smt",
                "--config-dir",
                config_path,
                "--config-name",
                "experiment",
                "+mode=train",
                "train_dataset.input_csv_path=" + self.csv_ds_file,
                "val_dataset.input_csv_path=" + self.csv_ds_file,
            ],
            check=True,
        )
        self.assertEqual(return_from_process.returncode, 0)

    @parameterized.expand(config_path_list)
    def test_run_validate_config_from_command_line(self, config_path: str) -> None:
        return_from_process = subprocess.run(
            [
                "pytorch-smt",
                "--config-dir",
                config_path,
                "--config-name",
                "experiment",
                "+mode=validate-config",
            ],
            check=True,
        )
        self.assertEqual(return_from_process.returncode, 0)

    @parameterized.expand(config_path_list)
    def test_run_validate_config_from_python_script(self, config_path: str) -> None:
        script_path = os.path.join(
            os.path.dirname(__file__),
            "..",
            "pytorch_segmentation_models_trainer",
            "config_utils.py",
        )
        return_from_process = subprocess.run(
            [
                "python3",
                script_path,
                "--config-dir",
                config_path,
                "--config-name",
                "experiment",
                "+mode=validate-config",
            ],
            check=True,
        )
        self.assertEqual(return_from_process.returncode, 0)
