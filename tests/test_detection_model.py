# -*- coding: utf-8 -*-
"""
/***************************************************************************
 segmentation_models_trainer
                              -------------------
        begin                : 2021-08-16
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

import pytest
from hydra import compose, initialize
from parameterized import parameterized
from pytorch_segmentation_models_trainer.train import train

from tests.utils import CustomTestCase

current_dir = os.path.dirname(__file__)
detection_root_dir = os.path.join(
    current_dir, "testing_data", "data", "detection_data", "geo"
)


class Test_DetectionModel(CustomTestCase):
    @pytest.mark.skip(
        reason="Known segfault and high resource usage. Needs refactoring with mocks."
    )
    @parameterized.expand(
        [
            ("experiment_object_detection.yaml",),
            ("experiment_object_detection_with_callback.yaml",),
        ]
    )
    def test_train_object_detection_model(self, config_name) -> None:
        csv_path = os.path.join(detection_root_dir, "dsg_dataset.csv")
        with initialize(config_path="./test_configs"):
            cfg = compose(
                config_name=config_name,
                overrides=[
                    "train_dataset.input_csv_path=" + csv_path,
                    "train_dataset.root_dir=" + detection_root_dir,
                    "val_dataset.input_csv_path=" + csv_path,
                    "val_dataset.root_dir=" + detection_root_dir,
                ],
            )
            trainer = train(cfg)

    @pytest.mark.skip(
        reason="Known segfault and high resource usage. Needs refactoring with mocks."
    )
    def test_train_instance_segmentation_model(self) -> None:
        csv_path = os.path.join(detection_root_dir, "dsg_dataset.csv")
        with initialize(config_path="./test_configs"):
            cfg = compose(
                config_name="experiment_instance_segmentation.yaml",
                overrides=[
                    "train_dataset.input_csv_path=" + csv_path,
                    "train_dataset.root_dir=" + detection_root_dir,
                    "+train_dataset.return_mask=True",
                    "val_dataset.input_csv_path=" + csv_path,
                    "val_dataset.root_dir=" + detection_root_dir,
                    "+val_dataset.return_mask=True",
                    "++pl_trainer.fast_dev_run=true",
                ],
            )
            trainer = train(cfg)
