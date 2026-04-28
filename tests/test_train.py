# -*- coding: utf-8 -*-
"""
Tests for pytorch_segmentation_models_trainer.train

Refactored to mock Trainer.fit() and Model.setup() so tests don't run
actual training pipelines (I/O, GPU, etc.).
"""

import os
import unittest
from unittest.mock import MagicMock, patch

import pytorch_lightning as pl
from hydra import compose, initialize
from parameterized import parameterized

from pytorch_segmentation_models_trainer.model_loader.model import Model
from pytorch_segmentation_models_trainer.train import train

from tests.utils import CustomTestCase

config_name_list = [
    "experiment.yaml",
    "experiment_warmup.yaml",
    "experiment_warmup_and_img_callback.yaml",
]


class Test_Train(CustomTestCase):
    @parameterized.expand(config_name_list)
    @patch("pytorch_segmentation_models_trainer.train.Trainer")
    @patch.object(Model, "setup")
    def test_run_train_from_object(
        self, config_name: str, mock_setup, MockTrainer
    ) -> None:
        mock_trainer_instance = MagicMock(spec=pl.Trainer)
        MockTrainer.return_value = mock_trainer_instance

        with initialize(config_path="./test_configs"):
            cfg = compose(
                config_name=config_name,
                overrides=[
                    "train_dataset.input_csv_path=" + self.csv_ds_file,
                    "val_dataset.input_csv_path=" + self.csv_ds_file,
                    "++pl_trainer.fast_dev_run=true",
                ],
            )
            result = train(cfg)

        mock_trainer_instance.fit.assert_called_once()
        self.assertIs(result, mock_trainer_instance)

    @patch("pytorch_segmentation_models_trainer.train.Trainer")
    @patch.object(Model, "setup")
    def test_trainer_test_called_when_test_dataset_present(
        self, mock_setup, MockTrainer
    ) -> None:
        """trainer.test() must be called exactly once when test_dataset is in config."""
        mock_trainer_instance = MagicMock(spec=pl.Trainer)
        MockTrainer.return_value = mock_trainer_instance

        with initialize(config_path="./test_configs"):
            cfg = compose(
                config_name="experiment_with_test.yaml",
                overrides=[
                    "train_dataset.input_csv_path=" + self.csv_ds_file,
                    "val_dataset.input_csv_path=" + self.csv_ds_file,
                    "test_dataset.input_csv_path=" + self.csv_ds_file,
                ],
            )
            result = train(cfg)

        mock_trainer_instance.fit.assert_called_once()
        mock_trainer_instance.test.assert_called_once()
        self.assertIs(result, mock_trainer_instance)

    @patch("pytorch_segmentation_models_trainer.train.Trainer")
    @patch.object(Model, "setup")
    def test_trainer_test_not_called_when_test_dataset_absent(
        self, mock_setup, MockTrainer
    ) -> None:
        """trainer.test() must NOT be called when test_dataset is absent from config."""
        mock_trainer_instance = MagicMock(spec=pl.Trainer)
        MockTrainer.return_value = mock_trainer_instance

        with initialize(config_path="./test_configs"):
            cfg = compose(
                config_name="experiment.yaml",
                overrides=[
                    "train_dataset.input_csv_path=" + self.csv_ds_file,
                    "val_dataset.input_csv_path=" + self.csv_ds_file,
                    "++pl_trainer.fast_dev_run=true",
                ],
            )
            result = train(cfg)

        mock_trainer_instance.fit.assert_called_once()
        mock_trainer_instance.test.assert_not_called()
