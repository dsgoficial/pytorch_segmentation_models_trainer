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

from pytorch_segmentation_models_trainer.custom_callbacks.training_callbacks import (
    FinalMetricsCallback,
)
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

    @patch("pytorch_segmentation_models_trainer.train.Trainer")
    @patch.object(Model, "setup")
    def test_final_metrics_callback_added_by_default(
        self, mock_setup, MockTrainer
    ) -> None:
        """FinalMetricsCallback must be injected automatically when no flag is set."""
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
            train(cfg)

        call_kwargs = MockTrainer.call_args[1]
        callbacks = call_kwargs.get("callbacks", [])
        self.assertTrue(
            any(isinstance(cb, FinalMetricsCallback) for cb in callbacks),
            "FinalMetricsCallback should be present by default",
        )

    @patch("pytorch_segmentation_models_trainer.train.Trainer")
    @patch.object(Model, "setup")
    def test_final_metrics_callback_suppressed_when_flag_false(
        self, mock_setup, MockTrainer
    ) -> None:
        """FinalMetricsCallback must NOT be added when add_final_metrics_callback=false."""
        mock_trainer_instance = MagicMock(spec=pl.Trainer)
        MockTrainer.return_value = mock_trainer_instance

        with initialize(config_path="./test_configs"):
            cfg = compose(
                config_name="experiment.yaml",
                overrides=[
                    "train_dataset.input_csv_path=" + self.csv_ds_file,
                    "val_dataset.input_csv_path=" + self.csv_ds_file,
                    "++pl_trainer.fast_dev_run=true",
                    "++add_final_metrics_callback=false",
                ],
            )
            train(cfg)

        call_kwargs = MockTrainer.call_args[1]
        callbacks = call_kwargs.get("callbacks", [])
        self.assertFalse(
            any(isinstance(cb, FinalMetricsCallback) for cb in callbacks),
            "FinalMetricsCallback should be absent when flag is false",
        )

    @patch("pytorch_segmentation_models_trainer.train.Trainer")
    @patch.object(Model, "setup")
    def test_final_metrics_callback_not_duplicated(
        self, mock_setup, MockTrainer
    ) -> None:
        """When config already declares FinalMetricsCallback, only one instance must exist."""
        mock_trainer_instance = MagicMock(spec=pl.Trainer)
        MockTrainer.return_value = mock_trainer_instance

        with initialize(config_path="./test_configs"):
            cfg = compose(
                config_name="experiment_with_final_metrics_callback.yaml",
                overrides=[
                    "train_dataset.input_csv_path=" + self.csv_ds_file,
                    "val_dataset.input_csv_path=" + self.csv_ds_file,
                    "++pl_trainer.fast_dev_run=true",
                ],
            )
            train(cfg)

        call_kwargs = MockTrainer.call_args[1]
        callbacks = call_kwargs.get("callbacks", [])
        count = sum(1 for cb in callbacks if isinstance(cb, FinalMetricsCallback))
        self.assertEqual(count, 1, "FinalMetricsCallback must not be duplicated")
