# -*- coding: utf-8 -*-
"""
Unit tests for metrics_callbacks.py
"""

import unittest
from unittest.mock import MagicMock, patch
import numpy as np
import torch
import os
import shutil
import tempfile
from pathlib import Path

from pytorch_segmentation_models_trainer.custom_callbacks.metrics_callbacks import (
    ConfusionMatrixCallback,
    ClassificationReportCallback,
)


class TestMetricsCallbacks(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()

        # Mock Trainer
        self.trainer = MagicMock()
        self.trainer.log_dir = self.tmp_dir
        self.trainer.current_epoch = 0
        self.trainer.logger = MagicMock()

        # Mock Module
        self.pl_module = MagicMock()
        self.pl_module.device = "cpu"
        self.pl_module.side_effect = lambda x: torch.zeros(
            x.shape[0], 2, x.shape[2], x.shape[3]
        )  # 2 classes

    def tearDown(self):
        shutil.rmtree(self.tmp_dir)

    def test_confusion_matrix_callback_init(self):
        callback = ConfusionMatrixCallback(num_classes=2)
        self.assertEqual(callback.num_classes, 2)
        self.assertEqual(len(callback.class_names), 2)

    def test_on_sanity_check_end(self):
        callback = ConfusionMatrixCallback(num_classes=2)
        callback.on_sanity_check_end(self.trainer, self.pl_module)
        self.assertTrue(callback.save_outputs)
        self.assertTrue(os.path.exists(callback.output_path))

    def test_on_validation_batch_end(self):
        callback = ConfusionMatrixCallback(num_classes=2)
        callback.on_validation_epoch_start(self.trainer, self.pl_module)

        batch = {"image": torch.zeros(1, 3, 10, 10), "mask": torch.zeros(1, 1, 10, 10)}

        callback.on_validation_batch_end(self.trainer, self.pl_module, None, batch, 0)
        self.assertIsNotNone(callback.confmat)

    @patch("matplotlib.pyplot.savefig")
    @patch("matplotlib.pyplot.close")
    def test_on_validation_epoch_end(self, mock_close, mock_savefig):
        callback = ConfusionMatrixCallback(num_classes=2, log_every_n_epochs=1)
        callback.on_sanity_check_end(self.trainer, self.pl_module)
        callback.on_validation_epoch_start(self.trainer, self.pl_module)

        # Mock confmat compute
        callback.confmat.compute = MagicMock(
            return_value=torch.tensor([[50, 10], [5, 35]])
        )

        callback.on_validation_epoch_end(self.trainer, self.pl_module)
        mock_savefig.assert_called()

    def test_classification_report_callback_init(self):
        callback = ClassificationReportCallback(num_classes=2)
        self.assertEqual(callback.num_classes, 2)

    def test_classification_report_on_validation_batch_end(self):
        callback = ClassificationReportCallback(num_classes=2)
        batch = (torch.zeros(1, 3, 10, 10), torch.zeros(1, 1, 10, 10))
        callback.on_validation_batch_end(self.trainer, self.pl_module, None, batch, 0)
        self.assertGreater(len(callback.val_predictions), 0)
        self.assertGreater(len(callback.val_targets), 0)

    @patch("builtins.print")
    def test_classification_report_on_validation_epoch_end(self, mock_print):
        callback = ClassificationReportCallback(num_classes=2, log_every_n_epochs=1)
        callback.on_sanity_check_end(self.trainer, self.pl_module)

        callback.val_predictions = [0, 1, 0, 1]
        callback.val_targets = [0, 1, 1, 0]

        callback.on_validation_epoch_end(self.trainer, self.pl_module)

        # Check if report file was created
        report_dir = os.path.join(self.tmp_dir, "classification_reports")
        self.assertTrue(os.path.exists(report_dir))
        self.assertGreater(len(os.listdir(report_dir)), 0)


if __name__ == "__main__":
    unittest.main()
