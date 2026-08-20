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
        self.trainer.global_rank = 0
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

    def test_on_validation_batch_end_tuple(self):
        callback = ConfusionMatrixCallback(num_classes=2)
        callback.on_validation_epoch_start(self.trainer, self.pl_module)
        # Batch as tuple
        batch = (torch.zeros(1, 3, 10, 10), torch.zeros(1, 10, 10))
        callback.on_validation_batch_end(self.trainer, self.pl_module, None, batch, 0)
        self.assertIsNotNone(callback.confmat)

    def test_on_validation_batch_end_4d_targets(self):
        callback = ConfusionMatrixCallback(num_classes=2)
        callback.on_validation_epoch_start(self.trainer, self.pl_module)

        # 4D one-hot targets: [B, C, H, W]
        batch = {"image": torch.zeros(1, 3, 10, 10), "mask": torch.zeros(1, 2, 10, 10)}
        callback.on_validation_batch_end(self.trainer, self.pl_module, None, batch, 0)

        # 4D single channel targets: [B, 1, H, W]
        batch_single = {
            "image": torch.zeros(1, 3, 10, 10),
            "mask": torch.zeros(1, 1, 10, 10),
        }
        callback.on_validation_batch_end(
            self.trainer, self.pl_module, None, batch_single, 0
        )

    def test_on_validation_batch_end_confmat_none(self):
        callback = ConfusionMatrixCallback(num_classes=2)
        # self.confmat is None by default
        batch = {"image": torch.zeros(1, 3, 10, 10), "mask": torch.zeros(1, 10, 10)}
        callback.on_validation_batch_end(self.trainer, self.pl_module, None, batch, 0)
        # Should just return without error

    @patch("matplotlib.pyplot.savefig")
    @patch("matplotlib.pyplot.close")
    @patch("seaborn.heatmap")
    def test_on_validation_epoch_end_various_normalization(
        self, mock_heatmap, mock_close, mock_savefig
    ):
        for norm in ["true", "pred", "all", None]:
            callback = ConfusionMatrixCallback(
                num_classes=2, log_every_n_epochs=1, normalize=norm
            )
            callback.on_sanity_check_end(self.trainer, self.pl_module)
            callback.on_validation_epoch_start(self.trainer, self.pl_module)
            callback.confmat.compute = MagicMock(
                return_value=torch.tensor([[50, 10], [5, 35]])
            )

            # Mock logger
            mock_logger = MagicMock()
            self.trainer.logger = mock_logger
            with patch("matplotlib.pyplot.gcf") as mock_gcf:
                mock_fig = MagicMock()
                mock_gcf.return_value = mock_fig
                mock_fig.canvas.buffer_rgba.return_value = np.zeros(
                    (100, 100, 4), dtype=np.uint8
                ).tobytes()
                mock_fig.canvas.get_width_height.return_value = (100, 100)
                callback.on_validation_epoch_end(self.trainer, self.pl_module)

            mock_savefig.assert_called()

    def test_on_validation_epoch_end_skips(self):
        # 1. save_outputs is False
        callback = ConfusionMatrixCallback(num_classes=2)
        callback.on_validation_epoch_end(self.trainer, self.pl_module)

        # 2. log_every_n_epochs mismatch
        callback = ConfusionMatrixCallback(num_classes=2, log_every_n_epochs=5)
        callback.on_sanity_check_end(self.trainer, self.pl_module)
        callback.on_validation_epoch_start(self.trainer, self.pl_module)
        self.trainer.current_epoch = 1
        callback.on_validation_epoch_end(self.trainer, self.pl_module)

    @patch("builtins.print")
    @patch("seaborn.heatmap")
    def test_on_validation_epoch_end_exception(self, mock_heatmap, mock_print):
        callback = ConfusionMatrixCallback(num_classes=2, log_every_n_epochs=1)
        callback.on_sanity_check_end(self.trainer, self.pl_module)
        # Mock confmat
        callback.confmat = MagicMock()
        callback.confmat.compute.side_effect = RuntimeError("Mock compute error")

        callback.on_validation_epoch_end(self.trainer, self.pl_module)

        # Verify that print was called with something containing our error
        called_args = [str(call[0][0]) for call in mock_print.call_args_list]
        self.assertTrue(any("Mock compute error" in arg for arg in called_args))

    def test_classification_report_on_validation_batch_end_variants(self):
        callback = ClassificationReportCallback(num_classes=2)
        # Dict batch
        batch_dict = {
            "image": torch.zeros(1, 3, 10, 10),
            "mask": torch.zeros(1, 10, 10),
        }
        callback.on_validation_batch_end(
            self.trainer, self.pl_module, None, batch_dict, 0
        )

        # 4D one-hot targets in classification report
        batch_4d = (torch.zeros(1, 3, 10, 10), torch.zeros(1, 2, 10, 10))
        callback.on_validation_batch_end(
            self.trainer, self.pl_module, None, batch_4d, 0
        )

    @patch("builtins.print")
    def test_classification_report_on_validation_epoch_end_success(self, mock_print):
        callback = ClassificationReportCallback(num_classes=2, log_every_n_epochs=1)
        callback.on_sanity_check_end(self.trainer, self.pl_module)

        callback.val_predictions = [0, 1, 0, 1]
        callback.val_targets = [0, 1, 1, 0]

        callback.on_validation_epoch_end(self.trainer, self.pl_module)

        # Check if report file was created
        report_dir = os.path.join(self.tmp_dir, "classification_reports")
        self.assertTrue(os.path.exists(report_dir))
        self.assertGreater(len(os.listdir(report_dir)), 0)
        # Should have printed something
        mock_print.assert_called()

    @patch("builtins.print")
    def test_classification_report_on_validation_epoch_end_edge_cases(self, mock_print):
        # 1. save_outputs is False
        callback = ClassificationReportCallback(num_classes=2)
        callback.on_validation_epoch_end(self.trainer, self.pl_module)
        self.assertEqual(len(callback.val_predictions), 0)

        # 2. Length mismatch
        callback = ClassificationReportCallback(num_classes=2)
        callback.on_sanity_check_end(self.trainer, self.pl_module)
        callback.val_predictions = [0, 1]
        callback.val_targets = [0, 1, 0]
        callback.on_validation_epoch_end(self.trainer, self.pl_module)
        mock_print.assert_any_call(
            "WARNING: Prediction count (2) != Target count (3). Skipping confusion matrix."
        )

        # 3. Skip epoch
        callback = ClassificationReportCallback(num_classes=2, log_every_n_epochs=5)
        callback.on_sanity_check_end(self.trainer, self.pl_module)
        self.trainer.current_epoch = 1
        callback.val_predictions = [0]
        callback.val_targets = [0]
        callback.on_validation_epoch_end(self.trainer, self.pl_module)
        self.assertEqual(len(callback.val_predictions), 0)

        # 4. Empty data
        callback = ClassificationReportCallback(num_classes=2, log_every_n_epochs=1)
        callback.on_sanity_check_end(self.trainer, self.pl_module)
        callback.val_predictions = []
        callback.val_targets = []
        callback.on_validation_epoch_end(self.trainer, self.pl_module)
        # Should just return

    @patch("builtins.print")
    def test_classification_report_exception(self, mock_print):
        callback = ClassificationReportCallback(num_classes=2, log_every_n_epochs=1)
        callback.on_sanity_check_end(self.trainer, self.pl_module)
        callback.val_predictions = [0]
        callback.val_targets = [0]

        with patch(
            "sklearn.metrics.classification_report",
            side_effect=RuntimeError("Mock error"),
        ):
            callback.on_validation_epoch_end(self.trainer, self.pl_module)
            mock_print.assert_any_call(
                "Erro ao gerar relatório de classificação: Mock error"
            )


if __name__ == "__main__":
    unittest.main()
