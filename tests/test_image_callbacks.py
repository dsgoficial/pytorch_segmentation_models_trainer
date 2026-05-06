# -*- coding: utf-8 -*-
"""
Unit tests for image_callbacks.py
"""

import unittest
from unittest.mock import MagicMock, patch
import numpy as np
import torch
import os
import shutil
import tempfile
from pathlib import Path

from pytorch_segmentation_models_trainer.custom_callbacks.image_callbacks import (
    ImageSegmentationResultCallback,
    EnhancedImageSegmentationResultCallback,
)


class TestImageCallbacks(unittest.TestCase):
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

        # Mock Dataset and Dataloader
        self.dataset = MagicMock()
        self.dataset.get_path.side_effect = lambda i: f"image_{i}.tif"
        self.dataset.__getitem__.return_value = {
            "image": torch.zeros(3, 10, 10),
            "mask": torch.zeros(1, 10, 10),
        }

        self.dataloader = MagicMock()
        self.dataloader.dataset = self.dataset
        self.dataloader.batch_size = 1
        self.dataloader.__iter__.return_value = iter(
            [
                {
                    "image": torch.zeros(1, 3, 10, 10),
                    "mask": torch.zeros(1, 1, 10, 10),
                    "path": ["image_0.tif"],
                }
            ]
        )

        self.pl_module.val_dataloader.return_value = self.dataloader
        self.pl_module.side_effect = lambda x: torch.zeros(
            1, 1, 10, 10
        )  # Mock forward call

    def tearDown(self):
        shutil.rmtree(self.tmp_dir)

    def test_image_segmentation_result_callback_init(self):
        callback = ImageSegmentationResultCallback(n_samples=2)
        self.assertEqual(callback.n_samples, 2)
        self.assertFalse(callback.save_outputs)

    def test_on_sanity_check_end(self):
        callback = ImageSegmentationResultCallback()
        callback.on_sanity_check_end(self.trainer, self.pl_module)
        self.assertTrue(callback.save_outputs)
        self.assertTrue(os.path.exists(callback.output_path))

    @patch(
        "pytorch_segmentation_models_trainer.custom_callbacks.image_callbacks.generate_visualization"
    )
    def test_on_validation_epoch_end(self, mock_vis):
        # Mock visualization output
        mock_fig = MagicMock()
        mock_vis.return_value = (MagicMock(), mock_fig)

        callback = ImageSegmentationResultCallback(n_samples=1)
        callback.on_sanity_check_end(self.trainer, self.pl_module)

        with patch.object(callback, "save_plot_to_disk", return_value="dummy.png"):
            with patch.object(callback, "log_data_to_tensorboard"):
                callback.on_validation_epoch_end(self.trainer, self.pl_module)

        mock_vis.assert_called()

    def test_enhanced_callback_init(self):
        callback = EnhancedImageSegmentationResultCallback(
            num_classes=2, class_names=["bg", "fg"]
        )
        self.assertEqual(len(callback.class_colors), 2)
        self.assertEqual(callback.class_names, ["bg", "fg"])

    def test_enhanced_prepare_image_to_plot(self):
        callback = EnhancedImageSegmentationResultCallback(num_classes=2)
        image = torch.zeros(3, 10, 10)
        prepared = callback.prepare_image_to_plot(image)
        self.assertEqual(prepared.shape, (10, 10, 3))
        self.assertTrue(prepared.min() >= 0)
        self.assertTrue(prepared.max() <= 1)

    def test_enhanced_prepare_mask_to_plot(self):
        callback = EnhancedImageSegmentationResultCallback(num_classes=2)
        mask = torch.zeros(1, 10, 10)
        prepared = callback.prepare_mask_to_plot(mask)
        self.assertEqual(prepared.shape, (10, 10))
        self.assertEqual(prepared.dtype, np.uint8)


if __name__ == "__main__":
    unittest.main()
