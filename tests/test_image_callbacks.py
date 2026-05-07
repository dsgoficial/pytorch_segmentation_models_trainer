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
    ObjectDetectionResultCallback,
    FrameFieldResultCallback,
    PolygonRNNResultCallback,
    ModPolyMapperResultCallback,
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
        self.dataset.__getitem__.side_effect = lambda i: {
            "image": torch.zeros(3, 10, 10),
            "mask": torch.zeros(1, 10, 10),
        }
        self.dataset.grid_size = 28

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
        self.pl_module.side_effect = None  # Clear side effect

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

        with patch.object(
            self.pl_module, "forward", return_value=torch.zeros(1, 1, 10, 10)
        ):
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

    @patch(
        "pytorch_segmentation_models_trainer.custom_callbacks.image_callbacks.generate_visualization"
    )
    def test_enhanced_callback_execution(self, mock_vis):
        mock_vis.return_value = (np.zeros((3, 10, 10)), MagicMock())
        callback = EnhancedImageSegmentationResultCallback(
            n_samples=1, verbose=False, num_classes=2, max_workers=1
        )
        callback.on_sanity_check_end(self.trainer, self.pl_module)

        with patch.object(
            self.pl_module, "forward", return_value=torch.zeros(1, 1, 10, 10)
        ):
            with patch.object(callback, "_wait_and_log_to_tensorboard") as mock_log:
                callback.on_validation_epoch_end(self.trainer, self.pl_module)
                self.assertTrue(mock_log.called)

    def test_object_detection_callback(self):
        callback = ObjectDetectionResultCallback(n_samples=1)
        callback.on_sanity_check_end(self.trainer, self.pl_module)

        # Fixed outputs structure for detection: B list of dicts
        mock_outputs = [
            {"boxes": torch.tensor([[0, 0, 10, 10]]), "scores": torch.tensor([0.9])}
        ]

        with patch.object(self.pl_module, "forward", return_value=mock_outputs):
            self.dataloader.__iter__.return_value = iter(
                [
                    (
                        torch.ones(1, 3, 128, 128),
                        [{"boxes": torch.tensor([[0, 0, 10, 10]])}],
                        torch.tensor([0]),
                    )
                ]
            )
            with patch(
                "pytorch_segmentation_models_trainer.custom_callbacks.image_callbacks.visualize_image_with_bboxes"
            ) as mock_vis:
                mock_vis.return_value = [torch.ones(3, 128, 128)]
                callback.on_validation_epoch_end(self.trainer, self.pl_module)
                self.assertTrue(self.trainer.logger.experiment.add_image.called)

    @patch(
        "pytorch_segmentation_models_trainer.custom_callbacks.image_callbacks.generate_visualization"
    )
    def test_frame_field_callback(self, mock_vis):
        mock_vis.return_value = (MagicMock(), MagicMock())
        callback = FrameFieldResultCallback(n_samples=1)
        callback.on_sanity_check_end(self.trainer, self.pl_module)

        self.dataloader.__iter__.return_value = iter(
            [
                {
                    "image": torch.ones(1, 3, 10, 10),
                    "gt_polygons_image": torch.zeros(1, 2, 10, 10),
                    "path": ["test.tif"],
                }
            ]
        )

        # Fixed pl_module mock to return a dict with "seg" key (Tensor)
        with patch.object(
            self.pl_module, "forward", return_value={"seg": torch.zeros(1, 2, 10, 10)}
        ):
            with patch.object(callback, "save_plot_to_disk", return_value="dummy.png"):
                with patch.object(callback, "log_data_to_tensorboard") as mock_log:
                    callback.on_validation_epoch_end(self.trainer, self.pl_module)
                    self.assertTrue(mock_log.called)

    @patch(
        "pytorch_segmentation_models_trainer.utils.polygonrnn_utils.get_vertex_list_from_batch_tensors"
    )
    def test_polygon_rnn_callback(self, mock_utils):
        mock_utils.return_value = [np.array([[0, 0], [1, 1]])]
        callback = PolygonRNNResultCallback(n_samples=1)
        callback.on_sanity_check_end(self.trainer, self.pl_module)

        self.dataset.get_n_image_path_dict_list.return_value = {
            "test.tif": {
                "croped_images": torch.ones(1, 3, 224, 224),
                "shapely_polygon_list": [],
                "scale_h": 1,
                "scale_w": 1,
                "min_col": 0,
                "min_row": 0,
                "original_image": np.zeros((128, 128, 3)),
            }
        }
        self.pl_module.model.test.return_value = torch.zeros(1, 10)
        self.pl_module.val_seq_len = 10

        with patch.object(
            callback, "build_polygon_vis", return_value="dummy.png"
        ) as mock_build:
            callback.on_validation_epoch_end(self.trainer, self.pl_module)
            self.assertTrue(mock_build.called)

    @patch("albumentations.Compose")
    @patch(
        "pytorch_segmentation_models_trainer.utils.polygonrnn_utils.get_vertex_list_from_batch_tensors"
    )
    def test_mod_polymapper_callback(self, mock_utils, mock_alb):
        mock_utils.return_value = [np.array([[0, 0], [1, 1]])]
        mock_alb.return_value = lambda image: {"image": torch.ones(3, 10, 10)}

        callback = ModPolyMapperResultCallback(n_samples=1)
        callback.on_sanity_check_end(self.trainer, self.pl_module)

        combined = MagicMock()
        combined.iterables = {"polygon_rnn": self.dataloader}
        self.pl_module.val_dataloader.return_value = combined

        self.dataset.get_n_image_path_dict_list.return_value = {
            "test.tif": {
                "original_image": np.zeros((10, 10, 3)),
                "shapely_polygon_list": [],
            }
        }

        mock_model = MagicMock()
        self.pl_module.model = mock_model
        mock_model.polygonrnn_model.grid_size = 28
        mock_model.return_value = [
            {
                "polygonrnn_output": torch.zeros(1, 10),
                "scale_h": 1,
                "scale_w": 1,
                "min_row": 0,
                "min_col": 0,
            }
        ]

        with patch.object(
            callback, "build_obj_det_and_polygon_vis", return_value="dummy.png"
        ) as mock_build:
            callback.on_validation_epoch_end(self.trainer, self.pl_module)
            self.assertTrue(mock_build.called)


if __name__ == "__main__":
    unittest.main()
