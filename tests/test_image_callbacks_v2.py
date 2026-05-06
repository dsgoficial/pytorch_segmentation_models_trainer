# -*- coding: utf-8 -*-
import unittest
from unittest.mock import MagicMock, patch
import numpy as np
import torch
import os
import shutil
from pytorch_segmentation_models_trainer.custom_callbacks.image_callbacks import (
    ImageSegmentationResultCallback,
    EnhancedImageSegmentationResultCallback,
    ObjectDetectionResultCallback,
    FrameFieldResultCallback,
    PolygonRNNResultCallback,
    ModPolyMapperResultCallback,
)


class TestImageCallbacksV2(unittest.TestCase):
    def setUp(self):
        self.test_dir = "/tmp/test_image_callbacks"
        os.makedirs(self.test_dir, exist_ok=True)

        # Mock Trainer
        self.trainer = MagicMock()
        self.trainer.log_dir = self.test_dir
        self.trainer.current_epoch = 1

        # Mock Logger
        self.logger = MagicMock()
        self.trainer.logger = self.logger

        # Mock PL Module
        self.pl_module = MagicMock()
        self.pl_module.device = "cpu"

        # Mock Dataset and Dataloader
        self.dataset = MagicMock()
        self.dataset.__getitem__.side_effect = lambda i: {
            "image": torch.ones(3, 128, 128),
            "mask": torch.zeros(1, 128, 128),
        }
        self.dataset.get_path.return_value = "test_image.tif"
        self.dataset.grid_size = 28

        self.dataloader = MagicMock()
        self.dataloader.dataset = self.dataset
        self.dataloader.batch_size = 2
        # Default iter for segmentation
        self.dataloader.__iter__.return_value = iter(
            [
                {
                    "image": torch.ones(1, 3, 128, 128),
                    "mask": torch.zeros(1, 1, 128, 128),
                    "path": ["test_image.tif"],
                }
            ]
        )

        self.pl_module.val_dataloader.return_value = self.dataloader
        self.pl_module.return_value = torch.zeros(1, 1, 128, 128)

    def tearDown(self):
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    def test_image_segmentation_callback_init(self):
        callback = ImageSegmentationResultCallback(
            n_samples=5, output_path=self.test_dir
        )
        self.assertEqual(callback.n_samples, 5)

    @patch(
        "pytorch_segmentation_models_trainer.custom_callbacks.image_callbacks.generate_visualization"
    )
    def test_on_validation_epoch_end(self, mock_vis):
        mock_vis.return_value = (MagicMock(), MagicMock())
        callback = ImageSegmentationResultCallback(n_samples=1)
        callback.on_sanity_check_end(self.trainer, self.pl_module)
        with patch.object(callback, "save_plot_to_disk", return_value="dummy.png"):
            with patch.object(callback, "log_data_to_tensorboard") as mock_log:
                callback.on_validation_epoch_end(self.trainer, self.pl_module)
                self.assertTrue(mock_log.called)

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
                    "image": torch.ones(1, 3, 128, 128),
                    "gt_polygons_image": torch.zeros(1, 2, 128, 128),
                    "path": ["test.tif"],
                }
            ]
        )
        self.pl_module.return_value = {"seg": torch.zeros(1, 2, 128, 128)}

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
        # Mock albumentations compose to return identity-like
        mock_alb.return_value = lambda image: {"image": torch.ones(3, 128, 128)}

        callback = ModPolyMapperResultCallback(n_samples=1)
        callback.on_sanity_check_end(self.trainer, self.pl_module)

        # Setup PL Module for CombinedLoader/ModPolyMapper
        combined = MagicMock()
        combined.iterables = {"polygon_rnn": self.dataloader}
        self.pl_module.val_dataloader.return_value = combined

        self.dataset.get_n_image_path_dict_list.return_value = {
            "test.tif": {
                "original_image": np.zeros((128, 128, 3)),
                "shapely_polygon_list": [],
            }
        }

        # Mock model forward
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

    def test_enhanced_callback_execution(self):
        callback = EnhancedImageSegmentationResultCallback(
            n_samples=1, verbose=False, num_classes=2, max_workers=1
        )
        callback.on_sanity_check_end(self.trainer, self.pl_module)
        with patch.object(callback, "_wait_and_log_to_tensorboard") as mock_log:
            with patch(
                "pytorch_segmentation_models_trainer.custom_callbacks.image_callbacks.generate_visualization"
            ) as mock_vis:
                mock_vis.return_value = (MagicMock(), MagicMock())
                callback.on_validation_epoch_end(self.trainer, self.pl_module)
                self.assertTrue(mock_log.called)

    def test_object_detection_callback(self):
        callback = ObjectDetectionResultCallback(n_samples=1)
        callback.on_sanity_check_end(self.trainer, self.pl_module)
        self.pl_module.return_value = [
            {"boxes": torch.tensor([[0, 0, 10, 10]]), "scores": torch.tensor([0.9])}
        ]
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
            self.assertTrue(self.logger.experiment.add_image.called)


if __name__ == "__main__":
    unittest.main()
