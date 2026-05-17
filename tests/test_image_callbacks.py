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
from PIL import Image

from pytorch_segmentation_models_trainer.custom_models.variational_autoencoder import (
    VariationalAutoencoderOutput,
)
from pytorch_segmentation_models_trainer.custom_callbacks.image_callbacks import (
    ImageSegmentationResultCallback,
    EnhancedImageSegmentationResultCallback,
    ObjectDetectionResultCallback,
    FrameFieldResultCallback,
    FrameFieldOverlayedResultCallback,
    PolygonRNNResultCallback,
    ModPolyMapperResultCallback,
    AutoencoderResultCallback,
)


class TestImageCallbacks(unittest.TestCase):
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

        # Mock Dataset and Dataloader
        self.dataset = MagicMock()
        self.dataset.get_path.side_effect = lambda i: f"image_{i}.tif"
        self.dataset.__getitem__.side_effect = lambda i: {
            "image": torch.zeros(3, 10, 10),
            "mask": torch.zeros(1, 10, 10),
        }
        self.dataset.__len__ = MagicMock(return_value=10)
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

    def test_autoencoder_result_callback(self):
        # Mock dataset to return what AutoencoderDataset would return
        self.dataset.__getitem__.side_effect = lambda i: {
            "image": torch.zeros(3, 10, 10),
            "target": torch.zeros(3, 10, 10),
        }
        self.dataset.__len__.return_value = 1

        callback = AutoencoderResultCallback(n_samples=1)
        callback.on_sanity_check_end(self.trainer, self.pl_module)

        # Mock pl_module(image) call
        self.pl_module.return_value = torch.zeros(1, 3, 10, 10)

        with patch(
            "pytorch_segmentation_models_trainer.custom_callbacks.image_callbacks.generate_visualization"
        ) as mock_vis:
            mock_fig = MagicMock()
            mock_vis.return_value = (MagicMock(), mock_fig)

            with patch.object(callback, "save_plot_to_disk", return_value="dummy.png"):
                with patch.object(callback, "log_data_to_tensorboard"):
                    callback.on_validation_epoch_end(self.trainer, self.pl_module)

            mock_vis.assert_called_once()
            # Check if input_image and reconstructed_image were passed to generate_visualization
            args, kwargs = mock_vis.call_args
            self.assertIn("input_image", kwargs)
            self.assertIn("reconstructed_image", kwargs)

    def test_autoencoder_result_callback_accepts_vae_output(self):
        self.dataset.__getitem__.side_effect = lambda i: {
            "image": torch.zeros(3, 10, 10),
            "target": torch.zeros(3, 10, 10),
        }
        self.dataset.__len__.return_value = 1

        callback = AutoencoderResultCallback(n_samples=1)
        callback.on_sanity_check_end(self.trainer, self.pl_module)
        reconstruction = torch.zeros(1, 3, 10, 10)
        mu = torch.zeros(1, 2, 2, 2)
        self.pl_module.return_value = VariationalAutoencoderOutput(
            reconstruction=reconstruction,
            mu=mu,
            logvar=mu,
            z=mu,
        )

        with patch(
            "pytorch_segmentation_models_trainer.custom_callbacks.image_callbacks.generate_visualization"
        ) as mock_vis:
            mock_vis.return_value = (MagicMock(), MagicMock())
            with patch.object(callback, "save_plot_to_disk", return_value="dummy.png"):
                with patch.object(callback, "log_data_to_tensorboard"):
                    callback.on_validation_epoch_end(self.trainer, self.pl_module)

        mock_vis.assert_called_once()

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
                "boxes": torch.tensor([[0, 0, 1, 1]]),
                "labels": torch.tensor([1]),
                "scores": torch.tensor([0.9]),
            }
        ]

        with patch.object(
            callback, "build_obj_det_and_polygon_vis", return_value="dummy.png"
        ) as mock_build:
            callback.on_validation_epoch_end(self.trainer, self.pl_module)
            self.assertTrue(mock_build.called)

    def test_image_segmentation_result_callback_logging(self):
        callback = ImageSegmentationResultCallback()
        callback.on_sanity_check_end(self.trainer, self.pl_module)

        # Test real log_data_to_tensorboard
        dummy_img = Image.new("RGB", (10, 10))
        img_path = os.path.join(self.tmp_dir, "dummy.png")
        dummy_img.save(img_path)

        callback.log_data_to_tensorboard(img_path, "test_tag", self.trainer.logger, 0)
        self.trainer.logger.experiment.add_image.assert_called()

    def test_frame_field_result_callback_n_samples_none(self):
        callback = FrameFieldResultCallback(n_samples=None)
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
        with patch(
            "pytorch_segmentation_models_trainer.custom_callbacks.image_callbacks.generate_visualization"
        ) as mock_vis:
            mock_vis.return_value = (MagicMock(), MagicMock())
            with patch.object(
                self.pl_module,
                "forward",
                return_value={"seg": torch.zeros(1, 2, 10, 10)},
            ):
                callback.on_validation_epoch_end(self.trainer, self.pl_module)

    def test_enhanced_callback_verbose_and_summary(self):
        callback = EnhancedImageSegmentationResultCallback(
            num_classes=2, verbose=True, n_samples=1
        )
        callback.on_sanity_check_end(self.trainer, self.pl_module)

        # Test _log
        with patch("builtins.print") as mock_print:
            callback._log("test message")
            mock_print.assert_called()

        # Test summary logging at validation end
        self.pl_module.return_value = torch.zeros(1, 2, 10, 10)
        self.dataloader.__iter__.return_value = iter(
            [
                {
                    "image": torch.zeros(1, 3, 10, 10),
                    "mask": torch.zeros(1, 10, 10),
                    "path": ["test.tif"],
                }
            ]
        )
        callback.on_validation_epoch_end(self.trainer, self.pl_module)

    def test_enhanced_callback_error_handling(self):
        callback = EnhancedImageSegmentationResultCallback(num_classes=2, n_samples=1)
        callback.on_sanity_check_end(self.trainer, self.pl_module)

        # Trigger exception in batch processing
        self.dataloader.__iter__.return_value = iter(
            [{"image": torch.zeros(1, 3, 10, 10), "mask": None}]
        )
        callback.on_validation_epoch_end(self.trainer, self.pl_module)

    def test_enhanced_prepare_image_min_max_scaling(self):
        callback = EnhancedImageSegmentationResultCallback(
            num_classes=2, normalized_input=False
        )
        # Uniform image (max == min)
        img = torch.zeros(3, 10, 10)
        prepared = callback.prepare_image_to_plot(img)
        self.assertEqual(np.max(np.abs(prepared)), 0.0)

        # 1-channel image (repeated)
        img_1ch = torch.zeros(1, 10, 10)
        prepared_1ch = callback.prepare_image_to_plot(img_1ch)
        self.assertEqual(prepared_1ch.shape, (10, 10, 3))

    def test_enhanced_prepare_mask_else_branch(self):
        callback = EnhancedImageSegmentationResultCallback(num_classes=2)
        # 3D mask with channels last
        mask = torch.zeros(10, 10, 2)
        prepared = callback.prepare_mask_to_plot(mask)
        self.assertEqual(prepared.shape, (10, 10))

    def test_polygon_rnn_result_callback_execution(self):
        callback = PolygonRNNResultCallback(n_samples=1)
        callback.on_sanity_check_end(self.trainer, self.pl_module)

        self.dataset.get_n_image_path_dict_list.return_value = {
            "test.tif": {
                "croped_images": torch.ones(1, 3, 224, 224),
                "shapely_polygon_list": [],
                "scale_h": torch.tensor([1.0]),
                "scale_w": torch.tensor([1.0]),
                "min_col": torch.tensor([0]),
                "min_row": torch.tensor([0]),
                "original_image": np.zeros((128, 128, 3)),
            }
        }
        self.pl_module.model.test.return_value = torch.zeros(1, 10)
        self.pl_module.val_seq_len = 10

        with patch(
            "pytorch_segmentation_models_trainer.custom_callbacks.image_callbacks.generate_visualization"
        ) as mock_vis:
            mock_fig = MagicMock()
            mock_fig.get_axes.return_value = [MagicMock(), MagicMock()]  # 2 axes
            mock_vis.return_value = (MagicMock(), mock_fig)
            with patch("matplotlib.pyplot.gcf", return_value=mock_fig):
                with patch.object(
                    callback, "save_plot_to_disk", return_value="dummy.png"
                ):
                    with patch.object(callback, "log_data_to_tensorboard"):
                        callback.on_validation_epoch_end(self.trainer, self.pl_module)

    def test_enhanced_callback_advanced(self):
        # 1. Different colormaps
        callback = EnhancedImageSegmentationResultCallback(
            num_classes=2, colormap="viridis"
        )
        self.assertEqual(len(callback.class_colors), 2)

        # 2. prepare_image_to_plot (2 channels, many channels)
        callback.normalized_input = False
        img_2ch = torch.zeros(2, 10, 10)
        prepared_2ch = callback.prepare_image_to_plot(img_2ch)
        self.assertEqual(prepared_2ch.shape, (10, 10, 3))

        img_5ch = torch.zeros(5, 10, 10)
        prepared_5ch = callback.prepare_image_to_plot(img_5ch)
        self.assertEqual(prepared_5ch.shape, (10, 10, 3))

        # 3. prepare_mask_to_plot (3D one-hot)
        mask_3d = torch.zeros(2, 10, 10)  # 2 classes
        prepared_mask = callback.prepare_mask_to_plot(mask_3d)
        self.assertEqual(prepared_mask.shape, (10, 10))

        # 4. cleanup
        callback.on_train_end(self.trainer, self.pl_module)

    @patch("matplotlib.pyplot.savefig")
    def test_enhanced_callback_batch_formats(self, mock_savefig):
        callback = EnhancedImageSegmentationResultCallback(n_samples=1, num_classes=2)
        callback.on_sanity_check_end(self.trainer, self.pl_module)
        self.trainer.current_epoch = 1

        # Mock pl_module to return expected shape
        self.pl_module.return_value = torch.zeros(1, 2, 10, 10)

        # Test with tuple batch (image, mask)
        self.dataloader.__iter__.return_value = iter(
            [(torch.zeros(1, 3, 10, 10), torch.zeros(1, 1, 10, 10))]
        )
        callback.on_validation_epoch_end(self.trainer, self.pl_module)

        # Test with list batch [image, mask]
        self.dataloader.__iter__.return_value = iter(
            [[torch.zeros(1, 3, 10, 10), torch.zeros(1, 1, 10, 10)]]
        )
        callback.on_validation_epoch_end(self.trainer, self.pl_module)

    def test_enhanced_apply_colormap_ignore_index(self):
        callback = EnhancedImageSegmentationResultCallback(num_classes=2)
        ax = MagicMock()
        mask = np.full((10, 10), 255, dtype=np.uint8)  # All ignored
        callback.apply_colormap_to_axes(ax, [0], [mask])
        ax.imshow.assert_called()
        # Verify it used masked array
        args, kwargs = ax.imshow.call_args
        self.assertTrue(hasattr(args[0], "mask"))

    def test_enhanced_add_colorbar_legend(self):
        callback = EnhancedImageSegmentationResultCallback(
            num_classes=2, class_names=["A", "B"]
        )
        fig = MagicMock()
        axarr = [MagicMock()]
        mask = np.zeros((10, 10))
        callback.add_colorbar_legend(fig, axarr, mask)
        fig.colorbar.assert_called()

    @patch(
        "pytorch_segmentation_models_trainer.custom_callbacks.image_callbacks.get_tensorboard_image_seg_display"
    )
    def test_frame_field_overlayed_callback(self, mock_overlay):
        mock_overlay.return_value = torch.zeros(1, 3, 10, 10)
        callback = FrameFieldOverlayedResultCallback(n_samples=1)
        callback.on_sanity_check_end(self.trainer, self.pl_module)

        # 1. Successful execution
        self.pl_module.return_value = {
            "seg": torch.zeros(1, 2, 10, 10),
            "crossfield": torch.zeros(1, 2, 10, 10),
        }
        callback.on_validation_epoch_end(self.trainer, self.pl_module)
        self.assertTrue(self.trainer.logger.experiment.add_images.called)

        # 2. Missing "seg" in pred
        self.pl_module.return_value = {"crossfield": torch.zeros(1, 2, 10, 10)}
        callback.on_validation_epoch_end(self.trainer, self.pl_module)

    def test_image_segmentation_result_callback_misc(self):
        # 1-channel image needs 1-channel norm params or normalized_input=False
        callback = ImageSegmentationResultCallback(
            n_samples=None, normalized_input=False
        )
        # prepare_image_to_plot (1 channel)
        img = torch.zeros(1, 10, 10)
        prepared = callback.prepare_image_to_plot(img.numpy())
        self.assertEqual(prepared.shape, (10, 10))

        # prepare_mask_to_plot
        mask = torch.ones(1, 1, 10, 10)
        prepared_mask = callback.prepare_mask_to_plot(mask.numpy())
        self.assertEqual(prepared_mask.dtype, np.float64)

    def test_mod_polymapper_vis_logic(self):
        callback = ModPolyMapperResultCallback(n_samples=1)
        callback.on_sanity_check_end(self.trainer, self.pl_module)
        callback.show_label_scores = True

        detection_dict = {
            "boxes": torch.tensor([[0, 0, 10, 10]]),
            "labels": torch.tensor([1]),
            "scores": torch.tensor([0.9]),
            "polygonrnn_output": torch.zeros(1, 10),
            "scale_h": torch.tensor([1.0]),
            "scale_w": torch.tensor([1.0]),
            "min_row": torch.tensor([0]),
            "min_col": torch.tensor([0]),
        }
        prepared_item = {
            "original_image": np.zeros((100, 100, 3)),
            "shapely_polygon_list": [],
        }

        with patch(
            "pytorch_segmentation_models_trainer.custom_callbacks.image_callbacks.generate_visualization"
        ) as mock_vis:
            mock_fig = MagicMock()
            mock_fig.get_axes.return_value = [
                MagicMock(),
                MagicMock(),
                MagicMock(),
            ]  # 3 axes
            mock_vis.return_value = (MagicMock(), mock_fig)
            with patch("matplotlib.pyplot.gcf", return_value=mock_fig):
                with patch(
                    "pytorch_segmentation_models_trainer.custom_callbacks.image_callbacks.generate_bbox_visualization"
                ) as mock_bbox:
                    with patch.object(
                        callback, "save_plot_to_disk", return_value="dummy.png"
                    ):
                        with patch.object(callback, "log_data_to_tensorboard"):
                            callback.build_obj_det_and_polygon_vis(
                                "path.tif",
                                prepared_item["original_image"],
                                detection_dict,
                                [],
                                [],
                                self.trainer,
                            )
                            mock_bbox.assert_called()

    def test_prepare_image_to_plot_clips_to_0_1_after_denorm(self):
        # normalized_input=True + extreme values → denorm can exceed [0,1] → must be clipped
        callback = ImageSegmentationResultCallback(normalized_input=True)
        # Create an image with large positive values (as if model predicted out-of-range)
        # After denorm with ImageNet stats, some channels will exceed 1.0
        extreme_image = np.ones((3, 8, 8), dtype=np.float32) * 3.0
        result = callback.prepare_image_to_plot(extreme_image)
        self.assertLessEqual(result.max(), 1.0)
        self.assertGreaterEqual(result.min(), 0.0)
        self.assertEqual(result.shape, (8, 8, 3))

    def test_prepare_image_to_plot_clips_negative_after_denorm(self):
        callback = ImageSegmentationResultCallback(normalized_input=True)
        extreme_image = np.ones((3, 8, 8), dtype=np.float32) * -3.0
        result = callback.prepare_image_to_plot(extreme_image)
        self.assertGreaterEqual(result.min(), 0.0)
        self.assertLessEqual(result.max(), 1.0)

    def test_autoencoder_callback_no_denorm_when_normalized_input_false(self):
        # Regression: ToFloat datasets (images in [0,1]) must NOT apply ImageNet denorm.
        # Applying x*std+mean to [0,1] values → range compression + warm cast (sepia).
        callback = AutoencoderResultCallback(n_samples=1, normalized_input=False)
        # Image in [0,1] range (ToFloat output)
        image_01 = np.ones((3, 8, 8), dtype=np.float32) * 0.5
        result = callback.prepare_image_to_plot(image_01)
        # Without denorm, all values should remain 0.5
        np.testing.assert_allclose(result, np.full((8, 8, 3), 0.5, dtype=np.float32))

    def test_autoencoder_callback_denorm_sepia_regression(self):
        # When normalized_input=True (wrong for ToFloat data), applying ImageNet denorm
        # to [0,1] values compresses range and shifts channels unequally (sepia effect).
        # With normalized_input=False this must NOT happen.
        callback_wrong = AutoencoderResultCallback(n_samples=1, normalized_input=True)
        callback_correct = AutoencoderResultCallback(
            n_samples=1, normalized_input=False
        )
        image_01 = np.ones((3, 8, 8), dtype=np.float32) * 0.5

        wrong_result = callback_wrong.prepare_image_to_plot(image_01.copy())
        correct_result = callback_correct.prepare_image_to_plot(image_01.copy())

        # With wrong config: denorm shifts R/G/B differently → channels NOT equal
        r_wrong, g_wrong, b_wrong = (
            wrong_result[:, :, 0],
            wrong_result[:, :, 1],
            wrong_result[:, :, 2],
        )
        self.assertFalse(
            np.allclose(r_wrong, b_wrong),
            "Sepia: R and B should differ when denorm wrongly applied",
        )

        # With correct config: no denorm → all channels equal (0.5)
        np.testing.assert_allclose(
            correct_result, np.full((8, 8, 3), 0.5, dtype=np.float32)
        )

    def test_save_plot_to_disk_includes_sample_idx(self):
        """save_plot_to_disk must embed sample_idx in the filename when provided."""
        callback = AutoencoderResultCallback(n_samples=2)
        callback.on_sanity_check_end(self.trainer, self.pl_module)

        fig = MagicMock()
        path = callback.save_plot_to_disk(fig, "tile.tif", 0, sample_idx=3)
        self.assertIn("_idx3", path)

        path_no_idx = callback.save_plot_to_disk(fig, "tile.tif", 0)
        self.assertNotIn("_idx", path_no_idx)

    def test_autoencoder_callback_n_samples_capped_with_warning(self):
        """AutoencoderResultCallback warns and caps when n_samples > len(val_ds)."""
        self.dataset.__getitem__.side_effect = lambda i: {
            "image": torch.zeros(3, 10, 10),
            "target": torch.zeros(3, 10, 10),
        }
        self.dataset.__len__ = MagicMock(return_value=2)

        callback = AutoencoderResultCallback(n_samples=8)
        callback.on_sanity_check_end(self.trainer, self.pl_module)
        self.pl_module.return_value = torch.zeros(1, 3, 10, 10)

        with patch(
            "pytorch_segmentation_models_trainer.custom_callbacks.image_callbacks.generate_visualization"
        ) as mock_vis:
            mock_vis.return_value = (MagicMock(), MagicMock())
            with patch.object(callback, "save_plot_to_disk", return_value="dummy.png"):
                with patch.object(callback, "log_data_to_tensorboard"):
                    callback.on_validation_epoch_end(self.trainer, self.pl_module)

        # Only 2 calls (len(val_ds)=2), not 8
        self.assertEqual(mock_vis.call_count, 2)
        self.trainer.logger.warning.assert_called()

    def test_autoencoder_callback_filename_includes_idx(self):
        """Filename for each sample must contain _idx<i> to distinguish crops."""
        self.dataset.__getitem__.side_effect = lambda i: {
            "image": torch.zeros(3, 10, 10),
            "target": torch.zeros(3, 10, 10),
        }
        self.dataset.__len__ = MagicMock(return_value=3)

        callback = AutoencoderResultCallback(n_samples=3)
        callback.on_sanity_check_end(self.trainer, self.pl_module)
        self.pl_module.return_value = torch.zeros(1, 3, 10, 10)

        saved_names = []

        def capture_save(fig, name, epoch, sample_idx=None):
            path = f"report_image_{name}_idx{sample_idx}_epoch_{epoch}.png"
            saved_names.append(path)
            return path

        with patch(
            "pytorch_segmentation_models_trainer.custom_callbacks.image_callbacks.generate_visualization"
        ) as mock_vis:
            mock_vis.return_value = (MagicMock(), MagicMock())
            with patch.object(callback, "save_plot_to_disk", side_effect=capture_save):
                with patch.object(callback, "log_data_to_tensorboard"):
                    callback.on_validation_epoch_end(self.trainer, self.pl_module)

        self.assertEqual(len(saved_names), 3)
        for i, name in enumerate(saved_names):
            self.assertIn(f"_idx{i}", name)

    def test_log_every_k_epochs_skips_non_matching_epochs(self):
        """Base class must skip visualization when current_epoch % log_every_k_epochs != 0."""
        callback = ImageSegmentationResultCallback(n_samples=1, log_every_k_epochs=5)
        callback.on_sanity_check_end(self.trainer, self.pl_module)

        with patch(
            "pytorch_segmentation_models_trainer.custom_callbacks.image_callbacks.generate_visualization"
        ) as mock_vis:
            mock_vis.return_value = (MagicMock(), MagicMock())
            self.trainer.current_epoch = 3  # 3 % 5 != 0 → skip
            callback.on_validation_epoch_end(self.trainer, self.pl_module)
            mock_vis.assert_not_called()

            self.trainer.current_epoch = 5  # 5 % 5 == 0 → run
            with patch.object(callback, "save_plot_to_disk", return_value="dummy.png"):
                with patch.object(callback, "log_data_to_tensorboard"):
                    callback.on_validation_epoch_end(self.trainer, self.pl_module)
            mock_vis.assert_called()

    def test_autoencoder_log_every_k_epochs_respected(self):
        """AutoencoderResultCallback must skip when epoch doesn't match log_every_k_epochs."""
        self.dataset.__getitem__.side_effect = lambda i: {
            "image": torch.zeros(3, 10, 10),
        }
        self.dataset.__len__ = MagicMock(return_value=5)

        callback = AutoencoderResultCallback(n_samples=2, log_every_k_epochs=3)
        callback.on_sanity_check_end(self.trainer, self.pl_module)
        self.pl_module.return_value = torch.zeros(1, 3, 10, 10)

        with patch(
            "pytorch_segmentation_models_trainer.custom_callbacks.image_callbacks.generate_visualization"
        ) as mock_vis:
            mock_vis.return_value = (MagicMock(), MagicMock())
            self.trainer.current_epoch = 1  # 1 % 3 != 0 → skip
            callback.on_validation_epoch_end(self.trainer, self.pl_module)
            mock_vis.assert_not_called()

            self.trainer.current_epoch = 3  # 3 % 3 == 0 → run
            with patch.object(callback, "save_plot_to_disk", return_value="dummy.png"):
                with patch.object(callback, "log_data_to_tensorboard"):
                    callback.on_validation_epoch_end(self.trainer, self.pl_module)
            self.assertEqual(mock_vis.call_count, 2)  # n_samples=2

    def test_autoencoder_shuffle_indices_seed_picks_random_subset(self):
        """shuffle_indices_seed causes callback to sample random indices, not always first N."""
        ds_size = 20
        self.dataset.__len__ = MagicMock(return_value=ds_size)
        self.dataset.__getitem__.side_effect = lambda i: {
            "image": torch.zeros(3, 10, 10),
        }

        callback = AutoencoderResultCallback(n_samples=5, shuffle_indices_seed=42)
        callback.on_sanity_check_end(self.trainer, self.pl_module)
        self.pl_module.return_value = torch.zeros(1, 3, 10, 10)

        accessed = []
        original_getitem = self.dataset.__getitem__.side_effect

        def tracking_getitem(i):
            accessed.append(i)
            return {"image": torch.zeros(3, 10, 10)}

        self.dataset.__getitem__.side_effect = tracking_getitem

        with patch(
            "pytorch_segmentation_models_trainer.custom_callbacks.image_callbacks.generate_visualization"
        ) as mock_vis:
            mock_vis.return_value = (MagicMock(), MagicMock())
            with patch.object(callback, "save_plot_to_disk", return_value="dummy.png"):
                with patch.object(callback, "log_data_to_tensorboard"):
                    callback.on_validation_epoch_end(self.trainer, self.pl_module)

        self.assertEqual(len(accessed), 5)
        # With seed=42 and 20 items, accessed indices must not be [0,1,2,3,4]
        self.assertNotEqual(sorted(accessed), list(range(5)))

    def test_autoencoder_shuffle_indices_seed_is_reproducible(self):
        """Same seed gives same indices across two separate callback invocations."""
        ds_size = 20
        self.dataset.__len__ = MagicMock(return_value=ds_size)

        def make_callback():
            cb = AutoencoderResultCallback(n_samples=5, shuffle_indices_seed=7)
            cb.on_sanity_check_end(self.trainer, self.pl_module)
            self.pl_module.return_value = torch.zeros(1, 3, 10, 10)
            accessed = []
            self.dataset.__getitem__.side_effect = lambda i: accessed.append(i) or {
                "image": torch.zeros(3, 10, 10)
            }
            with patch(
                "pytorch_segmentation_models_trainer.custom_callbacks.image_callbacks.generate_visualization"
            ) as mock_vis:
                mock_vis.return_value = (MagicMock(), MagicMock())
                with patch.object(cb, "save_plot_to_disk", return_value="dummy.png"):
                    with patch.object(cb, "log_data_to_tensorboard"):
                        cb.on_validation_epoch_end(self.trainer, self.pl_module)
            return accessed

        indices_a = make_callback()
        indices_b = make_callback()
        self.assertEqual(indices_a, indices_b)

    def test_autoencoder_no_shuffle_seed_uses_first_n(self):
        """Without seed, callback accesses val_ds[0..n-1] sequentially."""
        ds_size = 10
        self.dataset.__len__ = MagicMock(return_value=ds_size)

        callback = AutoencoderResultCallback(n_samples=4)
        callback.on_sanity_check_end(self.trainer, self.pl_module)
        self.pl_module.return_value = torch.zeros(1, 3, 10, 10)

        accessed = []
        self.dataset.__getitem__.side_effect = lambda i: accessed.append(i) or {
            "image": torch.zeros(3, 10, 10)
        }

        with patch(
            "pytorch_segmentation_models_trainer.custom_callbacks.image_callbacks.generate_visualization"
        ) as mock_vis:
            mock_vis.return_value = (MagicMock(), MagicMock())
            with patch.object(callback, "save_plot_to_disk", return_value="dummy.png"):
                with patch.object(callback, "log_data_to_tensorboard"):
                    callback.on_validation_epoch_end(self.trainer, self.pl_module)

        self.assertEqual(accessed, [0, 1, 2, 3])

    # ------------------------------------------------------------------ #
    # use_basename_as_title                                                #
    # ------------------------------------------------------------------ #

    def test_get_title_default_returns_full_path(self):
        callback = ImageSegmentationResultCallback()
        self.assertEqual(
            callback._get_title("/some/dir/image_001.tif"),
            "/some/dir/image_001.tif",
        )

    def test_get_title_basename_mode_returns_stem(self):
        callback = ImageSegmentationResultCallback(use_basename_as_title=True)
        self.assertEqual(callback._get_title("/some/dir/image_001.tif"), "image_001")

    def test_get_title_dotted_name_drops_last_extension_only(self):
        callback = ImageSegmentationResultCallback(use_basename_as_title=True)
        self.assertEqual(callback._get_title("/path/to/tile.v2.tif"), "tile.v2")

    def test_image_segmentation_basename_title_in_visualization(self):
        self.dataset.get_path.side_effect = lambda i: f"/some/path/image_{i}.tif"
        callback = ImageSegmentationResultCallback(
            n_samples=1, use_basename_as_title=True
        )
        callback.on_sanity_check_end(self.trainer, self.pl_module)

        with patch(
            "pytorch_segmentation_models_trainer.custom_callbacks.image_callbacks.generate_visualization"
        ) as mock_vis:
            mock_vis.return_value = (MagicMock(), MagicMock())
            with patch.object(
                self.pl_module, "forward", return_value=torch.zeros(1, 1, 10, 10)
            ):
                with patch.object(
                    callback, "save_plot_to_disk", return_value="dummy.png"
                ):
                    with patch.object(callback, "log_data_to_tensorboard"):
                        callback.on_validation_epoch_end(self.trainer, self.pl_module)

        _, kwargs = mock_vis.call_args
        self.assertEqual(kwargs["fig_title"], "image_0")

    def test_autoencoder_basename_title_in_visualization(self):
        self.dataset.__getitem__.side_effect = lambda i: {
            "image": torch.zeros(3, 10, 10)
        }
        self.dataset.__len__ = MagicMock(return_value=1)
        self.dataset.get_path.side_effect = lambda i: f"/some/path/image_{i}.tif"

        callback = AutoencoderResultCallback(n_samples=1, use_basename_as_title=True)
        callback.on_sanity_check_end(self.trainer, self.pl_module)
        self.pl_module.return_value = torch.zeros(1, 3, 10, 10)

        with patch(
            "pytorch_segmentation_models_trainer.custom_callbacks.image_callbacks.generate_visualization"
        ) as mock_vis:
            mock_vis.return_value = (MagicMock(), MagicMock())
            with patch.object(callback, "save_plot_to_disk", return_value="dummy.png"):
                with patch.object(callback, "log_data_to_tensorboard"):
                    callback.on_validation_epoch_end(self.trainer, self.pl_module)

        _, kwargs = mock_vis.call_args
        self.assertEqual(kwargs["fig_title"], "image_0")

    def test_enhanced_callback_basename_title_in_visualization(self):
        self.dataset.get_path.side_effect = lambda i: f"/some/path/image_{i}.tif"
        callback = EnhancedImageSegmentationResultCallback(
            n_samples=1,
            verbose=False,
            num_classes=2,
            max_workers=1,
            use_basename_as_title=True,
        )
        callback.on_sanity_check_end(self.trainer, self.pl_module)
        self.pl_module.return_value = torch.zeros(1, 2, 10, 10)

        with patch(
            "pytorch_segmentation_models_trainer.custom_callbacks.image_callbacks.generate_visualization"
        ) as mock_vis:
            mock_vis.return_value = (np.zeros((3, 10, 10)), MagicMock())
            with patch.object(callback, "_wait_and_log_to_tensorboard"):
                callback.on_validation_epoch_end(self.trainer, self.pl_module)

        _, kwargs = mock_vis.call_args
        self.assertEqual(kwargs["fig_title"], "image_0")

    def test_frame_field_callback_basename_title(self):
        callback = FrameFieldResultCallback(n_samples=1, use_basename_as_title=True)
        callback.on_sanity_check_end(self.trainer, self.pl_module)
        self.dataloader.__iter__.return_value = iter(
            [
                {
                    "image": torch.ones(1, 3, 10, 10),
                    "gt_polygons_image": torch.zeros(1, 2, 10, 10),
                    "path": ["/some/path/test.tif"],
                }
            ]
        )

        with patch(
            "pytorch_segmentation_models_trainer.custom_callbacks.image_callbacks.generate_visualization"
        ) as mock_vis:
            mock_vis.return_value = (MagicMock(), MagicMock())
            with patch.object(
                self.pl_module,
                "forward",
                return_value={"seg": torch.zeros(1, 2, 10, 10)},
            ):
                with patch.object(
                    callback, "save_plot_to_disk", return_value="dummy.png"
                ):
                    with patch.object(callback, "log_data_to_tensorboard"):
                        callback.on_validation_epoch_end(self.trainer, self.pl_module)

        _, kwargs = mock_vis.call_args
        self.assertEqual(kwargs["fig_title"], "test")

    def test_frame_field_overlayed_basename_title(self):
        callback = FrameFieldOverlayedResultCallback(
            n_samples=1, use_basename_as_title=True
        )
        callback.on_sanity_check_end(self.trainer, self.pl_module)
        self.dataloader.__iter__.return_value = iter(
            [
                {
                    "image": torch.ones(1, 3, 10, 10),
                    "path": ["/some/path/test.tif"],
                }
            ]
        )
        self.pl_module.return_value = {
            "seg": torch.zeros(1, 2, 10, 10),
        }

        with patch(
            "pytorch_segmentation_models_trainer.custom_callbacks.image_callbacks.get_tensorboard_image_seg_display",
            return_value=torch.zeros(1, 3, 10, 10),
        ):
            callback.on_validation_epoch_end(self.trainer, self.pl_module)

        call_args = self.trainer.logger.experiment.add_images.call_args
        tag = call_args[0][0]
        self.assertIn("test", tag)
        self.assertNotIn("/some/path/", tag)

    def test_polygon_rnn_callback_basename_title(self):
        callback = PolygonRNNResultCallback(n_samples=1, use_basename_as_title=True)
        callback.on_sanity_check_end(self.trainer, self.pl_module)

        self.dataset.get_n_image_path_dict_list.return_value = {
            "/some/path/test.tif": {
                "croped_images": torch.ones(1, 3, 224, 224),
                "shapely_polygon_list": [],
                "scale_h": torch.tensor([1.0]),
                "scale_w": torch.tensor([1.0]),
                "min_col": torch.tensor([0]),
                "min_row": torch.tensor([0]),
                "original_image": np.zeros((128, 128, 3)),
            }
        }
        self.pl_module.model.test.return_value = torch.zeros(1, 10)
        self.pl_module.val_seq_len = 10

        with patch(
            "pytorch_segmentation_models_trainer.custom_callbacks.image_callbacks.generate_visualization"
        ) as mock_vis:
            mock_fig = MagicMock()
            mock_fig.get_axes.return_value = [MagicMock(), MagicMock()]
            mock_vis.return_value = (MagicMock(), mock_fig)
            with patch("matplotlib.pyplot.gcf", return_value=mock_fig):
                with patch.object(
                    callback, "save_plot_to_disk", return_value="dummy.png"
                ):
                    with patch.object(callback, "log_data_to_tensorboard"):
                        callback.on_validation_epoch_end(self.trainer, self.pl_module)

        _, kwargs = mock_vis.call_args
        self.assertEqual(kwargs["fig_title"], "test")

    def test_mod_polymapper_basename_title(self):
        callback = ModPolyMapperResultCallback(n_samples=1, use_basename_as_title=True)
        callback.on_sanity_check_end(self.trainer, self.pl_module)

        detection_dict = {
            "boxes": torch.tensor([[0, 0, 10, 10]]),
            "labels": torch.tensor([1]),
            "scores": torch.tensor([0.9]),
            "polygonrnn_output": torch.zeros(1, 10),
            "scale_h": torch.tensor([1.0]),
            "scale_w": torch.tensor([1.0]),
            "min_row": torch.tensor([0]),
            "min_col": torch.tensor([0]),
        }
        prepared_item = {
            "original_image": np.zeros((100, 100, 3)),
            "shapely_polygon_list": [],
        }

        with patch(
            "pytorch_segmentation_models_trainer.custom_callbacks.image_callbacks.generate_visualization"
        ) as mock_vis:
            mock_fig = MagicMock()
            mock_fig.get_axes.return_value = [MagicMock(), MagicMock(), MagicMock()]
            mock_vis.return_value = (MagicMock(), mock_fig)
            with patch("matplotlib.pyplot.gcf", return_value=mock_fig):
                with patch.object(
                    callback, "save_plot_to_disk", return_value="dummy.png"
                ):
                    with patch.object(callback, "log_data_to_tensorboard"):
                        callback.build_obj_det_and_polygon_vis(
                            "/some/path/tile.tif",
                            prepared_item["original_image"],
                            detection_dict,
                            [],
                            [],
                            self.trainer,
                        )

        _, kwargs = mock_vis.call_args
        self.assertEqual(kwargs["fig_title"], "tile")

    def test_object_detection_basename_title(self):
        callback = ObjectDetectionResultCallback(
            n_samples=1, use_basename_as_title=True
        )
        callback.on_sanity_check_end(self.trainer, self.pl_module)

        mock_outputs = [
            {"boxes": torch.tensor([[0, 0, 10, 10]]), "scores": torch.tensor([0.9])}
        ]
        self.dataset.get_path.side_effect = lambda i: f"/some/path/image_{i}.tif"

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

        call_args = self.trainer.logger.experiment.add_image.call_args
        tag = call_args[0][0]
        self.assertEqual(tag, "image_0")

    def test_enhanced_callback_batch_paths_basename_title(self):
        """Enhanced callback uses _get_title for batch paths when dataset has no get_path."""
        dataset_no_path = MagicMock(spec=["__len__", "__getitem__"])
        dataset_no_path.__len__.return_value = 1
        dataloader = MagicMock()
        dataloader.dataset = dataset_no_path
        dataloader.batch_size = 1
        dataloader.__iter__.return_value = iter(
            [
                {
                    "image": torch.zeros(1, 3, 10, 10),
                    "mask": torch.zeros(1, 1, 10, 10),
                    "path": ["/some/path/test.tif"],
                }
            ]
        )
        self.pl_module.val_dataloader.return_value = dataloader

        callback = EnhancedImageSegmentationResultCallback(
            n_samples=1,
            verbose=False,
            num_classes=2,
            max_workers=1,
            use_basename_as_title=True,
        )
        callback.on_sanity_check_end(self.trainer, self.pl_module)
        self.pl_module.return_value = torch.zeros(1, 2, 10, 10)

        with patch(
            "pytorch_segmentation_models_trainer.custom_callbacks.image_callbacks.generate_visualization"
        ) as mock_vis:
            mock_vis.return_value = (np.zeros((3, 10, 10)), MagicMock())
            with patch.object(callback, "_wait_and_log_to_tensorboard"):
                callback.on_validation_epoch_end(self.trainer, self.pl_module)

        _, kwargs = mock_vis.call_args
        self.assertEqual(kwargs["fig_title"], "test")

    def test_enhanced_callback_generic_sample_name_when_no_paths(self):
        """Enhanced callback falls back to sample_N when no get_path and no batch paths."""
        dataset_no_path = MagicMock(spec=["__len__", "__getitem__"])
        dataset_no_path.__len__.return_value = 1
        dataloader = MagicMock()
        dataloader.dataset = dataset_no_path
        dataloader.batch_size = 1
        dataloader.__iter__.return_value = iter(
            [
                {
                    "image": torch.zeros(1, 3, 10, 10),
                    "mask": torch.zeros(1, 1, 10, 10),
                }
            ]
        )
        self.pl_module.val_dataloader.return_value = dataloader

        callback = EnhancedImageSegmentationResultCallback(
            n_samples=1,
            verbose=False,
            num_classes=2,
            max_workers=1,
        )
        callback.on_sanity_check_end(self.trainer, self.pl_module)
        self.pl_module.return_value = torch.zeros(1, 2, 10, 10)

        with patch(
            "pytorch_segmentation_models_trainer.custom_callbacks.image_callbacks.generate_visualization"
        ) as mock_vis:
            mock_vis.return_value = (np.zeros((3, 10, 10)), MagicMock())
            with patch.object(callback, "_wait_and_log_to_tensorboard"):
                callback.on_validation_epoch_end(self.trainer, self.pl_module)

        _, kwargs = mock_vis.call_args
        self.assertIn("sample_", kwargs["fig_title"])
