# -*- coding: utf-8 -*-
"""
Unit tests for inference_processors.py
"""

import unittest
from unittest.mock import MagicMock, patch, call
import numpy as np
import torch
import rasterio
from rasterio.transform import Affine
import albumentations as A
from collections import defaultdict
import os
import shutil
from pathlib import Path

from pytorch_segmentation_models_trainer.tools.inference.inference_processors import (
    AbstractInferenceProcessor,
    SingleImageInfereceProcessor,
    MultiClassInferenceProcessor,
)
from pytorch_segmentation_models_trainer.tools.polygonization.polygonizer import (
    TemplatePolygonizerProcessor,
)
from pytorch_segmentation_models_trainer.tools.inference.export_inference import (
    RasterExportInferenceStrategy,
)
from pytorch_toolbelt.utils.torch_utils import (
    image_to_tensor,
    to_numpy,
)
from torch.utils.data import DataLoader
from pytorch_toolbelt.inference.tiles import ImageSlicer, TileMerger


# Dummy Model for testing
class DummySegmentationModel(torch.nn.Module):
    def __init__(self, num_classes=1):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, num_classes, 1)  # Simple conv to produce output
        self.num_classes = num_classes

    def forward(self, x):
        h, w = x.shape[-2:]
        return torch.randn(x.shape[0], self.num_classes, h, w)


# Dummy Export Strategy
class DummyExportStrategy(RasterExportInferenceStrategy):
    def __init__(self, output_dir="."):
        self.output_dir = output_dir
        self.saved_inferences = []

    def save_inference(self, inference_data, profile):
        self.saved_inferences.append((inference_data, profile))
        return os.path.join(self.output_dir, f"{profile['input_name']}.tif")


# Dummy implementation of AbstractInferenceProcessor for testing abstract methods
class DummyInferenceProcessor(AbstractInferenceProcessor):
    def make_inference(self, image: np.ndarray, **kwargs) -> dict:
        return {"seg": np.zeros(image.shape[:2] + (self.mask_bands,), dtype=np.uint8)}


class TestAbstractInferenceProcessor(unittest.TestCase):
    def setUp(self):
        self.model = MagicMock(spec=DummySegmentationModel)
        self.device = "cpu"
        self.batch_size = 1
        self.export_strategy = DummyExportStrategy()
        self.polygonizer = MagicMock(spec=TemplatePolygonizerProcessor)
        self.model_input_shape = (256, 256)
        self.mask_bands = 1
        self.normalize_mean = [0.5, 0.5, 0.5]
        self.normalize_std = [0.5, 0.5, 0.5]
        self.normalize_max_value = 255.0

        self.processor = DummyInferenceProcessor(
            model=self.model,
            device=self.device,
            batch_size=self.batch_size,
            export_strategy=self.export_strategy,
            polygonizer=self.polygonizer,
            model_input_shape=self.model_input_shape,
            step_shape=(128, 128),
            mask_bands=self.mask_bands,
            normalize_mean=self.normalize_mean,
            normalize_std=self.normalize_std,
            normalize_max_value=self.normalize_max_value,
            group_output_by_image_basename=False,
        )

        self.image_data = np.random.randint(0, 256, (512, 512, 3), dtype=np.uint8)
        self.image_path = "dummy_image.tif"
        self.transform = Affine(1.0, 0.0, 0.0, 0.0, -1.0, 512.0)

        # Mock rasterio.open for read_image_and_profile
        self.mock_raster_ds = MagicMock()
        self.mock_raster_ds.read.return_value = np.moveaxis(self.image_data, -1, 0)
        self.mock_raster_ds.profile = {
            "driver": "GTiff",
            "height": 512,
            "width": 512,
            "count": 3,
            "dtype": "uint8",
            "crs": "EPSG:4326",
            "transform": self.transform,
        }
        self.mock_raster_ds.__enter__.return_value = self.mock_raster_ds
        self.mock_raster_ds.__exit__.return_value = False

    @patch("rasterio.open")
    def test_read_image_and_profile(self, mock_rasterio_open):
        mock_rasterio_open.return_value = self.mock_raster_ds
        image, profile = self.processor.read_image_and_profile(self.image_path)
        mock_rasterio_open.assert_called_once_with(self.image_path, "r")
        np.testing.assert_array_equal(image, self.image_data)
        self.assertEqual(profile["height"], 512)
        self.assertEqual(profile["width"], 512)

    def test_get_normalization_function(self):
        norm_func = self.processor.get_normalization_function()
        self.assertIsInstance(norm_func, A.Normalize)
        dummy_img = np.zeros((10, 10, 3), dtype=np.uint8)
        normalized_img = norm_func(image=dummy_img)["image"]
        expected_val = (0 - 0.5 * 255.0) / (0.5 * 255.0)
        self.assertAlmostEqual(normalized_img[0, 0, 0], expected_val)


class TestSingleImageInferenceProcessor(unittest.TestCase):
    def setUp(self):
        self.model = MagicMock(spec=DummySegmentationModel)
        self.model.to.return_value = self.model
        self.model.eval.return_value = None
        self.device = "cpu"
        self.batch_size = 1
        self.export_strategy = DummyExportStrategy()
        self.polygonizer = MagicMock(spec=TemplatePolygonizerProcessor)
        self.model_input_shape = (256, 256)
        self.step_shape = (128, 128)
        self.mask_bands = 1
        self.normalize_mean = [0.5, 0.5, 0.5]
        self.normalize_std = [0.5, 0.5, 0.5]
        self.normalize_max_value = 255.0

        self.processor = SingleImageInfereceProcessor(
            model=self.model,
            device=self.device,
            batch_size=self.batch_size,
            export_strategy=self.export_strategy,
            polygonizer=self.polygonizer,
            model_input_shape=self.model_input_shape,
            step_shape=self.step_shape,
            mask_bands=self.mask_bands,
            normalize_mean=self.normalize_mean,
            normalize_std=self.normalize_std,
            normalize_max_value=self.normalize_max_value,
            group_output_by_image_basename=False,
            use_tta=False,
            tta_mode=None,
            tile_weight="mean",
        )

        self.image_data = np.random.randint(0, 256, (512, 512, 3), dtype=np.uint8)
        self.image_path = "dummy_image.tif"
        self.transform = Affine(1.0, 0.0, 0.0, 0.0, -1.0, 512.0)

        self.mock_raster_ds = MagicMock()
        self.mock_raster_ds.read.return_value = np.moveaxis(self.image_data, -1, 0)
        self.mock_raster_ds.profile = {
            "driver": "GTiff",
            "height": 512,
            "width": 512,
            "count": 3,
            "dtype": "uint8",
            "crs": "EPSG:4326",
            "transform": self.transform,
        }
        self.mock_raster_ds.__enter__.return_value = self.mock_raster_ds
        self.mock_raster_ds.__exit__.return_value = False

    def test_init_tta_mode_alias(self):
        processor_d4 = SingleImageInfereceProcessor(
            self.model,
            self.device,
            self.batch_size,
            self.export_strategy,
            tta_mode="d4",
        )
        self.assertTrue(processor_d4.use_tta)
        self.assertEqual(len(processor_d4.tta_augmentations), 8)

        processor_flip = SingleImageInfereceProcessor(
            self.model,
            self.device,
            self.batch_size,
            self.export_strategy,
            tta_mode="flip",
        )
        self.assertTrue(processor_flip.use_tta)
        self.assertEqual(len(processor_flip.tta_augmentations), 4)

        with self.assertRaises(ValueError):
            SingleImageInfereceProcessor(
                self.model,
                self.device,
                self.batch_size,
                self.export_strategy,
                tta_mode="invalid",
            )

    def test_gaussian_importance_map(self):
        patch_size = (11, 11)  # Odd size to ensure peak at 1.0
        gaussian_map = self.processor._gaussian_importance_map(patch_size)
        self.assertEqual(gaussian_map.shape, patch_size)
        self.assertAlmostEqual(gaussian_map.max(), 1.0, places=5)
        self.assertLess(gaussian_map.min(), 1.0)

    def test_resolve_tile_weight(self):
        self.assertEqual(self.processor._resolve_tile_weight("mean"), "mean")
        self.assertEqual(self.processor._resolve_tile_weight("pyramid"), "pyramid")
        gaussian_weight = self.processor._resolve_tile_weight("gaussian")
        self.assertEqual(gaussian_weight.shape, self.model_input_shape)
        with self.assertRaises(ValueError):
            self.processor._resolve_tile_weight("invalid_weight")

    @patch("rasterio.open")
    @patch.object(SingleImageInfereceProcessor, "save_inference")
    @patch.object(
        SingleImageInfereceProcessor,
        "make_inference",
        return_value={"seg": np.zeros((512, 512, 1), dtype=np.uint8)},
    )
    def test_process_basic_flow(
        self, mock_make_inference, mock_save_inference, mock_rasterio_open
    ):
        mock_rasterio_open.return_value = self.mock_raster_ds
        self.processor.process(self.image_path)
        mock_rasterio_open.assert_called_once_with(self.image_path, "r")
        mock_make_inference.assert_called_once()
        mock_save_inference.assert_called_once()

    @patch("rasterio.open")
    @patch(
        "pytorch_segmentation_models_trainer.tools.inference.inference_processors.ImageSlicer"
    )
    @patch(
        "pytorch_segmentation_models_trainer.tools.inference.inference_processors.TileMerger"
    )
    def test_make_inference_no_tta(
        self, MockTileMerger, MockImageSlicer, mock_rasterio_open
    ):
        mock_rasterio_open.return_value = self.mock_raster_ds
        mock_tiler_instance = MagicMock(spec=ImageSlicer)
        mock_tiler_instance.split.return_value = [
            np.zeros((3, 256, 256), dtype=np.float32)
        ] * 2
        mock_tiler_instance.crops = [(0, 0, 256, 256), (0, 0, 256, 256)]
        mock_tiler_instance.target_shape = (1, 512, 512)
        mock_tiler_instance.image_height = 512
        mock_tiler_instance.image_width = 512
        mock_tiler_instance.margin_top = 0
        mock_tiler_instance.margin_left = 0
        mock_tiler_instance.weight = "mean"
        mock_tiler_instance.crop_to_orignal_size.side_effect = lambda x: x
        MockImageSlicer.return_value = mock_tiler_instance

        mock_merger_instance = MagicMock(spec=TileMerger)
        mock_merger_instance.merge.return_value = torch.zeros(1, 512, 512)
        MockTileMerger.return_value = mock_merger_instance

        # Ensure model returns the tensor when called
        self.model.return_value = torch.randn(
            self.batch_size, self.mask_bands, 256, 256
        )
        inference_output = self.processor.make_inference(self.image_data)

        MockTileMerger.assert_called_once()
        self.assertIn("seg", inference_output)


class TestMultiClassInferenceProcessor(unittest.TestCase):
    def setUp(self):
        self.num_classes = 3
        self.model = MagicMock(spec=DummySegmentationModel)
        self.model.num_classes = self.num_classes
        self.model.to.return_value = self.model
        self.model.eval.return_value = None
        self.device = "cpu"
        self.batch_size = 1
        self.export_strategy = DummyExportStrategy()
        self.model_input_shape = (256, 256)
        self.step_shape = (128, 128)

        self.processor = MultiClassInferenceProcessor(
            model=self.model,
            device=self.device,
            batch_size=self.batch_size,
            export_strategy=self.export_strategy,
            num_classes=self.num_classes,
            model_input_shape=self.model_input_shape,
            step_shape=self.step_shape,
        )

        self.tmp_dir = "temp_multi_class_test_dir"
        os.makedirs(self.tmp_dir, exist_ok=True)
        self.image_data = np.random.randint(0, 256, (512, 512, 3), dtype=np.uint8)
        self.image_path = "dummy_image.tif"
        self.mock_raster_ds = MagicMock()
        self.mock_raster_ds.read.return_value = np.moveaxis(self.image_data, -1, 0)
        self.mock_raster_ds.profile = {
            "height": 512,
            "width": 512,
            "count": 3,
            "dtype": "uint8",
            "crs": "EPSG:4326",
            "transform": Affine.identity(),
        }
        self.mock_raster_ds.__enter__.return_value = self.mock_raster_ds
        self.mock_raster_ds.__exit__.return_value = False

    def tearDown(self):
        if os.path.exists(self.tmp_dir):
            shutil.rmtree(self.tmp_dir)

    def test_init_tta_mode_multiclass(self):
        processor_d4 = MultiClassInferenceProcessor(
            self.model,
            self.device,
            self.batch_size,
            self.export_strategy,
            num_classes=self.num_classes,
            tta_mode="d4",
        )
        self.assertEqual(processor_d4.tta_mode, "d4")
        with self.assertRaises(ValueError):
            MultiClassInferenceProcessor(
                self.model,
                self.device,
                self.batch_size,
                self.export_strategy,
                num_classes=self.num_classes,
                tta_mode="invalid",
            )

    def test_compute_confidence_maps(self):
        probs = np.array(
            [[[0.9, 0.1], [0.5, 0.5]], [[0.1, 0.9], [0.5, 0.5]]],  # Class 0  # Class 1
            dtype=np.float32,
        )

        confidence_maps = self.processor._compute_confidence_maps(probs)
        self.assertIn("max_prob", confidence_maps)
        self.assertIn("entropy", confidence_maps)
        self.assertIn("margin", confidence_maps)

        np.testing.assert_array_almost_equal(
            confidence_maps["max_prob"], np.array([[0.9, 0.9], [0.5, 0.5]])
        )
        np.testing.assert_array_almost_equal(
            confidence_maps["margin"], np.array([[0.8, 0.8], [0.0, 0.0]])
        )

    @patch("rasterio.open")
    def test_is_large_image(self, mock_rasterio_open):
        mock_ds = MagicMock()
        mock_ds.height = 10000
        mock_ds.width = 10000
        mock_ds.__enter__.return_value = mock_ds
        mock_rasterio_open.return_value = mock_ds
        self.processor.striped_threshold_pixels = 50_000_000
        self.assertTrue(self.processor._is_large_image("large.tif"))

    @patch("rasterio.open")
    @patch.object(MultiClassInferenceProcessor, "make_inference_striped")
    @patch.object(MultiClassInferenceProcessor, "_is_large_image", return_value=True)
    def test_process_large_image_flow(
        self, mock_is_large, mock_make_striped, mock_open
    ):
        self.processor.export_strategy.output_file_path = self.tmp_dir
        result = self.processor.process(self.image_path)
        mock_make_striped.assert_called_once()
        self.assertEqual(len(result), 0)


if __name__ == "__main__":
    unittest.main()
