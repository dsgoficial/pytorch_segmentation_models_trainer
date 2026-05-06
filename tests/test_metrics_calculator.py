# -*- coding: utf-8 -*-
"""
Unit tests for metrics_calculator.py
"""

import unittest
from unittest.mock import MagicMock, patch, mock_open
import numpy as np
import pandas as pd
import torch
import rasterio
from rasterio.transform import Affine
from rasterio.windows import Window
from omegaconf import OmegaConf, DictConfig
import json
import os

from pytorch_segmentation_models_trainer.tools.evaluation.metrics_calculator import (
    MetricsCalculator,
    _get_spatial_overlap,
)
from pytorch_segmentation_models_trainer.tools.evaluation.image_processing_worker import (
    process_single_image_worker,
)
from pytorch_segmentation_models_trainer.tools.data_handlers.raster_reader import (
    RasterFile,
)


# Helper function to create a dummy raster file in memory
def create_dummy_raster_in_memory(
    data: np.ndarray, transform: Affine, crs: str = "EPSG:4326"
):
    height, width = data.shape[-2:]  # Assuming (C, H, W) or (H, W)
    profile = {
        "driver": "GTiff",
        "height": height,
        "width": width,
        "count": data.shape[0] if data.ndim == 3 else 1,
        "dtype": str(data.dtype),
        "crs": crs,
        "transform": transform,
    }
    memfile = rasterio.MemoryFile()
    with memfile.open(**profile) as dst:
        if data.ndim == 3:
            for i in range(data.shape[0]):
                dst.write(data[i], i + 1)
        else:
            dst.write(data, 1)
    return memfile


class TestSpatialOverlap(unittest.TestCase):
    def test_full_overlap(self):
        # Two rasters with identical bounds
        transform = Affine(
            1.0, 0.0, 0.0, 0.0, -1.0, 10.0
        )  # Origin (0,10), 1m resolution
        data = np.zeros((10, 10), dtype=np.uint8)
        with create_dummy_raster_in_memory(data, transform) as r1:
            with create_dummy_raster_in_memory(data, transform) as r2:
                pred_window, gt_window, matched_shape = _get_spatial_overlap(
                    r1.name, r2.name
                )

            self.assertIsNotNone(pred_window)
            self.assertEqual(pred_window, Window(0, 0, 10, 10))
            self.assertEqual(gt_window, Window(0, 0, 10, 10))
            self.assertEqual(matched_shape, (10, 10))

    def test_partial_overlap(self):
        # Raster 1: (0,0) to (10,10)
        t1 = Affine(1.0, 0.0, 0.0, 0.0, -1.0, 10.0)
        d1 = np.zeros((10, 10), dtype=np.uint8)

        # Raster 2: (5,5) to (15,15)
        t2 = Affine(1.0, 0.0, 5.0, 0.0, -1.0, 5.0)
        d2 = np.zeros((10, 10), dtype=np.uint8)

        with create_dummy_raster_in_memory(d1, t1) as r1:
            with create_dummy_raster_in_memory(d2, t2) as r2:
                pred_window, gt_window, matched_shape = _get_spatial_overlap(
                    r1.name, r2.name
                )

            self.assertIsNotNone(pred_window)
            self.assertEqual(pred_window, Window(5, 5, 5, 5))
            self.assertEqual(gt_window, Window(0, 0, 5, 5))
            self.assertEqual(matched_shape, (5, 5))

    def test_no_overlap(self):
        # Raster 1: (0,0) to (10,10)
        t1 = Affine(1.0, 0.0, 0.0, 0.0, -1.0, 10.0)
        d1 = np.zeros((10, 10), dtype=np.uint8)

        # Raster 2: (11,11) to (21,21)
        t2 = Affine(1.0, 0.0, 11.0, 0.0, -1.0, -1.0)
        d2 = np.zeros((10, 10), dtype=np.uint8)

        with create_dummy_raster_in_memory(d1, t1) as r1:
            with create_dummy_raster_in_memory(d2, t2) as r2:
                result = _get_spatial_overlap(r1.name, r2.name)
            self.assertIsNone(result)

    def test_window_size_mismatch_warning(self):
        # Raster 1: (0,0) to (10,10)
        t1 = Affine(1.0, 0.0, 0.0, 0.0, -1.0, 10.0)
        d1 = np.zeros((10, 10), dtype=np.uint8)

        # Raster 2: Much larger difference to ensure warning (> 1 pixel)
        # Intersection from bounds will be (0.5, 0) to (10, 8.5)
        # For Raster 1: col_off=0.5/1.0=0.5, row_off=(10-8.5)/1.0=1.5, width=9.5, height=8.5
        # Rounding for Raster 1: Window(1, 2, 10, 9)
        # For Raster 2: col_off=(0.5-0.5)/2.0=0, row_off=(8.5-8.5)/-2.0=0, width=(10-0.5)/2.0=4.75, height=(8.5-0)/2.0=4.25
        # Rounding for Raster 2: Window(0, 0, 5, 4)
        t2 = Affine(2.0, 0.0, 0.5, 0.0, -2.0, 8.5)
        d2 = np.zeros((10, 10), dtype=np.uint8)

        with create_dummy_raster_in_memory(d1, t1) as r1:
            with create_dummy_raster_in_memory(d2, t2) as r2:
                with self.assertLogs(
                    "pytorch_segmentation_models_trainer.tools.evaluation.metrics_calculator",
                    level="WARNING",
                ) as cm:
                    pred_window, gt_window, matched_shape = _get_spatial_overlap(
                        r1.name, r2.name
                    )
                self.assertIn(
                    "Window size mismatch after spatial alignment:", cm.output[0]
                )


class TestMetricsCalculator(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = "temp_test_dir"
        os.makedirs(self.tmp_dir, exist_ok=True)

        self.transform = Affine(1.0, 0.0, 0.0, 0.0, -1.0, 10.0)
        self.raster_size = (10, 10)
        self.num_classes = 2

        self.config = OmegaConf.create(
            {
                "metrics": {
                    "iou": {
                        "_target_": "torchmetrics.JaccardIndex",
                        "task": "binary",
                        "num_classes": self.num_classes,
                    },
                    "f1": {
                        "_target_": "torchmetrics.F1Score",
                        "task": "binary",
                        "num_classes": self.num_classes,
                    },
                },
                "pixel_metrics": {
                    "accuracy": {
                        "_target_": "torchmetrics.Accuracy",
                        "task": "binary",
                        "num_classes": self.num_classes,
                    },
                },
                "object_metrics": {},
                "num_classes": self.num_classes,
                "experiment_name": "test_exp",
                "output_folder": self.tmp_dir,
                "threshold": 0.5,
                "target_crs": "EPSG:4326",
                "output_metrics_json": os.path.join(self.tmp_dir, "metrics.json"),
                "prediction_csv_path": os.path.join(self.tmp_dir, "predictions.csv"),
                "ground_truth_csv_path": os.path.join(self.tmp_dir, "gt.csv"),
            }
        )

        self.gt_df = pd.DataFrame(
            {
                "image": ["image_0.tif"],
                "mask": ["gt_mask_0.tif"],
                "width": [self.raster_size[1]],
                "height": [self.raster_size[0]],
                "id": ["0"],
            }
        )
        self.pred_df = pd.DataFrame(
            {
                "image": ["image_0.tif"],
                "mask": ["pred_mask_0.tif"],
                "width": [self.raster_size[1]],
                "height": [self.raster_size[0]],
                "id": ["0"],
            }
        )
        self.gt_df.to_csv(self.config.ground_truth_csv_path, index=False)
        self.pred_df.to_csv(self.config.prediction_csv_path, index=False)

        self.gt_mask_path = os.path.join(self.tmp_dir, "gt_mask_0.tif")
        self.pred_mask_path = os.path.join(self.tmp_dir, "pred_mask_0.tif")

        self.gt_data = np.zeros(self.raster_size, dtype=np.uint8)
        self.pred_data = np.zeros(self.raster_size, dtype=np.uint8)

        with create_dummy_raster_in_memory(self.gt_data, self.transform) as gt_memfile:
            with open(self.gt_mask_path, "wb") as f:
                f.write(gt_memfile.read())
        with create_dummy_raster_in_memory(
            self.pred_data, self.transform
        ) as pred_memfile:
            with open(self.pred_mask_path, "wb") as f:
                f.write(pred_memfile.read())

    def tearDown(self):
        if os.path.exists(self.tmp_dir):
            import shutil

            shutil.rmtree(self.tmp_dir)

    @patch(
        "pytorch_segmentation_models_trainer.tools.evaluation.metrics_calculator.process_single_image_worker"
    )
    def test_calculate_metrics_basic(self, mock_worker):
        mock_worker.return_value = {
            "image_name": "gt_mask_0",
            "pred_flat": np.zeros(100, dtype=np.int32),
            "gt_flat": np.zeros(100, dtype=np.int32),
        }

        calculator = MetricsCalculator(self.config)
        with patch.object(
            calculator,
            "_compute_confusion_matrix_fast",
            return_value=np.zeros((2, 2), dtype=np.int64),
        ):
            aggregated_metrics = calculator.calculate_metrics()

        self.assertIsNotNone(aggregated_metrics)
        self.assertIn("gt_mask_0", aggregated_metrics)
        self.assertTrue(os.path.exists(self.config.output_metrics_json))

    def test_calculate_metrics_empty_csv(self):
        empty_df = pd.DataFrame(columns=["image", "mask", "width", "height", "id"])
        empty_df.to_csv(self.config.prediction_csv_path, index=False)

        calculator = MetricsCalculator(self.config)
        with self.assertRaises(ValueError):
            calculator.calculate_metrics()

    def test_process_single_image_worker_integration(self):
        # We need to ensure paths are absolute or correct for the worker
        result = process_single_image_worker(
            image_id="test_id",
            prediction_path=os.path.abspath(self.pred_mask_path),
            ground_truth_path=os.path.abspath(self.gt_mask_path),
            num_classes=self.num_classes,
        )

        self.assertIsNotNone(result)
        self.assertEqual(result["image_id"], "test_id")
        self.assertIn("pred_flat", result)
        self.assertIn("gt_flat", result)


if __name__ == "__main__":
    unittest.main()
