# -*- coding: utf-8 -*-
"""
Unit tests for metrics_calculator.py
"""

import unittest
from unittest.mock import MagicMock, patch
import numpy as np
import pandas as pd
import torch
import rasterio
from rasterio.transform import Affine
from rasterio.windows import Window
from omegaconf import OmegaConf
import os
from pathlib import Path

from pytorch_segmentation_models_trainer.tools.evaluation.metrics_calculator import (
    MetricsCalculator,
    _get_spatial_overlap,
)
from pytorch_segmentation_models_trainer.tools.evaluation.image_processing_worker import (
    process_single_image_worker,
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

    @patch(
        "pytorch_segmentation_models_trainer.tools.evaluation.metrics_calculator.rasterio.open"
    )
    def test_spatial_overlap_open_error(self, mock_open):
        mock_open.side_effect = RuntimeError("open failed")
        self.assertIsNone(_get_spatial_overlap("pred.tif", "gt.tif"))


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

    def test_metric_instantiation_error_is_ignored(self):
        cfg = OmegaConf.create(
            {
                "metrics": {"bad": {"_target_": "missing.module.Metric"}},
                "pixel_metrics": {},
                "num_classes": 2,
                "output_folder": self.tmp_dir,
            }
        )
        calculator = MetricsCalculator(cfg)
        self.assertEqual(calculator.metrics_to_compute, {})

    def test_find_prediction_files_tasks_and_matching_variants(self):
        calculator = MetricsCalculator(self.config)
        found = calculator._find_prediction_files(self.tmp_dir)
        self.assertIn("pred_mask_0", found)

        self.assertEqual(
            calculator._find_matching_prediction(
                "pred_mask_0", {"pred_mask_0": "direct.tif"}
            ),
            "direct.tif",
        )
        self.assertEqual(
            calculator._find_matching_prediction("mask_tile", {"tile": "pred.tif"}),
            "pred.tif",
        )
        self.assertEqual(
            calculator._find_matching_prediction("abc_tile_xyz", {"tile": "pred.tif"}),
            "pred.tif",
        )
        self.assertIsNone(calculator._find_matching_prediction("missing", {}))

        rows = pd.DataFrame(
            [
                {"image": "", "mask": "", "prediction": ""},
                {"mask": "gt_mask_0.tif", "prediction": "pred_mask_0.tif"},
            ]
        )
        tasks = calculator._create_tasks(rows, {})
        self.assertEqual(len(tasks), 1)
        self.assertTrue(tasks[0]["gt_path"].endswith("gt_mask_0.tif"))

    def test_compute_metrics_for_single_pair_no_overlap_threshold_and_metric_error(
        self,
    ):
        calculator = MetricsCalculator(self.config)
        self.assertEqual(
            calculator._compute_metrics_for_single_pair(
                self.pred_mask_path,
                os.path.join(self.tmp_dir, "missing_gt.tif"),
            ),
            {},
        )

        good_metric = MagicMock(return_value=torch.tensor(1.0))
        bad_metric = MagicMock(side_effect=RuntimeError("metric failed"))
        calculator.pixel_metrics = {"good": good_metric}
        calculator.metrics_to_compute = {"bad": bad_metric}
        result = calculator._compute_metrics_for_single_pair(
            self.pred_mask_path,
            self.gt_mask_path,
            threshold=0.5,
        )

        self.assertEqual(result["good"], 1.0)
        self.assertNotIn("bad", result)

    def test_calculate_metrics_fallbacks_prediction_folder_and_parallel(self):
        fallback_cfg = OmegaConf.create(dict(self.config))
        fallback_cfg.ground_truth_csv_path = os.path.join(
            self.tmp_dir, "missing_gt.csv"
        )
        calculator = MetricsCalculator(fallback_cfg)
        with (
            patch.object(
                calculator,
                "_create_tasks",
                return_value=[
                    {
                        "image_name": "a",
                        "pred_path": self.pred_mask_path,
                        "gt_path": self.gt_mask_path,
                    },
                    {
                        "image_name": "b",
                        "pred_path": self.pred_mask_path,
                        "gt_path": self.gt_mask_path,
                    },
                ],
            ),
            patch.object(
                calculator,
                "_process_images_parallel",
                return_value=[{"image_name": "a", "iou": 1.0}],
            ) as process_parallel,
            patch.object(
                calculator,
                "_compute_metrics_from_results",
                return_value={"ok": True},
            ),
        ):
            output = calculator.calculate_metrics(
                predictions_folder=self.tmp_dir,
                ground_truth_csv=None,
                experiment_name="parallel_exp",
                parallel=True,
                num_workers=None,
            )

        self.assertEqual(output, {"ok": True})
        process_parallel.assert_called_once()

        no_csv_cfg = OmegaConf.create(dict(self.config))
        no_csv_cfg.ground_truth_csv_path = os.path.join(self.tmp_dir, "missing_gt.csv")
        no_csv_cfg.prediction_csv_path = os.path.join(self.tmp_dir, "missing_pred.csv")
        with self.assertRaises(ValueError):
            MetricsCalculator(no_csv_cfg).calculate_metrics()

        with patch.object(calculator, "_create_tasks", return_value=[]):
            with self.assertRaises(ValueError):
                calculator.calculate_metrics(
                    ground_truth_csv=self.config.prediction_csv_path
                )

    def test_processing_helpers_and_metric_aggregation_paths(self):
        calculator = MetricsCalculator(self.config)
        tasks = [
            {
                "image_name": "im",
                "pred_path": self.pred_mask_path,
                "gt_path": self.gt_mask_path,
            }
        ]

        with patch(
            "pytorch_segmentation_models_trainer.tools.evaluation.metrics_calculator.process_single_image_worker",
            return_value=None,
        ):
            self.assertEqual(calculator._process_images_sequential(tasks), [])

        with patch(
            "pytorch_segmentation_models_trainer.tools.evaluation.metrics_calculator.process_single_image_worker",
            return_value={"image_name": "im", "iou": 0.5, "accuracy": 0.75},
        ):
            parallel_results = calculator._process_images_parallel(
                tasks * 2, num_workers=1
            )
        self.assertEqual(len(parallel_results), 2)

        with (
            patch.object(calculator, "_save_results"),
            patch.object(
                calculator,
                "_prepare_output_directory",
                return_value=Path(self.tmp_dir),
            ),
        ):
            averaged = calculator._compute_metrics_from_results(
                [
                    {"image_name": "a", "iou": 0.5, "accuracy": 0.75},
                    {"image_name": "b", "iou": 1.0, "accuracy": 0.25},
                ],
                "avg_exp",
            )
            empty = calculator._compute_metrics_from_results([], "empty")

        self.assertEqual(averaged["overall"]["iou"], 0.75)
        self.assertEqual(empty, {})

        with (
            patch.object(calculator, "_save_results"),
            patch.object(
                calculator,
                "_prepare_output_directory",
                return_value=Path(self.tmp_dir),
            ),
        ):
            matrix_based = calculator._compute_metrics_from_results(
                [
                    {
                        "image_name": "cm",
                        "pred_flat": np.array([0, 1, 1, 0]),
                        "gt_flat": np.array([0, 1, 0, 0]),
                    }
                ],
                "cm_exp",
            )
        self.assertIn("Accuracy", matrix_based["overall"])

        cm = calculator._compute_confusion_matrix_fast(
            torch.tensor([0, 1, 1]),
            torch.tensor([0, 1, 0]),
        )
        self.assertEqual(cm.shape, (2, 2))
        self.assertIn(
            "JaccardIndex", calculator._metrics_from_confusion_matrix_aggregated(cm)
        )

    def test_process_images_parallel_future_exception(self):
        calculator = MetricsCalculator(self.config)

        class BadFuture:
            def result(self):
                raise RuntimeError("future failed")

        class DummyExecutor:
            def __init__(self, max_workers):
                self.max_workers = max_workers

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

            def submit(self, *args, **kwargs):
                return BadFuture()

        with (
            patch(
                "pytorch_segmentation_models_trainer.tools.evaluation.metrics_calculator.ThreadPoolExecutor",
                DummyExecutor,
            ),
            patch(
                "pytorch_segmentation_models_trainer.tools.evaluation.metrics_calculator.as_completed",
                lambda futures: list(futures),
            ),
        ):
            output = calculator._process_images_parallel(
                [{"image_name": "im"}],
                num_workers=None,
            )

        self.assertEqual(output, [])

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
