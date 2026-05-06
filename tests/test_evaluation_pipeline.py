# -*- coding: utf-8 -*-
"""
Unit tests for evaluation_pipeline.py
"""

import unittest
from unittest.mock import MagicMock, patch, call
import os
import shutil
import tempfile
from pathlib import Path
import pandas as pd
import numpy as np
from omegaconf import OmegaConf

from pytorch_segmentation_models_trainer.tools.evaluation.evaluation_pipeline import (
    EvaluationPipeline,
)


class TestEvaluationPipeline(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()
        self.base_dir = os.path.join(self.tmp_dir, "base")
        os.makedirs(self.base_dir, exist_ok=True)

        self.config = OmegaConf.create(
            {
                "experiments": [
                    {
                        "name": "exp1",
                        "output_folder": os.path.join(self.base_dir, "exp1"),
                        "checkpoint_path": "model1.ckpt",
                        "predict_config": "config1.yaml",
                    }
                ],
                "output": {
                    "base_dir": self.base_dir,
                    "files": {"summary_report": "summary.json"},
                    "structure": {
                        "reports_folder": "reports",
                        "comparison_folder": "comparisons",
                    },
                },
                "evaluation_dataset": {
                    "input_csv_path": os.path.join(self.base_dir, "dataset.csv"),
                    "build_csv_from_folders": {"enabled": False},
                },
                "pipeline_options": {
                    "parallel_inference": {"enabled": False},
                    "skip_existing_predictions": False,
                    "skip_existing_evaluations": False,
                    "load_predictions_from_folder": {"enabled": False},
                },
                "visualization": {
                    "comparison_plots": {"enabled": False},
                    "confusion_matrix": {
                        "save_individual": False,
                        "save_comparison": False,
                    },
                },
            }
        )

        # Create dummy dataset CSV
        self.dataset_csv = self.config.evaluation_dataset.input_csv_path
        pd.DataFrame({"image": ["im1.tif"], "mask": ["gt1.tif"]}).to_csv(
            self.dataset_csv, index=False
        )

        # Create exp folders
        os.makedirs(self.config.experiments[0].output_folder, exist_ok=True)

    def tearDown(self):
        shutil.rmtree(self.tmp_dir)

    @patch(
        "pytorch_segmentation_models_trainer.tools.evaluation.evaluation_pipeline.subprocess.run"
    )
    @patch(
        "pytorch_segmentation_models_trainer.tools.evaluation.evaluation_pipeline.MetricsCalculator"
    )
    @patch(
        "pytorch_segmentation_models_trainer.tools.evaluation.evaluation_pipeline.ResultsAggregator"
    )
    def test_run_basic_sequential(self, MockAggregator, MockMetricsCalc, mock_subproc):
        # Setup mocks
        mock_subproc.return_value = MagicMock(returncode=0)

        mock_metrics_calc_instance = MagicMock()
        MockMetricsCalc.return_value = mock_metrics_calc_instance
        mock_metrics_calc_instance.calculate_metrics.return_value = {
            "aggregated": {"iou": 0.8},
            "per_image": pd.DataFrame({"image": ["im1.tif"], "iou": [0.8]}),
            "num_classes": 2,
            "class_names": ["bg", "fg"],
            "output_dir": "some_dir",
        }

        mock_aggregator_instance = MagicMock()
        MockAggregator.return_value = mock_aggregator_instance
        mock_aggregator_instance.aggregate.return_value = {
            "num_experiments": 1,
            "experiments": {
                "exp1": mock_metrics_calc_instance.calculate_metrics.return_value
            },
        }

        pipeline = EvaluationPipeline(self.config)
        results = pipeline.run()

        self.assertEqual(results["num_experiments"], 1)
        mock_subproc.assert_called_once()
        mock_metrics_calc_instance.calculate_metrics.assert_called_once()
        mock_aggregator_instance.aggregate.assert_called_once()

        # Check if summary report was saved
        summary_path = os.path.join(self.base_dir, "summary.json")
        self.assertTrue(os.path.exists(summary_path))

    @patch(
        "pytorch_segmentation_models_trainer.tools.evaluation.csv_builder.DatasetCSVBuilder"
    )
    def test_prepare_dataset_build_from_folders(self, MockCSVBuilder):
        self.config.evaluation_dataset.build_csv_from_folders.enabled = True
        mock_builder = MagicMock()
        MockCSVBuilder.return_value = mock_builder

        pipeline = EvaluationPipeline(self.config)
        csv_path = pipeline._prepare_dataset()

        expected_csv_path = os.path.join(self.base_dir, "generated_dataset.csv")
        self.assertEqual(csv_path, expected_csv_path)
        mock_builder.build_csv.assert_called_once_with(expected_csv_path)

    def test_should_skip_prediction(self):
        pipeline = EvaluationPipeline(self.config)
        exp = self.config.experiments[0]

        # Should not skip if disabled in config
        self.config.pipeline_options.skip_existing_predictions = False
        self.assertFalse(pipeline._should_skip_prediction(exp))

        # Should skip if enabled and files exist
        self.config.pipeline_options.skip_existing_predictions = True
        Path(os.path.join(exp.output_folder, "pred.tif")).touch()
        self.assertTrue(pipeline._should_skip_prediction(exp))

    @patch("os.path.exists", return_value=True)
    @patch("os.path.isdir", return_value=True)
    def test_validate_predictions_folder(self, mock_isdir, mock_exists):
        pipeline = EvaluationPipeline(self.config)

        # Mock Path.glob to return files only for one pattern to keep count simple
        with patch(
            "pytorch_segmentation_models_trainer.tools.evaluation.evaluation_pipeline.Path.glob"
        ) as mock_glob:
            mock_glob.side_effect = [
                [Path("file1.tif")],
                [],
                [],
                [],
            ]  # 4 patterns: *.tif, *.tiff, *.TIF, *.TIFF
            with patch("os.path.getsize", return_value=100):
                result = pipeline._validate_predictions_folder("some_folder", "exp1")
                self.assertTrue(result["valid"])
                self.assertEqual(result["num_files"], 1)

    def test_load_existing_predictions(self):
        self.config.pipeline_options.load_predictions_from_folder.enabled = True
        self.config.pipeline_options.load_predictions_from_folder.base_folder = (
            self.base_dir
        )

        # Create subfolder for exp1 with a tif file
        exp1_pred_dir = os.path.join(self.base_dir, "exp1")
        os.makedirs(exp1_pred_dir, exist_ok=True)
        Path(os.path.join(exp1_pred_dir, "test.tif")).write_text(
            "dummy"
        )  # Ensure size > 0

        pipeline = EvaluationPipeline(self.config)
        preds_info = pipeline._load_existing_predictions()

        self.assertIn("exp1", preds_info)
        self.assertTrue(preds_info["exp1"]["loaded"])
        self.assertEqual(preds_info["exp1"]["num_predictions"], 1)


class TestDatasetCSVBuilder(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.images_folder = Path(self.temp_dir) / "images"
        self.masks_folder = Path(self.temp_dir) / "masks"
        self.images_folder.mkdir(parents=True)
        self.masks_folder.mkdir(parents=True)

    def tearDown(self):
        shutil.rmtree(self.temp_dir)

    def _create_dummy_tif(self, path):
        import rasterio
        from rasterio.transform import from_bounds

        # Use np from module level
        data = np.zeros((1, 10, 10), dtype=np.uint8)
        transform = from_bounds(0, 0, 10, 10, 10, 10)
        with rasterio.open(
            path,
            "w",
            driver="GTiff",
            height=10,
            width=10,
            count=1,
            dtype=rasterio.uint8,
            transform=transform,
        ) as dst:
            dst.write(data[0], 1)

    def test_csv_builder_same_basename(self):
        from pytorch_segmentation_models_trainer.tools.evaluation.csv_builder import (
            DatasetCSVBuilder,
        )

        self._create_dummy_tif(self.images_folder / "1.tif")
        self._create_dummy_tif(self.masks_folder / "1.tif")

        config = OmegaConf.create(
            {
                "images_folder": str(self.images_folder),
                "masks_folder": str(self.masks_folder),
                "image_pattern": "*.tif",
                "mask_pattern": "*.tif",
                "matching_strategy": "same_basename",
            }
        )
        builder = DatasetCSVBuilder(config)
        csv_path = Path(self.temp_dir) / "out.csv"
        df = builder.build_csv(str(csv_path))
        self.assertEqual(len(df), 1)


if __name__ == "__main__":
    unittest.main()
