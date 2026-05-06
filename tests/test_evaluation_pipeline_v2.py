# -*- coding: utf-8 -*-
import unittest
from unittest.mock import MagicMock, patch, patch
import os
import shutil
from pathlib import Path
from omegaconf import OmegaConf
from pytorch_segmentation_models_trainer.tools.evaluation.evaluation_pipeline import (
    EvaluationPipeline,
)


class TestEvaluationPipelineV2(unittest.TestCase):
    def setUp(self):
        self.test_dir = "/tmp/test_eval_pipeline"
        os.makedirs(self.test_dir, exist_ok=True)

        self.config = OmegaConf.create(
            {
                "experiments": [
                    {
                        "name": "exp1",
                        "predict_config": "p1.yaml",
                        "checkpoint_path": "c1.ckpt",
                        "output_folder": os.path.join(self.test_dir, "exp1"),
                    }
                ],
                "pipeline_options": {
                    "parallel_inference": {
                        "enabled": False,
                        "strategy": "sequential",
                        "gpus": None,
                    },
                    "skip_existing_predictions": False,
                    "skip_existing_evaluations": False,
                    "load_predictions_from_folder": {"enabled": False},
                },
                "evaluation_dataset": {
                    "input_csv_path": os.path.join(self.test_dir, "input.csv"),
                    "build_csv_from_folders": {"enabled": False},
                },
                "output": {
                    "base_dir": self.test_dir,
                    "files": {"summary_report": "summary.json"},
                    "structure": {"comparison_folder": "comparisons"},
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

    def tearDown(self):
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    @patch("os.path.exists", return_value=True)
    def test_prepare_dataset_existing_csv(self, mock_exists):
        pipeline = EvaluationPipeline(self.config)
        csv_path = pipeline._prepare_dataset()
        self.assertEqual(csv_path, self.config.evaluation_dataset.input_csv_path)

    @patch(
        "pytorch_segmentation_models_trainer.tools.evaluation.csv_builder.DatasetCSVBuilder"
    )
    def test_prepare_dataset_build_from_folders(self, mock_builder):
        self.config.evaluation_dataset.build_csv_from_folders.enabled = True
        pipeline = EvaluationPipeline(self.config)
        csv_path = pipeline._prepare_dataset()
        self.assertIn("generated_dataset.csv", csv_path)
        self.assertTrue(mock_builder.return_value.build_csv.called)

    @patch(
        "pytorch_segmentation_models_trainer.tools.evaluation.direct_folder_evaluator.prepare_evaluation_csv_from_folders"
    )
    def test_prepare_dataset_direct_folder(self, mock_direct):
        self.config.evaluation_dataset.direct_folder_evaluation = {
            "enabled": True,
            "ground_truth_folder": "/tmp/gt",
            "predictions_folder": "/tmp/pred",
        }
        pipeline = EvaluationPipeline(self.config)
        csv_path = pipeline._prepare_dataset()
        self.assertIn("direct_evaluation_dataset.csv", csv_path)
        self.assertTrue(mock_direct.called)

    def test_validate_predictions_folder(self):
        pipeline = EvaluationPipeline(self.config)
        folder = os.path.join(self.test_dir, "valid_folder")
        os.makedirs(folder, exist_ok=True)

        # Empty folder
        res = pipeline._validate_predictions_folder(folder, "exp1")
        self.assertFalse(res["valid"])

        # Folder with non-empty TIF
        tif_path = os.path.join(folder, "test.tif")
        with open(tif_path, "wb") as f:
            f.write(b"data")

        res = pipeline._validate_predictions_folder(folder, "exp1")
        self.assertTrue(res["valid"])
        self.assertEqual(res["num_files"], 1)

    @patch(
        "pytorch_segmentation_models_trainer.tools.evaluation.evaluation_pipeline.EvaluationPipeline._validate_predictions_folder"
    )
    def test_load_existing_predictions(self, mock_val):
        self.config.pipeline_options.load_predictions_from_folder.enabled = True
        mock_val.return_value = {"valid": True, "num_files": 10}

        pipeline = EvaluationPipeline(self.config)
        info = pipeline._load_existing_predictions()

        self.assertIn("exp1", info)
        self.assertTrue(info["exp1"]["loaded"])

    @patch("subprocess.run")
    @patch("os.path.exists", return_value=True)
    def test_run_single_prediction(self, mock_exists, mock_run):
        mock_run.return_value = MagicMock(returncode=0)
        pipeline = EvaluationPipeline(self.config)

        # Test CPU
        pipeline._run_single_prediction(self.config.experiments[0], "csv", gpu_id=-1)
        self.assertTrue(mock_run.called)
        args = mock_run.call_args[0][0]
        self.assertIn("device=cpu", args)

    @patch(
        "pytorch_segmentation_models_trainer.tools.evaluation.metrics_calculator.MetricsCalculator"
    )
    @patch(
        "pytorch_segmentation_models_trainer.tools.evaluation.results_aggregator.ResultsAggregator.aggregate"
    )
    def test_full_run_flow(self, mock_agg, mock_calc):
        # Setup mocks for a successful run
        pipeline = EvaluationPipeline(self.config)

        # Mocking sub-methods to avoid complex logic
        pipeline._prepare_dataset = MagicMock(return_value="test.csv")
        pipeline._run_predictions = MagicMock(
            return_value={
                "exp1": {"output_folder": "/tmp", "loaded": True, "num_predictions": 5}
            }
        )

        mock_calc.return_value.calculate_metrics.return_value = {
            "num_classes": 2,
            "class_names": ["a", "b"],
            "per_image": [1],
            "output_dir": "/tmp",
            "aggregated": {"iou": 0.8},
            "confusion_matrix": None,
        }
        mock_agg.return_value = {
            "num_experiments": 1,
            "experiments": {
                "exp1": {
                    "num_classes": 2,
                    "class_names": ["a", "b"],
                    "per_image": [1],
                    "output_dir": "/tmp",
                    "aggregated": {"iou": 0.8},
                    "confusion_matrix": None,
                }
            },
        }

        results = pipeline.run()
        self.assertEqual(results["num_experiments"], 1)
        self.assertTrue(os.path.exists(os.path.join(self.test_dir, "summary.json")))


if __name__ == "__main__":
    unittest.main()
