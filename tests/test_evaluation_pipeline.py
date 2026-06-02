# -*- coding: utf-8 -*-
"""
Unit tests for evaluation_pipeline.py
"""

import unittest
from unittest.mock import MagicMock, patch
import os
import shutil
import tempfile
from omegaconf import OmegaConf

from pytorch_segmentation_models_trainer.tools.evaluation.evaluation_pipeline import (
    EvaluationPipeline,
    _evaluate_experiment_worker,
    _run_prediction_worker,
)


class TestEvaluationPipeline(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()
        self.config = OmegaConf.create(
            {
                "experiments": [
                    {
                        "name": "exp1",
                        "predict_config": "config1.yaml",
                        "checkpoint_path": "model1.ckpt",
                        "output_folder": os.path.join(self.tmp_dir, "exp1"),
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
                    "input_csv_path": os.path.join(self.tmp_dir, "input.csv"),
                    "build_csv_from_folders": {"enabled": False},
                },
                "metrics": {"confusion_matrix": {"enabled": True}},
                "output": {
                    "base_dir": self.tmp_dir,
                    "files": {"summary_report": "summary.json"},
                    "structure": {"comparison_folder": "comparisons"},
                },
                "visualization": {
                    "comparison_plots": {"enabled": False},
                    "confusion_matrix": {
                        "save_individual": True,
                        "save_comparison": False,
                    },
                },
            }
        )

    def tearDown(self):
        shutil.rmtree(self.tmp_dir)

    def test_pipeline_init(self):
        pipeline = EvaluationPipeline(self.config)
        self.assertEqual(len(pipeline.experiments), 1)

        self.config.pipeline_options.parallel_inference.enabled = True
        self.config.visualization.comparison_plots.enabled = True
        with (
            patch(
                "pytorch_segmentation_models_trainer.tools.evaluation.evaluation_pipeline.GPUDistributor"
            ),
            patch(
                "pytorch_segmentation_models_trainer.tools.evaluation.evaluation_pipeline.ConfusionMatrixPlotter"
            ),
            patch(
                "pytorch_segmentation_models_trainer.tools.evaluation.evaluation_pipeline.ComparisonPlotter"
            ),
        ):
            pipeline = EvaluationPipeline(self.config)
        self.assertIsNotNone(pipeline.gpu_distributor)
        self.assertIsNotNone(pipeline.confusion_matrix_plotter)

    @patch("os.path.exists", return_value=True)
    def test_prepare_dataset_existing_csv(self, mock_exists):
        pipeline = EvaluationPipeline(self.config)
        csv_path = pipeline._prepare_dataset()
        self.assertEqual(csv_path, self.config.evaluation_dataset.input_csv_path)

    def test_prepare_dataset_missing_existing_csv_raises(self):
        pipeline = EvaluationPipeline(self.config)
        with self.assertRaises(FileNotFoundError):
            pipeline._prepare_dataset()

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

        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        with open(csv_path, "w") as f:
            f.write("image,mask\n")
        mock_direct.reset_mock()
        csv_path = pipeline._prepare_dataset()
        self.assertIn("direct_evaluation_dataset.csv", csv_path)
        mock_direct.assert_not_called()

    def test_validate_predictions_folder(self):
        pipeline = EvaluationPipeline(self.config)
        self.assertFalse(
            pipeline._validate_predictions_folder(
                os.path.join(self.tmp_dir, "missing"), "exp1"
            )["valid"]
        )
        file_path = os.path.join(self.tmp_dir, "not_dir.tif")
        with open(file_path, "wb") as f:
            f.write(b"x")
        self.assertFalse(
            pipeline._validate_predictions_folder(file_path, "exp1")["valid"]
        )

        folder = os.path.join(self.tmp_dir, "valid_folder")
        os.makedirs(folder, exist_ok=True)

        # Empty folder
        res = pipeline._validate_predictions_folder(folder, "exp1")
        self.assertFalse(res["valid"])

        empty_tif = os.path.join(folder, "empty.tif")
        open(empty_tif, "wb").close()
        res = pipeline._validate_predictions_folder(folder, "exp1")
        self.assertFalse(res["valid"])
        self.assertIn("empty", res["error"])

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

    @patch(
        "pytorch_segmentation_models_trainer.tools.evaluation.evaluation_pipeline.EvaluationPipeline._validate_predictions_folder"
    )
    def test_load_existing_predictions_variants_and_invalid(self, mock_val):
        self.config.experiments = [
            {
                "name": "exp_specific",
                "predict_config": "config.yaml",
                "checkpoint_path": "model.ckpt",
                "output_folder": os.path.join(self.tmp_dir, "default"),
                "precomputed_predictions_folder": os.path.join(
                    self.tmp_dir, "specific"
                ),
            },
            {
                "name": "exp_base",
                "predict_config": "config.yaml",
                "checkpoint_path": "model.ckpt",
                "output_folder": os.path.join(self.tmp_dir, "default2"),
            },
        ]
        self.config.pipeline_options.load_predictions_from_folder.enabled = True
        self.config.pipeline_options.load_predictions_from_folder.base_folder = (
            os.path.join(self.tmp_dir, "base")
        )
        mock_val.side_effect = [
            {"valid": False, "num_files": 0, "error": "bad folder"},
            {"valid": True, "num_files": 3, "error": None},
        ]

        pipeline = EvaluationPipeline(self.config)
        info = pipeline._load_existing_predictions()

        self.assertFalse(info["exp_specific"]["loaded"])
        self.assertTrue(info["exp_base"]["loaded"])

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

        self.config.experiments[0].overrides = {"trainer.max_epochs": 1}
        pipeline._run_single_prediction(self.config.experiments[0], "csv", gpu_id=0)
        args = mock_run.call_args[0][0]
        self.assertIn("device=cuda:0", args)
        self.assertIn("trainer.max_epochs=1", args)

    @patch("os.path.exists", return_value=False)
    def test_run_single_prediction_missing_script(self, mock_exists):
        pipeline = EvaluationPipeline(self.config)
        with self.assertRaises(FileNotFoundError):
            pipeline._run_single_prediction(
                self.config.experiments[0], "csv", gpu_id=-1
            )

    @patch("subprocess.run")
    @patch("os.path.exists", return_value=True)
    def test_run_single_prediction_failure(self, mock_exists, mock_run):
        mock_run.return_value = MagicMock(returncode=1, stdout="out", stderr="err")
        pipeline = EvaluationPipeline(self.config)
        with self.assertRaises(RuntimeError):
            pipeline._run_single_prediction(
                self.config.experiments[0], "csv", gpu_id=-1
            )

    @patch(
        "pytorch_segmentation_models_trainer.tools.evaluation.metrics_calculator.MetricsCalculator"
    )
    @patch(
        "pytorch_segmentation_models_trainer.tools.evaluation.results_aggregator.ResultsAggregator.aggregate"
    )
    def test_full_run_flow(self, mock_agg, mock_calc):
        pipeline = EvaluationPipeline(self.config)
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
        self.assertTrue(os.path.exists(os.path.join(self.tmp_dir, "summary.json")))

    def test_run_failure_and_visualization_enabled(self):
        self.config.visualization.comparison_plots.enabled = True
        with (
            patch(
                "pytorch_segmentation_models_trainer.tools.evaluation.evaluation_pipeline.ConfusionMatrixPlotter"
            ),
            patch(
                "pytorch_segmentation_models_trainer.tools.evaluation.evaluation_pipeline.ComparisonPlotter"
            ),
        ):
            pipeline = EvaluationPipeline(self.config)
        pipeline._prepare_dataset = MagicMock(side_effect=RuntimeError("boom"))
        with self.assertRaises(RuntimeError):
            pipeline.run()

        pipeline._prepare_dataset = MagicMock(return_value="dataset.csv")
        pipeline._run_predictions = MagicMock(return_value={})
        pipeline._evaluate_all_experiments = MagicMock(return_value={})
        pipeline.results_aggregator.aggregate = MagicMock(
            return_value={"num_experiments": 0, "experiments": {}}
        )
        pipeline._generate_visualizations = MagicMock()
        pipeline._save_summary_report = MagicMock()
        result = pipeline.run()
        self.assertEqual(result["num_experiments"], 0)
        pipeline._generate_visualizations.assert_called_once()

    def test_run_predictions_parallel_flow(self):
        self.config.pipeline_options.parallel_inference.enabled = True
        self.config.pipeline_options.parallel_inference.strategy = "parallel"
        pipeline = EvaluationPipeline(self.config)
        pipeline._run_predictions_parallel = MagicMock(return_value={"exp1": "done"})

        res = pipeline._run_predictions("csv")
        self.assertEqual(res["exp1"], "done")

        self.config.pipeline_options.load_predictions_from_folder.enabled = True
        pipeline._load_existing_predictions = MagicMock(return_value={"exp1": "loaded"})
        res = pipeline._run_predictions("csv")
        self.assertEqual(res["exp1"], "loaded")

        self.config.pipeline_options.load_predictions_from_folder.enabled = False
        self.config.pipeline_options.parallel_inference.enabled = False
        pipeline._run_predictions_sequential = MagicMock(return_value={"exp1": "seq"})
        res = pipeline._run_predictions("csv")
        self.assertEqual(res["exp1"], "seq")

    def test_run_predictions_sequential_skip_success_gpu_and_error(self):
        pipeline = EvaluationPipeline(self.config)
        pipeline.gpu_distributor = MagicMock(available_gpus=[2])
        pipeline._should_skip_prediction = MagicMock(return_value=True)
        info = pipeline._run_predictions_sequential("dataset.csv")
        self.assertTrue(info["exp1"]["skipped"])
        self.assertEqual(info["exp1"]["gpu_id"], 2)

        pipeline._should_skip_prediction = MagicMock(return_value=False)
        pipeline._run_single_prediction = MagicMock()
        info = pipeline._run_predictions_sequential("dataset.csv")
        self.assertFalse(info["exp1"]["skipped"])

        pipeline.gpu_distributor = None
        pipeline._run_single_prediction = MagicMock()
        info = pipeline._run_predictions_sequential("dataset.csv")
        self.assertEqual(info["exp1"]["gpu_id"], -1)

        pipeline._run_single_prediction = MagicMock(side_effect=RuntimeError("failed"))
        with self.assertRaises(RuntimeError):
            pipeline._run_predictions_sequential("dataset.csv")

    def test_run_predictions_parallel_skipped_success_and_failure(self):
        self.config.pipeline_options.parallel_inference.enabled = True
        pipeline = EvaluationPipeline(self.config)
        pipeline.gpu_distributor = MagicMock()
        pipeline.gpu_distributor.assign_experiments.return_value = {
            0: self.config.experiments
        }
        pipeline._should_skip_prediction = MagicMock(return_value=True)
        skipped = pipeline._run_predictions_parallel("dataset.csv")
        self.assertTrue(skipped["exp1"]["skipped"])

        class GoodFuture:
            def result(self):
                return {"output_folder": "out", "skipped": False, "gpu_id": 0}

        class BadFuture:
            def result(self):
                raise RuntimeError("worker failed")

        class DummyExecutor:
            def __init__(self, max_workers):
                pass

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

            def submit(self, fn, *args):
                return GoodFuture() if args[0].name == "exp1" else BadFuture()

        self.config.experiments.append(
            {
                "name": "exp2",
                "predict_config": "config2.yaml",
                "checkpoint_path": "model2.ckpt",
                "output_folder": os.path.join(self.tmp_dir, "exp2"),
            }
        )
        pipeline = EvaluationPipeline(self.config)
        pipeline.gpu_distributor = MagicMock()
        pipeline.gpu_distributor.assign_experiments.return_value = {
            0: self.config.experiments
        }
        pipeline._should_skip_prediction = MagicMock(
            side_effect=lambda exp: exp.name == "exp2"
        )
        with (
            patch(
                "pytorch_segmentation_models_trainer.tools.evaluation.evaluation_pipeline.ThreadPoolExecutor",
                DummyExecutor,
            ),
            patch(
                "pytorch_segmentation_models_trainer.tools.evaluation.evaluation_pipeline.as_completed",
                lambda futures: list(futures),
            ),
        ):
            info = pipeline._run_predictions_parallel("dataset.csv")
        self.assertTrue(info["exp2"]["skipped"])

        pipeline._should_skip_prediction = MagicMock(return_value=False)
        with (
            patch(
                "pytorch_segmentation_models_trainer.tools.evaluation.evaluation_pipeline.ThreadPoolExecutor",
                DummyExecutor,
            ),
            patch(
                "pytorch_segmentation_models_trainer.tools.evaluation.evaluation_pipeline.as_completed",
                lambda futures: list(futures),
            ),
        ):
            info = pipeline._run_predictions_parallel("dataset.csv")
        self.assertIn("error", info["exp2"])

    def test_should_skip_prediction_and_evaluation(self):
        pipeline = EvaluationPipeline(self.config)
        self.assertFalse(pipeline._should_skip_prediction(self.config.experiments[0]))
        self.config.pipeline_options.skip_existing_predictions = True
        self.assertFalse(pipeline._should_skip_prediction(self.config.experiments[0]))
        os.makedirs(self.config.experiments[0].output_folder, exist_ok=True)
        self.assertFalse(pipeline._should_skip_prediction(self.config.experiments[0]))
        with open(
            os.path.join(self.config.experiments[0].output_folder, "pred.tif"), "wb"
        ) as f:
            f.write(b"x")
        self.assertTrue(pipeline._should_skip_prediction(self.config.experiments[0]))

        self.assertFalse(pipeline._should_skip_evaluation("exp1"))
        self.config.pipeline_options.skip_existing_evaluations = True
        self.assertFalse(pipeline._should_skip_evaluation("exp1"))

    @patch(
        "pytorch_segmentation_models_trainer.tools.evaluation.evaluation_pipeline.MetricsCalculator"
    )
    def test_evaluate_all_experiments_paths(self, mock_metrics):
        pipeline = EvaluationPipeline(self.config)
        mock_metrics.return_value.calculate_metrics.return_value = {"ok": True}

        results = pipeline._evaluate_all_experiments(
            {"exp1": {"output_folder": "preds", "loaded": True, "num_predictions": 2}},
            "dataset.csv",
        )
        self.assertEqual(results["exp1"], {"ok": True})

        pipeline._should_skip_evaluation = MagicMock(return_value=True)
        results = pipeline._evaluate_all_experiments_sequential(
            {"exp1": {"output_folder": "preds"}},
            "dataset.csv",
        )
        self.assertEqual(results, {})

        pipeline._should_skip_evaluation = MagicMock(return_value=False)
        results = pipeline._evaluate_all_experiments_sequential(
            {
                "failed": {"error": "bad"},
                "missing": {"output_folder": "preds"},
                "exp1": {"output_folder": "preds"},
            },
            "dataset.csv",
        )
        self.assertEqual(results["exp1"], {"ok": True})

        mock_metrics.return_value.calculate_metrics.side_effect = RuntimeError(
            "calc failed"
        )
        results = pipeline._evaluate_all_experiments_sequential(
            {"exp1": {"output_folder": "preds"}},
            "dataset.csv",
        )
        self.assertEqual(results, {})

    def test_generate_visualizations(self):
        pipeline = EvaluationPipeline(self.config)
        pipeline.confusion_matrix_plotter = MagicMock()

        aggregated = {
            "experiments": {
                "exp1": {
                    "confusion_matrix": [[1, 0], [0, 1]],
                    "class_names": ["a", "b"],
                }
            }
        }
        pipeline._generate_visualizations(aggregated)
        self.assertTrue(pipeline.confusion_matrix_plotter.plot_single_experiment.called)

        self.config.visualization.confusion_matrix.save_comparison = True
        pipeline.confusion_matrix_plotter = MagicMock()
        pipeline._generate_visualizations(aggregated)
        self.assertTrue(pipeline.confusion_matrix_plotter.plot_comparison_grid.called)

        pipeline._generate_visualizations({"experiments": {}})

    def test_save_summary_report_logs_non_float_metrics(self):
        pipeline = EvaluationPipeline(self.config)
        aggregated = {
            "num_experiments": 1,
            "experiments": {
                "exp1": {
                    "num_classes": 2,
                    "class_names": ["a", "b"],
                    "per_image": [1, 2],
                    "output_dir": "out",
                    "aggregated": {"iou": 0.5, "label": "ok"},
                }
            },
        }
        pipeline._save_summary_report(aggregated, 60.0)
        self.assertTrue(os.path.exists(os.path.join(self.tmp_dir, "summary.json")))

    @patch("subprocess.run")
    def test_prediction_worker_success_and_failure(self, mock_run):
        exp = self.config.experiments[0]
        exp.overrides = {"x": 1}
        mock_run.return_value = MagicMock(returncode=0, stderr="")
        output = _run_prediction_worker(exp, 0, "dataset.csv", self.config)
        self.assertEqual(output["gpu_id"], 0)
        self.assertEqual(os.environ["CUDA_VISIBLE_DEVICES"], "0")

        mock_run.return_value = MagicMock(returncode=1, stderr="bad")
        with self.assertRaises(RuntimeError):
            _run_prediction_worker(exp, -1, "dataset.csv", self.config)

    @patch(
        "pytorch_segmentation_models_trainer.tools.evaluation.metrics_calculator.MetricsCalculator"
    )
    def test_evaluate_experiment_worker(self, mock_metrics):
        mock_metrics.return_value.calculate_metrics.return_value = {"metric": 1}
        name, results = _evaluate_experiment_worker(
            self.config.experiments[0],
            "preds",
            "dataset.csv",
            self.config,
        )
        self.assertEqual(name, "exp1")
        self.assertEqual(results, {"metric": 1})


if __name__ == "__main__":
    unittest.main()
