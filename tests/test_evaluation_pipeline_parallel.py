# -*- coding: utf-8 -*-
import unittest
from unittest.mock import MagicMock, patch
import os
from omegaconf import OmegaConf
from pytorch_segmentation_models_trainer.tools.evaluation.evaluation_pipeline import (
    EvaluationPipeline,
)


class TestEvaluationPipelineParallel(unittest.TestCase):
    def setUp(self):
        self.config = OmegaConf.create(
            {
                "experiments": [
                    {
                        "name": "exp1",
                        "output_folder": "/tmp/exp1",
                        "checkpoint_path": "c",
                        "predict_config": "p",
                    }
                ],
                "pipeline_options": {
                    "parallel_inference": {
                        "enabled": True,
                        "strategy": "parallel",
                        "gpus": None,
                    },
                    "skip_existing_predictions": False,
                    "skip_existing_evaluations": False,
                    "load_predictions_from_folder": {"enabled": False},
                },
                "metrics": {"confusion_matrix": {"enabled": True}},
                "output": {
                    "base_dir": "/tmp",
                    "structure": {"comparison_folder": "comp"},
                },
                "visualization": {
                    "comparison_plots": {"enabled": True},
                    "confusion_matrix": {
                        "save_individual": True,
                        "save_comparison": True,
                    },
                },
            }
        )

    @patch(
        "pytorch_segmentation_models_trainer.tools.evaluation.gpu_distributor.GPUDistributor"
    )
    @patch(
        "pytorch_segmentation_models_trainer.tools.evaluation.results_aggregator.ResultsAggregator"
    )
    def test_init_with_gpu_distributor(self, mock_agg, mock_gpu):
        pipeline = EvaluationPipeline(self.config)
        self.assertIsNotNone(pipeline.gpu_distributor)

    def test_run_predictions_parallel_flow(self):
        # We test the routing to parallel
        pipeline = EvaluationPipeline(self.config)
        pipeline._run_predictions_parallel = MagicMock(return_value={"exp1": "done"})

        res = pipeline._run_predictions("csv")
        self.assertEqual(res["exp1"], "done")

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
        self.assertTrue(pipeline.confusion_matrix_plotter.plot_comparison_grid.called)


if __name__ == "__main__":
    unittest.main()
