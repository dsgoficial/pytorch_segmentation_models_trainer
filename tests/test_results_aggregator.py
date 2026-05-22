# -*- coding: utf-8 -*-
"""
Unit tests for results_aggregator.py
"""

import unittest
import os
import json
import pandas as pd
import numpy as np
from omegaconf import OmegaConf, DictConfig
import shutil

from pytorch_segmentation_models_trainer.tools.evaluation.results_aggregator import (
    ResultsAggregator,
)


class TestResultsAggregator(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = "temp_aggregator_test_dir"
        os.makedirs(self.tmp_dir, exist_ok=True)

        self.output_json_path = os.path.join(self.tmp_dir, "overall_metrics.json")
        self.output_csv_path = os.path.join(self.tmp_dir, "overall_metrics.csv")

        self.base_config = OmegaConf.create(
            {
                "output": {
                    "base_dir": self.tmp_dir,
                    "structure": {"reports_folder": "reports"},
                },
                "output_folder": self.tmp_dir,
                "output_metrics_json": self.output_json_path,
                "output_metrics_csv": self.output_csv_path,
                "experiments": [
                    {
                        "name": "exp1",
                        "output_folder": os.path.join(self.tmp_dir, "exp1_out"),
                    },
                    {
                        "name": "exp2",
                        "output_folder": os.path.join(self.tmp_dir, "exp2_out"),
                    },
                ],
            }
        )

        # Create dummy experiment output folders
        for exp in self.base_config.experiments:
            os.makedirs(exp.output_folder, exist_ok=True)

    def tearDown(self):
        if os.path.exists(self.tmp_dir):
            shutil.rmtree(self.tmp_dir)

    def _create_dummy_metrics_json(self, exp_output_folder, exp_name):
        metrics_data = {
            "overall": {"iou": 0.8, "f1": 0.85, "accuracy": 0.9},
            "per_image": {
                "image1.tif": {"iou": 0.75, "f1": 0.8, "accuracy": 0.88},
                "image2.tif": {"iou": 0.85, "f1": 0.9, "accuracy": 0.92},
            },
        }
        json_path = os.path.join(exp_output_folder, f"{exp_name}_metrics.json")
        with open(json_path, "w") as f:
            json.dump(metrics_data, f)
        return json_path

    def test_aggregate_results_success(self):
        self._create_dummy_metrics_json(
            self.base_config.experiments[0].output_folder, "exp1"
        )
        self._create_dummy_metrics_json(
            self.base_config.experiments[1].output_folder, "exp2"
        )

        aggregator = ResultsAggregator(self.base_config)
        aggregated_metrics = aggregator.aggregate_results()

        self.assertIn("exp1", aggregated_metrics)
        self.assertIn("exp2", aggregated_metrics)
        self.assertTrue(os.path.exists(self.output_json_path))
        self.assertTrue(os.path.exists(self.output_csv_path))

    def test_aggregate_results_missing_json_file(self):
        self._create_dummy_metrics_json(
            self.base_config.experiments[0].output_folder, "exp1"
        )
        aggregator = ResultsAggregator(self.base_config)
        aggregated_metrics = aggregator.aggregate_results()
        self.assertIn("exp1", aggregated_metrics)
        self.assertNotIn("exp2", aggregated_metrics)

    def test_aggregate_results_invalid_empty_and_aggregated_output(self):
        invalid_path = os.path.join(
            self.base_config.experiments[0].output_folder, "exp1_metrics.json"
        )
        with open(invalid_path, "w") as f:
            f.write("{invalid")
        empty_path = os.path.join(
            self.base_config.experiments[1].output_folder, "exp2_metrics.json"
        )
        with open(empty_path, "w") as f:
            json.dump({}, f)

        aggregator = ResultsAggregator(self.base_config)
        self.assertEqual(aggregator.aggregate_results(), {})

        with open(invalid_path, "w") as f:
            json.dump({"aggregated": {"iou": 0.6}}, f)
        os.remove(empty_path)
        aggregated = aggregator.aggregate_results()
        self.assertIn("exp1", aggregated)
        self.assertTrue(os.path.exists(self.output_csv_path))

    def test_get_metrics_file_path(self):
        aggregator = ResultsAggregator(self.base_config)
        path = aggregator._get_metrics_file_path(self.base_config.experiments[0])
        self.assertEqual(
            path, os.path.join(self.tmp_dir, "exp1_out", "exp1_metrics.json")
        )
        custom_exp = OmegaConf.create(
            {
                "name": "custom",
                "output_folder": self.tmp_dir,
                "output_metrics_filename": "custom.json",
            }
        )
        self.assertEqual(
            aggregator._get_metrics_file_path(custom_exp),
            os.path.join(self.tmp_dir, "custom.json"),
        )

    def test_aggregate_method(self):
        all_results = {
            "exp1": {
                "aggregated": {"iou": 0.8, "f1": 0.85},
                "per_image": pd.DataFrame(
                    {"image": ["im1", "im2"], "iou": [0.75, 0.85], "f1": [0.8, 0.9]}
                ),
                "num_classes": 2,
            },
            "exp2": {
                "aggregated": {"iou": 0.7, "f1": 0.75},
                "per_image": pd.DataFrame(
                    {"image": ["im1", "im2"], "iou": [0.65, 0.75], "f1": [0.7, 0.8]}
                ),
                "num_classes": 2,
            },
        }
        aggregator = ResultsAggregator(self.base_config)
        aggregated = aggregator.aggregate(all_results)

        self.assertEqual(aggregated["num_experiments"], 2)
        self.assertIn("exp1", aggregated["experiments"])
        self.assertIn("rankings", aggregated)
        self.assertIn("statistics", aggregated)

        # Check rankings
        self.assertEqual(aggregated["rankings"]["iou"][0]["experiment"], "exp1")
        self.assertEqual(aggregated["rankings"]["iou"][1]["experiment"], "exp2")

    def test_aggregate_empty_and_none_results(self):
        aggregator = ResultsAggregator(self.base_config)
        empty = aggregator.aggregate({})
        self.assertEqual(empty["num_experiments"], 0)

        empty = aggregator.aggregate({"exp1": None})
        self.assertEqual(empty["num_experiments"], 0)

    def test_rankings_empty_missing_aggregated_and_none_values(self):
        aggregator = ResultsAggregator(self.base_config)
        self.assertEqual(aggregator._create_rankings({}), {})
        self.assertEqual(aggregator._create_rankings({"exp": {"per_image": []}}), {})

        rankings = aggregator._create_rankings(
            {
                "exp1": {"aggregated": {"iou": None, "f1": 0.8}},
                "exp2": {"aggregated": {"iou": 0.7, "f1": 0.7}},
            }
        )
        self.assertEqual(rankings["iou"][0]["experiment"], "exp2")

    def test_calculate_statistics_skips_empty_numeric_values(self):
        aggregator = ResultsAggregator(self.base_config)
        stats = aggregator._calculate_statistics(
            {
                "exp": {
                    "per_image": pd.DataFrame(
                        {"image": ["a", "b"], "iou": [np.nan, np.nan]}
                    )
                }
            }
        )
        self.assertEqual(stats["exp"], {})

    def test_create_summary(self):
        all_results = {
            "exp1": {
                "aggregated": {"iou": 0.8},
                "per_image": [{"image": "im1"}, {"image": "im2"}],
                "num_classes": 2,
                "class_names": ["bg", "fg"],
            }
        }
        aggregator = ResultsAggregator(self.base_config)
        summary = aggregator._create_summary(all_results)
        self.assertEqual(summary["num_experiments"], 1)
        self.assertEqual(summary["total_images_evaluated"], 2)
        self.assertIn("exp1", summary["experiments_summary"])

    def test_find_best_worst_experiment(self):
        all_results = {
            "exp1": {"aggregated": {"iou": 0.8}},
            "exp2": {"aggregated": {"iou": 0.6}},
        }
        aggregator = ResultsAggregator(self.base_config)
        best = aggregator._find_best_experiment(all_results, metric="iou")
        worst = aggregator._find_worst_experiment(all_results, metric="iou")

        self.assertEqual(best["name"], "exp1")
        self.assertEqual(worst["name"], "exp2")
        self.assertEqual(aggregator._find_best_experiment({}, metric="iou"), {})
        self.assertEqual(aggregator._find_worst_experiment({}, metric="iou"), {})

    def test_save_helpers_cover_empty_invalid_rankings_and_statistics(self):
        aggregator = ResultsAggregator(self.base_config)
        aggregator._save_aggregated_csv({})
        self.assertTrue(
            os.path.exists(os.path.join(self.tmp_dir, "aggregated_metrics.csv"))
        )

        aggregator._save_aggregated_csv({"bad": None})
        aggregator._save_rankings_csv(
            {"iou": [{"rank": 1, "experiment": "exp1", "value": 1.0}]}
        )
        aggregator._save_statistics_json({"exp1": {"iou": {"mean": 1.0}}})

        reports_dir = os.path.join(self.tmp_dir, "reports")
        self.assertTrue(os.path.exists(os.path.join(reports_dir, "ranking_iou.csv")))
        self.assertTrue(os.path.exists(os.path.join(reports_dir, "statistics.json")))

    def test_create_comparison_table(self):
        all_results = {
            "exp1": {"aggregated": {"iou": 0.8, "f1": 0.85}},
            "exp2": {"aggregated": {"iou": 0.7, "f1": 0.75}},
        }
        aggregator = ResultsAggregator(self.base_config)
        table = aggregator.create_comparison_table(all_results, metrics=["iou", "f1"])
        self.assertEqual(len(table), 2)
        self.assertIn("iou", table.columns)
        self.assertEqual(table[table["Experiment"] == "exp1"]["iou"].iloc[0], "0.8000")

        table = aggregator.create_comparison_table(
            {"exp1": {"aggregated": {"iou_class_0": 0.1, "iou_class_1": 0.2}}},
            metrics=None,
        )
        self.assertIn("iou_class_1", table.columns)
        self.assertNotIn("iou_class_0", table.columns)

    def test_get_improvement_matrix(self):
        all_results = {
            "baseline": {"aggregated": {"iou": 0.5}},
            "exp1": {"aggregated": {"iou": 0.75}},
        }
        aggregator = ResultsAggregator(self.base_config)
        matrix = aggregator.get_improvement_matrix(
            all_results, baseline_exp="baseline", metric="iou"
        )

        self.assertEqual(matrix["exp1"]["absolute"], 0.25)
        self.assertEqual(matrix["exp1"]["relative"], 0.5)
        self.assertEqual(matrix["exp1"]["percentage"], 50.0)

        self.assertEqual(
            aggregator.get_improvement_matrix(
                all_results, baseline_exp="missing", metric="iou"
            ),
            {},
        )
        zero_matrix = aggregator.get_improvement_matrix(
            {
                "baseline": {"aggregated": {"iou": 0.0}},
                "exp1": {"aggregated": {"iou": 0.2}},
            },
            baseline_exp="baseline",
            metric="iou",
        )
        self.assertEqual(zero_matrix["exp1"]["relative"], 0)


if __name__ == "__main__":
    unittest.main()
