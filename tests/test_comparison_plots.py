# -*- coding: utf-8 -*-
"""
Unit tests for comparison_plots.py
"""

import unittest
import os
import shutil
import tempfile
import pandas as pd
import numpy as np
from omegaconf import OmegaConf

from pytorch_segmentation_models_trainer.tools.visualization.comparison_plots import (
    ComparisonPlotter,
)


class TestComparisonPlotter(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()

        self.config = OmegaConf.create(
            {
                "visualization": {
                    "comparison_plots": {
                        "dpi": 100,
                        "save_format": "png",
                        "metrics_to_compare": ["iou", "f1"],
                        "style": "seaborn-v0_8-darkgrid",
                    }
                }
            }
        )

        # Mock experiment data
        self.exp_data = {
            "exp1": {
                "aggregated": {"iou": 0.8, "f1": 0.85, "iou_bg": 0.9, "iou_fg": 0.7},
                "per_image": pd.DataFrame(
                    {
                        "image": ["img1.tif", "img2.tif"],
                        "iou": [0.75, 0.85],
                        "f1": [0.8, 0.9],
                    }
                ),
                "class_names": ["bg", "fg"],
            },
            "exp2": {
                "aggregated": {"iou": 0.7, "f1": 0.75, "iou_bg": 0.8, "iou_fg": 0.6},
                "per_image": pd.DataFrame(
                    {
                        "image": ["img1.tif", "img2.tif"],
                        "iou": [0.65, 0.75],
                        "f1": [0.7, 0.8],
                    }
                ),
                "class_names": ["bg", "fg"],
            },
        }

    def tearDown(self):
        shutil.rmtree(self.tmp_dir)

    def test_plot_metrics_bar_chart(self):
        plotter = ComparisonPlotter(self.config)
        out_path = plotter.plot_metrics_bar_chart(self.exp_data, self.tmp_dir)
        self.assertTrue(os.path.exists(out_path))

    def test_plot_per_image_boxplot(self):
        plotter = ComparisonPlotter(self.config)
        out_path = plotter.plot_per_image_boxplot(
            self.exp_data, self.tmp_dir, metric="iou"
        )
        self.assertTrue(os.path.exists(out_path))

    def test_plot_radar_chart(self):
        # Radar chart needs at least 3 metrics
        self.config.visualization.comparison_plots.metrics_to_compare = [
            "iou",
            "f1",
            "acc",
        ]
        # Add acc to data
        for exp in self.exp_data.values():
            exp["aggregated"]["acc"] = 0.9

        plotter = ComparisonPlotter(self.config)
        out_path = plotter.plot_radar_chart(self.exp_data, self.tmp_dir)
        self.assertTrue(os.path.exists(out_path))

    def test_plot_best_worst_images(self):
        plotter = ComparisonPlotter(self.config)
        out_paths = plotter.plot_best_worst_images(
            self.exp_data, self.tmp_dir, metric="iou", n_images=1
        )
        self.assertIn("exp1", out_paths)
        self.assertTrue(os.path.exists(out_paths["exp1"]))

    def test_plot_metrics_distribution(self):
        plotter = ComparisonPlotter(self.config)
        out_path = plotter.plot_metrics_distribution(
            self.exp_data, self.tmp_dir, metric="iou"
        )
        self.assertTrue(os.path.exists(out_path))

    def test_plot_per_class_comparison(self):
        plotter = ComparisonPlotter(self.config)
        out_path = plotter.plot_per_class_comparison(
            self.exp_data, self.tmp_dir, metric_prefix="iou"
        )
        self.assertTrue(os.path.exists(out_path))


if __name__ == "__main__":
    unittest.main()
