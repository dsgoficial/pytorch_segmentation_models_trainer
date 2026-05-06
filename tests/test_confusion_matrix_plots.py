# -*- coding: utf-8 -*-
"""
Unit tests for confusion_matrix_plots.py
"""

import unittest
import os
import shutil
import tempfile
import numpy as np
from omegaconf import OmegaConf

from pytorch_segmentation_models_trainer.tools.visualization.confusion_matrix_plots import (
    ConfusionMatrixPlotter,
)


class TestConfusionMatrixPlotter(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()

        self.config = OmegaConf.create(
            {
                "metrics": {
                    "confusion_matrix": {
                        "normalize": "true",
                        "plot_config": {
                            "figsize": [8, 6],
                            "cmap": "Blues",
                            "dpi": 100,
                            "save_format": "png",
                            "annot_fontsize": 8,
                        },
                    }
                },
                "visualization": {"confusion_matrix": {}},  # Fallback
            }
        )

        self.cm = np.array([[50, 10], [5, 35]])
        self.class_names = ["bg", "fg"]

    def tearDown(self):
        shutil.rmtree(self.tmp_dir)

    def test_plot_single_experiment(self):
        plotter = ConfusionMatrixPlotter(self.config)
        out_path = plotter.plot_single_experiment(
            self.cm, self.class_names, "exp1", self.tmp_dir
        )
        self.assertTrue(os.path.exists(out_path))

    def test_plot_comparison_grid(self):
        plotter = ConfusionMatrixPlotter(self.config)
        exp_data = {
            "exp1": {"confusion_matrix": self.cm, "class_names": self.class_names},
            "exp2": {
                "confusion_matrix": self.cm * 0.8,
                "class_names": self.class_names,
            },
        }
        out_path = plotter.plot_comparison_grid(exp_data, self.tmp_dir)
        self.assertTrue(os.path.exists(out_path))

    def test_plot_difference_matrix(self):
        plotter = ConfusionMatrixPlotter(self.config)
        exp1_data = {
            "name": "exp1",
            "confusion_matrix": self.cm,
            "class_names": self.class_names,
        }
        exp2_data = {
            "name": "exp2",
            "confusion_matrix": self.cm * 0.9,
            "class_names": self.class_names,
        }
        out_path = plotter.plot_difference_matrix(exp1_data, exp2_data, self.tmp_dir)
        self.assertTrue(os.path.exists(out_path))

    def test_plot_per_class_accuracy(self):
        plotter = ConfusionMatrixPlotter(self.config)
        exp_data = {
            "exp1": {"confusion_matrix": self.cm, "class_names": self.class_names},
            "exp2": {
                "confusion_matrix": self.cm * 0.7,
                "class_names": self.class_names,
            },
        }
        out_path = plotter.plot_per_class_accuracy(exp_data, self.tmp_dir)
        self.assertTrue(os.path.exists(out_path))


if __name__ == "__main__":
    unittest.main()
