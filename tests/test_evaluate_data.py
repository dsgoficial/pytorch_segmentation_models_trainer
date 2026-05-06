# -*- coding: utf-8 -*-
"""
Unit tests for evaluate_experiments.py
"""

import unittest
from unittest.mock import MagicMock, patch
import os
import shutil
import tempfile
from pathlib import Path
from omegaconf import OmegaConf

from pytorch_segmentation_models_trainer.evaluate_experiments import (
    evaluate,
    setup_logging,
)


class TestEvaluateExperiments(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()
        self.config = OmegaConf.create(
            {
                "name": "test_pipeline",
                "output": {"base_dir": self.tmp_dir},
                "logging": {"level": "info", "save_to_file": False},
                "experiments": [],
                "pipeline_options": {"parallel_inference": {"enabled": False}},
            }
        )

    def tearDown(self):
        shutil.rmtree(self.tmp_dir)

    def test_setup_logging_basic(self):
        # Should not raise errors
        setup_logging(self.config)

    def test_setup_logging_to_file(self):
        self.config.logging.save_to_file = True
        self.config.logging.log_file = "test.log"
        setup_logging(self.config)
        self.assertTrue(os.path.exists(os.path.join(self.tmp_dir, "test.log")))

    @patch(
        "pytorch_segmentation_models_trainer.evaluate_experiments.EvaluationPipeline"
    )
    def test_evaluate_function(self, MockPipeline):
        mock_pipeline_instance = MagicMock()
        MockPipeline.return_value = mock_pipeline_instance
        mock_pipeline_instance.run.return_value = {"status": "success"}

        # Call evaluate directly (bypass @hydra.main decoration logic for unit test)
        # Note: evaluate is actually the 'evaluate' function defined in evaluate_experiments.py
        # but hydra wraps it. We can call the underlying function if needed or just call it as is.
        results = evaluate(self.config)

        self.assertEqual(results, {"status": "success"})
        mock_pipeline_instance.run.assert_called_once()

    @patch(
        "pytorch_segmentation_models_trainer.evaluate_experiments.EvaluationPipeline"
    )
    @patch("sys.exit")
    def test_evaluate_function_error(self, mock_exit, MockPipeline):
        mock_pipeline_instance = MagicMock()
        MockPipeline.return_value = mock_pipeline_instance
        mock_pipeline_instance.run.side_effect = Exception("Test error")

        evaluate(self.config)
        mock_exit.assert_called_with(1)


if __name__ == "__main__":
    unittest.main()
