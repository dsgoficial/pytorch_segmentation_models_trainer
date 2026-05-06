# -*- coding: utf-8 -*-
"""
Unit tests for validate_evaluation_config.py
"""

import unittest
from unittest.mock import MagicMock, patch
from omegaconf import DictConfig, OmegaConf

from pytorch_segmentation_models_trainer.validate_evaluation_config import (
    validate_experiment_config,
    validate_dataset_config,
    validate_metrics_config,
    validate_output_config,
)


class TestValidateEvaluationConfig(unittest.TestCase):

    def setUp(self):
        self.base_exp_config = OmegaConf.create(
            {
                "name": "test_exp",
                "predict_config": {"model_path": "/path/to/model.ckpt"},
                "checkpoint_path": "/path/to/checkpoint.pth",
                "output_folder": "/path/to/output",
                "inference_config": {
                    "processor": {
                        "_target_": "pytorch_segmentation_models_trainer.tools.inference.inference_processors.MultiClassInferenceProcessor",
                        "model": "null",
                        "device": "cpu",
                        "batch_size": 4,
                        "export_strategy": {
                            "_target_": "pytorch_segmentation_models_trainer.tools.inference.export_inference.RasterExportInferenceStrategy"
                        },
                    }
                },
                "dataset_config": {
                    "prediction_csv_path": "/path/to/predictions.csv",
                    "ground_truth_csv_path": "/path/to/gt.csv",
                },
            }
        )

    @patch(
        "pytorch_segmentation_models_trainer.validate_evaluation_config.os.path.exists"
    )
    def test_validate_experiment_config_valid(self, mock_exists):
        mock_exists.return_value = True
        errors, warnings = validate_experiment_config(self.base_exp_config, 0)
        self.assertEqual(len(errors), 0)
        self.assertEqual(len(warnings), 0)

    def test_validate_experiment_config_missing_required_fields(self):
        invalid_config = OmegaConf.create({})
        errors, warnings = validate_experiment_config(invalid_config, 0)
        self.assertGreater(len(errors), 0)
        self.assertIn("Experiment 0: Missing required field 'name'", errors)
        self.assertIn("Experiment 0: Missing required field 'predict_config'", errors)
        self.assertIn("Experiment 0: Missing required field 'checkpoint_path'", errors)
        self.assertIn("Experiment 0: Missing required field 'output_folder'", errors)

    @patch(
        "pytorch_segmentation_models_trainer.validate_evaluation_config.os.path.exists"
    )
    def test_validate_experiment_config_predict_config_model_path_warning(
        self, mock_exists
    ):
        mock_exists.return_value = True
        config = OmegaConf.merge(
            self.base_exp_config,
            {"predict_config": {"model_path": None, "model_id": "some_id"}},
        )
        errors, warnings = validate_experiment_config(config, 0)
        self.assertEqual(len(errors), 0)
        self.assertIn(
            "Experiment 0: 'predict_config.model_path' is empty. If using MLflow or similar for model retrieval, ensure 'predict_config.model_id' is set and correctly handled downstream.",
            warnings,
        )

    @patch(
        "pytorch_segmentation_models_trainer.validate_evaluation_config.os.path.exists"
    )
    def test_validate_experiment_config_inference_processor_target(self, mock_exists):
        mock_exists.return_value = True
        # Invalid target for processor
        invalid_config = OmegaConf.merge(
            self.base_exp_config,
            {"inference_config": {"processor": {"_target_": "invalid.path.Processor"}}},
        )
        errors, warnings = validate_experiment_config(invalid_config, 0)
        self.assertGreater(len(errors), 0)
        self.assertIn(
            "Experiment 0: 'inference_config.processor._target_' is not a valid class path or the class does not exist.",
            errors,
        )

        # Missing target for processor - merge doesn't remove, so we must delete or create from scratch
        invalid_config = OmegaConf.merge(self.base_exp_config, {})
        invalid_config.inference_config.processor = {}  # Overwrite with empty dict

        errors, warnings = validate_experiment_config(invalid_config, 0)
        self.assertGreater(len(errors), 0)
        self.assertIn(
            "Experiment 0: 'inference_config.processor._target_' is missing.", errors
        )

    @patch(
        "pytorch_segmentation_models_trainer.validate_evaluation_config.os.path.exists"
    )
    def test_validate_experiment_config_export_strategy_target(self, mock_exists):
        mock_exists.return_value = True
        # Invalid target for export strategy
        invalid_config = OmegaConf.merge(
            self.base_exp_config,
            {
                "inference_config": {
                    "processor": {
                        "export_strategy": {"_target_": "invalid.path.Strategy"}
                    }
                }
            },
        )
        errors, warnings = validate_experiment_config(invalid_config, 0)
        self.assertGreater(len(errors), 0)
        self.assertIn(
            "Experiment 0: 'inference_config.processor.export_strategy._target_' is not a valid class path or the class does not exist.",
            errors,
        )

        # Missing target for export strategy
        invalid_config = OmegaConf.merge(self.base_exp_config, {})
        invalid_config.inference_config.processor.export_strategy = (
            {}
        )  # Overwrite with empty dict

        errors, warnings = validate_experiment_config(invalid_config, 0)
        self.assertGreater(len(errors), 0)
        self.assertIn(
            "Experiment 0: 'inference_config.processor.export_strategy._target_' is missing.",
            errors,
        )

    @patch(
        "pytorch_segmentation_models_trainer.validate_evaluation_config.os.path.exists"
    )
    def test_validate_experiment_config_dataset_config_paths_exclusive(
        self, mock_exists
    ):
        mock_exists.return_value = True
        # Both prediction_csv_path and prediction_input_folder
        invalid_config = OmegaConf.merge(
            self.base_exp_config,
            {
                "dataset_config": {
                    "prediction_csv_path": "/path/to/csv",
                    "prediction_input_folder": "/path/to/folder",
                }
            },
        )
        errors, warnings = validate_experiment_config(invalid_config, 0)
        self.assertGreater(len(errors), 0)
        self.assertIn(
            "Experiment 0: 'dataset_config.prediction_csv_path' and 'dataset_config.prediction_input_folder' are mutually exclusive. Provide only one.",
            errors,
        )

        # Both ground_truth_csv_path and ground_truth_input_folder
        invalid_config = OmegaConf.merge(
            self.base_exp_config,
            {
                "dataset_config": {
                    "ground_truth_csv_path": "/path/to/csv",
                    "ground_truth_input_folder": "/path/to/folder",
                }
            },
        )
        errors, warnings = validate_experiment_config(invalid_config, 0)
        self.assertGreater(len(errors), 0)
        self.assertIn(
            "Experiment 0: 'dataset_config.ground_truth_csv_path' and 'dataset_config.ground_truth_input_folder' are mutually exclusive. Provide only one.",
            errors,
        )

    @patch(
        "pytorch_segmentation_models_trainer.validate_evaluation_config.os.path.exists"
    )
    def test_validate_experiment_config_dataset_config_missing_prediction_source(
        self, mock_exists
    ):
        mock_exists.return_value = True
        # Missing both prediction_csv_path and prediction_input_folder
        invalid_config = OmegaConf.merge(
            self.base_exp_config,
            {"dataset_config": {"prediction_csv_path": None}},
        )
        del (
            invalid_config.dataset_config.prediction_csv_path
        )  # Ensure it's truly missing
        errors, warnings = validate_experiment_config(invalid_config, 0)
        self.assertGreater(len(errors), 0)
        self.assertIn(
            "Experiment 0: Either 'dataset_config.prediction_csv_path' or 'dataset_config.prediction_input_folder' must be provided.",
            errors,
        )


class TestValidateConfigs(unittest.TestCase):
    def test_validate_dataset_config_csv_valid(self):
        config = OmegaConf.create(
            {
                "build_csv_from_folders": {"enabled": False},
                "input_csv_path": "valid.csv",
            }
        )
        with patch("os.path.exists", return_value=True):
            with patch("pandas.read_csv") as mock_read_csv:
                mock_df = MagicMock()
                mock_df.columns = ["image", "mask"]
                mock_df.__len__.return_value = 10
                mock_df.head.return_value.iterrows.return_value = []
                mock_read_csv.return_value = mock_df

                is_valid, errors, warnings = validate_dataset_config(config)
                self.assertTrue(is_valid)
                self.assertEqual(len(errors), 0)

    def test_validate_dataset_config_csv_missing(self):
        config = OmegaConf.create(
            {
                "build_csv_from_folders": {"enabled": False},
                "input_csv_path": "missing.csv",
            }
        )
        with patch("os.path.exists", return_value=False):
            is_valid, errors, warnings = validate_dataset_config(config)
            self.assertFalse(is_valid)
            self.assertIn("Dataset CSV not found: missing.csv", errors)

    def test_validate_dataset_config_folders_valid(self):
        config = OmegaConf.create(
            {
                "build_csv_from_folders": {
                    "enabled": True,
                    "images_folder": "images",
                    "masks_folder": "masks",
                    "image_pattern": "*.tif",
                }
            }
        )
        with patch("os.path.exists", return_value=True):
            with patch(
                "pytorch_segmentation_models_trainer.validate_evaluation_config.Path.glob",
                return_value=[MagicMock()],
            ):
                is_valid, errors, warnings = validate_dataset_config(config)
                self.assertTrue(is_valid)
                self.assertEqual(len(errors), 0)

    def test_validate_metrics_config(self):
        # Valid
        config = OmegaConf.create(
            {"segmentation_metrics": [{"_target_": "some.Metric"}]}
        )
        is_valid, errors, warnings = validate_metrics_config(config)
        self.assertTrue(is_valid)

        # Missing target
        invalid_config = OmegaConf.create({"segmentation_metrics": [{}]})
        is_valid, errors, warnings = validate_metrics_config(invalid_config)
        self.assertFalse(is_valid)
        self.assertIn("Metric 0: Missing '_target_' field", errors)

    def test_validate_output_config(self):
        config = OmegaConf.create({"base_dir": "output"})
        # Existing and writable
        with patch(
            "pytorch_segmentation_models_trainer.validate_evaluation_config.Path.exists",
            return_value=True,
        ):
            with patch("os.access", return_value=True):
                is_valid, errors, warnings = validate_output_config(config)
                self.assertTrue(is_valid)

        # Not existing, can create
        with patch(
            "pytorch_segmentation_models_trainer.validate_evaluation_config.Path.exists",
            return_value=False,
        ):
            with patch(
                "pytorch_segmentation_models_trainer.validate_evaluation_config.Path.mkdir"
            ):
                with patch(
                    "pytorch_segmentation_models_trainer.validate_evaluation_config.Path.rmdir"
                ):
                    is_valid, errors, warnings = validate_output_config(config)
                    self.assertTrue(is_valid)


if __name__ == "__main__":
    unittest.main()
