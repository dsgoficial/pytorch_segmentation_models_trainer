# -*- coding: utf-8 -*-
"""
Tests for evaluation_config dataclasses.
"""

import pytest
from omegaconf import OmegaConf, MissingMandatoryValue
from pytorch_segmentation_models_trainer.config_definitions.evaluation_config import (
    ExperimentConfig,
    BuildCSVFromFoldersConfig,
    EvaluationDatasetConfig,
    MetricsConfig,
    OutputConfig,
    VisualizationConfig,
    PipelineOptionsConfig,
    EvaluationPipelineConfig,
)


class TestEvaluationConfigs:
    def test_experiment_config(self):
        cfg = OmegaConf.structured(ExperimentConfig)
        with pytest.raises(MissingMandatoryValue):
            _ = cfg.name
        assert cfg.overrides is None

    def test_build_csv_from_folders_config(self):
        cfg = OmegaConf.structured(BuildCSVFromFoldersConfig)
        assert cfg.enabled is False
        assert cfg.image_pattern == "*.tif"
        with pytest.raises(MissingMandatoryValue):
            _ = cfg.images_folder

    def test_evaluation_dataset_config(self):
        cfg = OmegaConf.structured(EvaluationDatasetConfig)
        with pytest.raises(MissingMandatoryValue):
            _ = cfg.input_csv_path
        assert (
            OmegaConf.get_type(cfg.build_csv_from_folders) == BuildCSVFromFoldersConfig
        )

    def test_metrics_config(self):
        cfg = OmegaConf.structured(MetricsConfig)
        with pytest.raises(MissingMandatoryValue):
            _ = cfg.num_classes
        assert cfg.segmentation_metrics == []

    def test_output_config(self):
        cfg = OmegaConf.structured(OutputConfig)
        assert cfg.base_dir == "./evaluation_results"

    def test_visualization_config(self):
        cfg = OmegaConf.structured(VisualizationConfig)
        assert cfg.comparison_plots.enabled is True

    def test_pipeline_options_config(self):
        cfg = OmegaConf.structured(PipelineOptionsConfig)
        assert cfg.parallel_inference.enabled is False
        assert cfg.load_predictions_from_folder.enabled is False

    def test_evaluation_pipeline_config(self):
        cfg = OmegaConf.structured(EvaluationPipelineConfig)
        assert cfg.experiments == []
        assert OmegaConf.get_type(cfg.output) == OutputConfig
