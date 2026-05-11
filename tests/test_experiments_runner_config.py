# -*- coding: utf-8 -*-
"""
Tests for experiments_runner_config dataclasses.
"""

import pytest
from omegaconf import OmegaConf
from pytorch_segmentation_models_trainer.config_definitions.experiments_runner_config import (
    ExperimentsRunnerConfig,
)


class TestExperimentsRunnerConfig:
    def test_defaults(self):
        cfg = OmegaConf.structured(ExperimentsRunnerConfig)
        assert cfg.n_runs is None
        assert cfg.seeds is None
        assert cfg.output_base_dir == "outputs/experiments_runner"
        assert cfg.save_summary is True
        assert cfg.summary_metrics == ["val/loss"]
        assert cfg.resume is False
