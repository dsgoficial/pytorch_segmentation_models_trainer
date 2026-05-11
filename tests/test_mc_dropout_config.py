# -*- coding: utf-8 -*-
"""
Tests for mc_dropout_config dataclasses.
"""

import pytest
from omegaconf import OmegaConf, MissingMandatoryValue
from pytorch_segmentation_models_trainer.config_definitions.mc_dropout_config import (
    MCDropoutInferenceProcessorConfig,
)


class TestMCDropoutConfig:
    def test_defaults(self):
        cfg = OmegaConf.structured(MCDropoutInferenceProcessorConfig)
        assert cfg.n_samples == 10
        assert cfg.uncertainty_mode == "entropy"
        assert cfg.export_uncertainty_map is False
        assert cfg.model_input_shape == [512, 512]
        with pytest.raises(MissingMandatoryValue):
            _ = cfg.num_classes
