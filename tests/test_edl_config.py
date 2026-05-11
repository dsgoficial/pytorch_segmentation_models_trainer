# -*- coding: utf-8 -*-
"""
Tests for edl_config dataclasses.
"""

import pytest
from omegaconf import OmegaConf, MissingMandatoryValue
from pytorch_segmentation_models_trainer.config_definitions.edl_config import (
    EvidentialWrapperConfig,
    EvidentialMSELossConfig,
    EvidentialKLLossConfig,
    EvidentialWarmupCallbackConfig,
    EvidentialUncertaintyVisualizationCallbackConfig,
    EvidentialInferenceProcessorConfig,
)


class TestEDLConfigs:
    def test_evidential_wrapper_config(self):
        cfg = OmegaConf.structured(EvidentialWrapperConfig)
        assert "EvidentialWrapper" in cfg._target_
        with pytest.raises(MissingMandatoryValue):
            _ = cfg.model

    def test_evidential_mse_loss_config(self):
        cfg = OmegaConf.structured(EvidentialMSELossConfig)
        assert cfg.name == "edl_mse"
        with pytest.raises(MissingMandatoryValue):
            _ = cfg.num_classes

    def test_evidential_kl_loss_config(self):
        cfg = OmegaConf.structured(EvidentialKLLossConfig)
        assert cfg.name == "edl_kl"
        with pytest.raises(MissingMandatoryValue):
            _ = cfg.num_classes

    def test_evidential_warmup_callback_config(self):
        cfg = OmegaConf.structured(EvidentialWarmupCallbackConfig)
        assert cfg.warmup_epochs == 5
        assert cfg.freeze_encoder is False

    def test_evidential_uncertainty_viz_callback_config(self):
        cfg = OmegaConf.structured(EvidentialUncertaintyVisualizationCallbackConfig)
        assert cfg.num_images == 4
        assert cfg.log_every_n_epochs == 5

    def test_evidential_inference_processor_config(self):
        cfg = OmegaConf.structured(EvidentialInferenceProcessorConfig)
        assert cfg.device == "cuda"
        assert cfg.batch_size == 1
        assert cfg.model_input_shape == [512, 512]
        with pytest.raises(MissingMandatoryValue):
            _ = cfg.model_path
