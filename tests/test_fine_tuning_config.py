# -*- coding: utf-8 -*-
"""
Tests for fine_tuning_config dataclasses.
"""

import pytest
from omegaconf import OmegaConf
from pytorch_segmentation_models_trainer.config_definitions.fine_tuning_config import (
    LoraAdapterConfig,
    FineTuningConfig,
)


class TestFineTuningConfig:
    def test_lora_adapter_config(self):
        cfg = OmegaConf.structured(LoraAdapterConfig)
        assert cfg.r == 16
        assert cfg.lora_alpha == 32.0
        assert cfg.target_modules == []

    def test_fine_tuning_config(self):
        cfg = OmegaConf.structured(FineTuningConfig)
        assert cfg.strategy == "full"
        assert cfg.lora_config is None
        assert "encoder" in cfg.frozen_modules
        assert "decoder" in cfg.trainable_modules
