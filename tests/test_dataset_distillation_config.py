# -*- coding: utf-8 -*-
"""
Tests for dataset_distillation_config dataclasses.
"""

import pytest
from omegaconf import OmegaConf
from unittest.mock import patch, MagicMock
from pytorch_segmentation_models_trainer.config_definitions.dataset_distillation_config import (
    DatasetDistillationConfig,
    register_dataset_distillation_configs,
)


class TestDatasetDistillationConfig:
    def test_defaults(self):
        cfg = OmegaConf.structured(DatasetDistillationConfig)
        assert cfg.num_clusters == 100
        assert cfg.batch_size == 32
        assert cfg.random_seed == 42
        assert cfg.use_sqrt_heuristic is True

    def test_register_configs(self):
        with patch("hydra.core.config_store.ConfigStore.instance") as mock_cs_instance:
            mock_cs = MagicMock()
            mock_cs_instance.return_value = mock_cs
            register_dataset_distillation_configs()
            mock_cs.store.assert_called_once()
            args, kwargs = mock_cs.store.call_args
            assert kwargs["group"] == "dataset_distillation"
            assert kwargs["name"] == "base_dataset_distillation"
            assert kwargs["node"] == DatasetDistillationConfig
