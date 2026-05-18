# -*- coding: utf-8 -*-
"""
Tests for dataset_distillation_config dataclasses.
"""

from omegaconf import OmegaConf
from unittest.mock import patch, MagicMock
from pytorch_segmentation_models_trainer.config_definitions import (
    dataset_distillation_config as distillation_config,
)

DatasetDistillationConfig = distillation_config.DatasetDistillationConfig
register_dataset_distillation_configs = (
    distillation_config.register_dataset_distillation_configs
)


class TestDatasetDistillationConfig:
    def test_defaults(self):
        cfg = OmegaConf.structured(DatasetDistillationConfig)
        assert cfg.num_clusters == 100
        assert cfg.k is None
        assert cfg.mode == "vae_decode"
        assert cfg.latent == "mu"
        assert cfg.latent_reduction == "flatten"
        assert cfg.weight_mode == "sqrt"
        assert cfg.distilled_image_format == "auto"
        assert cfg.batch_size == 32
        assert cfg.random_seed == 42
        assert cfg.use_sqrt_heuristic is True

    def test_register_configs(self):
        with patch("hydra.core.config_store.ConfigStore.instance") as mock_cs_instance:
            mock_cs = MagicMock()
            mock_cs_instance.return_value = mock_cs
            register_dataset_distillation_configs()
            assert mock_cs.store.call_count == 2
            names = [call.kwargs["name"] for call in mock_cs.store.call_args_list]
            assert names == ["base_dataset_distillation", "vae_ddoq"]
            for call in mock_cs.store.call_args_list:
                assert call.kwargs["group"] == "dataset_distillation"
                assert call.kwargs["node"] == DatasetDistillationConfig
