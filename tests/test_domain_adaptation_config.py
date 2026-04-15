# -*- coding: utf-8 -*-
"""
Tests for domain_adaptation_config dataclasses.
"""
import pytest
from omegaconf import OmegaConf, MissingMandatoryValue

from pytorch_segmentation_models_trainer.config_definitions.domain_adaptation_config import (
    DataLoaderDAConfig,
    DomainAdaptationConfig,
    DomainAdaptationDatasetConfig,
    DomainAdaptationMethodConfig,
    PretrainedCheckpointConfig,
)


class TestPretrainedCheckpointConfig:
    def test_defaults(self):
        cfg = OmegaConf.structured(PretrainedCheckpointConfig)
        assert cfg.source_format == "pytorch_lightning"
        assert cfg.strict_loading is True

    def test_path_is_required(self):
        cfg = OmegaConf.structured(PretrainedCheckpointConfig)
        with pytest.raises(MissingMandatoryValue):
            _ = cfg.path

    def test_custom_values(self):
        cfg = OmegaConf.structured(
            PretrainedCheckpointConfig(
                path="/tmp/model.ckpt",
                source_format="pytorch",
                strict_loading=False,
            )
        )
        assert cfg.path == "/tmp/model.ckpt"
        assert cfg.source_format == "pytorch"
        assert cfg.strict_loading is False


class TestDomainAdaptationMethodConfig:
    def test_target_is_required(self):
        cfg = OmegaConf.structured(DomainAdaptationMethodConfig)
        with pytest.raises(MissingMandatoryValue):
            _ = cfg._target_

    def test_lambda_da_default(self):
        cfg = OmegaConf.structured(DomainAdaptationMethodConfig)
        assert cfg.lambda_da == 1.0

    def test_custom_lambda(self):
        cfg = OmegaConf.structured(
            DomainAdaptationMethodConfig(_target_="my.Method", lambda_da=0.5)
        )
        assert cfg.lambda_da == 0.5


class TestDataLoaderDAConfig:
    def test_defaults(self):
        cfg = OmegaConf.structured(DataLoaderDAConfig)
        assert cfg.batch_size == 8
        assert cfg.num_workers == 4
        assert cfg.shuffle is True
        assert cfg.pin_memory is True
        assert cfg.drop_last is True
        assert cfg.prefetch_factor == 2
        assert cfg.persistent_workers is False


class TestDomainAdaptationDatasetConfig:
    def test_target_is_required(self):
        cfg = OmegaConf.structured(DomainAdaptationDatasetConfig)
        with pytest.raises(MissingMandatoryValue):
            _ = cfg._target_

    def test_input_csv_is_required(self):
        cfg = OmegaConf.structured(DomainAdaptationDatasetConfig)
        with pytest.raises(MissingMandatoryValue):
            _ = cfg.input_csv_path

    def test_data_loader_defaults(self):
        cfg = OmegaConf.structured(DomainAdaptationDatasetConfig)
        assert cfg.data_loader.batch_size == 8


class TestDomainAdaptationConfig:
    def test_method_is_required(self):
        cfg = OmegaConf.structured(DomainAdaptationConfig)
        with pytest.raises(MissingMandatoryValue):
            _ = cfg.method

    def test_source_dataset_is_required(self):
        cfg = OmegaConf.structured(DomainAdaptationConfig)
        with pytest.raises(MissingMandatoryValue):
            _ = cfg.source_dataset

    def test_optional_val_datasets_default_to_none(self):
        cfg = OmegaConf.structured(DomainAdaptationConfig)
        assert cfg.source_val_dataset is None
        assert cfg.target_val_dataset is None

    def test_optional_pretrained_checkpoint_defaults_to_none(self):
        cfg = OmegaConf.structured(DomainAdaptationConfig)
        assert cfg.pretrained_checkpoint is None

    def test_feature_layers_defaults_to_empty_list(self):
        cfg = OmegaConf.structured(DomainAdaptationConfig)
        assert list(cfg.feature_layers) == []
