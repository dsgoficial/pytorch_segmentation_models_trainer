# -*- coding: utf-8 -*-
"""
Tests for inference_config dataclasses.
"""

import pytest
from omegaconf import OmegaConf, MissingMandatoryValue
from pytorch_segmentation_models_trainer.config_definitions.inference_config import (
    DataLoaderConfig,
    BuildCSVFromFolderConfig,
    InferenceDatasetConfig,
)


class TestInferenceConfigs:
    def test_dataloader_config(self):
        cfg = OmegaConf.structured(DataLoaderConfig)
        assert cfg.num_workers == 4
        assert cfg.prefetch_factor == 2

    def test_build_csv_from_folder_config(self):
        cfg = OmegaConf.structured(BuildCSVFromFolderConfig)
        assert cfg.enabled is False
        assert cfg.image_pattern == "*.tif"
        with pytest.raises(MissingMandatoryValue):
            _ = cfg.images_folder

    def test_inference_dataset_config(self):
        cfg = OmegaConf.structured(InferenceDatasetConfig)
        with pytest.raises(MissingMandatoryValue):
            _ = cfg.input_csv_path
        assert OmegaConf.get_type(cfg.data_loader) == DataLoaderConfig
