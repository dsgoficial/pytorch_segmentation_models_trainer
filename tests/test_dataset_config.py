# -*- coding: utf-8 -*-
"""
Tests for dataset_config dataclasses.
"""

import pytest
from omegaconf import OmegaConf, MissingMandatoryValue
from pytorch_segmentation_models_trainer.config_definitions.dataset_config import (
    DataLoaderConfig,
    DatasetConfig,
    RasterPatchDatasetConfig,
    CSVWindowedDatasetConfig,
    CSVWindowedImageDatasetConfig,
)


class TestDatasetConfigs:
    def test_dataloader_config(self):
        cfg = OmegaConf.structured(DataLoaderConfig)
        container = OmegaConf.to_container(cfg, resolve=False)
        assert container["shuffle"] is True
        assert container["prefetch_factor"] == "${hyperparameters.batch_size}"

    def test_dataset_config(self):
        cfg = OmegaConf.structured(DatasetConfig)
        with pytest.raises(MissingMandatoryValue):
            _ = cfg.input_csv_path
        assert "SegmentationDataset" in cfg._target_

    def test_raster_patch_dataset_config(self):
        cfg = OmegaConf.structured(RasterPatchDatasetConfig)
        with pytest.raises(MissingMandatoryValue):
            _ = cfg.image_dir
        assert cfg.patch_size == 256
        assert cfg.stride == 128

    def test_csv_windowed_dataset_config(self):
        cfg = OmegaConf.structured(CSVWindowedDatasetConfig)
        assert "CSVWindowedSegmentationDataset" in cfg._target_
        assert cfg.image_key == "image"
        assert cfg.n_classes == 2

    def test_csv_windowed_image_dataset_config(self):
        cfg = OmegaConf.structured(CSVWindowedImageDatasetConfig)
        assert "CSVWindowedImageDataset" in cfg._target_
        assert cfg.image_key == "image"
        assert cfg.use_rasterio is True
