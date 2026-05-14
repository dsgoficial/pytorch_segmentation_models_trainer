# -*- coding: utf-8 -*-
"""
Tests for dataset_config dataclasses.
"""

import pytest
from omegaconf import OmegaConf, MissingMandatoryValue
from pytorch_segmentation_models_trainer.config_definitions.dataset_config import (
    DataLoaderConfig,
    DatasetConfig,
    AutoencoderDatasetConfig,
    AutoencoderRandomCropDatasetConfig,
    WindowedImageAutoencoderDatasetConfig,
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

    def test_autoencoder_dataset_config(self):
        cfg = OmegaConf.structured(AutoencoderDatasetConfig)
        with pytest.raises(MissingMandatoryValue):
            _ = cfg.input_csv_path
        assert "image_dataset.AutoencoderDataset" in cfg._target_
        assert cfg.corruption_augmentation_list == []

    def test_autoencoder_random_crop_dataset_config(self):
        cfg = OmegaConf.structured(AutoencoderRandomCropDatasetConfig)
        with pytest.raises(MissingMandatoryValue):
            _ = cfg.image_dir
        assert "AutoencoderRandomCropDataset" in cfg._target_
        assert cfg.crop_size == [256, 256]
        assert cfg.samples_per_epoch == 10000
        assert cfg.split == "all"

    def test_windowed_image_autoencoder_dataset_config(self):
        cfg = OmegaConf.structured(WindowedImageAutoencoderDatasetConfig)
        with pytest.raises(MissingMandatoryValue):
            _ = cfg.image_dir
        assert "WindowedImageAutoencoderDataset" in cfg._target_
        assert cfg.crop_size == [256, 256]
        assert cfg.verify_windows is False
        assert cfg.window_index_cache is None
        assert cfg.serialize_rasterio_reads is False
        assert cfg.rasterio_lock_dir is None
        assert cfg.reopen_rasterio_on_read is False

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
