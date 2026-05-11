# -*- coding: utf-8 -*-
"""
Tests for coco_dataset_config dataclasses.
"""

import pytest
from omegaconf import OmegaConf, MissingMandatoryValue
from pytorch_segmentation_models_trainer.config_definitions.coco_dataset_config import (
    CocoDatasetInfoConfig,
    LicenseConfig,
    ImageConfig,
    CategoryConfig,
    AnnotationConfig,
    CocoDatasetConfig,
)


class TestCocoDatasetConfigs:
    def test_info_config(self):
        cfg = OmegaConf.structured(CocoDatasetInfoConfig)
        with pytest.raises(MissingMandatoryValue):
            _ = cfg.description

    def test_license_config(self):
        cfg = OmegaConf.structured(LicenseConfig)
        with pytest.raises(MissingMandatoryValue):
            _ = cfg.name

    def test_image_config(self):
        cfg = OmegaConf.structured(ImageConfig)
        with pytest.raises(MissingMandatoryValue):
            _ = cfg.file_name

    def test_category_config(self):
        cfg = OmegaConf.structured(CategoryConfig)
        with pytest.raises(MissingMandatoryValue):
            _ = cfg.name

    def test_annotation_config(self):
        cfg = OmegaConf.structured(AnnotationConfig)
        assert cfg.segmentation == [[]]
        assert cfg.bbox == [0, 0, 0, 0]
        with pytest.raises(MissingMandatoryValue):
            _ = cfg.area

    def test_coco_dataset_config(self):
        cfg = OmegaConf.structured(CocoDatasetConfig)
        assert OmegaConf.get_type(cfg.info) == CocoDatasetInfoConfig
        assert len(cfg.licenses) == 1
        assert len(cfg.images) == 1
        assert len(cfg.categories) == 1
        assert len(cfg.annotations) == 1
