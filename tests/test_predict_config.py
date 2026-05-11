# -*- coding: utf-8 -*-
"""
Tests for predict_config dataclasses.
"""

import pytest
from omegaconf import OmegaConf, MissingMandatoryValue
from pytorch_segmentation_models_trainer.config_definitions.predict_config import (
    InferenceImageReaderConfig,
    InferenceProcessorConfig,
    ExportStrategyConfig,
    PolygonizerConfig,
    PredictSingleImageConfig,
)


class TestPredictConfigs:
    def test_inference_image_reader_config(self):
        cfg = OmegaConf.structured(InferenceImageReaderConfig)
        with pytest.raises(MissingMandatoryValue):
            _ = cfg._target_
        with pytest.raises(MissingMandatoryValue):
            _ = cfg.input_csv_path
        assert cfg.key == "image"

    def test_inference_processor_config(self):
        cfg = OmegaConf.structured(InferenceProcessorConfig)
        with pytest.raises(MissingMandatoryValue):
            _ = cfg._target_
        assert cfg.use_tta is False
        assert cfg.tta_augmentations == ["rot0", "rot90", "rot180", "rot270"]
        assert cfg.tile_weight == "mean"

    def test_export_strategy_config(self):
        cfg = OmegaConf.structured(ExportStrategyConfig)
        with pytest.raises(MissingMandatoryValue):
            _ = cfg._target_

    def test_polygonizer_config(self):
        cfg = OmegaConf.structured(PolygonizerConfig)
        with pytest.raises(MissingMandatoryValue):
            _ = cfg._target_
        assert cfg.config is None

    def test_predict_single_image_config(self):
        cfg = OmegaConf.structured(PredictSingleImageConfig)
        with pytest.raises(MissingMandatoryValue):
            _ = cfg.checkpoint_path
        assert cfg.device == "cuda:0"
        assert cfg.inference_threshold == 0.5
        assert (
            OmegaConf.get_type(cfg.inference_image_reader) == InferenceImageReaderConfig
        )
        assert OmegaConf.get_type(cfg.inference_processor) == InferenceProcessorConfig
