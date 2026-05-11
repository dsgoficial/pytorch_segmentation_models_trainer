# -*- coding: utf-8 -*-
"""
Tests for loss_config_definition dataclasses.
"""

import pytest
from omegaconf import OmegaConf, MissingMandatoryValue
from unittest.mock import MagicMock, patch
from pytorch_segmentation_models_trainer.config_definitions.loss_config_definition import (
    SegParamsConfig,
    BaseLossConfig,
    SegLossConfig,
    CrossfieldAlignLossConfig,
    CrossfieldAlign90LossConfig,
    CrossfieldSmoothLossConfig,
    SegCrossfieldLossConfig,
    SegEdgeInteriorLossConfig,
    NormalizationParams,
    LossWeightConfig,
    CompoundLossConfig,
    CoefsConfig,
    SegLossParamsConfig,
    MultiLossConfig,
    LossParamsConfig,
    build_config,
)


class TestLossConfigs:
    def test_seg_params_config(self):
        cfg = OmegaConf.structured(SegParamsConfig)
        assert cfg.compute_interior is True
        assert cfg.compute_edge is True
        assert cfg.compute_vertex is True

    def test_base_loss_config(self):
        cfg = OmegaConf.structured(BaseLossConfig)
        with pytest.raises(MissingMandatoryValue):
            _ = cfg._target_
        with pytest.raises(MissingMandatoryValue):
            _ = cfg.name
        assert cfg.weight == 1.0

    def test_seg_loss_config(self):
        cfg = OmegaConf.structured(SegLossConfig)
        assert "SegLoss" in cfg._target_
        assert cfg.name == "seg"
        assert cfg.bce_coef == 0.5
        assert cfg.dice_coef == 0.5
        assert cfg.weight == 1.0

    def test_crossfield_losses(self):
        cfg1 = OmegaConf.structured(CrossfieldAlignLossConfig)
        assert cfg1.name == "crossfield_align"
        assert cfg1.weight == 1.0

        cfg2 = OmegaConf.structured(CrossfieldAlign90LossConfig)
        assert cfg2.name == "crossfield_align90"
        assert cfg2.weight == 0.2

        cfg3 = OmegaConf.structured(CrossfieldSmoothLossConfig)
        assert cfg3.name == "crossfield_smooth"
        assert cfg3.weight == 0.005

    def test_seg_crossfield_loss(self):
        cfg = OmegaConf.structured(SegCrossfieldLossConfig)
        assert cfg.name == "seg_crossfield"
        assert cfg.pred_channel == 0
        assert cfg.weight == [0, 0, 0.2]

    def test_seg_edge_interior_loss(self):
        cfg = OmegaConf.structured(SegEdgeInteriorLossConfig)
        assert cfg.name == "seg_edge_interior"
        assert cfg.weight == [0, 0, 0.2]

    def test_compound_loss_configs(self):
        norm = OmegaConf.structured(NormalizationParams)
        assert norm.min_samples == 10
        assert norm.max_samples == 1000

        lw = OmegaConf.structured(LossWeightConfig)
        with pytest.raises(MissingMandatoryValue):
            _ = lw.loss
        assert lw.weight == 1.0

        compound = OmegaConf.structured(CompoundLossConfig)
        assert compound.losses == []
        assert compound.epoch_thresholds == [0, 5, 10]
        assert compound.normalization_params.min_samples == 10

    def test_legacy_configs(self):
        coefs = OmegaConf.structured(CoefsConfig)
        assert coefs.seg == 10
        assert coefs.crossfield_align == 1

        seg_params = OmegaConf.structured(SegLossParamsConfig)
        assert seg_params.bce_coef == 1.0
        assert seg_params.dice_coef == 0.2

        multi = OmegaConf.structured(MultiLossConfig)
        assert len(multi.defaults) == 3
        with pytest.raises(MissingMandatoryValue):
            _ = multi.normalization_params

        loss_params = OmegaConf.structured(LossParamsConfig)
        assert loss_params.compound_loss is None
        assert loss_params.seg_loss_params.bce_coef == 1.0

    def test_build_config(self):
        # build_config(cfg: DictConfig) -> None
        mock_cfg = OmegaConf.create({"test": "value"})
        with patch(
            "pytorch_segmentation_models_trainer.config_definitions.loss_config_definition.logger"
        ) as mock_logger:
            build_config(mock_cfg)
            mock_logger.info.assert_called()
