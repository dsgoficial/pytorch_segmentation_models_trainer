# -*- coding: utf-8 -*-
"""
Tests for loss_config_definition dataclasses.
"""

import pytest
from omegaconf import OmegaConf, MissingMandatoryValue
from unittest.mock import patch
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
    WeightedDiceCrossEntropyLossConfig,
    WeightedLovaszSCELossConfig,
    WeightedJMLSCELossConfig,
    WeightedDiceSCELossConfig,
    WeightedJMLSCEGCBLLossConfig,
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


class TestModernSegmentationLossConfigs:
    """Tests for modern segmentation loss config dataclasses."""

    def test_weighted_dice_ce_defaults(self):
        cfg = OmegaConf.structured(WeightedDiceCrossEntropyLossConfig)
        assert "WeightedDiceCrossEntropyLoss" in cfg._target_
        assert cfg.dice_weight == 0.75
        assert cfg.ce_weight == 0.25
        assert cfg.smooth_factor == 0.0
        assert cfg.ignore_index == 255

    def test_weighted_dice_ce_requires_num_classes(self):
        cfg = OmegaConf.structured(WeightedDiceCrossEntropyLossConfig)
        with pytest.raises(MissingMandatoryValue):
            _ = cfg.num_classes

    def test_weighted_dice_ce_override(self):
        cfg = OmegaConf.structured(WeightedDiceCrossEntropyLossConfig)
        OmegaConf.update(cfg, "num_classes", 7)
        OmegaConf.update(cfg, "dice_weight", 0.5)
        OmegaConf.update(cfg, "ce_weight", 0.5)
        assert cfg.num_classes == 7
        assert cfg.dice_weight == 0.5
        assert cfg.ce_weight == 0.5

    def test_weighted_lovasz_sce_defaults(self):
        cfg = OmegaConf.structured(WeightedLovaszSCELossConfig)
        assert "WeightedLovaszSCELoss" in cfg._target_
        assert cfg.lovasz_weight == 0.25
        assert cfg.sce_weight == 0.75
        assert cfg.ignore_index == 255
        assert cfg.sce_alpha == 1.0
        assert cfg.sce_beta == 0.5
        assert cfg.per_image is False
        assert cfg.lovasz_classes == "present"

    def test_weighted_lovasz_sce_requires_num_classes(self):
        cfg = OmegaConf.structured(WeightedLovaszSCELossConfig)
        with pytest.raises(MissingMandatoryValue):
            _ = cfg.num_classes

    def test_weighted_lovasz_sce_override(self):
        cfg = OmegaConf.structured(WeightedLovaszSCELossConfig)
        OmegaConf.update(cfg, "num_classes", 6)
        OmegaConf.update(cfg, "per_image", True)
        OmegaConf.update(cfg, "lovasz_classes", "all")
        assert cfg.num_classes == 6
        assert cfg.per_image is True
        assert cfg.lovasz_classes == "all"

    def test_weighted_jml_sce_defaults(self):
        cfg = OmegaConf.structured(WeightedJMLSCELossConfig)
        assert "WeightedJMLSCELoss" in cfg._target_
        assert cfg.jml_weight == 0.25
        assert cfg.sce_weight == 0.75
        assert cfg.ignore_index == 255
        assert cfg.sce_alpha == 1.0
        assert cfg.sce_beta == 0.5
        assert cfg.smooth == pytest.approx(1e-3)
        assert cfg.jml_classes == "present"
        assert cfg.label_smoothing == 0.0

    def test_weighted_jml_sce_requires_num_classes(self):
        cfg = OmegaConf.structured(WeightedJMLSCELossConfig)
        with pytest.raises(MissingMandatoryValue):
            _ = cfg.num_classes

    def test_weighted_jml_sce_override(self):
        cfg = OmegaConf.structured(WeightedJMLSCELossConfig)
        OmegaConf.update(cfg, "num_classes", 4)
        OmegaConf.update(cfg, "label_smoothing", 0.1)
        OmegaConf.update(cfg, "jml_classes", "all")
        assert cfg.num_classes == 4
        assert cfg.label_smoothing == pytest.approx(0.1)
        assert cfg.jml_classes == "all"

    def test_weighted_dice_sce_defaults(self):
        cfg = OmegaConf.structured(WeightedDiceSCELossConfig)
        assert "WeightedDiceSCELoss" in cfg._target_
        assert cfg.dice_weight == 0.25
        assert cfg.sce_weight == 0.75
        assert cfg.smooth_factor == 0.0
        assert cfg.ignore_index == 255
        assert cfg.sce_alpha == 1.0
        assert cfg.sce_beta == 0.5

    def test_weighted_dice_sce_requires_num_classes(self):
        cfg = OmegaConf.structured(WeightedDiceSCELossConfig)
        with pytest.raises(MissingMandatoryValue):
            _ = cfg.num_classes

    def test_weighted_dice_sce_override(self):
        cfg = OmegaConf.structured(WeightedDiceSCELossConfig)
        OmegaConf.update(cfg, "num_classes", 3)
        OmegaConf.update(cfg, "sce_beta", 0.8)
        assert cfg.num_classes == 3
        assert cfg.sce_beta == pytest.approx(0.8)

    def test_weighted_jml_sce_gcbl_defaults(self):
        cfg = OmegaConf.structured(WeightedJMLSCEGCBLLossConfig)
        assert "WeightedJMLSCEGCBLLoss" in cfg._target_
        assert cfg.jml_weight == 0.25
        assert cfg.sce_weight == 0.75
        assert cfg.gcbl_weight == pytest.approx(0.1)
        assert cfg.gcbl_embed_dim == 32
        assert cfg.gcbl_temperature == pytest.approx(0.07)
        assert cfg.gcbl_max_samples == 512

    def test_weighted_jml_sce_gcbl_requires_num_classes(self):
        cfg = OmegaConf.structured(WeightedJMLSCEGCBLLossConfig)
        with pytest.raises(MissingMandatoryValue):
            _ = cfg.num_classes

    def test_weighted_jml_sce_gcbl_override(self):
        cfg = OmegaConf.structured(WeightedJMLSCEGCBLLossConfig)
        OmegaConf.update(cfg, "num_classes", 7)
        OmegaConf.update(cfg, "gcbl_weight", 0.2)
        OmegaConf.update(cfg, "gcbl_embed_dim", 64)
        OmegaConf.update(cfg, "gcbl_max_samples", 256)
        assert cfg.num_classes == 7
        assert cfg.gcbl_weight == pytest.approx(0.2)
        assert cfg.gcbl_embed_dim == 64
        assert cfg.gcbl_max_samples == 256

    def test_all_modern_losses_have_distinct_targets(self):
        targets = [
            OmegaConf.structured(WeightedDiceCrossEntropyLossConfig)._target_,
            OmegaConf.structured(WeightedLovaszSCELossConfig)._target_,
            OmegaConf.structured(WeightedJMLSCELossConfig)._target_,
            OmegaConf.structured(WeightedDiceSCELossConfig)._target_,
            OmegaConf.structured(WeightedJMLSCEGCBLLossConfig)._target_,
        ]
        assert (
            len(set(targets)) == 5
        ), "All loss configs must have unique _target_ values"

    def test_all_modern_losses_point_to_loss_module(self):
        configs = [
            WeightedDiceCrossEntropyLossConfig,
            WeightedLovaszSCELossConfig,
            WeightedJMLSCELossConfig,
            WeightedDiceSCELossConfig,
            WeightedJMLSCEGCBLLossConfig,
        ]
        for cls in configs:
            cfg = OmegaConf.structured(cls)
            assert (
                "custom_losses.loss" in cfg._target_
            ), f"{cls.__name__}._target_ must point to custom_losses.loss module"

    def test_modern_losses_yaml_roundtrip(self):
        """Structured configs must survive a YAML round-trip."""
        for cls in [
            WeightedDiceCrossEntropyLossConfig,
            WeightedLovaszSCELossConfig,
            WeightedJMLSCELossConfig,
            WeightedDiceSCELossConfig,
            WeightedJMLSCEGCBLLossConfig,
        ]:
            cfg = OmegaConf.structured(cls)
            OmegaConf.update(cfg, "num_classes", 7)
            yaml_str = OmegaConf.to_yaml(cfg)
            restored = OmegaConf.create(yaml_str)
            assert restored.num_classes == 7
