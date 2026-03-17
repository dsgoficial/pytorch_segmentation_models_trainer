# -*- coding: utf-8 -*-
"""
Tests for pytorch_segmentation_models_trainer.custom_losses.loss_builder
"""
import pytest
import torch
import torch.nn as nn
from unittest.mock import MagicMock, patch
from omegaconf import OmegaConf

from pytorch_segmentation_models_trainer.custom_losses.loss_builder import (
    LossWrapper,
    build_compound_loss_from_config,
    build_loss_from_config,
    validate_loss_config,
)
from pytorch_segmentation_models_trainer.custom_losses.base_loss import MultiLoss, Loss


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_compound_cfg():
    """Return a minimal valid compound_loss DictConfig."""
    return OmegaConf.create({
        "losses": [
            {
                "loss": {"_target_": "torch.nn.BCEWithLogitsLoss"},
                "weight": 1.0,
            }
        ]
    })


def _make_full_cfg(key="compound_loss"):
    if key == "compound_loss":
        return OmegaConf.create({
            "loss_params": {
                "compound_loss": {
                    "losses": [
                        {
                            "loss": {"_target_": "torch.nn.BCEWithLogitsLoss"},
                            "weight": 1.0,
                        }
                    ]
                }
            }
        })
    elif key == "multi_loss":
        return OmegaConf.create({
            "loss_params": {
                "multi_loss": {}
            }
        })


# ---------------------------------------------------------------------------
# LossWrapper tests
# ---------------------------------------------------------------------------

class TestLossWrapper:
    def test_wraps_standard_loss(self):
        loss_fn = nn.BCEWithLogitsLoss()
        wrapper = LossWrapper(loss_fn)
        assert wrapper.is_custom_loss is False
        assert wrapper.name == "BCEWithLogitsLoss"

    def test_wraps_standard_loss_with_custom_name(self):
        loss_fn = nn.BCEWithLogitsLoss()
        wrapper = LossWrapper(loss_fn, name="my_bce")
        assert wrapper.name == "my_bce"

    def test_wraps_custom_loss(self):
        mock_custom = MagicMock(spec=Loss)
        mock_custom.name = "custom_seg"
        wrapper = LossWrapper(mock_custom)
        assert wrapper.is_custom_loss is True
        assert wrapper.name == "custom_seg"

    def test_forward_standard_loss_returns_tuple(self):
        loss_fn = nn.MSELoss()
        wrapper = LossWrapper(loss_fn)
        pred = torch.randn(2, 1)
        gt = torch.randn(2, 1)
        result = wrapper(pred, gt)
        assert isinstance(result, tuple)
        loss_value, extra = result
        assert isinstance(loss_value, torch.Tensor)
        assert extra == {}

    def test_forward_custom_loss_delegates(self):
        mock_custom = MagicMock(spec=Loss)
        mock_custom.name = "custom"
        expected_loss = torch.tensor(0.5)
        mock_custom.return_value = (expected_loss, {"info": 1.0})
        wrapper = LossWrapper(mock_custom)
        pred = torch.randn(2, 1)
        gt = torch.randn(2, 1)
        result = wrapper(pred, gt, normalize=True)
        assert mock_custom.call_count == 1
        assert result[0] is expected_loss

    def test_repr(self):
        loss_fn = nn.BCEWithLogitsLoss()
        wrapper = LossWrapper(loss_fn, name="bce")
        assert "bce" in repr(wrapper)


# ---------------------------------------------------------------------------
# build_compound_loss_from_config tests
# ---------------------------------------------------------------------------

class TestBuildCompoundLoss:
    def test_valid_config_returns_multiloss(self):
        cfg = _make_compound_cfg()
        result = build_compound_loss_from_config(cfg)
        assert isinstance(result, MultiLoss)

    def test_multiple_losses_in_multiloss(self):
        cfg = OmegaConf.create({
            "losses": [
                {"loss": {"_target_": "torch.nn.BCEWithLogitsLoss"}, "weight": 1.0},
                {"loss": {"_target_": "torch.nn.MSELoss"}, "weight": 0.5},
            ]
        })
        result = build_compound_loss_from_config(cfg)
        assert isinstance(result, MultiLoss)
        assert len(result.loss_funcs) == 2

    def test_missing_losses_raises_value_error(self):
        cfg = OmegaConf.create({"epoch_thresholds": [0, 5]})
        with pytest.raises(ValueError, match="losses"):
            build_compound_loss_from_config(cfg)

    def test_none_cfg_raises_value_error(self):
        with pytest.raises((ValueError, Exception)):
            build_compound_loss_from_config(None)

    def test_missing_loss_key_in_item_raises_value_error(self):
        cfg = OmegaConf.create({
            "losses": [{"weight": 1.0}]
        })
        with pytest.raises(ValueError, match="missing 'loss'"):
            build_compound_loss_from_config(cfg)


# ---------------------------------------------------------------------------
# build_loss_from_config tests
# ---------------------------------------------------------------------------

class TestBuildLossFromConfig:
    def test_compound_loss_path(self):
        cfg = _make_full_cfg(key="compound_loss")
        result = build_loss_from_config(cfg)
        assert isinstance(result, MultiLoss)

    def test_legacy_multi_loss_path(self):
        cfg = _make_full_cfg(key="multi_loss")
        mock_result = MagicMock(spec=MultiLoss)
        with patch(
            "pytorch_segmentation_models_trainer.custom_losses.base_loss.build_combined_loss",
            return_value=mock_result,
        ):
            result = build_loss_from_config(cfg)
            assert result is mock_result

    def test_no_valid_config_raises_value_error(self):
        cfg = OmegaConf.create({"loss_params": {}})
        with pytest.raises(ValueError):
            build_loss_from_config(cfg)


# ---------------------------------------------------------------------------
# validate_loss_config tests
# ---------------------------------------------------------------------------

class TestValidateLossConfig:
    def test_valid_config_returns_true(self):
        cfg = _make_compound_cfg()
        assert validate_loss_config(cfg) is True

    def test_missing_losses_raises_value_error(self):
        cfg = OmegaConf.create({"epoch_thresholds": [0, 5]})
        with pytest.raises(ValueError, match="losses"):
            validate_loss_config(cfg)

    def test_missing_target_raises_value_error(self):
        cfg = OmegaConf.create({
            "losses": [{"loss": {"name": "seg"}, "weight": 1.0}]
        })
        with pytest.raises(ValueError, match="_target_"):
            validate_loss_config(cfg)

    def test_duplicate_names_raises_value_error(self):
        cfg = OmegaConf.create({
            "losses": [
                {"loss": {"_target_": "torch.nn.BCEWithLogitsLoss", "name": "same"}, "weight": 1.0},
                {"loss": {"_target_": "torch.nn.MSELoss", "name": "same"}, "weight": 0.5},
            ]
        })
        with pytest.raises(ValueError, match="[Dd]uplicate"):
            validate_loss_config(cfg)
