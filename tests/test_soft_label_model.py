# -*- coding: utf-8 -*-
"""Tests for SoftLabelModel._shared_step dict-unwrap and _compute_loss re-wrap."""

import pytest
import torch
from unittest.mock import MagicMock, patch, call

from pytorch_segmentation_models_trainer.model_loader.soft_label_model import (
    SoftLabelModel,
)
from pytorch_segmentation_models_trainer.custom_losses.soft_label_loss import (
    SoftLabelWeightedCELoss,
)
from pytorch_segmentation_models_trainer.custom_losses.base_loss import MultiLoss

B, C, H, W = 2, 4, 8, 8


def _mock_self():
    """Build a minimal SoftLabelModel instance for unit testing unbound methods.

    Uses object.__new__ + nn.Module.__init__ to satisfy:
    - isinstance(obj, SoftLabelModel) == True  (required by super())
    - nn.Module internal state initialised  (required by __setattr__)
    - Model.__init__ bypassed  (avoids dataset/loss instantiation)
    """
    model = object.__new__(SoftLabelModel)
    torch.nn.Module.__init__(model)
    model.cfg = MagicMock()
    model.cfg.get.side_effect = lambda key, default=None: default
    model._w_conf = None
    return model


def _p_soft():
    raw = torch.rand(B, C, H, W)
    return raw / raw.sum(dim=1, keepdim=True)


def _w_conf():
    return torch.rand(B, 1, H, W)


# ---------------------------------------------------------------------------
# _shared_step: dict-mask unwrap
# ---------------------------------------------------------------------------


class TestSharedStepUnwrap:
    def test_w_conf_stored_when_dict_mask(self):
        model = _mock_self()
        p = _p_soft()
        w = _w_conf()
        batch = {"image": torch.rand(B, 3, H, W), "mask": {"mask": p, "w_conf": w}}

        with patch.object(
            SoftLabelModel.__bases__[0], "_shared_step", return_value=None
        ) as mock_parent:
            SoftLabelModel._shared_step(model, batch, "train")

        assert model._w_conf is not None
        assert torch.equal(model._w_conf, w)

    def test_batch_mask_flattened_before_parent(self):
        model = _mock_self()
        p = _p_soft()
        w = _w_conf()
        batch = {"image": torch.rand(B, 3, H, W), "mask": {"mask": p, "w_conf": w}}

        with patch.object(
            SoftLabelModel.__bases__[0], "_shared_step", return_value=None
        ) as mock_parent:
            SoftLabelModel._shared_step(model, batch, "train")

        called_batch = mock_parent.call_args[0][0]
        assert isinstance(called_batch["mask"], torch.Tensor)
        assert torch.equal(called_batch["mask"], p)

    def test_w_conf_none_when_plain_tensor_mask(self):
        model = _mock_self()
        p = _p_soft()
        batch = {"image": torch.rand(B, 3, H, W), "mask": p}

        with patch.object(
            SoftLabelModel.__bases__[0], "_shared_step", return_value=None
        ):
            SoftLabelModel._shared_step(model, batch, "train")

        assert model._w_conf is None

    def test_plain_tensor_mask_unchanged_in_batch(self):
        model = _mock_self()
        p = _p_soft()
        batch = {"image": torch.rand(B, 3, H, W), "mask": p}

        with patch.object(
            SoftLabelModel.__bases__[0], "_shared_step", return_value=None
        ) as mock_parent:
            SoftLabelModel._shared_step(model, batch, "train")

        called_batch = mock_parent.call_args[0][0]
        assert called_batch["mask"] is p

    def test_prefix_forwarded_to_parent(self):
        model = _mock_self()
        batch = {"image": torch.rand(B, 3, H, W), "mask": _p_soft()}

        with patch.object(
            SoftLabelModel.__bases__[0], "_shared_step", return_value=None
        ) as mock_parent:
            SoftLabelModel._shared_step(model, batch, "val")

        assert mock_parent.call_args[0][1] == "val"

    def test_dict_mask_without_w_conf_key(self):
        """Mask dict may omit w_conf (dataset without w_conf column)."""
        model = _mock_self()
        p = _p_soft()
        batch = {"image": torch.rand(B, 3, H, W), "mask": {"mask": p}}

        with patch.object(
            SoftLabelModel.__bases__[0], "_shared_step", return_value=None
        ):
            SoftLabelModel._shared_step(model, batch, "train")

        # w_conf is None when key is absent
        assert model._w_conf is None


# ---------------------------------------------------------------------------
# _compute_loss: dict re-wrap
# ---------------------------------------------------------------------------


class TestComputeLossRewrap:
    def test_parent_receives_dict_when_w_conf_set(self):
        model = _mock_self()
        model._w_conf = _w_conf()
        p = _p_soft()
        pred = torch.randn(B, C, H, W)

        with patch.object(
            SoftLabelModel.__bases__[0],
            "_compute_loss",
            return_value=(torch.tensor(0.0), {}, {}),
        ) as mock_parent:
            SoftLabelModel._compute_loss(model, pred, p)

        called_masks = mock_parent.call_args[0][1]
        assert isinstance(called_masks, dict)
        assert "mask" in called_masks and "w_conf" in called_masks
        assert called_masks["mask"] is p
        assert torch.equal(called_masks["w_conf"], model._w_conf)

    def test_parent_receives_tensor_when_no_w_conf(self):
        model = _mock_self()
        model._w_conf = None
        p = _p_soft()
        pred = torch.randn(B, C, H, W)

        with patch.object(
            SoftLabelModel.__bases__[0],
            "_compute_loss",
            return_value=(torch.tensor(0.0), {}, {}),
        ) as mock_parent:
            SoftLabelModel._compute_loss(model, pred, p)

        called_masks = mock_parent.call_args[0][1]
        assert called_masks is p

    def test_output_forwarded_unchanged(self):
        model = _mock_self()
        model._w_conf = None
        expected = (torch.tensor(1.5), {"loss": torch.tensor(1.5)}, {})

        with patch.object(
            SoftLabelModel.__bases__[0], "_compute_loss", return_value=expected
        ):
            result = SoftLabelModel._compute_loss(
                model, torch.randn(B, C, H, W), _p_soft()
            )

        assert result is expected
