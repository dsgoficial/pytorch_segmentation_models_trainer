# -*- coding: utf-8 -*-
"""
Tests for TerraTorchSegmentationWrapper (Phase 4).

TerraTorch is mocked so these tests run without installing terratorch.
The _prepare_input helper and head wiring are tested with a fake backbone.
"""

from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn as nn

# ─── Helpers ───────────────────────────────────────────────────────��─────────

B, C_IN, H, W = 2, 6, 64, 64
EMBED_DIM = 32
PATCH_SIZE = 16
NUM_TOKENS = (H // PATCH_SIZE) * (W // PATCH_SIZE)  # 16


def _make_fake_terratorch_module():
    """A tiny nn.Module that mimics a TerraTorch backbone returning patch tokens."""

    class _FakeBackbone(nn.Module):
        embed_dim = EMBED_DIM
        patch_size = PATCH_SIZE

        def forward(self, x):
            # x: (B, C, T, H, W)
            B = x.shape[0]
            H_p = x.shape[-2] // PATCH_SIZE
            W_p = x.shape[-1] // PATCH_SIZE
            tokens = torch.randn(B, H_p * W_p, EMBED_DIM)
            return tokens

    return _FakeBackbone()


def _make_fake_factory(backbone):
    factory = MagicMock()
    factory.build_model.return_value = backbone
    return factory


# ─── Import under mock ──────────────────��─────────────────────────────────────


def _build_wrapper(backbone, num_classes=2, head_type="linear"):
    """Build TerraTorchSegmentationWrapper with a mocked factory."""
    fake_factory = _make_fake_factory(backbone)
    with patch.dict(
        "sys.modules",
        {
            "terratorch": MagicMock(),
            "terratorch.models": MagicMock(),
            "terratorch.models.backbones": MagicMock(),
            "terratorch.models.backbones.prithvi_model_factory": MagicMock(
                PrithviModelFactory=MagicMock(return_value=fake_factory)
            ),
        },
    ):
        from pytorch_segmentation_models_trainer.custom_models.terratorch_models import (
            TerraTorchSegmentationWrapper,
        )

        wrapper = TerraTorchSegmentationWrapper(
            backbone_name="prithvi_100M",
            num_classes=num_classes,
            in_channels=C_IN,
            num_frames=1,
            img_size=H,
            head_type=head_type,
            pretrained_path=None,
        )
    # Replace the backbone with our fake one directly
    wrapper.backbone = backbone
    return wrapper


# ─── Tests ─────────────────────────────────────────────────────────────��─────


class TestTerraTorchWrapper:
    def test_import_error_when_terratorch_missing(self):
        with patch.dict("sys.modules", {"terratorch": None}):
            with pytest.raises(ImportError, match="terratorch"):
                from pytorch_segmentation_models_trainer.custom_models.terratorch_models import (
                    TerraTorchSegmentationWrapper,
                )

                TerraTorchSegmentationWrapper(
                    backbone_name="prithvi_100M",
                    num_classes=2,
                    in_channels=6,
                    num_frames=1,
                    img_size=64,
                    head_type="linear",
                )

    def test_prepare_input_4d_to_5d(self):
        backbone = _make_fake_terratorch_module()
        wrapper = _build_wrapper(backbone)
        x = torch.randn(B, C_IN, H, W)
        x5d = wrapper._prepare_input(x)
        assert x5d.ndim == 5
        assert x5d.shape == (B, C_IN, 1, H, W)

    def test_prepare_input_5d_unchanged(self):
        backbone = _make_fake_terratorch_module()
        wrapper = _build_wrapper(backbone)
        x = torch.randn(B, C_IN, 2, H, W)
        x5d = wrapper._prepare_input(x)
        assert x5d.shape == (B, C_IN, 2, H, W)

    def test_forward_single_temporal(self):
        backbone = _make_fake_terratorch_module()
        wrapper = _build_wrapper(backbone, num_classes=2, head_type="linear")
        x = torch.randn(B, C_IN, H, W)
        with torch.no_grad():
            out = wrapper(x)
        assert isinstance(out, torch.Tensor)
        assert out.shape[0] == B
        assert out.shape[1] == 2

    def test_forward_output_spatial_size(self):
        backbone = _make_fake_terratorch_module()
        wrapper = _build_wrapper(backbone, num_classes=2, head_type="linear")
        x = torch.randn(B, C_IN, H, W)
        with torch.no_grad():
            out = wrapper(x)
        assert out.shape[-2] == H
        assert out.shape[-1] == W

    def test_unsupported_head_type_raises(self):
        # head_type="fpn" requires SMP – just validate the error message
        backbone = _make_fake_terratorch_module()
        with pytest.raises((ValueError, ImportError)):
            _build_wrapper(backbone, head_type="invalid_head")
