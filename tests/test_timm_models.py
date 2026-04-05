# -*- coding: utf-8 -*-
"""
Tests for TimmEncoderWithSMPDecoder (Phase 3).

Uses a real tiny timm model (resnet18) to validate the full forward pass
on CPU with small tensors.  No pretrained weights are downloaded
(pretrained=False).
"""
import pytest
import torch

try:
    import timm  # noqa: F401
    TIMM_AVAILABLE = True
except ImportError:
    TIMM_AVAILABLE = False

try:
    import segmentation_models_pytorch  # noqa: F401
    SMP_AVAILABLE = True
except ImportError:
    SMP_AVAILABLE = False

pytestmark = pytest.mark.skipif(
    not (TIMM_AVAILABLE and SMP_AVAILABLE),
    reason="timm and segmentation-models-pytorch are required",
)

from pytorch_segmentation_models_trainer.custom_models.timm_models import (
    TimmEncoderWithSMPDecoder,
)

B, H, W = 2, 128, 128


class TestTimmEncoderWithSMPDecoder:
    def test_unet_decoder_instantiation(self):
        model = TimmEncoderWithSMPDecoder(
            timm_model_name="resnet18",
            num_classes=2,
            decoder_type="unet",
            pretrained=False,
        )
        assert model is not None

    def test_unet_forward_shape(self):
        model = TimmEncoderWithSMPDecoder(
            timm_model_name="resnet18",
            num_classes=2,
            decoder_type="unet",
            pretrained=False,
        )
        model.eval()
        x = torch.randn(B, 3, H, W)
        with torch.no_grad():
            out = model(x)
        assert out.shape[0] == B
        assert out.shape[1] == 2

    def test_fpn_decoder_instantiation(self):
        model = TimmEncoderWithSMPDecoder(
            timm_model_name="resnet18",
            num_classes=3,
            decoder_type="fpn",
            pretrained=False,
        )
        assert model is not None

    def test_fpn_forward_shape(self):
        model = TimmEncoderWithSMPDecoder(
            timm_model_name="resnet18",
            num_classes=3,
            decoder_type="fpn",
            pretrained=False,
        )
        model.eval()
        x = torch.randn(B, 3, H, W)
        with torch.no_grad():
            out = model(x)
        assert out.shape[0] == B
        assert out.shape[1] == 3

    def test_unsupported_decoder_raises(self):
        with pytest.raises(ValueError, match="Unsupported decoder_type"):
            TimmEncoderWithSMPDecoder(
                timm_model_name="resnet18",
                num_classes=2,
                decoder_type="deeplabv3",  # not supported in wrapper
                pretrained=False,
            )

    def test_custom_in_channels(self):
        model = TimmEncoderWithSMPDecoder(
            timm_model_name="resnet18",
            num_classes=2,
            decoder_type="unet",
            in_channels=6,
            pretrained=False,
        )
        model.eval()
        x = torch.randn(B, 6, H, W)
        with torch.no_grad():
            out = model(x)
        assert out.shape[0] == B
        assert out.shape[1] == 2

    def test_import_error_when_timm_missing(self):
        from unittest.mock import patch
        with patch.dict("sys.modules", {"timm": None}):
            with pytest.raises(ImportError, match="timm"):
                TimmEncoderWithSMPDecoder(
                    timm_model_name="resnet18",
                    num_classes=2,
                    pretrained=False,
                )


class TestSMPTuPrefix:
    """Validate that the SMP 'tu-' prefix path works end-to-end."""

    def test_smp_unet_with_tu_resnet18(self):
        import segmentation_models_pytorch as smp
        model = smp.Unet(
            encoder_name="tu-resnet18",
            encoder_weights=None,
            in_channels=3,
            classes=2,
        )
        model.eval()
        x = torch.randn(B, 3, H, W)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (B, 2, H, W)
