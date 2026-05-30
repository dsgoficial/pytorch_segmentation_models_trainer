# -*- coding: utf-8 -*-
"""
Unit tests for EvidentialWrapper and is_evidential_model.

All tests run on CPU and use a minimal segmentation model (2-layer CNN)
to avoid any SMP / HuggingFace dependencies.
"""

import pytest
import torch
import torch.nn as nn

from pytorch_segmentation_models_trainer.custom_models.edl_wrapper import (
    EvidentialWrapper,
    is_evidential_model,
)

B, K, H, W = 2, 4, 32, 32


# ---------------------------------------------------------------------------
# Minimal segmentation model stub
# ---------------------------------------------------------------------------


class TinySegModel(nn.Module):
    """Minimal model that returns [B, K, H, W] logits."""

    def __init__(self, in_channels=3, num_classes=K):
        super().__init__()
        self.encoder = nn.Conv2d(in_channels, 8, 3, padding=1)
        self.decoder = nn.Conv2d(8, num_classes, 1)

    def forward(self, x):
        return self.decoder(torch.relu(self.encoder(x)))


class TinySegModelWithTupleOutput(nn.Module):
    """Model that returns a tuple (logits, aux) like some torchvision models."""

    def __init__(self, num_classes=K):
        super().__init__()
        self.conv = nn.Conv2d(3, num_classes, 1)

    def forward(self, x):
        out = self.conv(x)
        return (out, torch.zeros_like(out))


class TinySegModelWithDictOutput(nn.Module):
    """Model that returns a dict like torchvision DeepLab."""

    def __init__(self, num_classes=K):
        super().__init__()
        self.conv = nn.Conv2d(3, num_classes, 1)

    def forward(self, x):
        return {"out": self.conv(x)}


# ---------------------------------------------------------------------------
# Output structure
# ---------------------------------------------------------------------------


class TestEvidentialWrapperOutputKeys:
    def _make(self, freeze_encoder=False):
        return EvidentialWrapper(TinySegModel(), freeze_encoder=freeze_encoder)

    def test_output_is_dict(self):
        wrapper = self._make()
        x = torch.randn(B, 3, H, W)
        out = wrapper(x)
        assert isinstance(out, dict)

    def test_required_keys_present(self):
        wrapper = self._make()
        x = torch.randn(B, 3, H, W)
        out = wrapper(x)
        for key in ("logits", "evidence", "alpha", "probs", "uncertainty"):
            assert key in out, f"Missing key: {key}"

    def test_shapes(self):
        wrapper = self._make()
        x = torch.randn(B, 3, H, W)
        out = wrapper(x)
        assert out["logits"].shape == (B, K, H, W)
        assert out["evidence"].shape == (B, K, H, W)
        assert out["alpha"].shape == (B, K, H, W)
        assert out["probs"].shape == (B, K, H, W)
        assert out["uncertainty"].shape == (B, 1, H, W)


# ---------------------------------------------------------------------------
# Mathematical invariants
# ---------------------------------------------------------------------------


class TestEvidentialWrapperInvariants:
    def _make(self):
        return EvidentialWrapper(TinySegModel())

    def test_evidence_non_negative(self):
        wrapper = self._make()
        out = wrapper(torch.randn(B, 3, H, W))
        assert (out["evidence"] >= 0).all()

    def test_alpha_at_least_one(self):
        wrapper = self._make()
        out = wrapper(torch.randn(B, 3, H, W))
        assert (out["alpha"] >= 1.0 - 1e-6).all()

    def test_probs_sum_to_one(self):
        wrapper = self._make()
        out = wrapper(torch.randn(B, 3, H, W))
        probs_sum = out["probs"].sum(dim=1)
        assert torch.allclose(probs_sum, torch.ones_like(probs_sum), atol=1e-5)

    def test_uncertainty_in_range(self):
        wrapper = self._make()
        out = wrapper(torch.randn(B, 3, H, W))
        unc = out["uncertainty"]
        assert (unc > 0).all()
        assert (unc <= 1.0 + 1e-5).all()

    def test_max_uncertainty_from_very_negative_logits(self):
        """When logits are very negative, evidence ≈ 0, alpha ≈ 1, u → 1."""

        # Use a model that directly returns a fixed very-negative tensor
        class NegativeLogitModel(nn.Module):
            def __init__(self, k):
                super().__init__()
                self.k = k

            def forward(self, x):
                return torch.full((x.shape[0], self.k, x.shape[2], x.shape[3]), -100.0)

        wrapper = EvidentialWrapper(NegativeLogitModel(K))
        out = wrapper(torch.randn(B, 3, H, W))
        # Softplus(-100) ≈ 0 → alpha ≈ 1 → S = K → u = K/K = 1
        assert out["uncertainty"].mean().item() > 0.99

    def test_low_uncertainty_with_strong_logits(self):
        """Very large positive logit for one class → u should be low."""
        model = TinySegModel()
        # Zero out weights, set bias of class 0 to large value
        with torch.no_grad():
            model.decoder.weight.zero_()
            model.decoder.bias.zero_()
            model.decoder.bias[0] = 50.0
        wrapper = EvidentialWrapper(model)
        out = wrapper(torch.randn(B, 3, H, W))
        assert out["uncertainty"].mean().item() < 0.3


# ---------------------------------------------------------------------------
# Encoder freeze
# ---------------------------------------------------------------------------


class TestEncoderFreeze:
    def test_freeze_encoder_true(self):
        wrapper = EvidentialWrapper(TinySegModel(), freeze_encoder=True)
        encoder = wrapper._find_encoder()
        assert encoder is not None
        for param in encoder.parameters():
            assert not param.requires_grad

    def test_freeze_encoder_false(self):
        wrapper = EvidentialWrapper(TinySegModel(), freeze_encoder=False)
        encoder = wrapper._find_encoder()
        assert encoder is not None
        for param in encoder.parameters():
            assert param.requires_grad

    def test_unfreeze_encoder(self):
        wrapper = EvidentialWrapper(TinySegModel(), freeze_encoder=True)
        wrapper.unfreeze_encoder()
        encoder = wrapper._find_encoder()
        for param in encoder.parameters():
            assert param.requires_grad

    def test_decoder_always_trainable_when_frozen(self):
        wrapper = EvidentialWrapper(TinySegModel(), freeze_encoder=True)
        # decoder parameters should remain trainable
        decoder_params = list(wrapper.model.decoder.parameters())
        assert all(p.requires_grad for p in decoder_params)


# ---------------------------------------------------------------------------
# Handling of non-standard model outputs
# ---------------------------------------------------------------------------


class TestEvidentialWrapperOutputHandling:
    def test_tuple_output(self):
        model = TinySegModelWithTupleOutput()
        wrapper = EvidentialWrapper(model)
        out = wrapper(torch.randn(B, 3, H, W))
        assert out["logits"].shape == (B, K, H, W)

    def test_dict_output(self):
        model = TinySegModelWithDictOutput()
        wrapper = EvidentialWrapper(model)
        out = wrapper(torch.randn(B, 3, H, W))
        assert out["logits"].shape == (B, K, H, W)


# ---------------------------------------------------------------------------
# is_evidential_model
# ---------------------------------------------------------------------------


class TestIsEvidentialModel:
    def test_true_for_wrapper_instance(self):
        wrapper = EvidentialWrapper(TinySegModel())
        assert is_evidential_model(wrapper)

    def test_false_for_plain_model(self):
        model = TinySegModel()
        assert not is_evidential_model(model)

    def test_true_for_hydra_config_with_target(self):
        from omegaconf import OmegaConf

        cfg = OmegaConf.create(
            {
                "_target_": "pytorch_segmentation_models_trainer.custom_models.edl_wrapper.EvidentialWrapper",
                "freeze_encoder": False,
            }
        )
        assert is_evidential_model(cfg)

    def test_false_for_other_target(self):
        from omegaconf import OmegaConf

        cfg = OmegaConf.create({"_target_": "segmentation_models_pytorch.Unet"})
        assert not is_evidential_model(cfg)
