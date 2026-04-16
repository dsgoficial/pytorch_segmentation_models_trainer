# -*- coding: utf-8 -*-
"""
Unit tests for EDL utility functions (pytorch_segmentation_models_trainer.custom_losses.edl_utils).

All tests run on CPU and require no external data.
"""

import pytest
import torch

from pytorch_segmentation_models_trainer.custom_losses.edl_utils import (
    dirichlet_kl_divergence,
    dirichlet_strength,
    edl_kl_regulariser,
    epistemic_uncertainty,
    kl_divergence_to_uniform,
    one_hot_encode,
)

B, K, H, W = 2, 4, 8, 8


# ---------------------------------------------------------------------------
# one_hot_encode
# ---------------------------------------------------------------------------


class TestOneHotEncode:
    def test_hard_labels_shape(self):
        labels = torch.zeros(B, H, W, dtype=torch.long)
        out = one_hot_encode(labels, K)
        assert out.shape == (B, K, H, W)

    def test_hard_labels_values(self):
        labels = torch.zeros(B, H, W, dtype=torch.long)
        labels[0, :, :] = 1  # second class
        out = one_hot_encode(labels, K)
        assert out[0, 1, 0, 0].item() == 1.0
        assert out[0, 0, 0, 0].item() == 0.0

    def test_hard_labels_sum_to_one(self):
        labels = torch.randint(0, K, (B, H, W))
        out = one_hot_encode(labels, K)
        assert torch.allclose(out.sum(dim=1), torch.ones(B, H, W))

    def test_soft_labels_passthrough(self):
        soft = torch.rand(B, K, H, W)
        out = one_hot_encode(soft, K)
        assert out.shape == (B, K, H, W)
        assert torch.allclose(out, soft.float())

    def test_ignore_index_zeroed(self):
        labels = torch.zeros(B, H, W, dtype=torch.long)
        labels[0, 0, 0] = 255  # ignore pixel
        out = one_hot_encode(labels, K)
        # Ignore pixel should be all zeros
        assert out[0, :, 0, 0].sum().item() == 0.0

    def test_wrong_dim_raises(self):
        labels = torch.zeros(B, K, H, W, dtype=torch.long)  # 4-D int tensor
        with pytest.raises(ValueError):
            one_hot_encode(labels, K)


# ---------------------------------------------------------------------------
# dirichlet_strength
# ---------------------------------------------------------------------------


class TestDirichletStrength:
    def test_shape(self):
        alpha = torch.ones(B, K, H, W) * 2.0
        S = dirichlet_strength(alpha)
        assert S.shape == (B, 1, H, W)

    def test_value_uniform_alpha(self):
        alpha = torch.ones(B, K, H, W) * 3.0
        S = dirichlet_strength(alpha)
        expected = K * 3.0
        assert torch.allclose(S, torch.full_like(S, expected))


# ---------------------------------------------------------------------------
# epistemic_uncertainty
# ---------------------------------------------------------------------------


class TestEpistemicUncertainty:
    def test_shape(self):
        alpha = torch.ones(B, K, H, W)
        u = epistemic_uncertainty(alpha)
        assert u.shape == (B, 1, H, W)

    def test_max_when_no_evidence(self):
        # alpha_k = 1 for all k → S = K → u = K/K = 1
        alpha = torch.ones(B, K, H, W)
        u = epistemic_uncertainty(alpha)
        assert torch.allclose(u, torch.ones_like(u))

    def test_decreases_with_evidence(self):
        # High evidence on one class → low uncertainty
        alpha_low = torch.ones(B, K, H, W)
        alpha_high = torch.ones(B, K, H, W)
        alpha_high[:, 0, :, :] = 100.0
        u_low = epistemic_uncertainty(alpha_low)
        u_high = epistemic_uncertainty(alpha_high)
        assert (u_high < u_low).all()

    def test_range(self):
        alpha = torch.rand(B, K, H, W).abs() + 1.0
        u = epistemic_uncertainty(alpha)
        assert (u > 0).all()
        assert (u <= 1.0 + 1e-5).all()


# ---------------------------------------------------------------------------
# dirichlet_kl_divergence
# ---------------------------------------------------------------------------


class TestDirichletKLDivergence:
    def test_shape(self):
        alpha = torch.ones(B, K, H, W) * 2.0
        beta = torch.ones(B, K, H, W)
        kl = dirichlet_kl_divergence(alpha, beta)
        assert kl.shape == (B, H, W)

    def test_identical_distributions_zero(self):
        alpha = torch.ones(B, K, H, W) * 3.0
        kl = dirichlet_kl_divergence(alpha, alpha)
        assert torch.allclose(kl, torch.zeros_like(kl), atol=1e-5)

    def test_non_negative(self):
        alpha = torch.rand(B, K, H, W).abs() + 1.0
        beta = torch.rand(B, K, H, W).abs() + 1.0
        kl = dirichlet_kl_divergence(alpha, beta)
        assert (kl >= 0).all()


# ---------------------------------------------------------------------------
# kl_divergence_to_uniform
# ---------------------------------------------------------------------------


class TestKLToUniform:
    def test_uniform_alpha_zero(self):
        # Dir(1,...,1) KL with itself = 0
        alpha = torch.ones(B, K, H, W)
        kl = kl_divergence_to_uniform(alpha)
        assert torch.allclose(kl, torch.zeros_like(kl), atol=1e-5)

    def test_non_uniform_positive(self):
        alpha = torch.ones(B, K, H, W)
        alpha[:, 0, :, :] = 10.0  # concentrated distribution
        kl = kl_divergence_to_uniform(alpha)
        assert (kl > 0).all()


# ---------------------------------------------------------------------------
# edl_kl_regulariser
# ---------------------------------------------------------------------------


class TestEDLKLRegulariser:
    def test_shape(self):
        alpha = torch.ones(B, K, H, W) * 2.0
        y = one_hot_encode(torch.zeros(B, H, W, dtype=torch.long), K)
        kl = edl_kl_regulariser(alpha, y)
        assert kl.shape == (B, H, W)

    def test_non_negative(self):
        alpha = torch.rand(B, K, H, W).abs() + 1.0
        y = one_hot_encode(torch.randint(0, K, (B, H, W)), K)
        kl = edl_kl_regulariser(alpha, y)
        assert (kl >= 0).all()

    def test_correct_high_confidence_not_penalised_heavily(self):
        """KL should be lower when evidence is concentrated on the correct class."""
        y = one_hot_encode(torch.zeros(B, H, W, dtype=torch.long), K)  # class 0

        # Correct class has high evidence
        alpha_correct = torch.ones(B, K, H, W)
        alpha_correct[:, 0, :, :] = 50.0  # class 0

        # Wrong class has high evidence
        alpha_wrong = torch.ones(B, K, H, W)
        alpha_wrong[:, 1, :, :] = 50.0  # class 1

        kl_correct = edl_kl_regulariser(alpha_correct, y)
        kl_wrong = edl_kl_regulariser(alpha_wrong, y)

        assert kl_correct.mean() < kl_wrong.mean()
