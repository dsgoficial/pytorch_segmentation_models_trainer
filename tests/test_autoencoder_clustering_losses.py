# -*- coding: utf-8 -*-
"""Tests for autoencoder clustering losses (DEC, Center, ClusteringAwareVAELoss).

References:
    - Xie et al., "Unsupervised Deep Embedding for Clustering Analysis", ICML 2016.
    - Guo et al., "Deep Convolutional Embedded Clustering", IJCAI 2017.
    - Wen et al., "A Discriminative Feature Learning Approach for Deep Face
      Recognition", ECCV 2016.
"""

import pytest
import torch
import torch.nn as nn

from pytorch_segmentation_models_trainer.custom_losses.autoencoder_clustering_losses import (
    CenterLoss,
    ClusteringAwareVAELoss,
    DECSoftAssignmentLoss,
)
from pytorch_segmentation_models_trainer.custom_losses.autoencoder_losses import (
    VariationalAutoencoderLoss,
)
from pytorch_segmentation_models_trainer.custom_models.variational_autoencoder import (
    VariationalAutoencoderOutput,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

N, D, K = 32, 16, 4  # samples, embedding dim, clusters


def _embeddings(n=N, d=D, requires_grad=False):
    return torch.randn(n, d, requires_grad=requires_grad)


def _centers(k=K, d=D):
    return nn.Parameter(torch.randn(k, d))


def _vae_output(batch=4, channels=3, h=16, w=16, latent_dim=D):
    recon = torch.randn(batch, channels, h, w, requires_grad=True)
    mu = torch.randn(batch, latent_dim, requires_grad=True)
    logvar = torch.zeros(batch, latent_dim)
    z = mu + torch.randn_like(mu) * (0.5 * logvar).exp()
    return VariationalAutoencoderOutput(reconstruction=recon, mu=mu, logvar=logvar, z=z)


def _spatial_vae_output(batch=4, channels=3, h=16, w=16, latent_c=8, latent_h=4):
    recon = torch.randn(batch, channels, h, w, requires_grad=True)
    mu = torch.randn(batch, latent_c, latent_h, latent_h, requires_grad=True)
    logvar = torch.zeros_like(mu)
    z = mu + torch.randn_like(mu) * (0.5 * logvar).exp()
    return VariationalAutoencoderOutput(reconstruction=recon, mu=mu, logvar=logvar, z=z)


# ===========================================================================
# DECSoftAssignmentLoss
# ===========================================================================


class TestDECSoftAssignmentLoss:
    def test_output_keys(self):
        loss_fn = DECSoftAssignmentLoss(n_clusters=K, embedding_dim=D)
        result = loss_fn(_embeddings(), _centers())
        assert "loss" in result

    def test_loss_is_scalar(self):
        loss_fn = DECSoftAssignmentLoss(n_clusters=K, embedding_dim=D)
        result = loss_fn(_embeddings(), _centers())
        assert result["loss"].shape == ()

    def test_kl_nonnegative(self):
        loss_fn = DECSoftAssignmentLoss(n_clusters=K, embedding_dim=D)
        result = loss_fn(_embeddings(), _centers())
        assert result["loss"].item() >= 0.0

    def test_q_is_valid_distribution(self):
        loss_fn = DECSoftAssignmentLoss(n_clusters=K, embedding_dim=D)
        emb = _embeddings()
        centers = _centers()
        q = loss_fn._compute_q(emb, centers)
        row_sums = q.sum(dim=1)
        assert torch.allclose(row_sums, torch.ones(N), atol=1e-5)

    def test_p_is_valid_distribution(self):
        loss_fn = DECSoftAssignmentLoss(n_clusters=K, embedding_dim=D)
        emb = _embeddings()
        centers = _centers()
        q = loss_fn._compute_q(emb, centers)
        p = loss_fn._compute_p(q)
        row_sums = p.sum(dim=1)
        assert torch.allclose(row_sums, torch.ones(N), atol=1e-5)

    def test_gradient_flows_to_embeddings(self):
        loss_fn = DECSoftAssignmentLoss(n_clusters=K, embedding_dim=D)
        emb = _embeddings(requires_grad=True)
        centers = _centers()
        result = loss_fn(emb, centers)
        result["loss"].backward()
        assert emb.grad is not None

    def test_gradient_flows_to_centers(self):
        loss_fn = DECSoftAssignmentLoss(n_clusters=K, embedding_dim=D)
        emb = _embeddings()
        centers = _centers()
        result = loss_fn(emb, centers)
        result["loss"].backward()
        assert centers.grad is not None

    def test_alpha_changes_q_values(self):
        loss_fn1 = DECSoftAssignmentLoss(n_clusters=K, embedding_dim=D, alpha=1.0)
        loss_fn2 = DECSoftAssignmentLoss(n_clusters=K, embedding_dim=D, alpha=2.0)
        emb = _embeddings()
        centers = _centers()
        q1 = loss_fn1._compute_q(emb, centers.detach())
        q2 = loss_fn2._compute_q(emb, centers.detach())
        assert not torch.allclose(q1, q2)

    def test_initialize_centers_sets_data(self):
        loss_fn = DECSoftAssignmentLoss(n_clusters=K, embedding_dim=D)
        new_centers = torch.randn(K, D)
        old_data = loss_fn.cluster_centers.data.clone()
        loss_fn.initialize_centers(new_centers)
        assert not torch.allclose(loss_fn.cluster_centers.data, old_data)
        assert torch.allclose(loss_fn.cluster_centers.data, new_centers)

    def test_invalid_n_clusters(self):
        with pytest.raises(ValueError, match="n_clusters"):
            DECSoftAssignmentLoss(n_clusters=1, embedding_dim=D)

    def test_invalid_alpha(self):
        with pytest.raises(ValueError, match="alpha"):
            DECSoftAssignmentLoss(n_clusters=K, embedding_dim=D, alpha=0.0)

    def test_clusters_must_match_embedding_dim(self):
        loss_fn = DECSoftAssignmentLoss(n_clusters=K, embedding_dim=D)
        emb = _embeddings(d=D)
        wrong_centers = nn.Parameter(torch.randn(K, D + 1))
        with pytest.raises(RuntimeError):
            loss_fn(emb, wrong_centers)


# ===========================================================================
# CenterLoss
# ===========================================================================


class TestCenterLoss:
    def test_output_keys(self):
        loss_fn = CenterLoss(n_clusters=K, embedding_dim=D)
        result = loss_fn(_embeddings(), _centers())
        assert "loss" in result
        assert "raw_center_loss" in result

    def test_loss_is_scalar(self):
        loss_fn = CenterLoss(n_clusters=K, embedding_dim=D)
        result = loss_fn(_embeddings(), _centers())
        assert result["loss"].shape == ()

    def test_loss_nonnegative(self):
        loss_fn = CenterLoss(n_clusters=K, embedding_dim=D)
        result = loss_fn(_embeddings(), _centers())
        assert result["loss"].item() >= 0.0

    def test_zero_when_embeddings_at_centers(self):
        loss_fn = CenterLoss(n_clusters=K, embedding_dim=D)
        centers_data = torch.randn(K, D)
        centers = nn.Parameter(centers_data.clone())
        # Each embedding equals one of the centers
        idx = torch.arange(N) % K
        emb = centers_data[idx]
        result = loss_fn(emb, centers)
        assert result["loss"].item() < 1e-6

    def test_gradient_to_embeddings(self):
        loss_fn = CenterLoss(n_clusters=K, embedding_dim=D)
        emb = _embeddings(requires_grad=True)
        result = loss_fn(emb, _centers())
        result["loss"].backward()
        assert emb.grad is not None

    def test_gradient_to_centers(self):
        loss_fn = CenterLoss(n_clusters=K, embedding_dim=D)
        emb = _embeddings()
        centers = _centers()
        result = loss_fn(emb, centers)
        result["loss"].backward()
        assert centers.grad is not None

    def test_lambda_scales_loss(self):
        emb = _embeddings()
        centers_data = torch.randn(K, D)
        loss_fn1 = CenterLoss(n_clusters=K, embedding_dim=D, lambda_center=1.0)
        loss_fn2 = CenterLoss(n_clusters=K, embedding_dim=D, lambda_center=2.0)
        r1 = loss_fn1(emb, nn.Parameter(centers_data.clone()))
        r2 = loss_fn2(emb, nn.Parameter(centers_data.clone()))
        assert torch.allclose(r2["loss"], r1["loss"] * 2, atol=1e-5)

    def test_invalid_n_clusters(self):
        with pytest.raises(ValueError, match="n_clusters"):
            CenterLoss(n_clusters=1, embedding_dim=D)

    def test_invalid_lambda(self):
        with pytest.raises(ValueError, match="lambda_center"):
            CenterLoss(n_clusters=K, embedding_dim=D, lambda_center=-1.0)


# ===========================================================================
# ClusteringAwareVAELoss
# ===========================================================================


class TestClusteringAwareVAELoss:
    def _make_loss(self, **kwargs):
        defaults = dict(
            n_clusters=K,
            embedding_dim=D,
            gamma=0.1,
            delta=0.01,
            reconstruction_loss="mse",
        )
        defaults.update(kwargs)
        return ClusteringAwareVAELoss(**defaults)

    def test_output_keys_present(self):
        loss_fn = self._make_loss()
        output = _vae_output()
        target = torch.randn_like(output.reconstruction)
        result = loss_fn(output, target)
        for key in (
            "loss",
            "reconstruction_loss",
            "kl_loss",
            "dec_loss",
            "center_loss",
        ):
            assert key in result, f"missing key: {key}"

    def test_total_loss_is_scalar(self):
        loss_fn = self._make_loss()
        output = _vae_output()
        result = loss_fn(output, torch.randn_like(output.reconstruction))
        assert result["loss"].shape == ()

    def test_backward_completes(self):
        loss_fn = self._make_loss()
        output = _vae_output()
        result = loss_fn(output, torch.randn_like(output.reconstruction))
        result["loss"].backward()
        assert output.mu.grad is not None

    def test_vae_latent_mu(self):
        loss_fn = self._make_loss(vae_latent="mu")
        output = _vae_output()
        # Should not raise
        loss_fn(output, torch.randn_like(output.reconstruction))

    def test_vae_latent_z(self):
        loss_fn = self._make_loss(vae_latent="z")
        output = _vae_output()
        loss_fn(output, torch.randn_like(output.reconstruction))

    def test_gamma_zero_omits_dec_loss(self):
        loss_fn = self._make_loss(gamma=0.0)
        output = _vae_output()
        result = loss_fn(output, torch.randn_like(output.reconstruction))
        assert result["dec_loss"].item() == 0.0

    def test_delta_zero_omits_center_loss(self):
        loss_fn = self._make_loss(delta=0.0)
        output = _vae_output()
        result = loss_fn(output, torch.randn_like(output.reconstruction))
        assert result["center_loss"].item() == 0.0

    def test_spatial_latent_4d(self):
        loss_fn = ClusteringAwareVAELoss(
            n_clusters=K,
            embedding_dim=8,
            gamma=0.1,
            delta=0.01,
            reconstruction_loss="mse",
        )
        output = _spatial_vae_output(latent_c=8, latent_h=4)
        result = loss_fn(output, torch.randn_like(output.reconstruction))
        assert result["loss"].shape == ()

    def test_type_error_wrong_output(self):
        loss_fn = self._make_loss()
        with pytest.raises(TypeError):
            loss_fn(torch.randn(4, 3, 16, 16), torch.randn(4, 3, 16, 16))

    def test_pure_reconstruction_matches_vae_loss(self):
        output = _vae_output()
        target = torch.randn_like(output.reconstruction)

        wrapper = ClusteringAwareVAELoss(
            n_clusters=K,
            embedding_dim=D,
            gamma=0.0,
            delta=0.0,
            beta=0.0,
            reconstruction_loss="mse",
            reconstruction_weight=1.0,
        )
        ref_loss = VariationalAutoencoderLoss(
            reconstruction_loss="mse", reconstruction_weight=1.0, beta=0.0
        )

        wrapper_result = wrapper(output, target)
        ref_result = ref_loss(output, target)

        assert torch.allclose(wrapper_result["loss"], ref_result["loss"], atol=1e-6)

    def test_initialize_centers_from_embeddings(self):
        loss_fn = self._make_loss()
        emb = torch.randn(64, D)
        old = loss_fn.cluster_centers.data.clone()
        loss_fn.initialize_centers_from_embeddings(emb)
        assert not torch.allclose(loss_fn.cluster_centers.data, old)

    def test_single_cluster_centers_parameter(self):
        loss_fn = self._make_loss()
        named = dict(loss_fn.named_parameters())
        center_keys = [k for k in named if "cluster_centers" in k]
        assert (
            len(center_keys) == 1
        ), f"expected 1 cluster_centers param, got {center_keys}"

    def test_invalid_vae_latent(self):
        with pytest.raises(ValueError, match="vae_latent"):
            self._make_loss(vae_latent="invalid")

    def test_invalid_latent_reduction(self):
        with pytest.raises(ValueError, match="latent_reduction"):
            self._make_loss(latent_reduction="invalid")

    def test_spatial_latent_flatten_mode(self):
        loss_fn = ClusteringAwareVAELoss(
            n_clusters=K,
            embedding_dim=8 * 4 * 4,
            gamma=0.1,
            delta=0.01,
            reconstruction_loss="mse",
            latent_reduction="flatten",
        )
        output = _spatial_vae_output(latent_c=8, latent_h=4)
        result = loss_fn(output, torch.randn_like(output.reconstruction))
        assert result["loss"].shape == ()

    def test_invalid_latent_ndim_raises(self):
        loss_fn = self._make_loss()
        output = _vae_output()
        # Replace mu with a 3-D tensor to trigger ValueError in _reduce_latents
        bad_mu = torch.randn(4, 8, 4)
        bad_output = VariationalAutoencoderOutput(
            reconstruction=output.reconstruction,
            mu=bad_mu,
            logvar=torch.zeros(4, 8, 4),
            z=bad_mu,
        )
        with pytest.raises(ValueError, match="2-D"):
            loss_fn(bad_output, torch.randn_like(output.reconstruction))
