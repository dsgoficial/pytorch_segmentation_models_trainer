# -*- coding: utf-8 -*-
"""Tests for ClusterCentersWarmStartCallback."""

from unittest.mock import MagicMock

import pytest
import torch

from pytorch_segmentation_models_trainer.config_definitions.autoencoder_config import (
    ClusterCentersWarmStartCallbackConfig,
)
from pytorch_segmentation_models_trainer.custom_callbacks.cluster_centers_warm_start_callback import (
    ClusterCentersWarmStartCallback,
)
from pytorch_segmentation_models_trainer.custom_losses.autoencoder_clustering_losses import (
    ClusteringAwareVAELoss,
)
from pytorch_segmentation_models_trainer.custom_models.variational_autoencoder import (
    VariationalAutoencoderOutput,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

N_CLUSTERS, EMB_DIM = 4, 8
BATCH_SIZE, C, H, W = 4, 3, 16, 16


def _make_loss(**kwargs):
    defaults = dict(
        n_clusters=N_CLUSTERS,
        embedding_dim=EMB_DIM,
        gamma=0.1,
        delta=0.01,
        reconstruction_loss="mse",
    )
    defaults.update(kwargs)
    return ClusteringAwareVAELoss(**defaults)


def _flat_vae_output(batch=BATCH_SIZE):
    mu = torch.randn(batch, EMB_DIM)
    logvar = torch.zeros(batch, EMB_DIM)
    return VariationalAutoencoderOutput(
        reconstruction=torch.randn(batch, C, H, W),
        mu=mu,
        logvar=logvar,
        z=mu + 0.1 * torch.randn_like(mu),
    )


def _spatial_vae_output(batch=BATCH_SIZE, latent_c=EMB_DIM, latent_h=4):
    mu = torch.randn(batch, latent_c, latent_h, latent_h)
    logvar = torch.zeros_like(mu)
    return VariationalAutoencoderOutput(
        reconstruction=torch.randn(batch, C, H, W),
        mu=mu,
        logvar=logvar,
        z=mu + 0.1 * torch.randn_like(mu),
    )


def _make_trainer(n_batches=8, batch_size=BATCH_SIZE):
    trainer = MagicMock()
    trainer.train_dataloader = [
        {"image": torch.randn(batch_size, C, H, W)} for _ in range(n_batches)
    ]
    return trainer


def _make_pl_module(loss, output_factory=None):
    if output_factory is None:
        output_factory = lambda images: _flat_vae_output(batch=images.shape[0])
    pl_module = MagicMock()
    pl_module.loss_function = loss
    pl_module.side_effect = output_factory
    return pl_module


# ===========================================================================
# Happy path
# ===========================================================================


def test_warm_start_changes_centers():
    loss = _make_loss()
    old_centers = loss.cluster_centers.data.clone()
    cb = ClusterCentersWarmStartCallback(max_samples=64, image_key="image")
    trainer = _make_trainer()
    pl_module = _make_pl_module(loss)

    cb.on_train_start(trainer, pl_module)

    assert not torch.allclose(loss.cluster_centers.data, old_centers)


def test_warm_start_uses_mu_by_default():
    captured = []

    def forward_fn(images):
        out = _flat_vae_output(batch=images.shape[0])
        captured.append(("mu", out.mu))
        return out

    loss = _make_loss()
    cb = ClusterCentersWarmStartCallback(max_samples=32, vae_latent="mu")
    trainer = _make_trainer(n_batches=4)
    pl_module = _make_pl_module(loss, output_factory=forward_fn)

    cb.on_train_start(trainer, pl_module)
    assert len(captured) > 0
    assert all(name == "mu" for name, _ in captured)


def test_warm_start_uses_z_when_configured():
    loss = _make_loss()
    cb = ClusterCentersWarmStartCallback(max_samples=32, vae_latent="z")
    trainer = _make_trainer(n_batches=4)
    pl_module = _make_pl_module(loss)
    # Should complete without error using z
    cb.on_train_start(trainer, pl_module)
    assert loss.cluster_centers.data.shape == (N_CLUSTERS, EMB_DIM)


def test_warm_start_handles_spatial_latent():
    loss = ClusteringAwareVAELoss(
        n_clusters=N_CLUSTERS,
        embedding_dim=EMB_DIM,
        gamma=0.1,
        delta=0.01,
        reconstruction_loss="mse",
    )
    cb = ClusterCentersWarmStartCallback(max_samples=32, vae_latent="mu")
    trainer = _make_trainer(n_batches=4)
    pl_module = _make_pl_module(
        loss,
        output_factory=lambda images: _spatial_vae_output(
            batch=images.shape[0], latent_c=EMB_DIM, latent_h=2
        ),
    )
    cb.on_train_start(trainer, pl_module)
    assert loss.cluster_centers.data.shape == (N_CLUSTERS, EMB_DIM)


# ===========================================================================
# Skip conditions
# ===========================================================================


def test_skips_when_loss_is_not_clustering_aware():
    cb = ClusterCentersWarmStartCallback(max_samples=64)
    trainer = _make_trainer()
    pl_module = MagicMock()
    pl_module.loss_function = MagicMock()  # not ClusteringAwareVAELoss

    cb.on_train_start(trainer, pl_module)

    # model should never be called (no forward pass)
    pl_module.assert_not_called()


def test_skips_when_no_loss_function_attr():
    cb = ClusterCentersWarmStartCallback(max_samples=64)
    trainer = _make_trainer()
    pl_module = MagicMock(spec=[])  # no attributes

    cb.on_train_start(trainer, pl_module)  # must not raise


# ===========================================================================
# max_samples
# ===========================================================================


def test_respects_max_samples():
    """Stops collecting after max_samples regardless of dataloader length."""
    collected = []

    def forward_fn(images):
        collected.append(images.shape[0])
        return _flat_vae_output(batch=images.shape[0])

    loss = _make_loss()
    cb = ClusterCentersWarmStartCallback(max_samples=N_CLUSTERS + 1, image_key="image")
    trainer = _make_trainer(n_batches=100, batch_size=BATCH_SIZE)
    pl_module = _make_pl_module(loss, output_factory=forward_fn)

    cb.on_train_start(trainer, pl_module)

    total = sum(collected)
    assert total <= N_CLUSTERS + 1 + BATCH_SIZE  # at most one batch over


# ===========================================================================
# Model mode
# ===========================================================================


def test_eval_mode_called_before_collection():
    loss = _make_loss()
    cb = ClusterCentersWarmStartCallback(max_samples=16)
    trainer = _make_trainer(n_batches=2)
    pl_module = _make_pl_module(loss)

    cb.on_train_start(trainer, pl_module)

    pl_module.eval.assert_called()


def test_train_mode_restored_after_collection():
    loss = _make_loss()
    cb = ClusterCentersWarmStartCallback(max_samples=16)
    trainer = _make_trainer(n_batches=2)
    pl_module = _make_pl_module(loss)

    cb.on_train_start(trainer, pl_module)

    pl_module.train.assert_called()


# ===========================================================================
# Validation
# ===========================================================================


def test_invalid_vae_latent_raises():
    with pytest.raises(ValueError, match="vae_latent"):
        ClusterCentersWarmStartCallback(vae_latent="invalid")


# ===========================================================================
# Config dataclass
# ===========================================================================


def test_config_dataclass_has_correct_target():
    cfg = ClusterCentersWarmStartCallbackConfig()
    assert "ClusterCentersWarmStartCallback" in cfg._target_


def test_config_dataclass_defaults():
    cfg = ClusterCentersWarmStartCallbackConfig()
    assert cfg.max_samples == 4096
    assert cfg.vae_latent == "mu"
    assert cfg.image_key == "image"
