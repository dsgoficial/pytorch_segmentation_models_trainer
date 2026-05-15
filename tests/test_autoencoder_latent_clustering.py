# -*- coding: utf-8 -*-
from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn as nn
from omegaconf import OmegaConf

from pytorch_segmentation_models_trainer.custom_metrics import (
    autoencoder_latent_clustering as latent_clustering,
)
from pytorch_segmentation_models_trainer.custom_callbacks.latent_clustering_callback import (
    AutoencoderLatentClusteringCallback,
)
from pytorch_segmentation_models_trainer.custom_models import (
    variational_autoencoder as vae_models,
)
from pytorch_segmentation_models_trainer.model_loader import autoencoder_model
from pytorch_segmentation_models_trainer.model_loader import (
    variational_autoencoder_model,
)
from pytorch_segmentation_models_trainer.tools.kmeans import kmeans_calculator

AutoencoderLatentClusteringMetrics = (
    latent_clustering.AutoencoderLatentClusteringMetrics
)
VariationalAutoencoderOutput = vae_models.VariationalAutoencoderOutput
AutoencoderModel = autoencoder_model.AutoencoderModel
VariationalAutoencoderModel = variational_autoencoder_model.VariationalAutoencoderModel
MiniBatchKMeans = kmeans_calculator.MiniBatchKMeans

LATENT_METRICS_TARGET = (
    "pytorch_segmentation_models_trainer.custom_metrics."
    "autoencoder_latent_clustering.AutoencoderLatentClusteringMetrics"
)


class TinyLatentAutoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder_conv = nn.Conv2d(3, 4, kernel_size=1)
        self.decoder_conv = nn.Conv2d(4, 3, kernel_size=1)

    def encode(self, x):
        return self.encoder_conv(x)

    def forward(self, x):
        return self.decoder_conv(self.encode(x))


class TinyLatentVAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder_conv = nn.Conv2d(3, 4, kernel_size=1)
        self.decoder_conv = nn.Conv2d(4, 3, kernel_size=1)

    def forward(self, x):
        mu = self.encoder_conv(x)
        logvar = torch.zeros_like(mu)
        z = mu + 1.0
        reconstruction = self.decoder_conv(mu)
        return VariationalAutoencoderOutput(reconstruction, mu, logvar, z)


class TinyLatentVAEWithEncode(TinyLatentVAE):
    def encode(self, x):
        mu = self.encoder_conv(x)
        logvar = torch.zeros_like(mu)
        return mu, logvar


def _clustered_embeddings(device):
    base = torch.tensor(
        [
            [0.0, 0.0],
            [0.1, 0.0],
            [0.0, 0.1],
            [4.0, 4.0],
            [4.1, 4.0],
            [4.0, 4.1],
        ],
        device=device,
    )
    return base.float()


def test_latent_clustering_metrics_contract_and_device():
    embeddings = _clustered_embeddings(torch.device("cpu"))
    metric = AutoencoderLatentClusteringMetrics(
        n_clusters=2,
        max_samples=6,
        kmeans_batch_size=3,
        kmeans_max_iter=10,
        random_state=0,
        compute_silhouette=True,
        compute_dunn=True,
    )

    metric.update(embeddings)
    result = metric.compute()

    expected = {
        "latent_calinski_harabasz",
        "latent_davies_bouldin",
        "latent_dunn",
        "latent_silhouette",
    }
    assert set(result) == expected
    for value in result.values():
        assert isinstance(value, torch.Tensor)
        assert value.shape == torch.Size([])
        assert value.dtype == embeddings.dtype
        assert value.device == embeddings.device
        assert torch.isfinite(value)


def test_latent_clustering_metrics_accept_spatial_latents_and_labels():
    embeddings = torch.zeros(6, 2, 2, 2)
    embeddings[:3, 0] = 0.0
    embeddings[3:, 0] = 5.0
    target_labels = torch.tensor([0, 0, 0, 1, 1, 1])
    metric = AutoencoderLatentClusteringMetrics(
        n_clusters=2,
        max_samples=6,
        random_state=0,
        label_key="domain",
    )

    reduced = metric.reduce_latents(embeddings)
    metric.update(embeddings, target_labels=target_labels)
    result = metric.compute()

    assert reduced.shape == (6, 2)
    assert reduced.dtype == embeddings.dtype
    assert "latent_adjusted_rand" in result
    assert "latent_normalized_mutual_info" in result
    assert result["latent_adjusted_rand"].device == embeddings.device


def test_latent_clustering_metrics_respects_max_samples():
    metric = AutoencoderLatentClusteringMetrics(n_clusters=2, max_samples=4)
    embeddings = torch.randn(8, 3)

    sampled, labels = metric._sample(embeddings, torch.arange(8))

    assert sampled.shape == (4, 3)
    assert labels.tolist() == [0, 1, 2, 3]


def test_latent_clustering_metrics_validates_inputs():
    metric = AutoencoderLatentClusteringMetrics(n_clusters=4)
    metric.update(torch.randn(3, 2))

    with pytest.raises(ValueError, match="n_clusters"):
        metric.compute()

    with pytest.raises(ValueError, match="2D or 4D"):
        metric.reduce_latents(torch.randn(2, 3, 4))


def test_latent_clustering_metrics_uses_existing_minibatch_kmeans():
    embeddings = _clustered_embeddings(torch.device("cpu"))
    metric = AutoencoderLatentClusteringMetrics(n_clusters=2, random_state=0)

    with patch(
        (
            "pytorch_segmentation_models_trainer.custom_metrics."
            "autoencoder_latent_clustering.MiniBatchKMeans"
        ),
        wraps=MiniBatchKMeans,
    ) as kmeans_mock:
        metric.update(embeddings)
        metric.compute()

    assert kmeans_mock.called


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA is not available",
)
def test_latent_clustering_metrics_keep_computation_on_cuda():
    embeddings = _clustered_embeddings(torch.device("cuda"))
    metric = AutoencoderLatentClusteringMetrics(n_clusters=2, random_state=0)

    metric.update(embeddings)
    result = metric.compute()

    assert all(value.device.type == "cuda" for value in result.values())


def test_latent_clustering_callback_logs_autoencoder_validation_epoch_end():
    cfg = OmegaConf.create(
        {
            "model": {"_target_": "unused"},
            "loss": {"_target_": "unused"},
            "optimizer": {"_target_": "torch.optim.Adam", "lr": 0.001},
            "scheduler_list": [],
        }
    )

    with patch.object(
        AutoencoderModel,
        "get_model",
        return_value=TinyLatentAutoencoder(),
    ):
        with patch.object(
            AutoencoderModel,
            "get_loss_function",
            return_value=nn.MSELoss(),
        ):
            pl_model = AutoencoderModel(cfg)

    pl_model.log = MagicMock()
    pl_model.log_dict = MagicMock()
    callback = AutoencoderLatentClusteringCallback(
        n_clusters=2,
        max_samples=4,
        kmeans_max_iter=5,
        random_state=0,
    )
    batch = {
        "image": torch.cat(
            [torch.zeros(2, 3, 4, 4), torch.ones(2, 3, 4, 4)],
            dim=0,
        ),
        "target": torch.zeros(4, 3, 4, 4),
    }

    pl_model.validation_step(batch, 0)
    callback.on_validation_batch_end(None, pl_model, None, batch, 0)
    callback.on_validation_epoch_end(None, pl_model)

    logged = pl_model.log_dict.call_args.args[0]
    assert "val/latent_calinski_harabasz" in logged
    assert "val/latent_davies_bouldin" in logged


def test_latent_clustering_callback_uses_mu_for_vae_by_default():
    cfg = OmegaConf.create(
        {
            "model": {"_target_": "unused"},
            "loss": {"_target_": "unused"},
            "optimizer": {"_target_": "torch.optim.Adam", "lr": 0.001},
            "scheduler_list": [],
        }
    )
    loss_fn = MagicMock(return_value={"loss": torch.tensor(0.0)})

    with patch.object(
        VariationalAutoencoderModel,
        "get_model",
        return_value=TinyLatentVAE(),
    ):
        with patch.object(
            VariationalAutoencoderModel,
            "get_loss_function",
            return_value=loss_fn,
        ):
            pl_model = VariationalAutoencoderModel(cfg)

    pl_model.log = MagicMock()
    pl_model.log_dict = MagicMock()
    callback = AutoencoderLatentClusteringCallback(
        n_clusters=2,
        max_samples=4,
        kmeans_max_iter=5,
        random_state=0,
    )
    batch = {
        "image": torch.cat(
            [torch.zeros(2, 3, 4, 4), torch.ones(2, 3, 4, 4)],
            dim=0,
        ),
        "target": torch.zeros(4, 3, 4, 4),
    }

    pl_model.validation_step(batch, 0)
    callback.on_validation_batch_end(None, pl_model, None, batch, 0)

    stored = callback.val_latent_metrics._embeddings[0]
    expected_mu = pl_model.model(batch["image"]).mu.detach()
    assert torch.allclose(stored, expected_mu)


def test_latent_clustering_callback_handles_vae_encode_tuple():
    cfg = OmegaConf.create(
        {
            "model": {"_target_": "unused"},
            "loss": {"_target_": "unused"},
            "optimizer": {"_target_": "torch.optim.Adam", "lr": 0.001},
            "scheduler_list": [],
        }
    )
    loss_fn = MagicMock(return_value={"loss": torch.tensor(0.0)})

    with patch.object(
        VariationalAutoencoderModel,
        "get_model",
        return_value=TinyLatentVAEWithEncode(),
    ):
        with patch.object(
            VariationalAutoencoderModel,
            "get_loss_function",
            return_value=loss_fn,
        ):
            pl_model = VariationalAutoencoderModel(cfg)

    callback = AutoencoderLatentClusteringCallback(n_clusters=2, max_samples=4)
    batch = {"image": torch.randn(4, 3, 4, 4), "target": torch.zeros(4, 3, 4, 4)}

    callback.on_validation_batch_end(None, pl_model, None, batch, 0)

    stored = callback.val_latent_metrics._embeddings[0]
    expected_mu, _ = pl_model.model.encode(batch["image"])
    assert torch.allclose(stored, expected_mu)
