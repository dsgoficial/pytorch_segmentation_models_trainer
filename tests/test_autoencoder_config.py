# -*- coding: utf-8 -*-
import pytest
from omegaconf import MissingMandatoryValue, OmegaConf

from pytorch_segmentation_models_trainer.config_definitions import (
    autoencoder_config,
)

AutoencoderLatentClusteringMetricsConfig = (
    autoencoder_config.AutoencoderLatentClusteringMetricsConfig
)
GenericVariationalAutoencoderConfig = (
    autoencoder_config.GenericVariationalAutoencoderConfig
)
VariationalAutoencoderLossConfig = autoencoder_config.VariationalAutoencoderLossConfig
VariationalAutoencoderModelConfig = autoencoder_config.VariationalAutoencoderModelConfig


def test_generic_variational_autoencoder_config():
    cfg = OmegaConf.structured(GenericVariationalAutoencoderConfig)

    assert "GenericVariationalAutoencoder" in cfg._target_
    assert cfg.use_huggingface is False
    assert cfg.in_channels == 3
    assert cfg.latent_dim is None
    assert cfg.pretrained is True
    with pytest.raises(MissingMandatoryValue):
        _ = cfg.encoder_name


def test_variational_autoencoder_loss_config():
    cfg = OmegaConf.structured(VariationalAutoencoderLossConfig)

    assert "VariationalAutoencoderLoss" in cfg._target_
    assert cfg.reconstruction_loss == "mse"
    assert cfg.reconstruction_weight == 1.0
    assert cfg.beta == 1.0
    assert cfg.free_bits == pytest.approx(0.0)
    assert cfg.kl_balance is False
    assert cfg.smooth_l1_beta == pytest.approx(0.1)
    assert cfg.smooth_l1_weight == pytest.approx(0.8)
    assert cfg.ms_ssim_weight == pytest.approx(0.2)
    assert cfg.ms_ssim_data_range == pytest.approx(1.0)
    assert list(cfg.ms_ssim_sigmas) == [0.5, 1.0, 2.0, 4.0, 8.0]
    assert cfg.ms_ssim_alpha == pytest.approx(1.0)
    assert cfg.ms_ssim_compensation == pytest.approx(1.0)


def test_variational_autoencoder_model_config():
    cfg = OmegaConf.structured(VariationalAutoencoderModelConfig)

    assert "VariationalAutoencoderModel" in cfg._target_


def test_autoencoder_latent_clustering_metrics_config():
    cfg = OmegaConf.structured(AutoencoderLatentClusteringMetricsConfig)

    assert "AutoencoderLatentClusteringMetrics" in cfg._target_
    assert cfg.max_samples == 2048
    assert cfg.kmeans_max_iter == 50
    assert cfg.kmeans_batch_size == 1024
    assert cfg.normalize is True
    assert cfg.latent_reduction == "adaptive_avg_pool"
    assert cfg.compute_silhouette is False
    assert cfg.compute_dunn is False
    assert cfg.label_key is None
    assert cfg.vae_latent == "mu"
    with pytest.raises(MissingMandatoryValue):
        _ = cfg.n_clusters
