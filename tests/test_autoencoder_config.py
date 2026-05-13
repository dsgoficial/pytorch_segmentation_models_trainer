# -*- coding: utf-8 -*-
import pytest
from omegaconf import MissingMandatoryValue, OmegaConf

from pytorch_segmentation_models_trainer.config_definitions.autoencoder_config import (
    GenericVariationalAutoencoderConfig,
    VariationalAutoencoderLossConfig,
    VariationalAutoencoderModelConfig,
)


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


def test_variational_autoencoder_model_config():
    cfg = OmegaConf.structured(VariationalAutoencoderModelConfig)

    assert "VariationalAutoencoderModel" in cfg._target_
