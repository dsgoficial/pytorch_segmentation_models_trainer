# -*- coding: utf-8 -*-
"""Hydra dataclass configs for deterministic and variational autoencoders."""

from dataclasses import dataclass
from typing import Optional

from hydra.core.config_store import ConfigStore
from omegaconf import MISSING


@dataclass
class GenericVariationalAutoencoderConfig:
    """Configuration for ``GenericVariationalAutoencoder``.

    Example YAML:
        model:
          _target_: pytorch_segmentation_models_trainer.custom_models.variational_autoencoder.GenericVariationalAutoencoder
          encoder_name: resnet18
          use_huggingface: false
          in_channels: 3
          latent_dim: 128
          pretrained: false
    """

    _target_: str = (
        "pytorch_segmentation_models_trainer.custom_models"
        ".variational_autoencoder.GenericVariationalAutoencoder"
    )
    encoder_name: str = MISSING
    use_huggingface: bool = False
    in_channels: int = 3
    latent_dim: Optional[int] = None
    pretrained: bool = True


@dataclass
class VariationalAutoencoderLossConfig:
    """Configuration for ``VariationalAutoencoderLoss``.

    Example YAML:
        loss:
          _target_: pytorch_segmentation_models_trainer.custom_losses.autoencoder_losses.VariationalAutoencoderLoss
          reconstruction_loss: mse
          reconstruction_weight: 1.0
          beta: 1.0
    """

    _target_: str = (
        "pytorch_segmentation_models_trainer.custom_losses"
        ".autoencoder_losses.VariationalAutoencoderLoss"
    )
    reconstruction_loss: str = "mse"
    reconstruction_weight: float = 1.0
    beta: float = 1.0


@dataclass
class VariationalAutoencoderModelConfig:
    """Configuration for ``VariationalAutoencoderModel``.

    Example YAML:
        pl_model:
          _target_: pytorch_segmentation_models_trainer.model_loader.variational_autoencoder_model.VariationalAutoencoderModel
    """

    _target_: str = (
        "pytorch_segmentation_models_trainer.model_loader"
        ".variational_autoencoder_model.VariationalAutoencoderModel"
    )


cs = ConfigStore.instance()
cs.store(
    group="model",
    name="generic_variational_autoencoder",
    node=GenericVariationalAutoencoderConfig,
)
cs.store(
    group="loss",
    name="variational_autoencoder_loss",
    node=VariationalAutoencoderLossConfig,
)
cs.store(
    group="pl_model",
    name="variational_autoencoder_model",
    node=VariationalAutoencoderModelConfig,
)
