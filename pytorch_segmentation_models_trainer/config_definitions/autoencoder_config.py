# -*- coding: utf-8 -*-
"""Hydra dataclass configs for deterministic and variational autoencoders."""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

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
    output_activation: Optional[str] = None
    logvar_clamp: Optional[List[float]] = None
    use_progressive_decoder: bool = False
    base_channels: int = 128
    min_channels: int = 32
    upsample_mode: str = "bilinear"


@dataclass
class VariationalAutoencoderLossConfig:
    """Configuration for ``VariationalAutoencoderLoss``.

    Example YAML:
        loss:
          _target_: pytorch_segmentation_models_trainer.custom_losses.autoencoder_losses.VariationalAutoencoderLoss
          reconstruction_loss: mse
          reconstruction_weight: 1.0
          beta: 1.0
          free_bits: 0.25
          kl_balance: true
    """

    _target_: str = (
        "pytorch_segmentation_models_trainer.custom_losses"
        ".autoencoder_losses.VariationalAutoencoderLoss"
    )
    reconstruction_loss: str = "mse"
    reconstruction_weight: float = 1.0
    beta: float = 1.0
    free_bits: float = 0.0
    kl_balance: bool = False
    smooth_l1_beta: float = 0.1


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


@dataclass
class KLAnnealingCallbackConfig:
    """Configuration for ``KLAnnealingCallback``.

    Example YAML:
        callbacks:
          - _target_: pytorch_segmentation_models_trainer.custom_callbacks.kl_annealing_callback.KLAnnealingCallback
            max_beta: 1.0
            min_beta: 0.0
            annealing_steps: 5000
            schedule: cosine
            use_epochs: false
    """

    _target_: str = (
        "pytorch_segmentation_models_trainer.custom_callbacks"
        ".kl_annealing_callback.KLAnnealingCallback"
    )
    max_beta: float = 1.0
    min_beta: float = 0.0
    annealing_steps: int = MISSING
    schedule: str = "linear"
    use_epochs: bool = False
    cycle_length: int = 100
    cycle_ratio: float = 0.5
    start_after: int = 0


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
cs.store(
    group="callbacks",
    name="kl_annealing",
    node=KLAnnealingCallbackConfig,
)
