# -*- coding: utf-8 -*-
"""Hydra dataclass configs for deterministic and variational autoencoders."""

from dataclasses import dataclass, field
from typing import List, Optional

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
          reconstruction_loss: smooth_l1_ms_ssim
          reconstruction_weight: 1.0
          beta: 1.0
          free_bits: 0.25
          kl_balance: true
          smooth_l1_beta: 0.1
          smooth_l1_weight: 0.8
          ms_ssim_weight: 0.2
          ms_ssim_data_range: 1.0
          ms_ssim_alpha: 1.0
          ms_ssim_compensation: 1.0
          ms_ssim_input_is_normalized: true
          ms_ssim_mean: [0.485, 0.456, 0.406]
          ms_ssim_std: [0.229, 0.224, 0.225]
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
    smooth_l1_weight: float = 0.8
    ms_ssim_weight: float = 0.2
    ms_ssim_data_range: float = 1.0
    ms_ssim_sigmas: List[float] = field(
        default_factory=lambda: [0.5, 1.0, 2.0, 4.0, 8.0]
    )
    ms_ssim_alpha: float = 1.0
    ms_ssim_compensation: float = 1.0
    ms_ssim_input_is_normalized: bool = False
    ms_ssim_mean: Optional[List[float]] = None
    ms_ssim_std: Optional[List[float]] = None
    ms_ssim_clamp: bool = True


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
class AutoencoderLatentClusteringCallbackConfig:
    """Configuration for autoencoder latent clustering callback.

    Example YAML:
        callbacks:
          - _target_: pytorch_segmentation_models_trainer.custom_callbacks.AutoencoderLatentClusteringCallback
          n_clusters: 8
          max_samples: 2048
          kmeans_max_iter: 50
          normalize: true
          compute_silhouette: true
          label_key: domain
          vae_latent: mu
    """

    _target_: str = (
        "pytorch_segmentation_models_trainer.custom_callbacks"
        ".AutoencoderLatentClusteringCallback"
    )
    n_clusters: int = MISSING
    max_samples: Optional[int] = 2048
    kmeans_max_iter: int = 50
    kmeans_batch_size: int = 1024
    tol: float = 1e-4
    random_state: Optional[int] = None
    normalize: bool = True
    latent_reduction: str = "adaptive_avg_pool"
    compute_silhouette: bool = False
    compute_dunn: bool = False
    label_key: Optional[str] = None
    vae_latent: str = "mu"
    latent_source: str = "auto"
    image_key: str = "image"


@dataclass
class PatienceWarmupCallbackConfig:
    """Configuration for ``PatienceWarmupCallback``.

    Example YAML:
        callbacks:
          - _target_: pytorch_segmentation_models_trainer.custom_callbacks.training_callbacks.PatienceWarmupCallback
            monitor: val/reconstruction_loss
            patience: 5
            min_delta: 0.001
            mode: min
            min_epochs: 3
    """

    _target_: str = (
        "pytorch_segmentation_models_trainer.custom_callbacks"
        ".training_callbacks.PatienceWarmupCallback"
    )
    monitor: str = "val/reconstruction_loss"
    patience: int = 5
    min_delta: float = 1e-4
    mode: str = "min"
    min_epochs: int = 0


@dataclass
class ClusterCentersWarmStartCallbackConfig:
    """Configuration for ``ClusterCentersWarmStartCallback``.

    Example YAML:
        callbacks:
          - _target_: pytorch_segmentation_models_trainer.custom_callbacks.ClusterCentersWarmStartCallback
            max_samples: 4096
            image_key: image
            vae_latent: mu
    """

    _target_: str = (
        "pytorch_segmentation_models_trainer.custom_callbacks"
        ".ClusterCentersWarmStartCallback"
    )
    max_samples: int = 4096
    image_key: str = "image"
    vae_latent: str = "mu"
    kmeans_max_iter: int = 100
    kmeans_batch_size: int = 1024
    random_state: Optional[int] = None


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
    name="autoencoder_latent_clustering",
    node=AutoencoderLatentClusteringCallbackConfig,
)
cs.store(
    group="callbacks",
    name="kl_annealing",
    node=KLAnnealingCallbackConfig,
)
cs.store(
    group="callbacks",
    name="cluster_centers_warm_start",
    node=ClusterCentersWarmStartCallbackConfig,
)
cs.store(
    group="callbacks",
    name="patience_warmup",
    node=PatienceWarmupCallbackConfig,
)
