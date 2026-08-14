# -*- coding: utf-8 -*-
"""
/***************************************************************************
 pytorch_segmentation_models_trainer
                              -------------------
        begin                : 2026-04-15
        git sha              : $Format:%H$
        copyright            : (C) 2026 by Philipe Borba - Cartographic Engineer
                                                            @ Brazilian Army
        email                : philipeborba at gmail dot com
 ***************************************************************************/
/***************************************************************************
 *                                                                         *
 *   This program is free software; you can redistribute it and/or modify  *
 *   it under the terms of the GNU General Public License as published by  *
 *   the Free Software Foundation; either version 2 of the License, or     *
 *   (at your option) any later version.                                   *
 *                                                                         *
 ****

Hydra dataclass configurations for Evidential Deep Learning components.

These configs are registered in the Hydra ConfigStore so they can be used
directly in YAML files via the ``_target_`` mechanism.
"""

import logging
from dataclasses import dataclass, field
from typing import Any, List, Optional, Union

from hydra.core.config_store import ConfigStore
from omegaconf import MISSING

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Model wrapper
# ---------------------------------------------------------------------------


@dataclass
class EvidentialWrapperConfig:
    """Configuration for the EvidentialWrapper model.

    Example YAML::

        model:
          _target_: pytorch_segmentation_models_trainer.custom_models.edl_wrapper.EvidentialWrapper
          freeze_encoder: true
          model:
            _target_: segmentation_models_pytorch.Unet
            encoder_name: resnet50
            encoder_weights: imagenet
            in_channels: 3
            classes: 5
    """

    _target_: str = (
        "pytorch_segmentation_models_trainer.custom_models.edl_wrapper.EvidentialWrapper"
    )
    freeze_encoder: bool = False
    model: Any = MISSING


# ---------------------------------------------------------------------------
# Loss functions
# ---------------------------------------------------------------------------


@dataclass
class EvidentialMSELossConfig:
    """Configuration for EvidentialMSELoss.

    Use inside a compound_loss configuration with weight 1.0 (constant).
    """

    _target_: str = (
        "pytorch_segmentation_models_trainer.custom_losses.edl_loss.EvidentialMSELoss"
    )
    name: str = "edl_mse"
    num_classes: int = MISSING
    ignore_index: int = 255


@dataclass
class EvidentialKLLossConfig:
    """Configuration for EvidentialKLLoss.

    Typically given weight 0.0 at epoch 0 and annealed to 1.0 via the
    ``compound_loss`` epoch_thresholds mechanism.
    """

    _target_: str = (
        "pytorch_segmentation_models_trainer.custom_losses.edl_loss.EvidentialKLLoss"
    )
    name: str = "edl_kl"
    num_classes: int = MISSING
    ignore_index: int = 255


# ---------------------------------------------------------------------------
# Callbacks
# ---------------------------------------------------------------------------


@dataclass
class EvidentialWarmupCallbackConfig:
    """Configuration for EvidentialWarmupCallback.

    Controls encoder freezing and KL annealing schedule.
    ``freeze_encoder`` should match the value in EvidentialWrapperConfig.
    """

    _target_: str = (
        "pytorch_segmentation_models_trainer.custom_callbacks.edl_callbacks.EvidentialWarmupCallback"
    )
    warmup_epochs: int = 5
    freeze_encoder: bool = False
    partial_unfreeze_epoch: int = 10


@dataclass
class EvidentialUncertaintyVisualizationCallbackConfig:
    """Configuration for EvidentialUncertaintyVisualizationCallback."""

    _target_: str = (
        "pytorch_segmentation_models_trainer.custom_callbacks.edl_callbacks"
        ".EvidentialUncertaintyVisualizationCallback"
    )
    num_images: int = 4
    log_every_n_epochs: int = 5
    norm_params: Optional[Any] = None


# ---------------------------------------------------------------------------
# Inference processor
# ---------------------------------------------------------------------------


@dataclass
class EvidentialInferenceProcessorConfig:
    """Configuration for EvidentialInferenceProcessor."""

    _target_: str = (
        "pytorch_segmentation_models_trainer.tools.inference.edl_inference_processor"
        ".EvidentialInferenceProcessor"
    )
    model_path: str = MISSING
    device: str = "cuda"
    batch_size: int = 1
    model_input_shape: List[int] = field(default_factory=lambda: [512, 512])
    step_shape: Optional[List[int]] = None
    output_uncertainty_path: Optional[str] = None
    export_alpha: bool = False


# ---------------------------------------------------------------------------
# Register with Hydra ConfigStore
# ---------------------------------------------------------------------------

cs = ConfigStore.instance()

cs.store(group="model", name="evidential_wrapper", node=EvidentialWrapperConfig)
cs.store(group="loss", name="edl_mse_loss", node=EvidentialMSELossConfig)
cs.store(group="loss", name="edl_kl_loss", node=EvidentialKLLossConfig)
cs.store(
    group="callbacks",
    name="edl_warmup",
    node=EvidentialWarmupCallbackConfig,
)
cs.store(
    group="callbacks",
    name="edl_uncertainty_viz",
    node=EvidentialUncertaintyVisualizationCallbackConfig,
)
