# -*- coding: utf-8 -*-
"""
/***************************************************************************
 pytorch_segmentation_models_trainer
                              -------------------
        begin                : 2021-05-08
        git sha              : $Format:%H$
        copyright            : (C) 2021 by Philipe Borba - Cartographic Engineer
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
"""
import dataclasses
from dataclasses import dataclass, field
import logging
from typing import Any, Dict, List, Optional, Union

import hydra
from hydra.core.config_store import ConfigStore
from hydra.utils import instantiate
from omegaconf import MISSING, DictConfig, OmegaConf


# ============================================================================
# Individual Loss Configurations
# ============================================================================

@dataclass
class SegParamsConfig:
    compute_interior: bool = True
    compute_edge: bool = True
    compute_vertex: bool = True


@dataclass
class BaseLossConfig:
    """Base configuration for any loss function"""
    _target_: str = MISSING
    name: str = MISSING
    weight: Union[float, List[float]] = 1.0  # Static weight or list for interpolation


@dataclass
class SegLossConfig(BaseLossConfig):
    _target_: str = "pytorch_segmentation_models_trainer.custom_losses.base_loss.SegLoss"
    name: str = "seg"
    gt_channel_selector: Union[SegParamsConfig, int] = field(default_factory=SegParamsConfig)
    bce_coef: float = 0.5
    dice_coef: float = 0.5
    tversky_focal_coef: float = 0.0
    weight: float = 1.0


@dataclass
class CrossfieldAlignLossConfig(BaseLossConfig):
    _target_: str = "pytorch_segmentation_models_trainer.custom_losses.base_loss.CrossfieldAlignLoss"
    name: str = "crossfield_align"
    weight: float = 1.0


@dataclass
class CrossfieldAlign90LossConfig(BaseLossConfig):
    _target_: str = "pytorch_segmentation_models_trainer.custom_losses.base_loss.CrossfieldAlign90Loss"
    name: str = "crossfield_align90"
    weight: float = 0.2


@dataclass
class CrossfieldSmoothLossConfig(BaseLossConfig):
    _target_: str = "pytorch_segmentation_models_trainer.custom_losses.base_loss.CrossfieldSmoothLoss"
    name: str = "crossfield_smooth"
    weight: float = 0.005


@dataclass
class SegCrossfieldLossConfig(BaseLossConfig):
    _target_: str = "pytorch_segmentation_models_trainer.custom_losses.base_loss.SegCrossfieldLoss"
    name: str = "seg_crossfield"
    pred_channel: int = 0
    weight: Union[float, List[float]] = field(default_factory=lambda: [0, 0, 0.2])


@dataclass
class SegEdgeInteriorLossConfig(BaseLossConfig):
    _target_: str = "pytorch_segmentation_models_trainer.custom_losses.base_loss.SegEdgeInteriorLoss"
    name: str = "seg_edge_interior"
    weight: Union[float, List[float]] = field(default_factory=lambda: [0, 0, 0.2])


# ============================================================================
# Multi-Loss Configuration
# ============================================================================

@dataclass
class LossWeightConfig:
    """Configuration for a single loss with its weight"""
    loss: Any = MISSING  # The loss configuration (SegLossConfig, etc.)
    weight: Union[float, List[float]] = 1.0  # Weight can be static or dynamic


@dataclass
class CompoundLossConfig:
    """
    Configuration for compound loss with multiple losses and weights.
    
    Example YAML usage:
    
    compound_loss:
      epoch_thresholds: [0, 5, 10]
      losses:
        - loss:
            _target_: pytorch_segmentation_models_trainer.custom_losses.base_loss.SegLoss
            name: seg
            gt_channel_selector: 0
            bce_coef: 0.5
            dice_coef: 0.5
          weight: 10.0
        
        - loss:
            _target_: pytorch_segmentation_models_trainer.custom_losses.base_loss.CrossfieldAlignLoss
            name: crossfield_align
          weight: 1.0
          
        - loss:
            _target_: pytorch_segmentation_models_trainer.custom_losses.base_loss.SegCrossfieldLoss
            name: seg_interior_crossfield
            pred_channel: 0
          weight: [0, 0, 0.2]  # Dynamic weight that interpolates based on epoch
    """
    losses: List[LossWeightConfig] = field(default_factory=list)
    epoch_thresholds: Optional[List[float]] = field(default_factory=lambda: [0, 5, 10])
    pre_processes: Optional[List[Any]] = None
    normalization_params: NormalizationParams = field(default_factory=NormalizationParams)


# ============================================================================
# Legacy Configurations (for backward compatibility)
# ============================================================================

@dataclass
class CoefsConfig:
    """Legacy coefficient configuration - kept for backward compatibility"""
    epoch_thresholds: List[float] = field(default_factory=lambda: [0, 5, 10])
    seg: float = 10
    crossfield_align: float = 1
    crossfield_align90: float = 0.2
    crossfield_smooth: float = 0.005
    seg_interior_crossfield: List[float] = field(default_factory=lambda: [0, 0, 0.2])
    seg_edge_crossfield: List[float] = field(default_factory=lambda: [0, 0, 0.2])
    seg_edge_interior: List[float] = field(default_factory=lambda: [0, 0, 0.2])


@dataclass
class SegLossParamsConfig:
    bce_coef: float = 1.0
    dice_coef: float = 0.2
    use_dist: bool = True
    use_size: bool = True
    w0: float = 50
    sigma: float = 10


@dataclass
class NormalizationParams:
    min_samples: int = 10
    max_samples: int = 1000


@dataclass
class MultiLossConfig:
    defaults: List[Any] = field(
        default_factory=lambda: [
            {"normalization_params": "norm"},
            {"coefs": "coefs"},
            {"seg_loss_params": "seg_loss_params"},
        ]
    )
    normalization_params: NormalizationParams = MISSING
    coefs: CoefsConfig = MISSING
    seg_loss_params: SegLossParamsConfig = MISSING


@dataclass
class LossParamsConfig:
    multi_loss: MultiLossConfig = field(default_factory=MultiLossConfig)
    seg_loss_params: SegParamsConfig = field(default_factory=SegLossParamsConfig)
    # New compound loss configuration
    compound_loss: Optional[CompoundLossConfig] = None


# ============================================================================
# Register configurations with Hydra
# ============================================================================

logger = logging.getLogger(__name__)

cs = ConfigStore.instance()

# Register individual loss configs
cs.store(group="loss", name="seg_loss", node=SegLossConfig)
cs.store(group="loss", name="crossfield_align_loss", node=CrossfieldAlignLossConfig)
cs.store(group="loss", name="crossfield_align90_loss", node=CrossfieldAlign90LossConfig)
cs.store(group="loss", name="crossfield_smooth_loss", node=CrossfieldSmoothLossConfig)
cs.store(group="loss", name="seg_crossfield_loss", node=SegCrossfieldLossConfig)
cs.store(group="loss", name="seg_edge_interior_loss", node=SegEdgeInteriorLossConfig)


# Register legacy configs for backward compatibility
cs.store(name="seg_loss_config", node=SegLossConfig)
cs.store(group="normalization_params", name="norm", node=NormalizationParams)
cs.store(group="coefs", name="coefs", node=CoefsConfig)
cs.store(group="seg_loss_params", name="seg_loss_params", node=SegLossParamsConfig)
cs.store(name="multi_loss", node=MultiLossConfig)

# Register compound loss config
cs.store(name="compound_loss", node=CompoundLossConfig)

@hydra.main(config_name="compound_loss_config")
def build_config(cfg: DictConfig) -> None:
    logger.info(OmegaConf.to_yaml(cfg))


if __name__ == "__main__":
    build_config()