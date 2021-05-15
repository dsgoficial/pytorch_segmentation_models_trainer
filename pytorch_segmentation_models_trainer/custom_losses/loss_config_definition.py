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
from typing import Any, Dict, List

import hydra
from hydra.core.config_store import ConfigStore
from hydra.utils import instantiate
from omegaconf import MISSING, DictConfig, OmegaConf


@dataclass
class SegParamsConfig:
    compute_interior: bool = True
    compute_edge: bool = True
    compute_vertex: bool = True

@dataclass
class SegLossConfig:
    _target_: str = "pytorch_segmentation_models_trainer.custom_losses.base_loss.SegLoss"
    name: str = "seg"
    gt_channel_selector: SegParamsConfig = field(default_factory=SegParamsConfig)
    bce_coef: float = 0.5
    dice_coef: float = 0.5

@dataclass
class CrossfieldAlignLossConfig:
    _target_: str = "pytorch_segmentation_models_trainer.custom_losses.base_loss.CrossfieldAlignLoss"
    name: str = "crossfield_align"

@dataclass
class CoefsConfig:
    epoch_thresholds: List[float] = field(default_factory=lambda: [0, 5, 10])
    seg: float = 10
    crossfield_align: float = 1
    crossfield_align90: float = 0.2
    crossfield_smooth: float = 0.005
    seg_interior_crossfield: List[float] = field(default_factory=lambda: [0, 0, 0.2])
    seg_edge_crossfield: List[float] = field(default_factory=lambda: [0, 0, 0.2])
    seg_edge_interior: List[float] = field(default_factory= lambda: [0, 0, 0.2])

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
    defaults: List[Any] = field(default_factory=lambda: [
        {"normalization_params": "norm"},
        {"coefs": "coefs"},
        {"seg_loss_params": "seg_loss_params"}
    ])
    normalization_params: NormalizationParams = MISSING
    coefs: CoefsConfig = MISSING
    seg_loss_params: SegLossParamsConfig = MISSING

logger = logging.getLogger(__name__)

cs = ConfigStore.instance()
cs.store(name="seg_loss_config", node=SegLossConfig)
cs.store(group="normalization_params", name="norm", node=NormalizationParams)
cs.store(group="coefs", name="coefs", node=CoefsConfig)
cs.store(group="seg_loss_params", name="seg_loss_params", node=SegLossParamsConfig)
cs.store(name="multi_loss", node=MultiLossConfig)

@hydra.main(config_name="multi_loss")
def build_config(cfg: DictConfig) -> None:
    logger.info(
        OmegaConf.to_yaml(cfg)
    )

if __name__=="__main__":
    build_config()