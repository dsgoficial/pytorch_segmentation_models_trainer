# -*- coding: utf-8 -*-
"""
/***************************************************************************
 pytorch_segmentation_models_trainer
                              -------------------
        begin                : 2021-07-19
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
from typing import Any, Dict, List

import hydra
import torch
from hydra.core.config_store import ConfigStore
from hydra.utils import instantiate
from omegaconf import MISSING, DictConfig, OmegaConf
from pytorch_segmentation_models_trainer.custom_callbacks.training_callbacks import \
    FrameFieldComputeWeightNormLossesCallback
from pytorch_segmentation_models_trainer.custom_losses.loss_config_definition import \
    LossParamsConfig
from pytorch_segmentation_models_trainer.model_loader.frame_field_model import \
    FrameFieldSegmentationPLModel
from pytorch_segmentation_models_trainer.model_loader.model import Model
from pytorch_segmentation_models_trainer.config_definitions.dataset_config import DatasetConfig

@dataclass
class BackboneConfig:
    name: str = "resnet152"
    input_width: int = 224
    input_height: int = 224

@dataclass
class PLTrainerConfig:
    max_epochs: str = "${hyperparameters.epochs}"
    gpus: int = -1
    precision: int = 32
    default_root_dir: str = "/experiment_data/${backbone.name}_${hyperparameters.model_name}"

@dataclass
class SegParams:
    compute_interior: bool = True
    compute_edge: bool = True
    compute_vertex: bool = False

@dataclass
class OptimizerConfig:
    _target_: str = "torch.optim.AdamW"
    lr: str = "${hyperparameters.max_lr}"
    weight_decay: float = 1e-3

@dataclass
class Hyperparameters:
    model_name: str = "unet"
    backbone: str = MISSING
    batch_size: int = 16
    epochs: int = 10
    max_lr: float = 1e-2
    classes: int = 1

@dataclass
class SchedulerConfig:
    _target_: str = "torch.optim.lr_scheduler.OneCycleLR"
    max_lr: str = "${hyperparameters.max_lr}"
    steps_per_epoch: int = 5161
    epochs: str = "${hyperparameters.epochs}"

@dataclass
class SchedulerItemConfig:
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    name: str = "learning_rate"
    interval: str = "step"
    frequency: int = 1
    monitor: str = "avg_val_loss"

@dataclass
class CallbackConfig:
    _target_: str = "pytorch_lightning.callbacks.LearningRateMonitor"

@dataclass
class MetricConfig:
    _target_: str = MISSING

@dataclass
class TrainConfig:
    pl_model: Model = field(default_factory=Model)
    model: torch.nn.Module = field(default_factory=torch.nn.Module)
    backbone: BackboneConfig = field(default_factory=BackboneConfig)
    compute_seg: bool = True
    compute_crossfield: bool = False
    device: str = "cpu"
    seg_params: SegParams = field(default_factory=SegParams)
    loss_params: LossParamsConfig = field(default_factory=LossParamsConfig)
    optimizer: List[OptimizerConfig] = field(default_factory=lambda : [OptimizerConfig()])
    hyperparameters: Hyperparameters = field(default_factory=Hyperparameters)
    scheduler_list: List[SchedulerItemConfig] = field(default_factory=lambda : [SchedulerItemConfig()])
    callbacks: List[CallbackConfig] = field(default_factory=[CallbackConfig()])
    pl_trainer: PLTrainerConfig = field(default_factory=PLTrainerConfig)
    train_dataset: DatasetConfig = field(default_factory=DatasetConfig)
    val_dataset: DatasetConfig = field(default_factory=DatasetConfig)
    metrics: List[MetricConfig] = MISSING
