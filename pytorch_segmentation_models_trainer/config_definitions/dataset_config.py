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
from pytorch_segmentation_models_trainer.utils.os_utils import \
    import_module_from_cfg

@dataclass
class DataLoaderConfig:
    shuffle: bool = True
    num_workers: int = 6
    pin_memory: bool = True
    drop_last: bool = True
    prefetch_factor: str = "${hyperparameters.batch_size}"

@dataclass
class DatasetConfig:
    _target_: str = "pytorch_segmentation_models_trainer.dataset_loader.dataset.SegmentationDataset"
    input_csv_path: str = MISSING
    root_dir: str = MISSING
    gpu_augmentation_list: List = field(default_factory=list)
    augmentation_list: List = field(default_factory=list)
    data_loader: DataLoaderConfig = field(default_factory=DataLoaderConfig)

@dataclass
class FrameFieldDatasetConfig(DatasetConfig):
    _target_: str = "pytorch_segmentation_models_trainer.dataset_loader.dataset.FrameFieldSegmentationDataset"
    return_distance_mask: str = "${loss_params.seg_loss_params.use_dist}"
    return_size_mask: str = "${loss_params.seg_loss_params.use_size}"
    image_width: str = "${backbone.input_width}"
    image_height: str = "${backbone.input_height}"