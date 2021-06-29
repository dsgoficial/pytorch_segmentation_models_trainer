# -*- coding: utf-8 -*-
"""
/***************************************************************************
 pytorch_segmentation_models_trainer
                              -------------------
        begin                : 2021-06-29
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
import logging
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Dict, List

import hydra
import torch
import torchvision
from hydra.core.config_store import ConfigStore
from hydra.utils import instantiate
from omegaconf import MISSING, DictConfig, OmegaConf
from pytorch_segmentation_models_trainer.custom_models.utils import \
    _SimpleSegmentationModel


@dataclass
class DeepLab101SegmentationBackbone:
    _target_: str = "torchvision.models.segmentation.deeplabv3_resnet101"
    pretrained: bool = True
    num_classes: int = 21


@dataclass
class DeepLab50SegmentationBackbone:
    _target_: str = "torchvision.models.segmentation.deeplabv3_resnet50"
    pretrained: bool = True
    num_classes: int = 21


@dataclass
class FCN101SegmentationBackbone:
    _target_: str = "torchvision.models.segmentation.fcn_resnet101"
    pretrained: bool = True
    num_classes: int = 21


@dataclass
class FCN50SegmentationBackbone:
    _target_: str = "torchvision.models.segmentation.fcn_resnet50"
    pretrained: bool = True
    num_classes: int = 21


@dataclass
class UNetResNetSegmentationBackbone:
    _target_: "pytorch_segmentation_models_trainer.custom_models.unet_resnet.UNetResNetBackbone"
    encoder_depth: int = 34
    num_filters: int = 32
    dropout_2d: float = 0.2
    pretrained: bool = True
    is_deconv: bool = False


@dataclass
class BaseSegmentationModel:
    features: 1
    name: MISSING
    backbone: MISSING
    output_conv_kernel: int = MISSING

    def __post_init__(self):
        self.backbone.classifier = torch.nn.Sequential(
            *list(self.backbone.classifier.children())[:-1],
            torch.nn.Conv2d(
                self.output_conv_kernel, self.features, kernel_size=(1, 1), stride=(1, 1)
            )
        )


@dataclass
class DeepLab101(BaseSegmentationModel):
    backbone: DeepLab101SegmentationBackbone = field(
        default_factory=DeepLab101SegmentationBackbone)
    name: str "deeplab101"
    output_conv_kernel: 256


@dataclass
class DeepLab50(BaseSegmentationModel):
    backbone: DeepLab50SegmentationBackbone = field(
        default_factory=DeepLab50SegmentationBackbone)
    name: str "deeplab50"
    output_conv_kernel: 256


@dataclass
class FCN101(BaseSegmentationModel):
    backbone: FCN101SegmentationBackbone = field(
        default_factory=FCN101SegmentationBackbone)
    name: "fcn101"
    output_conv_kernel: 512


@dataclass
class FCN50(BaseSegmentationModel):
    backbone: FCN101SegmentationBackbone = field(
        default_factory=FCN101SegmentationBackbone)
    name: "fcn50"
    output_conv_kernel: 512


@dataclass
class UNetResNet:
    backbone: UNetResNetSegmentationBackbone = field(
        default_factory=UNetResNetSegmentationBackbone)
    name: "unet_resnet"

    def __post_init__(self):
        self.backbone = _SimpleSegmentationModel(
            self.backbone,
            classifier=torch.nn.Identity()
        )
