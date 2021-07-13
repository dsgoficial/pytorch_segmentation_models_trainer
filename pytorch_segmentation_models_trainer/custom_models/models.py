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

import os
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
from segmentation_models_pytorch.base import SegmentationHead
from segmentation_models_pytorch.base import initialization as init
from pytorch_segmentation_models_trainer.custom_models.utils import \
    _SimpleSegmentationModel
from pytorch_segmentation_models_trainer.custom_models.hrnet_models import seg_hrnet_ocr
from torchvision.datasets.utils import download_url

current_dir = os.path.dirname(__file__)

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
    _target_: str = "pytorch_segmentation_models_trainer.custom_models.unet_resnet.UNetResNetBackbone"
    encoder_depth: int = 34
    num_filters: int = 32
    dropout_2d: float = 0.2
    pretrained: bool = True
    is_deconv: bool = False

class BaseSegmentationModel(torch.nn.Module):
    def __init__(self, backbone, output_conv_kernel=None, features=1):
        super(BaseSegmentationModel, self).__init__()
        self.backbone = backbone
        self.output_conv_kernel = output_conv_kernel
        self.features = features
    
    def __post_init__(self):
        self.backbone = instantiate(self.backbone)
        if self.output_conv_kernel is not None:
            self.backbone.classifier = torch.nn.Sequential(
                *list(self.backbone.classifier.children())[:-1],
                torch.nn.Conv2d(
                    self.output_conv_kernel,
                    self.features,
                    kernel_size=(1, 1),
                    stride=(1, 1)
                )
            )

    def forward(self, x):
        return self.backbone(x)['out']

class DeepLab101(BaseSegmentationModel):
    def __init__(self, backbone=None, features=128):
        backbone = DeepLab101SegmentationBackbone if backbone is None else backbone
        super(DeepLab101, self).__init__(backbone=backbone,
                                         output_conv_kernel=256, features=features)
        super(DeepLab101, self).__post_init__()

class DeepLab50(BaseSegmentationModel):
    def __init__(self, backbone=None, features=128):
        backbone = DeepLab50SegmentationBackbone if backbone is None else backbone
        super(DeepLab50, self).__init__(backbone=backbone,
                                         output_conv_kernel=256, features=features)
        super(DeepLab50, self).__post_init__()

class FCN101(BaseSegmentationModel):
    def __init__(self, backbone=None, features=256):
        backbone = FCN50SegmentationBackbone if backbone is None else backbone
        super(FCN101, self).__init__(backbone=backbone,
                                         output_conv_kernel=512, features=features)
        super(FCN101, self).__post_init__()

class FCN50(BaseSegmentationModel):
    def __init__(self, backbone=None, features=256):
        backbone = FCN50SegmentationBackbone if backbone is None else backbone
        super(FCN50, self).__init__(backbone=backbone,
                                         output_conv_kernel=512, features=features)
        super(FCN50, self).__post_init__()

class UNetResNet(BaseSegmentationModel):

    def __post_init__(self):
        self.backbone = _SimpleSegmentationModel(
            instantiate(self.backbone),
            classifier=torch.nn.Identity()
        )
    
    def __init__(self, features=256):
        super(UNetResNet, self).__init__(backbone=UNetResNetSegmentationBackbone,
                                         output_conv_kernel=None, features=features)
        super(UNetResNet, self).__post_init__()

class HRNetOCRW48(torch.nn.Module):
    pretrained_weights_names = {
        'cityscapes' : 'hrnet_ocr_w48_cityscapes',
        'lip' : 'hrnet_ocr_w48_lip',
        'pascal' : 'hrnet_ocr_w48_pascal',
    }
    pretrained_weights_download_links = {
        'cityscapes': 'https://github.com/hsfzxjy/models.storage/releases/download/HRNet-OCR/hrnet_ocr_cs_trainval_8227_torch11.pth',
        'lip': 'https://github.com/hsfzxjy/models.storage/releases/download/HRNet-OCR/hrnet_ocr_lip_5648_torch04.pth',
        'pascal': 'https://github.com/hsfzxjy/models.storage/releases/download/HRNet-OCR/hrnet_ocr_pascal_ctx_5618_torch11.pth'
    }
    pretrained_dataset_outputs = {
        'cityscapes' : 19,
        'lip' : 20,
        'pascal' : 59,
    }
    def __init__(self, in_channels=3, classes=1, upsampling=4, activation=None, pretrained=None):
        super(HRNetOCRW48, self).__init__()
        self.cfg = OmegaConf.load(
            os.path.join(current_dir, "hrnet_models", "conf", "hrnet_ocr_w48.yml")
        )
        self.cfg.n_classes = self.pretrained_dataset_outputs[pretrained] \
            if pretrained is not None and pretrained in self.pretrained_dataset_outputs \
                else self.cfg.n_classes
        self.out_channels = self.cfg.n_classes
        self.backbone = instantiate(self.cfg)
        self.segmentation_head = SegmentationHead(
            in_channels=self.cfg.n_classes,
            out_channels=classes,
            activation=activation,
            kernel_size=1,
            upsampling=upsampling,
        )
        pretrained_weights = self.get_pretrained_weights(pretrained) if pretrained is not None \
            else self.cfg.pretrained

        self.backbone.init_weights(pretrained_weights)
        init.initialize_head(self.segmentation_head)
    
    def get_pretrained_weights(self, pretrained):
        if pretrained not in self.pretrained_weights_names:
            raise RuntimeError("Invalid weight. Allowed values are cityscapes, lip and pascal")
        root_name = os.path.join(current_dir, 'pretrained')
        file_name = f'{self.pretrained_weights_names[pretrained]}.pth'
        weight_file = os.path.join(root_name, file_name)
        if os.path.isfile(weight_file):
            return weight_file
        download_url(
            url=self.pretrained_weights_download_links[pretrained],
            root=root_name,
            filename=file_name    
        )
        return weight_file
    
    def forward(self, x, apply_segmentation_head=False):
        return self.backbone(x)['out'] if not apply_segmentation_head \
            else self.segmentation_head(self.backbone(x)['out'])

if __name__ == "__main__":
    x = HRNetOCRW48()
    print(x)