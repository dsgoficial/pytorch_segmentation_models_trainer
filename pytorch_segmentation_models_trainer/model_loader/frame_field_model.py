# -*- coding: utf-8 -*-
"""
/***************************************************************************
 pytorch_segmentation_models_trainer
                              -------------------
        begin                : 2021-03-19
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
 *   Code inspired by the one in                                           *
 *   https://github.com/Lydorn/Polygonization-by-Frame-Field-Learning/     *
 ****
"""
import albumentations as A
import pytorch_lightning as pl
import torch
import torch.nn as nn
from hydra.utils import instantiate

from torch.utils.data import DataLoader

from typing import List, Any

from pytorch_segmentation_models_trainer.model_loader.model import Model

class FrameFieldBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        use_batchnorm=True,
        compute_seg=True,
        compute_crossfield=True,
        seg_params=None
    ):
        super().__init__()
        self.crossfield_channels = 4
        self.seg_params = {
            "compute_interior": True,
            "compute_edge": True,
            "compute_vertex": False
        }
        if seg_params is not None:
            for param in seg_params:
                if param not in self.seg_params:
                    continue
                if not isinstance(param, bool):
                    raise ValueError(f"Parameter {param} must be boolean!")
                self.seg_params[param] = seg_params
        self.compute_seg = compute_seg
        self.compute_crossfield = compute_crossfield
        self.use_batchnorm = use_batchnorm
        self.in_channels = in_channels
        self.seg_channels = sum(self.seg_params.values())
        self.seg_module = self.get_seg_module()
        self.crosfield_module = self.get_crossfield_module()
        
    def get_seg_module(self):
        if not self.seg_module:
            return None
    
    def get_crossfield_module(self):
        if not self.compute_crossfield:
            return None
        self.crossfield_conv1 = torch.nn.Conv2d(
            self.in_channels + self.seg_channels,
            self.in_channels,
            kernel_size=3,
            padding=1
        )
        self.batch_norm = torch.nn.BatchNorm2d(self.in_channels)
        self.module_activation = torch.nn.ELU()
        self.crossfield_conv2 = torch.nn.Conv2d(
            self.in_channels,
            self.crossfield_channels,
            kernel_size=1
        )
        self.frame_field_activation = torch.nn.Tanh()

        parameter_list = [
            self.crossfield_conv1,
            self.batch_norm,
            self.module_activation,
            self.crossfield_conv2,
            self.frame_field_activation
        ] if self.use_batchnorm else [
            self.conv1,
            self.module_activation,
            self.conv2,
            self.frame_field_activation
        ]

        return torch.nn.Sequential(
            parameter_list
        ) 

    def forward(self, x):
        return self.crossfield_module(x)



class FrameFieldModel(Model):
    def __init__(self, cfg):
        super(FrameFieldModel).__init__(cfg)

    def get_model(self):
        self.segmentation_block = instantiate(self.cfg.model.segmentation_block)
        backbone_output = 
        self.crossfield_model = FrameFieldBlock(backbone_output)
        return super().get_model()
    
    def get_loss_function(self):
        """Multi-loss model

        Returns:
            [type]: [description]
        """
        return None
    
    def training_step(self, batch, batch_idx):
        # TODO
        return super().training_step(batch, batch_idx)
    
    def validation_step(self, batch, batch_idx):
        # TODO
        return super().validation_step(batch, batch_idx)