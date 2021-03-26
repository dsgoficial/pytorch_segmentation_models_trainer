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
import torch
import torch.nn as nn
from hydra.utils import instantiate

from typing import List, Any

from pytorch_segmentation_models_trainer.model_loader.model import Model
class FrameFieldModel(nn.Module):
    def __init__(
        self,
        segmentation_backbone,
        use_batchnorm: bool =True,
        compute_seg: bool =True,
        compute_crossfield: bool=True,
        seg_params: dict=None
    ):
        """[summary]

        Args:
            segmentation_backbone (pytorch model): Chosen segmentation module
            use_batchnorm (bool, optional): Enables the use of batchnorm. 
             Defaults to True.
            compute_seg (bool, optional): Enables computing the segmentation.
             Defaults to True.
            compute_crossfield (bool, optional): Enables computing the crossfield.
             Defaults to True.
            seg_params (dict, optional): Additional segmentation parameters.
             Defaults to None.

        Raises:
            ValueError: [description]
        """
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
        self.segmentation_backbone = segmentation_backbone
        self.compute_seg = compute_seg
        self.compute_crossfield = compute_crossfield
        self.use_batchnorm = use_batchnorm
        self.backbone_output = self.get_out_channels(
            self.segmentation_backbone
        )
        self.seg_channels = sum(self.seg_params.values())
        self.seg_module = self.get_seg_module()
        self.crosfield_module = self.get_crossfield_module()

    def get_seg_module(self) -> torch.nn.Sequential:
        """Prepares the seg module

        Returns:
            torch.nn.Sequential: Sequential module that computes the seg.
        """
        if not self.compute_seg:
            return None
        return torch.nn.Sequential(
            torch.nn.Conv2d(self.backbone_output, self.backbone_output, 3, padding=1),
            torch.nn.BatchNorm2d(self.backbone_output),
            torch.nn.ELU(),
            torch.nn.Conv2d(self.backbone_output, self.seg_channels, 1),
            torch.nn.Sigmoid()
        )

    def get_crossfield_module(self) -> torch.nn.Sequential:
        """Prepares the crossfield module

        Returns:
            torch.nn.Sequential: Sequential module that computes the crossfield.
        """
        if not self.compute_crossfield:
            return None
        self.crossfield_conv1 = torch.nn.Conv2d(
            self.backbone_output + self.seg_channels,
            self.backbone_output,
            kernel_size=3,
            padding=1
        )
        self.batch_norm = torch.nn.BatchNorm2d(self.backbone_output)
        self.module_activation = torch.nn.ELU()
        self.crossfield_conv2 = torch.nn.Conv2d(
            self.backbone_output,
            self.crossfield_channels,
            kernel_size=1
        )
        self.frame_field_activation = torch.nn.Tanh()
        return torch.nn.Sequential(
            self.crossfield_conv1,
            self.batch_norm if self.use_batchnorm else nn.Identity(),
            self.module_activation,
            self.crossfield_conv2,
            self.frame_field_activation
        )

    def forward(self, x):
        output_dict = dict()
        segmentation_features = self.segmentation_backbone(x)
        if self.compute_seg:
            seg = self.seg_module(segmentation_features)
            detached_seg = seg.clone().detach()
            segmentation_features = torch.cat(
                [
                    segmentation_features,
                    detached_seg
                ], dim=1
            )
            output_dict["seg"] = seg
        if self.compute_crossfield:
            output_dict["crossfield"] = 2 * self.crosfield_module(segmentation_features)
        return output_dict

    @staticmethod
    def get_out_channels(module):
        """Method reused from 
        https://github.com/Lydorn/Polygonization-by-Frame-Field-Learning/blob/master/frame_field_learning/model.py

        Args:
            module ([type]): [description]

        Returns:
            [type]: [description]
        """
        if hasattr(module, "out_channels"):
            return module.out_channels
        children = list(module.children())
        i = 1
        out_channels = None
        while out_channels is None and i <= len(children):
            last_child = children[-i]
            out_channels = FrameFieldModel.get_out_channels(last_child)
            i += 1
        # If we get out of the loop but out_channels is None,
        # then the prev child of the parent module will be checked, etc.
        return out_channels

class FrameFieldSegmentationModel(Model):
    def __init__(self, cfg):
        super(FrameFieldSegmentationModel).__init__(cfg)

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
