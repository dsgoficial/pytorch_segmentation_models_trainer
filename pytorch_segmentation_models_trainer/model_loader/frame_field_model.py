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
from omegaconf.dictconfig import DictConfig
from pytorch_segmentation_models_trainer.custom_losses.base_loss import (
    MultiLoss, build_combined_loss)
from pytorch_segmentation_models_trainer.model_loader.model import Model
from segmentation_models_pytorch.base.initialization import (
    initialize_decoder, initialize_head)

class FrameFieldModel(nn.Module):
    def __init__(
        self,
        segmentation_model,
        use_batchnorm: bool = True,
        replace_seg_head: bool = True,
        compute_seg: bool = True,
        compute_crossfield: bool = True,
        seg_params: dict = None,
        module_activation: str = None,
        frame_field_activation: str = None,
    ):
        """[summary]

        Args:
            segmentation_model (pytorch model): Chosen segmentation module
            use_batchnorm (bool, optional): Enables the use of batchnorm. 
             Defaults to True.
            replace_seg_head (bool, optional): Enables computing the segmentation.
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
            "compute_vertex": True
        }
        if seg_params is not None:
            for param, value in seg_params.items():
                if param not in self.seg_params:
                    continue
                if not isinstance(value, bool):
                    raise ValueError(f"Parameter {param} must be boolean!")
                self.seg_params[param] = value
        self.segmentation_model = instantiate(segmentation_model) \
            if isinstance(segmentation_model, (str, DictConfig)) else segmentation_model
        self.replace_seg_head = replace_seg_head
        self.compute_seg = compute_seg
        self.compute_crossfield = compute_crossfield
        self.use_batchnorm = use_batchnorm
        self.frame_field_activation = frame_field_activation
        self.module_activation = module_activation
        self.backbone_output = self.get_out_channels(
            self.segmentation_model.decoder if self.replace_seg_head \
                else self.segmentation_model.sementation_head
        )
        self.seg_channels = sum(self.seg_params.values())
        self.seg_module = self.get_seg_module()
        self.crossfield_module = self.get_crossfield_module()
        self.initialize()

    def get_seg_module(self) -> torch.nn.Sequential:
        """Prepares the seg module

        Returns:
            torch.nn.Sequential: Sequential module that computes the seg.
        """
        if not self.replace_seg_head:
            return self.segmentation_model.segmentation_head
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
        return torch.nn.Sequential(
            torch.nn.Conv2d(
                self.backbone_output + self.seg_channels,
                self.backbone_output,
                kernel_size=3,
                padding=1
            ),
            torch.nn.BatchNorm2d(self.backbone_output) if self.use_batchnorm \
                else nn.Identity(),
            torch.nn.ELU() if self.module_activation is None \
                else instantiate(self.module_activation),
            torch.nn.Conv2d(
                self.backbone_output,
                self.crossfield_channels,
                kernel_size=1
            ),
            torch.nn.Tanh() if self.frame_field_activation is None \
                else instantiate(self.frame_field_activation)
        )

    def forward(self, x):
        output_dict = dict()
        encoder_feats = self.segmentation_model.encoder(x)
        decoder_output = self.segmentation_model.decoder(*encoder_feats)
        if self.compute_seg:
            segmentation_features = self.seg_module(decoder_output)
            detached_segmentation_features = segmentation_features.clone().detach()
            decoder_output = torch.cat(
                [
                    decoder_output,
                    detached_segmentation_features
                ], dim=1
            )
            output_dict["seg"] = segmentation_features
        if self.compute_crossfield:
            output_dict["crossfield"] = 2 * self.crossfield_module(decoder_output)
        return output_dict

    def initialize(self):
        if self.replace_seg_head:
            initialize_head(self.seg_module)
        if self.compute_crossfield:
            initialize_decoder(self.crossfield_module)

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

class FrameFieldSegmentationPLModel(Model):
    def __init__(self, cfg):
        super(FrameFieldSegmentationPLModel, self).__init__(cfg)

    def get_loss_function(self) -> MultiLoss:
        """Multi-loss model defined in frame field article
        Returns:
            MultiLoss: Multi loss object
        """
        return build_combined_loss(self.cfg)

    def training_step(self, batch, batch_idx):
        pred = self.model(batch['image'])
        loss, individual_metrics_dict, extra_dict = self.loss_function(pred, batch, epoch=self.current_epoch)
        y_pred = pred["seg"][:, 0, ...]
        y_true = batch["gt_polygons_image"][:, 0, ...]
        evaluated_metrics = self.evaluate_metrics(
            y_pred, y_true, step_type='train'
        )
        tensorboard_logs = {k: {'train': v} for k, v in evaluated_metrics.items()}
        # use log_dict instead of log
        self.log_dict(
            evaluated_metrics, on_step=True, on_epoch=False, prog_bar=True, logger=False
        )
        return {'loss' : loss, 'log': tensorboard_logs}

    def validation_step(self, batch, batch_idx):
        pred = self.model(batch['image'])
        loss, individual_metrics_dict, extra_dict = self.loss_function(pred, batch, epoch=self.current_epoch)
        y_pred = pred["seg"][:, 0, ...]
        y_true = batch["gt_polygons_image"][:, 0, ...]
        evaluated_metrics = self.evaluate_metrics(
            y_pred, y_true, step_type='train'
        )
        tensorboard_logs = {k: {'val': v} for k, v in evaluated_metrics.items()}
        # use log_dict instead of log
        self.log_dict(
            evaluated_metrics, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True, logger=False
        )
        self.log('validation_loss', loss, on_step=True, on_epoch=True, sync_dist=True)
        return {'val_loss': loss, 'log': tensorboard_logs}
