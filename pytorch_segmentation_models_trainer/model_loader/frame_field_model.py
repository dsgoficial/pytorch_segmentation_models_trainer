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
from logging import log
from collections import OrderedDict
import torch
import torch.nn as nn
import segmentation_models_pytorch as smp
from hydra.utils import instantiate
from omegaconf.dictconfig import DictConfig
from pytorch_segmentation_models_trainer.custom_losses.base_loss import (
    MultiLoss, build_combined_loss)
from pytorch_segmentation_models_trainer.model_loader.model import Model
from segmentation_models_pytorch.base.initialization import (
    initialize_decoder, initialize_head)
from pytorch_segmentation_models_trainer.custom_metrics import metrics
from pytorch_segmentation_models_trainer.utils import tensor_utils
from tqdm import tqdm

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
        frame_field_activation: str = None
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
        if hasattr(self.segmentation_model, 'decoder'):
            self.backbone_output = self.get_out_channels(
                self.segmentation_model.decoder if self.replace_seg_head \
                    else self.segmentation_model.segmentation_head
            )
        else:
            self.backbone_output = self.get_out_channels(self.segmentation_model)
        self.seg_channels = sum(self.seg_params.values())
        self.upsampling = self.get_upsampling_method()
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
            self.upsampling,
            torch.nn.Sigmoid()
        )
    
    def get_upsampling_method(self) -> torch.nn.Module:
        return list(self.segmentation_model.segmentation_head.children())[1] \
            if hasattr(self.segmentation_model, 'segmentation_head') \
                else torch.nn.Identity()
        

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
    
    def get_output(self, x):
        if isinstance(self.segmentation_model, (smp.Unet, smp.UnetPlusPlus, smp.FPN, \
                smp.PSPNet, smp.PAN, smp.DeepLabV3, smp.DeepLabV3Plus)):
            encoder_feats = self.segmentation_model.encoder(x)
            decoder_output = self.segmentation_model.decoder(*encoder_feats)
        else:
            decoder_output = self.segmentation_model(x)
        return decoder_output if not isinstance(decoder_output, OrderedDict) else decoder_output['out']

    def forward(self, x):
        output_dict = OrderedDict()
        decoder_output = self.get_output(x)
        if self.compute_seg:
            segmentation_features = self.seg_module(decoder_output)
            detached_segmentation_features = segmentation_features.clone().detach()
            decoder_output = torch.cat(
                [
                    self.upsampling(decoder_output),
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
        self.loss_norm_is_initializated = False

    def get_loss_function(self) -> MultiLoss:
        """Multi-loss model defined in frame field article
        Returns:
            MultiLoss: Multi loss object
        """
        return build_combined_loss(self.cfg)
    
    def set_encoder_trainable(self, trainable=False):
        return self.set_model_component_trainable(
            self.model.segmentation_model.encoder,
            'Encoder',
            trainable=trainable
        )
    
    def set_decoder_trainable(self, trainable=False):
        return self.set_model_component_trainable(
            self.model.segmentation_model.decoder,
            'Decoder',
            trainable=trainable
        )
    
    def set_seg_module_trainable(self, trainable=False):
        return self.set_model_component_trainable(
            self.seg_module,
            'Seg Module',
            trainable=trainable
        )
    
    def set_model_component_trainable(self, component, component_name, trainable=False):
        for child in component.children():
            for param in child.parameters():
                param.requires_grad = trainable
        print(f"{component_name} weights set to trainable={trainable}")
        return
    
    def set_only_crossfield_trainable(self, trainable=False):
        self.set_encoder_trainable(trainable=trainable)
        self.set_decoder_trainable(trainable=trainable)
        self.set_seg_module_trainable(trainable=trainable)

    
    def compute_iou_metrics(self, y_pred, y_true, individual_metrics_dict):
        iou_thresholds = [0.1, 0.25, 0.5, 0.75, 0.9]
        for iou_threshold in iou_thresholds:
            iou = metrics.iou(
                y_pred.reshape(y_pred.shape[0], -1),
                y_true.reshape(y_true.shape[0], -1),
                threshold=iou_threshold
            )
            mean_iou = torch.mean(iou)
            individual_metrics_dict[f"IoU_{iou_threshold}"] = mean_iou
    
    def compute_loss_norms(self, dl, total_batches):
        self.loss_function.reset_norm()

        t = None
        if self.local_rank == 0:
            t = tqdm(total=total_batches, desc="Init loss norms", leave=False)  # Initialise

        batch_i = 0
        while batch_i < total_batches:
            batch = next(iter(dl))
            # Update loss norms
            batch = tensor_utils.batch_to_cuda(batch) if self.cfg.device == 'cuda' else batch
            pred = self.model(batch['image'])
            self.loss_function.update_norm(pred, batch, batch["image"].shape[0])
            if t is not None:
                t.update(1)
            batch_i += 1

        # Now sync loss norms across GPUs:
        world_size = self.get_world_size()
        if world_size > 1:
            self.loss_function.sync(world_size)
    
    def get_world_size(self):
        if self.cfg.device == 'cpu' or self.cfg.pl_trainer.gpus == 0:
            return 1
        elif isinstance(self.cfg.pl_trainer.gpus, list):
            return len(self.cfg.pl_trainer.gpus)
        elif self.cfg.pl_trainer.gpus == -1:
            return torch.cuda.device_count()
        else:
            return self.cfg.pl_trainer.gpus

    def training_step(self, batch, batch_idx):
        batch['image'] = batch['image'] if self.gpu_train_transform is None \
            else self.gpu_train_transform(batch['image'])
        pred = self.model(batch['image'])
        loss, individual_metrics_dict, extra_dict = self.loss_function(pred, batch, epoch=self.current_epoch)
        y_pred = pred["seg"][:, 0, ...]
        y_true = batch["gt_polygons_image"][:, 0, ...]
        if 'seg' in pred:
            self.compute_iou_metrics(y_pred, y_true, individual_metrics_dict)
            self.log_dict(individual_metrics_dict,\
                prog_bar=True, on_step=True, on_epoch=True, logger=True, sync_dist=True)
        # evaluated_metrics = self.evaluate_metrics(
        #     y_pred, y_true.long(), step_type='train'
        # )
        evaluated_metrics = self.train_metrics(y_pred, y_true.long())
        tensorboard_logs = {k: {'train': v} for k, v in evaluated_metrics.items()}
        tensorboard_logs.update(
            {
                k: {'train': v} for k, v in individual_metrics_dict.items()
            }
        )
        # use log_dict instead of log
        self.log_dict(
            evaluated_metrics, on_step=True, on_epoch=True, prog_bar=False, logger=True, sync_dist=True
        )
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        return {'loss' : loss, 'log': tensorboard_logs}

    def validation_step(self, batch, batch_idx):
        image = batch['image'] if self.gpu_val_transform is None \
            else self.gpu_val_transform(batch['image'])
        pred = self.model(image)
        loss, individual_metrics_dict, extra_dict = self.loss_function(pred, batch, epoch=self.current_epoch)
        y_pred = pred["seg"][:, 0, ...]
        y_true = batch["gt_polygons_image"][:, 0, ...]
        if 'seg' in pred:
            self.compute_iou_metrics(y_pred, y_true, individual_metrics_dict)
            self.log_dict(individual_metrics_dict, prog_bar=True, \
                on_step=False, on_epoch=True, logger=True, sync_dist=True)
        # evaluated_metrics = self.evaluate_metrics(
        #     y_pred, y_true.long(), step_type='val'
        # )
        evaluated_metrics = self.val_metrics(y_pred, y_true.long())
        tensorboard_logs = {k: {'val': v} for k, v in evaluated_metrics.items()}
        tensorboard_logs.update(
            {
                k: {'val': v} for k, v in individual_metrics_dict.items()
            }
        )
        # use log_dict instead of log
        self.log_dict(
            evaluated_metrics, on_step=False, on_epoch=True, prog_bar=False, sync_dist=True, logger=False
        )
        self.log('validation_loss', loss, on_step=True, on_epoch=True, sync_dist=True)
        return {'val_loss': loss, 'log': tensorboard_logs}

    def training_epoch_end(self, outputs):
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        tensorboard_logs = {'avg_loss': {'train' : avg_loss}}
        tensorboard_logs.update(
            self.compute_average_metrics(outputs, self.train_metrics)
        )
        tensorboard_logs.update(
            {
                'avg_'+name: {
                    'train': torch.stack([x['log'][name]['train'] for x in outputs]).mean().detach()
                } for name in outputs[0]['log'].keys() if name not in list(map('train_{0}'.format,self.train_metrics.keys()))
            }
        )
        self.log_dict(tensorboard_logs, logger=True)
        self.log('avg_train_loss', avg_loss, logger=True)

    def validation_epoch_end(self, outputs):
        # OPTIONAL
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        tensorboard_logs = {'avg_loss': {'val' : avg_loss}}
        tensorboard_logs.update(
            self.compute_average_metrics(outputs, self.val_metrics, step_type='val')
        )
        tensorboard_logs.update(
            {
                'avg_'+name: {
                    'val': torch.stack([x['log'][name]['val'] for x in outputs]).mean().detach()
                } for name in outputs[0]['log'].keys() if name not in self.train_metrics.keys()
            }
        )
        self.log_dict(tensorboard_logs, logger=True)
        self.log('avg_train_loss', avg_loss, logger=True)
