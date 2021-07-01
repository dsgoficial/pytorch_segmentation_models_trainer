# -*- coding: utf-8 -*-
"""
/***************************************************************************
 pytorch_segmentation_models_trainer
                              -------------------
        begin                : 2021-03-01
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
import albumentations as A
import pytorch_lightning as pl
import torchmetrics
import torch
import torch.nn as nn
from hydra.utils import instantiate

from torch.utils.data import DataLoader

from typing import List, Any
from pytorch_segmentation_models_trainer.utils.model_utils import (
    replace_activation
)
class Model(pl.LightningModule):
    """[summary]

    Args:
        pl ([type]): [description]
    """
    def __init__(self, cfg):
        super(Model, self).__init__()
        self.cfg = cfg
        self.model = self.get_model()
        self.train_ds = instantiate(self.cfg.train_dataset)
        self.val_ds = instantiate(self.cfg.val_dataset)
        self.loss_function = self.get_loss_function()
        metrics = torchmetrics.MetricCollection(
            [instantiate(i) for i in self.cfg.metrics]
        )
        self.train_metrics = metrics.clone(prefix='train_')
        self.val_metrics = metrics.clone(prefix='val_')
        # self.train_metrics = self.get_metrics()
        # self.val_metrics = self.get_metrics()
        self.gpu_train_transform = None if 'gpu_augmentation_list' not in self.cfg.train_dataset \
            else self.get_gpu_augmentations(self.cfg.train_dataset.gpu_augmentation_list)
        self.gpu_val_transform = None if 'gpu_augmentation_list' not in self.cfg.val_dataset \
            else self.get_gpu_augmentations(self.cfg.val_dataset.gpu_augmentation_list)

    def get_model(self):
        model = instantiate(self.cfg.model)
        if 'replace_model_activation' in self.cfg:
            old_activation = instantiate(self.cfg.replace_model_activation.old_activation)
            new_activation = instantiate(self.cfg.replace_model_activation.new_activation)
            replace_activation(model, old_activation, new_activation)
        return model

    def get_metrics(self):
        return nn.ModuleDict([[self.get_metric_name(i), instantiate(i)] for i in self.cfg.metrics])

    def get_metric_name(self, x):
        return x['_target_'].split('.')[-1]

    def get_gpu_augmentations(self, augmentation_list):
        return torch.nn.Sequential(*[
            instantiate(aug) for aug in augmentation_list
        ])

    def evaluate_metrics(self, predicted_masks, masks, step_type='train'):
        if step_type not in ['train', 'val']:
            raise NotImplementedError
        iterate_dict = self.train_metrics if step_type == 'train' \
            else self.val_metrics
        return {
            name: metric(predicted_masks, masks) \
                for name, metric in iterate_dict.items()
        }

    def get_loss_function(self):
        return instantiate(self.cfg.loss)

    def get_optimizer(self):
        return instantiate(self.cfg.optimizer, params=self.parameters())

    def set_encoder_trainable(self, trainable=False):
        """Freezes or unfreezes the model encoder.

        Args:
            trainable (bool, optional): Sets the encoder weights trainable.
            Defaults to False.
        """
        for child in self.model.encoder.children():
            for param in child.parameters():
                param.requires_grad = trainable
        print(f"\nEncoder weights set to trainable={trainable}\n")
        return

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        # REQUIRED
        optimizer = self.get_optimizer()
        scheduler_list = []
        if 'scheduler_list' not in self.cfg:
            return [optimizer], scheduler_list
        for item in self.cfg.scheduler_list:
            dict_item = dict(item)
            dict_item['scheduler'] = instantiate(item.scheduler, optimizer=optimizer)
            scheduler_list.append(dict_item)
        return [optimizer], scheduler_list

    def train_dataloader(self):
        return DataLoader(
            self.train_ds,
            batch_size=self.cfg.hyperparameters.batch_size,
            shuffle=self.cfg.train_dataset.data_loader.shuffle,
            num_workers=self.cfg.train_dataset.data_loader.num_workers,
            pin_memory=self.cfg.train_dataset.data_loader.pin_memory \
                if 'pin_memory' in self.cfg.train_dataset.data_loader else True,
            drop_last=self.cfg.train_dataset.data_loader.drop_last \
                if 'drop_last' in self.cfg.train_dataset.data_loader else True,
            prefetch_factor=self.cfg.train_dataset.data_loader.prefetch_factor \
                if 'prefetch_factor' in self.cfg.train_dataset.data_loader \
                    else 4*self.hyperparameters.batch_size
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_ds,
            batch_size=self.cfg.hyperparameters.batch_size,
            shuffle=self.cfg.val_dataset.data_loader.shuffle if 'shuffle' \
                in self.cfg.val_dataset.data_loader else False,
            num_workers=self.cfg.val_dataset.data_loader.num_workers,
            pin_memory=self.cfg.val_dataset.data_loader.pin_memory \
                if 'pin_memory' in self.cfg.val_dataset.data_loader else True,
            drop_last=self.cfg.val_dataset.data_loader.drop_last \
                if 'drop_last' in self.cfg.val_dataset.data_loader else True,
            prefetch_factor=self.cfg.val_dataset.data_loader.prefetch_factor \
                if 'prefetch_factor' in self.cfg.val_dataset.data_loader \
                    else 4*self.hyperparameters.batch_size
        )

    def training_step(self, batch, batch_idx):
        images, masks = batch.values()
        masks = masks.long()
        predicted_masks = self(images)
        loss = self.loss_function(predicted_masks, masks)
        # evaluated_metrics = self.evaluate_metrics(
        #     predicted_masks, masks, step_type='train'
        # )
        evaluated_metrics = self.train_metrics(predicted_masks, masks)
        tensorboard_logs = {k: {'train': v} for k, v in evaluated_metrics.items()}
        # use log_dict instead of log
        self.log_dict(
            evaluated_metrics, on_step=True, on_epoch=False, prog_bar=True, logger=False
        )
        return {'loss' : loss, 'log': tensorboard_logs}

    def validation_step(self, batch, batch_idx):
        images, masks = batch.values()
        masks = masks.long()
        predicted_masks = self(images)
        loss = self.loss_function(predicted_masks, masks)
        # evaluated_metrics = self.evaluate_metrics(
        #     predicted_masks, masks, step_type='val'
        # )
        evaluated_metrics = self.val_metrics(predicted_masks, masks)
        tensorboard_logs = {k: {'val': v} for k, v in evaluated_metrics.items()}
        # use log_dict instead of log
        self.log_dict(
            evaluated_metrics, on_step=True, on_epoch=True, prog_bar=True, logger=False
        )
        self.log('validation_loss', loss, on_step=True, on_epoch=True)
        return {'val_loss': loss, 'log': tensorboard_logs}
    
    def compute_average_metrics(self, outputs, metric_dict, step_type='train'):
        return {
            'avg_'+name: {
                step_type: torch.stack([x['log'][name if step_type in name else f'{step_type}_'+name][step_type] for x in outputs]).mean()
            } for name in metric_dict.keys()
        }
    
    def training_epoch_end(self, outputs):
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        tensorboard_logs = {'avg_loss': {'train' : avg_loss}}
        tensorboard_logs.update(
            self.compute_average_metrics(outputs, self.train_metrics)
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
        return {'avg_val_loss': avg_loss,
                'log': tensorboard_logs}
