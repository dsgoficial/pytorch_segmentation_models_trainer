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
import torch
import torch.nn as nn
from hydra.utils import instantiate

from torch.utils.data import DataLoader

from typing import List, Any

class Model(pl.LightningModule):
    """[summary]

    Args:
        pl ([type]): [description]
    """
    def __init__(self, cfg):
        super(Model, self).__init__()
        self.cfg = cfg
        self.model = instantiate(self.cfg.model)
        self.train_ds = instantiate(self.cfg.train_dataset)
        self.val_ds = instantiate(self.cfg.val_dataset)
        self.loss_function = self.get_loss_function()
        self.train_metrics = self.get_metrics()
        self.val_metrics = self.get_metrics()
    
    def get_metrics(self):
        return nn.ModuleDict([[self.get_metric_name(i), instantiate(i)] for i in self.cfg.metrics])

    def get_metric_name(self, x):
        return x['_target_'].split('.')[-1]


    def evaluate_metrics(self, predicted_masks, masks, step_type='train'):
        if step_type not in ['train', 'val']:
            raise NotImplementedError
        return {
            name: metric(predicted_masks, masks) \
                for name, metric in self.train_metrics.items()
        } if step_type == 'train' else {
            name: metric(predicted_masks, masks) \
                for name, metric in self.val_metrics.items()
        }

    def get_loss_function(self):
        return instantiate(self.cfg.loss)

    def get_optimizer(self):
        return instantiate(self.cfg.optimizer, params=self.parameters())

    def get_scheduler(self, optimizer):
        if 'scheduler' not in self.cfg:
            return torch.optim.lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=self.cfg.hyperparameters.max_lr,
                epochs=self.cfg.hyperparameters.epochs,
                steps_per_epoch=len(self.train_dataloader())
            )
        return instantiate(self.cfg.scheduler, optimizer=optimizer)

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
        lr_scheduler = {
            'scheduler': self.get_scheduler(optimizer),
            'name': 'learning_rate',
            'interval': 'step',
            'frequency': 1
        }
        return [optimizer], [lr_scheduler]

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
        evaluated_metrics = self.evaluate_metrics(
            predicted_masks, masks, step_type='train'
        )
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
        evaluated_metrics = self.evaluate_metrics(
            predicted_masks, masks, step_type='val'
        )
        tensorboard_logs = {k: {'val': v} for k, v in evaluated_metrics.items()}
        # use log_dict instead of log
        self.log_dict(
            evaluated_metrics, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True, logger=False
        )
        return {'val_loss': loss, 'log': tensorboard_logs}
    
    def compute_average_metrics(self, outputs, metric_dict, step_type='train'):
        return {
            'avg_'+name: {
                step_type: torch.stack([x['log'][name][step_type] for x in outputs]).mean()
            } for name in metric_dict.keys()
        }
    
    def training_epoch_end(self, outputs):
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        tensorboard_logs = {'avg_loss': {'train' : avg_loss}}
        tensorboard_logs.update(
            self.compute_average_metrics(outputs, self.train_metrics)
        )
        return {'avg_train_loss': avg_loss,
                'log': tensorboard_logs}

    def validation_epoch_end(self, outputs):
        # OPTIONAL
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        tensorboard_logs = {'avg_loss': {'val' : avg_loss}}
        tensorboard_logs.update(
            self.compute_average_metrics(outputs, self.val_metrics, step_type='val')
        )
        return {'avg_val_loss': avg_loss,
                'log': tensorboard_logs}
