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
from hydra.utils import instantiate
from pytorch_lightning.metrics import (
    Accuracy, F1, IoU, Precision, Recall
)
#precision e recall com problema no pytorch lightning 1.2, 
# retirar e depois ver o que fazer
from torch.utils.data import DataLoader

class Model(pl.LightningModule):
    """[summary]

    Args:
        pl ([type]): [description]
    """
    def __init__(self, cfg):
        super(Model, self).__init__()
        self.cfg = cfg
        self.model = instantiate(self.cfg.model)
        self.loss_function = self.get_loss_function()
        metrics = pl.metrics.MetricCollection(
            [
                Accuracy(),
                Precision(),
                Recall(),
                F1(num_classes=self.cfg.model.classes)
            ]
        )
        self.train_metrics = metrics.clone()
        self.validation_metrics = metrics.clone()

    def get_loss_function(self):
        return instantiate(self.cfg.loss)

    def get_optimizer(self):
        return instantiate(self.cfg.optimizer, params=self.parameters())

    def get_scheduler(self, optimizer):
        return torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=self.cfg.hyperparameters.max_lr,
            epochs=self.cfg.hyperparameters.epochs,
            steps_per_epoch=len(self.train_dataloader())
        )

    def forward(self, x):
        return self.model(x.float())

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
        train_ds = instantiate(self.cfg.train_dataset)
        train_dl = DataLoader(
            train_ds,
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
        return train_dl

    def val_dataloader(self):
        val_ds = instantiate(self.cfg.val_dataset)
        val_dl = DataLoader(
            val_ds,
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
        return val_dl

    def training_step(self, batch, batch_idx):
        images, masks = batch.values()
        masks = masks.long()
        predicted_masks = self(images)
        loss = self.loss_function(predicted_masks, masks)
        self.train_metrics(predicted_masks, masks)
        # use log_dict instead of log
        self.log_dict(
            self.train_metrics, on_step=True, on_epoch=False
        )
        return loss

    def validation_step(self, batch, batch_idx):
        images, masks = batch.values()
        masks = masks.long()
        predicted_masks = self(images)
        loss = self.loss_function(predicted_masks, masks)
        self.validation_metrics(predicted_masks, masks)
        # use log_dict instead of log
        self.log_dict(
            self.validation_metrics, on_step=True, on_epoch=True
        )
        return {'val_loss': loss}

    def validation_epoch_end(self, outputs):
        # OPTIONAL
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
   
        tensorboard_logs = {'val_loss': avg_loss}
        return {'avg_val_loss': avg_loss,
                'log': tensorboard_logs}
