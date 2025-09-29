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
from pytorch_segmentation_models_trainer.utils.model_utils import replace_activation


class Model(pl.LightningModule):
    """Base Model class compatible with PyTorch Lightning 2.0+"""

    def __init__(self, cfg):
        super(Model, self).__init__()
        self.cfg = cfg
        self.model = self.get_model()
        self.train_ds = instantiate(self.cfg.train_dataset, _recursive_=False)
        self.val_ds = instantiate(self.cfg.val_dataset, _recursive_=False)
        self.loss_function = self.get_loss_function()
        
        # Save hyperparameters for better checkpointing
        self.save_hyperparameters(ignore=['model', 'loss_function', 'train_ds', 'val_ds'])
        
        if "metrics" in self.cfg:
            metrics = torchmetrics.MetricCollection(
                [instantiate(i, _recursive_=False) for i in self.cfg.metrics]
            )
            # Use forward slash for grouping in TensorBoard
            self.train_metrics = metrics.clone(prefix="train/")
            self.val_metrics = metrics.clone(prefix="val/")
        
        self.gpu_train_transform = (
            None
            if "gpu_augmentation_list" not in self.cfg.train_dataset
            else self.get_gpu_augmentations(
                self.cfg.train_dataset.gpu_augmentation_list
            )
        )
        self.gpu_val_transform = (
            None
            if "gpu_augmentation_list" not in self.cfg.val_dataset
            else self.get_gpu_augmentations(self.cfg.val_dataset.gpu_augmentation_list)
        )

    def get_model(self):
        model = instantiate(self.cfg.model, _recursive_=False)
        if "replace_model_activation" in self.cfg:
            old_activation = instantiate(
                self.cfg.replace_model_activation.old_activation, _recursive_=False
            )
            new_activation = instantiate(
                self.cfg.replace_model_activation.new_activation, _recursive_=False
            )
            replace_activation(model, old_activation, new_activation)
        return model

    def get_gpu_augmentations(self, augmentation_list):
        return torch.nn.Sequential(
            *[instantiate(aug, _recursive_=False) for aug in augmentation_list]
        )

    def get_loss_function(self):
        return instantiate(self.cfg.loss, _recursive_=False)

    def get_optimizer(self):
        return instantiate(
            self.cfg.optimizer, params=self.parameters(), _recursive_=False
        )

    def set_encoder_trainable(self, trainable=False):
        """Freezes or unfreezes the model encoder."""
        for child in self.model.encoder.children():
            for param in child.parameters():
                param.requires_grad = trainable
        print(f"\nEncoder weights set to trainable={trainable}\n")
        return

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        optimizer = self.get_optimizer()
        scheduler_list = []
        if "scheduler_list" not in self.cfg:
            return [optimizer], scheduler_list
        for item in self.cfg.scheduler_list:
            dict_item = dict(item)
            dict_item["scheduler"] = instantiate(
                item.scheduler, optimizer=optimizer, _recursive_=False
            )
            scheduler_list.append(dict_item)
        return [optimizer], scheduler_list

    def train_dataloader(self):
        return DataLoader(
            self.train_ds,
            batch_size=self.cfg.hyperparameters.batch_size,
            shuffle=self.cfg.train_dataset.data_loader.shuffle,
            num_workers=self.cfg.train_dataset.data_loader.num_workers,
            pin_memory=self.cfg.train_dataset.data_loader.pin_memory
            if "pin_memory" in self.cfg.train_dataset.data_loader
            else True,
            drop_last=self.cfg.train_dataset.data_loader.drop_last
            if "drop_last" in self.cfg.train_dataset.data_loader
            else True,
            prefetch_factor=self.cfg.train_dataset.data_loader.prefetch_factor
            if "prefetch_factor" in self.cfg.train_dataset.data_loader
            else 2,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_ds,
            batch_size=self.cfg.hyperparameters.batch_size,
            shuffle=self.cfg.val_dataset.data_loader.shuffle
            if "shuffle" in self.cfg.val_dataset.data_loader
            else False,
            num_workers=self.cfg.val_dataset.data_loader.num_workers,
            pin_memory=self.cfg.val_dataset.data_loader.pin_memory
            if "pin_memory" in self.cfg.val_dataset.data_loader
            else True,
            drop_last=self.cfg.val_dataset.data_loader.drop_last
            if "drop_last" in self.cfg.val_dataset.data_loader
            else True,
            prefetch_factor=self.cfg.val_dataset.data_loader.prefetch_factor
            if "prefetch_factor" in self.cfg.val_dataset.data_loader
            else 2,
        )

    def training_step(self, batch, batch_idx):
        images, masks = batch.values()
        masks = masks.long()
        predicted_masks = self(images)
        loss = self.loss_function(predicted_masks, masks)
        
        # Log loss with forward slash for grouping
        self.log("loss/train", loss, on_step=True, on_epoch=True, prog_bar=True)
        
        # Compute and log metrics - automatically prefixed with train/
        if hasattr(self, 'train_metrics'):
            metrics = self.train_metrics(predicted_masks, masks)
            self.log_dict(metrics, on_step=True, on_epoch=True, prog_bar=False)
        
        return loss

    def validation_step(self, batch, batch_idx):
        images, masks = batch.values()
        masks = masks.long()
        predicted_masks = self(images)
        loss = self.loss_function(predicted_masks, masks)
        
        # Log loss with forward slash for grouping
        self.log("loss/val", loss, on_step=False, on_epoch=True, prog_bar=True)
        
        # Compute and log metrics - automatically prefixed with val/
        if hasattr(self, 'val_metrics'):
            metrics = self.val_metrics(predicted_masks, masks)
            self.log_dict(metrics, on_step=False, on_epoch=True, prog_bar=False)
        
        return loss

    # Removed training_epoch_end and validation_epoch_end
    # Lightning 2.0+ automatically aggregates metrics
