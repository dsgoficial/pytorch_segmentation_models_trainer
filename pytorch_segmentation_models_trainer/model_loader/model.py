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
        self.steps_per_epoch = None
    
    def setup(self, stage=None):
        """Extract dataset info when dataloaders are ready"""
        if stage == 'fit' or stage is None:
            self._compute_steps_per_epoch_from_trainer()
    
    def _compute_steps_per_epoch_from_trainer(self):
        """Compute steps_per_epoch from trainer's dataloader"""
        try:
            train_dataloader = None
            
            if hasattr(self.trainer, 'train_dataloader') and self.trainer.train_dataloader is not None:
                train_dataloader = self.trainer.train_dataloader
            elif hasattr(self.trainer, 'datamodule') and self.trainer.datamodule is not None:
                if hasattr(self.trainer.datamodule, 'train_dataloader'):
                    train_dataloader = self.trainer.datamodule.train_dataloader()
            
            if train_dataloader is None:
                print("⚠️  Warning: Could not access training dataloader yet")
                return
            
            if hasattr(train_dataloader, 'dataset'):
                dataset_size = len(train_dataloader.dataset)
            elif hasattr(train_dataloader, '__len__'):
                dataset_size = len(train_dataloader) * getattr(train_dataloader, 'batch_size', 1)
            else:
                print("⚠️  Warning: Cannot determine dataset size")
                return
            
            batch_size = getattr(train_dataloader, 'batch_size', 1)
            accumulate_grad_batches = getattr(self.trainer, 'accumulate_grad_batches', 1)
            
            self.steps_per_epoch = dataset_size // (batch_size * accumulate_grad_batches)
            
            print(f"\n{'='*60}")
            print(f"✅ AUTOMATIC TRAINING CONFIGURATION")
            print(f"{'='*60}")
            print(f"Dataset size:           {dataset_size:>10,}")
            print(f"Batch size:             {batch_size:>10}")
            print(f"Gradient accumulation:  {accumulate_grad_batches:>10}")
            print(f"Steps per epoch:        {self.steps_per_epoch:>10,}")
            print(f"{'='*60}\n")
            
        except Exception as e:
            print(f"⚠️  Could not compute steps_per_epoch: {e}")
            self.steps_per_epoch = None

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
        
        if self.steps_per_epoch is None:
            self._compute_steps_per_epoch_from_trainer()
        
        for item in self.cfg.scheduler_list:
            dict_item = dict(item)
            
            scheduler_target = item.scheduler.get('_target_', '')
            is_one_cycle = 'OneCycleLR' in scheduler_target
            
            if not ('OneCycleLR' in scheduler_target):
                dict_item["scheduler"] = instantiate(
                    item.scheduler, optimizer=optimizer, _recursive_=False
                )
                scheduler_list.append(dict_item)
                continue
            scheduler_config = dict(item.scheduler)
            
            needs_auto_steps = (
                'steps_per_epoch' not in scheduler_config or 
                scheduler_config.get('steps_per_epoch') in [None, 'auto', -1]
            )
            
            if needs_auto_steps:
                if self.steps_per_epoch is not None:
                    scheduler_config['steps_per_epoch'] = self.steps_per_epoch
                    print(f"✅ OneCycleLR: steps_per_epoch = {self.steps_per_epoch:,}")
                else:
                    raise ValueError("Cannot determine steps_per_epoch for OneCycleLR!")
            else:
                provided_steps = scheduler_config['steps_per_epoch']
                print(f"ℹ️  OneCycleLR: Using provided steps_per_epoch = {provided_steps:,}")
            
            dict_item["scheduler"] = instantiate(
                scheduler_config, optimizer=optimizer, _recursive_=False
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
        self.log("loss/train", loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        
        # Compute and log metrics - automatically prefixed with train/
        if hasattr(self, 'train_metrics'):
            metrics = self.train_metrics(predicted_masks, masks)
            self.log_dict(metrics, on_step=True, on_epoch=True, prog_bar=False, sync_dist=True)
        
        return loss

    def validation_step(self, batch, batch_idx):
        images, masks = batch.values()
        masks = masks.long()
        predicted_masks = self(images)
        loss = self.loss_function(predicted_masks, masks)
        
        # Log loss with forward slash for grouping
        self.log("loss/val", loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        
        # Compute and log metrics - automatically prefixed with val/
        if hasattr(self, 'val_metrics'):
            metrics = self.val_metrics(predicted_masks, masks)
            self.log_dict(metrics, on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)
        
        return loss

    # Removed training_epoch_end and validation_epoch_end
    # Lightning 2.0+ automatically aggregates metrics
