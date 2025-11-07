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
import logging
import albumentations as A
import pytorch_lightning as pl
import torchmetrics
import torch
import torch.nn as nn
from hydra.utils import instantiate

from torch.utils.data import DataLoader

from typing import List, Any, Union, Dict, Tuple
from pytorch_segmentation_models_trainer.utils.model_utils import replace_activation
from pytorch_segmentation_models_trainer.custom_losses.base_loss import MultiLoss

logger = logging.getLogger(__name__)


class Model(pl.LightningModule):
    """Base Model class compatible with PyTorch Lightning 2.0+"""

    def __init__(self, cfg, inference_mode=False):
        super(Model, self).__init__()
        self.cfg = cfg
        self.model = self.get_model()
        self.train_ds = instantiate(self.cfg.train_dataset, _recursive_=False) if "train_dataset" in self.cfg else None
        self.val_ds = instantiate(self.cfg.val_dataset, _recursive_=False) if "val_dataset" in self.cfg else None
        if inference_mode:
            return
        self.loss_function = self.get_loss_function()
        
        # NEW: Determine if using compound loss (MultiLoss)
        self.use_compound_loss = isinstance(self.loss_function, MultiLoss)
        
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
        
        # NEW: Log loss configuration
        logger.info(f"Initialized Model with loss function: {self.loss_function}")
        if self.use_compound_loss:
            logger.info(f"Using compound loss with {len(self.loss_function.loss_funcs)} components")
        self.check_if_should_normalize()
    
    def setup(self, stage=None):
        """Extract dataset info when dataloaders are ready"""
        if stage == 'fit' or stage is None:
            self._compute_steps_from_config()
    
    def _compute_device_count(self):
        """
        Compute the number of devices that will be used for training.
        Handles various PyTorch Lightning device specifications.
        """
        device_count = 1  # Default to single device
        accelerator = "auto"  # Default accelerator
        
        # Get accelerator and devices from config
        if 'hyperparameters' in self.cfg:
            accelerator = self.cfg.hyperparameters.get('accelerator', 'auto')
            devices = self.cfg.hyperparameters.get('devices', 'auto')
        elif 'pl_trainer' in self.cfg:
            accelerator = self.cfg.pl_trainer.get('accelerator', 'auto')
            devices = self.cfg.pl_trainer.get('devices', 'auto')
        else:
            devices = 'auto'
        
        # Handle different device specifications
        if devices in ['auto', -1, '-1']:
            # Use all available devices for the accelerator
            if accelerator in ['gpu', 'cuda', 'auto']:
                device_count = torch.cuda.device_count() if torch.cuda.is_available() else 1
            else:
                device_count = 1  # CPU, MPS, etc.
        
        elif isinstance(devices, int):
            # Specific number of devices
            device_count = max(1, devices)
        
        elif isinstance(devices, (list, tuple)):
            # List of device IDs
            device_count = len(devices)
        
        elif isinstance(devices, str):
            # Handle string specifications like "0,1" or "2"
            if ',' in devices:
                device_count = len(devices.split(','))
            else:
                try:
                    device_count = int(devices)
                except ValueError:
                    # Fallback for unexpected string formats
                    device_count = 1
        
        return max(1, device_count)  # Ensure at least 1

    def _compute_steps_from_config(self):
        """
        Compute steps_per_epoch by reading the CSV file from config.
        This is called during configure_optimizers before trainer is available.
        """
        try:
            # Get train_dataset config
            if 'train_dataset' not in self.cfg:
                print("⚠️  'train_dataset' not found in config")
                return None
            
            dataset_cfg = self.cfg.train_dataset
            
            # Get CSV path
            if 'input_csv_path' not in dataset_cfg:
                print("⚠️  'input_csv_path' not found in train_dataset config")
                return None
            
            csv_path = dataset_cfg.input_csv_path
            
            # Read CSV to get dataset size
            import pandas as pd
            import os
            
            if not os.path.exists(csv_path):
                print(f"⚠️  CSV file not found: {csv_path}")
                return None
            
            df = pd.read_csv(csv_path)
            dataset_size = len(df)
            
            # Get batch size from config (check multiple locations)
            batch_size = None
            
            # Try data_loader.batch_size first
            if 'data_loader' in dataset_cfg and 'batch_size' in dataset_cfg.data_loader:
                batch_size = dataset_cfg.data_loader.batch_size
            
            # Fallback to hyperparameters.batch_size
            if batch_size is None and 'hyperparameters' in self.cfg:
                batch_size = self.cfg.hyperparameters.get('batch_size')
            
            # Fallback to top-level batch_size
            if batch_size is None:
                batch_size = self.cfg.get('batch_size')
            
            if batch_size is None:
                print("⚠️  Could not find batch_size in config")
                return None
            
            # Compute device count using improved method
            device_count = self._compute_device_count()
            
            # Get gradient accumulation steps
            accumulate_grad_batches = 1
            
            # Check both hyperparameters and pl_trainer config
            if 'hyperparameters' in self.cfg:
                accumulate_grad_batches = self.cfg.hyperparameters.get(
                    'accumulate_grad_batches', 1
                )
            elif 'pl_trainer' in self.cfg:
                accumulate_grad_batches = self.cfg.pl_trainer.get(
                    'accumulate_grad_batches', 1
                )
            
            # Calculate steps per epoch
            effective_batch_size = batch_size * accumulate_grad_batches * device_count
            steps_per_epoch = dataset_size // effective_batch_size
            
            # Print summary
            print(f"\n{'='*60}")
            print(f"✅ AUTO-COMPUTED STEPS_PER_EPOCH FROM CONFIG")
            print(f"{'='*60}")
            print(f"CSV path:               {csv_path}")
            print(f"Dataset size:           {dataset_size:>10,} samples")
            print(f"Batch size:             {batch_size:>10}")
            print(f"Devices:                {device_count:>10}")
            print(f"Gradient accumulation:  {accumulate_grad_batches:>10}")
            print(f"Effective batch size:   {effective_batch_size:>10}")
            print(f"Steps per epoch:        {steps_per_epoch:>10,}")
            print(f"{'='*60}\n")
            
            return steps_per_epoch
            
        except Exception as e:
            print(f"⚠️  Error computing steps_per_epoch from config: {e}")
            import traceback
            traceback.print_exc()
            return None

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

    def get_loss_function(self) -> Union[nn.Module, MultiLoss]:
        """
        Get the loss function from configuration.
        
        Supports three modes:
        1. NEW: Compound loss via loss_params.compound_loss (recommended)
        2. LEGACY: Multi loss via loss_params.multi_loss (backward compatible)
        3. SIMPLE: Direct loss specification via cfg.loss
        
        Returns:
            Loss function (can be MultiLoss or simple nn.Module)
        """
        # Check for compound loss configuration (NEW)
        if hasattr(self.cfg, 'loss_params') and hasattr(self.cfg.loss_params, 'compound_loss'):
            if self.cfg.loss_params.compound_loss is not None:
                logger.info("Building compound loss from loss_params.compound_loss")
                from pytorch_segmentation_models_trainer.custom_losses.loss_builder import (
                    build_compound_loss_from_config
                )
                return build_compound_loss_from_config(self.cfg.loss_params.compound_loss)
        
        # Check for legacy multi_loss configuration
        if hasattr(self.cfg, 'loss_params') and hasattr(self.cfg.loss_params, 'multi_loss'):
            logger.info("Building loss from legacy multi_loss configuration")
            from pytorch_segmentation_models_trainer.custom_losses.loss_builder import (
                build_loss_from_config
            )
            return build_loss_from_config(self.cfg)
        
        # Fall back to simple loss specification
        if "loss" in self.cfg:
            logger.info("Building simple loss from cfg.loss")
            return instantiate(self.cfg.loss, _recursive_=False)
        
        # If nothing is specified, raise an error
        raise ValueError(
            "No loss configuration found. Please specify one of:\n"
            "  - cfg.loss_params.compound_loss (recommended)\n"
            "  - cfg.loss_params.multi_loss (legacy)\n"
            "  - cfg.loss (simple)"
        )

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
        """Configure optimizer and schedulers with automatic OneCycleLR setup"""
        optimizer = self.get_optimizer()
        scheduler_list = []
        
        if "scheduler_list" not in self.cfg:
            return [optimizer], scheduler_list
        
        for item in self.cfg.scheduler_list:
            dict_item = dict(item)
            
            # Check if this is OneCycleLR scheduler
            scheduler_target = item.scheduler.get('_target_', '')
            is_one_cycle = 'OneCycleLR' in scheduler_target
            
            if is_one_cycle:
                scheduler_config = dict(item.scheduler)
                
                # Check if steps_per_epoch needs to be set automatically
                needs_auto_steps = (
                    'steps_per_epoch' not in scheduler_config or 
                    scheduler_config.get('steps_per_epoch') in [None, 'auto', -1]
                )
                
                if needs_auto_steps:
                    # Compute steps_per_epoch from config
                    steps_per_epoch = self._compute_steps_from_config()
                    
                    if steps_per_epoch is not None:
                        scheduler_config['steps_per_epoch'] = steps_per_epoch
                        print(f"✅ OneCycleLR: Using steps_per_epoch = {steps_per_epoch:,}")
                    else:
                        raise ValueError(
                            "\n" + "="*60 + "\n"
                            "ERROR: Cannot determine steps_per_epoch for OneCycleLR!\n"
                            "="*60 + "\n"
                            "Could not automatically detect dataset size.\n"
                            "Please manually specify in your config:\n\n"
                            "scheduler_list:\n"
                            "  - scheduler:\n"
                            "      _target_: torch.optim.lr_scheduler.OneCycleLR\n"
                            "      steps_per_epoch: 4092  # dataset_size / batch_size\n"
                            "      max_lr: 0.001\n"
                            "      epochs: 200\n"
                            "\n"
                            "To calculate: dataset_size / batch_size\n"
                            "Example: 98201 / 24 = 4092\n"
                            "="*60
                        )
                else:
                    provided_steps = scheduler_config['steps_per_epoch']
                    print(f"ℹ️  OneCycleLR: Using provided steps_per_epoch = {provided_steps:,}")
                
                dict_item["scheduler"] = instantiate(
                    scheduler_config, optimizer=optimizer, _recursive_=False
                )
            else:
                # Other schedulers - instantiate normally
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

    def _compute_loss(
        self, 
        predicted_masks: torch.Tensor, 
        masks: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], Dict[str, Any]]:
        """
        Compute loss handling both simple and compound loss functions.
        
        Args:
            predicted_masks: Model predictions
            masks: Ground truth masks
            
        Returns:
            Tuple of (total_loss, individual_losses_dict, extra_info_dict)
            For simple losses, individual_losses_dict will be empty
        """
        if self.use_compound_loss:
            # Check config for normalization flag
            should_normalize = True
            if hasattr(self.cfg, 'loss_params'):
                if hasattr(self.cfg.loss_params, 'compound_loss'):
                    should_normalize = self.cfg.loss_params.compound_loss.get(
                        'normalize_losses', 
                        True  # Default to True
                    )
            
            # Compound loss returns (total_loss, individual_losses, extra_info)
            # Need to handle different batch formats
            if isinstance(masks, dict):
                # FrameField-style batch with gt_polygons_image
                pred_batch = {"seg": predicted_masks}
                gt_batch = masks
            else:
                # Simple segmentation batch
                pred_batch = predicted_masks
                gt_batch = masks
            
            return self.loss_function(
                pred_batch, 
                gt_batch, 
                normalize=should_normalize,  # Use config flag
                epoch=self.current_epoch
            )
        else:
            # Simple loss just returns scalar
            loss = self.loss_function(predicted_masks, masks)
            return loss, {}, {}

    def training_step(self, batch, batch_idx):
        """Training step - now supports both simple and compound losses."""
        images, masks = batch.values()
        masks = masks.long()
        predicted_masks = self(images)
        
        # Compute loss (handles both simple and compound)
        loss, individual_losses, extra_info = self._compute_loss(predicted_masks, masks)
        
        # Log total loss
        self.log("loss/train", loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        
        # NEW: Log individual losses if using compound loss
        if individual_losses:
            for loss_name, loss_value in individual_losses.items():
                self.log(
                    f"losses/train_{loss_name}", 
                    loss_value, 
                    on_step=True, 
                    on_epoch=True,
                    sync_dist=False
                )
        
        # NEW: Log extra info if available
        if extra_info:
            for loss_name, extra_dict in extra_info.items():
                for key, value in extra_dict.items():
                    self.log(
                        f"extra/train_{loss_name}_{key}",
                        value,
                        on_step=True,
                        on_epoch=True,
                        sync_dist=False
                    )
        
        # Compute and log metrics - automatically prefixed with train/
        if hasattr(self, 'train_metrics'):
            metrics = self.train_metrics(predicted_masks, masks)
            self.log_dict(metrics, on_step=True, on_epoch=True, prog_bar=False, sync_dist=True)
        
        return loss

    def validation_step(self, batch, batch_idx):
        """Validation step - now supports both simple and compound losses."""
        images, masks = batch.values()
        masks = masks.long()
        predicted_masks = self(images)
        
        # Compute loss (handles both simple and compound)
        loss, individual_losses, extra_info = self._compute_loss(predicted_masks, masks)
        
        # Log total loss
        self.log("loss/val", loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        
        # NEW: Log individual losses if using compound loss
        if individual_losses:
            for loss_name, loss_value in individual_losses.items():
                self.log(
                    f"losses/val_{loss_name}",
                    loss_value,
                    on_step=False,
                    on_epoch=True,
                    sync_dist=False
                )
        
        # NEW: Log extra info if available
        if extra_info:
            for loss_name, extra_dict in extra_info.items():
                for key, value in extra_dict.items():
                    self.log(
                        f"extra/val_{loss_name}_{key}",
                        value,
                        on_step=False,
                        on_epoch=True,
                        sync_dist=False
                    )
        
        # Compute and log metrics - automatically prefixed with val/
        if hasattr(self, 'val_metrics'):
            metrics = self.val_metrics(predicted_masks, masks)
            self.log_dict(metrics, on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)
        
        return loss
    
    def check_if_should_normalize(self):
        self.should_normalize = False
        if hasattr(self.cfg, 'loss_params') and hasattr(self.cfg.loss_params, 'compound_loss'):
            self.should_normalize = self.cfg.loss_params.compound_loss.get(
                'normalize_losses', 
                True  # Default to True
            )
        return self.should_normalize

    # Removed training_epoch_end and validation_epoch_end
    # Lightning 2.0+ automatically aggregates metrics
    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        return self(batch) 
    