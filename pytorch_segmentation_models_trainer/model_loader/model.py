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
from pytorch_segmentation_models_trainer.dataset_loader.dataset import _worker_init_fn

logger = logging.getLogger(__name__)


class Model(pl.LightningModule):
    """Base Model class compatible with PyTorch Lightning 2.0+"""

    def __init__(self, cfg, inference_mode=False):
        super(Model, self).__init__()
        self.cfg = cfg
        self.model = self.get_model()
        if inference_mode:
            return
        self.train_ds = instantiate(self.cfg.train_dataset, _recursive_=False) if "train_dataset" in self.cfg else None
        self.val_ds = instantiate(self.cfg.val_dataset, _recursive_=False) if "val_dataset" in self.cfg else None
        self.loss_function = self.get_loss_function()

        # Dual-head: connect loss to model so it can access stored logits
        self._is_dual_head = getattr(self.loss_function, 'is_dual_head_loss', False)
        if self._is_dual_head:
            self.loss_function.set_model(self.model)

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
        
        # Per-class IoU: automatically enabled when class_definitions is present
        self._per_class_iou = None
        self._class_names = None
        if "class_definitions" in self.cfg:
            n_cls = len(self.cfg.class_definitions.names)
            self._per_class_iou = torchmetrics.JaccardIndex(
                task="multiclass", num_classes=n_cls,
                average="none", ignore_index=255,
            )
            self._class_names = list(self.cfg.class_definitions.names)
            logger.info(f"Per-class IoU monitoring enabled for {n_cls} classes")

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
        
        elif hasattr(devices, '__len__') and not isinstance(devices, str):
            # List of device IDs (list, tuple, or OmegaConf ListConfig)
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
                logger.warning("'train_dataset' not found in config")
                return None
            
            dataset_cfg = self.cfg.train_dataset
            
            # Get CSV path
            if 'input_csv_path' not in dataset_cfg:
                logger.warning("'input_csv_path' not found in train_dataset config")
                return None
            
            csv_path = dataset_cfg.input_csv_path

            import pandas as pd
            import os

            if not os.path.exists(csv_path):
                logger.warning("CSV file not found: %s", csv_path)
                return None

            # Use samples_per_epoch if defined (RandomCropSegmentationDataset),
            # otherwise fall back to CSV row count.
            # Grid mode: dataset size is determined by pre-computed grid positions.
            # When samples_per_epoch <= 0 (auto mode), read the computed
            # value from the instantiated dataset instead of the config.
            if dataset_cfg.get('grid_mode', False):
                if self.train_ds is not None:
                    dataset_size = len(self.train_ds)
                else:
                    logger.warning(
                        "Grid mode active but dataset not yet instantiated; "
                        "cannot compute steps_per_epoch"
                    )
                    return None
            elif 'samples_per_epoch' in dataset_cfg:
                dataset_size = dataset_cfg.samples_per_epoch
                if dataset_size <= 0 and self.train_ds is not None and hasattr(self.train_ds, 'samples_per_epoch'):
                    dataset_size = self.train_ds.samples_per_epoch
            else:
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
                logger.warning("Could not find batch_size in config")
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
            
            # Log summary
            logger.info(
                "AUTO-COMPUTED STEPS_PER_EPOCH: csv=%s, dataset_size=%d, "
                "batch_size=%d, devices=%d, accumulate=%d, effective_batch=%d, "
                "steps_per_epoch=%d",
                csv_path, dataset_size, batch_size, device_count,
                accumulate_grad_batches, effective_batch_size, steps_per_epoch,
            )
            
            return steps_per_epoch
            
        except Exception as e:
            logger.warning("Error computing steps_per_epoch from config: %s", e, exc_info=True)
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
        layer_decay = None
        if hasattr(self.cfg, 'hyperparameters'):
            layer_decay = getattr(self.cfg.hyperparameters, 'layer_decay', None)

        if layer_decay is not None and layer_decay < 1.0:
            param_groups = self._get_param_groups_with_layer_decay(layer_decay)
            # Cannot use Hydra instantiate with custom param_groups:
            # it converts Python dicts to OmegaConf DictConfig, losing tensor refs.
            from hydra.utils import get_class
            from omegaconf import OmegaConf
            optimizer_cls = get_class(self.cfg.optimizer._target_)
            opt_kwargs = OmegaConf.to_container(self.cfg.optimizer, resolve=True)
            opt_kwargs.pop('_target_')
            # lr and weight_decay already set per-group; keep remaining (e.g. betas)
            opt_kwargs.pop('lr', None)
            opt_kwargs.pop('weight_decay', None)
            return optimizer_cls(param_groups, **opt_kwargs)
        return instantiate(
            self.cfg.optimizer, params=self.parameters(), _recursive_=False
        )

    def _get_param_groups_with_layer_decay(self, layer_decay):
        """Create parameter groups with stage-wise LR decay for timm encoders.

        Follows MMSegmentation's LearningRateDecayOptimizerConstructor with
        stage_wise decay for ConvNeXt models.  Also applies no weight-decay
        to bias and normalization parameters (1-D tensors).

        Decay is applied by distance from output:
          decoder/head  : lr
          encoder stage N-1 : lr * decay
          encoder stage N-2 : lr * decay^2
          ...
          encoder stem  : lr * decay^(N+1)
        """
        import re

        base_lr = self.cfg.optimizer.lr
        weight_decay = getattr(self.cfg.optimizer, 'weight_decay', 0.05)

        # Detect number of encoder stages from parameter names.
        # SMP TimmUniversalEncoder: encoder.model.stages.X.blocks...
        # Also try downsample_layers.X for stem-based counting.
        max_stage = -1
        encoder_names_sample = []
        for name, _ in self.named_parameters():
            if 'encoder' in name:
                if len(encoder_names_sample) < 5:
                    encoder_names_sample.append(name)
                match = re.search(r'stages[\.\[](\d+)', name)
                if match:
                    max_stage = max(max_stage, int(match.group(1)))
        num_stages = max_stage + 1 if max_stage >= 0 else 0
        if num_stages == 0 and encoder_names_sample:
            logger.warning(
                f"LLRD: 0 stages detected. Sample encoder param names: "
                f"{encoder_names_sample}"
            )

        param_groups = []
        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue

            # Determine LR based on layer position
            if 'encoder' not in name:
                # Decoder / segmentation head: full LR
                group_lr = base_lr
            else:
                match = re.search(r'stages\.(\d+)', name)
                if match:
                    stage_id = int(match.group(1))
                    depth = num_stages - stage_id
                    group_lr = base_lr * (layer_decay ** depth)
                else:
                    # Stem or other encoder components: deepest decay
                    group_lr = base_lr * (layer_decay ** (num_stages + 1))

            # No weight decay for bias and normalization layers (1-D params)
            param_wd = 0.0 if param.ndim <= 1 else weight_decay

            param_groups.append({
                'params': [param],
                'lr': group_lr,
                'weight_decay': param_wd,
            })

        lr_values = [g['lr'] for g in param_groups]
        logger.info(
            f"Layer-wise LR decay (rate={layer_decay}, {num_stages} stages): "
            f"{len(param_groups)} param groups, "
            f"LR range: [{min(lr_values):.2e}, {max(lr_values):.2e}]"
        )
        return param_groups

    def set_encoder_trainable(self, trainable=False):
        """Freezes or unfreezes the model encoder."""
        for child in self.model.encoder.children():
            for param in child.parameters():
                param.requires_grad = trainable
        logger.info("Encoder weights set to trainable=%s", trainable)
        return

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        """Configure optimizer and schedulers with automatic OneCycleLR setup
        and optional LR warmup via LinearLR + SequentialLR."""
        optimizer = self.get_optimizer()
        scheduler_list = []

        # Read warmup config
        warmup_epochs = 0
        if hasattr(self.cfg, 'hyperparameters'):
            warmup_epochs = getattr(self.cfg.hyperparameters, 'warmup_epochs', 0)

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
                        logger.info("OneCycleLR: Using steps_per_epoch = %d", steps_per_epoch)
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
                    logger.info("OneCycleLR: Using provided steps_per_epoch = %d", provided_steps)

                dict_item["scheduler"] = instantiate(
                    scheduler_config, optimizer=optimizer, _recursive_=False
                )
            else:
                # Other schedulers - instantiate normally
                base_scheduler = instantiate(
                    item.scheduler, optimizer=optimizer, _recursive_=False
                )

                # Wrap with LinearLR warmup if configured
                if warmup_epochs > 0 and not is_one_cycle:
                    from torch.optim.lr_scheduler import LinearLR, SequentialLR

                    # Adjust T_max for CosineAnnealingLR so it finishes at the right epoch
                    if hasattr(base_scheduler, 'T_max'):
                        base_scheduler.T_max = base_scheduler.T_max - warmup_epochs

                    warmup_scheduler = LinearLR(
                        optimizer,
                        start_factor=1e-2,   # start at 1% of base lr
                        end_factor=1.0,
                        total_iters=warmup_epochs,
                    )
                    combined = SequentialLR(
                        optimizer,
                        schedulers=[warmup_scheduler, base_scheduler],
                        milestones=[warmup_epochs],
                    )
                    dict_item["scheduler"] = combined
                    logger.info(
                        f"LR warmup: {warmup_epochs} epochs linear "
                        f"(1% -> 100% of base lr), then main scheduler"
                    )
                else:
                    dict_item["scheduler"] = base_scheduler

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
            persistent_workers=self.cfg.train_dataset.data_loader.persistent_workers
            if "persistent_workers" in self.cfg.train_dataset.data_loader
            else False,
            worker_init_fn=_worker_init_fn,
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
            persistent_workers=self.cfg.val_dataset.data_loader.persistent_workers
            if "persistent_workers" in self.cfg.val_dataset.data_loader
            else False,
            worker_init_fn=_worker_init_fn,
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

    @staticmethod
    def _soft_to_hard_masks(masks, ignore_value=255):
        """Convert soft label masks (B,C,H,W) to hard masks (B,H,W) for metrics.

        Pixels where all class channels are zero (ignore regions) are mapped
        to ``ignore_value`` instead of class 0 (which argmax would return).
        """
        valid_mask = masks.sum(dim=1) > 0  # (B, H, W)
        hard_masks = masks.argmax(dim=1)  # (B, H, W)
        hard_masks[~valid_mask] = ignore_value
        return hard_masks

    def _shared_step(self, batch, prefix):
        """Shared logic for training and validation steps.

        Args:
            batch: dict with 'image', 'mask', and optionally 'hard_mask' tensors.
            prefix: 'train' or 'val' (used for log keys).
        """
        is_train = prefix == "train"
        images = batch["image"]
        masks = batch["mask"]

        # Soft labels arrive as float (B,C,H,W); hard labels as int/long (B,H,W)
        if masks.is_floating_point():
            hard_masks = self._soft_to_hard_masks(masks)  # for metrics only
        else:
            masks = masks.long()
            hard_masks = masks

        # Dual-head support: store hard_mask from batch in model for loss access
        if self._is_dual_head:
            if "hard_mask" in batch:
                self.model.last_hard_mask = batch["hard_mask"].long()
            else:
                self.model.last_hard_mask = hard_masks

        predicted_masks = self(images)

        # Dual-head loss: pass epoch for consistency warmup
        if self._is_dual_head:
            loss = self.loss_function(predicted_masks, masks, epoch=self.current_epoch)
            individual_losses, extra_info = {}, {}
        # Compute loss — with OHEM if model supports it
        elif hasattr(self.model, 'compute_ohem_loss') and getattr(self.model, 'ohem_ratio', 0) > 0 and is_train:
            loss = self.model.compute_ohem_loss(self.loss_function, predicted_masks, masks)
            individual_losses, extra_info = {}, {}
        else:
            loss, individual_losses, extra_info = self._compute_loss(predicted_masks, masks)

        # Add MoE auxiliary loss if model exposes it
        if hasattr(self.model, 'last_aux_loss') and self.model.last_aux_loss is not None:
            moe_aux_loss = self.model.last_aux_loss
            loss = loss + moe_aux_loss
            self.log(f"losses/{prefix}_moe_aux", moe_aux_loss,
                     on_step=is_train, on_epoch=True, sync_dist=True)

        # MoE routing diagnostics
        if hasattr(self.model, 'get_moe_diagnostics'):
            should_log_moe = (not is_train) or (self.global_step % 50 == 0)
            if should_log_moe:
                diag = self.model.get_moe_diagnostics()
                for k, v in diag.items():
                    self.log(f"{k}/{prefix}", v,
                             on_step=is_train, on_epoch=True, sync_dist=False)
                # Expert-class affinity (validation only, more expensive)
                if not is_train and hasattr(self.model, 'get_expert_class_affinity'):
                    affinity = self.model.get_expert_class_affinity(hard_masks)
                    for k, v in affinity.items():
                        self.log(f"{k}/val", v,
                                 on_step=False, on_epoch=True, sync_dist=False)

        # MEDOE expert-specific loss
        if hasattr(self.model, 'compute_expert_loss') and self.model.last_expert_outputs is not None:
            expert_loss = self.model.compute_expert_loss(
                self.model.last_expert_outputs, hard_masks
            )
            loss = loss + expert_loss
            self.log(f"losses/{prefix}_expert", expert_loss,
                     on_step=is_train, on_epoch=True, sync_dist=True)

        # MEDOE diagnostics
        if hasattr(self.model, 'get_medoe_diagnostics'):
            should_log = (not is_train) or (self.global_step % 50 == 0)
            if should_log:
                medoe_diag = self.model.get_medoe_diagnostics(
                    hard_masks if not is_train else None
                )
                for k, v in medoe_diag.items():
                    self.log(f"{k}/{prefix}", v,
                             on_step=is_train, on_epoch=True, sync_dist=False)
            # Free stored tensors after use
            self.model.last_expert_outputs = None
            self.model.last_gate_weights = None
            if hasattr(self.model, '_gate_weights_with_grad'):
                self.model._gate_weights_with_grad = None

        # Dual-head Kendall diagnostics
        if self._is_dual_head:
            should_log_dh = (not is_train) or (self.global_step % 50 == 0)
            if should_log_dh:
                self.log(f"kendall/{prefix}_sigma_hard",
                         self.loss_function.log_sigma_hard.exp(),
                         on_step=is_train, on_epoch=True, sync_dist=False)
                self.log(f"kendall/{prefix}_sigma_soft",
                         self.loss_function.log_sigma_soft.exp(),
                         on_step=is_train, on_epoch=True, sync_dist=False)
                self.log(f"kendall/{prefix}_sigma_consist",
                         self.loss_function.log_sigma_consist.exp(),
                         on_step=is_train, on_epoch=True, sync_dist=False)
            # Clean up stored tensors
            if hasattr(self.model, 'last_logits_A'):
                self.model.last_logits_A = None
                self.model.last_logits_B = None

        # Log total loss
        self.log(f"loss/{prefix}", loss,
                 on_step=is_train, on_epoch=True, prog_bar=True, sync_dist=True)

        # Log individual losses if using compound loss
        if individual_losses:
            for loss_name, loss_value in individual_losses.items():
                self.log(f"losses/{prefix}_{loss_name}", loss_value,
                         on_step=is_train, on_epoch=True, sync_dist=False)

        # Log extra info if available
        if extra_info:
            for loss_name, extra_dict in extra_info.items():
                for key, value in extra_dict.items():
                    self.log(f"extra/{prefix}_{loss_name}_{key}", value,
                             on_step=is_train, on_epoch=True, sync_dist=False)

        # Compute and log metrics
        metrics_attr = f'{prefix}_metrics'
        if hasattr(self, metrics_attr):
            metrics = getattr(self, metrics_attr)(predicted_masks, hard_masks)
            self.log_dict(metrics, on_step=is_train, on_epoch=True,
                          prog_bar=False, sync_dist=True)

        # Per-class IoU (validation only)
        if not is_train and self._per_class_iou is not None:
            per_class = self._per_class_iou(predicted_masks, hard_masks)
            for i, name in enumerate(self._class_names):
                self.log(f"val_iou/{name}", per_class[i],
                         on_step=False, on_epoch=True, sync_dist=True)

        return loss

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self._shared_step(batch, "val")
    
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
    