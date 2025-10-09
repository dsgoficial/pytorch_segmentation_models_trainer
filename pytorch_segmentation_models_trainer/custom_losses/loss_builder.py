# -*- coding: utf-8 -*-
"""
/***************************************************************************
 pytorch_segmentation_models_trainer
                              -------------------
        begin                : 2025-10-06
        git sha              : $Format:%H$
        copyright            : (C) 2025 by Philipe Borba - Cartographic Engineer
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
import torch
import torch.nn as nn
from typing import List, Optional, Union, Any, Dict, Tuple
from omegaconf import DictConfig, OmegaConf
from hydra.utils import instantiate

from pytorch_segmentation_models_trainer.custom_losses.base_loss import MultiLoss, Loss

logger = logging.getLogger(__name__)


class LossWrapper(nn.Module):
    """
    Wrapper to make any loss function compatible with MultiLoss.
    
    This wrapper ensures that any loss (PyTorch standard, third-party, etc.)
    can work with the compound loss system by providing a consistent interface.
    
    Inherits from nn.Module so it can be added to nn.ModuleList in MultiLoss.
    """
    def __init__(self, loss_func: nn.Module, name: str = None):
        """
        Args:
            loss_func: The loss function to wrap
            name: Optional name for the loss (auto-generated if not provided)
        """
        super(LossWrapper, self).__init__()
        
        self.is_custom_loss = isinstance(loss_func, Loss)
        self.loss_func = loss_func
        self.set_norm_updated(False)
        
        # Set name
        if name:
            self.name = name
        elif hasattr(loss_func, 'name'):
            self.name = loss_func.name
        else:
            self.name = loss_func.__class__.__name__
        
        # For compatibility with MultiLoss
        if not self.is_custom_loss:
            # Create dummy norm for non-custom losses
            self.norm = nn.Parameter(
                torch.Tensor([1.0], device='cpu'),
                requires_grad=False,
            )
    
    def reset_norm(self):
        """Reset normalization (only for custom losses)"""
        if self.is_custom_loss:
            self.loss_func.reset_norm()
    
    def set_norm_updated(self, updated):
        self.norm_updated = updated
    
    def update_norm(self, pred_batch, gt_batch, nums):
        """Update normalization (only for custom losses)"""
        if self.is_custom_loss:
            self.loss_func.update_norm(pred_batch, gt_batch, nums)
    
    def sync(self, world_size):
        """Sync across GPUs (only for custom losses)"""
        if self.is_custom_loss:
            self.loss_func.sync(world_size)
    
    def forward(self, pred_batch, gt_batch, normalize=True) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Call the loss function with appropriate arguments.
        
        Args:
            pred_batch: Predictions
            gt_batch: Ground truth
            normalize: Whether to normalize (only used for custom losses)
            
        Returns:
            Tuple of (loss_value, extra_info_dict)
        """
        if self.is_custom_loss:
            # Custom loss - supports normalize parameter
            return self.loss_func(pred_batch, gt_batch, normalize=normalize)
        else:
            # Standard PyTorch or third-party loss - doesn't support normalize
            try:
                # Try calling with both arguments
                loss_value = self.loss_func(pred_batch, gt_batch)
            except TypeError as e:
                # If that fails, try with just prediction (some losses work this way)
                try:
                    loss_value = self.loss_func(pred_batch)
                except Exception as e2:
                    logger.error(f"Error calling loss {self.name}: {e}")
                    logger.error(f"Secondary error: {e2}")
                    raise
            
            # Return in the format expected by MultiLoss
            return loss_value, {}
    
    def __call__(self, pred_batch, gt_batch, normalize=True) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Alias for forward() to maintain compatibility"""
        return self.forward(pred_batch, gt_batch, normalize=normalize)
    
    def __repr__(self):
        return f"LossWrapper({self.name}, custom={self.is_custom_loss})"


def build_compound_loss_from_config(
    compound_loss_cfg: DictConfig,
    pre_processes: Optional[List] = None
) -> MultiLoss:
    """
    Build a MultiLoss object from a CompoundLossConfig.
    
    This function instantiates all individual losses and their weights,
    then creates a MultiLoss that combines them.
    
    Args:
        compound_loss_cfg: Configuration containing losses and their weights
        pre_processes: Optional list of pre-processing functions
        
    Returns:
        MultiLoss object ready for training
        
    Example configuration:
        compound_loss:
          epoch_thresholds: [0, 5, 10]
          losses:
            - loss:
                _target_: pytorch_segmentation_models_trainer.custom_losses.base_loss.SegLoss
                name: seg
                gt_channel_selector: 0
                bce_coef: 0.5
                dice_coef: 0.5
              weight: 10.0
            
            - loss:
                _target_: segmentation_models_pytorch.losses.DiceLoss
                mode: binary
              weight: 1.0
    """
    if not compound_loss_cfg or 'losses' not in compound_loss_cfg:
        raise ValueError("compound_loss_cfg must contain 'losses' field")
    
    loss_funcs = []
    weights = []
    
    logger.info("Building compound loss with the following components:")
    
    # Instantiate each loss and extract its weight
    for idx, loss_weight_cfg in enumerate(compound_loss_cfg.losses):
        if 'loss' not in loss_weight_cfg:
            raise ValueError(f"Loss configuration at index {idx} missing 'loss' field")
        
        # Instantiate the loss function
        loss_cfg = loss_weight_cfg.loss
        loss_func = instantiate(loss_cfg, _recursive_=False)
        
        # Get the name for logging
        loss_name = loss_cfg.get('name', None)
        
        # Wrap non-custom losses to make them compatible
        if not isinstance(loss_func, Loss):
            if loss_name is None:
                loss_name = loss_func.__class__.__name__
            logger.info(f"  Wrapping non-custom loss: {loss_name}")
            loss_func = LossWrapper(loss_func, name=loss_name)
        
        loss_funcs.append(loss_func)
        
        # Extract weight (can be from loss_weight_cfg.weight or loss_cfg.weight)
        weight = loss_weight_cfg.get('weight', loss_cfg.get('weight', 1.0))
        weights.append(weight)
        
        # Log the configuration
        weight_str = f"{weight}" if isinstance(weight, (int, float)) else f"dynamic{weight}"
        
        # Safely get the loss name for logging
        if hasattr(loss_func, 'name'):
            display_name = loss_func.name
        else:
            display_name = loss_func.__class__.__name__
        
        logger.info(f"  [{idx}] {display_name} (weight={weight_str})")
    
    # Get epoch thresholds and pre-processes
    epoch_thresholds = compound_loss_cfg.get('epoch_thresholds', None)
    if pre_processes is None:
        pre_processes = compound_loss_cfg.get('pre_processes', None)
    
    # Create the MultiLoss
    multi_loss = MultiLoss(
        loss_funcs=loss_funcs,
        weights=weights,
        epoch_thresholds=epoch_thresholds,
        pre_processes=pre_processes
    )
    
    logger.info(f"Compound loss created with {len(loss_funcs)} components")
    return multi_loss


def build_loss_from_config(cfg: DictConfig, pre_processes: Optional[List] = None) -> MultiLoss:
    """
    Build a loss function from configuration.
    
    This function supports both the new compound_loss configuration and
    the legacy loss configuration for backward compatibility.
    
    Args:
        cfg: Configuration object containing loss parameters
        pre_processes: Optional list of pre-processing functions
        
    Returns:
        MultiLoss object or raises error if configuration is invalid
    """
    # Check for new compound loss configuration
    if 'compound_loss' in cfg.loss_params and cfg.loss_params.compound_loss is not None:
        logger.info("Using new compound loss configuration")
        return build_compound_loss_from_config(
            cfg.loss_params.compound_loss,
            pre_processes=pre_processes
        )
    
    # Check for legacy configuration
    elif 'multi_loss' in cfg.loss_params or "multiloss" in cfg.loss_params:
        logger.info("Using legacy multi_loss configuration")
        # Import here to avoid circular dependency
        from pytorch_segmentation_models_trainer.custom_losses.base_loss import build_combined_loss
        return build_combined_loss(cfg, pre_processes=pre_processes)
    
    else:
        raise ValueError(
            "No valid loss configuration found. Please provide either "
            "'compound_loss' or 'multi_loss' in cfg.loss_params"
        )


def validate_loss_config(compound_loss_cfg: DictConfig) -> bool:
    """
    Validate a compound loss configuration.
    
    Args:
        compound_loss_cfg: Configuration to validate
        
    Returns:
        True if valid, raises ValueError otherwise
    """
    if not compound_loss_cfg:
        raise ValueError("compound_loss_cfg is None or empty")
    
    if 'losses' not in compound_loss_cfg:
        raise ValueError("compound_loss_cfg must contain 'losses' field")
    
    if not compound_loss_cfg.losses:
        raise ValueError("compound_loss_cfg.losses cannot be empty")
    
    # Validate each loss configuration
    for idx, loss_weight_cfg in enumerate(compound_loss_cfg.losses):
        if 'loss' not in loss_weight_cfg:
            raise ValueError(f"Loss at index {idx} missing 'loss' field")
        
        loss_cfg = loss_weight_cfg.loss
        
        if '_target_' not in loss_cfg:
            raise ValueError(f"Loss at index {idx} missing '_target_' field")
        
        # Check that name is unique (if provided)
        names = []
        for lw in compound_loss_cfg.losses:
            if 'name' in lw.loss:
                names.append(lw.loss.name)
        
        if len(names) != len(set(names)):
            duplicates = [name for name in names if names.count(name) > 1]
            raise ValueError(f"Duplicate loss names found: {set(duplicates)}")
    
    logger.info("Loss configuration validation passed")
    return True
