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
from typing import List, Optional, Union
from omegaconf import DictConfig, OmegaConf
from hydra.utils import instantiate

from pytorch_segmentation_models_trainer.custom_losses.base_loss import MultiLoss

logger = logging.getLogger(__name__)


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
                _target_: pytorch_segmentation_models_trainer.custom_losses.base_loss.CrossfieldAlignLoss
                name: crossfield_align
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
        loss_funcs.append(loss_func)
        
        # Extract weight (can be from loss_weight_cfg.weight or loss_cfg.weight)
        weight = loss_weight_cfg.get('weight', loss_cfg.get('weight', 1.0))
        weights.append(weight)
        
        # Log the configuration
        weight_str = f"{weight}" if isinstance(weight, (int, float)) else f"dynamic{weight}"
        logger.info(f"  [{idx}] {loss_func.name} (weight={weight_str})")
    
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
    elif 'multi_loss' in cfg.loss_params:
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
        
        if 'name' not in loss_cfg:
            raise ValueError(f"Loss at index {idx} missing 'name' field")
        
        # Check that name is unique
        names = [lw.loss.name for lw in compound_loss_cfg.losses]
        if len(names) != len(set(names)):
            duplicates = [name for name in names if names.count(name) > 1]
            raise ValueError(f"Duplicate loss names found: {duplicates}")
    
    logger.info("Loss configuration validation passed")
    return True