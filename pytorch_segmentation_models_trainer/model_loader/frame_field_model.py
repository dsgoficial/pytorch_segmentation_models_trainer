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
import logging
from omegaconf.dictconfig import DictConfig
from pytorch_segmentation_models_trainer.model_loader.model import Model
from pytorch_segmentation_models_trainer.custom_losses.loss_builder import build_loss_from_config
from pytorch_segmentation_models_trainer.custom_losses.base_loss import ComputeSegGrads

logger = logging.getLogger(__name__)


class FrameFieldSegmentationPLModel(Model):
    """
    PyTorch Lightning model for frame field segmentation.
    
    Now supports flexible compound loss configuration via YAML.
    """
    
    def __init__(self, cfg: DictConfig):
        """
        Initialize the model with configuration.
        
        Args:
            cfg: Hydra configuration containing model, loss, and training parameters
        """
        super(FrameFieldSegmentationPLModel, self).__init__(cfg)
        
        # Initialize the frame field model (same as before)
        self.model = self._build_model(cfg)
        
        # Build loss function with the NEW flexible system
        self.loss_function = self._build_loss_function(cfg)
        
        # Other initializations (metrics, transforms, etc.)
        self._setup_metrics(cfg)
        self._setup_transforms(cfg)
        
        logger.info(f"Initialized FrameFieldSegmentationPLModel with loss: {self.loss_function}")
    
    def _build_model(self, cfg: DictConfig):
        """Build the frame field model architecture."""
        # Your existing model building code
        from pytorch_segmentation_models_trainer.model_loader.frame_field_model import FrameFieldModel
        from hydra.utils import instantiate
        
        return FrameFieldModel(
            segmentation_model=instantiate(cfg.backbone, _recursive_=False),
            compute_seg=cfg.compute_seg,
            compute_crossfield=cfg.compute_crossfield,
            seg_params=cfg.seg_params,
            # ... other parameters
        )
    
    def _build_loss_function(self, cfg: DictConfig):
        """
        Build the loss function from configuration.
        
        This method now supports both:
        1. NEW: compound_loss configuration (flexible YAML-based)
        2. OLD: multi_loss configuration (backward compatible)
        
        Returns:
            MultiLoss object configured from YAML
        """
        # Prepare pre-processing functions if needed
        pre_processes = []
        
        # Add gradient computation if using coupling losses
        need_seg_grads = False
        
        if cfg.compute_seg:
            # Check if we need seg gradients for coupling losses
            if cfg.seg_params.compute_interior and cfg.compute_crossfield:
                need_seg_grads = True
            if cfg.seg_params.compute_edge and cfg.compute_crossfield:
                need_seg_grads = True
            if cfg.seg_params.compute_interior and cfg.seg_params.compute_edge:
                need_seg_grads = True
        
        if need_seg_grads:
            pre_processes.append(ComputeSegGrads(cfg.device))
            logger.info("Added ComputeSegGrads preprocessor for coupling losses")
        
        # Build loss using the new flexible system
        # This automatically detects whether you're using compound_loss or multi_loss
        try:
            loss_function = build_loss_from_config(
                cfg, 
                pre_processes=pre_processes if pre_processes else None
            )
            return loss_function
        except Exception as e:
            logger.error(f"Failed to build loss function: {e}")
            raise
    
    def _setup_metrics(self, cfg: DictConfig):
        """Setup metrics for training/validation."""
        # Your existing metrics setup
        pass
    
    def _setup_transforms(self, cfg: DictConfig):
        """Setup GPU transforms for training/validation."""
        # Your existing transform setup
        pass
    
    def on_train_start(self):
        """
        Called at the start of training.
        Compute normalization values for losses.
        """
        super().on_train_start()
        
        # Compute loss normalization (same as before)
        if hasattr(self, 'loss_function') and hasattr(self.loss_function, 'reset_norm'):
            logger.info("Computing loss normalization...")
            self._compute_loss_normalization()
    
    def _compute_loss_normalization(self):
        """Compute normalization values for the loss function."""
        from tqdm import tqdm
        import torch
        from pytorch_segmentation_models_trainer.utils import tensor_utils
        
        # Get dataloader
        dl = self.train_dataloader()
        
        # Number of batches to use for normalization
        normalization_params = self.cfg.loss_params.get('normalization_params', None)
        if normalization_params:
            max_batches = normalization_params.get('max_samples', 1000) // self.cfg.hyperparameters.batch_size
        else:
            max_batches = min(100, len(dl))
        
        # Reset loss norms
        self.loss_function.reset_norm()
        
        # Compute norms
        self.model.eval()
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(dl, total=max_batches, desc="Computing loss norms")):
                if batch_idx >= max_batches:
                    break
                
                # Move batch to device
                batch = tensor_utils.batch_to_cuda(batch) if self.cfg.device == "cuda" else batch
                
                # Forward pass
                pred = self.model(batch["image"])
                
                # Update loss norms
                self.loss_function.update_norm(pred, batch, batch["image"].shape[0])
        
        # Sync across GPUs if using distributed training
        world_size = self._get_world_size()
        if world_size > 1:
            self.loss_function.sync(world_size)
        
        self.model.train()
        logger.info("Loss normalization computed")
        logger.info(f"Loss function: {self.loss_function}")
    
    def _get_world_size(self):
        """Get the world size for distributed training."""
        if self.cfg.device == "cpu" or self.cfg.pl_trainer.devices == 0:
            return 1
        elif isinstance(self.cfg.pl_trainer.devices, list):
            return len(self.cfg.pl_trainer.devices)
        elif self.cfg.pl_trainer.devices == -1:
            import torch
            return torch.cuda.device_count()
        else:
            return self.cfg.pl_trainer.devices
    
    def training_step(self, batch, batch_idx):
        """
        Training step - unchanged from before.
        The loss_function handles everything internally.
        """
        # Apply GPU transforms if any
        if hasattr(self, 'gpu_train_transform') and self.gpu_train_transform is not None:
            batch["image"] = self.gpu_train_transform(batch["image"])
        
        # Forward pass
        pred = self.model(batch["image"])
        
        # Compute loss - the MultiLoss handles all the component losses
        loss, individual_losses_dict, extra_dict = self.loss_function(
            pred, batch, epoch=self.current_epoch
        )
        
        # Log total loss
        self.log("loss/train", loss, on_step=True, on_epoch=True, prog_bar=True)
        
        # Log individual component losses
        for loss_name, loss_value in individual_losses_dict.items():
            self.log(f"losses/train_{loss_name}", loss_value, on_step=True, on_epoch=True)
        
        # Log any extra information from losses
        for loss_name, extra_info in extra_dict.items():
            for key, value in extra_info.items():
                self.log(f"extra/train_{loss_name}_{key}", value, on_step=True, on_epoch=True)
        
        # Compute and log metrics (IoU, etc.)
        if "seg" in pred:
            self._log_segmentation_metrics(pred["seg"], batch["gt_polygons_image"], prefix="train")
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        """
        Validation step - unchanged from before.
        """
        # Apply GPU transforms if any
        if hasattr(self, 'gpu_val_transform') and self.gpu_val_transform is not None:
            batch["image"] = self.gpu_val_transform(batch["image"])
        
        # Forward pass
        pred = self.model(batch["image"])
        
        # Compute loss
        loss, individual_losses_dict, extra_dict = self.loss_function(
            pred, batch, epoch=self.current_epoch
        )
        
        # Log total loss
        self.log("loss/val", loss, on_step=False, on_epoch=True, prog_bar=True)
        
        # Log individual losses
        for loss_name, loss_value in individual_losses_dict.items():
            self.log(f"losses/val_{loss_name}", loss_value, on_step=False, on_epoch=True)
        
        # Compute and log metrics
        if "seg" in pred:
            self._log_segmentation_metrics(pred["seg"], batch["gt_polygons_image"], prefix="val")
        
        return loss
    
    def _log_segmentation_metrics(self, pred_seg, gt_seg, prefix="train"):
        """Log segmentation metrics like IoU."""
        # Your existing metrics logging
        pass


# Example usage in training script:
"""
from pytorch_segmentation_models_trainer.model_loader.frame_field_model import FrameFieldSegmentationPLModel
from omegaconf import DictConfig
import hydra

@hydra.main(config_path="conf", config_name="config")
def train(cfg: DictConfig):
    # The model now automatically uses the new compound loss system
    model = FrameFieldSegmentationPLModel(cfg)
    
    # Rest of training code remains the same
    trainer = Trainer(**cfg.pl_trainer)
    trainer.fit(model)

if __name__ == "__main__":
    train()
"""
