# -*- coding: utf-8 -*-
"""
Student model for Dataset Distillation (DDOQ).
Inherits from the base Model class to maintain consistency with the framework.
"""

import logging
import torch.nn.functional as F
from pytorch_segmentation_models_trainer.model_loader.model import Model

logger = logging.getLogger(__name__)


class StudentSegmentationModel(Model):
    """
    Segmentation model trained on a distilled coreset using DDOQ weights.
    Implements: min_theta sum(w * Loss(x, y, theta)).
    """

    def __init__(self, cfg):
        super().__init__(cfg)
        self.use_soft_labels = cfg.get("use_soft_labels", False)
        logger.info(
            f"Initialized StudentSegmentationModel (Soft Labels: {self.use_soft_labels})"
        )

    def _shared_step(self, batch, prefix):
        """
        Reimplementation of the shared_step with DDOQ specific weight handling.
        """
        is_train = prefix == "train"

        # 1. Unpack images and masks using parent logic
        images, masks = self._unpack_batch(batch)

        # 2. Extract DDOQ weights if in training mode
        weights = None
        if is_train:
            if isinstance(batch, dict):
                weight_key = self.cfg.get("weight_key", "weight")
                weights = batch.get(weight_key)
            elif isinstance(batch, (list, tuple)) and len(batch) == 3:
                weights = batch[2]

        # 3. Handle label types for metrics
        if masks.is_floating_point():
            hard_masks = self._soft_to_hard_masks(masks)
        else:
            masks = masks.long()
            hard_masks = masks

        # 4. Forward pass
        predicted_masks = self(images)

        # 5. Compute loss (possibly weighted)
        loss, individual_losses, extra_info = self._compute_loss(
            predicted_masks, masks, weights=weights if is_train else None
        )

        # 6. Logging and Metrics (consistent with Model class)
        self.log(
            f"loss/{prefix}",
            loss,
            on_step=is_train,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )

        # Handle metrics
        metrics_attr = f"{prefix}_metrics"
        if hasattr(self, metrics_attr):
            # Resolve predicted_masks for metrics (handle dict/EDL case)
            if isinstance(predicted_masks, dict) and "probs" in predicted_masks:
                predicted_masks_for_metrics = predicted_masks["probs"]
            else:
                predicted_masks_for_metrics = predicted_masks

            preds_ready = self._prepare_preds_for_metrics(predicted_masks_for_metrics)
            if preds_ready is not None:
                metrics = getattr(self, metrics_attr)(preds_ready, hard_masks)
                self.log_dict(
                    metrics,
                    on_step=is_train,
                    on_epoch=True,
                    prog_bar=False,
                    sync_dist=True,
                )

        return loss

    def _compute_loss(self, predicted_masks, masks, weights=None):
        """
        Extended loss computation to support DDOQ weighting.
        """
        # If weights are provided and we are in training, apply the weighted loss logic
        if self.training and weights is not None:
            if self.use_soft_labels:
                # Default DDOQ path for soft labels
                log_preds = F.log_softmax(predicted_masks, dim=1)
                loss_pixel = F.kl_div(log_preds, masks, reduction="none")
                # Average over spatial dimensions and sum over classes
                loss_per_image = loss_pixel.sum(dim=1).mean(dim=(1, 2))
                return (loss_per_image * weights).mean(), {}, {}

            # Use the configured loss function with reduction='none'
            old_reduction = self._get_loss_reduction()
            try:
                self._set_loss_reduction("none")
                # Leverage parent _compute_loss to handle MultiLoss, CompoundLoss, etc.
                loss_val, individual_losses, extra_info = super()._compute_loss(
                    predicted_masks, masks
                )

                # Spatial reduction: (B, H, W) -> (B,) or (B, C, H, W) -> (B,)
                if loss_val.ndim >= 3:
                    loss_per_image = loss_val.mean(dim=list(range(1, loss_val.ndim)))
                elif loss_val.ndim == 0:
                    loss_per_image = loss_val.expand(weights.shape[0])
                else:
                    loss_per_image = loss_val

                weighted_loss = (loss_per_image * weights).mean()
                return weighted_loss, individual_losses, extra_info
            finally:
                self._set_loss_reduction(old_reduction)

        # Standard fallback for validation or when no weights are supplied
        return super()._compute_loss(predicted_masks, masks)

    def _get_loss_reduction(self):
        """Helper to get current loss reduction."""
        return getattr(self.loss_function, "reduction", "mean")

    def _set_loss_reduction(self, reduction):
        """Helper to set loss reduction recursively if needed."""
        if hasattr(self.loss_function, "reduction"):
            self.loss_function.reduction = reduction
        elif hasattr(self.loss_function, "loss_funcs"):  # Support for MultiLoss
            for loss_func in self.loss_function.loss_funcs:
                if hasattr(loss_func, "reduction"):
                    loss_func.reduction = reduction
