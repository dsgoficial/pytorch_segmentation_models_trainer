# -*- coding: utf-8 -*-
"""Soft-label weighted cross-entropy loss for multi-source consensus training."""

from typing import Dict, Optional, Union

import torch
import torch.nn.functional as F

from pytorch_segmentation_models_trainer.custom_losses.base_loss import Loss


class SoftLabelWeightedCELoss(Loss):
    """Pixel-wise cross-entropy loss weighted by per-pixel confidence maps.

    Implements the noise-aware weak supervision loss from Xiao et al. (2026):

        L(i) = W_conf(i) · [-Σ_c P_soft(i,c) · log(pred(i,c))]
        L     = mean_i [L(i)]

    When ``w_conf`` is absent the formula reduces to the standard soft
    cross-entropy mean (equivalent to uniform confidence).

    Args:
        name: Loss identifier used by MultiLoss logging.
        num_classes: Number of segmentation classes C.
        weight_key: Key in a dict gt_batch for the per-pixel confidence weights
            tensor ``(B, 1, H, W)`` float.
        mask_key: Key in a dict gt_batch for the soft label distribution tensor
            ``(B, C, H, W)`` float.
        **kwargs: Accepted for Hydra / ConfigStore compatibility.

    Example YAML:
        loss:
          _target_: pytorch_segmentation_models_trainer.custom_losses.soft_label_loss.SoftLabelWeightedCELoss
          name: soft_label_ce
          num_classes: 6
          weight_key: w_conf
          mask_key: mask
    """

    def __init__(
        self,
        name: str = "soft_label_ce",
        num_classes: int = 2,
        weight_key: str = "w_conf",
        mask_key: str = "mask",
        **kwargs,
    ) -> None:
        super().__init__(name)
        self.num_classes = num_classes
        self.weight_key = weight_key
        self.mask_key = mask_key

    def compute(
        self,
        pred_batch: Union[torch.Tensor, Dict[str, torch.Tensor]],
        gt_batch: Union[torch.Tensor, Dict[str, torch.Tensor]],
    ) -> torch.Tensor:
        """Compute the confidence-weighted soft cross-entropy.

        Args:
            pred_batch: Logits ``(B, C, H, W)``, or a dict with a ``"seg"`` key
                (FrameField / compound-loss path).
            gt_batch: Soft label tensor ``(B, C, H, W)`` float, or a dict with
                ``mask_key`` and optionally ``weight_key`` entries.

        Returns:
            Scalar loss tensor.
        """
        if isinstance(pred_batch, dict):
            logits: torch.Tensor = pred_batch.get(
                "seg", next(iter(pred_batch.values()))
            )
        else:
            logits = pred_batch

        if isinstance(gt_batch, dict):
            p_soft: torch.Tensor = gt_batch[self.mask_key]
            w_conf: Optional[torch.Tensor] = gt_batch.get(self.weight_key)
        else:
            p_soft = gt_batch
            w_conf = None

        log_prob = F.log_softmax(logits, dim=1)  # (B, C, H, W)
        ce_per_pixel = -(p_soft * log_prob).sum(dim=1, keepdim=True)  # (B, 1, H, W)

        if w_conf is not None:
            return (ce_per_pixel * w_conf).mean()
        return ce_per_pixel.mean()
