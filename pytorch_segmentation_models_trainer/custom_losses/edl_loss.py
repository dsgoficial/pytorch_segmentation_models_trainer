# -*- coding: utf-8 -*-
"""
/***************************************************************************
 pytorch_segmentation_models_trainer
                              -------------------
        begin                : 2026-04-15
        git sha              : $Format:%H$
        copyright            : (C) 2026 by Philipe Borba - Cartographic Engineer
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

Evidential Deep Learning loss functions for semantic segmentation.

Two losses are provided so they can be combined via the framework's existing
``CompoundLoss`` / ``MultiLoss`` mechanism with epoch-based weight scheduling
(KL annealing):

    EvidentialMSELoss  — MSE integrated over Dirichlet distribution (eq. 4
                         of Sensoy et al. 2018)
    EvidentialKLLoss   — KL divergence regulariser (eq. 8, after removing
                         evidence for correct class)

Both losses accept either:
  * a plain tensor ``[B, K, H, W]`` (logits — Softplus applied internally)
  * a dict with key ``"alpha"``  (pre-computed Dirichlet params, Softplus
    already applied by the EvidentialWrapper)

Reference:
    Sensoy et al. (2018). Evidential Deep Learning to Quantify Classification
    Uncertainty. NeurIPS 2018.  https://arxiv.org/abs/1806.01768
"""

import logging

import torch
from torch import Tensor

from pytorch_segmentation_models_trainer.custom_losses.base_loss import Loss
from pytorch_segmentation_models_trainer.custom_losses.edl_utils import (
    dirichlet_strength,
    edl_kl_regulariser,
    one_hot_encode,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers shared by both losses
# ---------------------------------------------------------------------------


def _extract_alpha(pred_batch) -> Tensor:
    """Return Dirichlet alpha from a dict or raw logit tensor.

    If ``pred_batch`` is a dict we expect the ``EvidentialWrapper`` to have
    already computed ``alpha = Softplus(logits) + 1``.  If it is a plain
    tensor we apply Softplus and add 1 here (fallback for users who did not
    wrap the model).

    Args:
        pred_batch: Dict with key ``"alpha"`` or raw tensor ``[B, K, H, W]``.

    Returns:
        Alpha tensor ``[B, K, H, W]``, values >= 1.
    """
    if isinstance(pred_batch, dict):
        if "alpha" not in pred_batch:
            raise KeyError(
                "EDL loss received a dict without 'alpha' key. "
                "Make sure your model is wrapped with EvidentialWrapper."
            )
        return pred_batch["alpha"]
    # Raw logit tensor — apply Softplus + 1
    return torch.nn.functional.softplus(pred_batch) + 1.0


def _mean_over_valid_pixels(per_pixel: Tensor, y: Tensor) -> Tensor:
    """Average ``per_pixel`` [B, H, W] only over pixels with valid labels.

    Ignore pixels are those where all one-hot channels are 0 (produced by
    ``one_hot_encode`` when the original label was >= num_classes).

    Args:
        per_pixel: Loss per pixel, shape ``[B, H, W]``.
        y:         One-hot labels ``[B, K, H, W]``.

    Returns:
        Scalar mean loss.
    """
    valid = y.sum(dim=1) > 0  # [B, H, W]
    if valid.any():
        return per_pixel[valid].mean()
    return per_pixel.mean()


# ---------------------------------------------------------------------------
# MSE loss (integrated over Dirichlet)
# ---------------------------------------------------------------------------


class EvidentialMSELoss(Loss):
    """MSE loss integrated analytically over the Dirichlet distribution.

    For each pixel and class k the expected squared error between the
    predicted probability and the one-hot label is:

        E[(y_k - p_k)²] = (y_k - α_k/S)²  +  α_k(S - α_k) / (S²(S+1))
                        = bias²             +  variance term

    Summing over classes gives the per-pixel loss.

    Args:
        name:        Loss name used for logging (default ``"edl_mse"``).
        num_classes: Number of segmentation classes K.
        ignore_index: Label value to ignore (default 255).
    """

    def __init__(
        self,
        name: str = "edl_mse",
        num_classes: int = 2,
        ignore_index: int = 255,
    ):
        super().__init__(name)
        self.num_classes = num_classes
        self.ignore_index = ignore_index

    def compute(self, pred_batch, gt_batch) -> Tensor:
        """Compute EDL MSE loss.

        Args:
            pred_batch: Dict with ``"alpha"`` key or raw logit tensor.
            gt_batch:   Labels ``[B, H, W]`` long or ``[B, K, H, W]`` float.

        Returns:
            Scalar loss value.
        """
        alpha = _extract_alpha(pred_batch)  # [B, K, H, W]
        y = one_hot_encode(gt_batch, self.num_classes).to(alpha.device)

        S = dirichlet_strength(alpha)  # [B, 1, H, W]
        p_hat = alpha / S  # [B, K, H, W]

        # Bias² term
        bias_sq = (y - p_hat) ** 2  # [B, K, H, W]

        # Variance term
        var = alpha * (S - alpha) / (S**2 * (S + 1))  # [B, K, H, W]

        per_pixel = (bias_sq + var).sum(dim=1)  # [B, H, W]

        return _mean_over_valid_pixels(per_pixel, y)


# ---------------------------------------------------------------------------
# KL regulariser loss
# ---------------------------------------------------------------------------


class EvidentialKLLoss(Loss):
    """KL divergence regulariser for EDL.

    Following Sensoy et al. (2018) eq. (8): before computing the KL the
    evidence of the correct class is removed so the network is not penalised
    for *correct* high-confidence predictions:

        α̃_k = y_k + (1 - y_k) · α_k

    Then KL[ Dir(α̃) || Dir(1,...,1) ] is computed analytically.

    This loss is typically given weight 0 at the start of training and
    annealed to 1.0 over time using the ``CompoundLoss`` epoch-threshold
    mechanism.

    Args:
        name:        Loss name used for logging (default ``"edl_kl"``).
        num_classes: Number of segmentation classes K.
        ignore_index: Label value to ignore (default 255).
    """

    def __init__(
        self,
        name: str = "edl_kl",
        num_classes: int = 2,
        ignore_index: int = 255,
    ):
        super().__init__(name)
        self.num_classes = num_classes
        self.ignore_index = ignore_index

    def compute(self, pred_batch, gt_batch) -> Tensor:
        """Compute EDL KL regulariser.

        Args:
            pred_batch: Dict with ``"alpha"`` key or raw logit tensor.
            gt_batch:   Labels ``[B, H, W]`` long or ``[B, K, H, W]`` float.

        Returns:
            Scalar loss value.
        """
        alpha = _extract_alpha(pred_batch)  # [B, K, H, W]
        y = one_hot_encode(gt_batch, self.num_classes).to(alpha.device)

        per_pixel = edl_kl_regulariser(alpha, y)  # [B, H, W]

        return _mean_over_valid_pixels(per_pixel, y)
