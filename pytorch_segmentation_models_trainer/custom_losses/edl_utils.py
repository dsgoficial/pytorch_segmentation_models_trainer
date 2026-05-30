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

Utility functions for Evidential Deep Learning (EDL).

Reference:
    Sensoy et al. (2018). Evidential Deep Learning to Quantify Classification
    Uncertainty. NeurIPS 2018.  https://arxiv.org/abs/1806.01768
"""

import torch
import torch.nn.functional as F
from torch import Tensor

# ---------------------------------------------------------------------------
# Label helpers
# ---------------------------------------------------------------------------


def one_hot_encode(labels: Tensor, num_classes: int) -> Tensor:
    """Convert segmentation labels to one-hot float tensors.

    Accepts either:
      - Hard labels  ``[B, H, W]``  int/long  → converts to ``[B, K, H, W]`` float
      - Soft labels  ``[B, K, H, W]`` float   → returned as-is (passthrough)

    Pixels with label value >= ``num_classes`` (e.g. ignore_index=255) are mapped
    to an all-zeros vector so they do not contribute to the loss.

    Args:
        labels: Ground-truth labels.
        num_classes: Number of classes K.

    Returns:
        Float tensor of shape ``[B, K, H, W]`` with values in ``[0, 1]``.
    """
    if labels.dim() == 4 and labels.is_floating_point():
        # Already one-hot / soft labels
        return labels.float()

    if labels.dim() != 3:
        raise ValueError(f"Hard labels must be 3-D [B, H, W], got shape {labels.shape}")

    B, H, W = labels.shape
    # Clamp to valid range so F.one_hot doesn't error on ignore pixels
    clamped = labels.clamp(0, num_classes - 1).long()
    one_hot = F.one_hot(clamped, num_classes=num_classes)  # [B, H, W, K]
    one_hot = one_hot.permute(0, 3, 1, 2).float()  # [B, K, H, W]

    # Zero out ignore pixels (original label >= num_classes)
    ignore_mask = (labels >= num_classes).unsqueeze(1)  # [B, 1, H, W]
    one_hot = one_hot.masked_fill(ignore_mask, 0.0)

    return one_hot


# ---------------------------------------------------------------------------
# Dirichlet statistics helpers
# ---------------------------------------------------------------------------


def dirichlet_strength(alpha: Tensor) -> Tensor:
    """Compute the Dirichlet strength S = sum_k alpha_k.

    Args:
        alpha: Dirichlet parameters, shape ``[B, K, H, W]``, values >= 1.

    Returns:
        Strength tensor of shape ``[B, 1, H, W]``.
    """
    return alpha.sum(dim=1, keepdim=True)


def epistemic_uncertainty(alpha: Tensor) -> Tensor:
    """Compute pixel-wise epistemic uncertainty u = K / S.

    When the model has no evidence (all alpha_k ≈ 1) the uncertainty is
    maximal (u ≈ 1).  When evidence is concentrated on one class u → 0.

    Args:
        alpha: Dirichlet parameters, shape ``[B, K, H, W]``, values >= 1.

    Returns:
        Uncertainty map of shape ``[B, 1, H, W]`` with values in ``(0, 1]``.
    """
    K = alpha.shape[1]
    S = dirichlet_strength(alpha)
    return K / S


# ---------------------------------------------------------------------------
# KL divergence between Dirichlet distributions
# ---------------------------------------------------------------------------


def dirichlet_kl_divergence(alpha: Tensor, beta: Tensor) -> Tensor:
    """Analytic KL[ Dir(alpha) || Dir(beta) ] per pixel.

    Uses the formula:
        KL = lgamma(S_a) - lgamma(S_b)
           - sum_k [ lgamma(alpha_k) - lgamma(beta_k) ]
           + sum_k [ (alpha_k - beta_k) * (digamma(alpha_k) - digamma(S_a)) ]

    where S_a = sum alpha_k, S_b = sum beta_k.

    Args:
        alpha: Shape ``[B, K, H, W]``, values > 0.
        beta:  Shape ``[B, K, H, W]``, values > 0 (target Dirichlet params).

    Returns:
        Per-pixel KL, shape ``[B, H, W]``, values >= 0.
    """
    alpha = alpha.clamp(min=1e-6)
    beta = beta.clamp(min=1e-6)

    S_a = alpha.sum(dim=1)  # [B, H, W]
    S_b = beta.sum(dim=1)  # [B, H, W]

    kl = (
        torch.lgamma(S_a)
        - torch.lgamma(S_b)
        - (torch.lgamma(alpha) - torch.lgamma(beta)).sum(dim=1)
        + (
            (alpha - beta) * (torch.digamma(alpha) - torch.digamma(S_a.unsqueeze(1)))
        ).sum(dim=1)
    )
    return kl.clamp(min=0.0)


def kl_divergence_to_uniform(alpha: Tensor) -> Tensor:
    """KL[ Dir(alpha) || Dir(1,...,1) ] per pixel.

    The uniform Dirichlet Dir(1,...,1) represents maximum ignorance. This
    term penalises predictions that are *more* concentrated than a uniform
    distribution. Called after removing evidence for the correct class so
    that the model is not penalised for *correct* high-confidence outputs.

    Args:
        alpha: Shape ``[B, K, H, W]``, values >= 1.

    Returns:
        Per-pixel KL, shape ``[B, H, W]``, values >= 0.
    """
    K = alpha.shape[1]
    ones = torch.ones_like(alpha)
    return dirichlet_kl_divergence(alpha, ones)


# ---------------------------------------------------------------------------
# EDL-specific KL regulariser
# ---------------------------------------------------------------------------


def edl_kl_regulariser(alpha: Tensor, y: Tensor) -> Tensor:
    """Compute the EDL KL regulariser per pixel.

    Following Sensoy et al. (2018) eq. (8): before computing the KL we
    remove the evidence from the ground-truth class so the network is not
    penalised for *correct* high-confidence predictions:

        alpha_tilde_k = y_k + (1 - y_k) * alpha_k

    Args:
        alpha: Dirichlet parameters ``[B, K, H, W]``, values >= 1.
        y:     One-hot labels ``[B, K, H, W]``, values in ``{0, 1}`` (or soft).

    Returns:
        Per-pixel KL, shape ``[B, H, W]``.
    """
    alpha_tilde = y + (1.0 - y) * alpha
    return kl_divergence_to_uniform(alpha_tilde)
