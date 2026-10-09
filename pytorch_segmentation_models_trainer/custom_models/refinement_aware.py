# -*- coding: utf-8 -*-
"""
/***************************************************************************
 pytorch_segmentation_models_trainer
                              -------------------
        begin                : 2026-10-09
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

Refinement-aware base training for category splitting (few-shot).

When a base model is trained with a *superclass* (e.g. two land-cover
classes merged into one), the cross-entropy loss gives no reason to keep
the internal structure of that superclass in the feature space, which is
exactly what a later few-shot split needs. This wrapper adds two
label-free auxiliary objectives on the decoder features of an smp model:

* **Canny edges** — a 1x1 head predicts the Canny edge map of the input
  image (computed on the fly with kornia, no labels), so boundaries that
  exist *inside* the merged superclass (e.g. field limits) stay encoded;
* **sub-prototypes** — ``M`` learnable prototypes of the superclass; the
  superclass pixels are softly assigned to them (cosine / temperature),
  and the loss asks for sharp per-pixel assignments, balanced use of the
  prototypes and separated prototypes (an unsupervised clustering of the
  superclass, in the spirit of part/sub-class prototypes).

The forward still returns plain logits, so metrics, TTA and checkpoints
work as for the wrapped model. ``Model._shared_step`` adds
``compute_auxiliary_losses(masks)`` to the loss.
"""

from typing import Dict, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from pytorch_segmentation_models_trainer.custom_models.edl_wrapper import (
    _instantiate_if_config,
)

_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


def canny_targets(
    images: Tensor,
    size: Tuple[int, int],
    mean: Sequence[float] = _IMAGENET_MEAN,
    std: Sequence[float] = _IMAGENET_STD,
    low_threshold: float = 0.1,
    high_threshold: float = 0.2,
) -> Tensor:
    """Binary Canny edge map of normalized images at the feature resolution.

    The images are de-normalized with ``mean``/``std``, clamped to
    ``[0, 1]``, converted to grayscale, passed through
    ``kornia.filters.canny`` and max-pooled to ``size`` (an edge anywhere in
    a feature cell marks the cell).

    Args:
        images: Normalized images ``(B, C, H, W)`` (first 3 channels used).
        size: Output ``(h, w)``.
        mean: Per-channel normalization mean.
        std: Per-channel normalization std.
        low_threshold: Canny low threshold (gradient magnitude in ``[0, 1]``).
        high_threshold: Canny high threshold.

    Returns:
        Float tensor ``(B, 1, h, w)`` with values in ``{0, 1}``.
    """
    import kornia

    x = images[:, :3].float()
    m = x.new_tensor(mean[:3]).view(1, -1, 1, 1)
    s = x.new_tensor(std[:3]).view(1, -1, 1, 1)
    rgb = (x * s + m).clamp(0.0, 1.0)
    gray = kornia.color.rgb_to_grayscale(rgb)
    _, edges = kornia.filters.canny(
        gray, low_threshold=low_threshold, high_threshold=high_threshold
    )
    return (F.adaptive_max_pool2d(edges, size) > 0).float()


class RefinementAwareWrapper(nn.Module):
    """smp segmentation model + label-free superclass-structure objectives.

    Args:
        model: smp ``SegmentationModel`` (``encoder``, ``decoder``,
            ``segmentation_head``), e.g. ``smp.UPerNet``, or its Hydra config
            (instantiated here, as in ``EvidentialWrapper``).
        superclass_index: Label of the merged superclass in the training
            masks (e.g. 3 = low vegetation).
        edge_weight: Weight of the Canny edge loss (0 disables it).
        edge_region: ``all`` (every valid pixel) or ``superclass`` (only
            superclass pixels, i.e. boundaries inside the merged class).
        edge_low_threshold: Canny low threshold.
        edge_high_threshold: Canny high threshold.
        edge_max_pos_weight: Upper bound of the positive weight of the
            edge BCE (``#neg / #pos`` per batch, edges are sparse).
        normalization_mean: Mean used to normalize the input images.
        normalization_std: Std used to normalize the input images.
        subprototype_weight: Weight of the sub-prototype loss (0 disables).
        num_subprototypes: Number ``M >= 2`` of sub-prototypes.
        temperature: Softmax temperature of the cosine assignments.
        separation_margin: Cosine above which two prototypes are penalized.
        separation_weight: Weight of the separation term inside the
            sub-prototype loss.
        ignore_index: Label ignored in the masks.
    """

    def __init__(
        self,
        model: nn.Module,
        superclass_index: int,
        edge_weight: float = 0.1,
        edge_region: str = "all",
        edge_low_threshold: float = 0.1,
        edge_high_threshold: float = 0.2,
        edge_max_pos_weight: float = 10.0,
        normalization_mean: Sequence[float] = _IMAGENET_MEAN,
        normalization_std: Sequence[float] = _IMAGENET_STD,
        subprototype_weight: float = 0.1,
        num_subprototypes: int = 4,
        temperature: float = 0.1,
        separation_margin: float = 0.0,
        separation_weight: float = 1.0,
        ignore_index: int = 255,
    ) -> None:
        super().__init__()
        if edge_region not in ("all", "superclass"):
            raise ValueError(
                f"edge_region must be 'all' or 'superclass', got {edge_region!r}."
            )
        if subprototype_weight > 0 and num_subprototypes < 2:
            raise ValueError("num_subprototypes must be >= 2.")
        model = _instantiate_if_config(model)
        self.model = model
        self.superclass_index = int(superclass_index)
        self.edge_weight = float(edge_weight)
        self.edge_region = edge_region
        self.edge_low_threshold = edge_low_threshold
        self.edge_high_threshold = edge_high_threshold
        self.edge_max_pos_weight = edge_max_pos_weight
        self.normalization_mean = tuple(normalization_mean)
        self.normalization_std = tuple(normalization_std)
        self.subprototype_weight = float(subprototype_weight)
        self.temperature = temperature
        self.separation_margin = separation_margin
        self.separation_weight = separation_weight
        self.ignore_index = ignore_index
        channels = model.segmentation_head[0].in_channels
        self.edge_head: Optional[nn.Conv2d] = (
            nn.Conv2d(channels, 1, kernel_size=1) if self.edge_weight > 0 else None
        )
        self.subprototypes: Optional[nn.Parameter] = (
            nn.Parameter(torch.randn(num_subprototypes, channels) * 0.02)
            if self.subprototype_weight > 0
            else None
        )
        self.last_features: Optional[Tensor] = None
        self.last_images: Optional[Tensor] = None

    def gfss_unwrap(self) -> nn.Module:
        """The wrapped segmentation model (used by the GFSS backbone)."""
        return self.model

    def forward(self, x: Tensor) -> Tensor:
        """Logits of the wrapped model; keeps decoder features for the loss."""
        features = self.model.decoder(self.model.encoder(x))
        self.last_features = features
        self.last_images = x
        return self.model.segmentation_head(features)

    def _edge_loss(self, features: Tensor, images: Tensor, region: Tensor) -> Tensor:
        logits = self.edge_head(features)
        if not region.any():
            return logits.sum() * 0.0
        with torch.no_grad():
            target = canny_targets(
                images,
                logits.shape[-2:],
                self.normalization_mean,
                self.normalization_std,
                self.edge_low_threshold,
                self.edge_high_threshold,
            )
        r = region.unsqueeze(1)
        pos = target[r].sum()
        neg = r.sum() - pos
        pos_weight = (neg / pos.clamp(min=1.0)).clamp(
            min=1.0, max=self.edge_max_pos_weight
        )
        return F.binary_cross_entropy_with_logits(
            logits[r], target[r], pos_weight=pos_weight
        )

    def _subprototype_loss(self, features: Tensor, superclass: Tensor) -> Tensor:
        """Sharp + balanced assignment of superclass pixels, separated prototypes.

        ``mean H(a_i) - H(mean a_i) + λ·mean_{j<k} relu(cos(P_j, P_k) - m)``
        over the superclass pixels ``i`` (``a_i`` = softmax of the cosine
        similarities to the prototypes ``P`` divided by the temperature).
        """
        if not superclass.any():
            return self.subprototypes.sum() * 0.0
        z = F.normalize(features.permute(0, 2, 3, 1)[superclass], dim=-1)
        p = F.normalize(self.subprototypes, dim=-1)
        a = F.softmax(z @ p.t() / self.temperature, dim=-1)
        sharp = -(a * a.clamp(min=1e-12).log()).sum(-1).mean()
        mean_a = a.mean(0)
        balance = (mean_a * mean_a.clamp(min=1e-12).log()).sum()
        cos = p @ p.t()
        j, k = torch.triu_indices(len(p), len(p), offset=1)
        separation = F.relu(cos[j, k] - self.separation_margin).mean()
        return sharp + balance + self.separation_weight * separation

    def compute_auxiliary_losses(self, masks: Tensor) -> Dict[str, Tensor]:
        """Weighted auxiliary losses of the last forward (then freed).

        Args:
            masks: Hard label masks ``(B, H, W)`` of the batch.

        Returns:
            ``{"edge": ..., "subproto": ...}`` for the enabled terms (empty
            when no forward is pending).
        """
        features, images = self.last_features, self.last_images
        self.last_features = self.last_images = None
        if features is None:
            return {}
        size = features.shape[-2:]
        m = F.interpolate(masks.unsqueeze(1).float(), size=size, mode="nearest")
        m = m.squeeze(1).long()
        superclass = m == self.superclass_index
        out: Dict[str, Tensor] = {}
        if self.edge_head is not None:
            region = (
                superclass
                if self.edge_region == "superclass"
                else (m != self.ignore_index)
            )
            out["edge"] = self.edge_weight * self._edge_loss(features, images, region)
        if self.subprototypes is not None:
            out["subproto"] = self.subprototype_weight * self._subprototype_loss(
                features, superclass
            )
        return out
