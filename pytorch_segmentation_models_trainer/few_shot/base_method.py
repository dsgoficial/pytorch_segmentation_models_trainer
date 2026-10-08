# -*- coding: utf-8 -*-
"""
/***************************************************************************
 pytorch_segmentation_models_trainer
                              -------------------
        begin                : 2026-10-07
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
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from pytorch_segmentation_models_trainer.few_shot.hierarchy import ClassHierarchy


class BaseGFSSMethod(nn.Module, ABC):
    """Abstract GFSS method operating in the feature space of a frozen base.

    Instantiated from ``gfss.method`` (Hydra ``_target_``) and owned by
    ``GFSSModel`` (the LightningModule), which handles data, the frozen
    backbone (``FrozenLinearHeadSegmenter``), upsampling and metrics. All
    tensors a method receives are at the feature resolution; support masks
    are downsampled by nearest neighbour and use the novel class indices of
    the hierarchy, ``not_novel_index`` for "labelled, not any novel class"
    (support regime S-novel) and 255 for ignored pixels.

    Lifecycle:

    1. ``setup(hierarchy, base_weight, base_bias)`` once, before anything else.
    2. ``init_from_support(features, masks)`` with the whole support set.
    3. ``support_loss(features, masks)`` per training step (inductive
       adaptation via ``trainer.fit``); ``None`` means nothing to train.
    4. If ``transductive``, ``adapt_to_query(support_features,
       support_masks, query_features)`` before predicting each test batch.
    5. ``forward(features)`` returns logits over ``hierarchy.num_classes``.

    Subclasses must accept ``**kwargs`` (Hydra).
    """

    transductive: bool = False

    def __init__(self, **kwargs) -> None:
        super().__init__()
        self.hierarchy: Optional[ClassHierarchy] = None
        self.not_novel_index: int = 254
        self.base_output: str = "softmax"

    def setup(
        self,
        hierarchy: ClassHierarchy,
        base_weight: Tensor,
        base_bias: Tensor,
        not_novel_index: int = 254,
        base_output: str = "softmax",
    ) -> None:
        """Receive the task hierarchy and the frozen base classifier.

        Args:
            hierarchy: Mother → children mapping of the task.
            base_weight: ``(num_base_classes, F)`` base classifier weight.
            base_bias: ``(num_base_classes,)`` base classifier bias.
            not_novel_index: Support label meaning "not any novel class".
            base_output: ``softmax`` (cross-entropy base model) or
                ``evidential`` (Dirichlet base model, ``EvidentialWrapper``).
        """
        if base_output not in {"softmax", "evidential"}:
            raise ValueError(
                f"base_output must be 'softmax' or 'evidential', got {base_output!r}."
            )
        self.hierarchy = hierarchy
        self.not_novel_index = not_novel_index
        self.base_output = base_output
        self.register_buffer("base_weight", base_weight.clone())
        self.register_buffer("base_bias", base_bias.clone())

    def base_alpha(self, logits: Tensor) -> Tensor:
        """Dirichlet parameters ``softplus(logits) + 1`` of an evidential base."""
        return F.softplus(logits) + 1.0

    def base_probabilities(self, logits: Tensor) -> Tensor:
        """Base class probabilities: softmax, or the Dirichlet mean ``α / S``."""
        if self.base_output == "evidential":
            alpha = self.base_alpha(logits)
            return alpha / alpha.sum(dim=1, keepdim=True)
        return torch.softmax(logits, dim=1)

    def init_from_support(self, features: Tensor, masks: Tensor) -> None:
        """Initialise from the full support set (default: nothing)."""
        return None

    def support_loss(self, features: Tensor, masks: Tensor) -> Optional[Tensor]:
        """Loss of one support batch (default: nothing to train)."""
        return None

    def adapt_to_query(
        self, support_features: Tensor, support_masks: Tensor, query_features: Tensor
    ) -> None:
        """Transductive adaptation to one query batch (default: nothing)."""
        return None

    def uncertainty_names(self) -> List[str]:
        """Names of the maps returned by ``uncertainty`` (default: none).

        Methods that estimate per-pixel uncertainty override this and
        ``uncertainty(features) -> {name: (B, h, w)}`` with values in
        ``[0, 1]``; ``GFSSModel`` then evaluates them (AURC of the split
        decisions, per-tile Spearman with the tile error).
        """
        return []

    def variant_names(self) -> List[str]:
        """Names of extra decoding variants (default: none).

        Methods with variants (e.g. a threshold sweep) override this and
        ``decode_variants(features) -> {name: logits}``; ``GFSSModel``
        evaluates each one with its own ``GFSSMetrics``
        (``test/var/<name>/...``).
        """
        return []

    def decode_variants(self, features: Tensor) -> Dict[str, Tensor]:
        """Logits of the extra decoding variants (default: none)."""
        return {}

    @abstractmethod
    def forward(self, features: Tensor) -> Tensor:
        """Logits ``(B, num_classes, h, w)`` for features ``(B, F, h, w)``."""
