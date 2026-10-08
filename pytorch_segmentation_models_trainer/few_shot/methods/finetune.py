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

from typing import List, Optional, Tuple

import torch
from torch import Tensor, nn

from pytorch_segmentation_models_trainer.few_shot.backbone import (
    FrozenLinearHeadSegmenter,
)
from pytorch_segmentation_models_trainer.few_shot.base_method import BaseGFSSMethod
from pytorch_segmentation_models_trainer.few_shot.losses import (
    hierarchical_kd,
    novel_prototypes,
    projected_ce,
)

_ROWS = {"children", "novel", "all"}
_INITS = {"mother", "prototype"}


class FineTune(BaseGFSSMethod):
    """Fine-tuning baselines on the support (transfer learning, B1).

    A linear head over all classes: base rows copied from the base model and
    each novel row initialised from its **mother** (``init: mother``, the
    "initialise by the coarse class" of CCDA) or from the support prototype
    scaled to the mother's norm with the mother's bias (``init: prototype``).
    The rows in ``train_rows`` are trained in ``trainer.fit`` on

    ``CE_S + kd_weight · KL(π_new2old(p) ‖ p_snapshot)``

    where the support CE projects "not novel" pixels onto the sum of base
    probabilities (DIaM Eq. 5) and the KD sums each novel class into its
    mother (DIaM Eq. 11–12 generalised; "mother = Σ children" of CCDA).

    * ``train_rows: children`` — only the rows of the superclass children
      (B1a, isolated fine-tuning of the split);
    * ``train_rows: novel`` — only the novel rows;
    * ``train_rows: all`` — the whole head.

    The backbone is fine-tuned too when ``gfss.backbone.trainable`` is
    ``decoder``, ``all`` or ``lora`` (B1b, M3); then the KD snapshot comes
    from a frozen copy of the base model (``GFSSModel`` passes it).

    Args:
        train_rows: ``children``, ``novel`` or ``all``.
        init: ``mother`` or ``prototype``.
        kd_weight: Weight β of the hierarchical KD (0 = no KD).
        not_novel_weight: CE weight of "not novel" support pixels.

    Example YAML::

        gfss:
          backbone: {trainable: none}     # B1a; "all" for B1b
          method:
            _target_: pytorch_segmentation_models_trainer.few_shot.methods.finetune.FineTune
            train_rows: children
            init: mother
            kd_weight: 0.0
        pl_trainer:
          max_steps: 200
    """

    def __init__(
        self,
        train_rows: str = "children",
        init: str = "mother",
        kd_weight: float = 0.0,
        not_novel_weight: float = 1.0,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        if train_rows not in _ROWS:
            raise ValueError(
                f"train_rows must be one of {sorted(_ROWS)}, got {train_rows!r}."
            )
        if init not in _INITS:
            raise ValueError(f"init must be one of {sorted(_INITS)}, got {init!r}.")
        self.train_rows = train_rows
        self.init = init
        self.kd_weight = float(kd_weight)
        self.not_novel_weight = float(not_novel_weight)

    @property
    def requires_snapshot(self) -> bool:
        """The KD needs the frozen base model's logits on the support."""
        return self.kd_weight > 0

    def _train_index(self) -> List[int]:
        h = self.hierarchy
        if self.train_rows == "all":
            return list(range(h.num_classes))
        if self.train_rows == "novel":
            return list(h.novel_classes)
        return sorted(c for m in h.mothers for c in h.children_of(m))

    def setup(self, hierarchy, base_weight, base_bias, not_novel_index=254, **kwargs):
        super().setup(hierarchy, base_weight, base_bias, not_novel_index, **kwargs)
        n, dim = hierarchy.num_classes, base_weight.shape[1]
        weight = base_weight.new_zeros(n, dim)
        bias = base_bias.new_zeros(n)
        weight[: hierarchy.num_base_classes] = base_weight
        bias[: hierarchy.num_base_classes] = base_bias
        idx = self._train_index()
        self.register_buffer("fixed_weight", weight)
        self.register_buffer("fixed_bias", bias)
        self.register_buffer("train_index", torch.tensor(idx, dtype=torch.long))
        self.train_weight = nn.Parameter(weight[idx].clone())
        self.train_bias = nn.Parameter(bias[idx].clone())

    def init_from_support(self, features: Tensor, masks: Tensor) -> None:
        """Novel rows from the mother (or the scaled prototype)."""
        h = self.hierarchy
        mothers = [h.mother_of(c) for c in h.novel_classes]
        rows_w = self.base_weight[mothers].clone()
        rows_b = self.base_bias[mothers].clone()
        if self.init == "prototype":
            proto = novel_prototypes(features, masks, h).T  # (n_novel, F), unit norm
            rows_w = proto * self.base_weight[mothers].norm(dim=1, keepdim=True)
        with torch.no_grad():
            self.fixed_weight[h.novel_classes] = rows_w
            self.fixed_bias[h.novel_classes] = rows_b
            self.train_weight.copy_(self.fixed_weight[self.train_index])
            self.train_bias.copy_(self.fixed_bias[self.train_index])

    def head(self) -> Tuple[Tensor, Tensor]:
        """Current head (num_classes, F), (num_classes,)."""
        weight = self.fixed_weight.index_copy(0, self.train_index, self.train_weight)
        bias = self.fixed_bias.index_copy(0, self.train_index, self.train_bias)
        return weight, bias

    def forward(self, features: Tensor) -> Tensor:
        return FrozenLinearHeadSegmenter.linear(features, *self.head())

    def support_loss(
        self, features: Tensor, masks: Tensor, snapshot_logits: Optional[Tensor] = None
    ) -> Tensor:
        """Projected CE on the support (+ hierarchical KD to the snapshot)."""
        probas = torch.softmax(self(features), dim=1)
        loss = projected_ce(
            probas.unsqueeze(0),
            masks,
            self.hierarchy,
            self.not_novel_index,
            self.not_novel_weight,
        )[0]
        if self.kd_weight > 0:
            if snapshot_logits is None:
                raise ValueError(
                    "kd_weight > 0 needs the base model's snapshot_logits."
                )
            snapshot = self.base_probabilities(snapshot_logits)
            valid = probas.new_ones(probas.shape[0], 1, *probas.shape[2:])
            kd = hierarchical_kd(
                probas.unsqueeze(1), snapshot.unsqueeze(1), valid, self.hierarchy
            )
            loss = loss + self.kd_weight * kd.mean()
        return loss
