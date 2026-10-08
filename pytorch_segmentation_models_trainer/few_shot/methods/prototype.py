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

from torch import Tensor

import torch

from pytorch_segmentation_models_trainer.few_shot.backbone import (
    FrozenLinearHeadSegmenter,
)
from pytorch_segmentation_models_trainer.few_shot.base_method import BaseGFSSMethod
from pytorch_segmentation_models_trainer.few_shot.losses import novel_prototypes


class PrototypeImprinting(BaseGFSSMethod):
    """Training-free baseline: novel rows imprinted from support prototypes.

    The base classifier is kept unchanged; each novel class gets a row equal
    to the L2-normalised mean support feature of its pixels (as DIaM's
    initialisation), multiplied by ``scale``, with bias ``bias``:

    * ``scale: base_norm`` (default) — mean L2 norm of the base rows, so the
      novel logits live on the scale of the base ones (weight imprinting,
      Qi et al. 2018) [adaptation: imprinting in PIFS assumes a cosine
      classifier, whereas the base models here use a linear one];
    * ``scale: mother_norm`` — L2 norm of the novel class's mother row;
    * ``scale: unit`` — DIaM's initial classifier before any optimisation;
    * a float — fixed scale.

    * ``bias: zero`` (default), ``base_mean`` (mean of the base biases) or
      ``mother`` (the mother's bias). With ``scale: mother_norm`` and
      ``bias: mother`` a pixel goes to the novel class iff its features are
      closer (in cosine) to the novel prototype than to the mother's row;
      ``mother_*`` options use the hierarchy (not hierarchy-agnostic).

    Example YAML::

        gfss:
          method:
            _target_: pytorch_segmentation_models_trainer.few_shot.methods.prototype.PrototypeImprinting
            scale: base_norm
            bias: zero
        pl_trainer:
          max_steps: 0
    """

    def __init__(self, scale="base_norm", bias: str = "zero", **kwargs) -> None:
        super().__init__(**kwargs)
        if not (
            scale in {"base_norm", "mother_norm", "unit"}
            or isinstance(scale, (int, float))
        ):
            raise ValueError(
                "scale must be 'base_norm', 'mother_norm', 'unit' or a number, "
                f"got {scale!r}."
            )
        if bias not in {"zero", "base_mean", "mother"}:
            raise ValueError(
                f"bias must be 'zero', 'base_mean' or 'mother', got {bias!r}."
            )
        self.scale = scale
        self.bias = bias

    def init_from_support(self, features: Tensor, masks: Tensor) -> None:
        proto = novel_prototypes(features, masks, self.hierarchy)  # (F, n_novel)
        mothers = [self.hierarchy.mother_of(c) for c in self.hierarchy.novel_classes]
        if self.scale == "base_norm":
            factor = self.base_weight.norm(dim=1).mean()
        elif self.scale == "mother_norm":
            factor = self.base_weight[mothers].norm(dim=1).unsqueeze(0)
        elif self.scale == "unit":
            factor = 1.0
        else:
            factor = float(self.scale)
        self.register_buffer("novel_weight", (proto * factor).T.contiguous())
        if self.bias == "zero":
            novel_bias = self.base_bias.new_zeros(len(mothers))
        elif self.bias == "base_mean":
            novel_bias = self.base_bias.mean().repeat(len(mothers))
        else:
            novel_bias = self.base_bias[mothers].clone()
        self.register_buffer("novel_bias", novel_bias)

    def forward(self, features: Tensor) -> Tensor:
        weight = torch.cat([self.base_weight, self.novel_weight], dim=0)
        bias = torch.cat([self.base_bias, self.novel_bias])
        return FrozenLinearHeadSegmenter.linear(features, weight, bias)
