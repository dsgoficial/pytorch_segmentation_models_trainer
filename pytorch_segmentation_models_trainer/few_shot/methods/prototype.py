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
    initialisation), multiplied by ``scale`` and with zero bias:

    * ``scale: base_norm`` (default) — mean L2 norm of the base rows, so the
      novel logits live on the scale of the base ones (weight imprinting,
      Qi et al. 2018) [adaptation: imprinting in PIFS assumes a cosine
      classifier, whereas the base models here use a linear one];
    * ``scale: unit`` — DIaM's initial classifier before any optimisation;
    * a float — fixed scale.

    Example YAML::

        gfss:
          method:
            _target_: pytorch_segmentation_models_trainer.few_shot.methods.prototype.PrototypeImprinting
            scale: base_norm
        pl_trainer:
          max_steps: 0
    """

    def __init__(self, scale="base_norm", **kwargs) -> None:
        super().__init__(**kwargs)
        if not (scale in {"base_norm", "unit"} or isinstance(scale, (int, float))):
            raise ValueError(
                f"scale must be 'base_norm', 'unit' or a number, got {scale!r}."
            )
        self.scale = scale

    def init_from_support(self, features: Tensor, masks: Tensor) -> None:
        proto = novel_prototypes(features, masks, self.hierarchy)
        if self.scale == "base_norm":
            factor = self.base_weight.norm(dim=1).mean()
        elif self.scale == "unit":
            factor = 1.0
        else:
            factor = float(self.scale)
        self.register_buffer("novel_weight", (proto * factor).T.contiguous())

    def forward(self, features: Tensor) -> Tensor:
        weight = torch.cat([self.base_weight, self.novel_weight], dim=0)
        bias = torch.cat(
            [self.base_bias, self.base_bias.new_zeros(self.novel_weight.shape[0])]
        )
        return FrozenLinearHeadSegmenter.linear(features, weight, bias)
