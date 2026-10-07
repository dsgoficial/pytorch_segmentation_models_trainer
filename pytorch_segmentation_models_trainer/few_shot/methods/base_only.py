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

import torch
from torch import Tensor

from pytorch_segmentation_models_trainer.few_shot.backbone import (
    FrozenLinearHeadSegmenter,
)
from pytorch_segmentation_models_trainer.few_shot.base_method import BaseGFSSMethod


class BaseOnly(BaseGFSSMethod):
    """Lower bound: the frozen base model, never predicting novel classes.

    Base logits are kept unchanged and every novel class receives a logit
    far below the minimum, so the novel IoU is 0 and base metrics equal the
    base model's. Useful as the floor of the GFSS tables and as a pipeline
    smoke test.

    Example YAML::

        gfss:
          method:
            _target_: pytorch_segmentation_models_trainer.few_shot.methods.base_only.BaseOnly
    """

    def forward(self, features: Tensor) -> Tensor:
        base = FrozenLinearHeadSegmenter.linear(
            features, self.base_weight, self.base_bias
        )
        n_novel = len(self.hierarchy.novel_classes)
        floor = base.amin(dim=1, keepdim=True) - 1e4
        return torch.cat([base, floor.expand(-1, n_novel, -1, -1)], dim=1)
