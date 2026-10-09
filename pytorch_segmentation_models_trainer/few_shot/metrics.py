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

from typing import Dict, List, Optional, Sequence

import torch
import torch.nn.functional as F
from torch import Tensor
from torchmetrics import Metric

from pytorch_segmentation_models_trainer.few_shot.hierarchy import ClassHierarchy


def boundary_band(target: Tensor, width: int, ignore_index: int = 255) -> Tensor:
    """Boolean mask of the pixels within ``width`` pixels of a label change.

    A pixel is in the band when its ``(2·width+1)²`` window holds two different
    valid labels (the *trimap* band around the ground-truth boundaries).
    ``ignore_index`` pixels and the tile border are not boundaries, and
    ``ignore_index`` pixels are never in the band.

    Args:
        target: Label map ``(..., H, W)``.
        width: Band half-width in pixels (``>= 1``).
        ignore_index: Label excluded from the boundaries.

    Returns:
        Boolean tensor with the shape of ``target``.
    """
    shape = target.shape
    t = target.reshape(-1, 1, shape[-2], shape[-1]).float()
    valid = t != ignore_index
    k = 2 * width + 1
    # max_pool2d pads with -inf, so the tile border never creates a change.
    hi = F.max_pool2d(torch.where(valid, t, -1.0), k, stride=1, padding=width)
    lo = -F.max_pool2d(
        torch.where(valid, -t, -float(ignore_index) - 1.0),
        k,
        stride=1,
        padding=width,
    )
    band = (hi >= 0) & (hi != lo) & valid
    return band.reshape(shape)


class GFSSMetrics(Metric):
    """Confusion-matrix metrics for generalized few-shot segmentation.

    Accumulates two confusion matrices over all evaluated pixels:

    * ``confmat[gt, pred]`` (``num_classes × num_classes``) of the adapted
      model;
    * ``base_confmat[gt, base_pred]`` (``num_classes × num_base_classes``) of
      the frozen base model, when its predictions are given.

    ``compute()`` returns scalar tensors (absent classes give ``nan`` and are
    skipped in the means):

    * ``iou/<c>``, ``precision/<c>``, ``recall/<c>`` per class;
    * ``miou``, ``miou_base``, ``miou_novel`` and ``oem_score``
      (``0.4·miou_base + 0.6·miou_novel``, OpenEarthMap Few-Shot Challenge);
    * ``locality``: pixel accuracy of the adapted model divided by that of
      the base model, over ground-truth pixels of *untouched* classes (base
      classes that are not mothers); 1 means no change there;
    * ``split_ceiling/<c>`` for each mother and novel child: fraction of the
      true pixels of ``c`` that the base model assigns to its mother — an
      upper bound for any method that only splits the mother;
    * with ``boundary_width > 0``, ``boundary/iou/<c>``, ``boundary/miou``,
      ``boundary/miou_base`` and ``boundary/miou_novel``: IoU restricted to
      the pixels within ``boundary_width`` pixels of a ground-truth label
      change (trimap band, :func:`boundary_band`).

    Args:
        hierarchy: Class hierarchy of the task.
        class_names: Optional names used in the keys (default: indices).
        ignore_index: Target value excluded from every matrix.
        prefix: Prepended to every key (e.g. ``"test/"``).
        boundary_width: Half-width in pixels of the boundary band; 0
            disables the boundary metrics.
    """

    full_state_update = False

    def __init__(
        self,
        hierarchy: ClassHierarchy,
        class_names: Optional[Sequence[str]] = None,
        ignore_index: int = 255,
        prefix: str = "",
        boundary_width: int = 0,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        if boundary_width < 0:
            raise ValueError(f"boundary_width must be >= 0, got {boundary_width}.")
        self.boundary_width = int(boundary_width)
        self.hierarchy = hierarchy
        n, nb = hierarchy.num_classes, hierarchy.num_base_classes
        if class_names is not None and len(class_names) != n:
            raise ValueError(
                f"class_names must have {n} entries, got {len(class_names)}."
            )
        self.class_names: List[str] = (
            [str(c) for c in class_names] if class_names else [str(i) for i in range(n)]
        )
        self.ignore_index = ignore_index
        self.prefix = prefix
        self.add_state(
            "confmat", torch.zeros(n, n, dtype=torch.long), dist_reduce_fx="sum"
        )
        self.add_state(
            "base_confmat", torch.zeros(n, nb, dtype=torch.long), dist_reduce_fx="sum"
        )
        if self.boundary_width:
            self.add_state(
                "boundary_confmat",
                torch.zeros(n, n, dtype=torch.long),
                dist_reduce_fx="sum",
            )

    @staticmethod
    def _bincount(target: Tensor, pred: Tensor, rows: int, cols: int) -> Tensor:
        idx = target * cols + pred
        return torch.bincount(idx, minlength=rows * cols).view(rows, cols)

    def update(
        self, pred: Tensor, target: Tensor, base_pred: Optional[Tensor] = None
    ) -> None:
        """Accumulate one batch of label maps (same shape, integer classes)."""
        valid = target != self.ignore_index
        t = target[valid].long()
        n, nb = self.hierarchy.num_classes, self.hierarchy.num_base_classes
        self.confmat += self._bincount(t, pred[valid].long(), n, n)
        if base_pred is not None:
            self.base_confmat += self._bincount(t, base_pred[valid].long(), n, nb)
        if self.boundary_width:
            band = boundary_band(target, self.boundary_width, self.ignore_index)
            self.boundary_confmat += self._bincount(
                target[band].long(), pred[band].long(), n, n
            )

    @staticmethod
    def _iou(cm: Tensor) -> Tensor:
        tp = cm.diag()
        gt, pr = cm.sum(1), cm.sum(0)
        nan = torch.tensor(float("nan"), dtype=torch.float64, device=cm.device)
        return torch.where(gt > 0, tp / (gt + pr - tp).clamp(min=1), nan)

    def compute(self) -> Dict[str, Tensor]:
        """Return the metric dictionary described in the class docstring."""
        cm = self.confmat.double()
        tp = cm.diag()
        gt, pr = cm.sum(1), cm.sum(0)
        nan = torch.tensor(float("nan"), dtype=torch.float64, device=cm.device)
        iou = self._iou(cm)
        precision = torch.where(pr > 0, tp / pr.clamp(min=1), nan)
        recall = torch.where(gt > 0, tp / gt.clamp(min=1), nan)

        h = self.hierarchy
        out: Dict[str, Tensor] = {}
        for i, name in enumerate(self.class_names):
            out[f"iou/{name}"] = iou[i]
            out[f"precision/{name}"] = precision[i]
            out[f"recall/{name}"] = recall[i]
        base_idx = list(range(h.num_base_classes))
        out["miou"] = iou.nanmean()
        out["miou_base"] = iou[base_idx].nanmean()
        out["miou_novel"] = iou[h.novel_classes].nanmean()
        out["oem_score"] = 0.4 * out["miou_base"] + 0.6 * out["miou_novel"]
        if self.boundary_width:
            b_iou = self._iou(self.boundary_confmat.double())
            for i, name in enumerate(self.class_names):
                out[f"boundary/iou/{name}"] = b_iou[i]
            out["boundary/miou"] = b_iou.nanmean()
            out["boundary/miou_base"] = b_iou[base_idx].nanmean()
            out["boundary/miou_novel"] = b_iou[h.novel_classes].nanmean()

        bcm = self.base_confmat.double()
        if bcm.sum() > 0:
            untouched = [c for c in base_idx if c not in h.mothers]
            n_px = cm[untouched].sum()
            # advanced indexing with two equal lists picks the diagonal (hits)
            acc_after = cm[untouched, untouched].sum() / n_px
            acc_before = bcm[untouched, untouched].sum() / n_px
            out["locality"] = (
                acc_after / acc_before if n_px > 0 and acc_before > 0 else nan
            )
            for mother in h.mothers:
                for child in h.children_of(mother):
                    total = bcm[child].sum()
                    out[f"split_ceiling/{self.class_names[child]}"] = (
                        bcm[child, mother] / total if total > 0 else nan
                    )
        return {f"{self.prefix}{k}": v.float() for k, v in out.items()}
