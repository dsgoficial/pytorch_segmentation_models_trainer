# -*- coding: utf-8 -*-
"""Boundary label relaxation loss (Zhu et al., CVPR 2019)."""

from typing import Dict, Optional, Union

import torch
import torch.nn.functional as F

from pytorch_segmentation_models_trainer.custom_losses.base_loss import Loss


class BoundaryLabelRelaxationLoss(Loss):
    """Boundary label relaxation loss.

    Implements the loss from Zhu, Y. et al., "Improving Semantic Segmentation
    via Video Propagation and Label Relaxation," CVPR 2019 — only the
    label-relaxation component (video propagation is out of scope here).

    For each pixel, let :math:`\\mathcal{N}` be the set of classes present in
    the 3x3 neighbourhood of its hard label. Instead of maximizing the
    probability of the annotated class alone, the loss maximizes the
    probability mass assigned to the *union* of :math:`\\mathcal{N}`:

        L_boundary(i) = -log( sum_{C in N(i)} P(C | i) )

    When a pixel is "interior" (all 3x3 neighbours share its class,
    :math:`|\\mathcal{N}|=1`), the formula reduces exactly to standard
    per-pixel cross-entropy — this loss is a strict generalization of CE,
    not a separate loss added on top of it. There is no extra hyperparameter
    besides the (fixed, 3x3) neighbourhood window used to define
    :math:`\\mathcal{N}`.

    Neighbourhood membership is computed with a 3x3 max-pool over the
    per-class one-hot label map (no explicit per-pixel Python loop):
    ``membership[c] = 1`` at pixel i iff class c appears anywhere in the
    3x3 window centred at i.

    Pixels equal to ``ignore_index`` are excluded from the loss (their
    contribution to the mean) and are also excluded as a *neighbour* class
    source for adjacent valid pixels, so ignored regions cannot leak a
    spurious "relaxed" class into a valid pixel's neighbourhood set.

    Args:
        name: Loss identifier used by MultiLoss logging.
        num_classes: Number of segmentation classes C.
        ignore_index: Label value excluded from the loss. Defaults to 255,
            matching the convention used elsewhere in this project
            (e.g. ``WeightedDiceCrossEntropyLoss``,
            ``Model._soft_to_hard_masks``).
        mask_key: Key in a dict gt_batch holding the hard-label tensor.
        **kwargs: Accepted for Hydra / ConfigStore compatibility.

    Example YAML:
        loss:
          _target_: pytorch_segmentation_models_trainer.custom_losses.boundary_relaxation_loss.BoundaryLabelRelaxationLoss
          name: boundary_relaxation
          num_classes: 6
    """

    def __init__(
        self,
        name: str = "boundary_relaxation",
        num_classes: int = 2,
        ignore_index: Optional[int] = None,
        mask_key: str = "mask",
        **kwargs,
    ) -> None:
        super().__init__(name)
        self.num_classes = num_classes
        self.ignore_index = 255 if ignore_index is None else ignore_index
        self.mask_key = mask_key

    def compute(
        self,
        pred_batch: Union[torch.Tensor, Dict[str, torch.Tensor]],
        gt_batch: Union[torch.Tensor, Dict[str, torch.Tensor]],
    ) -> torch.Tensor:
        """Compute the boundary label relaxation loss.

        Args:
            pred_batch: Logits ``(B, C, H, W)``, or a dict with a ``"seg"``
                key (FrameField / compound-loss path).
            gt_batch: Hard-label tensor ``(B, H, W)`` int/long, or a dict
                with a ``mask_key`` entry holding that tensor.

        Returns:
            Scalar loss tensor (mean over non-ignored pixels).
        """
        if isinstance(pred_batch, dict):
            logits: torch.Tensor = pred_batch.get(
                "seg", next(iter(pred_batch.values()))
            )
        else:
            logits = pred_batch

        gt = gt_batch[self.mask_key] if isinstance(gt_batch, dict) else gt_batch
        gt = gt.long()

        valid = gt != self.ignore_index  # (B, H, W)
        gt_clamped = gt.clamp(min=0, max=self.num_classes - 1)

        one_hot = F.one_hot(gt_clamped, num_classes=self.num_classes)
        one_hot = one_hot.permute(0, 3, 1, 2).float()  # (B, C, H, W)
        # Ignored pixels contribute nothing as a neighbour class source.
        one_hot = one_hot * valid.unsqueeze(1).float()

        # membership[b, c, i] = 1 iff class c is present in the 3x3 window
        # around pixel i (includes the centre pixel itself).
        membership = F.max_pool2d(one_hot, kernel_size=3, stride=1, padding=1)

        probs = F.softmax(logits, dim=1)
        relaxed_mass = (probs * membership).sum(dim=1)  # (B, H, W)
        per_pixel_loss = -torch.log(relaxed_mass.clamp(min=1e-7))

        valid_f = valid.float()
        denom = valid_f.sum().clamp(min=1.0)
        return (per_pixel_loss * valid_f).sum() / denom
