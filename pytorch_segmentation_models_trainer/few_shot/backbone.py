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

from typing import Sequence

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class FrozenLinearHeadSegmenter(nn.Module):
    """Frozen base segmenter split into feature extractor and linear head.

    Wraps a segmentation_models_pytorch model whose
    segmentation_head[0] is a 1x1 convolution (UPerNet, DeepLabV3+,
    FPN...). features(x) is the decoder output (the input of that
    convolution, at the decoder resolution) and weight/bias are the
    base classifier, so linear(features(x), weight, bias) reproduces the
    base logits before upsampling. This is the feature space where DIaM,
    ClassTrans and prototype methods operate (ClassTrans' official code
    obtains the same tensor by replacing the smp segmentation head with
    Identity).

    The wrapped model is frozen and kept in eval mode even when the parent
    module is put in training mode (BatchNorm statistics never change).

    Args:
        model: smp segmentation model with a 1x1 convolutional head.

    Raises:
        ValueError: If the head is not a 1x1 convolution (e.g. smp U-Net,
            whose head is 3x3 and therefore not a per-pixel linear classifier).
    """

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        head = model.segmentation_head[0]
        if not isinstance(head, nn.Conv2d) or tuple(head.kernel_size) != (1, 1):
            raise ValueError(
                "GFSS feature-space methods need a 1x1 convolutional "
                f"segmentation head; got {head}."
            )
        self.model = model
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.model.eval()

    def train(self, mode: bool = True) -> "FrozenLinearHeadSegmenter":
        """Keep the frozen model in eval mode regardless of mode."""
        super().train(False)
        return self

    @property
    def weight(self) -> Tensor:
        """Base classifier weight, (num_base_classes, F) (detached copy)."""
        return self.model.segmentation_head[0].weight.detach().flatten(1).clone()

    @property
    def bias(self) -> Tensor:
        """Base classifier bias, (num_base_classes,) (detached copy)."""
        b = self.model.segmentation_head[0].bias
        if b is None:
            return self.weight.new_zeros(self.weight.shape[0])
        return b.detach().clone()

    @torch.no_grad()
    def features(self, x: Tensor) -> Tensor:
        """Decoder output (B, F, h, w) for images x."""
        return self.model.decoder(self.model.encoder(x))

    @staticmethod
    def linear(features: Tensor, weight: Tensor, bias: Tensor) -> Tensor:
        """Per-pixel linear classifier: (B, F, h, w) -> (B, C, h, w)."""
        return torch.einsum("bfhw,cf->bchw", features, weight) + bias.view(1, -1, 1, 1)

    def upsample(self, logits: Tensor, size: Sequence[int]) -> Tensor:
        """Resize logits to size as the smp head does (bilinear, aligned
        corners) and apply the head activation."""
        if tuple(logits.shape[-2:]) != tuple(size):
            logits = F.interpolate(
                logits, size=tuple(size), mode="bilinear", align_corners=True
            )
        return self.model.segmentation_head[2](logits)

    @torch.no_grad()
    def base_logits(self, x: Tensor) -> Tensor:
        """Logits of the frozen base model at the input resolution."""
        logits = self.linear(self.features(x), self.weight, self.bias)
        return self.upsample(logits, x.shape[-2:])
