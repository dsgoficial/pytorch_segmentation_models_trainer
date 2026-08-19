# -*- coding: utf-8 -*-
"""
/***************************************************************************
 pytorch_segmentation_models_trainer
                              -------------------
        begin                : 2024-01-01
        copyright            : (C) 2024 by Philipe Borba - Cartographic Engineer
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

"""
TerraTorch geospatial foundation model wrappers
===============================================

Integrates TerraTorch foundation model encoders (Prithvi, Clay, SatMAE, etc.)
with SMP-style decoders so they produce plain ``(B, num_classes, H, W)``
tensors compatible with this framework's training loop.

TerraTorch models are typically ViT-based and operate on multi-temporal
satellite imagery.  This wrapper handles:

* Single-temporal inputs ``(B, C, H, W)`` by auto-unsqueezing to
  ``(B, C, 1, H, W)``.
* Multi-temporal inputs ``(B, C, T, H, W)`` passed through unchanged.
* Dense prediction via an SMP ``UperNet``-style head or a simple linear
  patch-projection head.

Requires ``terratorch`` to be installed::

    pip install terratorch
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class _LinearSegHead(nn.Module):
    """Simple linear head: project patch tokens → dense mask."""

    def __init__(self, embed_dim: int, num_classes: int, patch_size: int) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(embed_dim, num_classes, kernel_size=1)

    def forward(self, tokens: torch.Tensor, hw: Tuple[int, int]) -> torch.Tensor:
        H, W = hw
        B, N, C = tokens.shape
        x = tokens.transpose(1, 2).reshape(B, C, H, W)
        x = self.proj(x)
        return F.interpolate(
            x, scale_factor=self.patch_size, mode="bilinear", align_corners=False
        )


class TerraTorchSegmentationWrapper(nn.Module):
    """
    Combines a **TerraTorch** foundation model encoder with a lightweight
    segmentation head to produce a full segmentation model.

    Args:
        backbone_name: TerraTorch model name, e.g. ``"prithvi_100M"``,
            ``"prithvi_300M"``, ``"clay_small"``.
        num_classes: Number of segmentation output channels.
        in_channels: Number of spectral bands in the input image.
        num_frames: Number of time steps.  Use ``1`` for single-temporal
            inputs.  TerraTorch Prithvi models default to ``3`` frames but
            can be configured for ``1``.
        img_size: Spatial patch size expected by the backbone (e.g. 224).
        head_type: ``"linear"`` for a simple linear projection head.
            ``"fpn"`` attaches an SMP FPN decoder (requires multi-scale
            feature extraction support in the backbone).
        pretrained_path: Optional path / URL to pretrained backbone weights.
            If ``None``, weights are random (useful for testing).
        backbone_kwargs: Extra keyword arguments forwarded to the TerraTorch
            model constructor.

    Example YAML config::

        model:
          _target_: pytorch_segmentation_models_trainer.custom_models.terratorch_models.TerraTorchSegmentationWrapper
          backbone_name: prithvi_100M
          num_classes: 2
          in_channels: 6
          num_frames: 1
          img_size: 224
          head_type: linear
          pretrained_path: /weights/prithvi_100M.pt
    """

    def __init__(
        self,
        backbone_name: str,
        num_classes: int,
        in_channels: int = 6,
        num_frames: int = 1,
        img_size: int = 224,
        head_type: str = "linear",
        pretrained_path: Optional[str] = None,
        **backbone_kwargs,
    ) -> None:
        super().__init__()

        try:
            import terratorch  # noqa: F401
        except ImportError as e:
            raise ImportError(
                "The 'terratorch' package is required to use "
                "TerraTorchSegmentationWrapper.  "
                "Install it with:  pip install terratorch"
            ) from e

        from terratorch.models.backbones.prithvi_model_factory import (
            PrithviModelFactory,
        )

        factory = PrithviModelFactory()
        self.backbone = factory.build_model(
            task="segmentation",
            backbone=backbone_name,
            backbone_pretrained_cfg_overlay=(
                {
                    "file": pretrained_path,
                }
                if pretrained_path
                else None
            ),
            in_channels=in_channels,
            num_frames=num_frames,
            img_size=img_size,
            **backbone_kwargs,
        )

        self.num_frames = num_frames
        self.num_classes = num_classes
        embed_dim: int = getattr(self.backbone, "embed_dim", 768)
        patch_size: int = getattr(self.backbone, "patch_size", 16)

        head_type = head_type.lower()
        if head_type == "linear":
            self.head = _LinearSegHead(
                embed_dim=embed_dim,
                num_classes=num_classes,
                patch_size=patch_size,
            )
        elif head_type == "fpn":
            try:
                import segmentation_models_pytorch as smp
                from segmentation_models_pytorch.base import SegmentationHead
            except ImportError as exc:
                raise ImportError(
                    "segmentation-models-pytorch is required for head_type='fpn'."
                ) from exc
            # FPN decoder expects encoder_channels list
            encoder_channels = [0, embed_dim // 4, embed_dim // 2, embed_dim, embed_dim]
            self.decoder = smp.decoders.fpn.decoder.FPNDecoder(
                encoder_channels=encoder_channels,
                encoder_depth=4,
                pyramid_channels=256,
                segmentation_channels=128,
                dropout=0.2,
                merge_policy="add",
            )
            self.head = SegmentationHead(
                in_channels=128,
                out_channels=num_classes,
                activation=None,
                kernel_size=3,
                upsampling=patch_size,
            )
        else:
            raise ValueError(
                f"Unsupported head_type='{head_type}'. Choose 'linear' or 'fpn'."
            )

    def _prepare_input(self, x: torch.Tensor) -> torch.Tensor:
        """Accept ``(B, C, H, W)`` and expand to ``(B, C, T, H, W)``."""
        if x.ndim == 4:
            x = x.unsqueeze(2)  # (B, C, 1, H, W)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._prepare_input(x)
        B, C, T, H, W = x.shape

        features = self.backbone(x)

        # TerraTorch returns dict or tensor depending on version/config
        if isinstance(features, dict):
            tokens = features.get("last_hidden_state", features.get("features"))
        elif isinstance(features, (list, tuple)):
            tokens = features[-1]
        else:
            tokens = features

        if tokens.ndim == 3:
            # (B, N, embed_dim) patch tokens → pass to linear head
            h_patches = H // getattr(self.backbone, "patch_size", 16)
            w_patches = W // getattr(self.backbone, "patch_size", 16)
            return self.head(tokens, (h_patches, w_patches))
        else:
            # (B, C, H', W') feature map → pass directly to head
            return self.head(tokens)
