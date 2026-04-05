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
timm-based segmentation models
================================

**Recommended path (zero new code):**
Most timm backbones are already available inside
``segmentation_models_pytorch`` via the ``"tu-"`` encoder prefix::

    model:
      _target_: segmentation_models_pytorch.Unet
      encoder_name: tu-convnext_base   # any timm model name prefixed with tu-
      encoder_weights: imagenet
      in_channels: 3
      classes: 2

Use :class:`TimmEncoderWithSMPDecoder` only when you need a timm backbone
that is *not* registered as an SMP encoder, or when you want fine-grained
control over the decoder architecture.
"""

from typing import List, Optional

import torch
import torch.nn as nn


class TimmEncoderWithSMPDecoder(nn.Module):
    """
    Combines a **timm** feature-extraction backbone with an
    **SMP** decoder to produce a full segmentation model.

    The timm encoder is created with ``features_only=True`` so it returns
    multi-scale feature maps.  The feature channel sizes are read from
    ``encoder.feature_info`` and forwarded to the chosen SMP decoder.

    Supported decoders (``decoder_type``):
        * ``"unet"``        – :class:`segmentation_models_pytorch.UnetDecoder`
        * ``"fpn"``         – :class:`segmentation_models_pytorch.FPNDecoder`
        * ``"pan"``         – :class:`segmentation_models_pytorch.PANDecoder`
        * ``"pspnet"``      – :class:`segmentation_models_pytorch.PSPDecoder`
        * ``"deeplabv3"``   – :class:`segmentation_models_pytorch.DeepLabV3Decoder`
        * ``"deeplabv3plus"`` – :class:`segmentation_models_pytorch.DeepLabV3PlusDecoder`

    Args:
        timm_model_name: Any timm model name, e.g. ``"convnext_base"``,
            ``"vit_base_patch16_224"``.
        num_classes: Number of segmentation output channels.
        decoder_type: SMP decoder to attach.  Defaults to ``"unet"``.
        in_channels: Number of input image channels.  Defaults to 3.
        pretrained: Load timm ImageNet pretrained weights.  Defaults to
            ``True``.
        decoder_channels: Output channel sizes for each decoder stage (only
            used by UNet decoder).  Defaults to ``(256, 128, 64, 32, 16)``.
        upsampling: Final upsampling factor applied by the segmentation head.

    Example YAML config::

        model:
          _target_: pytorch_segmentation_models_trainer.custom_models.timm_models.TimmEncoderWithSMPDecoder
          timm_model_name: convnext_base
          num_classes: 2
          decoder_type: fpn
          pretrained: true
    """

    def __init__(
        self,
        timm_model_name: str,
        num_classes: int,
        decoder_type: str = "unet",
        in_channels: int = 3,
        pretrained: bool = True,
        decoder_channels: tuple = (256, 128, 64, 32, 16),
        upsampling: int = 4,
    ) -> None:
        super().__init__()

        try:
            import timm
        except ImportError as e:
            raise ImportError(
                "The 'timm' package is required to use TimmEncoderWithSMPDecoder.  "
                "Install it with:  pip install timm"
            ) from e

        try:
            import segmentation_models_pytorch as smp
            from segmentation_models_pytorch.base import SegmentationHead
        except ImportError as e:
            raise ImportError(
                "The 'segmentation_models_pytorch' package is required.  "
                "Install it with:  pip install segmentation-models-pytorch"
            ) from e

        # Build encoder
        self.encoder = timm.create_model(
            timm_model_name,
            pretrained=pretrained,
            features_only=True,
            in_chans=in_channels,
        )
        encoder_channels: List[int] = [
            fi["num_chs"] for fi in self.encoder.feature_info
        ]
        # Prepend 0 to follow SMP convention (first entry = stem/input)
        encoder_channels_smp = [0] + encoder_channels

        decoder_type = decoder_type.lower()

        if decoder_type == "unet":
            self.decoder = smp.decoders.unet.decoder.UnetDecoder(
                encoder_channels=encoder_channels_smp,
                decoder_channels=decoder_channels[: len(encoder_channels)],
                n_blocks=len(encoder_channels),
                use_batchnorm=True,
                attention_type=None,
            )
            head_in_channels = decoder_channels[len(encoder_channels) - 1]
        elif decoder_type == "fpn":
            self.decoder = smp.decoders.fpn.decoder.FPNDecoder(
                encoder_channels=encoder_channels_smp,
                encoder_depth=len(encoder_channels),
                pyramid_channels=256,
                segmentation_channels=128,
                dropout=0.2,
                merge_policy="add",
            )
            head_in_channels = 128
        elif decoder_type == "pan":
            self.decoder = smp.decoders.pan.decoder.PANDecoder(
                encoder_channels=encoder_channels_smp,
                encoder_depth=len(encoder_channels),
                upscale_factor=1,
            )
            head_in_channels = 32
        else:
            raise ValueError(
                f"Unsupported decoder_type='{decoder_type}'. "
                f"Choose from: 'unet', 'fpn', 'pan'."
            )

        self.segmentation_head = SegmentationHead(
            in_channels=head_in_channels,
            out_channels=num_classes,
            activation=None,
            kernel_size=3,
            upsampling=upsampling,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.encoder(x)
        # SMP decoders expect the feature list prepended with the input tensor
        decoder_output = self.decoder(*([x] + features))
        return self.segmentation_head(decoder_output)
