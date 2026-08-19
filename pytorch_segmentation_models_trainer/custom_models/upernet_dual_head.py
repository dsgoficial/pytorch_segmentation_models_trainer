"""
UPerNet with dual decoder heads for joint hard/soft label training.

Architecture:
    encoder (shared, e.g. ConvNeXtV2-Tiny)
        ├── decoder_A (UPerNet) → seg_head_A → logits_A  (hard labels)
        └── decoder_B (UPerNet) → seg_head_B → logits_B  (soft labels)

forward() returns logits from the primary head (or average of both)
and stores last_logits_A / last_logits_B for the loss function to access.

Follows the same attribute-storage pattern as UPerNetMoE (last_aux_loss).

Usage via Hydra YAML:
    model:
      _target_: pytorch_segmentation_models_trainer.custom_models.upernet_dual_head.UPerNetDualHead
      encoder_name: tu-convnextv2_tiny.fcmae_ft_in22k_in1k_384
      encoder_weights: imagenet
      in_channels: 3
      classes: 6
      inference_head: "average"   # "A", "B", or "average"
"""

import torch
import torch.nn as nn
from typing import Any, Callable, Dict, Optional, Union

from segmentation_models_pytorch.base import (
    SegmentationHead,
)
from segmentation_models_pytorch.base import initialization as init
from segmentation_models_pytorch.encoders import get_encoder
from segmentation_models_pytorch.decoders.upernet.decoder import UPerNetDecoder


class UPerNetDualHead(nn.Module):
    """UPerNet with two independent decoders sharing one encoder.

    Head A is supervised with hard labels (concordância masks).
    Head B is supervised with soft labels (Dawid-Skene posteriors).
    A consistency loss couples them during training.

    The model stores intermediate logits as attributes so the
    DualHeadKendallLoss can access them without changing the
    standard forward() → loss(pred, target) interface.

    Args:
        encoder_name: timm/smp encoder name.
        encoder_depth: Number of encoder stages (default 5).
        encoder_weights: Pretrained weights (default "imagenet").
        decoder_channels: Intermediate channel count per decoder (default 256).
        decoder_use_norm: Normalization for decoder conv blocks.
        in_channels: Input image channels (default 3).
        classes: Number of output segmentation classes (default 6).
        activation: Output activation (default None = raw logits).
        upsampling: Upsampling factor for segmentation head (default 4).
        inference_head: Which head to use at inference:
            "A" (hard-trained), "B" (soft-trained), or "average" (default).
    """

    def __init__(
        self,
        encoder_name: str = "tu-convnextv2_tiny.fcmae_ft_in22k_in1k_384",
        encoder_depth: int = 5,
        encoder_weights: Optional[str] = "imagenet",
        decoder_channels: int = 256,
        decoder_use_norm: Union[bool, str, Dict[str, Any]] = "batchnorm",
        in_channels: int = 3,
        classes: int = 6,
        activation: Optional[Union[str, Callable]] = None,
        upsampling: int = 4,
        inference_head: str = "average",
        **kwargs: Any,
    ):
        super().__init__()

        self.num_classes = classes
        if inference_head not in ("A", "B", "average"):
            raise ValueError(
                f"inference_head must be 'A', 'B', or 'average', got '{inference_head}'"
            )
        self.inference_head = inference_head

        # Shared encoder
        self.encoder = get_encoder(
            encoder_name,
            in_channels=in_channels,
            depth=encoder_depth,
            weights=encoder_weights,
            **{k: v for k, v in kwargs.items() if v is not None},
        )

        # Two independent decoders
        decoder_kwargs = dict(
            encoder_channels=self.encoder.out_channels,
            encoder_depth=encoder_depth,
            decoder_channels=decoder_channels,
            use_norm=decoder_use_norm,
        )
        self.decoder_A = UPerNetDecoder(**decoder_kwargs)
        self.decoder_B = UPerNetDecoder(**decoder_kwargs)

        # Two independent segmentation heads
        head_kwargs = dict(
            in_channels=decoder_channels,
            out_channels=classes,
            activation=activation,
            kernel_size=1,
            upsampling=upsampling,
        )
        self.seg_head_A = SegmentationHead(**head_kwargs)
        self.seg_head_B = SegmentationHead(**head_kwargs)

        # Attributes for loss to read (same pattern as UPerNetMoE.last_aux_loss)
        self.last_logits_A: Optional[torch.Tensor] = None
        self.last_logits_B: Optional[torch.Tensor] = None
        # Hard mask stored by _shared_step for the loss
        self.last_hard_mask: Optional[torch.Tensor] = None

        self.name = f"upernet-dual-{encoder_name}"
        self.classification_head = None
        self._initialize()

    def _initialize(self):
        """Initialize decoder and head weights."""
        init.initialize_decoder(self.decoder_A)
        init.initialize_decoder(self.decoder_B)
        init.initialize_head(self.seg_head_A)
        init.initialize_head(self.seg_head_B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through shared encoder and both decoders.

        Stores logits_A and logits_B as attributes for the loss function.
        Returns the inference output (configurable via inference_head).

        Args:
            x: Input image (B, C, H, W)

        Returns:
            logits: (B, classes, H, W) — from head A, B, or their average.
        """
        features = self.encoder(x)

        # Decoder A (hard labels head)
        decoded_A = self.decoder_A(features)
        logits_A = self.seg_head_A(decoded_A)

        # Decoder B (soft labels head)
        decoded_B = self.decoder_B(features)
        logits_B = self.seg_head_B(decoded_B)

        # Store for loss function access
        self.last_logits_A = logits_A
        self.last_logits_B = logits_B

        # Return based on inference mode
        if self.inference_head == "A":
            return logits_A
        elif self.inference_head == "B":
            return logits_B
        else:  # "average"
            return (logits_A + logits_B) / 2.0
