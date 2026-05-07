# -*- coding: utf-8 -*-
"""
Generic Autoencoder combining SMP and Transformers.
"""

from typing import Optional, Union, Tuple, List
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import segmentation_models_pytorch as smp
except ImportError:
    smp = None

try:
    from transformers import AutoModel
except ImportError:
    AutoModel = None


class HuggingFaceEncoderAdapter(nn.Module):
    """
    Adapter for HuggingFace models to return spatial feature maps.
    """

    def __init__(self, hf_model_name: str, **kwargs):
        super().__init__()
        if AutoModel is None:
            raise ImportError("Please install transformers: pip install transformers")

        self.model = AutoModel.from_pretrained(hf_model_name, **kwargs)
        # Try to detect output channels from config
        self.out_channels = getattr(self.model.config, "hidden_size", 768)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outputs = self.model(x)
        # last_hidden_state is (B, N, C) for ViTs or (B, C, H, W) for ConvNets
        last_hidden_state = outputs.last_hidden_state

        if last_hidden_state.ndim == 3:
            # (B, N, C) -> (B, C, H, W)
            # This is a naive reshape, assuming square patches
            B, N, C = last_hidden_state.shape
            H = W = int(N**0.5)
            if H * W == N:
                last_hidden_state = last_hidden_state.permute(0, 2, 1).reshape(
                    B, C, H, W
                )
            else:
                # Handle cases with CLS token: (B, 1 + N, C)
                H = W = int((N - 1) ** 0.5)
                if H * W == N - 1:
                    last_hidden_state = (
                        last_hidden_state[:, 1:, :].permute(0, 2, 1).reshape(B, C, H, W)
                    )
                else:
                    raise ValueError(
                        f"Cannot reshape last_hidden_state of shape {last_hidden_state.shape} to 2D."
                    )

        return last_hidden_state


class GenericDecoder(nn.Module):
    """
    Simple decoder to reconstruct image from bottleneck.
    """

    def __init__(self, in_channels: int, out_channels: int, scale_factor: int = 32):
        super().__init__()
        # Simple stack of transposed convolutions or bilinear upsampling + conv
        self.decoder = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Upsample(
                scale_factor=scale_factor, mode="bilinear", align_corners=False
            ),
            nn.Conv2d(in_channels // 2, out_channels, kernel_size=3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(x)


class GenericAutoencoder(nn.Module):
    """
    Generic Autoencoder that can use SMP or HuggingFace encoders.

    Example YAML:
        model:
          _target_: pytorch_segmentation_models_trainer.custom_models.generic_autoencoder.GenericAutoencoder
          encoder_name: resnet18
          use_huggingface: false
          in_channels: 3
    """

    def __init__(
        self,
        encoder_name: str,
        use_huggingface: bool = False,
        in_channels: int = 3,
        latent_dim: Optional[int] = None,
        pretrained: bool = True,
        **kwargs,
    ):
        super().__init__()

        if use_huggingface:
            self.encoder = HuggingFaceEncoderAdapter(encoder_name, **kwargs)
            encoder_out_channels = self.encoder.out_channels
            # Assume 16x downsampling for HF if not specified
            self.scale_factor = 16
        else:
            if smp is None:
                raise ImportError("Please install segmentation-models-pytorch")
            self.encoder = smp.encoders.get_encoder(
                encoder_name,
                in_channels=in_channels,
                weights="imagenet" if pretrained else None,
            )
            encoder_out_channels = self.encoder.out_channels[-1]
            # Find total stride
            self.scale_factor = 2 ** (len(self.encoder.out_channels) - 1)

        self.latent_proj = nn.Identity()
        if latent_dim is not None:
            self.latent_proj = nn.Conv2d(
                encoder_out_channels, latent_dim, kernel_size=1
            )
            encoder_out_channels = latent_dim

        self.decoder = GenericDecoder(
            in_channels=encoder_out_channels,
            out_channels=in_channels,
            scale_factor=self.scale_factor,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if isinstance(self.encoder, HuggingFaceEncoderAdapter):
            bottleneck = self.encoder(x)
        else:
            features = self.encoder(x)
            bottleneck = features[-1]

        latent = self.latent_proj(bottleneck)
        reconstructed = self.decoder(latent)

        # Ensure output size matches input size exactly
        if reconstructed.shape[-2:] != x.shape[-2:]:
            reconstructed = F.interpolate(
                reconstructed, size=x.shape[-2:], mode="bilinear", align_corners=False
            )

        return reconstructed
