# -*- coding: utf-8 -*-
"""Variational autoencoder architectures for image reconstruction."""

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from pytorch_segmentation_models_trainer.custom_models.generic_autoencoder import (
    GenericDecoder,
    HuggingFaceEncoderAdapter,
    smp,
)


@dataclass
class VariationalAutoencoderOutput:
    """Output produced by variational autoencoders.

    Args:
        reconstruction: Reconstructed image tensor with shape ``(B, C, H, W)``.
        mu: Mean tensor of the approximate posterior ``q(z|x)``.
        logvar: Log-variance tensor of the approximate posterior ``q(z|x)``.
        z: Sampled latent tensor produced with the reparameterization trick.

    Returns:
        Dataclass carrying all tensors needed by ``VariationalAutoencoderLoss``.
    """

    reconstruction: torch.Tensor
    mu: torch.Tensor
    logvar: torch.Tensor
    z: torch.Tensor


class GenericVariationalAutoencoder(nn.Module):
    """Generic VAE using SMP or HuggingFace encoders and a convolutional decoder.

    Args:
        encoder_name: SMP encoder name or HuggingFace model identifier.
        use_huggingface: When ``True``, builds a HuggingFace encoder adapter.
        in_channels: Number of input and reconstruction channels.
        latent_dim: Number of latent channels. Defaults to encoder output channels.
        pretrained: Whether to request ImageNet weights for SMP encoders.
        **kwargs: Extra arguments forwarded to the HuggingFace adapter.

    Returns:
        ``VariationalAutoencoderOutput`` from ``forward``.

    Example YAML:
        model:
          _target_: pytorch_segmentation_models_trainer.custom_models.variational_autoencoder.GenericVariationalAutoencoder
          encoder_name: resnet18
          use_huggingface: false
          in_channels: 3
          latent_dim: 128
          pretrained: false
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
        self.use_huggingface = use_huggingface

        if use_huggingface:
            self.encoder = HuggingFaceEncoderAdapter(encoder_name, **kwargs)
            encoder_out_channels = self.encoder.out_channels
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
            self.scale_factor = 2 ** (len(self.encoder.out_channels) - 1)

        latent_channels = encoder_out_channels if latent_dim is None else latent_dim
        if latent_channels < 2:
            raise ValueError("latent_dim must be at least 2 when provided")
        self.mu_proj = nn.Conv2d(encoder_out_channels, latent_channels, kernel_size=1)
        self.logvar_proj = nn.Conv2d(
            encoder_out_channels, latent_channels, kernel_size=1
        )
        self.decoder = GenericDecoder(
            in_channels=latent_channels,
            out_channels=in_channels,
            scale_factor=self.scale_factor,
        )

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode an image batch into posterior mean and log-variance tensors.

        Args:
            x: Input image tensor with shape ``(B, C, H, W)``.

        Returns:
            Tuple ``(mu, logvar)`` describing ``q(z|x)``.
        """
        if self.use_huggingface:
            bottleneck = self.encoder(x)
        else:
            features = self.encoder(x)
            bottleneck = features[-1]
        return self.mu_proj(bottleneck), self.logvar_proj(bottleneck)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Sample latent tensor with the reparameterization trick.

        Args:
            mu: Posterior mean tensor.
            logvar: Posterior log-variance tensor.

        Returns:
            Differentiable sample ``z = mu + eps * exp(0.5 * logvar)``.
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor, output_size: tuple[int, int]) -> torch.Tensor:
        """Decode a latent tensor back to image space.

        Args:
            z: Latent tensor sampled from ``q(z|x)``.
            output_size: Target spatial size ``(H, W)``.

        Returns:
            Reconstructed image tensor with the requested spatial size.
        """
        reconstruction = self.decoder(z)
        if reconstruction.shape[-2:] != output_size:
            reconstruction = F.interpolate(
                reconstruction,
                size=output_size,
                mode="bilinear",
                align_corners=False,
            )
        return reconstruction

    def forward(self, x: torch.Tensor) -> VariationalAutoencoderOutput:
        """Run the VAE forward pass.

        Args:
            x: Input image tensor with shape ``(B, C, H, W)``.

        Returns:
            ``VariationalAutoencoderOutput`` with reconstruction, posterior
            parameters, and sampled latent tensor.
        """
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        reconstruction = self.decode(z, x.shape[-2:])
        return VariationalAutoencoderOutput(reconstruction, mu, logvar, z)
