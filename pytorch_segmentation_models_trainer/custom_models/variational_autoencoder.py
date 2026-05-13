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

    The spatial resolution of the latent space is controlled by two mutually
    exclusive parameters:

    * For SMP encoders: ``encoder_depth`` sets how many downsampling stages
      the encoder computes.  The resulting scale factor is ``2 ** encoder_depth``
      and the latent spatial size is ``H / 2**encoder_depth × W / 2**encoder_depth``.
      Common choices for 224 × 224 inputs:

      ============  ===========  =============
      encoder_depth  scale_factor  latent size
      ============  ===========  =============
      3              8             28 × 28
      4              16            14 × 14
      5 (default)    32             7 × 7
      ============  ===========  =============

    * For HuggingFace encoders: ``feature_level`` selects which index in the
      list of hidden states the encoder returns (``-1`` = deepest / most
      down-sampled, ``-2`` = one level shallower, etc.).  The scale factor is
      inferred from the actual spatial size of the selected feature map at the
      first forward pass and cached for the decoder.

    Args:
        encoder_name: SMP encoder name or HuggingFace model identifier.
        use_huggingface: When ``True``, builds a HuggingFace encoder adapter.
        in_channels: Number of input and reconstruction channels.
        latent_dim: Number of latent channels.  Defaults to the channel count
            of the selected encoder feature level.
        pretrained: Whether to request ImageNet weights for SMP encoders.
        encoder_depth: Number of downsampling stages for SMP encoders.
            Must be ``None`` when ``use_huggingface=True``.
            Defaults to ``5`` (full encoder depth), matching the original
            behaviour.
        feature_level: Index into the HuggingFace hidden-state list that
            selects the feature map used as the VAE bottleneck.  Negative
            indices count from the deepest level (``-1``).  Ignored when
            ``use_huggingface=False``.  Defaults to ``-1`` (deepest level).
        **kwargs: Extra arguments forwarded to the HuggingFace adapter.

    Raises:
        ValueError: If both ``encoder_depth`` and ``use_huggingface=True`` are
            set, or if ``encoder_depth`` is outside ``[1, 5]``.
        ImportError: If ``segmentation-models-pytorch`` is not installed and
            ``use_huggingface=False``.

    Returns:
        ``VariationalAutoencoderOutput`` from ``forward``.

    Example YAML — DDOQ-compatible latent (28 × 28 × 4 for 224 × 224 input):

    .. code-block:: yaml

        model:
          _target_: pytorch_segmentation_models_trainer.custom_models.variational_autoencoder.GenericVariationalAutoencoder
          encoder_name: resnet18
          use_huggingface: false
          in_channels: 3
          latent_dim: 4
          pretrained: false
          encoder_depth: 3        # 2^3 = 8x downsampling → 28×28 latent

    Example YAML — full-depth encoder (original behaviour):

    .. code-block:: yaml

        model:
          _target_: pytorch_segmentation_models_trainer.custom_models.variational_autoencoder.GenericVariationalAutoencoder
          encoder_name: resnet18
          use_huggingface: false
          in_channels: 3
          latent_dim: 128
          pretrained: false
          encoder_depth: 5        # 2^5 = 32x downsampling → 7×7 latent

    Example YAML — HuggingFace encoder, second-deepest feature level:

    .. code-block:: yaml

        model:
          _target_: pytorch_segmentation_models_trainer.custom_models.variational_autoencoder.GenericVariationalAutoencoder
          encoder_name: facebook/dinov2-base
          use_huggingface: true
          in_channels: 3
          latent_dim: 64
          feature_level: -2
    """

    def __init__(
        self,
        encoder_name: str,
        use_huggingface: bool = False,
        in_channels: int = 3,
        latent_dim: Optional[int] = None,
        pretrained: bool = True,
        encoder_depth: Optional[int] = None,
        feature_level: Optional[int] = None,
        **kwargs,
    ):
        super().__init__()

        self.use_huggingface = use_huggingface

        # ── Encoder ───────────────────────────────────────────────────────
        if use_huggingface:
            if encoder_depth is not None:
                raise ValueError(
                    "'encoder_depth' is only valid for SMP encoders. "
                    "Use 'feature_level' to control the latent resolution "
                    "for HuggingFace encoders."
                )
            self.feature_level = feature_level if feature_level is not None else -1
            self.encoder = HuggingFaceEncoderAdapter(encoder_name, **kwargs)
            encoder_out_channels = self.encoder.out_channels
            # scale_factor for HuggingFace is inferred lazily on first forward
            # pass because it depends on the actual input resolution.
            self._hf_scale_factor: Optional[int] = None

        else:
            if feature_level is not None:
                raise ValueError(
                    "'feature_level' is only valid for HuggingFace encoders. "
                    "Use 'encoder_depth' to control the latent resolution "
                    "for SMP encoders."
                )
            if smp is None:
                raise ImportError("Please install segmentation-models-pytorch")

            depth = encoder_depth if encoder_depth is not None else 5
            if not (1 <= depth <= 5):
                raise ValueError(
                    f"'encoder_depth' must be between 1 and 5, got {depth}."
                )

            self.encoder = smp.encoders.get_encoder(
                encoder_name,
                in_channels=in_channels,
                depth=depth,
                weights="imagenet" if pretrained else None,
            )
            # out_channels has shape (depth + 1,): first element is the
            # input channel count, the rest are the encoder stage outputs.
            encoder_out_channels = self.encoder.out_channels[-1]
            self.scale_factor = 2**depth

        # ── Latent projections ────────────────────────────────────────────
        latent_channels = encoder_out_channels if latent_dim is None else latent_dim
        if latent_channels < 2:
            raise ValueError("'latent_dim' must be at least 2 when provided.")

        self.mu_proj = nn.Conv2d(encoder_out_channels, latent_channels, kernel_size=1)
        self.logvar_proj = nn.Conv2d(
            encoder_out_channels, latent_channels, kernel_size=1
        )

        # ── Decoder ───────────────────────────────────────────────────────
        # For HuggingFace encoders the scale_factor is set during the first
        # forward pass; pass a placeholder of 0 and update it then.
        self.decoder = GenericDecoder(
            in_channels=latent_channels,
            out_channels=in_channels,
            scale_factor=0 if use_huggingface else self.scale_factor,
        )

    # ── Private helpers ───────────────────────────────────────────────────

    def _get_smp_bottleneck(self, x: torch.Tensor) -> torch.Tensor:
        """Return the deepest feature map produced by the SMP encoder."""
        features = self.encoder(x)
        # features[0] is the input; features[-1] is the deepest stage output.
        return features[-1]

    def _get_hf_bottleneck(self, x: torch.Tensor) -> tuple[torch.Tensor, int]:
        """Return the selected HuggingFace feature map and its scale factor."""
        hidden_states = self.encoder(x)  # list of feature maps
        bottleneck = hidden_states[self.feature_level]

        # Infer and cache scale_factor from the actual spatial dimensions.
        h_in, w_in = x.shape[-2], x.shape[-1]
        h_feat, w_feat = bottleneck.shape[-2], bottleneck.shape[-1]
        scale_h = h_in // h_feat
        scale_w = w_in // w_feat
        if scale_h != scale_w:
            raise RuntimeError(
                f"Non-square scale factor ({scale_h} × {scale_w}) is not "
                "supported. Check the selected 'feature_level'."
            )
        return bottleneck, scale_h

    # ── Public API ────────────────────────────────────────────────────────

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode an image batch into posterior mean and log-variance tensors.

        Args:
            x: Input image tensor with shape ``(B, C, H, W)``.

        Returns:
            Tuple ``(mu, logvar)`` describing ``q(z|x)``.
        """
        if self.use_huggingface:
            bottleneck, scale_factor = self._get_hf_bottleneck(x)
            if self._hf_scale_factor is None:
                # First call: wire the scale_factor into the decoder.
                self._hf_scale_factor = scale_factor
                self.decoder.scale_factor = scale_factor
        else:
            bottleneck = self._get_smp_bottleneck(x)

        return self.mu_proj(bottleneck), self.logvar_proj(bottleneck)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Sample a latent tensor with the reparameterization trick.

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
