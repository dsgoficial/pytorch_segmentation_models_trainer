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
    """Simple convolutional decoder that reconstructs an image from a bottleneck tensor.

    Args:
        in_channels: Number of input channels from the bottleneck.
        out_channels: Number of output channels (usually the input image channels).
        scale_factor: Spatial upsampling factor applied via bilinear interpolation.
            Set to ``0`` to skip upsampling (e.g. when the encoder and input
            share the same spatial size).
        output_activation: Optional final activation applied to the logit output.
            Supported values: ``None`` (no activation, unbounded output),
            ``"sigmoid"`` (output in ``[0, 1]``, use with targets in ``[0, 1]``),
            ``"tanh"`` (output in ``[-1, 1]``, use with targets in ``[-1, 1]``).

    Raises:
        ValueError: If ``output_activation`` is not one of the supported values.
    """

    _VALID_ACTIVATIONS = (None, "sigmoid", "tanh")

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        scale_factor: int = 32,
        output_activation: Optional[str] = None,
    ):
        super().__init__()
        if output_activation not in self._VALID_ACTIVATIONS:
            raise ValueError(
                f"output_activation must be one of {self._VALID_ACTIVATIONS}; "
                f"got {output_activation!r}"
            )
        self.scale_factor = scale_factor
        self.output_activation = output_activation
        self.conv1 = nn.Conv2d(in_channels, in_channels // 2, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(in_channels // 2, out_channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.conv1(x))
        if self.scale_factor > 0:
            x = F.interpolate(
                x,
                scale_factor=float(self.scale_factor),
                mode="bilinear",
                align_corners=False,
            )
        x = self.conv2(x)
        if self.output_activation == "sigmoid":
            return torch.sigmoid(x)
        if self.output_activation == "tanh":
            return torch.tanh(x)
        return x


class GenericAutoencoder(nn.Module):
    """Generic Autoencoder that can use SMP or HuggingFace encoders.

    Args:
        encoder_name: SMP encoder name or HuggingFace model identifier.
        use_huggingface: When ``True``, builds a HuggingFace encoder adapter.
        in_channels: Number of input and reconstruction channels.
        latent_dim: Optional bottleneck channel reduction via a 1×1 conv.
        pretrained: Whether to request ImageNet weights for SMP encoders.
        output_activation: Optional final activation on the decoder output.
            Use ``"sigmoid"`` when targets are in ``[0, 1]`` (e.g. uint8/255),
            ``"tanh"`` for ``[-1, 1]``, or ``None`` for unbounded output.
        use_progressive_decoder: When ``True``, uses ``ProgressiveDecoder``
            instead of ``GenericDecoder``.  Recommended when
            ``scale_factor >= 4`` to avoid single-step bilinear artefacts.
        base_channels: Starting channel count for ``ProgressiveDecoder``.
            Ignored when ``use_progressive_decoder=False``.
        min_channels: Minimum channel count for ``ProgressiveDecoder`` stages.
            Ignored when ``use_progressive_decoder=False``.
        **kwargs: Extra arguments forwarded to the HuggingFace adapter.

    Example YAML:
        model:
          _target_: pytorch_segmentation_models_trainer.custom_models.generic_autoencoder.GenericAutoencoder
          encoder_name: resnet18
          use_huggingface: false
          in_channels: 3
          output_activation: sigmoid
          use_progressive_decoder: true
    """

    def __init__(
        self,
        encoder_name: str,
        use_huggingface: bool = False,
        in_channels: int = 3,
        latent_dim: Optional[int] = None,
        pretrained: bool = True,
        output_activation: Optional[str] = None,
        use_progressive_decoder: bool = False,
        base_channels: int = 128,
        min_channels: int = 32,
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

        if use_progressive_decoder:
            self.decoder = ProgressiveDecoder(
                in_channels=encoder_out_channels,
                out_channels=in_channels,
                scale_factor=self.scale_factor,
                base_channels=base_channels,
                min_channels=min_channels,
                output_activation=output_activation,
            )
        else:
            self.decoder = GenericDecoder(
                in_channels=encoder_out_channels,
                out_channels=in_channels,
                scale_factor=self.scale_factor,
                output_activation=output_activation,
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


class ProgressiveDecoder(nn.Module):
    """Convolutional decoder with progressive upsampling.

    Replaces ``GenericDecoder`` for cases where a single bilinear interpolation
    is insufficient — typically when ``scale_factor >= 4`` and the bottleneck
    has few channels (e.g. ``latent_dim=4`` with ``scale_factor=8``).

    Each upsampling stage doubles the spatial resolution and applies two
    convolutional layers with ReLU activations.  This matches the standard
    decoder design used in VAEs for image reconstruction (e.g. the decoder
    of Stable Diffusion's KL-regularized autoencoder).

    Args:
        in_channels: Number of bottleneck channels (``latent_dim``).
        out_channels: Number of output channels (input image channels).
        scale_factor: Total spatial upsampling factor.  Must be a power of 2.
            Each stage doubles the resolution, so ``log2(scale_factor)``
            stages are created.
        base_channels: Number of channels in the first decoder stage.
            Halved at each subsequent stage down to ``min_channels``.
        min_channels: Minimum number of channels in any decoder stage.
        output_activation: Final activation.  ``None``, ``"sigmoid"``,
            or ``"tanh"``.

    Raises:
        ValueError: If ``scale_factor`` is not a power of 2, or if
            ``output_activation`` is not one of the supported values.

    Example — replacing GenericDecoder in GenericVariationalAutoencoder:

    .. code-block:: python

        self.decoder = ProgressiveDecoder(
            in_channels=latent_channels,
            out_channels=in_channels,
            scale_factor=self.scale_factor,   # 8 for encoder_depth=3
            base_channels=128,
            output_activation=output_activation,
        )
    """

    _VALID_ACTIVATIONS = (None, "sigmoid", "tanh")

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        scale_factor: int = 8,
        base_channels: int = 128,
        min_channels: int = 32,
        output_activation: Optional[str] = None,
    ):
        super().__init__()

        if output_activation not in self._VALID_ACTIVATIONS:
            raise ValueError(
                f"output_activation must be one of {self._VALID_ACTIVATIONS}; "
                f"got {output_activation!r}"
            )

        import math

        n_stages = int(math.log2(scale_factor))
        if 2**n_stages != scale_factor:
            raise ValueError(f"scale_factor must be a power of 2; got {scale_factor}.")

        self.scale_factor = scale_factor
        self.output_activation = output_activation

        # ── Input projection: bottleneck → base_channels ──────────────────
        self.input_proj = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

        # ── Progressive upsampling stages ─────────────────────────────────
        stages = []
        ch_in = base_channels
        for i in range(n_stages):
            ch_out = max(ch_in // 2, min_channels)
            stages.append(
                nn.Sequential(
                    # Bilinear upsample 2× then refine with two convolutions.
                    nn.Upsample(scale_factor=2.0, mode="bilinear", align_corners=False),
                    nn.Conv2d(ch_in, ch_out, kernel_size=3, padding=1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(ch_out, ch_out, kernel_size=3, padding=1),
                    nn.ReLU(inplace=True),
                )
            )
            ch_in = ch_out

        self.stages = nn.ModuleList(stages)

        # ── Output projection: last stage channels → image channels ───────
        self.output_proj = nn.Conv2d(ch_in, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_proj(x)
        for stage in self.stages:
            x = stage(x)
        x = self.output_proj(x)

        if self.output_activation == "sigmoid":
            return torch.sigmoid(x)
        if self.output_activation == "tanh":
            return torch.tanh(x)
        return x
