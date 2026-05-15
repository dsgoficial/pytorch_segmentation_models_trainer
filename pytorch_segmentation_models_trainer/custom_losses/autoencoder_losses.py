# -*- coding: utf-8 -*-
"""Loss functions for autoencoder training."""

from typing import Dict, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from kornia.losses import MS_SSIMLoss

from pytorch_segmentation_models_trainer.custom_models.variational_autoencoder import (
    VariationalAutoencoderOutput,
)


class VariationalAutoencoderLoss(nn.Module):
    """Composite VAE loss with reconstruction and analytic KL terms.

    Args:
        reconstruction_loss: Reconstruction term. Supported values are
            ``"mse"``, ``"l1"``, ``"smooth_l1"``, ``"ms_ssim"``,
            ``"smooth_l1_ms_ssim"``, and ``"bce_with_logits"``.
        reconstruction_weight: Multiplicative weight for the reconstruction term.
        beta: Multiplicative weight for the KL term.
        free_bits: Minimum KL (in nats) per latent spatial position. Positions
            below this floor have their KL gradient blocked, preventing
            over-regularisation of active dimensions while discouraging
            posterior collapse. Set to 0.0 to disable (default). Recommended
            range: 0.1–0.5 nats for spatial VAEs.
        kl_balance: When ``True``, scales the KL term by
            ``(C * H * W) / (Cz * Hz * Wz)`` so it is proportional to the
            same total-information budget as the reconstruction term. This
            matches the theoretical ELBO and is useful when input dimensions
            greatly exceed latent dimensions (e.g. ``encoder_depth=5``).
            The raw ``kl_loss`` key in the output dict is never affected;
            only ``weighted_kl_loss`` and ``loss`` reflect the scaling.
        smooth_l1_beta: Threshold for the ``smooth_l1`` reconstruction mode.
            Controls the L2-to-L1 transition point in Huber loss. Only used
            when ``reconstruction_loss`` includes ``"smooth_l1"``. Default:
            ``0.1``.
        smooth_l1_weight: Weight of the Smooth L1 component when
            ``reconstruction_loss="smooth_l1_ms_ssim"``.
        ms_ssim_weight: Weight of the MS-SSIM component when
            ``reconstruction_loss="smooth_l1_ms_ssim"``.
        ms_ssim_data_range: Dynamic range of image tensors passed to MS-SSIM.
            Use ``1.0`` for ``[0, 1]`` tensors and ``255.0`` for raw uint8-scale
            tensors.
        ms_ssim_sigmas: Gaussian sigma values used by the multi-scale SSIM
            windows.
        ms_ssim_alpha: Alpha parameter forwarded to ``kornia.losses.MS_SSIMLoss``.
            Default ``1.0`` disables Kornia's internal L1 blend so the term is
            pure MS-SSIM loss.
        ms_ssim_compensation: Compensation factor forwarded to
            ``kornia.losses.MS_SSIMLoss``.
        **kwargs: Reserved for Hydra compatibility.

    Returns:
        A dictionary with scalar tensors:

        - ``loss``: total ELBO loss.
        - ``reconstruction_loss``: raw reconstruction term (no weight).
        - ``kl_loss``: raw KL per latent dim (free_bits applied; no balance/beta).
        - ``weighted_reconstruction_loss``: ``reconstruction_weight × recon``.
        - ``weighted_kl_loss``: ``beta × kl_balance_factor × kl``.

    Example YAML:
        loss:
          _target_: pytorch_segmentation_models_trainer.custom_losses.autoencoder_losses.VariationalAutoencoderLoss
          reconstruction_loss: smooth_l1_ms_ssim
          reconstruction_weight: 1.0
          beta: 1.0
          free_bits: 0.25
          kl_balance: true
          smooth_l1_beta: 0.1
          smooth_l1_weight: 0.8
          ms_ssim_weight: 0.2
          ms_ssim_data_range: 1.0
    """

    _VALID_RECONSTRUCTION_LOSSES = {
        "mse",
        "l1",
        "smooth_l1",
        "ms_ssim",
        "smooth_l1_ms_ssim",
        "bce_with_logits",
    }

    def __init__(
        self,
        reconstruction_loss: str = "mse",
        reconstruction_weight: float = 1.0,
        beta: float = 1.0,
        free_bits: float = 0.0,
        kl_balance: bool = False,
        smooth_l1_beta: float = 0.1,
        smooth_l1_weight: float = 0.8,
        ms_ssim_weight: float = 0.2,
        ms_ssim_data_range: float = 1.0,
        ms_ssim_sigmas: Sequence[float] = (0.5, 1.0, 2.0, 4.0, 8.0),
        ms_ssim_alpha: float = 1.0,
        ms_ssim_compensation: float = 1.0,
        **kwargs,
    ):
        super().__init__()
        self.reconstruction_loss = reconstruction_loss
        self.reconstruction_weight = reconstruction_weight
        self.beta = beta
        self.free_bits = free_bits
        self.kl_balance = kl_balance
        self.smooth_l1_beta = smooth_l1_beta
        self.smooth_l1_weight = smooth_l1_weight
        self.ms_ssim_weight = ms_ssim_weight
        self.ms_ssim_data_range = ms_ssim_data_range
        self.ms_ssim_sigmas = tuple(ms_ssim_sigmas)
        self.ms_ssim_alpha = ms_ssim_alpha
        self.ms_ssim_compensation = ms_ssim_compensation
        self.extra_kwargs = kwargs

        if reconstruction_loss not in self._VALID_RECONSTRUCTION_LOSSES:
            raise ValueError(
                "reconstruction_loss must be one of 'mse', 'l1', 'smooth_l1', "
                "'ms_ssim', 'smooth_l1_ms_ssim', or 'bce_with_logits'; got "
                f"{reconstruction_loss!r}"
            )

        if smooth_l1_weight < 0 or ms_ssim_weight < 0:
            raise ValueError("smooth_l1_weight and ms_ssim_weight must be >= 0")

        self.ms_ssim_loss = None
        if reconstruction_loss in {"ms_ssim", "smooth_l1_ms_ssim"}:
            self.ms_ssim_loss = MS_SSIMLoss(
                sigmas=self.ms_ssim_sigmas,
                data_range=ms_ssim_data_range,
                alpha=ms_ssim_alpha,
                compensation=ms_ssim_compensation,
                reduction="mean",
            )

    def _compute_reconstruction_loss(
        self, reconstruction: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        """Compute the configured reconstruction loss.

        Args:
            reconstruction: Reconstructed image tensor.
            target: Target image tensor.

        Returns:
            Scalar reconstruction loss tensor.
        """
        if self.reconstruction_loss == "mse":
            return F.mse_loss(reconstruction, target)
        if self.reconstruction_loss == "l1":
            return F.l1_loss(reconstruction, target)
        if self.reconstruction_loss == "smooth_l1":
            return F.smooth_l1_loss(reconstruction, target, beta=self.smooth_l1_beta)
        if self.reconstruction_loss == "ms_ssim":
            return self._compute_ms_ssim_loss(reconstruction, target)
        if self.reconstruction_loss == "smooth_l1_ms_ssim":
            components = self._compute_reconstruction_components(reconstruction, target)
            return components["reconstruction_loss"]
        return F.binary_cross_entropy_with_logits(reconstruction, target)

    def _compute_ms_ssim_loss(
        self, reconstruction: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        """Compute MS-SSIM reconstruction loss.

        Args:
            reconstruction: Reconstructed image tensor.
            target: Target image tensor.

        Returns:
            Scalar MS-SSIM loss tensor.
        """
        if self.ms_ssim_loss is None:
            raise RuntimeError("MS-SSIM loss was not initialised")
        return self.ms_ssim_loss(reconstruction, target)

    def _compute_reconstruction_components(
        self, reconstruction: torch.Tensor, target: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """Compute reconstruction loss and optional component losses.

        Args:
            reconstruction: Reconstructed image tensor.
            target: Target image tensor.

        Returns:
            Dictionary containing the scalar ``reconstruction_loss`` and any
            available component losses.
        """
        if self.reconstruction_loss != "smooth_l1_ms_ssim":
            reconstruction_loss = self._compute_reconstruction_loss(
                reconstruction, target
            )
            components = {"reconstruction_loss": reconstruction_loss}
            if self.reconstruction_loss == "ms_ssim":
                components["ms_ssim_loss"] = reconstruction_loss
            return components

        smooth_l1_loss = F.smooth_l1_loss(
            reconstruction, target, beta=self.smooth_l1_beta
        )
        ms_ssim_loss = self._compute_ms_ssim_loss(reconstruction, target)
        weighted_smooth_l1_loss = self.smooth_l1_weight * smooth_l1_loss
        weighted_ms_ssim_loss = self.ms_ssim_weight * ms_ssim_loss

        return {
            "reconstruction_loss": weighted_smooth_l1_loss + weighted_ms_ssim_loss,
            "smooth_l1_loss": smooth_l1_loss,
            "ms_ssim_loss": ms_ssim_loss,
            "weighted_smooth_l1_loss": weighted_smooth_l1_loss,
            "weighted_ms_ssim_loss": weighted_ms_ssim_loss,
        }

    def _compute_kl_loss(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Compute KL divergence from ``q(z|x)`` to ``N(0, I)``.

        When ``free_bits > 0``, each latent position's KL is clamped to at
        least ``free_bits`` nats before averaging. Positions below the floor
        contribute a constant to the loss (gradient is zero for those
        positions), which prevents the KL from over-regularising dimensions
        that the encoder has not yet learned to use.

        Args:
            mu: Posterior mean tensor (B, Cz, Hz, Wz).
            logvar: Posterior log-variance tensor, same shape as ``mu``.

        Returns:
            Scalar KL loss per latent position (nats).
        """
        kl_per_dim = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
        if self.free_bits > 0:
            kl_per_dim = kl_per_dim.clamp(min=self.free_bits)
        return kl_per_dim.mean()

    def forward(
        self, output: VariationalAutoencoderOutput, target: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """Compute total VAE loss.

        Args:
            output: VAE output containing reconstruction, ``mu``, and ``logvar``.
            target: Clean reconstruction target tensor.

        Returns:
            Dictionary with total loss and detached component tensors.
        """
        if not isinstance(output, VariationalAutoencoderOutput):
            raise TypeError(
                "VariationalAutoencoderLoss expects VariationalAutoencoderOutput"
            )

        reconstruction_components = self._compute_reconstruction_components(
            output.reconstruction, target
        )
        reconstruction_loss = reconstruction_components["reconstruction_loss"]
        kl_loss = self._compute_kl_loss(output.mu, output.logvar)

        if self.kl_balance:
            data_numel = float(target[0].numel())
            latent_numel = float(output.mu[0].numel())
            kl_loss_for_total = kl_loss * (data_numel / latent_numel)
        else:
            kl_loss_for_total = kl_loss

        total_loss = (
            self.reconstruction_weight * reconstruction_loss
            + self.beta * kl_loss_for_total
        )

        result = {
            "loss": total_loss,
            "reconstruction_loss": reconstruction_loss.detach(),
            "kl_loss": kl_loss.detach(),
            "weighted_reconstruction_loss": (
                self.reconstruction_weight * reconstruction_loss
            ).detach(),
            "weighted_kl_loss": (self.beta * kl_loss_for_total).detach(),
        }
        result.update(
            {
                name: value.detach()
                for name, value in reconstruction_components.items()
                if name != "reconstruction_loss"
            }
        )
        return result
