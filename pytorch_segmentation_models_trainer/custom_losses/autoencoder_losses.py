# -*- coding: utf-8 -*-
"""Loss functions for autoencoder training."""

from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from pytorch_segmentation_models_trainer.custom_models.variational_autoencoder import (
    VariationalAutoencoderOutput,
)


class VariationalAutoencoderLoss(nn.Module):
    """Composite VAE loss with reconstruction and analytic KL terms.

    Args:
        reconstruction_loss: Reconstruction term. Supported values are
            ``"mse"``, ``"l1"``, ``"smooth_l1"``, and ``"bce_with_logits"``.
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
            when ``reconstruction_loss="smooth_l1"``. Default: ``0.1``.
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
          reconstruction_loss: smooth_l1
          reconstruction_weight: 1.0
          beta: 1.0
          free_bits: 0.25
          kl_balance: true
          smooth_l1_beta: 0.1
    """

    def __init__(
        self,
        reconstruction_loss: str = "mse",
        reconstruction_weight: float = 1.0,
        beta: float = 1.0,
        free_bits: float = 0.0,
        kl_balance: bool = False,
        smooth_l1_beta: float = 0.1,
        **kwargs,
    ):
        super().__init__()
        self.reconstruction_loss = reconstruction_loss
        self.reconstruction_weight = reconstruction_weight
        self.beta = beta
        self.free_bits = free_bits
        self.kl_balance = kl_balance
        self.smooth_l1_beta = smooth_l1_beta
        self.extra_kwargs = kwargs

        if reconstruction_loss not in {"mse", "l1", "smooth_l1", "bce_with_logits"}:
            raise ValueError(
                "reconstruction_loss must be one of 'mse', 'l1', 'smooth_l1', or "
                f"'bce_with_logits'; got {reconstruction_loss!r}"
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
        return F.binary_cross_entropy_with_logits(reconstruction, target)

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

        reconstruction_loss = self._compute_reconstruction_loss(
            output.reconstruction, target
        )
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

        return {
            "loss": total_loss,
            "reconstruction_loss": reconstruction_loss.detach(),
            "kl_loss": kl_loss.detach(),
            "weighted_reconstruction_loss": (
                self.reconstruction_weight * reconstruction_loss
            ).detach(),
            "weighted_kl_loss": (self.beta * kl_loss_for_total).detach(),
        }
