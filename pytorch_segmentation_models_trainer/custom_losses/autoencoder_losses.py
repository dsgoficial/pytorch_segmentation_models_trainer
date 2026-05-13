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
            ``"mse"``, ``"l1"``, and ``"bce_with_logits"``.
        reconstruction_weight: Multiplicative weight for the reconstruction term.
        beta: Multiplicative weight for the KL term.
        **kwargs: Reserved for Hydra compatibility.

    Returns:
        A dictionary with scalar tensors ``loss``, ``reconstruction_loss``, and
        ``kl_loss``.

    Example YAML:
        loss:
          _target_: pytorch_segmentation_models_trainer.custom_losses.autoencoder_losses.VariationalAutoencoderLoss
          reconstruction_loss: mse
          reconstruction_weight: 1.0
          beta: 1.0
    """

    def __init__(
        self,
        reconstruction_loss: str = "mse",
        reconstruction_weight: float = 1.0,
        beta: float = 1.0,
        **kwargs,
    ):
        super().__init__()
        self.reconstruction_loss = reconstruction_loss
        self.reconstruction_weight = reconstruction_weight
        self.beta = beta
        self.extra_kwargs = kwargs

        if reconstruction_loss not in {"mse", "l1", "bce_with_logits"}:
            raise ValueError(
                "reconstruction_loss must be one of 'mse', 'l1', or "
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
        return F.binary_cross_entropy_with_logits(reconstruction, target)

    def _compute_kl_loss(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Compute KL divergence from ``q(z|x)`` to ``N(0, I)``.

        Args:
            mu: Posterior mean tensor.
            logvar: Posterior log-variance tensor.

        Returns:
            Scalar KL loss averaged by batch size.
        """
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return kl_loss / mu.shape[0]

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
        total_loss = (
            self.reconstruction_weight * reconstruction_loss + self.beta * kl_loss
        )

        return {
            "loss": total_loss,
            "reconstruction_loss": reconstruction_loss.detach(),
            "kl_loss": kl_loss.detach(),
        }
