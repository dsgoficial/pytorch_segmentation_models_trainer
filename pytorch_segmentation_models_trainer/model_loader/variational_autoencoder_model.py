# -*- coding: utf-8 -*-
"""PyTorch Lightning module for variational autoencoder training."""

import torch

from pytorch_segmentation_models_trainer.custom_models.variational_autoencoder import (
    VariationalAutoencoderOutput,
)
from pytorch_segmentation_models_trainer.model_loader.model import Model


class VariationalAutoencoderModel(Model):
    """LightningModule for VAE reconstruction training.

    Args:
        cfg: Hydra training configuration.
        inference_mode: When ``True``, skips dataset and loss instantiation.

    Returns:
        Training, validation, and test steps return the scalar total loss.

    Example YAML:
        pl_model:
          _target_: pytorch_segmentation_models_trainer.model_loader.\
variational_autoencoder_model.VariationalAutoencoderModel

        loss:
          _target_: pytorch_segmentation_models_trainer.custom_losses.\
autoencoder_losses.VariationalAutoencoderLoss
          reconstruction_loss: mse
          beta: 1.0

        callbacks:
          - _target_: pytorch_segmentation_models_trainer.custom_callbacks.\
AutoencoderLatentClusteringCallback
            n_clusters: 8
            vae_latent: mu
    """

    def __init__(self, cfg, inference_mode=False):
        """Create the VAE LightningModule and optional latent metrics.

        Args:
            cfg: Hydra training configuration.
            inference_mode: When ``True``, skips dataset and loss setup.

        Returns:
            ``None``.
        """
        super().__init__(cfg, inference_mode=inference_mode)

    def _log_loss_dict(self, loss_dict: dict, prefix: str) -> torch.Tensor:
        """Log total loss and any available VAE component losses.

        Args:
            loss_dict: Dictionary containing at least the ``loss`` key.
            prefix: Lightning log prefix.

        Returns:
            Scalar total loss tensor.
        """
        loss = loss_dict["loss"]
        for name, value in loss_dict.items():
            self.log(
                f"{prefix}/{name}",
                value,
                on_step=(prefix == "train"),
                on_epoch=True,
                prog_bar=(name == "loss"),
                sync_dist=True,
            )
        return loss

    def _shared_step(self, batch, prefix: str) -> torch.Tensor:
        """Run a reconstruction step and log VAE loss components.

        Args:
            batch: Batch dictionary with ``image`` and optional ``target``.
            prefix: Logging namespace for this step.

        Returns:
            Scalar total loss tensor.
        """
        images = batch["image"]
        targets = batch.get("target", images)
        output = self(images)
        loss_result = self.loss_function(output, targets)

        if isinstance(loss_result, dict):
            loss = self._log_loss_dict(loss_result, prefix)
        else:
            loss = loss_result
            self.log(
                f"{prefix}/loss",
                loss_result,
                on_step=(prefix == "train"),
                on_epoch=True,
                prog_bar=True,
                sync_dist=True,
            )

        metrics_attr = f"{prefix}_metrics"
        if hasattr(self, metrics_attr):
            reconstruction = (
                output.reconstruction
                if isinstance(output, VariationalAutoencoderOutput)
                else output
            )
            if isinstance(reconstruction, torch.Tensor):
                metrics = getattr(self, metrics_attr)(reconstruction, targets)
                self.log_dict(
                    metrics,
                    on_step=(prefix == "train"),
                    on_epoch=True,
                    prog_bar=False,
                    sync_dist=True,
                )

        return loss

    def training_step(self, batch, batch_idx):
        """Run one training step.

        Args:
            batch: Batch dictionary with ``image`` and optional ``target``.
            batch_idx: Batch index supplied by Lightning.

        Returns:
            Scalar training loss tensor.
        """
        return self._shared_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        """Run one validation step.

        Args:
            batch: Batch dictionary with ``image`` and optional ``target``.
            batch_idx: Batch index supplied by Lightning.

        Returns:
            Scalar validation loss tensor.
        """
        return self._shared_step(batch, "val")

    def test_step(self, batch, batch_idx):
        """Run one test step.

        Args:
            batch: Batch dictionary with ``image`` and optional ``target``.
            batch_idx: Batch index supplied by Lightning.

        Returns:
            Scalar test loss tensor.
        """
        return self._shared_step(batch, "test")

    def forward(self, x):
        """Delegate the forward pass to the configured VAE module.

        Args:
            x: Input image tensor.

        Returns:
            Output produced by ``self.model``.
        """
        return self.model(x)
