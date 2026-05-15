# -*- coding: utf-8 -*-
"""PyTorch Lightning module for variational autoencoder training."""

import torch
from hydra.utils import instantiate

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

        latent_metrics:
          _target_: pytorch_segmentation_models_trainer.custom_metrics.\
autoencoder_latent_clustering.AutoencoderLatentClusteringMetrics
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
        if not inference_mode and "latent_metrics" in self.cfg:
            self.val_latent_metrics = instantiate(
                self.cfg.latent_metrics, _recursive_=False
            )
            self.test_latent_metrics = instantiate(
                self.cfg.latent_metrics, _recursive_=False
            )

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

        self._update_latent_metrics(output, batch, prefix)
        return loss

    def _update_latent_metrics(self, output, batch, prefix: str) -> None:
        """Accumulate VAE latents for epoch-level clustering metrics.

        Args:
            output: VAE output from the current step.
            batch: Batch dictionary, optionally containing label keys.
            prefix: Logging namespace for this step.

        Returns:
            ``None``.
        """
        metrics_attr = f"{prefix}_latent_metrics"
        if not hasattr(self, metrics_attr) or not isinstance(
            output, VariationalAutoencoderOutput
        ):
            return

        latent_metrics = getattr(self, metrics_attr)
        latent_name = getattr(latent_metrics, "vae_latent", "mu")
        latents = output.mu if latent_name == "mu" else output.z
        labels = self._get_latent_metric_labels(batch, latent_metrics)
        latent_metrics.update(latents, target_labels=labels)

    def _get_latent_metric_labels(self, batch, latent_metrics):
        """Return optional labels configured for latent clustering metrics.

        Args:
            batch: Batch dictionary.
            latent_metrics: Latent metrics object with optional ``label_key``.

        Returns:
            Label tensor or ``None``.
        """
        label_key = getattr(latent_metrics, "label_key", None)
        if label_key is None or not isinstance(batch, dict) or label_key not in batch:
            return None
        labels = batch[label_key]
        if isinstance(labels, torch.Tensor):
            return labels
        return torch.as_tensor(labels)

    def _log_latent_metrics_epoch_end(self, prefix: str) -> None:
        """Compute, log, and reset accumulated latent clustering metrics.

        Args:
            prefix: Logging namespace, usually ``val`` or ``test``.

        Returns:
            ``None``.
        """
        metrics_attr = f"{prefix}_latent_metrics"
        if not hasattr(self, metrics_attr):
            return
        latent_metrics = getattr(self, metrics_attr)
        if not getattr(latent_metrics, "_embeddings", None):
            return
        metrics = latent_metrics.compute()
        self.log_dict(
            {f"{prefix}/{name}": value for name, value in metrics.items()},
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            sync_dist=True,
        )
        latent_metrics.reset()

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

    def on_validation_epoch_end(self):
        """Compute and log accumulated validation latent clustering metrics."""
        self._log_latent_metrics_epoch_end("val")

    def on_test_epoch_end(self):
        """Compute and log accumulated test latent clustering metrics."""
        self._log_latent_metrics_epoch_end("test")

    def forward(self, x):
        """Delegate the forward pass to the configured VAE module.

        Args:
            x: Input image tensor.

        Returns:
            Output produced by ``self.model``.
        """
        return self.model(x)
