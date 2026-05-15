# -*- coding: utf-8 -*-
"""Callbacks for autoencoder latent-space diagnostics."""

from typing import Optional

import pytorch_lightning as pl
import torch

from pytorch_segmentation_models_trainer.custom_metrics.autoencoder_latent_clustering import (
    AutoencoderLatentClusteringMetrics,
)
from pytorch_segmentation_models_trainer.custom_models.variational_autoencoder import (
    VariationalAutoencoderOutput,
)


class AutoencoderLatentClusteringCallback(pl.callbacks.Callback):
    """Log clustering metrics for autoencoder latent embeddings.

    Args:
        n_clusters: Number of clusters used by K-Means.
        max_samples: Maximum number of accumulated samples per epoch.
        kmeans_max_iter: Maximum number of mini-batch K-Means iterations.
        kmeans_batch_size: Mini-batch size used by K-Means.
        tol: K-Means convergence tolerance.
        random_state: Optional seed passed to ``MiniBatchKMeans``.
        normalize: Whether to L2-normalize embeddings before clustering.
        latent_reduction: Reduction for spatial latents.
        compute_silhouette: Whether to compute silhouette score.
        compute_dunn: Whether to compute Dunn index.
        label_key: Optional batch key with labels for ARI/NMI.
        vae_latent: Which VAE tensor to use: ``"mu"`` or ``"z"``.
        latent_source: ``"auto"`` uses ``model.encode`` when available and VAE
            ``mu`` otherwise. ``"encode"`` requires ``model.encode``.
        image_key: Batch key containing input images.
        **kwargs: Ignored extra Hydra arguments for forward compatibility.

    Returns:
        ``None``. Metrics are logged through the LightningModule logger.

    Example YAML:
        callbacks:
          - _target_: pytorch_segmentation_models_trainer.custom_callbacks.AutoencoderLatentClusteringCallback
            n_clusters: 8
            max_samples: 2048
            kmeans_max_iter: 50
            normalize: true
            compute_silhouette: false
            vae_latent: mu
    """

    def __init__(
        self,
        n_clusters: int,
        max_samples: Optional[int] = 2048,
        kmeans_max_iter: int = 50,
        kmeans_batch_size: int = 1024,
        tol: float = 1e-4,
        random_state: Optional[int] = None,
        normalize: bool = True,
        latent_reduction: str = "adaptive_avg_pool",
        compute_silhouette: bool = False,
        compute_dunn: bool = False,
        label_key: Optional[str] = None,
        vae_latent: str = "mu",
        latent_source: str = "auto",
        image_key: str = "image",
        **kwargs,
    ):
        super().__init__()
        if latent_source not in ("auto", "encode"):
            raise ValueError("'latent_source' must be 'auto' or 'encode'.")
        self.image_key = image_key
        self.latent_source = latent_source
        self.val_latent_metrics = AutoencoderLatentClusteringMetrics(
            n_clusters=n_clusters,
            max_samples=max_samples,
            kmeans_max_iter=kmeans_max_iter,
            kmeans_batch_size=kmeans_batch_size,
            tol=tol,
            random_state=random_state,
            normalize=normalize,
            latent_reduction=latent_reduction,
            compute_silhouette=compute_silhouette,
            compute_dunn=compute_dunn,
            label_key=label_key,
            vae_latent=vae_latent,
        )
        self.test_latent_metrics = AutoencoderLatentClusteringMetrics(
            n_clusters=n_clusters,
            max_samples=max_samples,
            kmeans_max_iter=kmeans_max_iter,
            kmeans_batch_size=kmeans_batch_size,
            tol=tol,
            random_state=random_state,
            normalize=normalize,
            latent_reduction=latent_reduction,
            compute_silhouette=compute_silhouette,
            compute_dunn=compute_dunn,
            label_key=label_key,
            vae_latent=vae_latent,
        )
        self.extra_kwargs = kwargs

    def on_validation_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs,
        batch,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        """Accumulate validation latents after each validation batch."""
        self._update_latent_metrics(pl_module, batch, self.val_latent_metrics)

    def on_test_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs,
        batch,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        """Accumulate test latents after each test batch."""
        self._update_latent_metrics(pl_module, batch, self.test_latent_metrics)

    def on_validation_epoch_end(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        """Compute and log validation latent clustering metrics."""
        self._log_latent_metrics(pl_module, self.val_latent_metrics, "val")

    def on_test_epoch_end(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        """Compute and log test latent clustering metrics."""
        self._log_latent_metrics(pl_module, self.test_latent_metrics, "test")

    def _update_latent_metrics(
        self,
        pl_module: pl.LightningModule,
        batch,
        latent_metrics: AutoencoderLatentClusteringMetrics,
    ) -> None:
        images = self._get_images(batch)
        labels = self._get_labels(batch, latent_metrics)
        with torch.no_grad():
            latents = self._extract_latents(pl_module, images, latent_metrics)
        latent_metrics.update(latents, target_labels=labels)

    def _extract_latents(
        self,
        pl_module: pl.LightningModule,
        images: torch.Tensor,
        latent_metrics: AutoencoderLatentClusteringMetrics,
    ) -> torch.Tensor:
        model = getattr(pl_module, "model", pl_module)
        if hasattr(model, "encode"):
            encoded = model.encode(images)
            if isinstance(encoded, tuple):
                latent_name = getattr(latent_metrics, "vae_latent", "mu")
                if latent_name == "mu":
                    return encoded[0]
            else:
                return encoded
        if self.latent_source == "encode":
            raise ValueError("latent_source='encode' requires model.encode(images).")

        output = pl_module(images)
        if not isinstance(output, VariationalAutoencoderOutput):
            raise ValueError(
                "AutoencoderLatentClusteringCallback requires model.encode(images) "
                "or a VariationalAutoencoderOutput forward result."
            )
        latent_name = getattr(latent_metrics, "vae_latent", "mu")
        return output.mu if latent_name == "mu" else output.z

    def _get_images(self, batch) -> torch.Tensor:
        if isinstance(batch, dict):
            return batch[self.image_key]
        return batch[0]

    def _get_labels(self, batch, latent_metrics):
        label_key = getattr(latent_metrics, "label_key", None)
        if label_key is None or not isinstance(batch, dict) or label_key not in batch:
            return None
        labels = batch[label_key]
        if isinstance(labels, torch.Tensor):
            return labels
        return torch.as_tensor(labels)

    def _log_latent_metrics(
        self,
        pl_module: pl.LightningModule,
        latent_metrics: AutoencoderLatentClusteringMetrics,
        prefix: str,
    ) -> None:
        if not getattr(latent_metrics, "_embeddings", None):
            return
        metrics = latent_metrics.compute()
        pl_module.log_dict(
            {f"{prefix}/{name}": value for name, value in metrics.items()},
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            sync_dist=True,
        )
        latent_metrics.reset()
