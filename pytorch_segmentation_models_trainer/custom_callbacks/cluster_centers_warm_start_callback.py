# -*- coding: utf-8 -*-
"""Callback that warm-starts VAE cluster centers before Phase-2 training."""

from typing import Optional

import pytorch_lightning as pl
import torch

from pytorch_segmentation_models_trainer.custom_losses.autoencoder_clustering_losses import (
    ClusteringAwareVAELoss,
)


class ClusterCentersWarmStartCallback(pl.Callback):
    """Warm-start ``ClusteringAwareVAELoss`` cluster centers before training.

    Runs once at ``on_train_start``.  Iterates the training dataloader,
    collects latent embeddings from the model, then calls
    ``loss_function.initialize_centers_from_embeddings`` so that Phase-2
    clustering loss starts from a sensible K-Means solution rather than
    random noise.

    If ``pl_module.loss_function`` is not a ``ClusteringAwareVAELoss`` the
    callback is a no-op.

    Args:
        max_samples: Maximum number of embedding samples used to fit K-Means.
            Default ``4096``.
        image_key: Key in the batch dict that holds input images.
            Default ``"image"``.
        vae_latent: Which latent tensor to collect — ``"mu"`` or ``"z"``.
            Default ``"mu"`` (deterministic posterior mean).
        kmeans_max_iter: Maximum K-Means iterations forwarded to
            ``initialize_centers_from_embeddings``.  Default ``100``.
        kmeans_batch_size: Mini-batch size for K-Means.  Default ``1024``.
        random_state: Optional RNG seed.
        **kwargs: Ignored extra arguments for Hydra compatibility.

    Returns:
        ``None``.  Cluster centers in ``pl_module.loss_function`` are mutated
        in-place.

    Example YAML:
        callbacks:
          - _target_: pytorch_segmentation_models_trainer.custom_callbacks.ClusterCentersWarmStartCallback
            max_samples: 4096
            image_key: image
            vae_latent: mu
    """

    def __init__(
        self,
        max_samples: int = 4096,
        image_key: str = "image",
        vae_latent: str = "mu",
        kmeans_max_iter: int = 100,
        kmeans_batch_size: int = 1024,
        random_state: Optional[int] = None,
        **kwargs,
    ) -> None:
        super().__init__()
        if vae_latent not in ("mu", "z"):
            raise ValueError(f"vae_latent must be 'mu' or 'z', got {vae_latent!r}.")
        self.max_samples = max_samples
        self.image_key = image_key
        self.vae_latent = vae_latent
        self.kmeans_max_iter = kmeans_max_iter
        self.kmeans_batch_size = kmeans_batch_size
        self.random_state = random_state

    def on_train_start(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        """Collect embeddings and initialise cluster centers.

        Args:
            trainer: The PyTorch Lightning trainer.
            pl_module: The LightningModule being trained.

        Returns:
            ``None``.
        """
        if not self._has_clustering_loss(pl_module):
            return

        embeddings = self._collect_embeddings(trainer, pl_module)
        pl_module.loss_function.initialize_centers_from_embeddings(
            embeddings,
            kmeans_max_iter=self.kmeans_max_iter,
            kmeans_batch_size=self.kmeans_batch_size,
            random_state=self.random_state,
        )

    def _has_clustering_loss(self, pl_module: pl.LightningModule) -> bool:
        return hasattr(pl_module, "loss_function") and isinstance(
            pl_module.loss_function, ClusteringAwareVAELoss
        )

    def _collect_embeddings(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> torch.Tensor:
        """Iterate train_dataloader and accumulate latent vectors.

        Args:
            trainer: The Lightning trainer (provides ``train_dataloader``).
            pl_module: The model used for inference.

        Returns:
            Float tensor ``(N, D)`` or ``(N, C, H, W)`` with accumulated
            latents, capped at ``max_samples``.
        """
        chunks = []
        total = 0

        pl_module.eval()
        try:
            with torch.no_grad():
                for batch in trainer.train_dataloader:
                    images = batch[self.image_key]
                    output = pl_module(images)
                    latent = output.mu if self.vae_latent == "mu" else output.z
                    chunks.append(latent.detach().cpu())
                    total += latent.shape[0]
                    if total >= self.max_samples:
                        break
        finally:
            pl_module.train()

        all_embeddings = torch.cat(chunks, dim=0)
        return all_embeddings[: self.max_samples]
