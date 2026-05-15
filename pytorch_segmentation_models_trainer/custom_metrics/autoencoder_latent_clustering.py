# -*- coding: utf-8 -*-
"""GPU-friendly latent clustering metrics for autoencoder encoders."""

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torchmetrics.functional.clustering import (
    adjusted_rand_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    dunn_index,
    normalized_mutual_info_score,
)

from pytorch_segmentation_models_trainer.tools.kmeans import kmeans_calculator

MiniBatchKMeans = kmeans_calculator.MiniBatchKMeans


class AutoencoderLatentClusteringMetrics:
    """Compute clustering diagnostics over autoencoder latent embeddings.

    The metric accumulates latent tensors during an epoch, clusters them with
    the framework's PyTorch ``MiniBatchKMeans``, and computes TorchMetrics
    clustering scores on the same device as the embeddings.  It is intended for
    validation/test epoch logging, not per-step optimization.

    Args:
        n_clusters: Number of clusters used by K-Means.
        max_samples: Maximum number of samples used per epoch.  ``None`` uses
            all accumulated embeddings.
        kmeans_max_iter: Maximum number of mini-batch K-Means iterations.
        kmeans_batch_size: Mini-batch size used by K-Means.
        tol: K-Means convergence tolerance.
        random_state: Optional seed passed to ``MiniBatchKMeans``.
        normalize: Whether to L2-normalize embeddings before clustering.
        latent_reduction: Reduction for spatial latents.
            ``"adaptive_avg_pool"`` converts ``(B, C, H, W)`` to ``(B, C)``;
            ``"flatten"`` converts to ``(B, C*H*W)``.
        compute_silhouette: Whether to compute the PyTorch silhouette score.
            This is O(N²), so keep ``max_samples`` bounded.
        compute_dunn: Whether to compute TorchMetrics Dunn index.
        label_key: Optional batch key with labels for ARI/NMI.
        vae_latent: Which VAE tensor should be used by Lightning integration:
            ``"mu"`` or ``"z"``.
        **kwargs: Ignored extra Hydra arguments for forward compatibility.

    Returns:
        ``compute`` returns a dictionary of scalar tensors suitable for
        ``LightningModule.log_dict``.

    Example:
        metric = AutoencoderLatentClusteringMetrics(
            n_clusters=8,
            max_samples=2048,
            kmeans_max_iter=50,
            normalize=True,
            compute_silhouette=True,
            label_key="domain",
            vae_latent="mu",
        )
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
        **kwargs,
    ):
        if n_clusters < 2:
            raise ValueError("'n_clusters' must be at least 2.")
        if max_samples is not None and max_samples < n_clusters:
            raise ValueError("'max_samples' must be >= n_clusters when provided.")
        if latent_reduction not in ("adaptive_avg_pool", "flatten"):
            raise ValueError(
                "'latent_reduction' must be 'adaptive_avg_pool' or 'flatten'."
            )
        if vae_latent not in ("mu", "z"):
            raise ValueError("'vae_latent' must be 'mu' or 'z'.")

        self.n_clusters = n_clusters
        self.max_samples = max_samples
        self.kmeans_max_iter = kmeans_max_iter
        self.kmeans_batch_size = kmeans_batch_size
        self.tol = tol
        self.random_state = random_state
        self.normalize = normalize
        self.latent_reduction = latent_reduction
        self.compute_silhouette = compute_silhouette
        self.compute_dunn = compute_dunn
        self.label_key = label_key
        self.vae_latent = vae_latent
        self._embeddings: List[torch.Tensor] = []
        self._target_labels: List[torch.Tensor] = []

    def reduce_latents(self, latents: torch.Tensor) -> torch.Tensor:
        """Convert latent tensors to a 2D embedding matrix.

        Args:
            latents: Tensor with shape ``(B, D)`` or ``(B, C, H, W)``.

        Returns:
            Tensor with shape ``(B, D)``.

        Raises:
            ValueError: If ``latents`` is not 2D or 4D.
        """
        if latents.ndim == 2:
            return latents.float()
        if latents.ndim != 4:
            raise ValueError("Latent embeddings must be 2D or 4D tensors.")
        if self.latent_reduction == "flatten":
            return latents.flatten(start_dim=1).float()
        pooled = F.adaptive_avg_pool2d(latents, output_size=1)
        return pooled.flatten(start_dim=1).float()

    def update(
        self,
        latents: torch.Tensor,
        target_labels: Optional[torch.Tensor] = None,
    ) -> None:
        """Accumulate one batch of latents and optional ground-truth labels.

        Args:
            latents: Latent tensor from the encoder.
            target_labels: Optional labels used for ARI/NMI.

        Returns:
            ``None``.
        """
        self._embeddings.append(latents.detach())
        if target_labels is not None:
            self._target_labels.append(target_labels.detach())

    def reset(self) -> None:
        """Clear accumulated epoch state.

        Returns:
            ``None``.
        """
        self._embeddings.clear()
        self._target_labels.clear()

    def compute(self) -> Dict[str, torch.Tensor]:
        """Cluster accumulated embeddings and return scalar metric tensors.

        Returns:
            Dictionary with unprefixed latent metric names.

        Raises:
            ValueError: If no embeddings were accumulated or if fewer samples
                than ``n_clusters`` are available.
        """
        if not self._embeddings:
            raise ValueError("No latent embeddings were accumulated.")

        embeddings = self.reduce_latents(torch.cat(self._embeddings, dim=0))
        target_labels = (
            torch.cat(self._target_labels, dim=0).long()
            if self._target_labels
            else None
        )
        embeddings, target_labels = self._sample(embeddings, target_labels)

        if embeddings.shape[0] < self.n_clusters:
            raise ValueError(
                f"n_clusters={self.n_clusters} requires at least "
                f"{self.n_clusters} samples, got {embeddings.shape[0]}."
            )

        if self.normalize:
            embeddings = F.normalize(embeddings, dim=1)

        kmeans = MiniBatchKMeans(
            n_clusters=self.n_clusters,
            max_iter=self.kmeans_max_iter,
            batch_size=self.kmeans_batch_size,
            tol=self.tol,
            device=embeddings.device,
            random_state=self.random_state,
        )
        kmeans.fit(embeddings)
        cluster_labels = kmeans.predict(embeddings)

        result: Dict[str, torch.Tensor] = {
            "latent_calinski_harabasz": calinski_harabasz_score(
                embeddings, cluster_labels
            ),
            "latent_davies_bouldin": davies_bouldin_score(embeddings, cluster_labels),
        }

        if self.compute_dunn:
            result["latent_dunn"] = dunn_index(embeddings, cluster_labels)
        if self.compute_silhouette:
            result["latent_silhouette"] = self._silhouette_score(
                embeddings, cluster_labels
            )
        if target_labels is not None:
            result["latent_adjusted_rand"] = adjusted_rand_score(
                cluster_labels, target_labels.to(cluster_labels.device)
            ).to(embeddings.device)
            nmi = normalized_mutual_info_score(
                cluster_labels,
                target_labels.to(cluster_labels.device),
            )
            result["latent_normalized_mutual_info"] = nmi.to(embeddings.device)

        return result

    def _sample(
        self, embeddings: torch.Tensor, target_labels: Optional[torch.Tensor]
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Apply deterministic max-sample truncation."""
        if self.max_samples is None or embeddings.shape[0] <= self.max_samples:
            return embeddings, target_labels
        embeddings = embeddings[: self.max_samples]
        if target_labels is None:
            return embeddings, None
        return embeddings, target_labels[: self.max_samples]

    def _silhouette_score(
        self, embeddings: torch.Tensor, labels: torch.Tensor
    ) -> torch.Tensor:
        """Compute mean silhouette score in PyTorch on the embedding device."""
        unique_labels = torch.unique(labels)
        if unique_labels.numel() < 2 or unique_labels.numel() >= embeddings.shape[0]:
            return torch.tensor(float("nan"), device=embeddings.device)

        distances = torch.cdist(embeddings, embeddings, p=2)
        scores = []
        for idx in range(embeddings.shape[0]):
            label = labels[idx]
            same = labels == label
            same[idx] = False
            if same.any():
                intra = distances[idx, same].mean()
            else:
                intra = torch.tensor(0.0, device=embeddings.device)

            nearest = None
            for other_label in unique_labels:
                if other_label == label:
                    continue
                other = labels == other_label
                candidate = distances[idx, other].mean()
                nearest = (
                    candidate if nearest is None else torch.minimum(nearest, candidate)
                )

            denom = torch.maximum(intra, nearest)
            score = torch.where(denom > 0, (nearest - intra) / denom, denom)
            scores.append(score)

        return torch.stack(scores).mean()
