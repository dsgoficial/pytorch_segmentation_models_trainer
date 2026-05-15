# -*- coding: utf-8 -*-
"""Clustering-aware loss functions for autoencoder / VAE training.

Implements three losses designed to be used together in a Phase-2 fine-tuning
regime (DCEC-style) where the encoder is encouraged to produce a well-structured
latent space while reconstruction quality is preserved.

References:
    - Xie et al., "Unsupervised Deep Embedding for Clustering Analysis",
      ICML 2016.  (DEC soft-assignment loss)
    - Guo et al., "Deep Convolutional Embedded Clustering", IJCAI 2017.
      (DCEC: joint reconstruction + clustering loss for convolutional AEs)
    - Wen et al., "A Discriminative Feature Learning Approach for Deep Face
      Recognition", ECCV 2016.  (Center loss)
"""

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from pytorch_segmentation_models_trainer.custom_losses.autoencoder_losses import (
    VariationalAutoencoderLoss,
)
from pytorch_segmentation_models_trainer.custom_models.variational_autoencoder import (
    VariationalAutoencoderOutput,
)
from pytorch_segmentation_models_trainer.tools.kmeans.kmeans_calculator import (
    MiniBatchKMeans,
)


def _reduce_latents(latents: torch.Tensor, mode: str) -> torch.Tensor:
    """Reduce spatial or flat latent tensors to shape ``(B, D)``.

    Args:
        latents: Tensor with shape ``(B, D)`` or ``(B, C, H, W)``.
        mode: ``"adaptive_avg_pool"`` or ``"flatten"``.

    Returns:
        Float tensor with shape ``(B, D)``.

    Raises:
        ValueError: If ``latents`` is not 2-D or 4-D.
    """
    if latents.ndim == 2:
        return latents.float()
    if latents.ndim != 4:
        raise ValueError("Latent tensor must be 2-D (B, D) or 4-D (B, C, H, W).")
    if mode == "flatten":
        return latents.flatten(start_dim=1).float()
    return F.adaptive_avg_pool2d(latents, output_size=1).flatten(start_dim=1).float()


class DECSoftAssignmentLoss(nn.Module):
    """Soft-assignment clustering loss from DEC / DCEC.

    Measures the KL divergence between the soft cluster assignment distribution
    ``Q`` (Student t-kernel) and the sharpened target distribution ``P``.
    Minimising this loss pushes the encoder to produce confident, well-separated
    cluster assignments.

    The loss module stores a default ``cluster_centers`` parameter for
    convenience when used standalone.  When used inside
    ``ClusteringAwareVAELoss`` the wrapper's shared centers are passed
    explicitly and the stored parameter is unused.

    Args:
        n_clusters: Number of cluster centers ``K``.  Must be ≥ 2.
        embedding_dim: Dimensionality of the embedding vectors ``D``.
        alpha: Degrees of freedom of the Student t-distribution used to
            compute soft assignments.  Default ``1.0`` (Cauchy kernel).
            Must be > 0.

    Returns:
        ``forward`` returns a dict with key ``"loss"`` (scalar KL divergence).

    Example YAML:
        loss:
          _target_: ...DECSoftAssignmentLoss
          n_clusters: 8
          embedding_dim: 128
          alpha: 1.0

    References:
        Xie et al., ICML 2016; Guo et al., IJCAI 2017.
    """

    def __init__(
        self,
        n_clusters: int,
        embedding_dim: int,
        alpha: float = 1.0,
    ) -> None:
        super().__init__()
        if n_clusters < 2:
            raise ValueError(f"n_clusters must be >= 2, got {n_clusters}.")
        if alpha <= 0:
            raise ValueError(f"alpha must be > 0, got {alpha}.")
        self.n_clusters = n_clusters
        self.embedding_dim = embedding_dim
        self.alpha = alpha
        self.cluster_centers = nn.Parameter(
            torch.empty(n_clusters, embedding_dim).normal_(std=0.01)
        )

    def _compute_q(
        self, embeddings: torch.Tensor, cluster_centers: torch.Tensor
    ) -> torch.Tensor:
        """Compute soft assignment matrix Q via Student t-kernel.

        Args:
            embeddings: Float tensor ``(B, D)``.
            cluster_centers: Float tensor ``(K, D)``.

        Returns:
            Soft assignment matrix ``(B, K)`` with rows summing to 1.
        """
        sq_dists = torch.cdist(embeddings, cluster_centers, p=2).pow(2)
        numerator = (1.0 + sq_dists / self.alpha).pow(-(self.alpha + 1.0) / 2.0)
        return numerator / numerator.sum(dim=1, keepdim=True)

    def _compute_p(self, q: torch.Tensor) -> torch.Tensor:
        """Compute sharpened target distribution P from Q.

        Args:
            q: Soft assignment matrix ``(B, K)``.

        Returns:
            Target distribution ``(B, K)`` with rows summing to 1.
        """
        freq = q.sum(dim=0, keepdim=True)
        p = q.pow(2) / freq
        return p / p.sum(dim=1, keepdim=True)

    def initialize_centers(self, centers: torch.Tensor) -> None:
        """Warm-start cluster centers from external tensor (e.g., K-Means).

        Args:
            centers: Tensor ``(K, D)`` with initial center positions.

        Returns:
            ``None``.
        """
        self.cluster_centers.data.copy_(centers)

    def forward(
        self,
        embeddings: torch.Tensor,
        cluster_centers: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Compute DEC soft-assignment KL loss.

        Args:
            embeddings: Latent vectors ``(B, D)``.
            cluster_centers: Center tensor ``(K, D)``.

        Returns:
            Dict with ``"loss"`` (scalar, non-negative KL divergence).
        """
        q = self._compute_q(embeddings, cluster_centers)
        p = self._compute_p(q).detach()
        loss = F.kl_div(q.log(), p, reduction="batchmean")
        return {"loss": loss}


class CenterLoss(nn.Module):
    """Intra-cluster compactness loss (Center Loss).

    Penalises the mean squared distance from each embedding to its nearest
    cluster center.  Minimising this loss produces compact, well-separated
    clusters.

    The loss module stores a default ``cluster_centers`` parameter for
    convenience when used standalone.  When used inside
    ``ClusteringAwareVAELoss`` the wrapper's shared centers are passed
    explicitly and the stored parameter is unused.

    Args:
        n_clusters: Number of cluster centers ``K``.  Must be ≥ 2.
        embedding_dim: Dimensionality of embedding vectors ``D``.
        lambda_center: Multiplicative weight applied to the raw center loss.
            Must be ≥ 0.  Default ``1.0``.

    Returns:
        ``forward`` returns a dict with ``"loss"`` (weighted) and
        ``"raw_center_loss"`` (unweighted, detached).

    Example YAML:
        loss:
          _target_: ...CenterLoss
          n_clusters: 8
          embedding_dim: 128
          lambda_center: 0.01

    References:
        Wen et al., ECCV 2016.
    """

    def __init__(
        self,
        n_clusters: int,
        embedding_dim: int,
        lambda_center: float = 1.0,
    ) -> None:
        super().__init__()
        if n_clusters < 2:
            raise ValueError(f"n_clusters must be >= 2, got {n_clusters}.")
        if lambda_center < 0:
            raise ValueError(f"lambda_center must be >= 0, got {lambda_center}.")
        self.n_clusters = n_clusters
        self.embedding_dim = embedding_dim
        self.lambda_center = lambda_center
        self.cluster_centers = nn.Parameter(
            torch.empty(n_clusters, embedding_dim).normal_(std=0.01)
        )

    def forward(
        self,
        embeddings: torch.Tensor,
        cluster_centers: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Compute center loss.

        Args:
            embeddings: Latent vectors ``(B, D)``.
            cluster_centers: Center tensor ``(K, D)``.

        Returns:
            Dict with ``"loss"`` (weighted scalar) and ``"raw_center_loss"``
            (unweighted, detached scalar).
        """
        dists = torch.cdist(embeddings, cluster_centers, p=2)
        assignments = dists.argmin(dim=1)
        nearest = cluster_centers[assignments]
        raw = 0.5 * (embeddings - nearest).pow(2).sum(dim=1).mean()
        return {
            "loss": self.lambda_center * raw,
            "raw_center_loss": raw.detach(),
        }


class ClusteringAwareVAELoss(nn.Module):
    """Composite VAE loss combining reconstruction, KL, DEC and center losses.

    Designed for Phase-2 DCEC-style fine-tuning: the model is first pre-trained
    with reconstruction loss only (good PSNR), then fine-tuned with this
    composite loss to encourage latent clustering while preserving reconstruction
    quality.

    A single ``cluster_centers`` parameter is shared between the DEC and center
    sub-losses.  Call ``initialize_centers_from_embeddings`` before starting
    Phase-2 training to warm-start the centers from a K-Means fit on the
    pre-trained encoder's embeddings.

    Total loss::

        L = L_reconstruction + beta * L_KL + gamma * L_DEC + delta * L_center

    Args:
        n_clusters: Number of cluster centers ``K``.
        embedding_dim: Dimensionality ``D`` of the (reduced) latent vector.
        gamma: Weight for the DEC soft-assignment KL loss.  Set to ``0.0``
            to disable.  Default ``0.1``.
        delta: Weight for the center loss.  Set to ``0.0`` to disable.
            Default ``0.01``.
        vae_latent: Which VAE tensor to cluster — ``"mu"`` or ``"z"``.
            Default ``"mu"``.
        latent_reduction: How to reduce spatial latents ``(B, C, H, W)`` to
            vectors ``(B, D)`` before clustering.  ``"adaptive_avg_pool"``
            (default) or ``"flatten"``.
        alpha: Student t degrees of freedom passed to
            ``DECSoftAssignmentLoss``.  Default ``1.0``.
        lambda_center: Multiplicative weight inside ``CenterLoss``.  Controls
            the scale of the raw center loss *before* ``delta`` is applied.
            Default ``1.0``.
        **vae_loss_kwargs: All keyword arguments accepted by
            ``VariationalAutoencoderLoss`` (``reconstruction_loss``,
            ``reconstruction_weight``, ``beta``, ``free_bits``, etc.).

    Returns:
        ``forward`` returns a dict containing all keys from
        ``VariationalAutoencoderLoss`` plus ``"dec_loss"``,
        ``"weighted_dec_loss"``, ``"center_loss"``, and
        ``"weighted_center_loss"``.

    Example YAML:
        loss:
          _target_: ...ClusteringAwareVAELoss
          n_clusters: 8
          embedding_dim: 128
          gamma: 0.1
          delta: 0.01
          vae_latent: mu
          reconstruction_loss: mse
          reconstruction_weight: 1.0
          beta: 0.0001

    References:
        Xie et al., ICML 2016; Guo et al., IJCAI 2017; Wen et al., ECCV 2016.
    """

    def __init__(
        self,
        n_clusters: int,
        embedding_dim: int,
        gamma: float = 0.1,
        delta: float = 0.01,
        vae_latent: str = "mu",
        latent_reduction: str = "adaptive_avg_pool",
        alpha: float = 1.0,
        lambda_center: float = 1.0,
        **vae_loss_kwargs,
    ) -> None:
        super().__init__()
        if vae_latent not in ("mu", "z"):
            raise ValueError(f"vae_latent must be 'mu' or 'z', got {vae_latent!r}.")
        if latent_reduction not in ("adaptive_avg_pool", "flatten"):
            raise ValueError(
                f"latent_reduction must be 'adaptive_avg_pool' or 'flatten', "
                f"got {latent_reduction!r}."
            )
        self.n_clusters = n_clusters
        self.embedding_dim = embedding_dim
        self.gamma = gamma
        self.delta = delta
        self.vae_latent = vae_latent
        self.latent_reduction = latent_reduction

        self.cluster_centers = nn.Parameter(
            torch.empty(n_clusters, embedding_dim).normal_(std=0.01)
        )
        self.dec_loss = DECSoftAssignmentLoss(
            n_clusters=n_clusters, embedding_dim=embedding_dim, alpha=alpha
        )
        self.center_loss_fn = CenterLoss(
            n_clusters=n_clusters,
            embedding_dim=embedding_dim,
            lambda_center=lambda_center,
        )
        # Sub-losses own their cluster_centers only when used standalone.
        # Inside this wrapper the single self.cluster_centers is shared,
        # so the duplicate parameters in the sub-modules are removed to
        # avoid double memory usage and spurious optimizer updates.
        del self.dec_loss._parameters["cluster_centers"]
        del self.center_loss_fn._parameters["cluster_centers"]
        self.vae_loss = VariationalAutoencoderLoss(**vae_loss_kwargs)

    def initialize_centers_from_embeddings(
        self,
        embeddings: torch.Tensor,
        kmeans_max_iter: int = 100,
        kmeans_batch_size: int = 1024,
        random_state: Optional[int] = None,
    ) -> None:
        """Warm-start cluster centers via K-Means on pre-trained embeddings.

        Should be called once before Phase-2 training begins, using embeddings
        collected from a full forward pass of the pre-trained encoder.

        Args:
            embeddings: Raw latent tensor ``(N, D)`` or ``(N, C, H, W)``.
                Will be reduced according to ``latent_reduction``.
            kmeans_max_iter: Max K-Means iterations.  Default ``100``.
            kmeans_batch_size: Mini-batch size for K-Means.  Default ``1024``.
            random_state: Optional RNG seed.

        Returns:
            ``None``.
        """
        reduced = _reduce_latents(embeddings, self.latent_reduction)
        kmeans = MiniBatchKMeans(
            n_clusters=self.n_clusters,
            max_iter=kmeans_max_iter,
            batch_size=kmeans_batch_size,
            device=reduced.device,
            random_state=random_state,
        )
        kmeans.fit(reduced)
        self.cluster_centers.data.copy_(kmeans.centroids)

    def forward(
        self,
        output: VariationalAutoencoderOutput,
        target: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Compute composite clustering-aware VAE loss.

        Args:
            output: VAE forward output containing ``reconstruction``, ``mu``,
                ``logvar``, and ``z``.
            target: Reconstruction target tensor ``(B, C, H, W)``.

        Returns:
            Dict of scalar tensors suitable for ``LightningModule.log_dict``.

        Raises:
            TypeError: If ``output`` is not a ``VariationalAutoencoderOutput``.
        """
        if not isinstance(output, VariationalAutoencoderOutput):
            raise TypeError(
                "ClusteringAwareVAELoss expects VariationalAutoencoderOutput, "
                f"got {type(output).__name__}."
            )

        vae_result = self.vae_loss(output, target)

        latent = output.mu if self.vae_latent == "mu" else output.z
        z = _reduce_latents(latent, self.latent_reduction)

        zero = torch.zeros((), device=z.device)

        if self.gamma > 0:
            dec_result = self.dec_loss(z, self.cluster_centers)
            raw_dec = dec_result["loss"]
            weighted_dec = self.gamma * raw_dec
        else:
            raw_dec = zero
            weighted_dec = zero

        if self.delta > 0:
            center_result = self.center_loss_fn(z, self.cluster_centers)
            raw_center = center_result["loss"]
            weighted_center = self.delta * raw_center
        else:
            raw_center = zero
            weighted_center = zero

        total = vae_result["loss"] + weighted_dec + weighted_center

        result = dict(vae_result)
        result["loss"] = total
        result["dec_loss"] = raw_dec.detach() if self.gamma > 0 else zero
        result["weighted_dec_loss"] = weighted_dec.detach() if self.gamma > 0 else zero
        result["center_loss"] = raw_center.detach() if self.delta > 0 else zero
        result["weighted_center_loss"] = (
            weighted_center.detach() if self.delta > 0 else zero
        )
        return result
