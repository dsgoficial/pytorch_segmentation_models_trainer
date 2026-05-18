# -*- coding: utf-8 -*-
"""
Dataset distillation utilities using Coreset of Medoids.
Inspired by "Dataset Distillation as Pushforward Optimal Quantization".
"""

from typing import List, Optional, Tuple
import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from tqdm import tqdm

from pytorch_segmentation_models_trainer.tools.kmeans.kmeans_calculator import (
    MiniBatchKMeans,
)


class KMeansClusteringTool:
    """
    Orchestrator for DDOQ (Dataset Distillation by Optimal Quantization).
    Handles clustering, weight calculation (Voronoi weights), and Medoid search.
    """

    def __init__(
        self,
        n_clusters: int,
        device: torch.device,
        random_state: int = 42,
    ):
        self.n_clusters = n_clusters
        self.device = device
        self.random_state = random_state
        self.model = None

    def fit(self, latents: torch.Tensor, batch_size: int = 1024, max_iter: int = 100):
        """Trains the MiniBatchKMeans model."""
        self.model = MiniBatchKMeans(
            n_clusters=self.n_clusters,
            device=self.device,
            random_state=self.random_state,
            batch_size=batch_size,
            max_iter=max_iter,
        )
        self.model.fit(latents)
        return self

    def predict(self, latents: torch.Tensor) -> torch.Tensor:
        """Assign embeddings to the fitted cluster centers.

        Args:
            latents: Embedding tensor with shape ``(N, D)``.

        Returns:
            Cluster id tensor with shape ``(N,)``.
        """
        if self.model is None:
            raise ValueError("Model must be fitted before predicting clusters.")
        return self.model.predict(latents)

    def get_cluster_weights(
        self, mode: str = "density", labels: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Calculates cluster weights for Student training.

        Args:
            mode:
                - "uniform": Classical K-Means coreset (all weights = 1.0).
                - "density": Vanilla DDOQ (weights proportional to counts).
                - "sqrt": Improved DDOQ (weights proportional to sqrt of counts).
            labels: Optional cluster assignment for every original image. When
                provided, weights use exact Voronoi masses from those labels;
                otherwise they fall back to Mini-Batch K-Means internal counts.
        """
        if mode == "uniform":
            # Classical K-Means: equal weights without density scaling.
            return torch.ones(self.n_clusters)

        if labels is not None:
            counts = torch.bincount(
                labels.detach().cpu().long(), minlength=self.n_clusters
            ).float()
        elif self.model is not None and self.model.counts is not None:
            counts = self.model.counts.clone().float().cpu()
        else:
            raise ValueError(
                "Model must be fitted before calculating density-based weights."
            )
        if mode == "density":
            weights = counts
        elif mode == "sqrt":
            weights = torch.sqrt(counts + 1e-8)
        else:
            raise ValueError(f"Invalid weight mode: {mode}")

        # DDOQ Normalization: Scales weights so the average is 1/sqrt(K).
        # This keeps Loss magnitude consistent across modes and coreset sizes.
        # preserving the original gradient scale for the optimizer.
        weights = (weights / (torch.sum(weights) + 1e-8)) * torch.sqrt(
            torch.tensor(self.n_clusters)
        )
        return weights

    def get_medoids_from_dataloader(
        self, dataloader: DataLoader
    ) -> Tuple[List[int], List[float]]:
        """
        OOM-safe search for Medoids (real samples closest to centroids).

        Returns:
            A tuple containing (medoid_indices, medoid_weights).
        """
        if self.model is None or self.model.centroids is None:
            raise ValueError("Model must be fitted before finding medoids.")

        centroids = self.model.centroids.to(self.device)
        min_distances = torch.full((self.n_clusters,), float("inf"), device=self.device)
        medoid_indices = [-1] * self.n_clusters

        # We assume the dataloader returns batches with an "index" or "id" field
        # or we track the global index manually if shuffle=False.
        global_idx = 0

        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Finding medoids"):
                if isinstance(batch, dict):
                    emb = batch.get("embedding")
                    if emb is None:
                        emb = batch.get("latents")

                    batch_ids = batch.get("id")
                    if batch_ids is None:
                        batch_ids = batch.get("index")
                else:
                    emb = batch
                    batch_ids = None

                if emb is None:
                    continue

                emb = emb.to(self.device).float()

                # Calculate L2 distances (Batch_Size, K)
                dist_matrix = torch.cdist(emb, centroids, p=2)

                # Find closest samples in this batch for each centroid
                batch_min_dist, batch_min_idx = torch.min(dist_matrix, dim=0)

                for k in range(self.n_clusters):
                    if batch_min_dist[k] < min_distances[k]:
                        min_distances[k] = batch_min_dist[k]
                        if batch_ids is not None:
                            medoid_indices[k] = int(batch_ids[batch_min_idx[k].item()])
                        else:
                            medoid_indices[k] = global_idx + batch_min_idx[k].item()

                global_idx += emb.shape[0]

        return medoid_indices


def extract_all_latents(
    model: pl.LightningModule, dataloader: DataLoader, device: torch.device
) -> torch.Tensor:
    """
    Extracts latents from all images in the dataloader using the model's encoder.
    """
    model.to(device)
    model.eval()

    inner_model = getattr(model, "model", model)

    latents = []
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Extracting latents"):
            if isinstance(batch, (list, tuple)):
                x = batch[0]
            elif isinstance(batch, dict):
                x = batch.get("image")
                if x is None:
                    x = batch.get("x")
            else:
                x = batch

            if x is None:
                raise ValueError("Could not extract image from batch.")

            x = x.to(device)

            if hasattr(inner_model, "encoder"):
                encoder = inner_model.encoder
                if (
                    hasattr(inner_model, "use_huggingface")
                    and inner_model.use_huggingface
                ):
                    z = encoder(x)
                else:
                    features = encoder(x)
                    z = (
                        features[-1]
                        if isinstance(features, (list, tuple))
                        else features
                    )
            else:
                z = inner_model(x)

            if hasattr(inner_model, "latent_proj"):
                z = inner_model.latent_proj(z)

            if z.ndim == 4:
                z = torch.mean(z, dim=(2, 3))
            elif z.ndim == 3:
                z = torch.mean(z, dim=1)

            latents.append(z.cpu())

    return torch.cat(latents, dim=0)


def save_ddoq_results(indices: List[int], weights: torch.Tensor, output_path: str):
    """Saves medoid indices and their Voronoi weights."""
    data = {"indices": indices, "weights": weights.cpu().tolist()}
    torch.save(data, output_path)
    print(f"DDOQ results saved to {output_path}")


def load_ddoq_results(path: str) -> Tuple[List[int], torch.Tensor]:
    """Loads medoid indices and Voronoi weights."""
    data = torch.load(path)
    return data["indices"], torch.tensor(data["weights"])
