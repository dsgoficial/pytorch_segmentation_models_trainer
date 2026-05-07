# -*- coding: utf-8 -*-
"""
Dataset distillation utilities using Coreset of Medoids.
Inspired by "Dataset Distillation as Pushforward Optimal Quantization".
"""

from typing import List, Optional, Union
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, Subset
import pytorch_lightning as pl
import numpy as np
from tqdm import tqdm

from pytorch_segmentation_models_trainer.tools.kmeans.kmeans_calculator import (
    MiniBatchKMeans,
)


def extract_all_latents(
    model: pl.LightningModule, dataloader: DataLoader, device: torch.device
) -> torch.Tensor:
    """
    Extracts latents from all images in the dataloader using the model's encoder.

    Args:
        model: Trained PyTorch Lightning module (e.g., AutoencoderSystem).
        dataloader: DataLoader containing unlabeled images.
        device: Device to perform extraction on (e.g., 'cuda' or 'cpu').

    Returns:
        A CPU tensor of shape (N, latent_dim) where N is the total number of samples.
    """
    model.to(device)
    model.eval()

    # Identify the inner model if it's a LightningModule wrapper
    inner_model = getattr(model, "model", model)

    latents = []
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Extracting latents"):
            # Handle different batch formats (images, [images, masks], etc.)
            if isinstance(batch, (list, tuple)):
                x = batch[0]
            elif isinstance(batch, dict):
                x = batch.get("image") or batch.get("x")
            else:
                x = batch

            if x is None:
                raise ValueError("Could not extract image from batch.")

            x = x.to(device)

            # Extract features following GenericAutoencoder-like structure
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
                # Fallback to model forward if no explicit encoder is found
                z = inner_model(x)

            # Apply latent projection if it exists (GenericAutoencoder uses this)
            if hasattr(inner_model, "latent_proj"):
                z = inner_model.latent_proj(z)

            # Global Average Pooling if the output is spatial (B, C, H, W)
            if z.ndim == 4:
                z = torch.mean(z, dim=(2, 3))
            elif z.ndim == 3:
                # Handle sequence models (B, L, C) -> (B, C)
                z = torch.mean(z, dim=1)

            latents.append(z.cpu())

    return torch.cat(latents, dim=0)


def find_coreset_medoids(
    latents: torch.Tensor, num_clusters: int, device: torch.device
) -> List[int]:
    """
    Performs K-Means clustering and finds the medoids (closest real samples to centroids).
    Uses MiniBatchKMeans (GPU support) if a CUDA device is provided, otherwise falls back
    to sklearn.cluster.KMeans on CPU.

    Args:
        latents: (N, latent_dim) tensor of embeddings.
        num_clusters: Number of clusters to find (size of the coreset).
        device: Device to perform distance calculations on.

    Returns:
        List of global dataset indices corresponding to the selected medoids.
    """

    if device.type == "cuda":
        # Use custom GPU-optimized MiniBatchKMeans
        kmeans = MiniBatchKMeans(
            n_clusters=num_clusters,
            device=device,
            random_state=42,
            batch_size=min(1024, latents.shape[0]),
        )
        kmeans.fit(latents)
        centroids = kmeans.centroids
        # Predict clusters for each latent vector
        labels = kmeans.predict(latents).cpu().numpy()
    else:
        # Fallback to scikit-learn for CPU
        from sklearn.cluster import KMeans

        latents_np = latents.numpy()
        kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init="auto")
        labels = kmeans.fit_predict(latents_np)
        centroids = torch.from_numpy(kmeans.cluster_centers_).to(device)

    # Move latents and centroids to the specified device for fast distance calculation
    latents_gpu = latents.to(device)
    centroids = centroids.to(device)
    medoid_indices = []

    for i in range(num_clusters):
        # Identify indices of latents belonging to this cluster
        cluster_mask = labels == i
        cluster_indices = np.where(cluster_mask)[0]

        if len(cluster_indices) == 0:
            continue

        # Extract latents for this cluster
        cluster_latents = latents_gpu[cluster_mask]
        centroid = centroids[i].unsqueeze(0)  # Shape: (1, latent_dim)

        # Calculate L2 distances between the centroid and all points in the cluster
        dists = torch.cdist(centroid, cluster_latents)  # Shape: (1, N_cluster)

        # Find the index of the point closest to the centroid within this cluster
        local_min_idx = torch.argmin(dists).item()
        global_min_idx = cluster_indices[local_min_idx]

        medoid_indices.append(int(global_min_idx))

    return medoid_indices


def create_distilled_dataloader(
    original_dataset: Dataset,
    medoid_indices: List[int],
    batch_size: int,
    num_workers: int = 4,
) -> DataLoader:
    """
    Creates a new DataLoader containing only the selected medoid samples.

    Args:
        original_dataset: The full unlabeled dataset.
        medoid_indices: Indices of the selected medoid samples.
        batch_size: Batch size for the new DataLoader.
        num_workers: Number of workers for data loading.

    Returns:
        A PyTorch DataLoader serving the distilled dataset.
    """
    distilled_subset = Subset(original_dataset, medoid_indices)

    return DataLoader(
        distilled_subset, batch_size=batch_size, num_workers=num_workers, shuffle=False
    )
