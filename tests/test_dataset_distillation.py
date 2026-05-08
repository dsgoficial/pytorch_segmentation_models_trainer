# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import pytorch_lightning as pl
import pytest
import os
from pytorch_segmentation_models_trainer.dataset_distillation import (
    extract_all_latents,
    KMeansClusteringTool,
    save_ddoq_results,
    load_ddoq_results,
)


class MockEncoder(nn.Module):
    def __init__(self, out_channels=128):
        super().__init__()
        self.conv = nn.Conv2d(3, out_channels, 3)

    def forward(self, x):
        return self.conv(x)


class MockModel(nn.Module):
    def __init__(self, latent_dim=64):
        super().__init__()
        self.encoder = MockEncoder(out_channels=128)
        self.latent_proj = nn.Conv2d(128, latent_dim, 1)

    def forward(self, x):
        z = self.encoder(x)
        return self.latent_proj(z)


class MockSystem(pl.LightningModule):
    def __init__(self, model):
        super().__init__()
        self.model = model


def test_extract_all_latents():
    device = torch.device("cpu")
    latent_dim = 64
    model = MockModel(latent_dim=latent_dim)
    system = MockSystem(model)

    # Create dummy dataset
    images = torch.randn(10, 3, 32, 32)
    dataset = TensorDataset(images)
    dataloader = DataLoader(dataset, batch_size=2)

    latents = extract_all_latents(system, dataloader, device)

    assert isinstance(latents, torch.Tensor)
    assert latents.shape == (10, latent_dim)


def test_kmeans_clustering_tool_weights():
    device = torch.device("cpu")
    num_clusters = 3
    tool = KMeansClusteringTool(n_clusters=num_clusters, device=device)

    # 3 clusters
    c1 = torch.randn(100, 8) + 5
    c2 = torch.randn(200, 8) - 5
    c3 = torch.randn(300, 8)
    latents = torch.cat([c1, c2, c3], dim=0)

    tool.fit(latents)

    # Test Uniform weights (Classical)
    weights_uniform = tool.get_cluster_weights(mode="uniform")
    assert torch.allclose(weights_uniform, torch.tensor([1.0, 1.0, 1.0]))

    # Test Density weights (Vanilla DDOQ)
    weights_density = tool.get_cluster_weights(mode="density")
    assert weights_density.shape == (num_clusters,)
    assert torch.allclose(weights_density.mean(), torch.tensor(1.0))

    # Test Sqrt heuristic weights (DDOQ-LULC)
    weights_sqrt = tool.get_cluster_weights(mode="sqrt")
    assert weights_sqrt.shape == (num_clusters,)
    assert torch.allclose(weights_sqrt.mean(), torch.tensor(1.0))
    # Sqrt should be more uniform than density for unbalanced clusters
    assert torch.std(weights_sqrt) < torch.std(weights_density)


def test_kmeans_clustering_tool_medoids():
    device = torch.device("cpu")
    num_clusters = 5
    num_samples = 100
    latents = torch.randn(num_samples, 8)

    tool = KMeansClusteringTool(n_clusters=num_clusters, device=device)
    tool.fit(latents)

    dataloader = DataLoader(latents, batch_size=20)
    medoid_indices = tool.get_medoids_from_dataloader(dataloader)

    assert len(medoid_indices) == num_clusters
    assert all(0 <= idx < num_samples for idx in medoid_indices)
    assert len(set(medoid_indices)) == num_clusters


def test_save_load_ddoq_results(tmp_path):
    output_path = os.path.join(tmp_path, "results.pt")
    indices = [1, 2, 3]
    weights = torch.tensor([0.2, 0.3, 0.5])

    save_ddoq_results(indices, weights, output_path)
    assert os.path.exists(output_path)

    loaded_indices, loaded_weights = load_ddoq_results(output_path)
    assert loaded_indices == indices
    assert torch.allclose(loaded_weights, weights)
