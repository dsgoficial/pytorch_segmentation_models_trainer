# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import pytorch_lightning as pl
import pytest
from pytorch_segmentation_models_trainer.dataset_distillation import (
    extract_all_latents,
    find_coreset_medoids,
    create_distilled_dataloader,
)


class MockEncoder(nn.Module):
    def __init__(self, out_channels=128):
        super().__init__()
        self.conv = nn.Conv2d(3, out_channels, 3)

    def forward(self, x):
        # Simulate returning a tensor (some encoders return a list, handled in code)
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
    assert latents.device.type == "cpu"


def test_find_coreset_medoids():
    device = torch.device("cpu")
    num_samples = 50
    latent_dim = 16
    num_clusters = 5

    # Create dummy latents with clear clusters to ensure KMeans works as expected
    latents = torch.randn(num_samples, latent_dim)

    medoid_indices = find_coreset_medoids(latents, num_clusters, device)

    assert isinstance(medoid_indices, list)
    assert len(medoid_indices) == num_clusters
    assert all(isinstance(idx, int) for idx in medoid_indices)
    assert all(0 <= idx < num_samples for idx in medoid_indices)
    # Check for uniqueness
    assert len(set(medoid_indices)) == num_clusters


def test_create_distilled_dataloader():
    num_samples = 20
    images = torch.randn(num_samples, 3, 32, 32)
    dataset = TensorDataset(images)
    medoid_indices = [0, 5, 10]

    dataloader = create_distilled_dataloader(dataset, medoid_indices, batch_size=2)

    assert isinstance(dataloader, DataLoader)
    assert len(dataloader.dataset) == 3

    batch = next(iter(dataloader))
    # TensorDataset returns a list [tensor]
    assert isinstance(batch, (list, tuple))
    assert batch[0].shape == (2, 3, 32, 32)
