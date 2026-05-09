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
    def __init__(self, out_channels=128, return_list=False):
        super().__init__()
        self.conv = nn.Conv2d(3, out_channels, 3)
        self.return_list = return_list

    def forward(self, x):
        feat = self.conv(x)
        return [feat, feat] if self.return_list else feat


class MockModel(nn.Module):
    def __init__(
        self,
        latent_dim=64,
        use_huggingface=False,
        no_proj=False,
        return_list=False,
        ndim=4,
    ):
        super().__init__()
        self.encoder = MockEncoder(out_channels=128, return_list=return_list)
        self.use_huggingface = use_huggingface
        self.ndim = ndim
        if not no_proj:
            self.latent_proj = (
                nn.Conv2d(128, latent_dim, 1)
                if ndim == 4
                else nn.Linear(128, latent_dim)
            )

    def forward(self, x):
        z = self.encoder(x)
        if isinstance(z, list):
            z = z[-1]

        if hasattr(self, "latent_proj"):
            # Ajusta formato para linear se necessário
            if isinstance(self.latent_proj, nn.Linear) and z.ndim == 4:
                z = torch.mean(z, dim=(2, 3))
            z = self.latent_proj(z)

        if self.ndim == 3 and z.ndim == 4:
            z = torch.mean(z, dim=2)
        elif self.ndim == 2 and z.ndim == 4:
            z = torch.mean(z, dim=(2, 3))

        return z


class MockModelSimple(nn.Module):
    def forward(self, x):
        return torch.mean(x, dim=(2, 3))


class MockSystem(pl.LightningModule):
    def __init__(self, model):
        super().__init__()
        self.model = model


def test_extract_all_latents_variations():
    device = torch.device("cpu")

    # Case 1: Standard with latent_proj and 4D output
    model = MockModel(latent_dim=64)
    system = MockSystem(model)
    dataloader = DataLoader(TensorDataset(torch.randn(4, 3, 32, 32)), batch_size=2)
    latents = extract_all_latents(system, dataloader, device)
    assert latents.shape == (4, 64)

    # Case 2: Use HuggingFace branch
    # Em MockModel, if use_huggingface is True, extract_all_latents calls encoder directly
    # Our MockModel encoder returns 128 channels.
    model_hf = MockModel(latent_dim=64, use_huggingface=True, no_proj=True)
    latents = extract_all_latents(MockSystem(model_hf), dataloader, device)
    # The MockModel logic in extract_all_latents when use_huggingface is true
    # will mean z = encoder(x), which has 128 channels at output of conv.
    # Then it is averaged to (4, 128)
    assert latents.shape == (4, 128)

    # Case 3: Encoder returns list
    model_list = MockModel(latent_dim=64, return_list=True)
    latents = extract_all_latents(MockSystem(model_list), dataloader, device)
    assert latents.shape == (4, 64)

    # Case 4: No encoder (Direct call)
    model_simple = MockModelSimple()
    latents = extract_all_latents(MockSystem(model_simple), dataloader, device)
    assert latents.shape == (4, 3)

    # Case 5: 3D output (e.g. transformers)
    model_3d = MockModel(latent_dim=64, ndim=3, no_proj=True)
    # Simulate 3D output from encoder
    model_3d.encoder.forward = lambda x: torch.randn(x.shape[0], 10, 128)
    latents = extract_all_latents(MockSystem(model_3d), dataloader, device)
    # 3D (B, L, D) -> averaged over L -> (B, D)
    assert latents.shape == (4, 128)

    # Case 6: Different batch formats (dict)
    dataloader_dict = DataLoader(
        [{"image": torch.randn(3, 32, 32)} for _ in range(4)], batch_size=2
    )
    latents = extract_all_latents(MockSystem(model), dataloader_dict, device)
    assert latents.shape == (4, 64)

    # Case 7: Different batch formats (list/tuple)
    dataloader_tuple = DataLoader(
        [(torch.randn(3, 32, 32), torch.tensor(1)) for _ in range(4)], batch_size=2
    )
    latents = extract_all_latents(MockSystem(model), dataloader_tuple, device)
    assert latents.shape == (4, 64)

    # Case 8: 2D output (B, D) - should stay as is
    model_2d = MockModel(latent_dim=64, ndim=2, no_proj=True)
    model_2d.encoder.forward = lambda x: torch.randn(x.shape[0], 128)
    latents = extract_all_latents(MockSystem(model_2d), dataloader, device)
    assert latents.shape == (4, 128)


def test_extract_all_latents_errors():
    device = torch.device("cpu")
    model = MockModelSimple()
    system = MockSystem(model)

    # Test x is None error using a custom iterator to avoid DataLoader collate error
    class NoneDataset:
        def __iter__(self):
            yield None

    with pytest.raises(ValueError, match="Could not extract image from batch"):
        extract_all_latents(system, NoneDataset(), device)


def test_kmeans_clustering_tool_weights():
    device = torch.device("cpu")
    num_clusters = 3
    tool = KMeansClusteringTool(n_clusters=num_clusters, device=device)

    # Test error before fit
    with pytest.raises(ValueError, match="Model must be fitted"):
        tool.get_cluster_weights(mode="density")

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
    expected_mean = torch.tensor(1.0 / (num_clusters**0.5))
    assert torch.allclose(weights_density.mean(), expected_mean)

    # Test Sqrt heuristic weights (DDOQ-LULC)
    weights_sqrt = tool.get_cluster_weights(mode="sqrt")
    assert weights_sqrt.shape == (num_clusters,)
    assert torch.allclose(weights_sqrt.mean(), expected_mean)

    # Test invalid mode
    with pytest.raises(ValueError, match="Invalid weight mode"):
        tool.get_cluster_weights(mode="invalid")


def test_kmeans_clustering_tool_medoids():
    device = torch.device("cpu")
    num_clusters = 5
    num_samples = 100
    latents = torch.randn(num_samples, 8)

    tool = KMeansClusteringTool(n_clusters=num_clusters, device=device)

    # Test error before fit
    dataloader = DataLoader(latents, batch_size=20)
    with pytest.raises(ValueError, match="Model must be fitted before finding medoids"):
        tool.get_medoids_from_dataloader(dataloader)

    tool.fit(latents)

    # Test with standard tensor dataloader
    medoid_indices = tool.get_medoids_from_dataloader(dataloader)
    assert len(medoid_indices) == num_clusters

    # Test with dict dataloader (id key)
    data_dict = [{"embedding": latents[i], "id": i} for i in range(num_samples)]
    dataloader_dict = DataLoader(data_dict, batch_size=20)
    medoid_indices = tool.get_medoids_from_dataloader(dataloader_dict)
    assert len(medoid_indices) == num_clusters
    assert all(idx in range(num_samples) for idx in medoid_indices)

    # Test with dict dataloader (latents and index keys)
    data_dict2 = [{"latents": latents[i], "index": i} for i in range(num_samples)]
    dataloader_dict2 = DataLoader(data_dict2, batch_size=20)
    medoid_indices = tool.get_medoids_from_dataloader(dataloader_dict2)
    assert len(medoid_indices) == num_clusters


def test_save_load_ddoq_results(tmp_path):
    output_path = os.path.join(tmp_path, "results.pt")
    indices = [1, 2, 3]
    weights = torch.tensor([0.2, 0.3, 0.5])

    save_ddoq_results(indices, weights, output_path)
    assert os.path.exists(output_path)

    loaded_indices, loaded_weights = load_ddoq_results(output_path)
    assert loaded_indices == indices
    assert torch.allclose(loaded_weights, weights)
