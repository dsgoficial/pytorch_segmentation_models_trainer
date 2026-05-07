# -*- coding: utf-8 -*-
import unittest
import torch
import numpy as np
from sklearn.cluster import MiniBatchKMeans as SklearnMiniBatchKMeans
from sklearn.metrics import adjusted_rand_score
from torch.utils.data import Dataset, DataLoader
from pytorch_segmentation_models_trainer.tools.kmeans.kmeans_calculator import (
    MiniBatchKMeans,
)
from pytorch_segmentation_models_trainer.tools.kmeans.kmeans_exporter import (
    KMeansClusteringTool,
)
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point


class EmbeddingDataset(Dataset):
    """Dataset de mock para testar interfaces de DataLoader."""

    def __init__(self, ids, embeddings, geometries):
        self.ids = ids
        self.embeddings = embeddings
        self.geometries = geometries

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        return {
            "id": self.ids[idx],
            "embedding": self.embeddings[idx],
            "geometry": self.geometries[idx],
        }


def collate_with_geometry(batch):
    """Collate function customizada que lida com geometrias do shapely."""
    ids = [item["id"] for item in batch]
    embeddings = torch.stack([item["embedding"] for item in batch])
    geometries = [item["geometry"] for item in batch]
    return {"id": ids, "embedding": embeddings, "geometry": geometries}


class TestMiniBatchKMeans(unittest.TestCase):
    # ... (SetUp mantido)
    def setUp(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.n_clusters = 3
        self.batch_size = 100
        # Criar dados sintéticos: 3 clusters bem separados
        torch.manual_seed(42)
        c1 = torch.randn(500, 32) + torch.tensor([5.0] * 32)
        c2 = torch.randn(500, 32) + torch.tensor([-5.0] * 32)
        c3 = torch.randn(500, 32) + torch.tensor([0.0] * 32)
        self.data = torch.cat([c1, c2, c3], dim=0)  # Mantém na CPU
        self.labels_true = np.concatenate(
            [np.zeros(500), np.ones(500), np.full(500, 2)]
        )

    def test_kmeans_plusplus_init_shape(self):
        """Testa se a inicialização K-Means++ retorna o shape correto."""
        kmeans = MiniBatchKMeans(n_clusters=self.n_clusters, device=self.device)
        centroids = kmeans._init_centroids(self.data)
        self.assertEqual(centroids.shape, (self.n_clusters, 32))
        self.assertEqual(centroids.device.type, self.device.type)

    def test_fit_predict_shapes(self):
        """Testa se o fit e predict retornam labels com shapes corretos."""
        kmeans = MiniBatchKMeans(
            n_clusters=self.n_clusters, batch_size=self.batch_size, device=self.device
        )
        kmeans.fit(self.data)
        labels = kmeans.predict(self.data)

        self.assertEqual(labels.shape, (self.data.shape[0],))
        self.assertEqual(kmeans.centroids.shape, (self.n_clusters, 32))

    def test_convergence_simple_data(self):
        """Testa se o algoritmo converge em dados simples (ARI alto)."""
        kmeans = MiniBatchKMeans(
            n_clusters=self.n_clusters,
            batch_size=self.batch_size,
            max_iter=50,
            device=self.device,
        )
        kmeans.fit(self.data)
        labels = kmeans.predict(self.data).cpu().numpy()

        ari = adjusted_rand_score(self.labels_true, labels)
        self.assertGreater(ari, 0.9, f"ARI muito baixo: {ari}")

    def test_fit_from_dataloader(self):
        """Testa o treinamento a partir de um DataLoader."""
        n_samples, n_features = 150, 16
        ids = [f"id_{i}" for i in range(n_samples)]
        embeddings = torch.randn(n_samples, n_features)
        geometries = [Point(i, i) for i in range(n_samples)]

        dataset = EmbeddingDataset(ids, embeddings, geometries)
        dataloader = DataLoader(
            dataset, batch_size=50, collate_fn=collate_with_geometry
        )

        kmeans = MiniBatchKMeans(n_clusters=self.n_clusters, device=self.device)
        kmeans.fit_from_dataloader(dataloader, init_samples=100)

        self.assertIsNotNone(kmeans.centroids)
        self.assertEqual(kmeans.centroids.shape, (self.n_clusters, n_features))


class TestKMeansClusteringTool(unittest.TestCase):
    def setUp(self):
        self.n_samples = 150
        self.n_features = 16
        self.n_clusters = 3

        self.ids = [f"id_{i}" for i in range(self.n_samples)]
        self.embeddings = torch.randn(self.n_samples, self.n_features)
        self.geometries = [Point(i, i) for i in range(self.n_samples)]

        self.tool = KMeansClusteringTool(n_clusters=self.n_clusters, batch_size=50)

    def test_run_clustering_returns_gdf(self):
        """Testa se a execução principal retorna um GeoDataFrame com as colunas esperadas."""
        gdf = self.tool.run(
            ids=self.ids, embeddings=self.embeddings, geometries=self.geometries
        )

        self.assertIsInstance(gdf, gpd.GeoDataFrame)
        self.assertEqual(len(gdf), self.n_samples)
        self.assertIn("id", gdf.columns)
        self.assertIn("cluster_id", gdf.columns)
        self.assertIn("geometry", gdf.columns)
        self.assertEqual(gdf["cluster_id"].nunique(), self.n_clusters)

    def test_run_from_dataloader(self):
        """Testa a execução da ferramenta a partir de um DataLoader."""
        dataset = EmbeddingDataset(self.ids, self.embeddings, self.geometries)
        dataloader = DataLoader(
            dataset, batch_size=50, collate_fn=collate_with_geometry
        )

        gdf = self.tool.run_from_dataloader(dataloader)

        self.assertIsInstance(gdf, gpd.GeoDataFrame)
        self.assertEqual(len(gdf), self.n_samples)
        self.assertIn("cluster_id", gdf.columns)
        self.assertEqual(gdf["cluster_id"].nunique(), self.n_clusters)


if __name__ == "__main__":
    unittest.main()
