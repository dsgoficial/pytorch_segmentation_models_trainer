# -*- coding: utf-8 -*-
import unittest
import torch
import numpy as np
import tempfile
from unittest.mock import patch
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

    def test_kmeans_plusplus_handles_zero_distances(self):
        """Covers the duplicate-points fallback in K-Means++ initialization."""
        kmeans = MiniBatchKMeans(n_clusters=3, device=self.device)
        data = torch.ones((8, 4))

        centroids = kmeans._init_centroids(data)

        self.assertEqual(centroids.shape, (3, 4))
        self.assertTrue(torch.all(centroids == 1))

    def test_fit_predict_shapes(self):
        """Testa se o fit e predict retornam labels com shapes corretos."""
        kmeans = MiniBatchKMeans(
            n_clusters=self.n_clusters, batch_size=self.batch_size, device=self.device
        )
        kmeans.fit(self.data)
        labels = kmeans.predict(self.data)

        self.assertEqual(labels.shape, (self.data.shape[0],))
        self.assertEqual(kmeans.centroids.shape, (self.n_clusters, 32))

    def test_fit_breaks_when_center_shift_is_below_tolerance(self):
        kmeans = MiniBatchKMeans(
            n_clusters=2, batch_size=4, max_iter=5, tol=1e9, device=self.device
        )

        returned = kmeans.fit(torch.ones((8, 2)))

        self.assertIs(returned, kmeans)

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

    def test_fit_from_dataloader_breaks_when_already_converged(self):
        dataset = torch.utils.data.TensorDataset(torch.ones((12, 2)))

        class TensorOnlyDataset(Dataset):
            def __len__(self):
                return len(dataset)

            def __getitem__(self, idx):
                return dataset[idx][0]

        dataloader = DataLoader(TensorOnlyDataset(), batch_size=4)
        kmeans = MiniBatchKMeans(
            n_clusters=2, batch_size=4, max_iter=3, tol=1e9, device=self.device
        )

        returned = kmeans.fit_from_dataloader(dataloader, init_samples=8)

        self.assertIs(returned, kmeans)

    def test_predict_before_fit_raises(self):
        kmeans = MiniBatchKMeans(n_clusters=self.n_clusters, device=self.device)

        with self.assertRaises(ValueError):
            kmeans.predict(self.data)

    def test_find_optimal_k_elbow_method(self):
        """Testa se a busca adaptativa de K retorna um valor razoável."""
        from pytorch_segmentation_models_trainer.tools.kmeans.kmeans_calculator import (
            find_optimal_k_elbow_method,
        )

        # Usamos dados com 3 clusters claros
        optimal_k = find_optimal_k_elbow_method(
            self.data, k_min=2, k_max=10, step=1, random_state=42
        )
        # O 'cotovelo' deve estar próximo de 3
        self.assertGreaterEqual(optimal_k, 2)
        self.assertLessEqual(optimal_k, 5)


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

    def test_run_accepts_numpy_and_returns_dataframe_without_geometry(self):
        df = self.tool.run(ids=self.ids, embeddings=self.embeddings.numpy())

        self.assertIsInstance(df, pd.DataFrame)
        self.assertNotIn("geometry", df.columns)

    def test_run_validates_input_types_and_lengths(self):
        with self.assertRaises(ValueError):
            self.tool.run(ids=self.ids, embeddings=[[1, 2]])
        with self.assertRaises(ValueError):
            self.tool.run(ids=self.ids[:-1], embeddings=self.embeddings)
        with self.assertRaises(ValueError):
            self.tool.run(
                ids=self.ids,
                embeddings=self.embeddings,
                geometries=self.geometries[:-1],
            )

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

    def test_run_from_dataloader_returns_dataframe_without_geometry(self):
        class NoGeometryDataset(Dataset):
            def __len__(self):
                return len(self_ids)

            def __getitem__(self, idx):
                return {"id": self_ids[idx], "embedding": self_embeddings[idx]}

        self_ids = self.ids
        self_embeddings = self.embeddings
        dataloader = DataLoader(NoGeometryDataset(), batch_size=50)

        df = self.tool.run_from_dataloader(dataloader)

        self.assertIsInstance(df, pd.DataFrame)
        self.assertNotIn("geometry", df.columns)

    def test_run_from_dataloader_rejects_non_dict_batches(self):
        dataloader = DataLoader(
            torch.utils.data.TensorDataset(self.embeddings), batch_size=10
        )
        self.tool.model.fit_from_dataloader = lambda *args, **kwargs: None

        with self.assertRaises(ValueError):
            self.tool.run_from_dataloader(dataloader)

    def test_export_helpers(self):
        gdf = gpd.GeoDataFrame(
            {"id": ["a"], "cluster_id": [0]},
            geometry=[Point(0, 0)],
            crs="EPSG:4326",
        )
        with tempfile.TemporaryDirectory() as tmp:
            output_path = f"{tmp}/clusters.parquet"
            self.tool.export_to_parquet(gdf, output_path)
            self.assertTrue(pd.read_parquet(output_path).shape[0], 1)

        with (
            patch("geopandas.GeoDataFrame.to_postgis") as mock_to_postgis,
            patch("sqlalchemy.create_engine") as mock_create_engine,
        ):
            self.tool.export_to_postgis(gdf, "clusters", "postgresql://example/db")

        mock_create_engine.assert_called_once_with("postgresql://example/db")
        mock_to_postgis.assert_called_once()


if __name__ == "__main__":
    unittest.main()
