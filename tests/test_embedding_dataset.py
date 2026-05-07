# -*- coding: utf-8 -*-
import unittest
import tempfile
import json
import shutil
from pathlib import Path
import pandas as pd
import torch
from shapely.geometry import Point
from pytorch_segmentation_models_trainer.dataset_loader.embedding_dataset import (
    EmbeddingDataset,
)


class TestEmbeddingDataset(unittest.TestCase):
    def setUp(self):
        self.test_dir = Path(tempfile.mkdtemp())
        self.n_samples = 10
        self.data = {
            "id": [f"id_{i}" for i in range(self.n_samples)],
            "embedding": [[float(i)] * 4 for i in range(self.n_samples)],
            "geometry": [Point(i, i).wkt for i in range(self.n_samples)],
        }

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def test_read_csv(self):
        csv_path = self.test_dir / "data.csv"
        # Converte lista de embeddings para string JSON para o CSV
        df_csv = pd.DataFrame(self.data)
        df_csv["embedding"] = df_csv["embedding"].apply(json.dumps)
        df_csv.to_csv(csv_path, index=False)

        ds = EmbeddingDataset(csv_path)
        self.assertEqual(len(ds), self.n_samples)
        item = ds[0]
        self.assertEqual(item["id"], "id_0")
        self.assertIsInstance(item["embedding"], torch.Tensor)
        self.assertEqual(item["embedding"].shape, (4,))
        self.assertIsInstance(item["geometry"], Point)

    def test_read_parquet(self):
        pq_path = self.test_dir / "data.parquet"
        pd.DataFrame(self.data).to_parquet(pq_path)

        ds = EmbeddingDataset(pq_path)
        self.assertEqual(len(ds), self.n_samples)
        self.assertIsInstance(ds[0]["embedding"], torch.Tensor)

    def test_read_geojson(self):
        json_path = self.test_dir / "data.geojson"
        features = []
        for i in range(self.n_samples):
            features.append(
                {
                    "type": "Feature",
                    "properties": {"id": f"id_{i}", "embedding": [float(i)] * 4},
                    "geometry": {"type": "Point", "coordinates": [float(i), float(i)]},
                }
            )
        geojson = {"type": "FeatureCollection", "features": features}

        with open(json_path, "w") as f:
            json.dump(geojson, f)

        ds = EmbeddingDataset(json_path)
        self.assertEqual(len(ds), self.n_samples)
        self.assertEqual(ds[0]["id"], "id_0")
        self.assertIsInstance(ds[0]["geometry"], Point)

    def test_read_without_geometry(self):
        csv_path = self.test_dir / "no_geom.csv"
        data_no_geom = {
            "id": [f"id_{i}" for i in range(5)],
            "embedding": [[float(i)] * 4 for i in range(5)],
        }
        df = pd.DataFrame(data_no_geom)
        df["embedding"] = df["embedding"].apply(json.dumps)
        df.to_csv(csv_path, index=False)

        # Define geometry_column como None ou uma coluna inexistente
        ds = EmbeddingDataset(csv_path, geometry_column=None)
        self.assertEqual(len(ds), 5)
        item = ds[0]
        self.assertNotIn("geometry", item)
        self.assertEqual(item["id"], "id_0")


if __name__ == "__main__":
    unittest.main()
