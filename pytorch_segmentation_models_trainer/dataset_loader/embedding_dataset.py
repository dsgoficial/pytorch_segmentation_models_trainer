# -*- coding: utf-8 -*-
import glob
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from shapely import wkt
from shapely.geometry import shape

logger = logging.getLogger(__name__)


class EmbeddingDataset(Dataset):
    """Dataset para ingestão de embeddings, IDs e geometrias de arquivos em disco.

    Suporta arquivos CSV, JSON/GeoJSON e Parquet/GeoParquet.

    Args:
        data_path: Caminho para um arquivo ou diretório contendo os arquivos de dados.
        id_column: Nome da coluna que contém o ID único.
        embedding_column: Nome da coluna que contém o embedding (pode ser uma lista ou string JSON).
        geometry_column: Nome da coluna que contém a geometria (opcional).
        extension: Filtro de extensão se data_path for um diretório (ex: 'parquet', 'csv', 'json').
    """

    def __init__(
        self,
        data_path: Union[str, Path],
        id_column: str = "id",
        embedding_column: str = "embedding",
        geometry_column: Optional[str] = "geometry",
        extension: Optional[str] = None,
    ) -> None:
        self.data_path = Path(data_path)
        self.id_column = id_column
        self.embedding_column = embedding_column
        self.geometry_column = geometry_column

        self.files = self._get_files(extension)
        self.df = self._load_data()
        self.len = len(self.df)

    def _get_files(self, extension: Optional[str]) -> List[Path]:
        if self.data_path.is_file():
            return [self.data_path]

        pattern = f"*.{extension}" if extension else "*"
        files = list(self.data_path.glob(pattern))
        # Filtra apenas extensões suportadas se não foi especificada
        if not extension:
            supported = {".csv", ".json", ".geojson", ".parquet"}
            files = [f for f in files if f.suffix.lower() in supported]

        if not files:
            raise FileNotFoundError(f"Nenhum arquivo encontrado em {self.data_path}")

        return sorted(files)

    def _load_data(self) -> pd.DataFrame:
        dfs = []
        for file in self.files:
            ext = file.suffix.lower()
            if ext == ".csv":
                dfs.append(pd.read_csv(file))
            elif ext == ".parquet":
                dfs.append(pd.read_parquet(file))
            elif ext in {".json", ".geojson"}:
                with open(file, "r") as f:
                    data = json.load(f)
                    # Se for GeoJSON clássico (FeatureCollection)
                    if (
                        isinstance(data, dict)
                        and data.get("type") == "FeatureCollection"
                    ):
                        rows = []
                        for feat in data["features"]:
                            row = feat["properties"].copy()
                            # Só adiciona geometria se a coluna foi pedida
                            if self.geometry_column:
                                row[self.geometry_column] = feat["geometry"]
                            rows.append(row)
                        dfs.append(pd.DataFrame(rows))
                    else:
                        dfs.append(pd.DataFrame(data))

        return pd.concat(dfs, ignore_index=True)

    def _parse_geometry(self, geom: Any) -> Any:
        if geom is None or pd.isna(geom):
            return None
        if isinstance(geom, str):
            try:
                return wkt.loads(geom)
            except Exception:
                # Tenta como JSON se falhar WKT
                try:
                    return shape(json.loads(geom))
                except Exception:
                    return geom
        elif isinstance(geom, dict):
            return shape(geom)
        return geom

    def __len__(self) -> int:
        return self.len

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.df.iloc[idx]

        # Processa Embedding
        emb = row[self.embedding_column]
        if isinstance(emb, str):
            emb = np.array(json.loads(emb))
        elif isinstance(emb, list):
            emb = np.array(emb)

        # Garante que é float32
        emb = torch.tensor(emb, dtype=torch.float32)

        res = {"id": row[self.id_column], "embedding": emb}

        # Processa Geometria apenas se a coluna foi definida e existe no DF
        if self.geometry_column and self.geometry_column in self.df.columns:
            res["geometry"] = self._parse_geometry(row[self.geometry_column])

        return res
