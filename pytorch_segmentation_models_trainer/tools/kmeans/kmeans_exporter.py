# -*- coding: utf-8 -*-
import logging
import torch
import numpy as np
import pandas as pd
import geopandas as gpd
from typing import List, Optional, Union, Any
from pytorch_segmentation_models_trainer.tools.kmeans.kmeans_calculator import (
    MiniBatchKMeans,
)

logger = logging.getLogger(__name__)


class KMeansClusteringTool:
    """Orquestrador para clusterização K-Means e exportação de resultados.

    Gerencia a entrada de dados (IDs, embeddings, geometrias), executa o
    K-Means em GPU e consolida os resultados em um GeoDataFrame para exportação.

    Args:
        n_clusters: Número de clusters desejado.
        batch_size: Tamanho do lote para treinamento e inferência.
        max_iter: Número máximo de iterações do K-Means.
        device: Dispositivo para execução (cpu ou cuda).
        random_state: Semente aleatória para reprodutibilidade.
    """

    def __init__(
        self,
        n_clusters: int = 8,
        batch_size: int = 1024,
        max_iter: int = 100,
        device: Optional[str] = None,
        random_state: Optional[int] = None,
    ):
        self.n_clusters = n_clusters
        self.batch_size = batch_size
        self.max_iter = max_iter
        self.device = device
        self.random_state = random_state
        self.model = MiniBatchKMeans(
            n_clusters=n_clusters,
            batch_size=batch_size,
            max_iter=max_iter,
            device=device,
            random_state=random_state,
        )

    def run(
        self,
        ids: List[Any],
        embeddings: Union[torch.Tensor, np.ndarray],
        geometries: Optional[List[Any]] = None,
        crs: str = "EPSG:4326",
    ) -> Union[pd.DataFrame, gpd.GeoDataFrame]:
        """Executa a clusterização e retorna um DataFrame ou GeoDataFrame consolidado.

        Args:
            ids: Lista de identificadores únicos para cada amostra.
            embeddings: Tensor ou array (N, D) com as features.
            geometries: Opcional. Lista de objetos de geometria.
            crs: Sistema de referência de coordenadas. Default: EPSG:4326.

        Returns:
            DataFrame contendo colunas ['id', 'cluster_id'] e opcionalmente 'geometry'.
        """
        logger.info(f"Iniciando clusterização K-Means para {len(ids)} amostras...")

        # Converte embeddings para torch.Tensor se necessário
        if isinstance(embeddings, np.ndarray):
            embeddings = torch.from_numpy(embeddings).float()
        elif isinstance(embeddings, torch.Tensor):
            embeddings = embeddings.float()
        else:
            raise ValueError("embeddings deve ser torch.Tensor ou numpy.ndarray")

        if len(ids) != embeddings.shape[0]:
            raise ValueError("O número de IDs e embeddings deve ser o mesmo.")

        if geometries is not None and len(ids) != len(geometries):
            raise ValueError("O número de IDs e geometrias deve ser o mesmo.")

        # Executa fit e predict
        self.model.fit(embeddings)
        labels = self.model.predict(embeddings).cpu().numpy()

        # Constrói o DataFrame
        df = pd.DataFrame({"id": ids, "cluster_id": labels})

        if geometries is not None:
            gdf = gpd.GeoDataFrame(df, geometry=geometries, crs=crs)
            logger.info("Clusterização concluída com sucesso (GeoDataFrame).")
            return gdf

        logger.info("Clusterização concluída com sucesso (DataFrame).")
        return df

    def run_from_dataloader(
        self,
        dataloader: torch.utils.data.DataLoader,
        crs: str = "EPSG:4326",
        init_samples: int = 2048,
    ) -> Union[pd.DataFrame, gpd.GeoDataFrame]:
        """Executa a clusterização consumindo dados de um DataLoader.

        O DataLoader deve retornar dicionários contendo ao menos as chaves
        'id' e 'embedding'. A chave 'geometry' é opcional.

        Args:
            dataloader: DataLoader do PyTorch.
            crs: Sistema de referência de coordenadas.
            init_samples: Quantidade de amostras para inicialização K-Means++.

        Returns:
            DataFrame ou GeoDataFrame com os resultados.
        """
        logger.info("Iniciando clusterização via DataLoader...")

        # 1. Treinamento
        self.model.fit_from_dataloader(dataloader, init_samples=init_samples)

        # 2. Inferência e coleta de metadados
        all_ids = []
        all_labels = []
        all_geometries = []
        has_geometry = False

        with torch.no_grad():
            for batch in dataloader:
                if not isinstance(batch, dict):
                    raise ValueError(
                        "run_from_dataloader requer que o DataLoader retorne dicts com 'id' e 'embedding'."
                    )

                emb = batch["embedding"].to(self.model.device).float()

                # Predição no batch
                dist = torch.cdist(emb, self.model.centroids, p=2)
                _, labels = torch.min(dist, dim=1)

                all_ids.extend(batch["id"])
                all_labels.append(labels.cpu())

                if "geometry" in batch:
                    all_geometries.extend(batch["geometry"])
                    has_geometry = True

        # Consolidação
        df = pd.DataFrame({"id": all_ids, "cluster_id": torch.cat(all_labels).numpy()})

        if has_geometry:
            gdf = gpd.GeoDataFrame(df, geometry=all_geometries, crs=crs)
            logger.info("Clusterização via DataLoader concluída (GeoDataFrame).")
            return gdf

        logger.info("Clusterização via DataLoader concluída (DataFrame).")
        return df

    def export_to_parquet(self, gdf: gpd.GeoDataFrame, output_path: str):
        """Exporta o GeoDataFrame para formato GeoParquet.

        Args:
            gdf: GeoDataFrame gerado por run().
            output_path: Caminho de saída (.parquet).
        """
        logger.info(f"Exportando resultados para {output_path}...")
        gdf.to_parquet(output_path)

    def export_to_postgis(
        self,
        gdf: gpd.GeoDataFrame,
        table_name: str,
        engine_url: str,
        if_exists: str = "replace",
    ):
        """Exporta o GeoDataFrame diretamente para um banco PostGIS.

        Args:
            gdf: GeoDataFrame gerado por run().
            table_name: Nome da tabela no banco.
            engine_url: URL de conexão do SQLAlchemy (ex: postgresql://user:pass@host/db).
            if_exists: Comportamento se a tabela já existir ('fail', 'replace', 'append').
        """
        from sqlalchemy import create_engine

        logger.info(f"Exportando resultados para tabela PostGIS: {table_name}...")
        engine = create_engine(engine_url)
        gdf.to_postgis(table_name, engine, if_exists=if_exists)
