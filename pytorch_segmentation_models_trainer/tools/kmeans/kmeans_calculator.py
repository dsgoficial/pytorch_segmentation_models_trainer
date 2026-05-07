# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
from typing import Optional, Union


class MiniBatchKMeans:
    """Implementação do Mini-Batch K-Means em PyTorch com suporte a GPU.

    Este algoritmo é uma variante do K-Means que utiliza pequenos lotes (mini-batches)
    de dados para atualizar os centroides, o que reduz o tempo de computação e
    o uso de memória, sendo ideal para grandes datasets.

    Args:
        n_clusters: Número de clusters.
        max_iter: Número máximo de iterações.
        batch_size: Tamanho do lote para cada atualização.
        tol: Tolerância para convergência baseada na mudança dos centroides.
        device: Dispositivo onde os cálculos serão realizados (cpu ou cuda).
        random_state: Semente para reprodutibilidade.
    """

    def __init__(
        self,
        n_clusters: int = 8,
        max_iter: int = 100,
        batch_size: int = 1024,
        tol: float = 1e-4,
        device: Optional[Union[str, torch.device]] = None,
        random_state: Optional[int] = None,
    ):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.batch_size = batch_size
        self.tol = tol
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.random_state = random_state
        self.centroids = None
        self.counts = None

        if self.random_state is not None:
            torch.manual_seed(self.random_state)

    def _init_centroids(self, x: torch.Tensor) -> torch.Tensor:
        """Inicialização K-Means++ amigável à memória.

        Args:
            x: Tensor de entrada (N, D), pode estar na CPU.

        Returns:
            Centroides iniciais (n_clusters, D) na self.device.
        """
        n_samples, n_features = x.shape
        centroids = torch.zeros((self.n_clusters, n_features), device=self.device)

        # Escolhe o primeiro centroide aleatoriamente
        idx = torch.randint(0, n_samples, (1,))
        centroids[0] = x[idx].to(self.device)

        # Para K-Means++, precisamos da distância mínima de cada ponto aos centroides já escolhidos
        # Mantemos min_dist na CPU se x estiver na CPU para economizar VRAM
        min_sq_dist = torch.full((n_samples,), float("inf"), device=x.device)

        for i in range(1, self.n_clusters):
            # Último centroide escolhido
            latest_centroid = centroids[i - 1 : i]  # (1, D)

            # Atualiza distâncias mínimas em blocos para evitar OOM
            for start in range(0, n_samples, self.batch_size):
                end = min(start + self.batch_size, n_samples)
                batch = x[start:end].to(self.device)

                # Distância do batch ao último centroide
                dist = torch.cdist(batch, latest_centroid, p=2).squeeze(1)
                dist_sq = dist.pow(2).to(x.device)

                min_sq_dist[start:end] = torch.min(min_sq_dist[start:end], dist_sq)

            # Seleciona próximo centroide baseado nas probabilidades
            if min_sq_dist.sum() == 0:
                idx = torch.randint(0, n_samples, (1,))
            else:
                idx = torch.multinomial(min_sq_dist, 1)

            centroids[i] = x[idx].to(self.device)

        return centroids

    def fit(self, x: torch.Tensor):
        """Treina o modelo usando Mini-Batch K-Means de forma eficiente em memória.

        Args:
            x: Tensor de entrada (N, D).
        """
        n_samples, n_features = x.shape

        # Inicialização (K-Means++ processa x em batches internamente)
        self.centroids = self._init_centroids(x)
        self.counts = torch.zeros(self.n_clusters, device=self.device)

        for i in range(self.max_iter):
            # Amostra um mini-batch (mantém índices na CPU e move apenas o batch para GPU)
            indices = torch.randint(0, n_samples, (min(self.batch_size, n_samples),))
            batch = x[indices].to(self.device)

            # Atribuição
            dist = torch.cdist(batch, self.centroids, p=2)
            _, labels = torch.min(dist, dim=1)

            # Atualização dos centroides (Incremental)
            old_centroids = self.centroids.clone()

            for j in range(self.n_clusters):
                mask = labels == j
                num_points = mask.sum()
                if num_points > 0:
                    self.counts[j] += num_points
                    eta = 1.0 / self.counts[j]
                    batch_mean = batch[mask].mean(dim=0)
                    self.centroids[j] = (1.0 - eta) * self.centroids[
                        j
                    ] + eta * batch_mean

            # Verificação de convergência
            center_shift = torch.norm(self.centroids - old_centroids)
            if center_shift < self.tol:
                break

        return self

    def fit_from_dataloader(
        self, dataloader: torch.utils.data.DataLoader, init_samples: int = 2048
    ):
        """Treina o modelo consumindo dados de um DataLoader.

        Args:
            dataloader: DataLoader que retorna batches de tensores ou dicts com chave "embedding".
            init_samples: Quantidade de amostras iniciais para rodar o K-Means++.
        """
        # 1. Inicialização: acumula amostras para o K-Means++
        init_data = []
        collected = 0
        for batch in dataloader:
            emb = batch["embedding"] if isinstance(batch, dict) else batch
            init_data.append(emb)
            collected += emb.shape[0]
            if collected >= init_samples:
                break

        x_init = torch.cat(init_data, dim=0)[:init_samples]
        self.centroids = self._init_centroids(x_init)
        self.counts = torch.zeros(self.n_clusters, device=self.device)

        # 2. Treinamento incremental
        for i in range(self.max_iter):
            converged = True
            for batch in dataloader:
                emb = batch["embedding"] if isinstance(batch, dict) else batch
                emb = emb.to(self.device)

                # Atribuição
                dist = torch.cdist(emb, self.centroids, p=2)
                _, labels = torch.min(dist, dim=1)

                # Atualização
                old_centroids = self.centroids.clone()
                for j in range(self.n_clusters):
                    mask = labels == j
                    num_points = mask.sum()
                    if num_points > 0:
                        self.counts[j] += num_points
                        eta = 1.0 / self.counts[j]
                        batch_mean = emb[mask].mean(dim=0)
                        self.centroids[j] = (1.0 - eta) * self.centroids[
                            j
                        ] + eta * batch_mean

                if torch.norm(self.centroids - old_centroids) > self.tol:
                    converged = False

            if converged:
                break

        return self

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Atribui cada ponto ao cluster mais próximo processando em batches.

        Args:
            x: Tensor de entrada (N, D).

        Returns:
            Tensor de labels (N,) no mesmo dispositivo que x.
        """
        if self.centroids is None:
            raise ValueError(
                "O modelo deve ser treinado (fit) antes de prever (predict)."
            )

        n_samples = x.shape[0]
        # Aloca labels no mesmo dispositivo de x (provavelmente CPU para datasets gigantes)
        labels = torch.zeros(n_samples, dtype=torch.long, device=x.device)

        for i in range(0, n_samples, self.batch_size):
            end = min(i + self.batch_size, n_samples)
            batch = x[i:end].to(self.device)

            dist = torch.cdist(batch, self.centroids, p=2)
            _, batch_labels = torch.min(dist, dim=1)

            # Move labels de volta para o dispositivo original de x
            labels[i:end] = batch_labels.to(x.device)

        return labels
