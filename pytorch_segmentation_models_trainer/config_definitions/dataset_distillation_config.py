# -*- coding: utf-8 -*-
from dataclasses import dataclass, field
from typing import Optional, List
from hydra.core.config_store import ConfigStore


@dataclass
class DatasetDistillationConfig:
    """
    Configuration for dataset distillation via Coreset of Medoids (Optimal Quantization).
    """

    num_clusters: int = 100
    batch_size: int = 32
    num_workers: int = 4
    device: str = "cuda"
    random_seed: int = 42

    # Paths
    unlabeled_dataloader_config: Optional[str] = None
    checkpoint_path: Optional[str] = None
    output_indices_path: str = "medoid_indices.pt"


def register_dataset_distillation_configs():
    cs = ConfigStore.instance()
    cs.store(
        group="dataset_distillation",
        name="base_dataset_distillation",
        node=DatasetDistillationConfig,
    )
