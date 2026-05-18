# -*- coding: utf-8 -*-
from dataclasses import dataclass
from typing import List, Optional
from hydra.core.config_store import ConfigStore


@dataclass
class DatasetDistillationConfig:
    """
    Configuration for dataset distillation via Coreset of Medoids (Optimal Quantization).
    """

    # Core budget. ``k`` is the preferred DDOQ name; ``num_clusters`` remains
    # for backward compatibility with existing configs.
    k: Optional[int] = None
    num_clusters: int = 100
    batch_size: int = 32
    num_workers: int = 4
    device: str = "cuda"
    random_seed: int = 42

    # DDOQ Specific
    mode: str = "vae_decode"
    latent: str = "mu"
    latent_reduction: str = "flatten"
    weight_mode: str = "sqrt"
    distilled_image_format: str = "auto"
    output_size: Optional[List[int]] = None
    kmeans_max_iter: int = 100
    kmeans_batch_size: int = 1024
    use_sqrt_heuristic: bool = True
    adaptive_k: bool = False
    k_min: int = 10
    k_max: int = 1000
    k_step: int = 50

    # Paths
    vae_config_path: Optional[str] = None
    vae_checkpoint_path: Optional[str] = None
    dataset_config_path: Optional[str] = None
    dataset_key: str = "train_dataset"
    output_dir: str = "ddoq_output"
    unlabeled_dataloader_config: Optional[str] = None
    checkpoint_path: Optional[str] = None
    output_indices_path: str = "medoid_indices.pt"
    output_weights_path: str = "ddoq_weights.pt"
    output_ddoq_results_path: str = "ddoq_results.pt"


def register_dataset_distillation_configs():
    cs = ConfigStore.instance()
    cs.store(
        group="dataset_distillation",
        name="base_dataset_distillation",
        node=DatasetDistillationConfig,
    )
    cs.store(
        group="dataset_distillation",
        name="vae_ddoq",
        node=DatasetDistillationConfig,
    )
