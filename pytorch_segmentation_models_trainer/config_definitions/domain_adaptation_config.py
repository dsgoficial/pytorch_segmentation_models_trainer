# -*- coding: utf-8 -*-
"""
/***************************************************************************
 pytorch_segmentation_models_trainer
                              -------------------
        begin                : 2026-04-14
        git sha              : $Format:%H$
        copyright            : (C) 2026 by Philipe Borba - Cartographic Engineer
                                                            @ Brazilian Army
        email                : philipeborba at gmail dot com
 ***************************************************************************/
/***************************************************************************
 *                                                                         *
 *   This program is free software; you can redistribute it and/or modify  *
 *   it under the terms of the GNU General Public License as published by  *
 *   the Free Software Foundation; either version 2 of the License, or     *
 *   (at your option) any later version.                                   *
 *                                                                         *
 ****
"""

from dataclasses import dataclass, field
from typing import Any, List, Optional

from omegaconf import MISSING


@dataclass
class PretrainedCheckpointConfig:
    """Configuration for loading pretrained weights as a warm-start for DA training.

    This is distinct from ``resume_from_checkpoint``: it loads only the model
    weights (no optimizer state, epoch counter, or scheduler state) so that the
    optimizer starts fresh for the adaptation phase.

    Args:
        path: Absolute path to the checkpoint file.
        source_format: Format of the checkpoint file.
            - ``"pytorch_lightning"``: ``.ckpt`` file saved by PL. Weights are
              stored under the ``"state_dict"`` key and prefixed with ``"model."``.
            - ``"pytorch"``: plain ``.pt`` / ``.pth`` file whose top-level keys
              are the model parameter names (no prefix).
        strict_loading: Passed to ``load_state_dict(strict=...)``. Set to
            ``False`` when loading a checkpoint whose architecture differs
            slightly from the current model (e.g., different number of classes).
    """

    path: str = MISSING
    source_format: str = "pytorch_lightning"
    strict_loading: bool = True


@dataclass
class DomainAdaptationMethodConfig:
    """Base configuration for a domain adaptation method.

    Concrete method configs should inherit from this class and add their own
    fields. The ``_target_`` field is resolved by Hydra to instantiate the
    correct ``BaseDomainAdaptationMethod`` subclass.

    Args:
        _target_: Fully-qualified class name of the method to instantiate.
        lambda_da: Global weighting factor applied to the DA loss before it is
            added to the segmentation loss: ``total = seg_loss + lambda_da * da_loss``.
    """

    _target_: str = MISSING
    lambda_da: float = 1.0


@dataclass
class DataLoaderDAConfig:
    """DataLoader settings for domain adaptation datasets.

    Args:
        batch_size: Number of samples per batch.
        num_workers: Number of subprocesses for data loading.
        shuffle: Whether to shuffle the data at each epoch.
        pin_memory: Whether to copy tensors into pinned memory before returning.
        drop_last: Whether to drop the last incomplete batch.
        prefetch_factor: Number of batches to prefetch per worker.
        persistent_workers: Whether to keep worker processes alive between epochs.
    """

    batch_size: int = 8
    num_workers: int = 4
    shuffle: bool = True
    pin_memory: bool = True
    drop_last: bool = True
    prefetch_factor: int = 2
    persistent_workers: bool = False


@dataclass
class DomainAdaptationDatasetConfig:
    """Dataset configuration for a single domain (source or target).

    Args:
        _target_: Fully-qualified class name of the dataset to instantiate.
        input_csv_path: Path to the CSV file listing image/mask pairs.
        data_loader: DataLoader settings.
    """

    _target_: str = MISSING
    input_csv_path: str = MISSING
    data_loader: DataLoaderDAConfig = field(default_factory=DataLoaderDAConfig)


@dataclass
class DomainAdaptationConfig:
    """Top-level configuration for the domain adaptation module.

    Args:
        method: Configuration for the DA method (subclass of
            ``BaseDomainAdaptationMethod``).
        source_dataset: Labeled source-domain training dataset.
        target_dataset: Unlabeled (or weakly labeled) target-domain training
            dataset. Used for DA loss computation only — masks are ignored
            unless ``method.requires_target_labels = True``.
        source_val_dataset: Optional labeled source-domain validation dataset.
            Used to detect catastrophic forgetting during training.
        target_val_dataset: Optional labeled target-domain validation dataset.
            Used to measure adaptation progress.
        feature_layers: List of layer name strings (e.g. ``["encoder.layer3"]``)
            whose outputs will be captured by ``FeatureExtractorHook`` and
            passed to ``compute_da_loss``. Only active when
            ``method.requires_features = True``.
        pretrained_checkpoint: Optional warm-start checkpoint configuration.
            When provided, model weights are loaded before training begins.
    """

    method: DomainAdaptationMethodConfig = MISSING
    source_dataset: DomainAdaptationDatasetConfig = MISSING
    target_dataset: DomainAdaptationDatasetConfig = MISSING
    source_val_dataset: Optional[Any] = None
    target_val_dataset: Optional[Any] = None
    feature_layers: List[str] = field(default_factory=list)
    pretrained_checkpoint: Optional[PretrainedCheckpointConfig] = None
