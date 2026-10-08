# -*- coding: utf-8 -*-
"""
/***************************************************************************
 pytorch_segmentation_models_trainer
                              -------------------
        begin                : 2026-10-07
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
from typing import Any, Dict, List, Optional

from hydra.core.config_store import ConfigStore
from omegaconf import MISSING


@dataclass
class FewShotEpisodesConfig:
    """Configuration of ``mode: build-fewshot-episodes``.

    Draws fixed support sets for generalized few-shot segmentation from a
    window index (the same CSV/Parquet consumed by
    ``MBTilesMaskWindowedDataset.window_index_cache``). A window is eligible
    for a novel class when that class covers at least ``min_novel_fraction``
    of the window's valid (non-``ignore_index``) pixels. For each draw and
    novel class the eligible windows are shuffled once and the first ``K``
    are taken for every ``K`` in ``shots``, so support sets are **nested**
    (K=1 ⊂ K=3 ⊂ ...), which pairs comparisons across K.

    Args:
        window_index_cache: Window index of the pool (usually the training
            split). Pixel (``row_off``/``col_off``/``width``/``height``) or
            bounds columns, as in ``MBTilesMaskWindowedDataset``.
        output_csv: Episodes file. One row per selected window, with the
            original window-index columns plus ``shots``, ``draw``,
            ``novel_class`` and ``novel_fraction``.
        novel_classes: Mask values of the novel classes.
        shots: Support sizes K (windows per novel class).
        n_draws: Number of independent support draws per K.
        min_novel_fraction: Eligibility threshold over valid pixels.
        seed: Base seed; draw ``d`` of class ``c`` uses ``(seed, d, c)``.
        mask_base_path: Prefix for relative mask paths in the index.
        window_index_mask_path_key: Column with the mask path.
        window_index_coordinate_mode: ``auto``, ``pixel`` or ``bounds``.
        ignore_index: Mask value excluded from the fractions.

    Example YAML::

        mode: build-fewshot-episodes
        fewshot_episodes:
          window_index_cache: experiment_configs/tiles_data/train_pampa.csv
          mask_base_path: /data/masks
          novel_classes: [5]
          shots: [1, 3, 5, 10]
          n_draws: 5
          min_novel_fraction: 0.05
          seed: 2026
          output_csv: outputs/episodes/pampa.csv
    """

    window_index_cache: str = MISSING
    output_csv: str = MISSING
    novel_classes: List[int] = MISSING
    shots: List[int] = field(default_factory=lambda: [1, 3, 5, 10])
    n_draws: int = 5
    min_novel_fraction: float = 0.05
    seed: int = 2026
    mask_base_path: Optional[str] = None
    window_index_mask_path_key: str = "mask_path"
    window_index_coordinate_mode: str = "auto"
    ignore_index: int = 255


@dataclass
class ClassCountsConfig:
    """Configuration of ``mode: count-class-pixels``.

    Counts the pixels of each class over a window index (e.g. the training
    split), after an optional remapping, and writes
    ``{"num_classes": n, "counts": [...]}`` to ``output_json`` — e.g. the
    base-class counts of ClassTrans' LDAM margins (``base_class_counts``).

    Example YAML::

        mode: count-class-pixels
        class_counts:
          window_index_cache: experiment_configs/tiles_data/train_pampa.csv
          mask_base_path: /data/masks
          num_classes: 5
          mask_class_mapping: {5: 3}
          output_json: outputs/class_counts/pampa_base.json
    """

    window_index_cache: str = MISSING
    output_json: str = MISSING
    num_classes: int = MISSING
    mask_class_mapping: Optional[Dict[int, int]] = None
    mask_base_path: Optional[str] = None
    window_index_mask_path_key: str = "mask_path"
    window_index_coordinate_mode: str = "auto"
    ignore_index: int = 255


@dataclass
class GFSSCheckpointConfig:
    """Frozen base model checkpoint of a GFSS task.

    Exactly one of ``path`` and ``from_runner`` must be set.

    Args:
        path: Explicit checkpoint file.
        from_runner: ``output_base_dir`` of the ExperimentsRunner that trained
            the base model; the checkpoint of the run whose seed equals the
            GFSS run's ``seed`` is used (read from ``runner_state.json``).
        source_format: ``pytorch_lightning`` or ``pytorch``.
        strict_loading: Strict ``load_state_dict`` (the base model config must
            match the checkpoint exactly; keep ``true``).
    """

    path: Optional[str] = None
    from_runner: Optional[str] = None
    source_format: str = "pytorch_lightning"
    strict_loading: bool = True


@dataclass
class GFSSConfig:
    """The ``gfss`` node consumed by ``GFSSModel``.

    The base model is built from the top-level ``model`` node (same
    architecture and number of classes as the checkpoint), loaded from
    ``base_checkpoint``, frozen, and extended with the novel classes of
    ``hierarchy`` by ``method``. ``train_dataset`` is the support set and
    ``test_dataset`` the query set.

    Args:
        hierarchy: ``{mother: [children]}`` (see ``ClassHierarchy``).
        base_checkpoint: Where the base weights come from.
        method: Hydra node of a ``BaseGFSSMethod`` subclass.
        not_novel_index: Support label for "labelled, not any novel class"
            (support regime S-novel, e.g. via ``mask_class_mapping``).
        ignore_index: Label ignored by losses and metrics.
        class_names: Names of the ``num_classes`` final classes used in the
            metric keys (default: ``class_definitions.names`` or indices).
        backbone: Fine-tuning baselines only: ``{trainable: none | decoder |
            all | lora, lora: {r, alpha, dropout, target_modules}}``. A
            frozen copy of the base model stays as reference (KD snapshot,
            base predictions). Default ``none`` (frozen backbone).
        uncertainty_eval: Options of ``GFSSUncertaintyMetrics`` for methods
            with uncertainty maps (``abstain_thresholds``, ``ece_bins``,
            ``n_bins``).

    Example YAML::

        pl_model:
          _target_: pytorch_segmentation_models_trainer.model_loader.gfss_model.GFSSModel
        gfss:
          hierarchy: {3: [3, 5]}
          base_checkpoint:
            from_runner: outputs/baselines/r2_pampa
          method:
            _target_: pytorch_segmentation_models_trainer.few_shot.methods.base_only.BaseOnly
    """

    hierarchy: Dict[int, List[int]] = MISSING
    base_checkpoint: GFSSCheckpointConfig = field(default_factory=GFSSCheckpointConfig)
    method: Any = MISSING
    not_novel_index: int = 254
    ignore_index: int = 255
    class_names: Optional[List[str]] = None
    uncertainty_eval: Optional[Dict[str, Any]] = None
    backbone: Optional[Dict[str, Any]] = None


def _register_configs() -> None:
    cs = ConfigStore.instance()
    cs.store(name="fewshot_episodes_config", node=FewShotEpisodesConfig)
    cs.store(name="gfss_config", node=GFSSConfig)
    cs.store(name="class_counts_config", node=ClassCountsConfig)


_register_configs()
