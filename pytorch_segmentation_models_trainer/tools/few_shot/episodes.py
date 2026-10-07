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

import logging
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np
import pandas as pd
import rasterio
from omegaconf import DictConfig, OmegaConf
from rasterio.windows import Window
from tqdm import tqdm

from pytorch_segmentation_models_trainer.config_definitions.few_shot_config import (
    FewShotEpisodesConfig,
)
from pytorch_segmentation_models_trainer.dataset_loader.mbtiles_mask_dataset import (
    MBTilesMaskWindowedDataset,
)

logger = logging.getLogger(__name__)


def compute_class_fractions(
    records: Sequence[Dict[str, Any]],
    classes: Sequence[int],
    ignore_index: int = 255,
) -> np.ndarray:
    """Fraction of each class over the valid pixels of each mask window.

    Args:
        records: Window records with ``mask_path`` (resolved), ``row_off``,
            ``col_off``, ``width`` and ``height`` (pixel units).
        classes: Mask values to measure.
        ignore_index: Mask value excluded from the denominator.

    Returns:
        ``float64`` array of shape ``(len(records), len(classes))``. Windows
        without valid pixels get 0.
    """
    out = np.zeros((len(records), len(classes)), dtype=np.float64)
    for i, rec in enumerate(tqdm(records, desc="Class fractions", leave=False)):
        window = Window(rec["col_off"], rec["row_off"], rec["width"], rec["height"])
        with rasterio.open(rec["mask_path"]) as src:
            mask = src.read(1, window=window)
        n_valid = int((mask != ignore_index).sum())
        if n_valid == 0:
            continue
        for j, cls in enumerate(classes):
            out[i, j] = float((mask == cls).sum()) / n_valid
    return out


def sample_episodes(
    fractions: np.ndarray,
    novel_classes: Sequence[int],
    shots: Sequence[int],
    n_draws: int,
    min_fraction: float,
    seed: int,
) -> pd.DataFrame:
    """Draw nested support sets per (draw, novel class).

    For each draw and novel class, eligible windows (fraction >= threshold)
    are permuted once with ``np.random.default_rng((seed, draw, class))``;
    the support of size K is the first K windows of that permutation.

    Args:
        fractions: Output of :func:`compute_class_fractions`; column ``j``
            corresponds to ``novel_classes[j]``.
        novel_classes: Novel class values.
        shots: Support sizes.
        n_draws: Number of draws.
        min_fraction: Eligibility threshold.
        seed: Base seed.

    Returns:
        DataFrame with columns ``shots``, ``draw``, ``novel_class`` and
        ``window_idx`` (row index into ``fractions``), in selection order.

    Raises:
        ValueError: If a class has fewer eligible windows than ``max(shots)``.
    """
    rows = []
    max_k = max(shots)
    for j, cls in enumerate(novel_classes):
        eligible = np.flatnonzero(fractions[:, j] >= min_fraction)
        if len(eligible) < max_k:
            raise ValueError(
                f"novel class {cls}: only {len(eligible)} eligible windows "
                f"(fraction >= {min_fraction}), {max_k} needed."
            )
        for draw in range(n_draws):
            rng = np.random.default_rng((seed, draw, int(cls)))
            order = rng.permutation(eligible)
            for k in shots:
                rows.extend(
                    {
                        "shots": k,
                        "draw": draw,
                        "novel_class": int(cls),
                        "window_idx": int(w),
                    }
                    for w in order[:k]
                )
    return pd.DataFrame(rows, columns=["shots", "draw", "novel_class", "window_idx"])


def build_fewshot_episodes(cfg: DictConfig) -> str:
    """Entry point of ``mode: build-fewshot-episodes``.

    Args:
        cfg: Config with a ``fewshot_episodes`` node
            (:class:`FewShotEpisodesConfig`).

    Returns:
        Path of the written episodes CSV.
    """
    ep = OmegaConf.merge(
        OmegaConf.structured(FewShotEpisodesConfig), cfg.fewshot_episodes
    )
    index_path = Path(ep.window_index_cache)
    df = (
        pd.read_csv(index_path)
        if index_path.suffix.lower() == ".csv"
        else pd.read_parquet(index_path)
    )
    records = MBTilesMaskWindowedDataset._read_window_index_cache(
        index_path,
        mask_path_key=ep.window_index_mask_path_key,
        coordinate_mode=ep.window_index_coordinate_mode,
        mask_base_path=Path(ep.mask_base_path) if ep.mask_base_path else None,
    )
    novel: List[int] = list(ep.novel_classes)
    fractions = compute_class_fractions(records, novel, ignore_index=ep.ignore_index)
    episodes = sample_episodes(
        fractions,
        novel,
        shots=list(ep.shots),
        n_draws=ep.n_draws,
        min_fraction=ep.min_novel_fraction,
        seed=ep.seed,
    )
    col = {c: j for j, c in enumerate(novel)}
    out = df.iloc[episodes.window_idx.to_numpy()].reset_index(drop=True)
    out["shots"] = episodes.shots.to_numpy()
    out["draw"] = episodes.draw.to_numpy()
    out["novel_class"] = episodes.novel_class.to_numpy()
    out["novel_fraction"] = [
        fractions[w, col[c]] for w, c in zip(episodes.window_idx, episodes.novel_class)
    ]
    output = Path(ep.output_csv)
    output.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output, index=False)
    logger.info(
        "Few-shot episodes: %d rows (%d draws x shots %s x classes %s) -> %s",
        len(out),
        ep.n_draws,
        list(ep.shots),
        novel,
        output,
    )
    return str(output)
