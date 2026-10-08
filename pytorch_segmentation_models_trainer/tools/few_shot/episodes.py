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
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
import rasterio
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from rasterio.windows import Window
from tqdm import tqdm

from pytorch_segmentation_models_trainer.config_definitions.few_shot_config import (
    FewShotEpisodesConfig,
)
from pytorch_segmentation_models_trainer.dataset_loader.mbtiles_mask_dataset import (
    MBTilesMaskWindowedDataset,
)
from pytorch_segmentation_models_trainer.few_shot.backbone import (
    FrozenLinearHeadSegmenter,
)
from pytorch_segmentation_models_trainer.tools.few_shot.support_selection import (
    support_diversity,
    tile_descriptors,
    weighted_kcenter,
)
from pytorch_segmentation_models_trainer.utils.checkpoint_loading import (
    load_pretrained_weights,
    resolve_checkpoint_from_runner,
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


def kcenter_episodes(
    fractions: np.ndarray,
    novel_classes: Sequence[int],
    shots: Sequence[int],
    n_draws: int,
    min_fraction: float,
    seed: int,
    embeddings: np.ndarray,
    uncertainty: Optional[np.ndarray] = None,
    gamma: float = 1.0,
) -> pd.DataFrame:
    """Nested support sets by (uncertainty-weighted) k-center over embeddings.

    Same eligibility and output as :func:`sample_episodes`; for each draw the
    first window is drawn at random among the eligible ones
    (``np.random.default_rng((seed, draw, class))``) and the others follow the
    greedy farthest-point order of :func:`weighted_kcenter` (distance to the
    selected set × ``uncertainty ** gamma``), so supports are nested.
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
        sub_u = None if uncertainty is None else uncertainty[eligible]
        for draw in range(n_draws):
            rng = np.random.default_rng((seed, draw, int(cls)))
            first = int(rng.integers(len(eligible)))
            local = weighted_kcenter(embeddings[eligible], max_k, first, sub_u, gamma)
            order = eligible[local]
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


def _selection_descriptors(sel, index_path: Path, n_windows: int):
    """Load the frozen base model and compute tile embeddings/uncertainties."""
    model = instantiate(sel["model"], _recursive_=False)
    ckpt = dict(sel.get("base_checkpoint") or {})
    path = ckpt.get("path")
    if not path:
        path = resolve_checkpoint_from_runner(
            ckpt["from_runner"], int(ckpt.get("seed", 42))
        )
    load_pretrained_weights(
        model,
        path,
        ckpt.get("source_format", "pytorch_lightning"),
        strict_loading=ckpt.get("strict_loading", True),
    )
    segmenter = FrozenLinearHeadSegmenter(model)
    if torch.cuda.is_available():
        segmenter = segmenter.cuda()
    dataset = instantiate(
        sel["dataset"], window_index_cache=str(index_path), _recursive_=False
    )
    if len(dataset) != n_windows:
        raise ValueError(
            f"selection.dataset has {len(dataset)} items but the window index has "
            f"{n_windows} windows; it must read the same windows in the same order."
        )
    emb, unc = tile_descriptors(
        segmenter,
        dataset,
        mothers=list(sel.get("mothers", [])),
        batch_size=int(sel.get("batch_size", 8)),
        pooling=sel.get("pooling", "superclass"),
    )
    emb = emb / np.maximum(np.linalg.norm(emb, axis=1, keepdims=True), 1e-12)
    return emb, unc


def build_fewshot_episodes(cfg: DictConfig) -> str:
    """Entry point of ``mode: build-fewshot-episodes``.

    Args:
        cfg: Config with a ``fewshot_episodes`` node
            (:class:`FewShotEpisodesConfig`).

    Returns:
        Path of the written episodes CSV.
    """
    # Resolve against the full config first: merging a detached node would
    # break interpolations such as ``${paths.masks_dir}``.
    ep = OmegaConf.merge(
        OmegaConf.structured(FewShotEpisodesConfig),
        OmegaConf.to_container(cfg.fewshot_episodes, resolve=True),
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
    sel = (
        OmegaConf.to_container(ep.selection, resolve=True)
        if ep.get("selection", None) is not None
        else None
    )
    method = sel.get("method", "random") if sel else "random"
    if method not in {"random", "kcenter"}:
        raise ValueError(
            f"selection.method must be 'random' or 'kcenter', got {method!r}."
        )
    emb, u, label = None, None, "random"
    if sel:
        emb, unc = _selection_descriptors(sel, index_path, len(records))
        u_name = sel.get("uncertainty")
        if u_name:
            if u_name not in unc:
                raise KeyError(
                    f"uncertainty {u_name!r} not available ({sorted(unc)}; vacuity and "
                    "dissonance need an evidential base model)."
                )
            u = unc[u_name]
    common = dict(
        shots=list(ep.shots),
        n_draws=ep.n_draws,
        min_fraction=ep.min_novel_fraction,
        seed=ep.seed,
    )
    if method == "kcenter":
        episodes = kcenter_episodes(
            fractions,
            novel,
            embeddings=emb,
            uncertainty=u,
            gamma=float(sel.get("gamma", 1.0)),
            **common,
        )
        label = "kcenter" + (f"+{sel['uncertainty']}" if u is not None else "")
    else:
        episodes = sample_episodes(fractions, novel, **common)
    col = {c: j for j, c in enumerate(novel)}
    out = df.iloc[episodes.window_idx.to_numpy()].reset_index(drop=True)
    out["shots"] = episodes.shots.to_numpy()
    out["draw"] = episodes.draw.to_numpy()
    out["novel_class"] = episodes.novel_class.to_numpy()
    out["novel_fraction"] = [
        fractions[w, col[c]] for w, c in zip(episodes.window_idx, episodes.novel_class)
    ]
    if sel:
        out["selection"] = label
        group = episodes.groupby(["shots", "draw", "novel_class"]).window_idx
        div = {key: support_diversity(emb[idx.to_numpy()]) for key, idx in group}
        out["diversity"] = [
            div[(k, d, c)]
            for k, d, c in zip(episodes.shots, episodes.draw, episodes.novel_class)
        ]
        if u is not None:
            out["uncertainty"] = u[episodes.window_idx.to_numpy()]
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
