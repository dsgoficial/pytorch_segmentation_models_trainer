# -*- coding: utf-8 -*-
"""
/***************************************************************************
 pytorch_segmentation_models_trainer
                              -------------------
        begin                : 2026-10-09
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

Separability probe of the frozen base features (``mode: separability-probe``).

Measures how well a pair of classes that the base model never had to
separate (e.g. the two children of a merged superclass) can be told apart
in its frozen decoder features, as a function of the number of support
tiles K. For each base seed and support episode: decoder features of the
ground-truth pixels of the two classes are collected on the support tiles
(fully labelled — this is a diagnostic of the representation, not a GFSS
method), a classifier is fitted, and it is scored on the same two classes
over the test windows:

* ``logreg``: standardized features + L2 logistic regression
  (``class_weight="balanced"``);
* ``proto``: cosine to the class means of L2-normalized features
  (training-free; score = cos(z, μ+) − cos(z, μ−), threshold 0).

Metrics: AUROC, balanced accuracy and average precision (positive class).
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader, Dataset

from pytorch_segmentation_models_trainer.config_definitions.few_shot_config import (
    SeparabilityProbeConfig,
)
from pytorch_segmentation_models_trainer.few_shot.backbone import (
    FrozenLinearHeadSegmenter,
)
from pytorch_segmentation_models_trainer.utils.checkpoint_loading import (
    load_pretrained_weights,
    resolve_checkpoint_from_runner,
)

logger = logging.getLogger(__name__)

_METRICS = (
    "auroc_logreg",
    "balacc_logreg",
    "ap_logreg",
    "auroc_proto",
    "balacc_proto",
)
_EPISODE_COLUMNS = (
    "shots",
    "draw",
    "novel_class",
    "novel_fraction",
    "selection",
    "diversity",
    "uncertainty",
)


def probe_scores(
    train_X: np.ndarray,
    train_y: np.ndarray,
    test_X: np.ndarray,
    test_y: np.ndarray,
    C: float = 1.0,
) -> Dict[str, float]:
    """Fit the two probes on the support pixels and score them on the test.

    Args:
        train_X: Support features ``(N, F)``.
        train_y: Support labels (1 = positive class, 0 = negative).
        test_X: Test features ``(M, F)``.
        test_y: Test labels.
        C: Inverse L2 regularization of the logistic regression.

    Returns:
        ``{metric: value}`` (all ``nan`` when a class is missing in the
        support or in the test).
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import (
        average_precision_score,
        balanced_accuracy_score,
        roc_auc_score,
    )
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    if len(np.unique(train_y)) < 2 or len(np.unique(test_y)) < 2:
        return {k: float("nan") for k in _METRICS}
    clf = make_pipeline(
        StandardScaler(),
        LogisticRegression(C=C, class_weight="balanced", max_iter=1000),
    )
    clf.fit(train_X, train_y)
    p = clf.predict_proba(test_X)[:, 1]

    def _unit(x):
        return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)

    z_tr, z_te = _unit(train_X), _unit(test_X)
    mu_pos = _unit(z_tr[train_y == 1].mean(0, keepdims=True))[0]
    mu_neg = _unit(z_tr[train_y == 0].mean(0, keepdims=True))[0]
    s = z_te @ mu_pos - z_te @ mu_neg
    return {
        "auroc_logreg": float(roc_auc_score(test_y, p)),
        "balacc_logreg": float(balanced_accuracy_score(test_y, p > 0.5)),
        "ap_logreg": float(average_precision_score(test_y, p)),
        "auroc_proto": float(roc_auc_score(test_y, s)),
        "balacc_proto": float(balanced_accuracy_score(test_y, s > 0)),
    }


@torch.no_grad()
def collect_pair_features(
    segmenter: FrozenLinearHeadSegmenter,
    dataset: Dataset,
    negative_class: int,
    positive_class: int,
    batch_size: int = 8,
    keep_fraction: float = 1.0,
    max_per_class: Optional[int] = None,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Decoder features of the ground-truth pixels of two classes.

    Masks are downsampled (nearest) to the feature resolution, as in
    ``GFSSModel``. Each pixel is kept with probability ``keep_fraction`` and
    each class is finally capped at ``max_per_class`` random pixels.

    Returns:
        ``(X, y)``: features ``(N, F)`` (float32) and labels (1 = positive).
    """
    rng = rng if rng is not None else np.random.default_rng(0)
    device = segmenter.weight.device
    feats: List[np.ndarray] = []
    labels: List[np.ndarray] = []
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    for batch in loader:
        images = batch["image"].to(device).float()
        masks = batch["mask"].to(device).long()
        f = segmenter.features(images)
        m = F.interpolate(masks.unsqueeze(1).float(), size=f.shape[-2:], mode="nearest")
        m = m.squeeze(1).long()
        sel = (m == negative_class) | (m == positive_class)
        if keep_fraction < 1.0:
            keep = torch.from_numpy(rng.random(tuple(sel.shape)) < keep_fraction)
            sel &= keep.to(sel.device)
        if sel.any():
            feats.append(f.permute(0, 2, 3, 1)[sel].float().cpu().numpy())
            labels.append((m[sel] == positive_class).long().cpu().numpy())
    channels = segmenter.weight.shape[1]
    if not feats:
        return np.zeros((0, channels), np.float32), np.zeros((0,), np.int64)
    X, y = np.concatenate(feats), np.concatenate(labels)
    if max_per_class is not None:
        idx = []
        for c in (0, 1):
            ic = np.flatnonzero(y == c)
            if len(ic) > max_per_class:
                ic = rng.choice(ic, max_per_class, replace=False)
            idx.append(ic)
        idx = np.sort(np.concatenate(idx))
        X, y = X[idx], y[idx]
    return X.astype(np.float32), y


def _load_segmenter(model_cfg, ckpt, seed: Optional[int]) -> FrozenLinearHeadSegmenter:
    if bool(ckpt.get("path")) == bool(ckpt.get("from_runner")):
        raise ValueError(
            "separability_probe.checkpoint needs exactly one of 'path' and "
            "'from_runner'."
        )
    path = ckpt.get("path") or resolve_checkpoint_from_runner(
        ckpt["from_runner"], int(seed)
    )
    model = instantiate(model_cfg, _recursive_=False)
    load_pretrained_weights(
        model,
        path,
        ckpt.get("source_format", "pytorch_lightning"),
        strict_loading=ckpt.get("strict_loading", True),
    )
    segmenter = FrozenLinearHeadSegmenter(model)
    return segmenter.cuda() if torch.cuda.is_available() else segmenter


def run_separability_probe(cfg: DictConfig) -> str:
    """Entry point of ``mode: separability-probe``; writes and returns the CSV.

    One row per base seed × episode (``shots``, ``draw``) with the support /
    test pixel counts and the metrics of :func:`probe_scores`.
    """
    p = OmegaConf.merge(
        OmegaConf.structured(SeparabilityProbeConfig),
        OmegaConf.to_container(cfg.separability_probe, resolve=True),
    )
    ckpt = OmegaConf.to_container(p.checkpoint)
    out = Path(p.output_csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(p.episodes_csv)
    if p.shots is not None:
        df = df[df["shots"].isin(list(p.shots))]
    if p.draws is not None:
        df = df[df["draw"].isin(list(p.draws))]
    if df.empty:
        raise ValueError(
            f"separability_probe: no episode selected from {p.episodes_csv} "
            f"(shots={p.shots}, draws={p.draws})."
        )
    window_cols = [c for c in df.columns if c not in _EPISODE_COLUMNS]
    seeds = list(p.seeds) if ckpt.get("from_runner") else [None]
    pair = dict(negative_class=p.negative_class, positive_class=p.positive_class)
    rows = []
    for seed in seeds:
        segmenter = _load_segmenter(cfg.model, ckpt, seed)
        rng = np.random.default_rng(p.seed)
        test_ds = instantiate(p.test_dataset, _recursive_=False)
        test_X, test_y = collect_pair_features(
            segmenter,
            test_ds,
            batch_size=p.batch_size,
            keep_fraction=p.test_keep_fraction,
            max_per_class=p.max_test_pixels_per_class,
            rng=rng,
            **pair,
        )
        for (k, d), sub in df.groupby(["shots", "draw"], sort=True):
            index = out.parent / f"probe_support_k{int(k):02d}_d{int(d):02d}.csv"
            sub[window_cols].drop_duplicates().to_csv(index, index=False)
            sup_ds = instantiate(
                p.support_dataset, window_index_cache=str(index), _recursive_=False
            )
            X, y = collect_pair_features(
                segmenter,
                sup_ds,
                batch_size=p.batch_size,
                max_per_class=p.max_support_pixels_per_class,
                rng=rng,
                **pair,
            )
            index.unlink()
            row = {
                "base_seed": seed,
                "shots": int(k),
                "draw": int(d),
                "n_support_neg": int((y == 0).sum()),
                "n_support_pos": int((y == 1).sum()),
                "n_test_neg": int((test_y == 0).sum()),
                "n_test_pos": int((test_y == 1).sum()),
            }
            row.update(probe_scores(X, y, test_X, test_y, C=p.C))
            rows.append(row)
            logger.info("Separability probe %s", row)
    pd.DataFrame(rows).to_csv(out, index=False)
    return str(out)
