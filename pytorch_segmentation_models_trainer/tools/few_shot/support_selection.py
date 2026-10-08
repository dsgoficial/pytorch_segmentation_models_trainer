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

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from pytorch_segmentation_models_trainer.few_shot.backbone import (
    FrozenLinearHeadSegmenter,
)
from pytorch_segmentation_models_trainer.few_shot.uncertainty import (
    dissonance,
    normalized_entropy,
    vacuity,
)
from pytorch_segmentation_models_trainer.tools.coreset.vendi_score import vendi_score


def weighted_kcenter(
    embeddings: np.ndarray,
    k: int,
    first: int,
    uncertainty: Optional[np.ndarray] = None,
    gamma: float = 1.0,
) -> List[int]:
    """Greedy (weighted) k-center order starting at ``first``.

    At each step the next item maximises
    ``min_{s in selected} ||e_i − e_s|| · u_i^gamma``; with ``uncertainty=None``
    or ``gamma=0`` this is the classic farthest-point traversal. The order is
    nested: the first K items of a larger selection are the K-selection.

    Args:
        embeddings: ``(N, D)`` item embeddings.
        k: Number of items (capped at N).
        first: Index of the first item.
        uncertainty: Optional ``(N,)`` non-negative weights.
        gamma: Exponent of the uncertainty weight.

    Returns:
        Selected indices in selection order.
    """
    n = embeddings.shape[0]
    weight = (
        np.ones(n) if uncertainty is None else np.asarray(uncertainty, float) ** gamma
    )
    order = [int(first)]
    dist = np.linalg.norm(embeddings - embeddings[first], axis=1)
    while len(order) < min(k, n):
        score = dist * weight
        score[order] = -np.inf
        nxt = int(np.argmax(score))
        order.append(nxt)
        dist = np.minimum(dist, np.linalg.norm(embeddings - embeddings[nxt], axis=1))
    return order


def support_diversity(embeddings: np.ndarray) -> float:
    """Vendi score (cosine kernel) of a support set: 1 = all alike, up to K."""
    return float(vendi_score(np.asarray(embeddings, dtype=np.float64)))


@torch.no_grad()
def tile_descriptors(
    segmenter: FrozenLinearHeadSegmenter,
    dataset: Dataset,
    mothers: Sequence[int],
    batch_size: int = 8,
    pooling: str = "superclass",
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """Per-tile embedding and base-model uncertainty, without labels.

    Args:
        segmenter: Frozen base model (features = decoder output).
        dataset: Items with an ``image`` tensor, in window-index order.
        mothers: Base classes to be split (e.g. ``[3]``, low vegetation).
        batch_size: Batch size of the forward passes.
        pooling: ``superclass`` — mean decoder feature over the pixels the
            base model predicts as a mother (tile mean if there are none);
            ``tile`` — mean over the whole tile.

    Returns:
        ``(N, F)`` embeddings and ``{name: (N,)}`` mean uncertainties over the
        superclass pixels (0 when there are none): ``base_entropy``
        (normalised entropy of the base probabilities) and, for an evidential
        base model, ``base_vacuity`` and ``base_dissonance``.

    Raises:
        ValueError: Unknown ``pooling``.
    """
    if pooling not in {"superclass", "tile"}:
        raise ValueError(f"pooling must be 'superclass' or 'tile', got {pooling!r}.")
    weight, bias = segmenter.weight, segmenter.bias
    embs: List[np.ndarray] = []
    unc: Dict[str, List[np.ndarray]] = {"base_entropy": []}
    if segmenter.evidential:
        unc.update(base_vacuity=[], base_dissonance=[])
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    for batch in tqdm(loader, desc="Tile descriptors", leave=False):
        images = batch["image"] if isinstance(batch, dict) else batch[0]
        images = images.to(weight.device).float()
        feats = segmenter.features(images)
        logits = segmenter.linear(feats, weight.to(feats), bias.to(feats))
        if segmenter.evidential:
            alpha = torch.nn.functional.softplus(logits) + 1.0
            probs = alpha / alpha.sum(1, keepdim=True)
        else:
            probs = torch.softmax(logits, dim=1)
        pred = probs.argmax(1)
        sc = torch.isin(pred, torch.tensor(list(mothers), device=pred.device)).float()
        n_sc = sc.flatten(1).sum(1)
        tile_mean = feats.mean(dim=(2, 3))
        sc_mean = (feats * sc.unsqueeze(1)).sum(dim=(2, 3)) / n_sc.clamp(
            min=1
        ).unsqueeze(1)
        use_sc = (n_sc > 0).unsqueeze(1) & (pooling == "superclass")
        embs.append(torch.where(use_sc, sc_mean, tile_mean).double().cpu().numpy())

        def sc_avg(m):
            return (
                ((m * sc).flatten(1).sum(1) / n_sc.clamp(min=1)).double().cpu().numpy()
            )

        unc["base_entropy"].append(sc_avg(normalized_entropy(probs)))
        if segmenter.evidential:
            unc["base_vacuity"].append(sc_avg(vacuity(alpha)))
            unc["base_dissonance"].append(
                sc_avg(dissonance((alpha - 1.0) / alpha.sum(1, keepdim=True)))
            )
    return np.concatenate(embs), {k: np.concatenate(v) for k, v in unc.items()}
