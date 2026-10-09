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

import warnings
from typing import Callable, Dict, List, Sequence

import numpy as np
import torch
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from torch import Tensor

from pytorch_segmentation_models_trainer.few_shot.backbone import (
    FrozenLinearHeadSegmenter,
)
from pytorch_segmentation_models_trainer.few_shot.base_method import BaseGFSSMethod

_FLOOR = -1e4


def _softmax_decision(clf: LogisticRegression, X: np.ndarray) -> np.ndarray:
    """Official ``proba_wo_d4p``: softmax of the decision function."""
    decision = clf.decision_function(X)
    if decision.ndim == 1:
        decision = np.c_[-decision, decision]
    decision = decision - decision.max(axis=1, keepdims=True)
    e = np.exp(decision)
    return e / e.sum(axis=1, keepdims=True)


def _balance(X, y, sampling: str, rng: np.random.RandomState):
    """Official ``handle_imbalance`` (``us``, ``os``, ``bg``)."""
    if sampling == "bg":
        _, counts = np.unique(y, return_counts=True)
        max_num_novel = np.sort(counts)[-2]
        idxs = np.where(y == 0)[0]
        if len(idxs) > max_num_novel:
            idxs = rng.choice(idxs, size=max_num_novel, replace=False)
        idxs = np.concatenate([idxs, np.where(y != 0)[0]])
        return X[idxs], y[idxs]
    classes, counts = np.unique(y, return_counts=True)
    idx = []
    if sampling == "us":
        n_min = counts[classes != 0].min()
        for c in classes:
            c_idx = np.where(y == c)[0]
            idx.append(c_idx[rng.permutation(len(c_idx))[:n_min]])
    else:  # "os"
        n_max = counts[classes != 0].max()
        for c in classes:
            c_idx = np.where(y == c)[0]
            if len(c_idx) < n_max:
                c_idx = rng.choice(c_idx, size=n_max, replace=True)
            elif len(c_idx) > n_max:
                c_idx = rng.choice(c_idx, size=n_max, replace=False)
            idx.append(c_idx)
    idx = np.concatenate(idx, axis=0)
    return X[idx], y[idx]


class _LogisticRegressionCV:
    """Official ``exLogisticRegressionCV``: C chosen among
    ``logspace(-5, 5, n_C)`` by stratified K-fold mean average precision,
    with class balancing on every training split and on the final fit."""

    def __init__(self, n_splits: int, n_C: int, sampling: str, rng) -> None:
        self.n_splits, self.n_C, self.sampling, self.rng = n_splits, n_C, sampling, rng

    def fit(self, X: np.ndarray, y: np.ndarray) -> "_LogisticRegressionCV":
        self.le = LabelEncoder().fit(y)
        y = self.le.transform(y)
        Cs = np.logspace(-5, 5, self.n_C)
        scores = np.zeros((self.n_splits, self.n_C))
        for i, (tr, te) in enumerate(
            StratifiedKFold(n_splits=self.n_splits).split(X, y)
        ):
            X_tr, y_tr = _balance(X[tr], y[tr], self.sampling, self.rng)
            for j, C in enumerate(Cs):
                clf = LogisticRegression(C=C).fit(X_tr, y_tr)
                proba = _softmax_decision(clf, X[te])
                scores[i, j] = np.mean(
                    [
                        average_precision_score(y[te] == c, proba[:, c])
                        for c in np.unique(y[te])
                    ]
                )
        X_b, y_b = _balance(X, y, self.sampling, self.rng)
        self.clf = LogisticRegression(C=Cs[int(np.argmax(scores.mean(axis=0)))]).fit(
            X_b, y_b
        )
        return self

    def predict_proba(self, X: np.ndarray, n_classes: int):
        """Probabilities over ``n_classes`` (zero for classes unseen in fit)
        and the 0/1 flags of the classes seen in fit (official
        ``predict_proba_w_blank``)."""
        proba = np.zeros((X.shape[0], n_classes))
        proba[:, self.le.classes_] = _softmax_decision(self.clf, X)
        flags = np.zeros(n_classes)
        flags[self.le.classes_] = 1.0
        return proba, flags


class BCM(BaseGFSSMethod):
    """BCM — base-class mining (Sakai et al., NeurIPS 2024).

    Port of the official ``Classifier`` (https://github.com/IBM/BCM,
    ``src/bcm.py``; parity test on its output). For each base class β
    mapped to novel classes, a logistic regression g_β on the frozen
    features of **all** support pixels separates "not novel" (label 0) from
    the novel classes of β (C by stratified 5-fold CV of the mean average
    precision, class balancing by ``sampling``). At inference the base
    prediction is kept, except where it is β and g_β predicts a novel class
    (decided at the output resolution, after upsampling the base and g_β
    probabilities separately, as in the official code).

    The base → novel mapping is ``mined`` as in the paper (top-``top_k`` base
    classes predicted on the novel support pixels) or taken from the
    ``hierarchy`` (each novel class's mother). Differences: ``sklearnex``
    replaced by ``sklearn`` (same API); the random generator is a seeded
    ``RandomState`` instead of NumPy's global one; in the ensemble, one-shot
    datasets are single support tiles (official: the i-th shot of every novel
    class — identical with one novel class) and the all-tiles weight is
    ``ensemble_full_weight`` for any K (official: a fixed list that gives 5
    only for K = 5).

    Args:
        mapping: ``mined`` (official) or ``hierarchy``.
        top_k: Base classes kept per novel class (``mined``; official 1).
        sampling: ``us`` (under-sampling, official default), ``os`` or ``bg``.
        beta: Power applied to the features before g_β (official 1.0;
            values < 1 need non-negative features, e.g. after a ReLU).
        n_splits: CV folds (official 5).
        n_C: Number of C values in ``logspace(-5, 5)`` (official 10).
        seed: Seed of the balancing sampler.
        ensemble: Shot-wise ensemble (paper §4.5): one model per support
            tile plus the model on all tiles, probabilities averaged with
            weights 1 and ``ensemble_full_weight`` (per class, over the
            models that saw it). Used only with more than one support tile.
        ensemble_full_weight: Weight of the all-tiles model (official 5).

    Example YAML::

        gfss:
          method:
            _target_: pytorch_segmentation_models_trainer.few_shot.methods.bcm.BCM
            mapping: mined
            top_k: 1
            sampling: us
        pl_trainer:
          max_steps: 0
    """

    def __init__(
        self,
        mapping: str = "mined",
        top_k: int = 1,
        sampling: str = "us",
        beta: float = 1.0,
        n_splits: int = 5,
        n_C: int = 10,
        seed: int = 0,
        ensemble: bool = False,
        ensemble_full_weight: float = 5.0,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        if mapping not in {"mined", "hierarchy"}:
            raise ValueError(
                f"mapping must be 'mined' or 'hierarchy', got {mapping!r}."
            )
        if sampling not in {"us", "os", "bg"}:
            raise ValueError(f"sampling must be 'us', 'os' or 'bg', got {sampling!r}.")
        self.mapping, self.top_k, self.sampling = mapping, int(top_k), sampling
        self.beta, self.n_splits, self.n_C, self.seed = (
            float(beta),
            int(n_splits),
            int(n_C),
            int(seed),
        )
        self.ensemble = bool(ensemble)
        self.ensemble_full_weight = float(ensemble_full_weight)
        self.table: Dict[int, List[int]] = {}
        self.models: Dict[int, List[_LogisticRegressionCV]] = {}
        self.model_weights: Dict[int, List[float]] = {}

    def _base_logits(self, features: Tensor) -> Tensor:
        return FrozenLinearHeadSegmenter.linear(
            features, self.base_weight, self.base_bias
        )

    def _mine_table(self, features: Tensor, masks: Tensor) -> Dict[int, List[int]]:
        base_pred = self._base_logits(features).argmax(1)
        table: Dict[int, List[int]] = {}
        for cls in self.hierarchy.novel_classes:
            found, counts = base_pred[masks == cls].unique(return_counts=True)
            ranked = sorted(
                zip(found.tolist(), counts.tolist()), key=lambda x: x[1], reverse=True
            )
            for base, _ in ranked[: self.top_k] if self.top_k != -1 else ranked:
                table.setdefault(base, []).append(cls)
        return dict(sorted(table.items()))

    def _features_np(self, features: Tensor) -> np.ndarray:
        X = (
            features.permute(0, 2, 3, 1)
            .reshape(-1, features.shape[1])
            .detach()
            .cpu()
            .numpy()
        )
        return np.power(X, self.beta) if self.beta != 1.0 else X

    def init_from_support(self, features: Tensor, masks: Tensor) -> None:
        """Mine (or read) the mapping and fit one g_β per mapped base class."""
        h = self.hierarchy
        if self.mapping == "hierarchy":
            self.table = {}
            for cls in h.novel_classes:
                if (masks == cls).any():
                    self.table.setdefault(h.mother_of(cls), []).append(cls)
            self.table = dict(sorted(self.table.items()))
        else:
            self.table = self._mine_table(features, masks)
        rng = np.random.RandomState(self.seed)
        X_all = self._features_np(features)
        m = masks.reshape(-1).cpu().numpy()
        n_tiles = features.shape[0]
        tile = np.repeat(np.arange(n_tiles), features.shape[2] * features.shape[3])
        use_ensemble = self.ensemble and n_tiles > 1
        self.models, self.model_weights = {}, {}
        for base, novels in self.table.items():
            y = np.zeros_like(m)
            for j, cls in enumerate(sorted(novels), start=1):
                y[m == cls] = j
            valid = m != 255
            subsets = (
                [valid & (tile == i) for i in range(n_tiles)] if use_ensemble else []
            )
            weights = [1.0] * len(subsets)
            subsets.append(valid)
            weights.append(self.ensemble_full_weight if use_ensemble else 1.0)
            self.models[base] = []
            for subset in subsets:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", ConvergenceWarning)
                    model = _LogisticRegressionCV(
                        self.n_splits, self.n_C, self.sampling, rng
                    ).fit(X_all[subset], y[subset])
                self.models[base].append(model)
            self.model_weights[base] = weights
        self.table = {k: sorted(v) for k, v in self.table.items()}

    def _novel_probas(self, features: Tensor, base: int) -> Tensor:
        b, _, hh, ww = features.shape
        n = len(self.table[base]) + 1
        X = self._features_np(features)
        total, norm = 0.0, 0.0
        for model, weight in zip(self.models[base], self.model_weights[base]):
            proba, flags = model.predict_proba(X, n)
            total = total + weight * proba
            norm = norm + weight * flags
        proba = total / np.maximum(norm, 1e-12)[None, :]
        return (
            torch.from_numpy(proba).to(features).view(b, hh, ww, n).permute(0, 3, 1, 2)
        )

    def decode_to(
        self, features: Tensor, size: Sequence[int], upsample: Callable
    ) -> Tensor:
        """Logits at the output resolution whose argmax is the official BCM
        prediction (base classes: log of the upsampled base probabilities;
        a novel class gets the top logit where it overrides)."""
        # official: the base PROBABILITIES are upsampled (not the logits)
        base_probas = upsample(
            self.base_probabilities(self._base_logits(features)), size
        )
        logits = torch.log(base_probas.clamp(min=1e-10))
        out = logits.new_full(
            (logits.shape[0], self.hierarchy.num_classes, *logits.shape[2:]), _FLOOR
        )
        out[:, : self.hierarchy.num_base_classes] = logits
        base_pred = base_probas.argmax(1)
        top = logits.max(1).values + 1.0
        for base, novels in self.table.items():
            here = base_pred == base
            if not here.any():
                continue
            g_pred = upsample(self._novel_probas(features, base), size).argmax(1)
            for j, cls in enumerate(novels, start=1):
                out[:, cls] = torch.where(here & (g_pred == j), top, out[:, cls])
        return out

    def forward(self, features: Tensor) -> Tensor:
        """Feature-resolution logits (same rule without upsampling)."""
        return self.decode_to(features, features.shape[-2:], lambda x, size: x)
