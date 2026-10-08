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

import json
from typing import Optional, Sequence, Union

import torch
from torch import Tensor

from pytorch_segmentation_models_trainer.few_shot.base_method import BaseGFSSMethod
from pytorch_segmentation_models_trainer.few_shot.losses import (
    class_prior,
    default_not_novel_weight,
    entropy_and_marginal_kl,
    hierarchical_kd,
    projected_ce,
    support_one_hot,
)
from pytorch_segmentation_models_trainer.few_shot.methods.diam import _logits


def sinkhorn(
    cost: Tensor,
    target: Tensor,
    source: Tensor,
    lam: float,
    eps: float = 1e-7,
    max_iter: int = 10000,
) -> Tensor:
    """Entropic optimal transport (Sinkhorn-Knopp), as in the official
    ``compute_optimal_transport`` (plus an iteration cap)."""
    plan = torch.exp(-lam * cost)
    plan = plan / plan.sum()
    u = torch.zeros(cost.shape[0], device=cost.device, dtype=cost.dtype)
    for _ in range(max_iter):
        if torch.max(torch.abs(u - plan.sum(1))) <= eps:
            break
        u = plan.sum(1)
        plan = plan * (target / u).reshape(-1, 1)
        plan = plan * (source / plan.sum(0)).reshape(1, -1)
    return plan


class ClassTrans(BaseGFSSMethod):
    """ClassTrans — Class Similarity Transition (Wang et al., CVPRW 2024).

    Port of the official ``TransitionClassifier``
    (https://github.com/earth-insights/ClassTrans, ``src/classifier.py``),
    which differs from the paper in several points (see the research notes,
    ``03_classtrans.md`` §10). Faithful to the **code**:

    * novel rows initialised by optimal transport between support novel
      features and base-predicted regions (``W_n = W_b · Tᵀ``);
    * classification logits ``W·f + b`` plus transition logits
      ``layer_scale ⊙ (S(f) · W_snapshot f)`` (snapshot logits without bias),
      with ``S = (W_c f + b_c) ⊗ (W_r f + b_r)`` (one linear layer each) and
      ``layer_scale`` starting at 0;
    * loss ``w_ce·CE_S + w_ent·H(p_q) + w_kl·KL(marginal ‖ π) + w_kd·KD``
      (query terms use the classification branch only), with the support CE
      replaced by LDAM from iteration ``ldam_start_iter`` on; SGD with
      momentum and weight decay; π re-estimated at ``pi_update_at``.

    Not ported: the OEM-specific post-processing of ``test.py`` and the use
    of the support valid mask for query terms. Generalisations: novel classes
    are summed into their mother (background in the original); one classifier
    per query map (the official code supports a single query image); the
    transition parameters are drawn once from the support, not per query.

    Args:
        weights: ``[w_ce, w_ent, w_kl, w_kd]`` (official: 650, 3, 16, 7).
        adapt_iter: SGD iterations per query batch (official: 130).
        lr: Learning rate (official: 9e-5).
        momentum: SGD momentum (official: 0.9).
        weight_decay: SGD weight decay (official: 5e-4).
        ldam_start_iter: First (0-based) iteration using LDAM (official: 101).
        ldam_max_margin: Largest LDAM margin (official: 6).
        class_counts: Pixel counts (or frequencies) of the ``num_classes``
            classes for the LDAM margins ``∝ n^(-1/4)``. ``None``: counted on
            the support labels, every base class receiving the number of
            "not novel"/base-labelled pixels [adaptation; the official code
            hard-codes OpenEarthMap counts].
        pi_update_at: Iterations (1-based) where π is re-estimated.
        fine_tune_base_classifier: Also optimise the base rows (official).
        ot_lambda: Entropic regularisation of the transport (official: 0.1).
        base_class_counts: Pixel counts of the ``num_base_classes`` base
            classes on the training set (``mode: count-class-pixels`` with the
            base mapping); novel counts then come from the support, as in the
            paper ("estimated via D_train and D_support"). A list, or the path
            of the JSON written by ``count-class-pixels``. Ignored when
            ``class_counts`` is given.
        hierarchical_mask: Ablation (HierTrans): the transition into each
            novel class only comes from its mother's column (hard
            hierarchical prior); base rows unchanged.
        not_novel_weight: CE weight of "not novel" pixels; ``None`` =
            official rule (0.01 with one support map per novel class, else 0.15).

    Example YAML::

        gfss:
          method:
            _target_: pytorch_segmentation_models_trainer.few_shot.methods.classtrans.ClassTrans
            weights: [650, 3, 16, 7]
            adapt_iter: 130
            lr: 9.0e-5
        pl_trainer:
          max_steps: 0
          inference_mode: false
    """

    transductive = True

    def __init__(
        self,
        weights: Sequence[float] = (650.0, 3.0, 16.0, 7.0),
        adapt_iter: int = 130,
        lr: float = 9e-5,
        momentum: float = 0.9,
        weight_decay: float = 5e-4,
        ldam_start_iter: int = 101,
        ldam_max_margin: float = 6.0,
        class_counts: Optional[Sequence[float]] = None,
        pi_update_at: Sequence[int] = (10, 20, 30, 40, 50, 100),
        fine_tune_base_classifier: bool = True,
        ot_lambda: float = 0.1,
        not_novel_weight: Optional[float] = None,
        hierarchical_mask: bool = False,
        base_class_counts: Optional[Union[Sequence[float], str]] = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.weights = [float(w) for w in weights]
        self.adapt_iter = int(adapt_iter)
        self.lr = float(lr)
        self.momentum = float(momentum)
        self.weight_decay = float(weight_decay)
        self.ldam_start_iter = int(ldam_start_iter)
        self.ldam_max_margin = float(ldam_max_margin)
        self.class_counts = (
            None if class_counts is None else [float(c) for c in class_counts]
        )
        self.pi_update_at = [int(i) for i in pi_update_at]
        self.fine_tune_base_classifier = fine_tune_base_classifier
        self.ot_lambda = float(ot_lambda)
        self.not_novel_weight = not_novel_weight
        self.hierarchical_mask = hierarchical_mask
        # A JSON path (mode count-class-pixels) is read lazily in
        # init_from_support: the file may be produced by an earlier pipeline step.
        self.base_class_counts = (
            base_class_counts
            if base_class_counts is None or isinstance(base_class_counts, str)
            else [float(c) for c in base_class_counts]
        )
        self._task_params: Optional[dict] = None

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def _support_counts(self, masks: Tensor) -> Tensor:
        h = self.hierarchy
        counts = torch.zeros(h.num_classes, dtype=torch.float64)
        for c in range(h.num_classes):
            counts[c] = float((masks == c).sum())
        base_like = float((masks == self.not_novel_index).sum()) + float(
            counts[: h.num_base_classes].sum()
        )
        counts[: h.num_base_classes] = base_like
        return counts.clamp(min=1.0)

    def init_from_support(self, features: Tensor, masks: Tensor) -> None:
        """Optimal-transport novel rows and random transition layers."""
        h = self.hierarchy
        nb, nn_ = h.num_base_classes, len(h.novel_classes)
        snap = _logits(
            features.unsqueeze(0),
            self.base_weight.T.unsqueeze(0),
            self.base_bias.unsqueeze(0),
        )
        pseudo = snap.argmax(dim=2)[0]  # (S, h, w)
        f = features.unsqueeze(0)
        cost = features.new_zeros(nn_, nb)
        for b in range(nb):
            base_vec = (
                (f * (pseudo == b).unsqueeze(0).unsqueeze(2))
                .mean(dim=(1, 3, 4))
                .squeeze()
            )
            for j, cls in enumerate(h.novel_classes):
                novel_vec = (
                    (f * (masks == cls).unsqueeze(0).unsqueeze(2))
                    .mean(dim=(1, 3, 4))
                    .squeeze()
                )
                cost[j, b] = torch.square(base_vec - novel_vec).sum()
        source = features.new_ones(nb) / nb
        target = features.new_ones(nn_) / nn_
        plan = sinkhorn(cost, target, source, self.ot_lambda)
        self.register_buffer("novel_weight", self.base_weight.T @ plan.T)
        self.register_buffer("novel_bias", features.new_zeros(nn_))
        dim = features.shape[1]
        row = torch.nn.init.normal_(features.new_zeros(dim, nb), mean=0, std=0.01)
        col = torch.nn.init.normal_(
            features.new_zeros(dim, h.num_classes), mean=0, std=0.01
        )
        self.register_buffer("row_weight", row)
        self.register_buffer("col_weight", col)
        if self.class_counts is not None:
            counts = torch.tensor(self.class_counts, dtype=torch.float64)
        elif self.base_class_counts is not None:
            if isinstance(self.base_class_counts, str):
                with open(self.base_class_counts) as f:
                    self.base_class_counts = [float(c) for c in json.load(f)["counts"]]
            if len(self.base_class_counts) != nb:
                raise ValueError(
                    f"base_class_counts must have {nb} entries, "
                    f"got {len(self.base_class_counts)}."
                )
            counts = self._support_counts(masks)
            counts[:nb] = torch.tensor(self.base_class_counts, dtype=torch.float64)
            counts = counts.clamp(min=1.0)
        else:
            counts = self._support_counts(masks)
        if counts.numel() != h.num_classes:
            raise ValueError(
                f"class_counts must have {h.num_classes} entries, got {counts.numel()}."
            )
        margins = 1.0 / torch.sqrt(torch.sqrt(counts / counts.sum()))
        margins = margins * (self.ldam_max_margin / margins.max())
        self.register_buffer("ldam_margins", margins.to(features.dtype))
        self._n_support = features.shape[0]

    # ------------------------------------------------------------------
    # Logits
    # ------------------------------------------------------------------

    def _initial_params(self, n_tasks: int) -> dict:
        def rep(t):
            return t.unsqueeze(0).repeat(n_tasks, *([1] * t.dim())).clone()

        nb, nc = self.hierarchy.num_base_classes, self.hierarchy.num_classes
        return {
            "base_w": rep(self.base_weight.T),
            "base_b": rep(self.base_bias),
            "novel_w": rep(self.novel_weight),
            "novel_b": rep(self.novel_bias),
            "row_w": rep(self.row_weight),
            "row_b": self.row_weight.new_zeros(n_tasks, nb),
            "col_w": rep(self.col_weight),
            "col_b": self.col_weight.new_zeros(n_tasks, nc),
            "layer_scale": self.col_weight.new_zeros(n_tasks, nc),
        }

    @staticmethod
    def _classification(f5: Tensor, p: dict) -> Tensor:
        w = torch.cat([p["base_w"], p["novel_w"]], dim=2)
        b = torch.cat([p["base_b"], p["novel_b"]], dim=1)
        return _logits(f5, w, b)

    def _transition(self, f5: Tensor, p: dict) -> Tensor:
        snapshot = torch.einsum("bochw,cC->boChw", f5, self.base_weight.T)
        row = _logits(f5, p["row_w"], p["row_b"])
        col = _logits(f5, p["col_w"], p["col_b"])
        matrix = torch.einsum("bochw,borhw->bocrhw", col, row)
        if self.hierarchical_mask:
            h = self.hierarchy
            mask = matrix.new_ones(h.num_classes, h.num_base_classes)
            for n in h.novel_classes:
                mask[n] = 0.0
                mask[n, h.mother_of(n)] = 1.0
            matrix = matrix * mask.view(1, 1, *mask.shape, 1, 1)
        out = torch.einsum("borchw,bochw->borhw", matrix, snapshot)
        return out * p["layer_scale"].unsqueeze(1).unsqueeze(3).unsqueeze(4)

    def _all_logits(self, f5: Tensor, p: dict) -> Tensor:
        return self._classification(f5, p) + self._transition(f5, p)

    def _ldam_ce(self, logits: Tensor, masks: Tensor, nn_weight: float) -> Tensor:
        one_hot = support_one_hot(
            masks, self.hierarchy, self.not_novel_index
        ).unsqueeze(0)
        shifted = logits - one_hot * self.ldam_margins.view(1, 1, -1, 1, 1)
        output = torch.where(one_hot.bool(), shifted, logits)
        return projected_ce(
            torch.softmax(output, dim=2),
            masks,
            self.hierarchy,
            self.not_novel_index,
            nn_weight,
        )

    # ------------------------------------------------------------------
    # Adaptation / prediction
    # ------------------------------------------------------------------

    def adapt_to_query(
        self, support_features: Tensor, support_masks: Tensor, query_features: Tensor
    ) -> None:
        """Optimise classifier + transition per query map (official ``optimize``)."""
        w_ce, w_ent, w_kl, w_kd = self.weights
        n_tasks = query_features.shape[0]
        p = self._initial_params(n_tasks)
        names = ["novel_w", "novel_b"]
        if self.fine_tune_base_classifier:
            names += ["base_w", "base_b"]
        names += ["col_w", "col_b", "row_w", "row_b", "layer_scale"]
        params = [p[n].requires_grad_() for n in names]
        optimizer = torch.optim.SGD(
            params, lr=self.lr, momentum=self.momentum, weight_decay=self.weight_decay
        )
        fs = support_features.unsqueeze(0)
        fq = query_features.unsqueeze(1)
        valid_q = fq.new_ones(n_tasks, 1, *fq.shape[-2:])
        snapshot = torch.softmax(
            _logits(
                fq,
                self.base_weight.T.unsqueeze(0).expand(n_tasks, -1, -1),
                self.base_bias.unsqueeze(0).expand(n_tasks, -1),
            ),
            dim=2,
        )
        nn_weight = (
            self.not_novel_weight
            if self.not_novel_weight is not None
            else default_not_novel_weight(
                self._n_support, len(self.hierarchy.novel_classes)
            )
        )
        with torch.no_grad():
            pi = class_prior(torch.softmax(self._all_logits(fq, p), 2), valid_q)
        for it in range(self.adapt_iter):
            logits_s = self._all_logits(fs, p)
            proba_q = torch.softmax(self._classification(fq, p), dim=2)
            kd = hierarchical_kd(proba_q, snapshot, valid_q, self.hierarchy)
            d_kl, entropy = entropy_and_marginal_kl(proba_q, valid_q, pi)
            if it < self.ldam_start_iter:
                ce = projected_ce(
                    torch.softmax(logits_s, 2),
                    support_masks,
                    self.hierarchy,
                    self.not_novel_index,
                    nn_weight,
                )
            else:
                ce = self._ldam_ce(logits_s, support_masks, nn_weight)
            loss = w_ce * ce + w_ent * entropy + w_kl * d_kl + w_kd * kd
            optimizer.zero_grad()
            loss.sum(0).backward()
            optimizer.step()
            if (it + 1) in self.pi_update_at and w_ent != 0:
                with torch.no_grad():
                    pi = class_prior(torch.softmax(self._all_logits(fq, p), 2), valid_q)
        self._task_params = {k: v.detach() for k, v in p.items()}

    def forward(self, features: Tensor) -> Tensor:
        """Classification + transition logits with the parameters of the last
        adapted batch (same size), or the initial ones otherwise."""
        p = self._task_params
        if p is None or p["base_w"].shape[0] != features.shape[0]:
            p = self._initial_params(features.shape[0])
        return self._all_logits(features.unsqueeze(1), p).squeeze(1)
