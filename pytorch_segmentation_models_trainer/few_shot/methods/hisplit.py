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
from typing import Dict, List, Optional, Sequence

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from pytorch_segmentation_models_trainer.few_shot.backbone import (
    FrozenLinearHeadSegmenter,
)
from pytorch_segmentation_models_trainer.few_shot.base_method import BaseGFSSMethod
from pytorch_segmentation_models_trainer.custom_losses.edl_utils import (
    edl_kl_regulariser,
)
from pytorch_segmentation_models_trainer.few_shot.losses import valid_mean
from pytorch_segmentation_models_trainer.few_shot.uncertainty import (
    dissonance,
    normalized_entropy,
    vacuity,
)

logger = logging.getLogger(__name__)

_EPS = 1e-10
_Q_TYPES = {"proto", "proto_prob", "linear", "trans", "edl"}
_TRAINABLE = {"linear", "trans", "edl"}


class HiSplit(BaseGFSSMethod):
    """HiSplit — hierarchical split of a base class (category splitting).

    The frozen base model gives ``p_base``; only the split of each mother
    ``m`` among its children is learned:

    ``p(child) = p_base(m) · q(child | x)``, ``Σ_children q = 1``,

    and every class outside the hierarchy keeps ``p_base``. ``q`` is a
    softmax over the children of each mother of per-child scores ``s_c(x)``:

    * ``proto``: ``τ · cos(f, μ_c)`` with support prototypes ``μ_c``
      (no training);
    * ``proto_prob``: diagonal Gaussian log-likelihood with per-child means
      and a variance shared by the children of a mother (``variance:
      shared``) or per child (``per_class``), floored at ``var_floor``
      (no training);
    * ``linear``: ``w_c·f + b_c`` initialised as ``proto`` and trained on the
      support by ``trainer.fit`` (``pl_trainer.max_steps > 0``);
    * ``trans``: ``linear`` plus, for each query map, ``adapt_iter`` SGD
      steps on ``w_ce·CE_support + w_ent·H(q) + w_kl·KL(q̄ ‖ π)``, where
      the entropy and the child proportions ``q̄`` are weighted by
      ``p_base(m)`` (restricted to the superclass) and ``π`` is
      self-estimated and re-estimated at ``pi_update_at`` (DIaM-style);
    * ``edl``: evidential pair head (D4 4a): evidence
      ``softplus(w_c·f + b_c)``, ``α = e + 1`` and ``q = α / S`` within each
      mother; trained by ``trainer.fit`` with the EDL MSE loss (Sensoy et al.
      2018) plus the KL regulariser annealed linearly over ``kl_anneal_steps``.

    With an evidential base model (``base_output: evidential``, R2-EDL),
    ``p_base`` is the Dirichlet mean and the split divides the mother's
    evidence **and** base rate by ``q`` (``e_c = e_m q_c``, ``a_c = a_m q_c``),
    so ``α_c = α_m q_c`` (Dirichlet aggregation is exact) and the beliefs stay
    non-negative (D4 4c).

    ``uncertainty(features)`` returns per-pixel maps in ``[0, 1]``:
    ``split_entropy`` (normalised entropy of the split of the mother predicted
    by the base model; 0 elsewhere), ``split_vacuity`` (``edl`` only),
    ``base_vacuity`` and ``dissonance`` (evidential base only; dissonance of
    the split opinion over all classes).

    Support targets of ``q``: a pixel labelled with a child is an example of
    that child; a pixel labelled ``not_novel_index`` (S-novel regime) that
    the base model predicts as mother ``m`` is an example of the child
    that keeps the mother's index (the "free negatives" of a fully annotated
    novel class; noisy where the base model is wrong). Other pixels are not
    used.

    Decoding: ``hierarchical`` (default) takes the base argmax and splits
    only pixels predicted as a mother, so predictions of the other classes
    are those of the base model (up to ties within ``decode_eps`` in logit
    space); ``flat`` takes the argmax of ``log p`` over all classes.

    Args:
        q: ``proto``, ``proto_prob``, ``linear`` or ``trans``.
        tau: Cosine temperature of ``proto`` (and init of ``linear``).
        variance: ``shared`` or ``per_class`` (``proto_prob``).
        var_floor: Minimum variance (``proto_prob``).
        decoding: ``hierarchical`` or ``flat``.
        decode_eps: Tie-break scale of the hierarchical decoding.
        trans_weights: ``[w_ce, w_ent, w_kl]`` (``trans``).
        adapt_iter: SGD iterations per query map (``trans``).
        lr: SGD learning rate (``trans``).
        pi_update_at: Iterations (1-based) where π is re-estimated (``trans``).
        kl_anneal_steps: Steps to reach full KL weight (``edl``).

    Example YAML::

        gfss:
          hierarchy: {3: [3, 5]}
          method:
            _target_: pytorch_segmentation_models_trainer.few_shot.methods.hisplit.HiSplit
            q: proto
            tau: 10.0
        pl_trainer:
          max_steps: 0      # > 0 for q: linear / trans
    """

    def __init__(
        self,
        q: str = "proto",
        tau: float = 10.0,
        variance: str = "shared",
        var_floor: float = 1e-4,
        decoding: str = "hierarchical",
        decode_eps: float = 1e-4,
        trans_weights: Sequence[float] = (1.0, 1.0, 1.0),
        adapt_iter: int = 50,
        lr: float = 1e-3,
        pi_update_at: Sequence[int] = (10,),
        kl_anneal_steps: int = 50,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        if q not in _Q_TYPES:
            raise ValueError(f"q must be one of {sorted(_Q_TYPES)}, got {q!r}.")
        if variance not in {"shared", "per_class"}:
            raise ValueError(
                f"variance must be 'shared' or 'per_class', got {variance!r}."
            )
        if decoding not in {"hierarchical", "flat"}:
            raise ValueError(
                f"decoding must be 'hierarchical' or 'flat', got {decoding!r}."
            )
        self.q = q
        self.tau = float(tau)
        self.variance = variance
        self.var_floor = float(var_floor)
        self.decoding = decoding
        self.decode_eps = float(decode_eps)
        self.trans_weights = [float(w) for w in trans_weights]
        self.adapt_iter = int(adapt_iter)
        self.lr = float(lr)
        self.pi_update_at = [int(i) for i in pi_update_at]
        self.kl_anneal_steps = int(kl_anneal_steps)
        self.transductive = q == "trans"
        self._edl_step = 0
        self._task_params: Optional[tuple] = None

    # ------------------------------------------------------------------
    # Setup / support
    # ------------------------------------------------------------------

    def setup(self, hierarchy, base_weight, base_bias, not_novel_index=254, **kwargs):
        super().setup(hierarchy, base_weight, base_bias, not_novel_index, **kwargs)
        self.child_classes: List[int] = [
            c for m in hierarchy.mothers for c in hierarchy.children_of(m)
        ]
        self.child_mother: List[int] = [
            hierarchy.mother_of(c) for c in self.child_classes
        ]
        n, dim = len(self.child_classes), base_weight.shape[1]
        if self.q in _TRAINABLE:
            self.q_weight = nn.Parameter(base_weight.new_zeros(n, dim))
            self.q_bias = nn.Parameter(base_weight.new_zeros(n))
        else:
            self.register_buffer("q_weight", base_weight.new_zeros(n, dim))
            self.register_buffer("q_bias", base_weight.new_zeros(n))
        self.register_buffer("q_var", base_weight.new_ones(n, dim))

    def _base_probs(self, features: Tensor) -> Tensor:
        return self.base_probabilities(self._base_logits(features))

    def _base_logits(self, features: Tensor) -> Tensor:
        return FrozenLinearHeadSegmenter.linear(
            features, self.base_weight, self.base_bias
        )

    def _targets(self, features: Tensor, masks: Tensor) -> Tensor:
        """Index into ``self.child_classes`` of each support pixel, or -1."""
        targets = torch.full_like(masks, -1)
        base_pred = self._base_probs(features).argmax(1)
        for i, (child, mother) in enumerate(zip(self.child_classes, self.child_mother)):
            targets[masks == child] = i
            if child == mother:
                targets[(masks == self.not_novel_index) & (base_pred == mother)] = i
        return targets

    def init_from_support(self, features: Tensor, masks: Tensor) -> None:
        """Per-child prototypes (and variances) from the support targets."""
        targets = self._targets(features, masks)
        flat = features.permute(0, 2, 3, 1).reshape(-1, features.shape[1])
        t = targets.reshape(-1)
        weight = torch.zeros_like(self.q_weight)
        var = torch.ones_like(self.q_var)
        for i, child in enumerate(self.child_classes):
            x = flat[t == i]
            if x.shape[0] == 0:
                logger.warning("HiSplit: no support pixel for class %d.", child)
                continue
            mu = x.mean(0)
            weight[i] = mu
            var[i] = x.var(0, unbiased=False) if x.shape[0] > 1 else var[i]
        if self.variance == "shared":
            for mother in self.hierarchy.mothers:
                idx = [i for i, m in enumerate(self.child_mother) if m == mother]
                var[idx] = var[idx].mean(0, keepdim=True)
        self.q_var.copy_(var.clamp(min=self.var_floor))
        with torch.no_grad():
            if self.q == "proto_prob":
                self.q_weight.copy_(weight)
            else:
                norm = weight / (weight.norm(dim=1, keepdim=True) + _EPS)
                self.q_weight.copy_(self.tau * norm)
            self.q_bias.zero_()

    # ------------------------------------------------------------------
    # Scores and probabilities
    # ------------------------------------------------------------------

    def _scores(self, features: Tensor, weight: Tensor, bias: Tensor) -> Tensor:
        """Per-child scores ``(B, n_children, h, w)``; weight may be per map."""
        if self.q == "proto":
            f = F.normalize(features, dim=1)
            return torch.einsum("bfhw,cf->bchw", f, weight)
        if self.q == "proto_prob":
            diff = features.unsqueeze(1) - weight.view(1, *weight.shape, 1, 1)
            var = self.q_var.view(1, *self.q_var.shape, 1, 1)
            return -0.5 * (diff.pow(2) / var + torch.log(var)).sum(2)
        if weight.dim() == 3:
            return torch.einsum("bfhw,bcf->bchw", features, weight) + bias.view(
                bias.shape[0], -1, 1, 1
            )
        return torch.einsum("bfhw,cf->bchw", features, weight) + bias.view(1, -1, 1, 1)

    def _groups(self):
        for mother in self.hierarchy.mothers:
            yield mother, [i for i, m in enumerate(self.child_mother) if m == mother]

    def _q(self, scores: Tensor) -> Tensor:
        """Split within the children of each mother: softmax of the scores,
        or the Dirichlet mean ``α / S`` for ``edl``."""
        q = torch.empty_like(scores)
        if self.q == "edl":
            alpha = F.softplus(scores) + 1.0
            for _, idx in self._groups():
                q[:, idx] = alpha[:, idx] / alpha[:, idx].sum(1, keepdim=True)
            return q
        for _, idx in self._groups():
            q[:, idx] = torch.softmax(scores[:, idx], dim=1)
        return q

    def split_probabilities(
        self, features: Tensor, q: Optional[Tensor] = None
    ) -> Tensor:
        """``p(c)`` over all classes ``(B, num_classes, h, w)``."""
        if q is None:
            q = self._q(
                self._scores(features, *self._current_params(features.shape[0]))
            )
        p_base = self._base_probs(features)
        out = p_base.new_zeros(
            p_base.shape[0], self.hierarchy.num_classes, *p_base.shape[2:]
        )
        out[:, : self.hierarchy.num_base_classes] = p_base
        for i, (child, mother) in enumerate(zip(self.child_classes, self.child_mother)):
            out[:, child] = p_base[:, mother] * q[:, i]
        return out

    def _current_params(self, n_maps: int):
        if self._task_params is not None and self._task_params[0].shape[0] == n_maps:
            return self._task_params
        return self.q_weight, self.q_bias

    # ------------------------------------------------------------------
    # Training / transductive adaptation
    # ------------------------------------------------------------------

    def _support_ce(self, features: Tensor, targets: Tensor, weight, bias) -> Tensor:
        q = self._q(self._scores(features, weight, bias))
        valid = targets >= 0
        idx = targets.clamp(min=0).unsqueeze(1)
        p = q.gather(1, idx).squeeze(1)
        ce = -torch.log(p + _EPS)
        return (ce * valid).sum() / valid.sum().clamp(min=1)

    def _edl_loss(self, features: Tensor, targets: Tensor) -> Tensor:
        """EDL MSE + annealed KL of the evidential pair head, per mother."""
        alpha = F.softplus(self._scores(features, self.q_weight, self.q_bias)) + 1.0
        kl_weight = min(1.0, self._edl_step / max(self.kl_anneal_steps, 1))
        total = alpha.new_zeros(())
        for _, idx in self._groups():
            in_group = (targets >= min(idx)) & (targets <= max(idx))
            if not in_group.any():
                continue
            a = alpha[:, idx]
            local = (targets - min(idx)).clamp(min=0, max=len(idx) - 1)
            y = F.one_hot(local, len(idx)).movedim(-1, 1).to(a.dtype)
            s = a.sum(1, keepdim=True)
            p = a / s
            mse = ((y - p) ** 2 + a * (s - a) / (s**2 * (s + 1))).sum(1)
            per_pixel = mse + kl_weight * edl_kl_regulariser(a, y)
            total = total + per_pixel[in_group].mean()
        return total

    def support_loss(self, features: Tensor, masks: Tensor) -> Optional[Tensor]:
        """Loss of q on the support targets (trainable variants): CE
        (``linear``/``trans``) or EDL MSE + annealed KL (``edl``)."""
        if self.q not in _TRAINABLE:
            return None
        targets = self._targets(features, masks)
        if self.q == "edl":
            loss = self._edl_loss(features, targets)
            self._edl_step += 1
            return loss
        return self._support_ce(features, targets, self.q_weight, self.q_bias)

    def _superclass_terms(self, q: Tensor, p_base: Tensor, pi: Tensor):
        """Entropy of q and KL(q̄ ‖ π), weighted by ``p_base(mother)``, per map."""
        ent = torch.zeros(q.shape[0], device=q.device, dtype=q.dtype)
        kl = torch.zeros_like(ent)
        for mother in self.hierarchy.mothers:
            idx = [i for i, m in enumerate(self.child_mother) if m == mother]
            w = p_base[:, mother].unsqueeze(1)
            qm = q[:, idx]
            h = -(qm * torch.log(qm + _EPS)).sum(1, keepdim=True)
            ent = ent + valid_mean(h, w, (1, 2, 3))
            marginal = valid_mean(qm, w, (2, 3))
            kl = kl + (marginal * torch.log(_EPS + marginal / (pi[:, idx] + _EPS))).sum(
                1
            )
        return ent, kl

    def _prior(self, q: Tensor, p_base: Tensor) -> Tensor:
        pi = torch.zeros(q.shape[0], q.shape[1], device=q.device, dtype=q.dtype)
        for mother in self.hierarchy.mothers:
            idx = [i for i, m in enumerate(self.child_mother) if m == mother]
            pi[:, idx] = valid_mean(q[:, idx], p_base[:, mother].unsqueeze(1), (2, 3))
        return pi.detach()

    def adapt_to_query(
        self, support_features: Tensor, support_masks: Tensor, query_features: Tensor
    ) -> None:
        """``trans``: per-map SGD on support CE + superclass entropy/prior."""
        w_ce, w_ent, w_kl = self.trans_weights
        n = query_features.shape[0]
        weight = self.q_weight.detach().unsqueeze(0).repeat(n, 1, 1).requires_grad_()
        bias = self.q_bias.detach().unsqueeze(0).repeat(n, 1).requires_grad_()
        optimizer = torch.optim.SGD([weight, bias], lr=self.lr)
        targets = self._targets(support_features, support_masks)
        p_base = self._base_probs(query_features)
        with torch.no_grad():
            pi = self._prior(
                self._q(self._scores(query_features, weight, bias)), p_base
            )
        for it in range(self.adapt_iter):
            ce = torch.stack(
                [
                    self._support_ce(support_features, targets, weight[i], bias[i])
                    for i in range(n)
                ]
            )
            q = self._q(self._scores(query_features, weight, bias))
            ent, kl = self._superclass_terms(q, p_base, pi)
            loss = w_ce * ce + w_ent * ent + w_kl * kl
            optimizer.zero_grad()
            loss.sum().backward()
            optimizer.step()
            if (it + 1) in self.pi_update_at:
                with torch.no_grad():
                    pi = self._prior(
                        self._q(self._scores(query_features, weight, bias)), p_base
                    )
        self._task_params = (weight.detach(), bias.detach())

    # ------------------------------------------------------------------
    # Uncertainty
    # ------------------------------------------------------------------

    def uncertainty_names(self) -> List[str]:
        """Keys returned by :meth:`uncertainty` (after ``setup``)."""
        names = ["split_entropy"]
        if self.q == "edl":
            names.append("split_vacuity")
        if self.base_output == "evidential":
            names += ["base_vacuity", "dissonance"]
        return names

    def uncertainty(self, features: Tensor) -> Dict[str, Tensor]:
        """Per-pixel uncertainty maps ``(B, h, w)`` in ``[0, 1]`` (see class doc)."""
        scores = self._scores(features, *self._current_params(features.shape[0]))
        q = self._q(scores)
        logits = self._base_logits(features)
        base_pred = logits.argmax(1)
        split_entropy = features.new_zeros(base_pred.shape)
        split_vacuity = features.new_zeros(base_pred.shape)
        for mother, idx in self._groups():
            here = base_pred == mother
            split_entropy = torch.where(
                here, normalized_entropy(q[:, idx]), split_entropy
            )
            if self.q == "edl":
                alpha_q = F.softplus(scores[:, idx]) + 1.0
                split_vacuity = torch.where(here, vacuity(alpha_q), split_vacuity)
        out = {"split_entropy": split_entropy}
        if self.q == "edl":
            out["split_vacuity"] = split_vacuity
        if self.base_output == "evidential":
            alpha = self.base_alpha(logits)
            strength = alpha.sum(1, keepdim=True)
            evidence = alpha - 1.0
            belief = evidence.new_zeros(
                evidence.shape[0], self.hierarchy.num_classes, *evidence.shape[2:]
            )
            belief[:, : self.hierarchy.num_base_classes] = evidence
            for i, (child, mother) in enumerate(
                zip(self.child_classes, self.child_mother)
            ):
                belief[:, child] = evidence[:, mother] * q[:, i]
            out["base_vacuity"] = vacuity(alpha)
            out["dissonance"] = dissonance(belief / strength)
        return out

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def forward(self, features: Tensor) -> Tensor:
        """Log-probabilities (``flat``) or hierarchical decoding logits."""
        q = self._q(self._scores(features, *self._current_params(features.shape[0])))
        p = self.split_probabilities(features, q)
        if self.decoding == "flat":
            return torch.log(p + _EPS)
        logits = torch.log(p + _EPS)
        p_base = self._base_probs(features)
        for i, (child, mother) in enumerate(zip(self.child_classes, self.child_mother)):
            logits[:, child] = torch.log(p_base[:, mother] + _EPS) + self.decode_eps * (
                q[:, i] - 1.0
            )
        return logits
