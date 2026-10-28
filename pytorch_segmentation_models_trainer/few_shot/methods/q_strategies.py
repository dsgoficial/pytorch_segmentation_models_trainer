# -*- coding: utf-8 -*-
"""
/***************************************************************************
 pytorch_segmentation_models_trainer
                              -------------------
        begin                : 2026-10-26
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

from __future__ import annotations

from abc import ABC
from typing import TYPE_CHECKING, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor

from pytorch_segmentation_models_trainer.few_shot.uncertainty import normalized_entropy

_EPS = 1e-10

if TYPE_CHECKING:
    from pytorch_segmentation_models_trainer.few_shot.methods.hisplit import HiSplit


class QStrategy(ABC):
    """One of ``HiSplit``'s 7 ``q`` mechanisms (``proto``/``proto_prob``/
    ``linear``/``trans``/``edl``/``logreg``/``linear_aef_gate``) -- same
    ``ABC`` + concrete-subclass shape already used elsewhere in the
    framework for this kind of dispatch (``ExportInferenceTemplate`` in
    ``tools/inference/export_inference.py``, ``AbstractConversionStrategy``
    in ``tools/dataset_handlers/convert_dataset.py``).

    Each strategy is a small, independently-constructable object owning
    ONLY the parameters that mechanism actually uses (``TransStrategy``'s
    ``adapt_iter``/``lr``/``pseudo_label_*``/``tversky_*``, ``EDLStrategy``'s
    ``kl_anneal_steps``/``edl_base_rate*``, ``LinearAefGateStrategy``'s
    ``aef_gate_epochs``/``aef_gate_lr``, ``ProtoProbStrategy``'s
    ``variance``/``var_floor``) -- NOT a flat bag of every method's
    parameters on ``HiSplit`` itself, which is what this split replaced
    (``HiSplit.__init__`` used to accept ~50 parameters, a mix of every
    q's own config and the orthogonal modifiers below; it now accepts
    ~30, only the genuinely shared ones).

    Built standalone (``q="trans"`` resolves to ``TransStrategy()`` with
    every default; ``q=TransStrategy(adapt_iter=10)`` for a non-default
    one; a Hydra sub-config with its own ``_target_`` also works, since
    ``HiSplit`` itself is instantiated with ``_recursive_=False`` and
    resolves ``q`` by hand -- see ``HiSplit.__init__``) and only attached
    to its owning ``HiSplit`` afterwards, via :meth:`bind` -- a strategy
    cannot take the ``HiSplit`` instance in its own ``__init__`` without
    a circular construction order (``HiSplit`` needs ``q`` built first to
    finish its own ``__init__``).

    The ~15 ORTHOGONAL MODIFIERS (``leak``/``widen``/``abstain_measure``/
    ``confidence_weight_measure``/``support_loss_weight_measure``/
    ``hard_decode_measure``/``bcm_teacher_weight``/``logreg_n_*``/
    ``aef_gate*``) stay on ``HiSplit`` itself, not owned by any one
    strategy, because they compose freely with (almost) any of the 7 --
    deliberate, the research actively tests those combinations.
    ``logreg_n_splits``/``n_C``/``sampling``/``n_jobs`` look ``logreg``-
    specific but are NOT: ``bcm_teacher_weight`` (meaningful only with
    ``trans``, but settable with any q) fits the exact same
    ``LogisticRegressionCV`` as its teacher, via ``HiSplit._fit_logreg``,
    so those 4 can't be owned by :class:`LogregStrategy` alone without a
    strategy that isn't ``logreg`` reaching into it. The AEF gate's own
    construction (``aef_gate_linear``, ``aef_dim``) also stays in
    ``HiSplit.setup`` for the same reason: ``aef_gate: true`` composes it
    with any trainable q, not only ``linear_aef_gate`` (:attr:`uses_aef_gate`
    is just the trigger a strategy opts into, the gate itself is shared
    infrastructure) -- only ``aef_gate_epochs``/``aef_gate_lr`` (the
    *fit*, used only by :class:`LinearAefGateStrategy`'s own self-fit)
    move here.

    ``edl_base_rate``/``edl_baserate`` (§2.7 of the analysis doc) is NOT
    an 8th ``q`` -- it's a modifier on top of ``q: edl`` owned by
    :class:`EDLStrategy` itself (``edl_base_rate: "uniform"|
    "inverse_frequency"``), not shared, since nothing else consumes it.

    Default implementations below match ``linear``'s behaviour (the
    shared per-mother-softmax-of-a-linear-score formula that ``trans``/
    ``edl``/``linear_aef_gate`` also start from) -- a subclass overrides
    only what actually differs for it."""

    #: Registry key (``HiSplit(q="...")``) -- also exposed as ``HiSplit.q``
    #: (a few call sites and log messages still want the bare name).
    name: str = ""
    #: ``q_weight``/``q_bias`` are ``nn.Parameter`` (trained by the generic
    #: ``pl_trainer`` loop via :meth:`HiSplit.support_loss`) instead of
    #: buffers (self-fit, or not fit via gradient at all).
    trainable: bool = False
    #: Whether this q needs ``HiSplit.adapt_to_query`` (per-task SGD
    #: against the query, ``trans`` only).
    transductive: bool = False
    #: Whether ``HiSplit._qfeat`` applies the AEF multiplicative gate for
    #: this q on its own (``linear_aef_gate``'s self-fit covers both the
    #: gate and the head together) -- independent of the ``aef_gate``
    #: flag, which composes the same gate with any *trainable* q instead.
    uses_aef_gate: bool = False
    #: Whether ``HiSplit.init_from_support`` must fit the BCM-style
    #: ``LogisticRegressionCV`` teacher (``logreg`` always needs its own
    #: fitted model to score with; other q's only need it when
    #: ``bcm_teacher_weight != 0`` -- handled separately, by that flag,
    #: not this one).
    needs_logreg_fit: bool = False
    #: ``proto_prob``-only config, read by the SHARED prototype-init code
    #: in ``HiSplit.init_from_support`` (which always computes per-child
    #: variance, even though only :class:`ProtoProbStrategy` consumes it)
    #: -- harmless default for every other strategy.
    variance: str = "shared"
    var_floor: float = 1e-4

    def __init__(self) -> None:
        self.h: Optional["HiSplit"] = None

    def bind(self, h: "HiSplit") -> None:
        """Attach the owning ``HiSplit`` -- called once, right after
        ``HiSplit.__init__`` resolves ``q`` into a strategy instance."""
        self.h = h

    def init_weight(self, raw_mean: Tensor) -> Tensor:
        """``q_weight`` from the per-child raw mean feature ``HiSplit.
        init_from_support`` already computed -- default: L2-normalised and
        scaled by ``HiSplit.tau`` (the prototype-as-cosine-anchor every q
        but ``proto_prob`` starts from, trained further from there by
        ``trans``/``edl``/``linear_aef_gate``'s own fit)."""
        norm = raw_mean / (raw_mean.norm(dim=1, keepdim=True) + 1e-10)
        return self.h.tau * norm

    def scores(self, features: Tensor, weight: Tensor, bias: Tensor) -> Tensor:
        """Per-child raw scores ``(B, n_children, h, w)`` (``weight`` may
        be ``(n, n_children, F)`` for ``trans``'s per-map task params) --
        default: the linear/bilinear score shared by ``linear``/``trans``/
        ``edl``/``linear_aef_gate``. Called with features already through
        ``HiSplit._qfeat`` (gate/Tukey-power) -- see ``HiSplit._scores``."""
        if weight.dim() == 3:
            return torch.einsum("bfhw,bcf->bchw", features, weight) + bias.view(
                bias.shape[0], -1, 1, 1
            )
        return torch.einsum("bfhw,cf->bchw", features, weight) + bias.view(
            1, -1, 1, 1
        )

    def normalize(self, scores: Tensor) -> Tensor:
        """``scores`` → ``q(child|x)`` -- default: softmax within each
        mother's children (``edl`` overrides with the Dirichlet mean)."""
        q = torch.empty_like(scores)
        for _, idx in self.h._groups():
            q[:, idx] = torch.softmax(scores[:, idx], dim=1)
        return q

    def init_from_support(self, features: Tensor, targets: Tensor) -> None:
        """Extra fit beyond the shared prototype init that ``HiSplit.
        init_from_support`` always does first (default: none -- the
        prototype values stay as the starting point ``linear``'s/
        ``trans``'s Adam/SGD then adjusts via :meth:`training_loss`).
        ``features``/``targets`` are the full, un-flattened tensors
        (same ones ``HiSplit.init_from_support`` itself received)."""
        return None

    def training_loss(
        self, features: Tensor, targets: Tensor, sample_weight: Optional[Tensor]
    ) -> Optional[Tensor]:
        """This q's own term of ``HiSplit.support_loss`` (default: the
        per-mother cross-entropy shared by ``linear``/``trans``;
        ``None`` for every non-:attr:`trainable` q, which has nothing for
        the generic ``pl_trainer`` loop to optimise)."""
        if not self.trainable:
            return None
        return self.h._support_ce(
            features, targets, self.h.q_weight, self.h.q_bias, sample_weight
        )

    def adapt_to_query(
        self, support_features: Tensor, support_masks: Tensor, query_features: Tensor
    ) -> None:
        """Transductive per-task adaptation (default: none -- only
        ``trans``, gated by :attr:`transductive`, is ever called)."""
        return None


class ProtoStrategy(QStrategy):
    """``s_c(x) = τ·cos(f(x), μ_c)`` -- zero parameters adjusted, decision
    by cosine distance to the per-child prototype (§2.2). No extra config
    of its own: ``tau`` stays on ``HiSplit`` (shared prototype anchor)."""

    name = "proto"

    def scores(self, features: Tensor, weight: Tensor, bias: Tensor) -> Tensor:
        f = F.normalize(features, dim=1)
        return torch.einsum("bfhw,cf->bchw", f, weight)


class ProtoProbStrategy(QStrategy):
    """``s_c(x) = log N(f(x); μ_c, Σ_c)`` -- Gaussian likelihood instead of
    plain cosine, still zero gradient (§2.3).

    Args:
        variance: ``"shared"`` (one Σ per mother) or ``"per_class"``.
        var_floor: Minimum variance (avoids collapse with small K).
    """

    name = "proto_prob"

    def __init__(self, variance: str = "shared", var_floor: float = 1e-4) -> None:
        super().__init__()
        if variance not in {"shared", "per_class"}:
            raise ValueError(
                f"variance must be 'shared' or 'per_class', got {variance!r}."
            )
        self.variance = variance
        self.var_floor = float(var_floor)

    def init_weight(self, raw_mean: Tensor) -> Tensor:
        return raw_mean  # not normalised/scaled by tau -- the raw per-child mean IS μ_c

    def scores(self, features: Tensor, weight: Tensor, bias: Tensor) -> Tensor:
        diff = features.unsqueeze(1) - weight.view(1, *weight.shape, 1, 1)
        var = self.h.q_var.view(1, *self.h.q_var.shape, 1, 1)
        return -0.5 * (diff.pow(2) / var + torch.log(var)).sum(2)


class LinearStrategy(QStrategy):
    """``s_c(x) = w_c·f(x) + b_c``, ``w_c, b_c`` initialised at the
    prototype and adjusted by the generic ``pl_trainer`` loop (CE) --
    scores/normalize/training_loss are all the :class:`QStrategy` default
    (§2.4); no extra config of its own, just :attr:`trainable`."""

    name = "linear"
    trainable = True


class TransStrategy(LinearStrategy):
    """``linear`` + transductive per-task refinement against the query
    (§2.5) -- :attr:`transductive` routes ``HiSplit.adapt_to_query`` here;
    scores/normalize stay ``LinearStrategy``'s (inherited), used through
    ``HiSplit._scores``/``_q`` so the gate/Tukey-power of ``HiSplit.
    _qfeat`` still apply even though this runs its own optimizer loop
    instead of ``pl_trainer``.

    Owns every parameter exclusive to the per-task SGD loop and the 3
    sub-modifiers only it ever consumes (:meth:`HiSplit._pseudo_label_terms`/
    ``_support_tversky``/``_bcm_teacher_term`` are only called from
    :meth:`adapt_to_query` below -- confirmed by grep, not inferred).
    ``bcm_teacher_weight`` itself (not the 4 ``logreg_n_*`` CV params the
    teacher's classifier needs) stays on ``HiSplit``, since the teacher
    fit is triggered from the SHARED ``init_from_support`` regardless of
    which q is active (see class doc of :class:`QStrategy`).

    Args:
        adapt_iter: SGD steps per query batch.
        lr: SGD learning rate.
        pi_update_at: Iterations (1-indexed) at which the class-prior
            ``π`` is re-estimated from the current ``q``.
        trans_weights: ``(w_ce, w_ent, w_kl)`` loss weights (Eq. 13).
        pseudo_label_weight: Weight of the confidence-gated pseudo-label
            CE term on query pixels (``0`` disables it).
        pseudo_label_contrast_weight: Weight of the hinge repulsion
            between the kept/novel pseudo-labelled feature centroids.
        pseudo_label_contrast_margin: Hinge margin for the term above.
        pseudo_label_threshold: Entropy threshold below which a
            pseudo-label is "confident enough" to use.
        pseudo_label_threshold_start: If set, the threshold is linearly
            interpolated from this value (at iteration 0) to
            ``pseudo_label_threshold`` (at ``adapt_iter - 1``); ``None``
            keeps it constant.
        pseudo_label_start_iter: Iteration at which pseudo-labelling
            begins (``0`` = from the start).
        pseudo_label_min_pixels: Minimum confident pixels per side
            (kept/novel) for the contrast term to apply on a given map.
        pseudo_label_teacher_period: If ``> 0``, the pseudo-label teacher
            is a periodically-refreshed snapshot of ``q`` (every this
            many iterations) instead of ``q`` itself.
        tversky_weight: Weight of the Tversky support term (``0``
            disables it).
        tversky_alpha: False-positive weight of the Tversky term.
        tversky_beta: False-negative weight of the Tversky term.
    """

    name = "trans"
    transductive = True

    def __init__(
        self,
        adapt_iter: int = 50,
        lr: float = 1e-3,
        pi_update_at: Sequence[int] = (10,),
        trans_weights: Sequence[float] = (1.0, 1.0, 1.0),
        pseudo_label_weight: float = 0.0,
        pseudo_label_contrast_weight: float = 0.0,
        pseudo_label_contrast_margin: float = 0.0,
        pseudo_label_threshold: float = 0.3,
        pseudo_label_threshold_start: Optional[float] = None,
        pseudo_label_start_iter: int = 0,
        pseudo_label_min_pixels: int = 1,
        pseudo_label_teacher_period: int = 0,
        tversky_weight: float = 0.0,
        tversky_alpha: float = 0.5,
        tversky_beta: float = 0.5,
    ) -> None:
        super().__init__()
        self.adapt_iter = int(adapt_iter)
        self.lr = float(lr)
        self.pi_update_at = [int(i) for i in pi_update_at]
        self.trans_weights = [float(w) for w in trans_weights]
        self.pseudo_label_weight = float(pseudo_label_weight)
        self.pseudo_label_contrast_weight = float(pseudo_label_contrast_weight)
        self.pseudo_label_contrast_margin = float(pseudo_label_contrast_margin)
        self.pseudo_label_threshold = float(pseudo_label_threshold)
        self.pseudo_label_threshold_start = (
            None
            if pseudo_label_threshold_start is None
            else float(pseudo_label_threshold_start)
        )
        self.pseudo_label_start_iter = int(pseudo_label_start_iter)
        self.pseudo_label_min_pixels = int(pseudo_label_min_pixels)
        self.pseudo_label_teacher_period = int(pseudo_label_teacher_period)
        self.tversky_weight = float(tversky_weight)
        self.tversky_alpha = float(tversky_alpha)
        self.tversky_beta = float(tversky_beta)

    def _pseudo_label_threshold(self, it: int) -> float:
        """Confidence threshold ``τ`` at iteration ``it`` -- constant at
        :attr:`pseudo_label_threshold`, or linearly interpolated from
        :attr:`pseudo_label_threshold_start` when set."""
        if self.pseudo_label_threshold_start is None:
            return self.pseudo_label_threshold
        frac = it / max(self.adapt_iter - 1, 1)
        return self.pseudo_label_threshold_start + (
            self.pseudo_label_threshold - self.pseudo_label_threshold_start
        ) * frac

    def _pseudo_label_terms(
        self,
        feat: Tensor,
        q: Tensor,
        base_pred: Tensor,
        it: int,
        q_teacher: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        """Confidence-gated pseudo-label CE and kept/novel contrast terms
        on query pixels, per map (see class doc); both ``0`` when disabled
        or before :attr:`pseudo_label_start_iter`.

        ``q`` is the differentiable split used by the CE term (the
        student); ``q_teacher`` (defaults to ``q`` itself) is the
        ``torch.no_grad()`` split whose entropy/arg-max decide which
        pixels are confident and what their pseudo-label is -- a
        periodically-refreshed snapshot of ``q`` when
        :attr:`pseudo_label_teacher_period` ``> 0`` (see
        :meth:`adapt_to_query`), or ``q`` itself (the degenerate,
        un-stabilised case) otherwise."""
        h = self.h
        if q_teacher is None:
            q_teacher = q
        n = feat.shape[0]
        zero = feat.new_zeros(n)
        use_ce = self.pseudo_label_weight != 0.0
        use_contrast = self.pseudo_label_contrast_weight != 0.0
        if (not use_ce and not use_contrast) or it < self.pseudo_label_start_iter:
            return zero, zero
        tau = self._pseudo_label_threshold(it)
        ce_term = zero.clone()
        contrast_term = zero.clone()
        qfeat = h._qfeat(feat) if use_contrast else None
        for mother, idx in h._groups():
            here = base_pred == mother
            qm = q[:, idx]
            qm_teacher = q_teacher[:, idx]
            with torch.no_grad():
                ent = normalized_entropy(qm_teacher)
                confident = here & (ent < tau)
                pseudo = qm_teacher.argmax(1)
            kept_local = idx.index(h.kept_row[mother])
            if use_ce:
                p_pseudo = qm.gather(1, pseudo.unsqueeze(1)).squeeze(1)
                ce_px = -torch.log(p_pseudo + _EPS)
                wgt = confident.to(qm.dtype) * (1.0 - ent)
                ce_term = ce_term + (ce_px * wgt).flatten(1).sum(1) / wgt.flatten(
                    1
                ).sum(1).clamp(min=_EPS)
            if use_contrast:
                kept_mask = confident & (pseudo == kept_local)
                novel_mask = confident & (pseudo != kept_local)
                n_kept = kept_mask.flatten(1).sum(1)
                n_novel = novel_mask.flatten(1).sum(1)
                flat = qfeat.flatten(2)
                p_kept = (flat * kept_mask.flatten(1).unsqueeze(1)).sum(
                    -1
                ) / n_kept.clamp(min=1).unsqueeze(1)
                p_novel = (flat * novel_mask.flatten(1).unsqueeze(1)).sum(
                    -1
                ) / n_novel.clamp(min=1).unsqueeze(1)
                cos = F.cosine_similarity(p_kept, p_novel, dim=1)
                hinge = (cos - self.pseudo_label_contrast_margin).clamp(min=0.0)
                active = (n_kept >= self.pseudo_label_min_pixels) & (
                    n_novel >= self.pseudo_label_min_pixels
                )
                contrast_term = contrast_term + torch.where(
                    active, hinge, torch.zeros_like(hinge)
                )
        return ce_term, contrast_term

    def adapt_to_query(
        self, support_features: Tensor, support_masks: Tensor, query_features: Tensor
    ) -> None:
        """Per-map SGD on support CE + superclass entropy/prior (+ the
        confidence-gated pseudo-label terms, Tversky and BCM-as-teacher
        sub-modifiers above). Sets ``HiSplit._task_params``."""
        h = self.h
        w_ce, w_ent, w_kl = self.trans_weights
        n = query_features.shape[0]
        weight = h.q_weight.detach().unsqueeze(0).repeat(n, 1, 1).requires_grad_()
        bias = h.q_bias.detach().unsqueeze(0).repeat(n, 1).requires_grad_()
        optimizer = torch.optim.SGD([weight, bias], lr=self.lr)
        targets = h._targets(support_features, support_masks)
        p_base = h._base_probs(query_features)
        base_pred = p_base.argmax(1)
        with torch.no_grad():
            pi = h._prior(self.normalize(h._scores(query_features, weight, bias)), p_base)
        teacher_params: Optional[tuple] = None
        for it in range(self.adapt_iter):
            if self.pseudo_label_teacher_period > 0 and (
                it % self.pseudo_label_teacher_period == 0
            ):
                teacher_params = (weight.detach().clone(), bias.detach().clone())
            ce = torch.stack(
                [
                    h._support_ce(support_features, targets, weight[i], bias[i])
                    for i in range(n)
                ]
            )
            q = self.normalize(h._scores(query_features, weight, bias))
            ent, kl = h._superclass_terms(q, p_base, pi)
            if teacher_params is None:
                q_teacher = q
            else:
                with torch.no_grad():
                    q_teacher = self.normalize(h._scores(query_features, *teacher_params))
            pseudo_ce, pseudo_contrast = self._pseudo_label_terms(
                query_features, q, base_pred, it, q_teacher=q_teacher
            )
            if self.tversky_weight != 0.0:
                tversky = torch.stack(
                    [
                        h._support_tversky(
                            support_features,
                            targets,
                            weight[i],
                            bias[i],
                            self.tversky_alpha,
                            self.tversky_beta,
                        )
                        for i in range(n)
                    ]
                )
            else:
                tversky = ce.new_zeros(n)
            bcm_teacher_ce = h._bcm_teacher_term(query_features, q)
            loss = (
                w_ce * ce
                + w_ent * ent
                + w_kl * kl
                + self.pseudo_label_weight * pseudo_ce
                + self.pseudo_label_contrast_weight * pseudo_contrast
                + self.tversky_weight * tversky
                + h.bcm_teacher_weight * bcm_teacher_ce
            )
            optimizer.zero_grad()
            loss.sum().backward()
            optimizer.step()
            if (it + 1) in self.pi_update_at:
                with torch.no_grad():
                    pi = h._prior(self.normalize(h._scores(query_features, weight, bias)), p_base)
        h._task_params = (weight.detach(), bias.detach())


class EDLStrategy(LinearStrategy):
    """Evidential (Dirichlet) head: ``e_c = softplus(s_c) ≥ 0``,
    ``α_c = e_c + W·a`` (``W·a`` = ``edl_prior``, uniform ``1`` unless
    ``edl_base_rate: inverse_frequency``), ``q(c|x) = α_c/S`` (§2.6-2.7).

    Args:
        kl_anneal_steps: Steps over which the EDL KL regulariser's weight
            is annealed from 0 to 1 (``HiSplit._edl_loss``).
        edl_base_rate: ``"uniform"`` (``W·a = 1``, default) or
            ``"inverse_frequency"`` (favours the rarer child in the
            support, §2.7).
        base_rate_tau: Inverse-frequency exponent (``a_c ∝ (n_c+s)^-τ``).
        base_rate_smoothing: Additive smoothing ``s`` in the formula above.
    """

    name = "edl"

    def __init__(
        self,
        kl_anneal_steps: int = 50,
        edl_base_rate: str = "uniform",
        base_rate_tau: float = 1.0,
        base_rate_smoothing: float = 1.0,
    ) -> None:
        super().__init__()
        if edl_base_rate not in {"uniform", "inverse_frequency"}:
            raise ValueError(
                f"edl_base_rate must be 'uniform' or 'inverse_frequency', "
                f"got {edl_base_rate!r}."
            )
        self.kl_anneal_steps = int(kl_anneal_steps)
        self.edl_base_rate = edl_base_rate
        self.base_rate_tau = float(base_rate_tau)
        self.base_rate_smoothing = float(base_rate_smoothing)

    def normalize(self, scores: Tensor) -> Tensor:
        alpha = self.h._edl_alpha(scores)
        q = torch.empty_like(scores)
        for _, idx in self.h._groups():
            q[:, idx] = alpha[:, idx] / alpha[:, idx].sum(1, keepdim=True)
        return q

    def init_from_support(self, features: Tensor, targets: Tensor) -> None:
        if self.edl_base_rate != "inverse_frequency":
            return
        h = self.h
        t_all = targets.reshape(-1)
        counts = torch.stack(
            [(t_all == i).sum() for i in range(len(h.child_classes))]
        ).to(h.edl_prior)
        prior = torch.empty_like(h.edl_prior)
        for _, idx in h._groups():
            a = (counts[idx] + self.base_rate_smoothing).pow(-self.base_rate_tau)
            prior[idx] = len(idx) * a / a.sum()
        h.edl_prior.copy_(prior)

    def training_loss(
        self, features: Tensor, targets: Tensor, sample_weight: Optional[Tensor]
    ) -> Optional[Tensor]:
        loss = self.h._edl_loss(features, targets, sample_weight, self.kl_anneal_steps)
        self.h._edl_step += 1
        return loss


class LogregStrategy(QStrategy):
    """BCM's classifier (``LogisticRegressionCV``, per mother) inside
    HiSplit's factorization instead of a gradient-trained head (§2.8) --
    ``scores`` are log-probabilities from the fitted ``sklearn`` models.
    No extra config of its own: the CV hyperparameters
    (``logreg_n_splits``/``n_C``/``sampling``/``n_jobs``) stay on
    ``HiSplit``, shared with the ``bcm_teacher_weight`` modifier (see
    class doc of :class:`QStrategy`). The fit itself is triggered by
    :attr:`needs_logreg_fit` from ``HiSplit.init_from_support``."""

    name = "logreg"
    needs_logreg_fit = True

    def scores(self, features: Tensor, weight: Tensor, bias: Tensor) -> Tensor:
        return self.h._logreg_scores(features)


class LinearAefGateStrategy(LinearStrategy):
    """AlphaEarth/OlmoEarth multiplicative gate on the frozen features
    before a linear head, both self-fit together (§8.14): ``features' =
    features · sigmoid(Linear(aef))``, ``q(c|x) = softmax_c(w_c·features'
    + b_c)`` -- the gate itself (``aef_gate_linear``, sized by ``HiSplit.
    aef_dim``) is shared infrastructure built by ``HiSplit.setup``
    whenever :attr:`uses_aef_gate` or the ``aef_gate`` flag is on (see
    class doc of :class:`QStrategy`); this strategy only triggers that
    and owns the self-fit's own Adam hyperparameters.

    Args:
        aef_gate_epochs: Adam epochs of the self-contained gate+head fit.
        aef_gate_lr: Adam learning rate of that fit.
    """

    name = "linear_aef_gate"
    #: not :attr:`QStrategy.trainable` in the generic ``pl_trainer`` sense
    #: -- :meth:`init_from_support` self-fits gate+head together instead
    #: (same self-fit pattern as :class:`LogregStrategy`), so ``pl_trainer.
    #: max_steps`` stays ``0``.
    trainable = False
    uses_aef_gate = True

    def __init__(self, aef_gate_epochs: int = 200, aef_gate_lr: float = 1e-2) -> None:
        super().__init__()
        self.aef_gate_epochs = int(aef_gate_epochs)
        self.aef_gate_lr = float(aef_gate_lr)

    def init_from_support(self, features: Tensor, targets: Tensor) -> None:
        self.h._fit_aef_gate(features, targets, self.aef_gate_epochs, self.aef_gate_lr)


#: Picked once by ``HiSplit.__init__`` when ``q`` is given as a plain
#: string (``HiSplit(q="trans")`` == ``HiSplit(q=TransStrategy())``, all
#: defaults) -- pass an already-constructed strategy instead for
#: non-default parameters (``HiSplit(q=TransStrategy(adapt_iter=10))``).
_Q_STRATEGIES = {
    cls.name: cls
    for cls in (
        ProtoStrategy,
        ProtoProbStrategy,
        LinearStrategy,
        TransStrategy,
        EDLStrategy,
        LogregStrategy,
        LinearAefGateStrategy,
    )
}
