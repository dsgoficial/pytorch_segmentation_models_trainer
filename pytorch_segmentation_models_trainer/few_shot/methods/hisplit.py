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
import warnings
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
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
from pytorch_segmentation_models_trainer.few_shot.methods.q_strategies import (
    QStrategy,
    _Q_STRATEGIES,
)
from pytorch_segmentation_models_trainer.few_shot.uncertainty import (
    dissonance,
    normalized_entropy,
    vacuity,
)

logger = logging.getLogger(__name__)

_EPS = 1e-10
_Q_TYPES = set(_Q_STRATEGIES)


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

    * ``logreg``: the classifier of BCM (cross-validated logistic regression,
      class balancing; ``sklearn``) fitted on the HiSplit support targets of
      each mother; ``q`` = its probabilities (no gradient training).
      ``logreg_n_jobs`` parallelizes its CV grid search (see
      ``BCM``/``_LogisticRegressionCV``); 1 (default) is sequential.

    Comparison options with BCM / H²EDL (defaults keep the original HiSplit):
    ``feature_power`` (Tukey's ladder of powers on the features used by
    ``q``, as BCM; negative values are clipped to 0 when ≠ 1), ``negatives:
    superclass | all`` ("not novel" support pixels become examples of the
    mother's kept child only where the base model predicts the mother — the
    default — or everywhere, as BCM's g_β; ``all`` needs a single mother) and,
    for ``edl``, ``edl_base_rate: inverse_frequency`` (H²EDL's tempered
    inverse-frequency base rate ``a_c ∝ (n_c + s)^-τ`` from the support
    targets, ``α = e + K·a``; the KL regulariser keeps the uniform target).

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

    Boundary of the superclass (P2) — the split alone cannot recover novel
    pixels the base model assigned to another class:

    * ``leak: true`` (hierarchical leak, "HierTrans"): learned coefficients
      ``t[n, c] = sigmoid(θ)`` move mass from every base class ``c`` other than
      the mother into each novel child ``n``:
      ``p'(n) = p_m q_n + Σ_c t[n,c] p_c`` and ``p'(c) = p_c (1 − Σ_n t[n,c])``
      (normalisation preserved). ``t`` is trained on the support (in
      ``trainer.fit``) with the likelihood of the final distribution: novel
      labels ``−log p'(n)``, "not novel" labels ``−log(1 − Σ_n p'(n))``, base
      labels ``−log p'(c)``. Hierarchical decoding first decides the
      superclass (which absorbs the leaked mass), then the split; a pixel
      outside the superclass either keeps the base class or becomes a child
      (uneven leak fractions never swap neighbours).
    * ``widen: prob | dissonance`` (superclass widened by a threshold):
      where the mother is the base model's **second** choice and
      ``p''(m) ≥ widen_threshold`` (``prob``) or the base dissonance is
      ``≥ widen_threshold`` (``dissonance``, evidential base only), the pixel
      becomes the novel child **if q prefers a novel child**; otherwise the
      base decision is kept [adaptation: the proposal says only "q decides";
      restricting the switch to the novel class avoids turning neighbours
      into the kept child].
    * ``novel_prior_weight``: multiplies the novel shares of ``q`` before
      renormalisation (logit/prior adjustment, ``w > 1`` favours novel).
    * ``sweep``: thresholds evaluated in the same pass as decoding variants
      ``widen_<thr>`` (``variant_names``/``decode_variants``), giving the
      preservation × correction curve.

    Uncertainty-gated abstention: ``abstain_measure`` (one of the keys of
    ``uncertainty()``, e.g. ``split_entropy``, ``split_vacuity``,
    ``dissonance``) makes the decoding call :meth:`abstention` and, on every
    pixel it flags (base model predicts a mother whose split uncertainty
    exceeds ``abstain_threshold``), forces the novel children's logits to
    ``-inf`` — the decision falls back to the mother's own kept child, as
    ``abstention`` already documents. Independent of ``widen``/``sweep`` and
    applied after them; ``None`` (default) disables it.

    Confidence-weighted split: ``confidence_weight_measure`` (a key of
    ``uncertainty()``) replaces ``abstain``'s hard cutoff with a continuous
    blend, for every pixel of every mother (not just where ``abstain``'s
    threshold happens to fire): ``q' = w·q + (1−w)·e_kept``, ``w = 1 −
    measure(x) ∈ [0, 1]`` and ``e_kept`` the one-hot of the mother's own
    kept child (same semantics as ``abstention``'s fallback, but smooth
    instead of a cliff at a threshold). Applied to every consumer of ``q``
    (:meth:`split_probabilities`, hierarchical decoding), so it works with
    ``decoding: flat`` too, unlike ``widen``/``abstain_measure``. Composes
    with ``abstain_measure``/``widen`` (applied before them); ``None``
    (default) disables it.

    Evidence-weighted training: ``support_loss_weight_measure`` (a key of
    ``uncertainty()``) reweights the per-pixel training loss of ``q``
    (``linear``/``trans``'s CE in :meth:`_support_ce`, ``edl``'s MSE+KL in
    :meth:`_edl_loss`) by ``1 − measure(x) ∈ [0, 1]``, computed once per
    ``support_loss`` call on the same support features already used for the
    loss — pixels where the base model's own evidence is weak (high
    ``measure``, e.g. ``base_vacuity``/``dissonance`` on an evidential base,
    or ``split_vacuity`` with ``q: edl``) count less towards training ``q``.
    ``None`` (default) disables it (every pixel weighted equally, the
    original behaviour).

    Confidence-gated pseudo-labelling (``trans`` only): on top of the usual
    support CE + superclass entropy/KL of :meth:`adapt_to_query`, every SGD
    iteration can also fit ``q`` to its own most confident predictions on
    the **query** itself — the population ``support_loss_weight_measure``
    could not help with, being just the ``K`` support pixels. For each
    mother, pixels in the superclass (``here``) whose split entropy is
    below a threshold ``τ`` are treated as confidently decided; their
    (stop-gradient) arg-max becomes a pseudo-label for two terms, both
    ``0.0`` (disabled) by default:

    * ``pseudo_label_weight``: cross-entropy of ``q`` against that
      pseudo-label, weighted by ``1 − entropy`` and averaged over the
      confident pixels of each map.
    * ``pseudo_label_contrast_weight``: hinge repulsion
      ``relu(cos(P_kept, P_novel) − margin)`` between the mean ``q``-space
      features of the confident pixels pseudo-labelled as the mother's
      kept child vs. as any novel child (skipped, per map, when either
      side has fewer than ``pseudo_label_min_pixels`` confident pixels) —
      the same hinge :class:`RefinementAwareWrapper`'s sub-prototypes use,
      but with the two sides chosen by the pseudo-label instead of an
      unsupervised cluster assignment.

    ``τ`` is ``pseudo_label_threshold`` by default, or linearly
    interpolated from ``pseudo_label_threshold_start`` up to
    ``pseudo_label_threshold`` over ``adapt_iter`` when the former is set
    (a confidence curriculum: only the most unambiguous pixels count
    early on, more are trusted as ``q`` improves). Neither term is applied
    before ``pseudo_label_start_iter``, to let plain support CE calibrate
    ``q`` first.

    Tversky support term (``trans`` only): ``_support_ce`` above averages
    cross-entropy uniformly over support pixels, which on a tiny, possibly
    imbalanced ``K``-shot support (few ``novel`` pixels against many
    ``kept`` ones) lets the easier/majority child dominate the per-episode
    SGD gradient. ``tversky_weight`` (``0.0``, disabled, by default) adds
    ``1 − mean_c Tversky_c`` (over children actually present in that
    episode's support, per :class:`RefinementAwareWrapper`-style soft
    TP/FP/FN, computed from the same ``q`` and support targets as
    ``_support_ce``) to the ``adapt_to_query`` loss, weighted equally
    across children instead of pixels. ``tversky_alpha``/``tversky_beta``
    (``0.5``/``0.5``, i.e. the Dice coefficient, by default) weight false
    positives vs. false negatives; raising ``tversky_beta`` above
    ``tversky_alpha`` penalises missing a child (false negative) more than
    over-predicting it, which can be used to push confidence specifically
    onto an under-predicted ``novel`` child.

    Uncertainty-gated hard decoding (any ``q``): BCM overrides the base
    model's prediction with a hard arg-max inside the mapped base class,
    instead of HiSplit's usual soft multiplicative factorization
    (``p(filha) = p_base(mãe)·q``); on the Mata Atlântica biome BCM beats
    every HiSplit variant tested (including ``hisplit_logreg``, the same
    classifier as BCM inside HiSplit's soft decoding — see
    ``pesquisa/analise_comparativa_hisplit_estado_da_arte_2026-10-16.md``
    §8.3), so the hard-vs-soft decision itself, not just the classifier,
    is a candidate explanation. ``hard_decode_measure`` (``None``,
    disabled, by default) snaps ``q`` to a one-hot at its own arg-max,
    per mother, wherever the named uncertainty measure (a key of
    :meth:`uncertainty`) is below ``hard_decode_threshold`` (confident) —
    a BCM-style hard override gated by uncertainty instead of applied
    unconditionally, applied in :meth:`split_probabilities` and
    :meth:`_decode` right after ``confidence_weight_measure`` (identity
    when both are ``None``). Uncertain pixels keep the usual soft ``q``
    unchanged.

    BCM-as-teacher distillation (``trans`` only): fits a
    ``LogisticRegressionCV`` teacher on the same ``K``-shot support as
    ``hisplit_logreg``/BCM (reusing :meth:`_fit_logreg`, regardless of
    ``q``) and, when ``bcm_teacher_weight`` (``0.0``, disabled, by
    default) is set, adds a cross-entropy of ``q`` against that
    teacher's (stop-gradient) arg-max on the query, weighted by the
    teacher's own confidence (``1 − entropy``) — distils BCM's decision
    into ``q`` instead of adopting BCM's hard-override mechanism
    wholesale (``hard_decode_measure`` above does that instead).

    AlphaEarth Foundations (AEF) multiplicative gate (``q: linear_aef_gate``
    only): modulates the frozen base features before the linear head —
    ``features' = features · sigmoid(Linear(aef))`` — with a 64-d external
    embedding (AlphaEarth Foundations, 10 m, phenology-aware) at the base's
    own native 64×64 grid (no resample needed for a 2.5 m/256 px tile).
    Motivated by a 4-stage separability probe (not inside this repo, see
    the companion research project's
    ``pesquisa/analise_comparativa_hisplit_estado_da_arte_2026-10-16.md``
    §8.14, Estágios 1-4) that measured this exact gate as the strongest
    and most consistent non-linear interaction between AEF and the base's
    features for a campo×cultivada-style split, confirmed in 2 biomes —
    plain concatenation (linear) and a generic MLP on the concatenation
    were weaker. The gate's ``nn.Linear`` and the head (``q_weight``/
    ``q_bias``) are fit together by :meth:`_fit_aef_gate`, a self-contained
    Adam loop inside :meth:`init_from_support` (like :meth:`_fit_logreg`'s
    sklearn fit) — not ``pl_trainer``'s generic training loop, so
    ``max_steps`` stays ``0`` as for ``logreg``/``proto``. The external
    embedding itself is loaded and provided by ``GFSSModel`` (via
    :meth:`set_aux`, re-aligned under every TTA pass) from
    ``gfss.aef_cache_dir`` — this method only consumes whatever tensor it
    is given, it does not know the cache format. ``K=1`` was not reliable
    for ANY mechanism tested in the probe (single-image support) — treat
    it with low confidence here too.

    Uncertainty-augmented AEF gate (``aef_gate_uncertainty: true``, combined
    with ``q: linear_aef_gate`` and an evidential base, ``base_output:
    evidential``/R2-EDL): concatenates the frozen base model's own
    per-pixel ``base_vacuity`` (:meth:`_base_vacuity`, the same quantity
    :meth:`uncertainty` reports under that name) as an extra channel to the
    AEF embedding before the gate's ``nn.Linear`` — ``features' = features
    · sigmoid(Linear([aef; base_vacuity]))``, input dim ``aef_dim + 1``
    instead of ``aef_dim``. ``base_vacuity`` is the only uncertainty
    measure usable here without a circular dependency: unlike
    ``split_vacuity``/``dissonance`` (which need ``q``, i.e. :meth:`_qfeat`
    itself, to compute), it is a function of the frozen base model's own
    Dirichlet output alone (:meth:`_base_vacuity`), so it is safe to read
    from inside :meth:`_qfeat`. Motivated by the user's question, in the
    companion research project's session, of whether the EDL checkpoint's
    effect on the gate (already shown to be biome-asymmetric when swapping
    the frozen checkpoint wholesale, §8.14 "Gate + checkpoint R2-EDL")
    could instead be an explicit, learned signal the gate itself consumes,
    rather than only an indirect effect of which checkpoint feeds it.
    Requires ``base_output: evidential`` (``setup`` raises ``ValueError``
    otherwise, like ``widen: dissonance``).

    Post-gate uncertainty blend (``aef_gate_uncertainty_blend: true``,
    combined with ``q: linear_aef_gate`` and an evidential base — an
    alternative to ``aef_gate_uncertainty`` above, not its composition):
    instead of giving the base model's ``base_vacuity`` to the gate's
    ``nn.Linear`` as an extra *input* (where the Adam fit is free to learn
    around/ignore it — exactly what happened empirically: in the companion
    research project, that channel's weight made no measurable difference,
    see §8.14 "gate+R2-EDL+incerteza (concat)"), this blends the gate's
    *output* deterministically back towards ``1`` (no modulation, raw
    features) wherever the base model itself is uncertain:
    ``gate' = w · gate + (1 − w)``, ``w = 1 − base_vacuity`` — reusing
    :meth:`_apply_confidence_weight`'s exact blend semantics
    (``w · q + (1 − w) · kept``), applied to the gate's multiplicative
    factor instead of to ``q``. No new learned parameters (unlike the
    concat variant, ``aef_gate_linear``'s input dim stays ``aef_dim``) —
    the rule is fixed, not something the fit can route around. Also
    requires ``base_output: evidential``.

    AEF gate composed with any trainable ``q`` (``aef_gate: true``,
    independent of ``q: linear_aef_gate``): the gate was originally its
    own ``q`` value, mutually exclusive with every other one — in
    particular with ``q: edl`` (the evidential pair head), whose own
    small, real, checkpoint-independent gain (companion research
    project, §8.4) had therefore never been combined with the gate's.
    Setting ``aef_gate: true`` applies the exact same multiplicative gate
    (:meth:`_qfeat`, unchanged) regardless of ``q``, as long as ``q`` is
    ``linear``, ``trans`` or ``edl`` (any :attr:`QStrategy.trainable`
    strategy, see ``q_strategies.py``) — the gate's
    ``aef_gate_linear`` is a real ``nn.Linear`` submodule, so it is
    automatically included in whatever the caller's generic training
    loop already optimises for that ``q`` (``GFSSModel``'s
    ``_trainable_parameters``), gradients flowing through
    :meth:`_scores` exactly like ``q_weight``/``q_bias`` do — **not**
    :meth:`_fit_aef_gate`'s self-contained Adam loop, which stays scoped
    to ``q == "linear_aef_gate"`` only (``pl_trainer.max_steps`` must
    therefore be ``> 0`` here, as it already is for any trainable ``q``,
    instead of staying ``0`` as ``linear_aef_gate``'s self-fit allows).
    ``aef_gate: true`` with a non-trainable, non-``linear_aef_gate`` ``q``
    (``proto``, ``proto_prob``, ``logreg``) raises ``ValueError`` in
    :meth:`setup` — the gate would never be fit, silently staying at its
    random initialisation.

    Teacher-student stabilisation: by default the pseudo-label/confidence
    decision above is read off the same ``q`` being optimised this very
    SGD step (student == teacher, degenerate). When
    ``pseudo_label_teacher_period > 0``, :meth:`adapt_to_query` instead
    snapshots ``(weight, bias)`` every that many iterations into a frozen
    "teacher" and uses *its* ``q`` (``torch.no_grad()``) for the
    confidence/pseudo-label decision, while the live "student" ``q`` is
    still what the CE/contrast terms optimise towards that decision —
    avoids chasing an instantaneous, noisier target (the standard
    Mean-Teacher idea, but a periodically-refreshed snapshot instead of a
    continuous EMA, to not couple teacher and student the way a
    low-decay EMA of the very same weights would). ``0`` (default)
    disables it.

    Each ``q`` mechanism (``proto``, ``proto_prob``, ``linear``, ``trans``,
    ``edl``, ``logreg``, ``linear_aef_gate``) is a :class:`QStrategy`
    subclass in ``q_strategies.py``, owning its own lean, mechanism-specific
    parameters (e.g. ``trans``'s ``adapt_iter``/``lr``/``pseudo_label_*``/
    ``tversky_*``, ``edl``'s ``kl_anneal_steps``, ``proto_prob``'s
    ``variance``/``var_floor``, ``linear_aef_gate``'s ``aef_gate_epochs``/
    ``aef_gate_lr`` — see each subclass's own docstring for its full
    parameter list). ``Args`` below are only the params that stay on
    ``HiSplit`` itself: either cross-cutting (apply regardless of ``q``) or
    shared infrastructure more than one mechanism can trigger (e.g.
    ``logreg_n_*``/:meth:`_fit_logreg`, used by ``q: logreg`` **and** by
    ``bcm_teacher_weight`` with any other ``q``).

    Args:
        q: ``"proto"`` / ``"proto_prob"`` / ``"linear"`` / ``"trans"`` /
            ``"edl"`` / ``"logreg"`` / ``"linear_aef_gate"`` picks that
            :class:`QStrategy` subclass with its own defaults; a
            :class:`QStrategy` instance (e.g. ``TransStrategy(adapt_iter=10)``)
            or a Hydra sub-config (``{"_target_": "...TransStrategy",
            "adapt_iter": 10}``) overrides its mechanism-specific params —
            see class doc and ``q_strategies.py``.
        tau: Cosine temperature of ``proto`` (and init of ``linear``/``trans``).
        decoding: ``hierarchical`` or ``flat``.
        decode_eps: Tie-break scale of the hierarchical decoding.
        feature_power: Tukey's ladder of powers applied to the features used
            by ``q`` (as BCM); negative values are clipped to 0 when ``!= 1``.
        negatives: ``superclass`` (default) or ``all`` — where "not novel"
            support pixels become examples of the mother's kept child (see
            class doc; ``all`` needs a single mother).
        logreg_n_splits: CV folds of the ``LogisticRegressionCV`` fitted by
            :meth:`_fit_logreg` (``q: logreg``, or any ``q`` with
            ``bcm_teacher_weight != 0``).
        logreg_n_C: Grid size of ``LogisticRegressionCV``'s ``C`` search.
        logreg_sampling: Class-balancing strategy of ``_fit_logreg``
            (see ``BCM``/``_LogisticRegressionCV``).
        logreg_n_jobs: Parallelism of the CV grid search; ``1`` (default)
            is sequential.
        seed: RNG seed for ``_fit_logreg``'s sampling.
        leak: Learn the hierarchical leak into novel children.
        leak_init: Initial ``θ`` of the leak (``sigmoid(-4) ≈ 0.018``).
        widen: ``none``, ``prob`` or ``dissonance``.
        widen_threshold: Threshold of ``widen``.
        novel_prior_weight: Prior weight of novel children in ``q``.
        sweep: Thresholds evaluated as extra decoding variants (needs
            ``widen`` other than ``none``).
        abstain_measure: Uncertainty measure gating abstention (a key of
            ``uncertainty()``), or ``None`` to disable it.
        abstain_threshold: Threshold of ``abstain_measure``.
        confidence_weight_measure: Uncertainty measure continuously
            blending ``q`` towards the mother's kept child (a key of
            ``uncertainty()``), or ``None`` to disable it.
        confidence_weight_power: Exponent applied to the measure before
            blending (``measure ** power``), e.g. ``2.0`` compresses it
            towards 0 for measures whose typical range is not already
            near an endpoint. ``1.0`` (default) leaves it unchanged.
        confidence_weight_scale: Multiplicative factor applied after the
            exponent, clamped back to ``[0, 1]`` before computing
            ``w = 1 − measure``. ``1.0`` (default) leaves it unchanged.
        support_loss_weight_measure: Uncertainty measure reweighting ``q``'s
            training loss by ``1 − measure`` (a key of ``uncertainty()``),
            or ``None`` to disable it (default; every pixel weighted
            equally).
        hard_decode_measure: Uncertainty measure gating the BCM-style
            hard override (a key of ``uncertainty()``, see class doc),
            or ``None`` to disable it (default).
        hard_decode_threshold: Threshold of ``hard_decode_measure``.
        bcm_teacher_weight: Weight of the cross-entropy of ``q`` against
            a BCM-style ``LogisticRegressionCV`` teacher on the query
            (``trans`` only, see class doc). ``0.0`` (default) disables
            it.
        aef_dim: Channel count of the external embedding consumed by
            ``q: linear_aef_gate`` (``64`` for AlphaEarth Foundations,
            the only one tested).
        aef_gate_uncertainty: Concatenate the base model's ``base_vacuity``
            to the AEF embedding before the gate (see class doc,
            ``linear_aef_gate`` only). Requires ``base_output: evidential``.
        aef_gate_uncertainty_blend: Deterministically blend the gate's
            output towards ``1`` (no modulation) by ``base_vacuity`` (see
            class doc, ``linear_aef_gate`` only; an alternative to
            ``aef_gate_uncertainty``, not its composition). Requires
            ``base_output: evidential``.
        aef_gate: Apply the AEF multiplicative gate together with any
            trainable ``q`` (``linear``, ``trans`` or ``edl`` — see class
            doc), instead of only through ``q: linear_aef_gate``. Trained
            by the caller's generic loop (``pl_trainer.max_steps > 0``),
            not :meth:`_fit_aef_gate`'s self-fit. ``False`` (default) —
            no behaviour change for existing configs.

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
        q: Union[str, QStrategy, Dict[str, Any]] = "proto",
        tau: float = 10.0,
        decoding: str = "hierarchical",
        decode_eps: float = 1e-4,
        feature_power: float = 1.0,
        negatives: str = "superclass",
        logreg_n_splits: int = 5,
        logreg_n_C: int = 10,
        logreg_sampling: str = "us",
        logreg_n_jobs: int = 1,
        seed: int = 0,
        leak: bool = False,
        leak_init: float = -4.0,
        widen: str = "none",
        widen_threshold: float = 0.3,
        novel_prior_weight: float = 1.0,
        sweep: Sequence[float] = (),
        abstain_measure: Optional[str] = None,
        abstain_threshold: float = 0.3,
        confidence_weight_measure: Optional[str] = None,
        confidence_weight_power: float = 1.0,
        confidence_weight_scale: float = 1.0,
        support_loss_weight_measure: Optional[str] = None,
        hard_decode_measure: Optional[str] = None,
        hard_decode_threshold: float = 0.3,
        bcm_teacher_weight: float = 0.0,
        aef_dim: int = 64,
        aef_gate_uncertainty: bool = False,
        aef_gate_uncertainty_blend: bool = False,
        aef_gate: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        if decoding not in {"hierarchical", "flat"}:
            raise ValueError(
                f"decoding must be 'hierarchical' or 'flat', got {decoding!r}."
            )
        if widen not in {"none", "prob", "dissonance"}:
            raise ValueError(
                f"widen must be 'none', 'prob' or 'dissonance', got {widen!r}."
            )
        if (widen != "none" or sweep or abstain_measure is not None) and (
            decoding != "hierarchical"
        ):
            raise ValueError("widen/sweep/abstain_measure require decoding: hierarchical.")
        if sweep and widen == "none":
            raise ValueError("sweep requires widen: prob or dissonance.")
        self.leak = bool(leak)
        self.leak_init = float(leak_init)
        self.widen = widen
        self.widen_threshold = float(widen_threshold)
        self.novel_prior_weight = float(novel_prior_weight)
        self.sweep = [float(t) for t in sweep]
        self.abstain_measure = abstain_measure
        self.abstain_threshold = float(abstain_threshold)
        self.confidence_weight_measure = confidence_weight_measure
        self.confidence_weight_power = float(confidence_weight_power)
        self.confidence_weight_scale = float(confidence_weight_scale)
        self.support_loss_weight_measure = support_loss_weight_measure
        self.hard_decode_measure = hard_decode_measure
        self.hard_decode_threshold = float(hard_decode_threshold)
        self.bcm_teacher_weight = float(bcm_teacher_weight)
        self.aef_dim = int(aef_dim)
        self.aef_gate_uncertainty = bool(aef_gate_uncertainty)
        self.aef_gate_uncertainty_blend = bool(aef_gate_uncertainty_blend)
        self.aef_gate = bool(aef_gate)
        self._aux: Optional[Tensor] = None
        self.tau = float(tau)
        self.decoding = decoding
        self.decode_eps = float(decode_eps)
        if negatives not in {"superclass", "all"}:
            raise ValueError(
                f"negatives must be 'superclass' or 'all', got {negatives!r}."
            )
        self.feature_power = float(feature_power)
        self.negatives = negatives
        self.logreg_args = (int(logreg_n_splits), int(logreg_n_C), logreg_sampling)
        self.logreg_n_jobs = int(logreg_n_jobs)
        self.seed = int(seed)
        self.logreg_models: Dict[int, object] = {}
        self._q_strategy = self._resolve_q_strategy(q)
        self._q_strategy.bind(self)
        self.q = self._q_strategy.name
        self.transductive = self._q_strategy.transductive
        self._edl_step = 0
        self._task_params: Optional[tuple] = None

    @staticmethod
    def _resolve_q_strategy(q: Union[str, QStrategy, Dict[str, Any]]) -> QStrategy:
        """``q="trans"`` (every default) | an already-built
        ``QStrategy(...)`` (non-default parameters, plain Python) | a
        Hydra sub-config with its own ``_target_`` (YAML: ``HiSplit`` is
        instantiated with ``_recursive_=False`` by ``GFSSModel``, so a
        nested ``_target_`` block arrives un-instantiated and must be
        resolved by hand here)."""
        if isinstance(q, str):
            if q not in _Q_STRATEGIES:
                raise ValueError(f"q must be one of {sorted(_Q_TYPES)}, got {q!r}.")
            return _Q_STRATEGIES[q]()
        if isinstance(q, QStrategy):
            return q
        from hydra.utils import instantiate

        return instantiate(q, _recursive_=False)

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
        if self._q_strategy.trainable:
            self.q_weight = nn.Parameter(base_weight.new_zeros(n, dim))
            self.q_bias = nn.Parameter(base_weight.new_zeros(n))
        else:
            self.register_buffer("q_weight", base_weight.new_zeros(n, dim))
            self.register_buffer("q_bias", base_weight.new_zeros(n))
        self.register_buffer("q_var", base_weight.new_ones(n, dim))
        self.register_buffer("edl_prior", base_weight.new_ones(n))  # W·a (uniform: 1)
        if self.negatives == "all" and len(hierarchy.mothers) > 1:
            raise ValueError("negatives: all needs a single mother class.")
        if self.widen == "dissonance" and self.base_output != "evidential":
            raise ValueError("widen: dissonance requires an evidential base model.")
        if self.abstain_measure is not None and self.abstain_measure not in (
            available := self.uncertainty_names()
        ):
            raise ValueError(
                f"abstain_measure {self.abstain_measure!r} not available "
                f"for this configuration ({sorted(available)})."
            )
        if self.confidence_weight_measure is not None and (
            self.confidence_weight_measure not in self.uncertainty_names()
        ):
            raise ValueError(
                f"confidence_weight_measure {self.confidence_weight_measure!r} not "
                f"available for this configuration ({sorted(self.uncertainty_names())})."
            )
        if self.hard_decode_measure is not None and (
            self.hard_decode_measure not in self.uncertainty_names()
        ):
            raise ValueError(
                f"hard_decode_measure {self.hard_decode_measure!r} not "
                f"available for this configuration ({sorted(self.uncertainty_names())})."
            )
        if self.support_loss_weight_measure is not None and (
            self.support_loss_weight_measure not in self.uncertainty_names()
        ):
            raise ValueError(
                f"support_loss_weight_measure {self.support_loss_weight_measure!r} not "
                f"available for this configuration ({sorted(self.uncertainty_names())})."
            )
        nb = hierarchy.num_base_classes
        self.novel_rows = [i for i, c in enumerate(self.child_classes) if c >= nb]
        self.kept_row: Dict[int, int] = {
            m: i for i, (c, m) in enumerate(zip(self.child_classes, self.child_mother))
            if c == m
        }
        mask = base_weight.new_zeros(n, nb)
        for i in self.novel_rows:
            mask[i] = 1.0
            mask[i, self.child_mother[i]] = 0.0
        self.register_buffer("leak_mask", mask)
        if self.leak:
            self.leak_logit = nn.Parameter(
                base_weight.new_full((n, nb), self.leak_init)
            )
        if self._q_strategy.uses_aef_gate or self.aef_gate:
            if self.aef_gate and not (
                self._q_strategy.trainable or self._q_strategy.uses_aef_gate
            ):
                raise ValueError(
                    f"aef_gate=True needs q to be trainable (pl_trainer-driven) "
                    f"or 'linear_aef_gate' (self-fit covers it already); got "
                    f"q={self.q!r}, whose aef_gate_linear would never be fit."
                )
            self.aux_names = ("aef",)
            if (
                self.aef_gate_uncertainty or self.aef_gate_uncertainty_blend
            ) and self.base_output != "evidential":
                raise ValueError(
                    "aef_gate_uncertainty/aef_gate_uncertainty_blend require an "
                    "evidential base model (base_output: evidential)."
                )
            gate_in_dim = self.aef_dim + (1 if self.aef_gate_uncertainty else 0)
            self.aef_gate_linear = nn.Linear(gate_in_dim, dim).to(base_weight)

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
                not_novel = masks == self.not_novel_index
                if self.negatives == "superclass":
                    not_novel = not_novel & (base_pred == mother)
                targets[not_novel] = i
        return targets

    def set_aux(self, aux: Dict[str, Tensor]) -> None:
        """Stores the ``"aef"`` entry (e.g. AEF or OlmoEarth embeddings)
        of the call that follows (``init_from_support``/``support_loss``/
        one TTA pass of ``decode_to`` — see :attr:`aux_names`), ``(B,
        aef_dim, h, w)`` at the same spatial resolution/orientation as
        ``features``, or ``None`` when missing. Only consumed by the AEF
        gate (:meth:`_qfeat`, :attr:`aux_names` empty for every other
        configuration)."""
        self._aux = aux.get("aef") if aux else None

    def _base_vacuity(self, features: Tensor) -> Tensor:
        """Per-pixel vacuity ``(B, h, w)`` of the frozen base model's own
        Dirichlet output (``base_output: evidential`` only) — the same
        quantity :meth:`uncertainty` reports as ``base_vacuity``, but
        computed directly from ``features``/``base_weight``/``base_bias``
        without going through ``q``/:meth:`_scores`, so it is safe to call
        from inside :meth:`_qfeat` itself (unlike ``split_vacuity``/
        ``dissonance``, which need ``q`` and would recurse)."""
        return vacuity(self.base_alpha(self._base_logits(features)))

    def _qfeat(self, features: Tensor) -> Tensor:
        """Features used by q: the AEF multiplicative gate (``q:
        linear_aef_gate``, or any trainable ``q`` with ``aef_gate: true``
        — see class doc — when :meth:`set_aux` was last given a tensor;
        with ``aef_gate_uncertainty``, the AEF embedding is concatenated
        with :meth:`_base_vacuity` before the gate's linear layer; with
        ``aef_gate_uncertainty_blend``, the gate's output is instead
        deterministically blended towards ``1`` by :meth:`_base_vacuity`
        afterwards — see class doc, the two are alternatives, not
        composed), then Tukey's power transform when ``feature_power`` ≠ 1."""
        if (self._q_strategy.uses_aef_gate or self.aef_gate) and self._aux is not None:
            gate_in = self._aux
            if gate_in.shape[1] != self.aef_dim:
                raise ValueError(
                    f"HiSplit linear_aef_gate: aux has {gate_in.shape[1]} channels "
                    f"but aef_dim={self.aef_dim} -- the external-embedding loader "
                    f"(e.g. gfss.aef_cache_loader's n_bands) and the method's "
                    f"aef_dim are independent config values with nothing syncing "
                    f"them automatically; set both to the same width."
                )
            if self.aef_gate_uncertainty:
                unc = self._base_vacuity(features).unsqueeze(1).to(gate_in)
                gate_in = torch.cat([gate_in, unc], dim=1)
            gate = torch.sigmoid(
                self.aef_gate_linear(gate_in.movedim(1, -1))
            ).movedim(-1, 1)
            if self.aef_gate_uncertainty_blend:
                w = (1.0 - self._base_vacuity(features)).unsqueeze(1).to(gate)
                gate = w * gate + (1.0 - w)
            features = features * gate
        if self.feature_power == 1.0:
            return features
        return features.clamp(min=0).pow(self.feature_power)

    def init_from_support(self, features: Tensor, masks: Tensor) -> None:
        """Per-child prototypes (and variances) from the support targets."""
        targets = self._targets(features, masks)
        qf = self._qfeat(features)
        flat = qf.permute(0, 2, 3, 1).reshape(-1, qf.shape[1])
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
        if self._q_strategy.variance == "shared":
            for mother in self.hierarchy.mothers:
                idx = [i for i, m in enumerate(self.child_mother) if m == mother]
                var[idx] = var[idx].mean(0, keepdim=True)
        self.q_var.copy_(var.clamp(min=self._q_strategy.var_floor))
        with torch.no_grad():
            self.q_weight.copy_(self._q_strategy.init_weight(weight))
            self.q_bias.zero_()
        t_all = targets.reshape(-1)
        if self._q_strategy.needs_logreg_fit or self.bcm_teacher_weight != 0.0:
            self._fit_logreg(flat.detach().cpu().numpy(), t_all.cpu().numpy())
        self._q_strategy.init_from_support(features, targets)

    def _fit_aef_gate(
        self, features: Tensor, targets: Tensor, epochs: int, lr: float
    ) -> None:
        """Jointly fits the AEF gate's ``nn.Linear`` and the linear head
        (``q_weight``/``q_bias``) by Adam, minimising the same
        per-mother cross-entropy as ``q: linear``'s ``pl_trainer``-driven
        path (:meth:`_support_ce`, called here directly) — self-contained
        like :meth:`_fit_logreg`, so these parameters are never touched by
        the generic training loop (``pl_trainer.max_steps`` stays ``0``,
        as for ``logreg``/``proto``). Uses whatever :meth:`set_aux` was
        last given (``None`` means an un-gated linear head, with a
        warning — a silent degradation that should not happen in a
        correctly configured run). ``epochs``/``lr`` come from
        ``LinearAefGateStrategy``, the only caller."""
        if self._aux is None:
            logger.warning(
                "HiSplit linear_aef_gate: no aux (AEF) set, fitting an "
                "un-gated linear head instead."
            )
        if (targets >= 0).sum() == 0:
            logger.warning("HiSplit linear_aef_gate: no labelled support pixel.")
            return
        weight = nn.Parameter(self.q_weight.detach().clone())
        bias = nn.Parameter(self.q_bias.detach().clone())
        params = list(self.aef_gate_linear.parameters()) + [weight, bias]
        opt = torch.optim.Adam(params, lr=lr)
        for _ in range(epochs):
            opt.zero_grad()
            loss = self._support_ce(features, targets, weight, bias)
            if not torch.isfinite(loss):
                logger.warning(
                    "HiSplit linear_aef_gate: non-finite support loss, "
                    "stopping the fit early."
                )
                break
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
            opt.step()
        with torch.no_grad():
            self.q_weight.copy_(weight)
            self.q_bias.copy_(bias)

    def _fit_logreg(self, X, t) -> None:
        """BCM's classifier per mother on the HiSplit targets."""
        from pytorch_segmentation_models_trainer.few_shot.methods.bcm import (
            _LogisticRegressionCV,
        )

        rng = np.random.RandomState(self.seed)
        self.logreg_models = {}
        for mother, idx in self._groups():
            sel = np.isin(t, idx)
            y = t[sel] - min(idx)
            if len(np.unique(y)) < 2:
                logger.warning(
                    "HiSplit logreg: mother %d has a single class in the support.",
                    mother,
                )
                continue
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self.logreg_models[mother] = _LogisticRegressionCV(
                    *self.logreg_args, rng, n_jobs=self.logreg_n_jobs
                ).fit(X[sel], y)

    # ------------------------------------------------------------------
    # Scores and probabilities
    # ------------------------------------------------------------------

    def _scores(self, features: Tensor, weight: Tensor, bias: Tensor) -> Tensor:
        """Per-child scores ``(B, n_children, h, w)``; weight may be per map.
        Dispatches to :attr:`_q_strategy` (see ``q_strategies.py``) after
        the shared gate/Tukey-power step (:meth:`_qfeat`)."""
        features = self._qfeat(features)
        return self._q_strategy.scores(features, weight, bias)

    def _logreg_scores(self, features: Tensor) -> Tensor:
        """Log-probabilities of the per-mother logistic regressions (uniform
        for a mother without a fitted model)."""
        b, _, hh, ww = features.shape
        X = (
            features.permute(0, 2, 3, 1)
            .reshape(-1, features.shape[1])
            .detach()
            .cpu()
            .numpy()
        )
        scores = features.new_zeros(b, len(self.child_classes), hh, ww)
        for mother, idx in self._groups():
            model = self.logreg_models.get(mother)
            if model is None:
                continue
            proba, _ = model.predict_proba(X, len(idx))
            proba = torch.from_numpy(proba).to(features).view(b, hh, ww, len(idx))
            scores[:, idx] = torch.log(proba.clamp(min=_EPS)).permute(0, 3, 1, 2)
        return scores

    def _edl_alpha(self, scores: Tensor) -> Tensor:
        """Dirichlet parameters of the pair head: ``softplus(s) + W·a``."""
        return F.softplus(scores) + self.edl_prior.view(1, -1, 1, 1)

    def _groups(self):
        for mother in self.hierarchy.mothers:
            yield mother, [i for i, m in enumerate(self.child_mother) if m == mother]

    def _q(self, scores: Tensor) -> Tensor:
        """Split within the children of each mother: softmax of the scores,
        or the Dirichlet mean ``α / S`` for ``edl`` (:attr:`_q_strategy`)."""
        return self._q_strategy.normalize(scores)

    def _adjust_prior(self, q: Tensor) -> Tensor:
        if self.novel_prior_weight == 1.0:
            return q
        w = torch.ones(q.shape[1], device=q.device, dtype=q.dtype)
        w[self.novel_rows] = self.novel_prior_weight
        q = q * w.view(1, -1, 1, 1)
        out = torch.empty_like(q)
        for _, idx in self._groups():
            out[:, idx] = q[:, idx] / q[:, idx].sum(1, keepdim=True)
        return out

    def _apply_confidence_weight(self, features: Tensor, q: Tensor) -> Tensor:
        """Continuously blend ``q`` towards each mother's kept child by
        ``1 − confidence_weight_measure(x)`` (see class doc); identity when
        ``confidence_weight_measure`` is ``None``. The measure is first
        raised to ``confidence_weight_power`` and multiplied by
        ``confidence_weight_scale`` (both default to ``1.0``, a no-op),
        then clamped back to ``[0, 1]``."""
        if self.confidence_weight_measure is None:
            return q
        maps = self.uncertainty(features)
        measure = maps[self.confidence_weight_measure]
        if self.confidence_weight_power != 1.0:
            measure = measure**self.confidence_weight_power
        if self.confidence_weight_scale != 1.0:
            measure = measure * self.confidence_weight_scale
        measure = measure.clamp(0.0, 1.0)
        w = (1.0 - measure).clamp(0.0, 1.0).unsqueeze(1)
        out = q.clone()
        for mother, idx in self._groups():
            kept_local = idx.index(self.kept_row[mother])
            e_kept = torch.zeros_like(q[:, idx])
            e_kept[:, kept_local] = 1.0
            out[:, idx] = w * q[:, idx] + (1.0 - w) * e_kept
        return out

    def _apply_hard_decode(self, features: Tensor, q: Tensor) -> Tensor:
        """Snap ``q`` to a one-hot at its own arg-max, per mother, wherever
        ``hard_decode_measure`` is below ``hard_decode_threshold``
        (confident) — a BCM-style hard override gated by uncertainty
        instead of HiSplit's usual soft factorization (see class doc);
        identity when ``hard_decode_measure`` is ``None``."""
        if self.hard_decode_measure is None:
            return q
        maps = self.uncertainty(features)
        measure = maps[self.hard_decode_measure]
        confident = (measure < self.hard_decode_threshold).unsqueeze(1)
        out = q.clone()
        for _, idx in self._groups():
            qm = q[:, idx]
            hard = F.one_hot(qm.argmax(1), len(idx)).movedim(-1, 1).to(qm.dtype)
            out[:, idx] = torch.where(confident, hard, qm)
        return out

    def leak_coefficients(self) -> Optional[Tensor]:
        """``t[child, base_class]`` (zero outside novel rows), or ``None``."""
        if not self.leak:
            return None
        t = torch.sigmoid(self.leak_logit) * self.leak_mask
        return t / t.sum(0, keepdim=True).clamp(min=1.0)  # Σ_n t[n, c] ≤ 1

    def _masses(self, p_base: Tensor, q: Tensor):
        """Superclass-level probabilities ``p''`` (B, Cb) and child masses
        (B, n_children), with the leak (if any) and the prior weight."""
        q = self._adjust_prior(q)
        level = p_base.clone()
        mass = torch.stack([p_base[:, m] for m in self.child_mother], dim=1) * q
        t = self.leak_coefficients()
        if t is not None:
            inflow = torch.einsum("nc,bchw->bnhw", t, p_base)
            level = level - p_base * t.sum(0).view(1, -1, 1, 1)
            mass = mass + inflow
            for i in self.novel_rows:
                level[:, self.child_mother[i]] = (
                    level[:, self.child_mother[i]] + inflow[:, i]
                )
        return level, mass

    def _compute_q(self, features: Tensor) -> Tensor:
        """``q(child|x)`` after confidence-weighting/hard-decode -- the
        same 3 lines :meth:`split_probabilities` and :meth:`_decode` each
        used to repeat inline, extracted so a caller (either of them, or
        :class:`GateEnsemble`, which averages this across 2 independently
        gated ``HiSplit`` instances before decoding) can get the raw ``q``
        without duplicating the confidence-weight/hard-decode steps."""
        q = self._q(self._scores(features, *self._current_params(features.shape[0])))
        q = self._apply_confidence_weight(features, q)
        return self._apply_hard_decode(features, q)

    def split_probabilities(
        self, features: Tensor, q: Optional[Tensor] = None
    ) -> Tensor:
        """``p(c)`` over all classes ``(B, num_classes, h, w)``."""
        if q is None:
            q = self._compute_q(features)
        level, mass = self._masses(self._base_probs(features), q)
        out = level.new_zeros(
            level.shape[0], self.hierarchy.num_classes, *level.shape[2:]
        )
        out[:, : self.hierarchy.num_base_classes] = level
        for i, child in enumerate(self.child_classes):
            out[:, child] = mass[:, i]
        return out

    def _current_params(self, n_maps: int):
        if self._task_params is not None and self._task_params[0].shape[0] == n_maps:
            return self._task_params
        return self.q_weight, self.q_bias

    # ------------------------------------------------------------------
    # Training / transductive adaptation
    # ------------------------------------------------------------------

    def _support_ce(
        self,
        features: Tensor,
        targets: Tensor,
        weight,
        bias,
        sample_weight: Optional[Tensor] = None,
    ) -> Tensor:
        q = self._q(self._scores(features, weight, bias))
        valid = targets >= 0
        idx = targets.clamp(min=0).unsqueeze(1)
        p = q.gather(1, idx).squeeze(1)
        ce = -torch.log(p + _EPS)
        if sample_weight is None:
            return (ce * valid).sum() / valid.sum().clamp(min=1)
        w = valid * sample_weight
        return (ce * w).sum() / w.sum().clamp(min=_EPS)

    def _support_tversky(
        self,
        features: Tensor,
        targets: Tensor,
        weight,
        bias,
        alpha: float,
        beta: float,
    ) -> Tensor:
        """``1 − mean_c Tversky_c`` of ``q`` against the support targets,
        averaged over children present in this episode's support (``trans``
        only, see class doc, whose ``tversky_alpha``/``tversky_beta`` are
        passed in here); ``0`` if none are present."""
        q = self._q(self._scores(features, weight, bias))
        c = q.shape[1]
        valid = targets >= 0
        y = F.one_hot(targets.clamp(min=0), c).movedim(-1, 1).to(q.dtype)
        v = valid.unsqueeze(1).to(q.dtype)
        tp = (q * y * v).sum(dim=(0, 2, 3))
        fp = (q * (1 - y) * v).sum(dim=(0, 2, 3))
        fn = ((1 - q) * y * v).sum(dim=(0, 2, 3))
        tversky = (tp + _EPS) / (tp + alpha * fp + beta * fn + _EPS)
        present = (y * v).sum(dim=(0, 2, 3)) > 0
        if not present.any():
            return q.new_zeros(())
        return 1.0 - tversky[present].mean()

    def _edl_loss(
        self,
        features: Tensor,
        targets: Tensor,
        sample_weight: Optional[Tensor],
        kl_anneal_steps: int,
    ) -> Tensor:
        """EDL MSE + annealed KL of the evidential pair head, per mother
        (``kl_anneal_steps`` comes from ``EDLStrategy``, the only caller)."""
        alpha = self._edl_alpha(self._scores(features, self.q_weight, self.q_bias))
        kl_weight = min(1.0, self._edl_step / max(kl_anneal_steps, 1))
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
            if sample_weight is None:
                total = total + per_pixel[in_group].mean()
            else:
                w = sample_weight[in_group]
                total = total + (per_pixel[in_group] * w).sum() / w.sum().clamp(
                    min=_EPS
                )
        return total

    def support_loss(self, features: Tensor, masks: Tensor) -> Optional[Tensor]:
        """Support loss: q's loss for trainable variants (CE for
        ``linear``/``trans``, EDL MSE + annealed KL for ``edl``), optionally
        weighted per pixel by ``support_loss_weight_measure`` (see class
        doc), plus, with ``leak``, the NLL of the final distribution;
        ``None`` if nothing to train."""
        loss = None
        sample_weight = None
        if self.support_loss_weight_measure is not None:
            measure = self.uncertainty(features)[self.support_loss_weight_measure]
            sample_weight = (1.0 - measure).clamp(0.0, 1.0)
        if self._q_strategy.trainable:
            targets = self._targets(features, masks)
            loss = self._q_strategy.training_loss(features, targets, sample_weight)
        if self.leak:
            leak_loss = self._final_nll(features, masks)
            loss = leak_loss if loss is None else loss + leak_loss
        return loss

    def _final_nll(self, features: Tensor, masks: Tensor) -> Tensor:
        """NLL of the support labels under the final distribution (trains the
        leak): novel/base labels ``−log p'(label)``, "not novel" labels
        ``−log(1 − Σ_novel p')``."""
        p = self.split_probabilities(features)
        novel = [self.child_classes[i] for i in self.novel_rows]
        not_novel = masks == self.not_novel_index
        valid = (masks != 255) & (not_novel | (masks < p.shape[1]))
        labels = torch.where(valid & ~not_novel, masks, torch.zeros_like(masks))
        p_label = p.gather(1, labels.unsqueeze(1)).squeeze(1)
        p_rest = 1.0 - p[:, novel].sum(1)
        prob = torch.where(not_novel, p_rest, p_label)
        nll = -torch.log(prob.clamp(min=_EPS))
        return (nll * valid).sum() / valid.sum().clamp(min=1)

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

    def _bcm_teacher_term(self, query_features: Tensor, q: Tensor) -> Tensor:
        """Cross-entropy of ``q`` against a BCM-style ``LogisticRegressionCV``
        teacher (fit on the support by :meth:`init_from_support`, see class
        doc), weighted by the teacher's own confidence (``1 − entropy``);
        ``0`` when disabled or no teacher was fit (``trans`` only)."""
        n = query_features.shape[0]
        if self.bcm_teacher_weight == 0.0 or not self.logreg_models:
            return q.new_zeros(n)
        with torch.no_grad():
            teacher_scores = self._logreg_scores(self._qfeat(query_features))
            teacher_q = self._q(teacher_scores)
        ce_term = q.new_zeros(n)
        for _, idx in self._groups():
            qm = q[:, idx]
            tm = teacher_q[:, idx]
            with torch.no_grad():
                conf = 1.0 - normalized_entropy(tm)
                pseudo = tm.argmax(1)
            p = qm.gather(1, pseudo.unsqueeze(1)).squeeze(1)
            ce_px = -torch.log(p + _EPS)
            ce_term = ce_term + (ce_px * conf).flatten(1).sum(1) / conf.flatten(
                1
            ).sum(1).clamp(min=_EPS)
        return ce_term

    def adapt_to_query(
        self, support_features: Tensor, support_masks: Tensor, query_features: Tensor
    ) -> None:
        """Transductive per-task adaptation -- dispatches to
        :attr:`_q_strategy` (only ``trans`` implements it, see
        ``q_strategies.py``; every other q's :attr:`transductive` is
        ``False`` so ``GFSSModel`` never calls this)."""
        self._q_strategy.adapt_to_query(support_features, support_masks, query_features)

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

    def abstention(self, features: Tensor, measure: str, threshold: float) -> Tensor:
        """Abstention map ``(B, h, w)``: pixels the base model assigns to a
        mother whose split uncertainty ``measure`` exceeds ``threshold``. Their
        decision goes back to the mother class (e.g. "low vegetation")
        instead of a child.

        Raises:
            KeyError: ``measure`` is not provided by this configuration.
        """
        maps = self.uncertainty(features)
        if measure not in maps:
            raise KeyError(
                f"uncertainty measure {measure!r} not available ({sorted(maps)})."
            )
        base_pred = self._base_logits(features).argmax(1)
        in_superclass = torch.isin(
            base_pred, torch.tensor(self.hierarchy.mothers, device=base_pred.device)
        )
        return in_superclass & (maps[measure] > threshold)

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
                alpha_q = self._edl_alpha(scores)[:, idx]
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

    def _decode(
        self, features: Tensor, threshold: Optional[float], q: Optional[Tensor] = None
    ) -> Tensor:
        """Hierarchical-decoding logits; ``threshold`` enables widening.
        ``q`` lets a caller (:class:`GateEnsemble`) supply an already-averaged
        ``q`` instead of this instance's own -- same role as the ``q``
        parameter of :meth:`split_probabilities`."""
        if q is None:
            q = self._compute_q(features)
        logits_base = self._base_logits(features)
        p_base = self.base_probabilities(logits_base)
        level, mass = self._masses(p_base, q)
        log_level = torch.log(level.clamp(min=_EPS))
        out = log_level.new_empty(
            level.shape[0], self.hierarchy.num_classes, *level.shape[2:]
        )
        # Stage 1 outside the superclasses: every base class gets the SAME
        # per-pixel shift log(1 − T_top), T_top = leaked fraction of the base
        # model's class at that pixel. The order among these classes stays the
        # base model's (uneven leak fractions cannot swap neighbours) and the
        # base class still competes with the superclass with its exact p''.
        # Without leak the shift is 0 (out = log p'').
        out[:, : self.hierarchy.num_base_classes] = torch.log(p_base.clamp(min=_EPS))
        t = self.leak_coefficients()
        if t is not None:
            kept = (1.0 - t.sum(0)).clamp(min=_EPS)  # (Cb,)
            shift = torch.log(kept)[p_base.argmax(1)]  # (B, h, w)
            out[:, : self.hierarchy.num_base_classes] += shift.unsqueeze(1)
        for i, (child, mother) in enumerate(zip(self.child_classes, self.child_mother)):
            share = mass[:, i] / level[:, mother].clamp(min=_EPS)
            out[:, child] = log_level[:, mother] + self.decode_eps * (share - 1.0)
        if self.abstain_measure is not None:
            abstain = self.abstention(features, self.abstain_measure, self.abstain_threshold)
            for i in self.novel_rows:
                child = self.child_classes[i]
                out[:, child] = torch.where(
                    abstain, out.new_full((), float("-inf")), out[:, child]
                )
        if threshold is None:
            return out
        top2 = level.topk(2, dim=1).indices
        best = log_level.max(1).values
        if self.widen == "dissonance":
            alpha = self.base_alpha(logits_base)
            gate = dissonance((alpha - 1.0) / alpha.sum(1, keepdim=True)) >= threshold
        for mother, idx in self._groups():
            if self.widen == "prob":
                gate = level[:, mother] >= threshold
            winner = mass[:, idx].argmax(1)
            for j, i in enumerate(idx):
                if i not in self.novel_rows:
                    continue
                widen = (
                    (top2[:, 0] != mother)
                    & (top2[:, 1] == mother)
                    & gate
                    & (winner == j)
                )
                child = self.child_classes[i]
                out[:, child] = torch.where(
                    widen, best + self.decode_eps, out[:, child]
                )
        return out

    def variant_names(self) -> List[str]:
        """Decoding variants evaluated besides the main one (``sweep``)."""
        return [f"widen_{t:g}" for t in self.sweep]

    def decode_variants(self, features: Tensor) -> Dict[str, Tensor]:
        """Logits of each ``sweep`` threshold."""
        return {f"widen_{t:g}": self._decode(features, t) for t in self.sweep}

    def forward(self, features: Tensor, q: Optional[Tensor] = None) -> Tensor:
        """Log-probabilities (``flat``) or hierarchical decoding logits
        (with widening when ``widen`` is set). ``q`` -- see
        :meth:`split_probabilities`/:meth:`_decode` -- lets
        :class:`GateEnsemble` drive the decode with an averaged ``q``."""
        if self.decoding == "flat":
            return torch.log(self.split_probabilities(features, q=q) + _EPS)
        return self._decode(
            features, None if self.widen == "none" else self.widen_threshold, q=q
        )
