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

from typing import Any, Dict, Optional, Union

from torch import Tensor

from pytorch_segmentation_models_trainer.few_shot.base_method import BaseGFSSMethod
from pytorch_segmentation_models_trainer.few_shot.hierarchy import ClassHierarchy
from pytorch_segmentation_models_trainer.few_shot.methods.hisplit import HiSplit


class GateEnsemble(BaseGFSSMethod):
    """Fixed-weight ensemble of 2 independently-gated :class:`HiSplit`
    experts sharing the same frozen base and ``K``-shot support, averaging
    their ``q`` before decoding -- the "média AEF+OlmoEarth" proposed
    method: 2 ``q: linear_aef_gate`` experts, one per external embedding
    product (AlphaEarth Foundations, OlmoEarth v1.3), each self-fitting
    its own gate+head on the same support, their decisions combined by a
    plain ``0.5/0.5`` average instead of choosing one product.

    **Not a mixture of experts, despite an earlier name/doc pass calling
    it that** (renamed 2026-10-28, user request, after a conceptual review
    of the terminology): the standard definition of MoE requires a gating
    network that produces input-dependent weights -- conditioned on each
    example/pixel, so the gate can "trust" a different expert in different
    regions of the input space (Jacobs & Jordan 1991; the same requirement
    in every modern deep-learning MoE, e.g. sparsely-gated layers). A fixed
    ``0.5/0.5`` (or any other constant split) has no such conditioning --
    it is a plain **ensemble** (probability/decision averaging), not MoE.
    This project *did* test an actual MoE-style mechanism -- a per-pixel
    routing weight computed from each gate's own split-confidence
    (``ensemble_splitconf`` in the companion research project's pilot
    script, genuinely input-dependent) -- and it **lost** to this fixed-
    weight ensemble with strong, consistent significance. Calling the
    adopted method "MoE" was therefore not just imprecise but backwards:
    the actual MoE-style mechanism is the one that did *not* make the cut.

    Motivated by the companion doctoral project's `run_gate_moe_grid.py`
    pilot script (outside this repo): concatenating both embeddings into
    ONE gate was already shown redundant (``aef_gate_uncertainty``, an
    extra input channel, made no measurable difference), so this instead
    averages between 2 separately-fit gates, the same pattern already
    validated by ``run_ensemble_two_biomes_grid.py`` for a different
    combination axis (biome instead of embedding product). That script's
    ``ensemble_avg`` arm (plain average, no routing) beat the per-pixel
    confidence-routed arm (``ensemble_splitconf``, the actual MoE
    mechanism, see above) more consistently against BCM across K -- this
    class implements exactly that arm as a first-class library method, not
    a one-off script, so it runs through the standard
    ``ExperimentsRunner``/``GFSSModel`` pipeline (metrics, TTA,
    checkpoints) like every other :class:`BaseGFSSMethod`.

    Averaging ``q`` (each expert's post-gate, post-softmax split
    distribution, already through :meth:`HiSplit._compute_q` -- confidence
    weighting/hard-decode included) rather than each expert's full
    ``split_probabilities`` is equivalent here and reuses the ``q=``
    override :meth:`HiSplit.split_probabilities`/``_decode``/``forward``
    already accept for this purpose: both experts share the exact same
    frozen base (``setup`` is called on both with the same
    ``base_weight``/``base_bias``), so ``p_base`` is identical for both and
    ``p_base · avg(q_a, q_b) == avg(p_base·q_a, p_base·q_b)``.

    All decode-time modifiers (``decoding``, ``widen``, ``leak``,
    ``sweep``, ``abstain_measure``, ``confidence_weight_measure``,
    ``hard_decode_measure``) are taken from **``expert_a`` only** -- the
    final decode is literally ``expert_a.forward(features, q=avg_q)``.
    ``expert_b`` only contributes its own gate-fitted ``q``; its own
    decode-time settings are never consulted. Configure them on
    ``expert_a``; ``expert_b`` only needs its ``q: linear_aef_gate``
    hyperparameters (``aef_dim``/``aef_gate_epochs``/``aef_gate_lr``) and
    whatever cross-cutting flags (e.g. ``bcm_teacher_weight``) matter to
    ITS OWN gate fit, if any.

    Not limited to 2 ``linear_aef_gate`` experts specifically -- any 2
    :class:`HiSplit` instances work (:meth:`support_loss`/
    :meth:`adapt_to_query` delegate to whichever expert is
    :attr:`trainable`/transductive) -- but that combination (both self-fit,
    neither trainable nor transductive, ``pl_trainer.max_steps: 0``) is the
    only one exercised by the companion research project and its tests.

    Args:
        expert_a: A :class:`HiSplit` instance, or a Hydra sub-config
            (``_target_: ...hisplit.HiSplit``) -- its decode-time settings
            drive the final decode (see above).
        expert_b: A second :class:`HiSplit` instance or sub-config --
            only its ``q`` contributes to the average.
        aux_name_a: ``gfss.aux_cache_loaders`` key routed to ``expert_a``'s
            own ``"aef"`` aux slot (:meth:`HiSplit.set_aux`). Default
            ``"aef"`` (AlphaEarth Foundations' conventional name).
        aux_name_b: ``gfss.aux_cache_loaders`` key routed to ``expert_b``'s
            own ``"aef"`` aux slot. Default ``"olmoearth"`` (matches
            ``OlmoEarthCacheLoader``'s own cache key).

    Example YAML::

        gfss:
          hierarchy: {3: [3, 5]}
          method:
            _target_: pytorch_segmentation_models_trainer.few_shot.methods.gate_ensemble.GateEnsemble
            expert_a:
              _target_: pytorch_segmentation_models_trainer.few_shot.methods.hisplit.HiSplit
              q:
                _target_: pytorch_segmentation_models_trainer.few_shot.methods.q_strategies.LinearAefGateStrategy
                aef_gate_epochs: 200
                aef_gate_lr: 1.0e-2
              tau: 10.0
              decoding: hierarchical
              aef_dim: 64
            expert_b:
              _target_: pytorch_segmentation_models_trainer.few_shot.methods.hisplit.HiSplit
              q:
                _target_: pytorch_segmentation_models_trainer.few_shot.methods.q_strategies.LinearAefGateStrategy
                aef_gate_epochs: 200
                aef_gate_lr: 1.0e-2
              tau: 10.0
              decoding: hierarchical
              aef_dim: 128
          aux_cache_loaders:
            aef: /path/to/aef_cache
            olmoearth:
              _target_: pytorch_segmentation_models_trainer.few_shot.olmoearth_cache.OlmoEarthCacheLoader
              cache_dir: /path/to/olmoearth_cache
              n_bands: 128
        pl_trainer:
          max_steps: 0
        train_dataset:
          return_metadata: true
        test_dataset:
          return_metadata: true
    """

    def __init__(
        self,
        expert_a: Union[HiSplit, Dict[str, Any]],
        expert_b: Union[HiSplit, Dict[str, Any]],
        aux_name_a: str = "aef",
        aux_name_b: str = "olmoearth",
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.expert_a = self._resolve_expert(expert_a)
        self.expert_b = self._resolve_expert(expert_b)
        if aux_name_a == aux_name_b:
            raise ValueError(
                f"aux_name_a and aux_name_b must differ, got {aux_name_a!r} twice."
            )
        self.aux_name_a = aux_name_a
        self.aux_name_b = aux_name_b
        self.aux_names = (aux_name_a, aux_name_b)
        self.transductive = bool(self.expert_a.transductive or self.expert_b.transductive)

    @staticmethod
    def _resolve_expert(expert: Union[HiSplit, Dict[str, Any]]) -> HiSplit:
        """An already-built :class:`HiSplit` (plain Python) or a Hydra
        sub-config with its own ``_target_`` (YAML: this method is itself
        instantiated with ``_recursive_=False`` by ``GFSSModel``, so nested
        ``_target_`` blocks arrive un-instantiated and must be resolved by
        hand here, same pattern as ``HiSplit._resolve_q_strategy``)."""
        if isinstance(expert, HiSplit):
            return expert
        from hydra.utils import instantiate

        resolved = instantiate(expert, _recursive_=False)
        if not isinstance(resolved, HiSplit):
            raise TypeError(
                f"GateEnsemble's experts must be HiSplit instances, got "
                f"{type(resolved).__name__}."
            )
        return resolved

    def setup(
        self,
        hierarchy: ClassHierarchy,
        base_weight: Tensor,
        base_bias: Tensor,
        not_novel_index: int = 254,
        base_output: str = "softmax",
    ) -> None:
        super().setup(hierarchy, base_weight, base_bias, not_novel_index, base_output)
        self.expert_a.setup(
            hierarchy, base_weight, base_bias, not_novel_index, base_output=base_output
        )
        self.expert_b.setup(
            hierarchy, base_weight, base_bias, not_novel_index, base_output=base_output
        )

    def init_from_support(self, features: Tensor, masks: Tensor) -> None:
        self.expert_a.init_from_support(features, masks)
        self.expert_b.init_from_support(features, masks)

    def set_aux(self, aux: Dict[str, Tensor]) -> None:
        """Routes the 2 named signals to each expert's own ``"aef"`` slot
        (:meth:`HiSplit.set_aux` always reads that one key, regardless of
        which product actually feeds a given expert)."""
        self.expert_a.set_aux({"aef": aux.get(self.aux_name_a)})
        self.expert_b.set_aux({"aef": aux.get(self.aux_name_b)})

    def support_loss(self, features: Tensor, masks: Tensor) -> Optional[Tensor]:
        losses = [
            l
            for l in (
                self.expert_a.support_loss(features, masks),
                self.expert_b.support_loss(features, masks),
            )
            if l is not None
        ]
        return (sum(losses) / len(losses)) if losses else None

    def adapt_to_query(
        self, support_features: Tensor, support_masks: Tensor, query_features: Tensor
    ) -> None:
        if self.expert_a.transductive:
            self.expert_a.adapt_to_query(support_features, support_masks, query_features)
        if self.expert_b.transductive:
            self.expert_b.adapt_to_query(support_features, support_masks, query_features)

    def forward(self, features: Tensor) -> Tensor:
        q = 0.5 * (self.expert_a._compute_q(features) + self.expert_b._compute_q(features))
        return self.expert_a.forward(features, q=q)
