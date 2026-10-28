# -*- coding: utf-8 -*-
"""
/***************************************************************************
 pytorch_segmentation_models_trainer
                              -------------------
        begin                : 2026-10-28
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


class TripleGateEnsemble(BaseGFSSMethod):
    """Fixed-weight ensemble of 3 independently-gated :class:`HiSplit`
    experts sharing the same frozen base and ``K``-shot support, averaging
    their ``q`` (equal ``1/3`` weight each) before decoding -- the
    3-source extension of :class:`~.gate_ensemble.GateEnsemble` ("média
    AEF+OlmoEarth"), investigated 2026-10-28 (user request) to test
    whether adding a third external embedding product (Tessera, see
    ``few_shot/tessera_cache.py``) beats the 2-source média with
    statistical significance.

    **Not a mixture of experts** -- same terminology correction as
    :class:`GateEnsemble` (see its docstring): the equal ``1/3`` weight is
    constant, not conditioned on the input, so this is a plain ensemble
    (probability averaging), not MoE in the standard (gating-network,
    input-dependent-weight) sense. Named ``TripleGateMoE`` until
    2026-10-28, renamed for the same reason as the 2-expert class.

    A separate class rather than generalizing :class:`GateEnsemble` to N
    experts: that class's name/interface (``expert_a``/``expert_b``,
    ``0.5/0.5``) is already the one cited throughout the companion
    research project's analysis as "the proposed method" -- this is a
    distinct, explicitly-3-way ablation, not a replacement. Equal ``1/3``
    weight is the direct analogue of the 2-expert ``0.5/0.5`` (itself
    confirmed near-optimal by a weight sweep, §8.14 of the analysis doc) --
    no weight sweep has been run for 3 experts; if this beats the 2-source
    média with significance, that would be a natural follow-up, not
    assumed here.

    All decode-time modifiers are taken from ``expert_a`` only, same
    convention as :class:`GateEnsemble` -- see its docstring for the
    ``p_base · avg(q_a, q_b, q_c) == avg(p_base·q_a, p_base·q_b, p_base·q_c)``
    reasoning (identical here, just 3 terms).

    Args:
        expert_a: A :class:`HiSplit` instance, or a Hydra sub-config --
            its decode-time settings drive the final decode.
        expert_b: A second :class:`HiSplit` instance or sub-config --
            only its ``q`` contributes to the average.
        expert_c: A third :class:`HiSplit` instance or sub-config --
            only its ``q`` contributes to the average.
        aux_name_a: ``gfss.aux_cache_loaders`` key routed to ``expert_a``.
            Default ``"aef"``.
        aux_name_b: ``gfss.aux_cache_loaders`` key routed to ``expert_b``.
            Default ``"olmoearth"``.
        aux_name_c: ``gfss.aux_cache_loaders`` key routed to ``expert_c``.
            Default ``"tessera"`` (matches :class:`TesseraCacheLoader`'s
            own cache key).
    """

    def __init__(
        self,
        expert_a: Union[HiSplit, Dict[str, Any]],
        expert_b: Union[HiSplit, Dict[str, Any]],
        expert_c: Union[HiSplit, Dict[str, Any]],
        aux_name_a: str = "aef",
        aux_name_b: str = "olmoearth",
        aux_name_c: str = "tessera",
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.expert_a = self._resolve_expert(expert_a)
        self.expert_b = self._resolve_expert(expert_b)
        self.expert_c = self._resolve_expert(expert_c)
        names = (aux_name_a, aux_name_b, aux_name_c)
        if len(set(names)) != 3:
            raise ValueError(f"aux_name_a/b/c must all differ, got {names!r}.")
        self.aux_name_a = aux_name_a
        self.aux_name_b = aux_name_b
        self.aux_name_c = aux_name_c
        self.aux_names = names
        self.transductive = bool(
            self.expert_a.transductive or self.expert_b.transductive or self.expert_c.transductive
        )

    @staticmethod
    def _resolve_expert(expert: Union[HiSplit, Dict[str, Any]]) -> HiSplit:
        """See :meth:`GateEnsemble._resolve_expert` -- identical reasoning."""
        if isinstance(expert, HiSplit):
            return expert
        from hydra.utils import instantiate

        resolved = instantiate(expert, _recursive_=False)
        if not isinstance(resolved, HiSplit):
            raise TypeError(
                f"TripleGateEnsemble's experts must be HiSplit instances, got "
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
        for expert in (self.expert_a, self.expert_b, self.expert_c):
            expert.setup(hierarchy, base_weight, base_bias, not_novel_index, base_output=base_output)

    def init_from_support(self, features: Tensor, masks: Tensor) -> None:
        for expert in (self.expert_a, self.expert_b, self.expert_c):
            expert.init_from_support(features, masks)

    def set_aux(self, aux: Dict[str, Tensor]) -> None:
        """Routes the 3 named signals to each expert's own ``"aef"`` slot
        (:meth:`HiSplit.set_aux` always reads that one key)."""
        self.expert_a.set_aux({"aef": aux.get(self.aux_name_a)})
        self.expert_b.set_aux({"aef": aux.get(self.aux_name_b)})
        self.expert_c.set_aux({"aef": aux.get(self.aux_name_c)})

    def support_loss(self, features: Tensor, masks: Tensor) -> Optional[Tensor]:
        losses = [
            l
            for l in (
                self.expert_a.support_loss(features, masks),
                self.expert_b.support_loss(features, masks),
                self.expert_c.support_loss(features, masks),
            )
            if l is not None
        ]
        return (sum(losses) / len(losses)) if losses else None

    def adapt_to_query(
        self, support_features: Tensor, support_masks: Tensor, query_features: Tensor
    ) -> None:
        for expert in (self.expert_a, self.expert_b, self.expert_c):
            if expert.transductive:
                expert.adapt_to_query(support_features, support_masks, query_features)

    def forward(self, features: Tensor) -> Tensor:
        q = (
            self.expert_a._compute_q(features)
            + self.expert_b._compute_q(features)
            + self.expert_c._compute_q(features)
        ) / 3.0
        return self.expert_a.forward(features, q=q)
