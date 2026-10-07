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

from typing import Optional, Sequence

import torch
from torch import Tensor

from pytorch_segmentation_models_trainer.few_shot.base_method import BaseGFSSMethod
from pytorch_segmentation_models_trainer.few_shot.losses import (
    class_prior,
    default_not_novel_weight,
    entropy_and_marginal_kl,
    hierarchical_kd,
    novel_prototypes,
    projected_ce,
)


def _logits(features: Tensor, weight: Tensor, bias: Tensor) -> Tensor:
    """``(T, O, F, h, w)`` x per-task ``(T, F, C)``, ``(T, C)`` -> ``(T, O, C, h, w)``."""
    out = torch.einsum("bochw,bcC->boChw", features, weight)
    return out + bias.unsqueeze(1).unsqueeze(3).unsqueeze(4)


class DIaM(BaseGFSSMethod):
    """DIaM (Hajimiri, Boudiaf & Ben Ayed, CVPR 2023), transductive.

    Port of the official ``Classifier`` (https://github.com/sinahmr/DIaM,
    ``src/classifier.py``): for every query batch, one linear classifier per
    query map (base rows + novel rows initialised with L2-normalised support
    prototypes) is optimised for ``adapt_iter`` SGD steps on

    ``w_ce·CE_S + w_kl·KL(marginal ‖ π) + w_ent·H(p_q) + w_kd·KD``

    with the support CE projected as in Eq. 5 and the KD of Eq. 11-12. The
    only generalisation is the hierarchy: novel classes are summed into
    their **mother** (the background in the original), and pixels labelled
    ``not_novel_index`` play the role of the original background label.
    With ``hierarchy = {0: [0, novel...]}`` the result matches the official
    code (parity test). Query pixels are all treated as valid (the official
    test uses the query ground truth only to drop ignored pixels).

    Args:
        weights: ``[w_ce, w_kl, w_ent, w_kd]`` (official: 100, 1, 1, 100).
        adapt_iter: SGD iterations per query batch (official: 100).
        lr: SGD learning rate, no momentum (official: 1.25e-3).
        pi_estimation: ``self`` (model's own predictions, official) or
            ``uniform``.
        pi_update_at: Iterations (1-based) where π is re-estimated.
        fine_tune_base_classifier: Also optimise the base rows (official).
        not_novel_weight: CE weight of "not novel" pixels; ``None`` = official
            rule (0.01 with one support map per novel class, else 0.15).

    Example YAML::

        gfss:
          method:
            _target_: pytorch_segmentation_models_trainer.few_shot.methods.diam.DIaM
            weights: [100, 1, 1, 100]
            adapt_iter: 100
            lr: 1.25e-3
        pl_trainer:
          max_steps: 0
          inference_mode: false
    """

    transductive = True

    def __init__(
        self,
        weights: Sequence[float] = (100.0, 1.0, 1.0, 100.0),
        adapt_iter: int = 100,
        lr: float = 1.25e-3,
        pi_estimation: str = "self",
        pi_update_at: Sequence[int] = (10,),
        fine_tune_base_classifier: bool = True,
        not_novel_weight: Optional[float] = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        if pi_estimation not in {"self", "uniform"}:
            raise ValueError(
                f"pi_estimation must be 'self' or 'uniform', got {pi_estimation!r}."
            )
        self.weights = [float(w) for w in weights]
        self.adapt_iter = int(adapt_iter)
        self.lr = float(lr)
        self.pi_estimation = pi_estimation
        self.pi_update_at = [int(i) for i in pi_update_at]
        self.fine_tune_base_classifier = fine_tune_base_classifier
        self.not_novel_weight = not_novel_weight
        self._task_weight: Optional[Tensor] = None
        self._task_bias: Optional[Tensor] = None

    def init_from_support(self, features: Tensor, masks: Tensor) -> None:
        """Novel rows = normalised support prototypes, zero bias."""
        proto = novel_prototypes(features, masks, self.hierarchy)
        self.register_buffer("novel_weight", proto)
        self.register_buffer("novel_bias", proto.new_zeros(proto.shape[1]))
        self._n_support = features.shape[0]

    def _initial_params(self, n_tasks: int):
        base_w = self.base_weight.T.unsqueeze(0).repeat(n_tasks, 1, 1)
        base_b = self.base_bias.unsqueeze(0).repeat(n_tasks, 1)
        novel_w = self.novel_weight.unsqueeze(0).repeat(n_tasks, 1, 1)
        novel_b = self.novel_bias.unsqueeze(0).repeat(n_tasks, 1)
        return base_w, base_b, novel_w, novel_b

    def _prior(self, probas: Tensor, valid: Tensor) -> Tensor:
        if self.pi_estimation == "uniform":
            n = probas.shape[2]
            return probas.new_full((probas.shape[0], n), 1.0 / n)
        return class_prior(probas, valid)

    def adapt_to_query(
        self, support_features: Tensor, support_masks: Tensor, query_features: Tensor
    ) -> None:
        """Optimise one classifier per query map (official ``optimize``)."""
        w_ce, w_kl, w_ent, w_kd = self.weights
        n_tasks = query_features.shape[0]
        base_w, base_b, novel_w, novel_b = self._initial_params(n_tasks)
        params = [novel_w, novel_b]
        if self.fine_tune_base_classifier:
            params += [base_w, base_b]
        for p in params:
            p.requires_grad_()
        optimizer = torch.optim.SGD(params, lr=self.lr)

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

        def weights_now():
            return torch.cat([base_w, novel_w], dim=2), torch.cat(
                [base_b, novel_b], dim=1
            )

        with torch.no_grad():
            pi = self._prior(torch.softmax(_logits(fq, *weights_now()), 2), valid_q)
        for it in range(self.adapt_iter):
            w, b = weights_now()
            proba_s = torch.softmax(_logits(fs, w, b), dim=2)
            proba_q = torch.softmax(_logits(fq, w, b), dim=2)
            kd = hierarchical_kd(proba_q, snapshot, valid_q, self.hierarchy)
            d_kl, entropy = entropy_and_marginal_kl(proba_q, valid_q, pi)
            ce = projected_ce(
                proba_s, support_masks, self.hierarchy, self.not_novel_index, nn_weight
            )
            loss = w_ce * ce + w_kl * d_kl + w_ent * entropy + w_kd * kd
            optimizer.zero_grad()
            loss.sum(0).backward()
            optimizer.step()
            if (
                (it + 1) in self.pi_update_at
                and self.pi_estimation == "self"
                and w_kl != 0
            ):
                with torch.no_grad():
                    pi = self._prior(
                        torch.softmax(_logits(fq, *weights_now()), 2), valid_q
                    )

        w, b = weights_now()
        self._task_weight, self._task_bias = w.detach(), b.detach()

    def forward(self, features: Tensor) -> Tensor:
        """Logits with the classifiers of the last adapted batch (same size),
        or with the initial classifier (base + prototypes) otherwise."""
        if (
            self._task_weight is not None
            and self._task_weight.shape[0] == features.shape[0]
        ):
            w, b = self._task_weight, self._task_bias
        else:
            base_w, base_b, novel_w, novel_b = self._initial_params(features.shape[0])
            w, b = torch.cat([base_w, novel_w], 2), torch.cat([base_b, novel_b], 1)
        return _logits(features.unsqueeze(1), w, b).squeeze(1)
