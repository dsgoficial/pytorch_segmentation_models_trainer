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

import torch
from torch import Tensor

from pytorch_segmentation_models_trainer.few_shot.hierarchy import ClassHierarchy

_EPS = 1e-10


def valid_mean(t: Tensor, valid: Tensor, dim) -> Tensor:
    """Mean of ``t`` over ``dim`` restricted to ``valid`` (DIaM's ``_valid_mean``)."""
    return (valid * t).sum(dim=dim) / (valid.sum(dim=dim) + _EPS)


def support_one_hot(
    masks: Tensor,
    hierarchy: ClassHierarchy,
    not_novel_index: int,
    ignore_index: int = 255,
) -> Tensor:
    """One-hot support targets ``(..., num_classes, h, w)`` from label maps.

    Novel and base labels go to their own channel. "Not any novel class"
    labels (``not_novel_index``) go to the channel of the first mother: the
    official DIaM/ClassTrans code encodes them as background (channel 0,
    the mother in standard GFSS) before projecting all base probabilities
    onto it. Ignored pixels get an all-zero vector.

    Args:
        masks: Integer labels ``(..., h, w)``.
        hierarchy: Task hierarchy.
        not_novel_index: Label meaning "labelled, not any novel class".
        ignore_index: Ignored label.
    """
    labels = masks.clone()
    labels[masks == not_novel_index] = hierarchy.mothers[0]
    valid = masks != ignore_index
    labels[~valid] = 0
    one_hot = torch.nn.functional.one_hot(labels.long(), hierarchy.num_classes)
    one_hot = one_hot.movedim(-1, -3).float()
    return one_hot * valid.unsqueeze(-3)


def projected_ce(
    probas: Tensor,
    masks: Tensor,
    hierarchy: ClassHierarchy,
    not_novel_index: int,
    not_novel_weight: float = 1.0,
    ignore_index: int = 255,
) -> Tensor:
    """Support cross-entropy with DIaM's projection π_S (Eq. 5).

    A pixel labelled ``not_novel_index`` only says "not any novel class",
    so its probability is the sum over all base classes; other labels use
    their own channel. As in the official code (``compute_wce``), the
    "not novel" term is weighted by ``not_novel_weight``.

    Args:
        probas: ``(T, S, C, h, w)`` probabilities (T tasks, S support maps).
        masks: ``(S, h, w)`` support labels.

    Returns:
        ``(T,)`` mean over valid support pixels, per task.
    """
    nb = hierarchy.num_base_classes
    not_novel = masks == not_novel_index
    valid = (masks != ignore_index).float()
    labels = masks.clone()
    labels[not_novel | (masks == ignore_index)] = 0
    index = (
        labels.long().unsqueeze(0).unsqueeze(2).expand(probas.shape[0], -1, 1, -1, -1)
    )
    p_label = probas.gather(2, index).squeeze(2)
    p_base = probas[:, :, :nb].sum(dim=2)
    p = torch.where(not_novel, p_base, p_label)
    weight = torch.where(
        not_novel, torch.full_like(p, not_novel_weight), torch.ones_like(p)
    )
    ce = -torch.log(p + _EPS) * weight
    return valid_mean(ce, valid.unsqueeze(0), (1, 2, 3))


def entropy_and_marginal_kl(probas: Tensor, valid: Tensor, pi: Tensor):
    """Conditional entropy and KL(marginal ‖ π) of query predictions.

    Args:
        probas: ``(T, 1, C, h, w)``.
        valid: ``(T, 1, h, w)``.
        pi: ``(T, C)`` class-proportion prior.

    Returns:
        ``(d_kl, entropy)``, each ``(T,)``.
    """
    entropy = -(probas * torch.log(probas + _EPS)).sum(2)
    entropy = valid_mean(entropy, valid, (1, 2, 3))
    marginal = valid_mean(probas, valid.unsqueeze(2), (1, 3, 4))
    d_kl = (marginal * torch.log(_EPS + marginal / (pi + _EPS))).sum(1)
    return d_kl, entropy


def hierarchical_kd(
    probas: Tensor, snapshot: Tensor, valid: Tensor, hierarchy: ClassHierarchy
) -> Tensor:
    """KL(π_new2old(p) ‖ p_snapshot) per task (DIaM Eq. 11-12), with novel
    classes summed into their mothers.

    Args:
        probas: ``(T, 1, C, h, w)`` current probabilities.
        snapshot: ``(T, 1, Cb, h, w)`` frozen base probabilities.
        valid: ``(T, 1, h, w)``.
    """
    adjusted = hierarchy.project_new_to_old(probas, dim=2)
    kl = (adjusted * torch.log(_EPS + adjusted / (_EPS + snapshot))).sum(dim=2)
    return valid_mean(kl, valid, (1, 2, 3))


def default_not_novel_weight(n_support: int, n_novel: int) -> float:
    """Official ``compute_wce``: 0.01 with one shot per novel class, else 0.15."""
    return 0.01 if n_support // max(n_novel, 1) == 1 else 0.15


def class_prior(probas: Tensor, valid: Tensor) -> Tensor:
    """Self-estimated π: mean predicted probabilities over valid pixels ``(T, C)``."""
    return valid_mean(probas, valid.unsqueeze(2), (1, 3, 4)).detach()


def novel_prototypes(
    features: Tensor, masks: Tensor, hierarchy: ClassHierarchy
) -> Tensor:
    """L2-normalised mean feature of each novel class ``(F, n_novel)``.

    Pools every support pixel labelled with the class (official DIaM
    ``init_prototypes``). Classes without pixels give a zero column.
    """
    cols = []
    for cls in hierarchy.novel_classes:
        m = (masks == cls).unsqueeze(1).float()
        cols.append(valid_mean(features, m, (0, 2, 3)))
    proto = torch.stack(cols, dim=1)
    return proto / (proto.norm(dim=0, keepdim=True) + _EPS)
