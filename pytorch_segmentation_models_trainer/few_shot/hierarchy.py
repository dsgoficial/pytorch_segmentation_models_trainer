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

from typing import Dict, List, Mapping

import torch
from torch import Tensor


class ClassHierarchy:
    """Mother → children mapping between a base model and its GFSS extension.

    The base model predicts ``num_base_classes`` classes (indices
    ``0 .. num_base_classes - 1``). Each *mother* is a base class whose pixels,
    during base training, also contained one or more *novel* classes. After
    adaptation the model predicts ``num_classes = num_base_classes + n_novel``
    classes: base classes keep their indices and novel classes take the
    contiguous indices ``num_base_classes .. num_classes - 1``.

    Every mother must list itself among its children (it keeps its index for
    the part that is not novel, e.g. ``3: [3, 5]`` splits "low vegetation" into
    "grassland" (3) and "cropland" (5)). Standard GFSS, where novel classes
    were hidden in the background during base training, is ``{0: [0, n1, ...]}``.

    Args:
        mapping: ``{mother: [children]}``. Keys and values may be ints or
            numeric strings (Hydra ``DictConfig`` is accepted).
        num_base_classes: Number of classes of the base model.

    Example YAML::

        gfss:
          hierarchy:
            3: [3, 5]     # low vegetation -> grassland (3), cropland (5)
            4: [4, 6]     # second mother, second novel class
    """

    def __init__(self, mapping: Mapping, num_base_classes: int) -> None:
        self.num_base_classes = int(num_base_classes)
        self._children: Dict[int, List[int]] = {
            int(m): [int(c) for c in children] for m, children in mapping.items()
        }
        self._validate()
        self.mothers: List[int] = sorted(self._children)
        self.novel_classes: List[int] = sorted(
            c
            for children in self._children.values()
            for c in children
            if c >= self.num_base_classes
        )
        self.num_classes = self.num_base_classes + len(self.novel_classes)
        lut = list(range(self.num_classes))
        for mother, children in self._children.items():
            for child in children:
                lut[child] = mother
        self.new_to_old: Tensor = torch.tensor(lut, dtype=torch.long)

    def _validate(self) -> None:
        if not self._children:
            raise ValueError("ClassHierarchy needs at least one mother class.")
        seen: Dict[int, int] = {}
        novel: List[int] = []
        for mother, children in self._children.items():
            if not 0 <= mother < self.num_base_classes:
                raise ValueError(
                    f"mother {mother} is not a base class "
                    f"(0..{self.num_base_classes - 1})."
                )
            if mother not in children:
                raise ValueError(
                    f"mother {mother} must be one of its children {children}."
                )
            own_novel = [c for c in children if c >= self.num_base_classes]
            if not own_novel:
                raise ValueError(f"mother {mother} has no novel child in {children}.")
            for child in children:
                if child in seen:
                    raise ValueError(
                        f"class {child} is assigned to more than one mother "
                        f"({seen[child]} and {mother})."
                    )
                seen[child] = mother
                if child < self.num_base_classes and child != mother:
                    raise ValueError(
                        f"base class {child} cannot be a child of mother {mother}."
                    )
            novel.extend(own_novel)
        expected = list(
            range(self.num_base_classes, self.num_base_classes + len(novel))
        )
        if sorted(novel) != expected:
            raise ValueError(
                f"novel classes {sorted(novel)} must be contiguous and start at "
                f"{self.num_base_classes} (expected {expected})."
            )

    def children_of(self, mother: int) -> List[int]:
        """Return the children of ``mother`` (including the mother itself)."""
        return list(self._children[mother])

    def mother_of(self, cls: int) -> int:
        """Return the base class that ``cls`` belonged to during base training."""
        return int(self.new_to_old[cls])

    def project_new_to_old(self, probs: Tensor, dim: int = 1) -> Tensor:
        """Sum the probabilities of each mother's children into the mother.

        Generalises DIaM's ``π_new2old`` (Eq. 11), which sums every novel class
        into the background, to arbitrary mothers.

        Args:
            probs: Probabilities with ``num_classes`` entries along ``dim``.
            dim: Class dimension.

        Returns:
            Tensor with ``num_base_classes`` entries along ``dim``.
        """
        if probs.shape[dim] != self.num_classes:
            raise ValueError(
                f"expected {self.num_classes} channels along dim {dim}, "
                f"got {probs.shape[dim]}."
            )
        dim = dim % probs.dim()
        shape = list(probs.shape)
        shape[dim] = self.num_base_classes
        index_shape = [1] * probs.dim()
        index_shape[dim] = self.num_classes
        index = self.new_to_old.to(probs.device).view(index_shape).expand_as(probs)
        out = probs.new_zeros(shape)
        return out.scatter_add(dim, index, probs)

    def to_old_labels(self, labels: Tensor, ignore_index: int = 255) -> Tensor:
        """Map label maps in the new class space to the base class space.

        Labels outside ``0 .. num_classes - 1`` (e.g. ``ignore_index``) are
        kept unchanged.
        """
        out = labels.clone()
        valid = (labels >= 0) & (labels < self.num_classes) & (labels != ignore_index)
        out[valid] = self.new_to_old.to(labels.device)[labels[valid]]
        return out

    def __repr__(self) -> str:
        return (
            f"ClassHierarchy({self._children}, "
            f"num_base_classes={self.num_base_classes})"
        )
