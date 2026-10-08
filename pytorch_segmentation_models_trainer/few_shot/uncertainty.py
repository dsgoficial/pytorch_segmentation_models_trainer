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

import math
from typing import Dict, Sequence

import torch
from torch import Tensor
from torchmetrics import Metric

_EPS = 1e-10


def vacuity(alpha: Tensor) -> Tensor:
    """Subjective-logic vacuity ``K / S`` of Dirichlet parameters ``(B, K, ...)``."""
    return alpha.shape[1] / alpha.sum(dim=1)


def dissonance(belief: Tensor) -> Tensor:
    """Jøsang's dissonance of belief masses ``(B, K, ...)`` (Σ b ≤ 1).

    ``Σ_k b_k · Σ_{j≠k} b_j Bal(b_j, b_k) / Σ_{j≠k} b_j``, with
    ``Bal(b_j, b_k) = 1 − |b_j − b_k| / (b_j + b_k)``. 1 for two equal
    beliefs, 0 for belief in a single class or no belief at all.
    """
    bj = belief.unsqueeze(1)  # (B, 1, K, ...)
    bk = belief.unsqueeze(2)  # (B, K, 1, ...)
    total = bj + bk
    bal = torch.where(
        total > 0, 1 - (bj - bk).abs() / total.clamp(min=_EPS), torch.zeros_like(total)
    )
    k = belief.shape[1]
    off = (1 - torch.eye(k, device=belief.device, dtype=belief.dtype)).view(
        1, k, k, *([1] * (belief.dim() - 2))
    )
    num = (bj * bal * off).sum(2)
    den = (bj * off).sum(2)
    return (
        belief * torch.where(den > 0, num / den.clamp(min=_EPS), torch.zeros_like(den))
    ).sum(1)


def normalized_entropy(probs: Tensor) -> Tensor:
    """Entropy over dim 1 divided by ``log K`` (0 when ``K = 1``)."""
    k = probs.shape[1]
    if k < 2:
        return probs.new_zeros(probs.shape[0], *probs.shape[2:])
    return -(probs * torch.log(probs + _EPS)).sum(1) / math.log(k)


def _spearman(x: Tensor, y: Tensor) -> Tensor:
    if x.numel() < 2:
        return torch.tensor(float("nan"))
    rx = x.argsort().argsort().double()
    ry = y.argsort().argsort().double()
    rx, ry = rx - rx.mean(), ry - ry.mean()
    den = (rx.norm() * ry.norm()).clamp(min=_EPS)
    return (rx * ry).sum() / den


class GFSSUncertaintyMetrics(Metric):
    """Evaluation of per-pixel uncertainty measures (values in ``[0, 1]``).

    For every named measure:

    * ``aurc/<name>``: area under the risk–coverage curve of the decisions
      inside ``region`` (e.g. the split decisions: pixels the base model
      assigns to a mother whose true class is one of its children), pixels
      retained from the least to the most uncertain; computed from a
      histogram with ``n_bins`` bins (ties inside a bin share its risk).
      Lower is better. Abstaining on uncertain pixels = sending them back
      to the mother class.
    * ``coverage@<t>/<name>`` and ``risk@<t>/<name>``: operating points of
      the abstention — fraction of the region retained (``u ≤ t``, at the
      histogram resolution) and error rate among the retained decisions
      (``nan`` when nothing is retained), for every ``t`` in
      ``abstain_thresholds``.
    * ``ece/<name>``: expected calibration error of the confidence
      ``1 − u`` against the correctness of the decisions in ``region``,
      with ``ece_bins`` equal-width bins.
    * ``tile_mean/<name>`` and ``tile_spearman/<name>``: mean uncertainty of
      each tile (over ``tile_valid`` pixels) and Spearman correlation
      between it and the tile error (1 − pixel accuracy) across tiles.

    Args:
        names: Measures that ``update`` receives.
        n_bins: Histogram bins for the AURC.
        prefix: Prepended to every key.
        abstain_thresholds: Uncertainty thresholds of the operating points.
        ece_bins: Bins of the ECE (fine histogram bins are grouped into them).
    """

    full_state_update = False

    def __init__(
        self,
        names: Sequence[str],
        n_bins: int = 1000,
        prefix: str = "",
        abstain_thresholds: Sequence[float] = (0.25, 0.5, 0.75),
        ece_bins: int = 10,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.names = list(names)
        self.n_bins = int(n_bins)
        self.prefix = prefix
        self.abstain_thresholds = [float(t) for t in abstain_thresholds]
        self.ece_bins = int(ece_bins)
        for name in self.names:
            self.add_state(
                f"count_{name}", torch.zeros(self.n_bins, dtype=torch.long), "sum"
            )
            self.add_state(
                f"errors_{name}", torch.zeros(self.n_bins, dtype=torch.long), "sum"
            )
            self.add_state(
                f"sum_u_{name}", torch.zeros(self.n_bins, dtype=torch.float64), "sum"
            )
            self.add_state(f"tile_u_{name}", [], dist_reduce_fx="cat")
        self.add_state("tile_err", [], dist_reduce_fx="cat")

    def update(
        self,
        uncertainty: Dict[str, Tensor],
        region: Tensor,
        correct: Tensor,
        tile_valid: Tensor,
    ) -> None:
        """Accumulate one batch.

        Args:
            uncertainty: ``{name: (B, H, W)}`` maps in ``[0, 1]``.
            region: ``(B, H, W)`` bool, pixels of the AURC.
            correct: ``(B, H, W)`` bool, prediction == ground truth.
            tile_valid: ``(B, H, W)`` bool, non-ignored pixels.
        """
        for name in uncertainty:
            if name not in self.names:
                raise KeyError(
                    f"uncertainty measure {name!r} not declared ({self.names})."
                )
        n_valid = tile_valid.flatten(1).sum(1).clamp(min=1)
        err = 1 - (correct & tile_valid).flatten(1).sum(1) / n_valid
        self.tile_err.append(err.double())
        for name, u in uncertainty.items():
            bins = (u.clamp(0, 1) * self.n_bins).long().clamp(max=self.n_bins - 1)
            sel = bins[region]
            getattr(self, f"count_{name}").add_(
                torch.bincount(sel, minlength=self.n_bins)
            )
            getattr(self, f"errors_{name}").add_(
                torch.bincount(sel[~correct[region]], minlength=self.n_bins)
            )
            getattr(self, f"sum_u_{name}").add_(
                torch.bincount(sel, weights=u[region].double(), minlength=self.n_bins)
            )
            tile_u = (u * tile_valid).flatten(1).sum(1) / n_valid
            getattr(self, f"tile_u_{name}").append(tile_u.double())

    def compute(self) -> Dict[str, Tensor]:
        out = {}
        errs = (
            self.tile_err
            if isinstance(self.tile_err, Tensor)
            else (torch.cat(self.tile_err) if self.tile_err else torch.zeros(0))
        )
        for name in self.names:
            count = getattr(self, f"count_{name}").double()
            errors = getattr(self, f"errors_{name}").double()
            total = count.sum()
            if total > 0:
                cum_n, cum_e = count.cumsum(0), errors.cumsum(0)
                risk = torch.where(
                    cum_n > 0, cum_e / cum_n.clamp(min=1), torch.zeros_like(cum_n)
                )
                aurc = (count / total * risk).sum()
            else:
                aurc = torch.tensor(float("nan"), dtype=torch.float64)
            out[f"aurc/{name}"] = aurc
            out.update(self._operating_points(name, count, errors, total))
            out[f"ece/{name}"] = self._ece(name, count, errors, total)
            tu = getattr(self, f"tile_u_{name}")
            tu = (
                tu
                if isinstance(tu, Tensor)
                else (torch.cat(tu) if tu else torch.zeros(0))
            )
            out[f"tile_mean/{name}"] = (
                tu.mean() if tu.numel() else torch.tensor(float("nan"))
            )
            out[f"tile_spearman/{name}"] = _spearman(tu.cpu(), errs.cpu())
        return {f"{self.prefix}{k}": v.float() for k, v in out.items()}

    def _operating_points(self, name, count, errors, total) -> Dict[str, Tensor]:
        out = {}
        nan = torch.tensor(float("nan"), dtype=torch.float64)
        for t in self.abstain_thresholds:
            keep = int(round(t * self.n_bins))
            kept = count[:keep].sum()
            out[f"coverage@{t:g}/{name}"] = kept / total if total > 0 else nan.clone()
            out[f"risk@{t:g}/{name}"] = (
                errors[:keep].sum() / kept if kept > 0 else nan.clone()
            )
        return out

    def _ece(self, name, count, errors, total) -> Tensor:
        if total == 0:
            return torch.tensor(float("nan"), dtype=torch.float64)
        coarse = (
            torch.arange(self.n_bins, device=count.device)
            * self.ece_bins
            // self.n_bins
        )
        n = count.new_zeros(self.ece_bins).index_add_(0, coarse, count)
        wrong = errors.new_zeros(self.ece_bins).index_add_(0, coarse, errors)
        sum_u = count.new_zeros(self.ece_bins).index_add_(
            0, coarse, getattr(self, f"sum_u_{name}").to(count.dtype)
        )
        filled = n > 0
        accuracy = 1.0 - wrong[filled] / n[filled]
        confidence = 1.0 - sum_u[filled] / n[filled]
        return (n[filled] / total * (accuracy - confidence).abs()).sum()
