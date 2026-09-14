# -*- coding: utf-8 -*-
"""Online Object-wise label Correction (O2C) — Liu et al., TGRS 2024 (AIO2).

Multiclass generalization of ``obj_wise_label_correction`` from the official
reference implementation (https://github.com/zhu-xlab/AIO2,
``utils/self_ensembling.py``), which is binary-only (single foreground class,
building-footprint segmentation). There is no multiclass version in the
paper or the official code — every design decision below that goes beyond a
literal per-class replay of the official binary logic is documented as such,
and mirrored in the article repository's decision log
(``doutorado_artigo_sam_noisy_labels/descricao_novos_experimentos.md``,
section "Adendo").
"""

import logging
from typing import Dict, Sequence

import numpy as np
import torch
from scipy import ndimage
from skimage.measure import label as cc_label

logger = logging.getLogger(__name__)

_VALID_CONFLICT_RESOLUTIONS = ("priority_order", "teacher_confidence")


def online_object_correction(
    noisy_hard: torch.Tensor,
    teacher_probs: torch.Tensor,
    classes_to_correct: Sequence[int],
    filter_size: int = -1,
    conflict_resolution: str = "priority_order",
) -> torch.Tensor:
    """Correct one noisy hard label using the teacher's predictions (O2C).

    Operates on a single image (not batched — call once per item in a
    batch; connected-component labelling has no batched GPU implementation,
    same constraint as the official code).

    For each class ``c`` in ``classes_to_correct`` (order matters — see
    ``conflict_resolution``), finds connected components of
    ``teacher_probs[c] > 0.5`` that do **not** overlap any pixel already
    labelled ``c`` in ``noisy_hard`` — an "object the noisy label appears to
    have missed" — and proposes adding them.

    Multiclass adaptations beyond the official (binary) algorithm:

    1. **Conflict resolution.** With one foreground class, a pixel can only
       ever be proposed once. With several classes, two channels can propose
       the same pixel. Two strategies (decision recorded 2026-09-14):

       - ``"priority_order"``: the first class in ``classes_to_correct``
         wins any disputed pixel. Deterministic, no dependency on model
         confidence, but the order is an arbitrary choice.
       - ``"teacher_confidence"``: ``argmax_c teacher_probs[c]`` wins at the
         disputed pixel — mirrors how SAM/SLICO order mask overwrites by
         confidence (``predicted_iou``/``area``, ascending) in
         ``tools.region_correction.correction.apply_region_correction``.

       Neither has support from the original paper; both are implemented
       and a pilot run decides which goes into the full experiment matrix
       (or both stay as an ablation axis).

    2. **"Fixed classes" restore.** After resolving conflicts, every pixel
       whose *original* ``noisy_hard`` class is not in
       ``classes_to_correct`` is restored, discarding any proposal for it —
       replicated verbatim from
       ``apply_region_correction`` (``correction.py:83-85``): eligibility is
       gated by the pixel's current class, never by the proposed one. The
       first draft of this function got this backwards (gated eligibility
       by which classes could *propose*, not which pixels could be
       *changed*), which let a protected class get silently overwritten by
       an eligible class's proposal — fixed here; see the article repo's
       decision log for the full writeup.

       Note this is not a literal transplant of the SAM/SLICO rule: there,
       ``classes_to_correct`` restricts only the *source* class — the
       majority-vote winner may be any of the 6 classes. Here it restricts
       *both* ends — proposals are only ever generated for target classes
       within the set (the per-class loop only iterates
       ``classes_to_correct``), and the final restore enforces source
       eligibility on top of that. Deliberate: O2C is target-class-driven
       (find missed objects *of class c*), not vote-driven like SAM, so
       there's no natural way for an out-of-scope target class to "win"
       here the way it can win a majority vote — scoping both ends to the
       same set keeps the correction strictly confined to the classes
       under study.

    3. **Soft boundary → hard erosion.** The official code can emit
       *continuous* soft labels at new-object boundaries (box-filtered
       confidence) for its binary sigmoid+BCE loss. Our loss
       (``WeightedDiceCrossEntropyLoss``) needs a hard multiclass target, so
       a continuous confidence isn't directly consumable without loss
       changes we decided not to make. ``filter_size > 0`` here instead
       *erodes* each newly-proposed class region: pixels where the local
       box-filtered average membership drops at or below 0.5 (i.e. the
       proposal's own boundary) are dropped from the correction, keeping
       only its more-confident interior. This approximates the same intent
       (don't trust a new object's edge as much as its interior) without
       requiring soft targets.

    Args:
        noisy_hard: ``(H, W)`` int/long tensor, the current noisy label.
        teacher_probs: ``(C, H, W)`` float tensor, softmax probabilities
            from the teacher model.
        classes_to_correct: Ordered sequence of eligible class indices.
            Order matters for ``conflict_resolution="priority_order"``, and
            no duplicates are allowed.
        filter_size: Odd box-filter size (pixels) for boundary erosion.
            ``<= 0`` disables it (hard proposals used as-is), matching the
            official code's "use hard labels" branch.
        conflict_resolution: ``"priority_order"`` or ``"teacher_confidence"``.

    Returns:
        ``(H, W)`` tensor, same dtype/device as ``noisy_hard`` — the
        corrected label.
    """
    if conflict_resolution not in _VALID_CONFLICT_RESOLUTIONS:
        raise ValueError(
            f"conflict_resolution must be one of {_VALID_CONFLICT_RESOLUTIONS}, "
            f"got {conflict_resolution!r}"
        )
    if filter_size > 0 and filter_size % 2 == 0:
        raise ValueError(f"filter_size must be odd, got {filter_size}")

    classes_to_correct = list(classes_to_correct)
    if not classes_to_correct:
        raise ValueError("classes_to_correct must not be empty")
    if len(classes_to_correct) != len(set(classes_to_correct)):
        raise ValueError("classes_to_correct must not contain duplicates")
    n_classes = teacher_probs.shape[0]
    if any(c < 0 or c >= n_classes for c in classes_to_correct):
        raise ValueError(
            f"classes_to_correct entries must be in [0, {n_classes}), "
            f"got {classes_to_correct}"
        )

    noisy_np = noisy_hard.detach().cpu().numpy()
    probs_np = teacher_probs.detach().cpu().numpy()

    proposals = _per_class_proposals(noisy_np, probs_np, classes_to_correct)
    assigned = _resolve_conflicts(
        proposals, probs_np, classes_to_correct, conflict_resolution
    )

    if filter_size > 0:
        assigned = _erode_boundaries(assigned, classes_to_correct, filter_size)

    corrected = noisy_np.copy()
    has_new = assigned >= 0
    corrected[has_new] = assigned[has_new]

    # "Fixed classes": restore every pixel whose ORIGINAL class is not
    # eligible for correction, regardless of what was proposed for it.
    # Verbatim mirror of apply_region_correction (SAM/SLICO), see docstring.
    non_target = ~np.isin(noisy_np, classes_to_correct)
    corrected[non_target] = noisy_np[non_target]

    return torch.from_numpy(corrected).to(
        dtype=noisy_hard.dtype, device=noisy_hard.device
    )


def _per_class_proposals(
    noisy_np: np.ndarray, probs_np: np.ndarray, classes_to_correct: Sequence[int]
) -> Dict[int, np.ndarray]:
    """Per-class candidate "missed object" masks (before conflict resolution).

    Port of the overlap-check core of ``obj_wise_label_correction``, run
    once per class instead of once (the official code has exactly one
    foreground class).
    """
    proposals = {}
    for c in classes_to_correct:
        pred_bin = probs_np[c] > 0.5
        noisy_bin = noisy_np == c
        labeled = cc_label(pred_bin)
        if labeled.max() == 0:
            proposals[c] = np.zeros_like(pred_bin, dtype=bool)
            continue
        overlap_ids = np.unique(labeled[noisy_bin])
        overlap_ids = overlap_ids[overlap_ids != 0]  # 0 = not-an-object, not a real id
        proposals[c] = (labeled > 0) & ~np.isin(labeled, overlap_ids)
    return proposals


def _resolve_conflicts(
    proposals: Dict[int, np.ndarray],
    probs_np: np.ndarray,
    classes_to_correct: Sequence[int],
    conflict_resolution: str,
) -> np.ndarray:
    """Return ``(H, W)`` int array: winning class per pixel, or ``-1`` (no proposal)."""
    shape = next(iter(proposals.values())).shape
    assigned = np.full(shape, -1, dtype=np.int64)

    if conflict_resolution == "priority_order":
        for c in classes_to_correct:  # first in the list wins disputed pixels
            claim = proposals[c] & (assigned == -1)
            assigned[claim] = c
        return assigned

    # teacher_confidence: highest teacher_probs[c] among proposing classes wins.
    stacked_probs = np.stack([probs_np[c] for c in classes_to_correct])  # (K, H, W)
    stacked_mask = np.stack(
        [proposals[c] for c in classes_to_correct]
    )  # (K, H, W) bool
    masked_probs = np.where(stacked_mask, stacked_probs, -np.inf)
    any_claim = stacked_mask.any(axis=0)
    best_idx = np.argmax(masked_probs, axis=0)  # index into classes_to_correct
    winning_class = np.asarray(classes_to_correct)[best_idx]
    assigned[any_claim] = winning_class[any_claim]
    return assigned


def _erode_boundaries(
    assigned: np.ndarray, classes_to_correct: Sequence[int], filter_size: int
) -> np.ndarray:
    """Shrink each newly-assigned class region inward (see module docstring point 3)."""
    result = assigned.copy()
    for c in classes_to_correct:
        mask = assigned == c
        if not mask.any():
            continue
        smoothed = ndimage.uniform_filter(mask.astype(np.float64), size=filter_size)
        keep = mask & (smoothed > 0.5)
        result[mask & ~keep] = -1
    return result
