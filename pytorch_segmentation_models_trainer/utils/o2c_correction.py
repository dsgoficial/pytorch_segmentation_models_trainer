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

GPU acceleration (2026-09-18 addendum)
---------------------------------------
This function is called once per image in a batch, every batch in
``correct_base="iter"`` mode (the default) — a real hot path. The default
(CPU) path round-trips through ``.cpu().numpy()`` and uses
``skimage``/``scipy`` for connected-component labelling and boundary
erosion, which forces a CUDA sync + host transfer on every call.

An earlier attempt at a GPU-native connected-components replacement
(``kornia.contrib.connected_components``) was verified and rejected: it's
an *approximate*, iterative label-propagation algorithm, not the exact
scanline algorithm skimage uses. Empirically, on realistic 256x256 masks it
mismatched skimage in 41/60 trials at its default ``num_iterations=100``,
and still needed ~512 iterations (5x default) to reach 0/60 mismatches in
that sample — with no formal convergence guarantee for an unseen mask
shape. Not acceptable for an operation that decides which pixels a
published method corrects.

The path below instead uses **cuCIM** (RAPIDS, ``uv sync --extra gpu-ml`` —
see the README's "AIO2 O2C GPU Acceleration" section) for
``label``/``uniform_filter``: connected-component labelling has an exact
solution (Rosenfeld & Pfaltz 1966) — every correct implementation,
including cuCIM's, produces an identical partition to skimage's, they only
differ in speed. cuCIM interoperates with PyTorch CUDA tensors via DLPack
(zero-copy — no host transfer at all). Requires an NVIDIA GPU with compute
capability 7.0+ (Volta or newer, e.g. V100) and a matching CUDA major
version for the ``cucim-cuXX`` wheel (the ``gpu-ml`` extra pins this).

**Not exercised by this repo's test suite** (no CUDA on the dev machine) —
only the CPU path and the ``use_gpu`` dispatch *decision* are covered by
tests. Before trusting the GPU path in a real training run: on the actual
training server, run one batch through both ``use_gpu=True`` and
``use_gpu=False`` on the *same* inputs and assert the outputs are
identical, the same equivalence check used to reject kornia above.
"""

import logging
from typing import Any, Dict, Optional, Sequence

import numpy as np
import torch
from scipy import ndimage
from skimage.measure import label as cc_label

logger = logging.getLogger(__name__)

_VALID_CONFLICT_RESOLUTIONS = ("priority_order", "teacher_confidence")

_cucim_available_cache: Optional[bool] = None


def _cucim_available() -> bool:
    """Whether cuCIM (+ CuPy) can be imported — cached after the first check."""
    global _cucim_available_cache
    if _cucim_available_cache is None:
        try:
            import cucim.scipy.ndimage  # noqa: F401
            import cucim.skimage.measure  # noqa: F401
            import cupy  # noqa: F401

            _cucim_available_cache = True
        except ImportError:
            _cucim_available_cache = False
    return _cucim_available_cache


def _resolve_use_gpu(use_gpu: Optional[bool], is_cuda: bool) -> bool:
    """Decide whether to run the cuCIM/CuPy path.

    ``None`` (default) — GPU path iff the input tensor is already on a CUDA
    device *and* cuCIM is importable; otherwise the CPU path, silently.
    ``True`` — force the GPU path, raising if cuCIM isn't available (rather
    than silently falling back, so a misconfigured server doesn't quietly
    train on the slow path without anyone noticing).
    ``False`` — force the CPU path regardless of device (useful to run the
    GPU-input/CPU-path equivalence check called for in the module docstring).
    """
    if use_gpu is None:
        return is_cuda and _cucim_available()
    if use_gpu and not _cucim_available():
        raise RuntimeError(
            "online_object_correction: use_gpu=True but cuCIM is not "
            "importable. Install it with `uv sync --extra gpu-ml` (see the "
            "README's 'AIO2 O2C GPU Acceleration' section). Leave use_gpu "
            "unset (None) to fall back to the CPU path automatically instead."
        )
    return bool(use_gpu)


def online_object_correction(
    noisy_hard: torch.Tensor,
    teacher_probs: torch.Tensor,
    classes_to_correct: Sequence[int],
    filter_size: int = -1,
    conflict_resolution: str = "priority_order",
    use_gpu: Optional[bool] = None,
) -> torch.Tensor:
    """Correct one noisy hard label using the teacher's predictions (O2C).

    Operates on a single image (not batched — call once per item in a
    batch; connected-component labelling has no batched GPU implementation
    in either backend used here, same constraint as the official code).

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
        use_gpu: ``None`` (default) auto-selects cuCIM/CuPy when
            ``noisy_hard`` is already on a CUDA device and cuCIM is
            importable, else the CPU (numpy/scipy/skimage) path. ``True``
            forces the GPU path (raises if cuCIM isn't available). ``False``
            forces the CPU path regardless of device — see the module
            docstring's GPU-acceleration note for why/how to use this for
            an equivalence check.

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

    use_gpu = _resolve_use_gpu(use_gpu, bool(noisy_hard.is_cuda))

    if use_gpu:
        import cupy as cp
        from cucim.scipy import ndimage as ndi
        from cucim.skimage.measure import label as label_fn

        xp = cp
        noisy_arr = cp.from_dlpack(
            torch.utils.dlpack.to_dlpack(noisy_hard.detach().contiguous())
        )
        probs_arr = cp.from_dlpack(
            torch.utils.dlpack.to_dlpack(teacher_probs.detach().contiguous())
        )
    else:
        xp = np
        ndi = ndimage
        label_fn = cc_label
        noisy_arr = noisy_hard.detach().cpu().numpy()
        probs_arr = teacher_probs.detach().cpu().numpy()

    proposals = _per_class_proposals(
        noisy_arr, probs_arr, classes_to_correct, xp, label_fn
    )
    assigned = _resolve_conflicts(
        proposals, probs_arr, classes_to_correct, conflict_resolution, xp
    )

    if filter_size > 0:
        assigned = _erode_boundaries(assigned, classes_to_correct, filter_size, xp, ndi)

    corrected = noisy_arr.copy()
    has_new = assigned >= 0
    corrected[has_new] = assigned[has_new]

    # "Fixed classes": restore every pixel whose ORIGINAL class is not
    # eligible for correction, regardless of what was proposed for it.
    # Verbatim mirror of apply_region_correction (SAM/SLICO), see docstring.
    non_target = ~xp.isin(noisy_arr, classes_to_correct)
    corrected[non_target] = noisy_arr[non_target]

    if use_gpu:
        out = torch.utils.dlpack.from_dlpack(corrected.toDlpack())
        return out.to(dtype=noisy_hard.dtype, device=noisy_hard.device)
    return torch.from_numpy(corrected).to(
        dtype=noisy_hard.dtype, device=noisy_hard.device
    )


def _per_class_proposals(
    noisy_arr, probs_arr, classes_to_correct: Sequence[int], xp, label_fn
) -> Dict[int, Any]:
    """Per-class candidate "missed object" masks (before conflict resolution).

    Port of the overlap-check core of ``obj_wise_label_correction``, run
    once per class instead of once (the official code has exactly one
    foreground class). ``xp``/``label_fn`` select the numpy+skimage or
    cupy+cuCIM backend — the logic is identical either way.
    """
    proposals = {}
    for c in classes_to_correct:
        pred_bin = probs_arr[c] > 0.5
        noisy_bin = noisy_arr == c
        labeled = label_fn(pred_bin)
        if labeled.max() == 0:
            proposals[c] = xp.zeros_like(pred_bin, dtype=bool)
            continue
        overlap_ids = xp.unique(labeled[noisy_bin])
        overlap_ids = overlap_ids[overlap_ids != 0]  # 0 = not-an-object, not a real id
        proposals[c] = (labeled > 0) & ~xp.isin(labeled, overlap_ids)
    return proposals


def _resolve_conflicts(
    proposals: Dict[int, Any],
    probs_arr,
    classes_to_correct: Sequence[int],
    conflict_resolution: str,
    xp,
):
    """Return ``(H, W)`` int array: winning class per pixel, or ``-1`` (no proposal)."""
    shape = next(iter(proposals.values())).shape
    assigned = xp.full(shape, -1, dtype=np.int64)

    if conflict_resolution == "priority_order":
        for c in classes_to_correct:  # first in the list wins disputed pixels
            claim = proposals[c] & (assigned == -1)
            assigned[claim] = c
        return assigned

    # teacher_confidence: highest teacher_probs[c] among proposing classes wins.
    stacked_probs = xp.stack([probs_arr[c] for c in classes_to_correct])  # (K, H, W)
    stacked_mask = xp.stack(
        [proposals[c] for c in classes_to_correct]
    )  # (K, H, W) bool
    masked_probs = xp.where(stacked_mask, stacked_probs, -np.inf)
    any_claim = stacked_mask.any(axis=0)
    best_idx = xp.argmax(masked_probs, axis=0)  # index into classes_to_correct
    winning_class = xp.asarray(classes_to_correct)[best_idx]
    assigned[any_claim] = winning_class[any_claim]
    return assigned


def _erode_boundaries(
    assigned, classes_to_correct: Sequence[int], filter_size: int, xp, ndi
):
    """Shrink each newly-assigned class region inward (see module docstring point 3)."""
    result = assigned.copy()
    for c in classes_to_correct:
        mask = assigned == c
        if not mask.any():
            continue
        smoothed = ndi.uniform_filter(mask.astype(np.float64), size=filter_size)
        keep = mask & (smoothed > 0.5)
        result[mask & ~keep] = -1
    return result
