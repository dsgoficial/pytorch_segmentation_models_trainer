# -*- coding: utf-8 -*-
"""Shared region-based majority-vote label correction.

Generator-agnostic core used by both ``tools.sam_correction`` (SAM AMG segments)
and ``tools.slico_correction`` (SLICO superpixels). The correction rule — majority
vote within each region, applied only to pixels whose original class is eligible
for correction — is identical regardless of how the region partition was produced;
only the partition source differs between the two callers.
"""

from pathlib import Path
from typing import Dict, FrozenSet, List, Tuple, Union

import numpy as np


def apply_region_correction(
    bags_raw: np.ndarray,
    segments: Union[List[Dict], np.ndarray],
    lulc_maps: List[np.ndarray],
    classes_to_correct: FrozenSet[int],
    num_classes: int = 6,
    include_bags: bool = True,
) -> np.ndarray:
    """Apply region-based majority-vote correction to a mask array.

    For each region that contains at least one pixel from ``classes_to_correct``,
    the winning class is determined by majority vote across all sources
    (optionally the original mask + each LULC map). The winner is written to
    every pixel in the region. Non-target class pixels are always restored from
    the original mask after processing.

    Args:
        bags_raw: Original mask array (H, W) uint8 — topographic vector source.
        segments: Region partition, in one of two formats:

            - ``List[Dict]`` — SAM AMG style: dicts with keys ``segmentation``
              (bool H×W), ``predicted_iou`` (float), ``area`` (int). Regions may
              overlap; they are processed ascending by ``(predicted_iou, area)``
              so higher-confidence, larger regions overwrite smaller/less-confident
              ones in overlapping pixels.
            - ``np.ndarray`` — dense integer label map (H, W), e.g. SLIC/SLICO
              output. Regions form a strict, non-overlapping partition (every
              pixel belongs to exactly one label), so processing is order-free
              and vectorized (no per-region Python loop).

        lulc_maps: List of auxiliary class arrays (H, W) uint8 used as extra
            votes alongside ``bags_raw``.
        classes_to_correct: Set of class indices eligible for correction.
        num_classes: Number of valid class indices (values >= num_classes ignored).
        include_bags: If True, ``bags_raw`` is counted as one vote source.

    Returns:
        Corrected mask array (H, W) uint8, same shape as ``bags_raw``.
    """
    if isinstance(segments, np.ndarray):
        corrected = _apply_from_label_map(
            bags_raw, segments, lulc_maps, classes_to_correct, num_classes, include_bags
        )
    else:
        corrected = _apply_from_mask_list(
            bags_raw, segments, lulc_maps, classes_to_correct, num_classes, include_bags
        )

    backup = bags_raw
    non_target = ~np.isin(backup, list(classes_to_correct))
    corrected[non_target] = backup[non_target]
    return corrected


def _apply_from_mask_list(
    bags_raw: np.ndarray,
    sam_masks: List[Dict],
    lulc_maps: List[np.ndarray],
    classes_to_correct: FrozenSet[int],
    num_classes: int,
    include_bags: bool,
) -> np.ndarray:
    corrected = bags_raw.copy()

    # Sort ascending so higher-confidence/larger masks overwrite last.
    sorted_masks = sorted(sam_masks, key=lambda m: (m["predicted_iou"], m["area"]))

    all_sources: List[np.ndarray] = []
    if include_bags:
        all_sources.append(bags_raw)
    all_sources.extend(lulc_maps)

    for m in sorted_masks:
        seg: np.ndarray = m["segmentation"]
        if not seg.any():
            continue
        if not np.isin(bags_raw[seg], list(classes_to_correct)).any():
            continue

        vote_counts = np.zeros(num_classes, dtype=np.int64)
        for src in all_sources:
            vals = src[seg]
            valid = vals[vals < num_classes]
            np.add.at(vote_counts, valid.astype(int), 1)

        if vote_counts.sum() == 0:
            continue

        winner = int(vote_counts.argmax())
        corrected[seg] = winner

    return corrected


def _apply_from_label_map(
    bags_raw: np.ndarray,
    label_map: np.ndarray,
    lulc_maps: List[np.ndarray],
    classes_to_correct: FrozenSet[int],
    num_classes: int,
    include_bags: bool,
) -> np.ndarray:
    corrected = bags_raw.copy()

    target_mask = np.isin(bags_raw, list(classes_to_correct))
    if not target_mask.any() or label_map.size == 0:
        return corrected

    n_labels = int(label_map.max()) + 1

    all_sources: List[np.ndarray] = []
    if include_bags:
        all_sources.append(bags_raw)
    all_sources.extend(lulc_maps)

    flat_labels = label_map.ravel().astype(np.int64)
    vote_counts = np.zeros((n_labels, num_classes), dtype=np.int64)
    for src in all_sources:
        vals = src.ravel()
        valid = vals < num_classes
        idx = flat_labels[valid] * num_classes + vals[valid].astype(np.int64)
        vote_counts += np.bincount(idx, minlength=n_labels * num_classes).reshape(
            n_labels, num_classes
        )

    has_votes = vote_counts.sum(axis=1) > 0
    winners = np.zeros(n_labels, dtype=np.uint8)
    winners[has_votes] = vote_counts[has_votes].argmax(axis=1).astype(np.uint8)

    touches_target = np.zeros(n_labels, dtype=bool)
    np.logical_or.at(touches_target, flat_labels[target_mask.ravel()], True)

    apply_mask = (touches_target & has_votes)[label_map]
    corrected[apply_mask] = winners[label_map][apply_mask]
    return corrected


def parse_correction_targets(raw: list) -> List[Tuple[FrozenSet[int], Path]]:
    """Parse the ``targets`` config list shared by SAM/SLICO correction configs.

    Args:
        raw: List of dicts with keys ``classes`` (list of int) and
            ``output_dir`` (str).

    Returns:
        List of ``(classes_to_correct, output_dir)`` tuples, ``classes_to_correct``
        as a ``frozenset`` and ``output_dir`` as a ``Path``.
    """
    parsed = []
    for t in raw:
        classes = frozenset(int(c) for c in t["classes"])
        out_dir = Path(t["output_dir"])
        parsed.append((classes, out_dir))
    return parsed


def chunk_cache_key(tile_name: str, window) -> str:
    """Build the cache key for a tile/window pair.

    Args:
        tile_name: Mask tile path or filename.
        window: Object with ``row_off``, ``col_off``, ``height``, ``width``
            attributes (e.g. ``rasterio.windows.Window``).

    Returns:
        Cache key string, e.g. ``"tile_001_r10_c20_h64_w128"``.
    """
    stem = Path(tile_name).stem
    return (
        f"{stem}_r{window.row_off}_c{window.col_off}_h{window.height}_w{window.width}"
    )
