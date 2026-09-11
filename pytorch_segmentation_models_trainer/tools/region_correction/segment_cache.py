# -*- coding: utf-8 -*-
"""NPZ-based caches for region-generation outputs.

Two formats are needed because SAM AMG and SLICO produce structurally different
partitions:

- :class:`SegmentListCache` — a list of possibly-overlapping boolean masks
  (SAM AMG style, one array per region plus a confidence/area scalar each).
- :class:`LabelMapCache` — a single dense integer label map (SLIC/SLICO style,
  every pixel assigned to exactly one region). Storing thousands of superpixels
  as separate boolean arrays (SAM's format) would be far more expensive to write
  and read than one dense array of the same shape as the mask.
"""

from pathlib import Path
from typing import Dict, List, Optional

import numpy as np


class SegmentListCache:
    """NPZ-based cache for lists of (possibly overlapping) segment masks.

    Stores segment masks keyed by a string identifier. Each entry contains
    per-segment boolean arrays (``seg_N``), confidence scalars (``iou_N``), and
    area scalars (``area_N``).

    Args:
        cache_dir: Directory for NPZ files. Pass ``""`` or ``None`` to disable.

    Example::

        cache = SegmentListCache(cache_dir="/data/sam_cache")
        masks = cache.get("tile_r0_c0_h1024_w1024")
        if masks is None:
            masks = mask_generator.generate(image)
            cache.put("tile_r0_c0_h1024_w1024", masks)
    """

    def __init__(self, cache_dir: Optional[str]) -> None:
        self._enabled = bool(cache_dir)
        self._root = Path(cache_dir) if cache_dir else None

    def get(self, key: str) -> Optional[List[Dict]]:
        """Load cached segment masks for ``key``, or None if absent/invalid.

        Returns None when:
        - cache is disabled
        - the NPZ file does not exist
        - the file was written with an old format lacking ``iou_N`` arrays
        """
        if not self._enabled:
            return None
        path = self._root / f"{key}.npz"
        if not path.exists():
            return None

        data = np.load(path, allow_pickle=False)
        n = int(data["n_masks"])
        if n > 0 and "iou_0" not in data:
            return None

        masks = []
        for i in range(n):
            masks.append(
                {
                    "segmentation": data[f"seg_{i}"].astype(bool),
                    "predicted_iou": float(data[f"iou_{i}"]),
                    "area": int(data[f"area_{i}"]),
                }
            )
        return masks

    def put(self, key: str, masks: List[Dict]) -> None:
        """Persist segment masks to disk under ``key``.

        No-op when cache is disabled.

        Args:
            key: Cache key (should be unique per chunk/patch).
            masks: List of segment mask dicts (``segmentation``, ``predicted_iou``, ``area``).
        """
        if not self._enabled:
            return
        self._root.mkdir(parents=True, exist_ok=True)
        path = self._root / f"{key}.npz"
        arrays: Dict[str, np.ndarray] = {"n_masks": np.array(len(masks))}
        for i, m in enumerate(masks):
            arrays[f"seg_{i}"] = m["segmentation"].astype(np.uint8)
            arrays[f"iou_{i}"] = np.array(m["predicted_iou"])
            arrays[f"area_{i}"] = np.array(m["area"])
        np.savez_compressed(path, **arrays)


class LabelMapCache:
    """NPZ-based cache for dense integer segment label maps (e.g. SLIC output).

    Unlike :class:`SegmentListCache`, a label map is a single (H, W) array —
    every pixel already belongs to exactly one region — so one array is stored
    per key instead of one per region.

    Args:
        cache_dir: Directory for NPZ files. Pass ``""`` or ``None`` to disable.

    Example::

        cache = LabelMapCache(cache_dir="/data/slico_cache")
        label_map = cache.get("tile_r0_c0_h1024_w1024_n1000")
        if label_map is None:
            label_map = slic(image, n_segments=1000, slic_zero=True)
            cache.put("tile_r0_c0_h1024_w1024_n1000", label_map)
    """

    def __init__(self, cache_dir: Optional[str]) -> None:
        self._enabled = bool(cache_dir)
        self._root = Path(cache_dir) if cache_dir else None

    def get(self, key: str) -> Optional[np.ndarray]:
        """Load the cached label map for ``key``, or None if absent/disabled."""
        if not self._enabled:
            return None
        path = self._root / f"{key}.npz"
        if not path.exists():
            return None
        data = np.load(path, allow_pickle=False)
        return data["label_map"]

    def put(self, key: str, label_map: np.ndarray) -> None:
        """Persist a label map to disk under ``key``. No-op when disabled."""
        if not self._enabled:
            return
        self._root.mkdir(parents=True, exist_ok=True)
        path = self._root / f"{key}.npz"
        np.savez_compressed(path, label_map=label_map.astype(np.int32))
