# -*- coding: utf-8 -*-
"""
/***************************************************************************
 pytorch_segmentation_models_trainer
                              -------------------
        begin                : 2026-10-21
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

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

_NODATA = -128


class AefCacheLoader:
    """Loads AlphaEarth Foundations (AEF) embeddings from a per-tile
    ``.npz`` cache for the ``BaseGFSSMethod.wants_aux``/``set_aux``
    mechanism (see ``few_shot/base_method.py`` and
    ``few_shot/methods/hisplit.py``'s ``q: linear_aef_gate``). Pure I/O
    and tensor alignment — this class has no knowledge of how a method
    uses the tensor it returns.

    Cache format (produced by an external pipeline, not part of this
    repo): ``<cache_dir>/<tile_id>.npz``, key ``'aef'``, shape
    ``(n_bands, h, w)`` ``int8``, ``-128`` = nodata. ``tile_id`` is the
    integer stem of the dataset's ``mask_path`` for that window (e.g.
    ``"42.tif"`` → ``42``).

    Args:
        cache_dir: Directory of ``<tile_id>.npz`` files.
        n_bands: Channel count of the cached embedding (``64`` for AEF).
    """

    def __init__(self, cache_dir: str, n_bands: int = 64) -> None:
        self.cache_dir = Path(cache_dir)
        self.n_bands = n_bands

    def _load_one(self, mask_path: str, size: Tuple[int, int]) -> torch.Tensor:
        """``(n_bands, *size)`` float32 in ``[-1, 1]``, ``0`` where nodata
        or the tile has no cache file (neutral fallback — never raises on
        a missing tile, since cache coverage is not guaranteed to be
        100% for every dataset)."""
        tile_id = Path(mask_path).stem
        f = self.cache_dir / f"{tile_id}.npz"
        if not f.exists():
            return torch.zeros(self.n_bands, *size)
        raw = np.load(f)["aef"]
        t = torch.from_numpy(raw).float()
        nodata = t == _NODATA
        aef = (t / 127.0).masked_fill(nodata, 0.0)
        if tuple(aef.shape[-2:]) != tuple(size):
            aef = F.interpolate(aef.unsqueeze(0), size=size, mode="nearest").squeeze(0)
        return aef

    def load_batch(
        self,
        mask_paths: List[str],
        device: torch.device,
        size: Tuple[int, int],
    ) -> Optional[torch.Tensor]:
        """``(B, n_bands, *size)`` tensor on ``device``, or ``None`` if
        ``mask_paths`` is empty."""
        if not mask_paths:
            return None
        tiles = [self._load_one(p, size) for p in mask_paths]
        return torch.stack(tiles).to(device)
