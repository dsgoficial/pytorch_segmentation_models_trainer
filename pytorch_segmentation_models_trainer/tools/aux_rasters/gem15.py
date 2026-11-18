# -*- coding: utf-8 -*-
"""
/***************************************************************************
 pytorch_segmentation_models_trainer
                              -------------------
        begin                : 2026-11-13
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

GEM-15 (Google Research): near-global surface (DSM) and terrain (DTM)
elevation at 15 m, from sub-meter optical stereo (mostly 2022-2024).
CC-BY 4.0, attribution "This dataset is produced by Google". Citation:
Batchu et al. (2026), EarthArXiv preprint, doi:10.31223/X55N75.

Data layout (read from the dataset's own static STAC catalog,
``https://storage.googleapis.com/climate-and-sustainability/gem-15/stac/``):
one item per 1x1 degree cell, id ``wm51sm21`` = west edge -51, south edge
-21, grouped in 10x10 degree folders (``m30_m60`` = south -30, west -60).
Each item has a ``dsm`` COG (5 float32 bands: dsm, dsm_percentile_5,
dsm_percentile_50, dsm_percentile_95, dsm_std) and a ``dtm`` COG (1 band),
both in the cell's UTM zone, NaN = nodata (water, low-texture surfaces).
"""

from __future__ import annotations

import functools
import logging
import math
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from pytorch_segmentation_models_trainer.tools.aux_rasters.common import (
    Bounds,
    read_to_tile_grid,
)

logger = logging.getLogger(__name__)

STAC_ROOT = "https://storage.googleapis.com/climate-and-sustainability/gem-15/stac/gem-15"

#: Band order of the ``dsm`` COG, as declared in each STAC item.
DSM_BANDS = ("dsm", "dsm_percentile_5", "dsm_percentile_50", "dsm_percentile_95", "dsm_std")

#: Channels produced by :func:`terrain_features`, in order.
FEATURE_NAMES = ("ndsm", "ndsm_p95", "dsm_std", "relief_p95_p5", "slope_deg", "tpi", "valid")


def _signed(v: int) -> str:
    """``-51 -> 'm51'``, ``0 -> 'p0'``, ``12 -> 'p12'`` (GEM-15's naming)."""
    return f"m{abs(v)}" if v < 0 else f"p{v}"


def cell_id(lon: float, lat: float) -> Tuple[str, str]:
    """``(group, cell)`` of the 1x1 degree GEM-15 cell containing a point,
    e.g. ``(-50.5, -20.5) -> ("m30_m60", "wm51sm21")``."""
    west, south = math.floor(lon), math.floor(lat)
    group = f"{_signed(10 * math.floor(south / 10))}_{_signed(10 * math.floor(west / 10))}"
    return group, f"w{_signed(west)}s{_signed(south)}"


def cells_for_bounds(bounds_wgs84: Bounds) -> List[Tuple[str, str]]:
    """Every ``(group, cell)`` a ``(west, south, east, north)`` box touches
    (usually one; up to four for a tile straddling cell edges)."""
    west, south, east, north = bounds_wgs84
    eps = 1e-9  # a box ending exactly on an integer degree does not enter the next cell
    out = []
    for lat in range(math.floor(south), math.floor(north - eps) + 1):
        for lon in range(math.floor(west), math.floor(east - eps) + 1):
            out.append(cell_id(lon + 0.5, lat + 0.5))
    return out


@functools.lru_cache(maxsize=4096)
def item_assets(group: str, cell: str) -> Optional[Dict[str, str]]:
    """``{"dsm": href, "dtm": href}`` from the cell's STAC item, or ``None``
    when the item does not exist (cell outside GEM-15's coverage). Cached
    per process: tiles of one batch share a few cells."""
    import requests

    url = f"{STAC_ROOT}/{group}/{cell}/{cell}.json"
    resp = requests.get(url, timeout=60)
    if resp.status_code == 404:
        return None
    resp.raise_for_status()
    assets = resp.json()["assets"]
    return {k: assets[k]["href"] for k in ("dsm", "dtm")}


def pixel_size_m(bounds_wgs84: Bounds, size: Tuple[int, int]) -> Tuple[float, float]:
    """``(dy, dx)`` ground size in meters of one pixel of the tile grid
    (spherical approximation at the box's mid latitude; enough for slope)."""
    west, south, east, north = bounds_wgs84
    m_per_deg = 111_320.0
    lat_mid = math.radians((south + north) / 2)
    dy = (north - south) * m_per_deg / size[0]
    dx = (east - west) * m_per_deg * math.cos(lat_mid) / size[1]
    return dy, dx


def download_tile(
    bounds_wgs84: Bounds,
    out_path: Path,
    size: Tuple[int, int] = (64, 64),
) -> bool:
    """Writes one tile's GEM-15 window as a per-tile ``.npz`` cache.

    Both COGs are bilinearly resampled (elevation is continuous) from 15 m
    UTM onto the tile's own WGS84 grid, ``size`` pixels (``64x64`` = the
    10 m grid of a 256 px / 2.5 m tile, matching the AEF/OlmoEarth caches).

    Cache keys: ``dsm`` ``(5, *size)`` float32 in :data:`DSM_BANDS` order,
    ``dtm`` ``(*size)`` float32, NaN = nodata; ``pixel_size_m`` ``(2,)``
    float32 ``(dy, dx)`` for :func:`terrain_features`.

    Args:
        bounds_wgs84: ``(west, south, east, north)`` in degrees.
        out_path: Output ``.npz`` path.
        size: Output grid ``(rows, cols)``.

    Returns:
        ``True`` if written; ``False`` (nothing written) when no GEM-15 cell
        covers the tile or the window is entirely nodata.
    """
    cells = [a for a in (item_assets(*c) for c in cells_for_bounds(bounds_wgs84)) if a]
    if not cells:
        logger.warning("GEM-15: no cell covers %s, skipping %s.", bounds_wgs84, out_path)
        return False
    dsm = read_to_tile_grid(
        [c["dsm"] for c in cells], bounds_wgs84, size, np.float32, np.nan, "bilinear"
    )
    dtm = read_to_tile_grid(
        [c["dtm"] for c in cells], bounds_wgs84, size, np.float32, np.nan, "bilinear"
    )[0]
    if np.isnan(dtm).all() and np.isnan(dsm).all():
        logger.warning("GEM-15: window fully nodata, skipping %s.", out_path)
        return False
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        dsm=dsm,
        dtm=dtm,
        pixel_size_m=np.asarray(pixel_size_m(bounds_wgs84, size), dtype=np.float32),
    )
    return True


def _box_mean(a: np.ndarray, radius: int) -> np.ndarray:
    """Mean over a ``(2r+1)^2`` window, edge-replicated (NaN-free input)."""
    k = 2 * radius + 1
    padded = np.pad(a, radius, mode="edge")
    c = padded.cumsum(0).cumsum(1)
    c = np.pad(c, ((1, 0), (1, 0)))
    s = c[k:, k:] - c[:-k, k:] - c[k:, :-k] + c[:-k, :-k]
    return s / (k * k)


def terrain_features(
    dsm: np.ndarray,
    dtm: np.ndarray,
    pixel_size: Sequence[float],
    tpi_radius: int = 5,
) -> np.ndarray:
    """Derived terrain/structure channels from one tile's GEM-15 cache.

    Relative quantities only: the dataset reports regional mean offsets of
    up to ~5 m against airborne lidar, so absolute heights are not used.

    Channels (:data:`FEATURE_NAMES`):
        ``ndsm`` = dsm - dtm (height above ground); ``ndsm_p95`` =
        dsm_percentile_95 - dtm; ``dsm_std`` (sub-15 m surface roughness);
        ``relief_p95_p5`` = p95 - p5; ``slope_deg`` (from dtm);
        ``tpi`` = dtm - mean(dtm over a ``(2r+1)^2`` window);
        ``valid`` = 1 where both dsm and dtm are valid.
    Invalid pixels are 0 in every channel except ``valid``.

    Args:
        dsm: ``(5, h, w)`` in :data:`DSM_BANDS` order.
        dtm: ``(h, w)``.
        pixel_size: ``(dy, dx)`` in meters.
        tpi_radius: TPI window radius in pixels.

    Returns:
        ``(7, h, w)`` float32.
    """
    valid = ~(np.isnan(dtm) | np.isnan(dsm).any(0))
    fill = float(np.nanmedian(dtm)) if valid.any() else 0.0
    dtm_f = np.where(valid, dtm, fill)
    dsm_f = np.where(valid[None], dsm, fill)
    dy, dx = float(pixel_size[0]), float(pixel_size[1])
    gy, gx = np.gradient(dtm_f, dy, dx)
    feats = np.stack(
        [
            dsm_f[0] - dtm_f,
            dsm_f[3] - dtm_f,
            dsm_f[4],
            dsm_f[3] - dsm_f[1],
            np.degrees(np.arctan(np.hypot(gx, gy))),
            dtm_f - _box_mean(dtm_f, tpi_radius),
        ]
    ).astype(np.float32)
    feats[:, ~valid] = 0.0
    return np.concatenate([feats, valid[None].astype(np.float32)])


class Gem15CacheLoader:
    """Loads :func:`terrain_features` from per-tile GEM-15 caches (see
    :func:`download_tile`) with the same ``load_batch(mask_paths, device,
    size)`` interface as
    :class:`~pytorch_segmentation_models_trainer.few_shot.aef_cache.AefCacheLoader`.

    ``tile_id`` is the integer stem of ``mask_path``. A tile with no cache
    file returns all zeros (``valid`` = 0), never raises.

    Args:
        cache_dir: Directory of ``<tile_id>.npz`` files.
        tpi_radius: Passed to :func:`terrain_features`.

    Example (Hydra)::

        aux_loader:
          _target_: pytorch_segmentation_models_trainer.tools.aux_rasters.gem15.Gem15CacheLoader
          cache_dir: /data/cache_gem15
          tpi_radius: 5
    """

    n_bands = len(FEATURE_NAMES)

    def __init__(self, cache_dir: str, tpi_radius: int = 5, **kwargs) -> None:
        self.cache_dir = Path(cache_dir)
        self.tpi_radius = tpi_radius

    def _load_one(self, mask_path: str, size: Tuple[int, int]) -> torch.Tensor:
        f = self.cache_dir / f"{Path(mask_path).stem}.npz"
        if not f.exists():
            return torch.zeros(self.n_bands, *size)
        z = np.load(f)
        feats = torch.from_numpy(
            terrain_features(z["dsm"], z["dtm"], z["pixel_size_m"], self.tpi_radius)
        )
        if tuple(feats.shape[-2:]) != tuple(size):
            feats = F.interpolate(feats.unsqueeze(0), size=size, mode="nearest").squeeze(0)
        return feats

    def load_batch(
        self, mask_paths: List[str], device, size: Tuple[int, int]
    ) -> Optional[torch.Tensor]:
        """``(B, 7, *size)`` float32 on ``device``; ``None`` if empty."""
        if not mask_paths:
            return None
        return torch.stack([self._load_one(p, size) for p in mask_paths]).to(device)
