# -*- coding: utf-8 -*-
"""
/***************************************************************************
 pytorch_segmentation_models_trainer
                              -------------------
        begin                : 2026-10-23
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

import logging
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)

_STORE_URL = "s3://olmoearth/olmoearth-v1-3-embeddings/embeddings.zarr"
_ENDPOINT = "https://data.source.coop"
_NODATA = -128


def utm_zone(lon: float) -> int:
    """UTM zone (1-60) of a WGS84 longitude, as the store groups its data."""
    return int((lon + 180) // 6) + 1


def dequantize_olmoearth(q: np.ndarray) -> np.ndarray:
    """OlmoEarth v1.3's own signed-square-root dequantization
    (``v = sign(q)·(|q|/127.5)²``) -- do **not** use
    :class:`AefCacheLoader`'s linear ``/127`` scheme for this product, the
    two are different quantization schemes. ``-128`` (nodata) becomes ``0``,
    matching :class:`AefCacheLoader`'s zero-fallback convention (a neutral
    value the gate's ``nn.Linear`` can learn to discount)."""
    v = q.astype(np.float32)
    out = (np.abs(v) / 127.5) ** 2.0 * np.sign(v)
    out[q == _NODATA] = 0.0
    return out


def download_tile(
    bounds_wgs84: Tuple[float, float, float, float],
    year: int,
    out_path: Path,
    size: Tuple[int, int] = (64, 64),
    n_bands: int = 64,
) -> bool:
    """Downloads one tile's OlmoEarth v1.3 window and saves it as a per-tile
    ``.npz`` cache (key ``'olmoearth'``, ``(n_bands, *size)`` int8, the raw
    quantized values as stored -- dequantize with :func:`dequantize_olmoearth`
    at load time; kept quantized here to keep the cache file 4x smaller than
    pre-dequantized float32).

    Reprojects the requested window from the store's own UTM zone grid into
    the tile's own WGS84-bounds grid with ``rasterio.warp.reproject``
    (nearest), the same "reproject the shared source onto each tile's own
    grid" pattern already used elsewhere in the companion research project
    (``MBTilesMaskWindowedDataset``) — not a naive pixel crop, which would
    drift for a tile whose native grid isn't exactly the store's zone grid.

    Args:
        bounds_wgs84: ``(west, south, east, north)`` in WGS84 degrees.
        year: Calendar year. The store only has 2017-2025; any other value
            returns ``False`` (nothing written) after logging a warning.
        out_path: Where to write ``<tile_id>.npz`` (``out_path`` itself).
        size: Output grid ``(rows, cols)`` -- the tile's own 10 m grid
            (``64, 64`` for a 256 px/2.5 m tile, matching
            :class:`AefCacheLoader`'s convention for the same tiles).
        n_bands: ``64`` (default; nested/Matryoshka-complete on its own, the
            width already used throughout the companion project's
            ``aef_dim``) or ``128`` (every dimension the model emits).

    Returns:
        ``True`` if a cache file was written; ``False`` if ``year`` is out
        of range or the window is entirely nodata (outside OlmoEarth's
        coverage for that year/location) -- no file is written either way,
        so a caller can tell "no data" apart from "not attempted yet" by
        checking whether ``out_path`` exists.
    """
    import rasterio
    from rasterio.transform import from_bounds
    from rasterio.warp import Resampling, reproject
    import zarr
    from obstore.store import S3Store
    from pyproj import Transformer

    zarr.config.set({"array.sharding_coalesce_max_gap_bytes": 0})
    s3 = S3Store.from_url(_STORE_URL, endpoint=_ENDPOINT, skip_signature=True)
    store = zarr.storage.ObjectStore(s3, read_only=True)
    root = zarr.open_group(store, mode="r")

    west, south, east, north = bounds_wgs84
    zone = root[f"utm{utm_zone((west + east) / 2):02d}"]
    years = zone["time"][:]
    match = np.where(years == year)[0]
    if len(match) == 0:
        logger.warning(
            "OlmoEarth: year %d not in the store (2017-2025 only), skipping %s.",
            year,
            out_path,
        )
        return False
    t = int(match[0])

    src_crs = zone.attrs["proj:code"]
    transformer = Transformer.from_crs("EPSG:4326", src_crs, always_xy=True)
    xs, ys = transformer.transform([west, east, west, east], [south, south, north, north])
    x0, x1 = min(xs), max(xs)
    y0, y1 = min(ys), max(ys)

    a, _, c, _, e, f = zone.attrs["spatial:transform"]
    col0 = max(int((x0 - c) / a) - 1, 0)
    col1 = int((x1 - c) / a) + 2
    row0 = max(int((y1 - f) / e) - 1, 0)  # e is negative (north-up grid)
    row1 = int((y0 - f) / e) + 2

    window = np.asarray(zone["embeddings"][t, :n_bands, row0:row1, col0:col1])
    if (window == _NODATA).all():
        logger.warning(
            "OlmoEarth: tile fully nodata for year %d, skipping %s.", year, out_path
        )
        return False

    src_transform = rasterio.transform.Affine(a, 0, c + col0 * a, 0, e, f + row0 * e)
    dst_transform = from_bounds(west, south, east, north, size[1], size[0])
    dst = np.full((n_bands, *size), _NODATA, dtype=np.int8)
    for b in range(n_bands):
        reproject(
            window[b],
            dst[b],
            src_transform=src_transform,
            src_crs=src_crs,
            dst_transform=dst_transform,
            dst_crs="EPSG:4326",
            resampling=Resampling.nearest,
            src_nodata=_NODATA,
            dst_nodata=_NODATA,
        )

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, olmoearth=dst)
    return True


class OlmoEarthCacheLoader:
    """Loads OlmoEarth v1.3 embeddings from a per-tile ``.npz`` cache (see
    :func:`download_tile`) for the ``BaseGFSSMethod.wants_aux``/``set_aux``
    mechanism -- a drop-in alternative to
    :class:`~pytorch_segmentation_models_trainer.few_shot.aef_cache.AefCacheLoader`
    (identical constructor and :meth:`load_batch` signature), so existing
    ``q: linear_aef_gate``/``aef_gate: true`` configs work unchanged, just
    pointing ``gfss.aef_cache_loader`` at this class and its own
    ``cache_dir`` instead of the default ``AefCacheLoader(aef_cache_dir)``.

    Cache format: ``<cache_dir>/<tile_id>.npz``, key ``'olmoearth'``,
    ``(n_bands, h, w)`` int8, ``-128`` = nodata -- OlmoEarth's own
    quantization, dequantized here with :func:`dequantize_olmoearth` (**not**
    AlphaEarth's linear ``/127`` scheme). ``tile_id`` is the integer stem of
    the dataset's ``mask_path`` for that window, exactly like
    :class:`AefCacheLoader`.

    Data: https://source.coop/olmoearth/olmoearth-v1-3-embeddings (Allen
    Institute for AI, OlmoEarth Artifact License — free use/redistribution
    for research, excludes military and extractive-industry use, requires
    attribution). Coverage per the dataset's own README: 2023-2025 global,
    2017-2022 "selected countries", being backfilled — confirmed directly
    against the live store (not just the README) for every year the
    companion research project's tiles actually use (2017, 2022, 2023,
    2025) in southern Brazil: real embeddings, not nodata, at 19 sampled
    tile centroids. 2016 is out of the store's range (2017-2025 only) and
    has no substitute here — :func:`download_tile` returns ``False`` for it.

    Args:
        cache_dir: Directory of ``<tile_id>.npz`` files (see
            :func:`download_tile`).
        n_bands: Channel count of the cached embedding (``64`` or ``128``,
            whatever :func:`download_tile` was run with).
    """

    def __init__(self, cache_dir: str, n_bands: int = 64) -> None:
        self.cache_dir = Path(cache_dir)
        self.n_bands = n_bands

    def _load_one(self, mask_path: str, size: Tuple[int, int]) -> torch.Tensor:
        """``(n_bands, *size)`` float32, dequantized -- ``0`` where nodata
        or the tile has no cache file (neutral fallback, never raises)."""
        tile_id = Path(mask_path).stem
        f = self.cache_dir / f"{tile_id}.npz"
        if not f.exists():
            return torch.zeros(self.n_bands, *size)
        raw = np.load(f)["olmoearth"]
        aef = torch.from_numpy(dequantize_olmoearth(raw))
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
