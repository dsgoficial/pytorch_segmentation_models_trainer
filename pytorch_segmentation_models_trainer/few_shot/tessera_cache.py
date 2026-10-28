# -*- coding: utf-8 -*-
"""
/***************************************************************************
 pytorch_segmentation_models_trainer
                              -------------------
        begin                : 2026-10-28
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
import math
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)

_DATA_BASE = "https://data.source.coop"
_MANIFEST_PATH = "tessera/tessera/npy/v1.1-cam/manifest.parquet"
_CELL_DEGREES = 0.1
_NODATA = -128


def dequantize_tessera(quantized: np.ndarray, scale: np.ndarray) -> np.ndarray:
    """Tessera's own dequantization: ``v = quantized * scale`` (the dataset
    stores int8 values together with a per-pixel float32 scale factor,
    **not** a fixed linear or sign-sqrt constant like AlphaEarth/OlmoEarth --
    confirmed against the live ``geotessera`` package's own
    ``dequantize_embedding`` and a real downloaded tile, see the companion
    research project's analysis doc). ``scale`` broadcasts over the last
    (band) axis if ``quantized`` has one more dimension than ``scale``.
    Non-finite scale (nodata, e.g. ocean/no-coverage pixels) becomes ``0``
    -- matches :class:`~.aef_cache.AefCacheLoader`'s zero-fallback
    convention, never ``NaN`` propagating into the model.
    """
    q = quantized.astype(np.float32)
    s = scale.astype(np.float32)
    if s.ndim == q.ndim - 1:
        s = s[..., np.newaxis]
    out = q * s
    out[~np.isfinite(out)] = 0.0
    return out


def _cell_center(lon: float, lat: float) -> Tuple[float, float]:
    """Snaps a point to the center of its 0.1x0.1 degree Tessera grid cell
    (cell centers sit at .x5 offsets, e.g. -51.45, -30.85 -- the dataset's
    own tiling convention, confirmed against the live manifest)."""
    cell_lon = round(math.floor(lon * 10) / 10 + _CELL_DEGREES / 2, 2)
    cell_lat = round(math.floor(lat * 10) / 10 + _CELL_DEGREES / 2, 2)
    return cell_lon, cell_lat


def open_manifest(cache_path: Optional[Path] = None, year: Optional[int] = None):
    """Downloads (or reuses a local copy of) the Tessera tile manifest --
    ~1.94M rows, one per (lon, lat, year) tile actually published, with
    ``grid_path``/``scales_path`` pointing at the data -- **not** the
    ``geotessera`` package's own registry fetch, which hung indefinitely
    from the companion research project's environment even though a plain
    HTTPS GET to the same host completed in ~5s (confirmed directly; root
    cause not identified, worked around by never calling into
    ``geotessera``'s HTTP client at runtime). Pass ``cache_path`` to avoid
    re-downloading the ~59MB file on every call (mirrors
    :func:`~.olmoearth_cache.open_store`'s ``root`` reuse pattern, just for
    the manifest instead of a live store connection). ``year`` pre-filters
    the returned frame (optional, saves memory for repeated single-year use).
    """
    import pandas as pd
    import requests

    if cache_path is not None and Path(cache_path).exists():
        df = pd.read_parquet(cache_path)
    else:
        url = f"{_DATA_BASE}/{_MANIFEST_PATH}"
        resp = requests.get(url, timeout=120)
        resp.raise_for_status()
        if cache_path is not None:
            Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
            Path(cache_path).write_bytes(resp.content)
            df = pd.read_parquet(cache_path)
        else:
            import io

            df = pd.read_parquet(io.BytesIO(resp.content))
    if year is not None:
        df = df[df["year"] == year]
    return df


def fetch_cell(
    grid_path: str,
    scales_path: str,
    cell_cache_dir: Optional[Path] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Downloads (or reuses a cached copy of) one full Tessera grid cell's
    quantized embedding ``(H, W, n_bands)`` int8 and its per-pixel scale
    ``(H, W)`` float32 -- a full cell is ~130MB (0.1x0.1 degree at ~10m),
    so ``cell_cache_dir`` lets a caller downloading many project tiles that
    share a cell (confirmed: 607 project tiles need only 308 unique cells)
    pay the download once per cell, not once per tile. Direct HTTPS GET
    (``s3://`` paths from the manifest translated to the public
    ``data.source.coop`` endpoint), same reasoning as :func:`open_manifest`
    for not using ``geotessera``'s own client.
    """
    import requests

    def _get(url: str, cache_file: Optional[Path]) -> bytes:
        if cache_file is not None and cache_file.exists():
            return cache_file.read_bytes()
        http_url = url.replace("s3://", f"{_DATA_BASE}/")
        resp = requests.get(http_url, timeout=120)
        resp.raise_for_status()
        if cache_file is not None:
            cache_file.parent.mkdir(parents=True, exist_ok=True)
            cache_file.write_bytes(resp.content)
        return resp.content

    import io

    grid_cache = Path(cell_cache_dir) / Path(grid_path).name if cell_cache_dir else None
    scales_cache = Path(cell_cache_dir) / Path(scales_path).name if cell_cache_dir else None
    quant = np.load(io.BytesIO(_get(grid_path, grid_cache)))
    scale = np.load(io.BytesIO(_get(scales_path, scales_cache)))
    return quant, scale


def download_tile(
    bounds_wgs84: Tuple[float, float, float, float],
    year: int,
    out_path: Path,
    size: Tuple[int, int] = (64, 64),
    n_bands: int = 128,
    manifest=None,
    cell_cache_dir: Optional[Path] = None,
) -> bool:
    """Downloads one tile's Tessera window and saves it as a per-tile
    ``.npz`` cache (keys ``'tessera'`` int8 ``(n_bands, *size)`` and
    ``'tessera_scale'`` float32 ``(*size,)`` -- kept quantized, dequantize
    with :func:`dequantize_tessera` at load time, same reasoning as
    :func:`~.olmoearth_cache.download_tile`).

    Tessera's grid is already native WGS84 degrees (unlike OlmoEarth's
    UTM-zone store), so this crops the requested ``bounds_wgs84`` window
    directly out of the cell array by linear pixel indexing -- no CRS
    reprojection needed -- then nearest-resizes to ``size`` if the crop
    doesn't land exactly on it.

    Args:
        bounds_wgs84: ``(west, south, east, north)`` in WGS84 degrees.
        year: Calendar year (2017-2025 published; others return ``False``).
        out_path: Where to write the per-tile ``.npz``.
        size: Output grid ``(rows, cols)``.
        n_bands: ``1``-``128`` (Tessera's native width; ``128`` default,
            every dimension the model emits).
        manifest: A frame from :func:`open_manifest`, reused across many
            calls (pass this when downloading more than a handful of
            tiles -- avoids re-reading the ~59MB manifest from disk/network
            every call).
        cell_cache_dir: Passed through to :func:`fetch_cell`.

    Returns:
        ``True`` if a cache file was written; ``False`` if the cell isn't
        in the manifest for ``year`` (no file written either way).
    """
    if manifest is None:
        manifest = open_manifest(year=year)

    west, south, east, north = bounds_wgs84
    cell_lon, cell_lat = _cell_center((west + east) / 2, (south + north) / 2)
    row = manifest[
        (manifest["year"] == year)
        & (np.isclose(manifest["lon"], cell_lon, atol=1e-6))
        & (np.isclose(manifest["lat"], cell_lat, atol=1e-6))
    ]
    if row.empty:
        logger.warning(
            "Tessera: cell (%.2f, %.2f) not in the manifest for year %d, skipping %s.",
            cell_lon,
            cell_lat,
            year,
            out_path,
        )
        return False
    row = row.iloc[0]

    quant, scale = fetch_cell(row["grid_path"], row["scales_path"], cell_cache_dir)
    H, W = quant.shape[0], quant.shape[1]
    cell_west, cell_east = cell_lon - _CELL_DEGREES / 2, cell_lon + _CELL_DEGREES / 2
    cell_south, cell_north = cell_lat - _CELL_DEGREES / 2, cell_lat + _CELL_DEGREES / 2

    col0 = max(int(round((west - cell_west) / _CELL_DEGREES * W)), 0)
    col1 = min(int(round((east - cell_west) / _CELL_DEGREES * W)), W)
    row0 = max(int(round((cell_north - north) / _CELL_DEGREES * H)), 0)
    row1 = min(int(round((cell_north - south) / _CELL_DEGREES * H)), H)
    if col1 <= col0 or row1 <= row0:
        logger.warning("Tessera: empty crop for bounds %s, skipping %s.", bounds_wgs84, out_path)
        return False

    crop_q = quant[row0:row1, col0:col1, :n_bands]  # (h, w, n_bands) -- channel-last,
    crop_s = scale[row0:row1, col0:col1]  # (h, w)      same layout Tessera itself uses

    if crop_q.shape[:2] != tuple(size):
        q_chw = torch.from_numpy(crop_q.astype(np.float32)).permute(2, 0, 1).unsqueeze(0)
        q_chw = F.interpolate(q_chw, size=size, mode="nearest").squeeze(0)
        crop_q = q_chw.permute(1, 2, 0).numpy()  # back to (h, w, n_bands)
        s_t = torch.from_numpy(crop_s.astype(np.float32)).unsqueeze(0).unsqueeze(0)
        crop_s = F.interpolate(s_t, size=size, mode="nearest").squeeze(0).squeeze(0).numpy()

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path, tessera=crop_q.astype(np.int8), tessera_scale=crop_s.astype(np.float32)
    )
    return True


class TesseraCacheLoader:
    """Loads Tessera embeddings from a per-tile ``.npz`` cache (see
    :func:`download_tile`) for the ``BaseGFSSMethod.wants_aux``/
    ``set_aux`` mechanism -- a drop-in alternative to
    :class:`~.aef_cache.AefCacheLoader`/
    :class:`~.olmoearth_cache.OlmoEarthCacheLoader` (identical constructor
    and :meth:`load_batch` signature).

    Cache format: ``<cache_dir>/<tile_id>.npz``, keys ``'tessera'``
    ``(h, w, n_bands)`` int8 (channel-last, Tessera's own native layout --
    **not** channel-first like AEF/OlmoEarth's cache; this loader converts
    to channel-first only on the tensor it returns) and ``'tessera_scale'``
    ``(h, w)`` float32 -- dequantized here with :func:`dequantize_tessera`
    (``quantized * scale``,
    **not** AlphaEarth's linear ``/127`` nor OlmoEarth's sign-sqrt scheme).
    ``tile_id`` is the integer stem of the dataset's ``mask_path`` for that
    window, exactly like the other two loaders.

    Data: TESSERA (University of Cambridge), 128-dim pixel embeddings from
    Sentinel-1+Sentinel-2 time series, 10m, global annual coverage
    2017-2025, CC0. Coverage confirmed directly against the live manifest
    for the companion research project's tiles: 606/607 present for 2024
    (1 isolated gap, not systemic) -- see the analysis doc, "Tessera —
    cobertura confirmada" (2026-10-28).

    Args:
        cache_dir: Directory of ``<tile_id>.npz`` files (see
            :func:`download_tile`).
        n_bands: Channel count of the cached embedding (``1``-``128``,
            whatever :func:`download_tile` was run with).
    """

    def __init__(self, cache_dir: str, n_bands: int = 128) -> None:
        self.cache_dir = Path(cache_dir)
        self.n_bands = n_bands

    def _load_one(self, mask_path: str, size: Tuple[int, int]) -> torch.Tensor:
        """``(n_bands, *size)`` float32, dequantized -- ``0`` where nodata
        or the tile has no cache file (neutral fallback, never raises)."""
        tile_id = Path(mask_path).stem
        f = self.cache_dir / f"{tile_id}.npz"
        if not f.exists():
            return torch.zeros(self.n_bands, *size)
        npz = np.load(f)
        dequant = dequantize_tessera(npz["tessera"], npz["tessera_scale"])  # (h, w, n_bands)
        emb = torch.from_numpy(dequant).permute(2, 0, 1)  # -> (n_bands, h, w)
        if tuple(emb.shape[-2:]) != tuple(size):
            emb = F.interpolate(emb.unsqueeze(0), size=size, mode="nearest").squeeze(0)
        return emb

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
