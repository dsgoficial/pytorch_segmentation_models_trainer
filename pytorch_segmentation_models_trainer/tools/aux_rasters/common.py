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
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Callable, Dict, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

logger = logging.getLogger(__name__)

Bounds = Tuple[float, float, float, float]

#: GDAL options for anonymous, header-only HTTP reads of public COGs
#: (avoids a directory listing request per file open).
GDAL_HTTP_ENV = dict(
    GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR",
    CPL_VSIL_CURL_ALLOWED_EXTENSIONS=".tif,.tiff",
    GDAL_HTTP_MAX_RETRY="5",
    GDAL_HTTP_RETRY_DELAY="2",
)


def gdal_path(uri: str) -> str:
    """Path GDAL can open: ``/vsicurl/<uri>`` for an ``http(s)://`` URI, the
    URI itself otherwise (a local file, already a ``/vsi...`` path)."""
    if uri.startswith(("http://", "https://")):
        return f"/vsicurl/{uri}"
    return uri


def read_to_tile_grid(
    sources: Sequence[str],
    bounds_wgs84: Bounds,
    size: Tuple[int, int],
    dtype,
    nodata,
    resampling: str = "nearest",
    bands: Optional[Sequence[int]] = None,
) -> np.ndarray:
    """Reads one or more source rasters onto a tile's own WGS84 grid.

    Each source is warped (``rasterio.vrt.WarpedVRT``, which reads only the
    COG blocks that intersect the tile) to the grid
    ``from_bounds(*bounds_wgs84, size[1], size[0])`` in EPSG:4326, the same
    "reproject the shared source onto each tile's own grid" convention as
    :func:`~pytorch_segmentation_models_trainer.few_shot.olmoearth_cache.download_tile`.
    When several sources cover the tile (a tile straddling two source
    files), a later source only fills the pixels still at ``nodata``.

    Args:
        sources: Local paths or ``http(s)://`` URIs of the source rasters.
        bounds_wgs84: ``(west, south, east, north)`` in degrees.
        size: Output ``(rows, cols)``.
        dtype: Output numpy dtype.
        nodata: Output fill value (``np.nan`` allowed for float dtypes).
        resampling: Any ``rasterio.enums.Resampling`` name (``"nearest"``
            for categorical/quantized data, ``"bilinear"`` for continuous).
        bands: 1-based band indexes to read (default: all bands of the
            first source).

    Returns:
        ``(n_bands, rows, cols)`` array of ``dtype``; all ``nodata`` where no
        source has valid data.
    """
    import rasterio
    from rasterio.enums import Resampling
    from rasterio.transform import from_bounds
    from rasterio.vrt import WarpedVRT

    rows, cols = size
    transform = from_bounds(*bounds_wgs84, cols, rows)
    out = None
    with rasterio.Env(**GDAL_HTTP_ENV):
        for src_uri in sources:
            with rasterio.open(gdal_path(src_uri)) as src:
                idx = list(bands) if bands is not None else list(range(1, src.count + 1))
                src_nodata = src.nodata if src.nodata is not None else nodata
                with WarpedVRT(
                    src,
                    crs="EPSG:4326",
                    transform=transform,
                    width=cols,
                    height=rows,
                    resampling=Resampling[resampling],
                    src_nodata=src_nodata,
                    nodata=nodata,
                ) as vrt:
                    data = vrt.read(idx).astype(dtype)
            if out is None:
                out = data
                continue
            missing = _is_nodata(out, nodata)
            out[missing] = data[missing]
    if out is None:
        raise ValueError("read_to_tile_grid: no source given.")
    return out


def _is_nodata(arr: np.ndarray, nodata) -> np.ndarray:
    """Boolean mask of ``nodata`` pixels (``NaN``-aware)."""
    if isinstance(nodata, float) and np.isnan(nodata):
        return np.isnan(arr)
    return arr == nodata


def tiles_from_manifest(
    manifest: pd.DataFrame,
    id_col: str = "id",
    bounds_cols: Sequence[str] = ("bbox_west", "bbox_south", "bbox_east", "bbox_north"),
) -> Dict[str, Bounds]:
    """``{tile_id: (west, south, east, north)}`` from a manifest table with
    one row per tile (the companion research project's ``manifest.csv``
    layout by default)."""
    missing = [c for c in (id_col, *bounds_cols) if c not in manifest.columns]
    if missing:
        raise ValueError(f"manifest is missing columns: {missing}")
    # zip over columns, not iterrows: iterrows upcasts an int id to float
    # when the bounds are float, which would name the cache "1.0.npz".
    ids = manifest[id_col].astype(str)
    bounds = zip(*(manifest[c].astype(float) for c in bounds_cols))
    return dict(zip(ids, (tuple(b) for b in bounds)))


def download_tiles(
    tiles: Dict[str, Bounds],
    out_dir: Path,
    download_one: Callable[[Bounds, Path], bool],
    max_workers: int = 8,
    overwrite: bool = False,
    desc: str = "tiles",
) -> Dict[str, list]:
    """Runs ``download_one(bounds, out_dir / f"{tile_id}.npz")`` for every
    tile, threaded (the work is HTTP-bound) and resumable (tiles whose
    ``.npz`` already exists are skipped unless ``overwrite``).

    Args:
        tiles: ``{tile_id: bounds_wgs84}`` (see :func:`tiles_from_manifest`).
        out_dir: Cache directory (created if needed).
        download_one: Writes one tile and returns ``True``, or returns
            ``False`` when the source has no data there (nothing written).
            An exception is caught and recorded as a failure for that tile.
        max_workers: Thread count.
        overwrite: Re-download tiles that already have a cache file.
        desc: Progress-bar label.

    Returns:
        ``{"written": [...], "skipped": [...], "no_data": [...],
        "failed": [...]}`` lists of tile ids (``failed`` holds
        ``(tile_id, error message)`` pairs).
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    report = {"written": [], "skipped": [], "no_data": [], "failed": []}
    todo = {}
    for tile_id, bounds in tiles.items():
        path = out_dir / f"{tile_id}.npz"
        if path.exists() and not overwrite:
            report["skipped"].append(tile_id)
        else:
            todo[tile_id] = (bounds, path)

    with ThreadPoolExecutor(max_workers=max(1, max_workers)) as pool:
        futures = {
            pool.submit(download_one, bounds, path): tile_id
            for tile_id, (bounds, path) in todo.items()
        }
        for fut in tqdm(as_completed(futures), total=len(futures), desc=desc):
            tile_id = futures[fut]
            try:
                ok = fut.result()
            except Exception as exc:  # recorded, never aborts the batch
                logger.warning("%s: tile %s failed: %s", desc, tile_id, exc)
                report["failed"].append((tile_id, str(exc)))
                continue
            report["written" if ok else "no_data"].append(tile_id)
    return report
