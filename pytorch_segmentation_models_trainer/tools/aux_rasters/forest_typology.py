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

Forest Typology (ForTy) 2020 v1.0 (Google / Nature Trace): per-class
forest-type probabilities at 10 m for the year 2020, CC-BY 4.0. Citation:
Neumann et al. (2026), preprint, doi:10.31223/X58R27.

Data layout: the Earth Engine collection is backed by public COGs in
``gs://nature-trace/export_rasters/forest_typology_2020_v1_0/``, named
``raster_<GZD>_<...>.tif`` by MGRS grid-zone designator (e.g. ``22J``),
in UTM, 5 uint8 bands (probability x 250; 255 = nodata). There is no
spatial index, so :func:`build_index` lists the files of the needed zones
and reads each header once (cached to JSON).
"""

from __future__ import annotations

import json
import logging
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from pytorch_segmentation_models_trainer.tools.aux_rasters.common import (
    GDAL_HTTP_ENV,
    Bounds,
    gdal_path,
    read_to_tile_grid,
)

logger = logging.getLogger(__name__)

BUCKET = "nature-trace"
PREFIX = "export_rasters/forest_typology_2020_v1_0/"
LIST_URL = f"https://storage.googleapis.com/storage/v1/b/{BUCKET}/o"
PUBLIC_URL = f"https://storage.googleapis.com/{BUCKET}/"

#: Stored bands, in file order (class 6, "Other land", is the complement).
BANDS = (
    "PrimaryForest",
    "NaturallyRegeneratingForest",
    "PlantedForest",
    "PlantationForest",
    "TreeCropsAndAgroforestry",
)
CLASS_NAMES = BANDS + ("OtherLand",)
NODATA = 255
SCALE = 250.0

_LAT_BANDS = "CDEFGHJKLMNPQRSTUVWX"  # MGRS latitude bands, 8 degrees from -80


def gzd(lon: float, lat: float) -> str:
    """MGRS grid-zone designator of a point, e.g. ``(-50.5, -25.5) -> '22J'``.
    Standard zones only (the Norway/Svalbard exceptions are not handled)."""
    zone = int((lon + 180) // 6) + 1
    band = _LAT_BANDS[min(max(int((lat + 80) // 8), 0), len(_LAT_BANDS) - 1)]
    return f"{zone}{band}"


def gzds_for_bounds(bounds: Iterable[Bounds]) -> List[str]:
    """Sorted set of GZDs touched by the corners of every box."""
    out = set()
    for west, south, east, north in bounds:
        for lon in (west, east):
            for lat in (south, north):
                out.add(gzd(lon, lat))
    return sorted(out)


def list_files(zone: str) -> List[str]:
    """Object names of every ForTy COG of one GZD (GCS JSON API, paginated)."""
    import requests

    names, token = [], None
    while True:
        params = {"prefix": f"{PREFIX}raster_{zone}_", "maxResults": 1000}
        if token:
            params["pageToken"] = token
        resp = requests.get(LIST_URL, params=params, timeout=60)
        resp.raise_for_status()
        data = resp.json()
        names += [o["name"] for o in data.get("items", []) if o["name"].endswith(".tif")]
        token = data.get("nextPageToken")
        if not token:
            return names


def _file_bounds(uri: str) -> List[float]:
    """WGS84 ``[west, south, east, north]`` of one COG (header-only read)."""
    import rasterio
    from rasterio.warp import transform_bounds

    with rasterio.Env(**GDAL_HTTP_ENV):
        with rasterio.open(gdal_path(uri)) as src:
            l, b, r, t = src.bounds
            return list(
                transform_bounds(src.crs, "EPSG:4326", min(l, r), min(b, t), max(l, r), max(b, t))
            )


def build_index(
    zones: Sequence[str],
    cache_path: Optional[Path] = None,
    max_workers: int = 16,
) -> Dict[str, List[float]]:
    """``{uri: [west, south, east, north]}`` for every ForTy COG of ``zones``.

    Zones already in ``cache_path`` (JSON) are not re-listed; new ones are
    added and the cache rewritten.

    Args:
        zones: GZDs to index (see :func:`gzds_for_bounds`).
        cache_path: Optional JSON cache ``{"zones": [...], "files": {...}}``.
        max_workers: Threads for the header reads.

    Returns:
        The index covering (at least) ``zones``.
    """
    cache = {"zones": [], "files": {}}
    if cache_path is not None and Path(cache_path).exists():
        cache = json.loads(Path(cache_path).read_text())
    todo = [z for z in zones if z not in cache["zones"]]
    for zone in todo:
        uris = [PUBLIC_URL + n for n in list_files(zone)]
        with ThreadPoolExecutor(max_workers=max(1, max_workers)) as pool:
            bounds = list(
                tqdm(pool.map(_file_bounds, uris), total=len(uris), desc=f"ForTy index {zone}")
            )
        cache["files"].update(dict(zip(uris, bounds)))
        cache["zones"].append(zone)
    if todo and cache_path is not None:
        Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
        Path(cache_path).write_text(json.dumps(cache))
    return cache["files"]


def files_for_bounds(index: Dict[str, List[float]], bounds_wgs84: Bounds) -> List[str]:
    """URIs in ``index`` whose WGS84 footprint intersects the box."""
    west, south, east, north = bounds_wgs84
    return [
        uri
        for uri, (w, s, e, n) in index.items()
        if w < east and e > west and s < north and n > south
    ]


def download_tile(
    bounds_wgs84: Bounds,
    out_path: Path,
    index: Dict[str, List[float]],
    size: Tuple[int, int] = (64, 64),
) -> bool:
    """Writes one tile's ForTy window as a per-tile ``.npz`` cache.

    Nearest-neighbour resampling (the stored values are quantized
    probabilities) onto the tile's own WGS84 grid. Cache key ``forty``:
    ``(5, *size)`` uint8 in :data:`BANDS` order, 255 = nodata; convert with
    :func:`forty_probabilities`.

    Args:
        bounds_wgs84: ``(west, south, east, north)`` in degrees.
        out_path: Output ``.npz`` path.
        index: From :func:`build_index`, covering the tile's GZD(s).
        size: Output grid ``(rows, cols)``.

    Returns:
        ``True`` if written; ``False`` (nothing written) when no file
        covers the tile or the window is entirely nodata.
    """
    uris = files_for_bounds(index, bounds_wgs84)
    if not uris:
        logger.warning("ForTy: no file covers %s, skipping %s.", bounds_wgs84, out_path)
        return False
    raw = read_to_tile_grid(uris, bounds_wgs84, size, np.uint8, NODATA, "nearest")
    if (raw == NODATA).all():
        logger.warning("ForTy: window fully nodata, skipping %s.", out_path)
        return False
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, forty=raw)
    return True


def forty_probabilities(raw: np.ndarray) -> np.ndarray:
    """``(6, h, w)`` float32 probabilities from a ``(5, h, w)`` uint8 cache:
    the 5 stored classes ``/250`` plus ``OtherLand = 1 - sum`` (clipped at
    0). Nodata pixels (any band 255) are 0 in all 6 channels."""
    nodata = (raw == NODATA).any(0)
    p = raw.astype(np.float32) / SCALE
    other = np.clip(1.0 - p.sum(0, keepdims=True), 0.0, 1.0)
    out = np.concatenate([p, other])
    out[:, nodata] = 0.0
    return out


class ForTyCacheLoader:
    """Loads :func:`forty_probabilities` from per-tile ForTy caches (see
    :func:`download_tile`) with the same ``load_batch(mask_paths, device,
    size)`` interface as
    :class:`~pytorch_segmentation_models_trainer.few_shot.aef_cache.AefCacheLoader`.
    A tile with no cache file returns zeros, never raises.

    Args:
        cache_dir: Directory of ``<tile_id>.npz`` files.

    Example (Hydra)::

        aux_loader:
          _target_: pytorch_segmentation_models_trainer.tools.aux_rasters.forest_typology.ForTyCacheLoader
          cache_dir: /data/cache_forty
    """

    n_bands = len(CLASS_NAMES)

    def __init__(self, cache_dir: str, **kwargs) -> None:
        self.cache_dir = Path(cache_dir)

    def _load_one(self, mask_path: str, size: Tuple[int, int]) -> torch.Tensor:
        f = self.cache_dir / f"{Path(mask_path).stem}.npz"
        if not f.exists():
            return torch.zeros(self.n_bands, *size)
        p = torch.from_numpy(forty_probabilities(np.load(f)["forty"]))
        if tuple(p.shape[-2:]) != tuple(size):
            p = F.interpolate(p.unsqueeze(0), size=size, mode="nearest").squeeze(0)
        return p

    def load_batch(
        self, mask_paths: List[str], device, size: Tuple[int, int]
    ) -> Optional[torch.Tensor]:
        """``(B, 6, *size)`` float32 on ``device``; ``None`` if empty."""
        if not mask_paths:
            return None
        return torch.stack([self._load_one(p, size) for p in mask_paths]).to(device)
