# -*- coding: utf-8 -*-
"""Scan unique RGB colour tuples from a raster (MBTiles or any rasterio source).

Uses parallel windowed reads via :class:`concurrent.futures.ThreadPoolExecutor`
so that large rasters are processed without loading the full image into memory.
Intended to build the ``color_map`` parameter for
:class:`~pytorch_segmentation_models_trainer.dataset_loader.mbtiles_dataset.MBTilesPolygonDataset`.
"""

import logging
import os
import sqlite3
from concurrent.futures import ThreadPoolExecutor, as_completed
from itertools import islice
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import numpy as np
import rasterio
from rasterio.windows import Window

logger = logging.getLogger(__name__)


def _get_mbtiles_tile_windows(raster_path: str) -> Optional[List[Window]]:
    """Query the MBTiles SQLite tile index and return tile-aligned Windows.

    Reads the ``tiles`` table at the maximum zoom level to discover which
    tiles actually exist in the database.  Only those positions are returned,
    so empty/nodata areas of the raster bbox are never read.

    MBTiles uses TMS y-axis convention (row 0 = bottom); rasterio uses
    row 0 = top.  The conversion applied here is
    ``row_off = (max_tile_row - tile_row) * 256``.

    Args:
        raster_path: String path to the MBTiles (SQLite) file.

    Returns:
        List of 256×256 :class:`~rasterio.windows.Window` objects — one per
        existing tile — or ``None`` if *raster_path* is not a valid MBTiles
        SQLite database or contains no tiles.
    """
    try:
        with sqlite3.connect(raster_path) as conn:
            row = conn.execute("SELECT MAX(zoom_level) FROM tiles").fetchone()
            if row is None or row[0] is None:
                return None
            max_zoom = int(row[0])

            bounds_row = conn.execute(
                "SELECT MIN(tile_column), MAX(tile_column), MIN(tile_row), MAX(tile_row) "
                "FROM tiles WHERE zoom_level = ?",
                (max_zoom,),
            ).fetchone()
            if bounds_row is None or bounds_row[0] is None:
                return None
            min_col, _max_col, min_row, max_row = bounds_row

            tile_coords = conn.execute(
                "SELECT tile_column, tile_row FROM tiles WHERE zoom_level = ?",
                (max_zoom,),
            ).fetchall()
    except Exception:
        return None

    windows: List[Window] = []
    for tile_col, tile_row in tile_coords:
        col_off = (tile_col - min_col) * 256
        row_off = (max_row - tile_row) * 256  # TMS y-axis inversion
        windows.append(Window(col_off, row_off, 256, 256))
    return windows


def _unique_colors_in_window(
    raster_path: str,
    window: Window,
    bands: List[int],
) -> Set[Tuple[int, int, int]]:
    """Read one raster window and return the set of unique RGB tuples.

    Opens its own file handle so the function is safe to call from multiple
    threads concurrently.

    Intermediate uint32 arrays are built in-place to minimise peak memory:
    only two H×W uint32 arrays exist simultaneously instead of four.

    Args:
        raster_path: String path passed to ``rasterio.open``.
        window: The raster window to read.
        bands: 1-based band indices; exactly three values expected.

    Returns:
        Set of ``(R, G, B)`` integer tuples found in the window.
    """
    with rasterio.open(raster_path) as src:
        data = src.read(bands, window=window)  # (3, H, W) uint8

    packed = data[0].astype(np.uint32)
    packed *= 65536
    tmp = data[1].astype(np.uint32)
    tmp *= 256
    packed += tmp
    del tmp
    packed += data[2]
    del data

    result: Set[Tuple[int, int, int]] = set()
    for v in np.unique(packed):
        v_int = int(v)
        result.add(((v_int >> 16) & 0xFF, (v_int >> 8) & 0xFF, v_int & 0xFF))
    return result


def scan_unique_colors(
    raster_path: Union[str, Path],
    bands: Optional[List[int]] = None,
    window_size: int = 1024,
    workers: Optional[int] = None,
    progress: bool = True,
) -> List[Tuple[int, int, int]]:
    """Return all unique RGB triplets present in *raster_path*, sorted.

    **MBTiles sources** are scanned via the SQLite ``tiles`` table: only
    existing 256×256 tiles are read, so empty areas of the raster bbox are
    skipped entirely.

    **Other rasterio sources** (GeoTIFF, VRT, …) fall back to a regular grid
    of ``window_size × window_size`` windows covering the full raster.

    In both cases task submission is **bounded**: at most ``2 × workers``
    futures are in flight simultaneously, preventing memory from growing with
    the total number of windows.

    Args:
        raster_path: Path to any rasterio-readable source (MBTiles, GeoTIFF,
            VRT, …).  For MBTiles the GDAL MBTILES driver is used automatically.
        bands: 1-based RGB band indices.  Defaults to ``[1, 2, 3]``.  The
            raster must have at least three bands (or ``bands`` must refer to
            valid existing bands).
        window_size: Tile size in pixels for the fallback grid scan (not used
            for MBTiles sources, which always use 256×256 native tiles).
            Default: ``1024``.
        workers: Thread count.  ``None`` → ``min(batch_size, cpu_count)``.
        progress: Show a ``tqdm`` progress bar on *stderr*.

    Returns:
        Sorted list of ``(R, G, B)`` tuples.

    Raises:
        ValueError: If the raster has fewer bands than requested or the wrong
            number of bands is given.
    """
    raster_path = Path(raster_path)

    with rasterio.open(raster_path) as src:
        raster_h = src.height
        raster_w = src.width
        n_bands = src.count

    if bands is None:
        bands = [1, 2, 3]

    if max(bands) > n_bands:
        raise ValueError(
            f"Band index {max(bands)} is out of range; raster '{raster_path.name}' "
            f"has only {n_bands} band(s).  Pass --bands with valid 1-based indices."
        )
    if len(bands) != 3:
        raise ValueError(
            f"Exactly 3 band indices are required for RGB scanning; got {bands}.  "
            "To duplicate a single band use e.g. '--bands 1,1,1'."
        )

    raster_str = str(raster_path)

    tile_windows = _get_mbtiles_tile_windows(raster_str)
    if tile_windows is not None:
        windows = tile_windows
        logger.debug("MBTiles tile index: %d existing tiles", len(windows))
    else:
        windows = []
        for row_off in range(0, raster_h, window_size):
            row_h = min(window_size, raster_h - row_off)
            for col_off in range(0, raster_w, window_size):
                col_w = min(window_size, raster_w - col_off)
                windows.append(Window(col_off, row_off, col_w, row_h))
        logger.debug("Grid scan: %d windows", len(windows))

    if not windows:
        return []

    n_workers = (
        workers if workers is not None else min(len(windows), os.cpu_count() or 1)
    )
    all_colors: Set[Tuple[int, int, int]] = set()

    if progress:
        from tqdm import tqdm

        pbar = tqdm(total=len(windows), desc="Scanning windows", unit="win", leave=True)
    else:
        pbar = None

    # Bounded submission: at most batch_size futures in flight at once.
    # Caps peak memory at ~2 × workers × (memory per window) regardless of
    # total window count.
    batch_size = max(n_workers * 2, 1)
    windows_iter = iter(windows)

    try:
        with ThreadPoolExecutor(max_workers=n_workers) as executor:
            while True:
                batch = list(islice(windows_iter, batch_size))
                if not batch:
                    break
                batch_futures = [
                    executor.submit(_unique_colors_in_window, raster_str, win, bands)
                    for win in batch
                ]
                for f in as_completed(batch_futures):
                    all_colors.update(f.result())
                    if pbar is not None:
                        pbar.update(1)
    finally:
        if pbar is not None:
            pbar.close()

    return sorted(all_colors)


def build_color_map(
    unique_colors: List[Tuple[int, int, int]],
) -> List[List[int]]:
    """Assign sequential class indices to *unique_colors*.

    ``(0, 0, 0)`` is treated as background and receives class ``0``.
    All other colors are assigned ``1, 2, 3, …`` in sorted order.

    Args:
        unique_colors: Sorted list of ``(R, G, B)`` tuples as returned by
            :func:`scan_unique_colors`.

    Returns:
        List of ``[R, G, B, class_idx]`` entries ready for the
        ``color_map`` field of
        :class:`~pytorch_segmentation_models_trainer.dataset_loader.mbtiles_dataset.MBTilesPolygonDataset`.
    """
    color_map: List[List[int]] = []
    non_background = [c for c in unique_colors if c != (0, 0, 0)]
    has_black = (0, 0, 0) in unique_colors

    if has_black:
        color_map.append([0, 0, 0, 0])

    for idx, (r, g, b) in enumerate(non_background, start=1):
        color_map.append([r, g, b, idx])

    return color_map


def scan_and_report(
    raster_path: Union[str, Path],
    bands: Optional[List[int]] = None,
    window_size: int = 1024,
    workers: Optional[int] = None,
    progress: bool = True,
) -> Dict[str, Any]:
    """Run :func:`scan_unique_colors` and return a structured report dict.

    Args:
        raster_path: Path to the mask raster.
        bands: 1-based RGB band indices; defaults to ``[1, 2, 3]``.
        window_size: Windowed-read tile size in pixels (fallback grid only).
        workers: Thread pool size; ``None`` for auto.
        progress: Show tqdm progress bar.

    Returns:
        Dict with keys:

        - ``raster``: absolute path string.
        - ``n_unique_colors``: integer count.
        - ``color_map``: ``[[R, G, B, class_idx], …]`` auto-assigned list.
    """
    unique = scan_unique_colors(
        raster_path=raster_path,
        bands=bands,
        window_size=window_size,
        workers=workers,
        progress=progress,
    )
    return {
        "raster": str(Path(raster_path).resolve()),
        "n_unique_colors": len(unique),
        "color_map": build_color_map(unique),
    }
