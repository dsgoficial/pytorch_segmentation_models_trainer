"""
tiff_remap.py — Remap pixel class values in raster files.

Walks a directory tree and remaps pixel values in-place via a user-supplied
integer→integer mapping, writing results to a mirrored output tree.
"""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np


def remap_raster(
    input_path: Path,
    output_path: Path,
    pixel_mapping: Dict[int, int],
) -> Tuple[Path, bool, Optional[str]]:
    """Remap pixel values in a single raster file.

    Reads the input raster, applies *pixel_mapping* (a dict of
    ``{old_value: new_value}``), and writes the result to *output_path*.
    Bands that have no entries in *pixel_mapping* are written unchanged.

    Args:
        input_path: Path to the source raster (must be readable by rasterio).
        output_path: Destination path for the remapped raster (GeoTIFF).
        pixel_mapping: Mapping from source pixel value to target pixel value.

    Returns:
        Tuple of ``(output_path, success, error_message)``.  *error_message*
        is ``None`` on success.

    Example YAML::

        pixel_mapping:
          8: 5
          6: 4
    """
    try:
        import rasterio

        output_path.parent.mkdir(parents=True, exist_ok=True)

        with rasterio.open(input_path) as src:
            profile = src.profile.copy()
            profile.update(driver="GTiff")
            data = src.read()

        remapped = data.copy()
        for old_val, new_val in pixel_mapping.items():
            remapped[data == old_val] = new_val

        with rasterio.open(output_path, "w", **profile) as dst:
            dst.write(remapped)

        return output_path, True, None
    except Exception as exc:  # noqa: BLE001
        return output_path, False, str(exc)


def remap_raster_folder(
    input_dir: Path,
    output_dir: Path,
    pixel_mapping: Dict[int, int],
    extensions: Tuple[str, ...] = (".tif", ".tiff"),
    n_workers: Optional[int] = None,
    progress: bool = True,
) -> Tuple[int, int]:
    """Remap all rasters in a directory tree.

    Walks *input_dir* recursively, remaps every file whose extension is in
    *extensions*, and writes results to a mirrored subtree under *output_dir*.

    Args:
        input_dir: Root directory to search for raster files.
        output_dir: Root directory for remapped output files.
        pixel_mapping: Mapping from source pixel value to target pixel value.
        extensions: File extensions to process (case-insensitive).
        n_workers: Number of threads.  Defaults to ``os.cpu_count()``.
        progress: Whether to display a tqdm progress bar.

    Returns:
        Tuple ``(n_success, n_errors)``.
    """
    try:
        from tqdm import tqdm
    except ImportError:  # pragma: no cover
        tqdm = None  # type: ignore[assignment]

    exts_lower = {e.lower() for e in extensions}
    files = [
        p
        for p in input_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in exts_lower
    ]

    if not files:
        return 0, 0

    workers = n_workers or os.cpu_count() or 1
    n_success = 0
    n_errors = 0

    jobs = []
    for src_path in files:
        rel = src_path.relative_to(input_dir)
        dst_path = output_dir / rel
        jobs.append((src_path, dst_path))

    iterator = jobs
    if progress and tqdm is not None:
        iterator = tqdm(jobs, desc="Remapping rasters", unit="file")

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(remap_raster, src, dst, pixel_mapping): (src, dst)
            for src, dst in iterator
        }
        for future in as_completed(futures):
            _, success, _ = future.result()
            if success:
                n_success += 1
            else:
                n_errors += 1

    return n_success, n_errors
