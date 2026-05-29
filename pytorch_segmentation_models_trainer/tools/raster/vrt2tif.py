"""
vrt2tif.py — Convert VRT (or any rasterio-readable) files to compressed GeoTIFF.

Walks a directory tree matching a glob pattern and converts each found file
to a tiled, compressed GeoTIFF.
"""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional, Tuple


def convert_to_geotiff(
    input_path: Path,
    output_path: Path,
    compression: str = "LZW",
    tile_size: int = 512,
) -> Tuple[Path, bool, Optional[str]]:
    """Convert a single rasterio-readable file to a compressed GeoTIFF.

    Preserves band descriptions if present.  The output is tiled with
    ``tile_size × tile_size`` blocks.

    Args:
        input_path: Source file path (VRT, GeoTIFF, MBTiles, etc.).
        output_path: Destination GeoTIFF path.
        compression: Rasterio compression codec (``"LZW"``, ``"DEFLATE"``,
            ``"JPEG"``, ``"NONE"``).
        tile_size: Tile block size in pixels for the output GeoTIFF.

    Returns:
        Tuple ``(output_path, success, error_message)``.  *error_message* is
        ``None`` on success.

    Example YAML::

        compression: LZW
        tile_size: 512
    """
    try:
        import rasterio

        output_path.parent.mkdir(parents=True, exist_ok=True)

        with rasterio.open(input_path) as src:
            profile = src.profile.copy()
            descriptions = src.descriptions
            data = src.read()

        profile.update(
            driver="GTiff",
            compress=compression,
            tiled=True,
            blockxsize=tile_size,
            blockysize=tile_size,
        )

        with rasterio.open(output_path, "w", **profile) as dst:
            dst.write(data)
            for i, desc in enumerate(descriptions, start=1):
                if desc:
                    dst.update_tags(i, name=desc)

        return output_path, True, None
    except Exception as exc:  # noqa: BLE001
        return output_path, False, str(exc)


def convert_folder(
    input_dir: Path,
    output_dir: Path,
    glob_pattern: str = "**/*.vrt",
    compression: str = "LZW",
    tile_size: int = 512,
    n_workers: int = 4,
    progress: bool = True,
) -> Tuple[int, int]:
    """Convert all files matching *glob_pattern* in *input_dir* to GeoTIFF.

    Files are written to a mirrored subtree under *output_dir* with the
    extension replaced by ``.tif``.

    Args:
        input_dir: Root directory to search.
        output_dir: Root directory for converted output.
        glob_pattern: Glob pattern relative to *input_dir*.
        compression: Compression codec forwarded to
            :func:`convert_to_geotiff`.
        tile_size: Tile block size forwarded to :func:`convert_to_geotiff`.
        n_workers: Number of worker threads.
        progress: Whether to display a tqdm progress bar.

    Returns:
        Tuple ``(n_success, n_errors)``.
    """
    try:
        from tqdm import tqdm
    except ImportError:  # pragma: no cover
        tqdm = None  # type: ignore[assignment]

    files = list(input_dir.glob(glob_pattern))

    if not files:
        return 0, 0

    jobs = []
    for src_path in files:
        rel = src_path.relative_to(input_dir).with_suffix(".tif")
        dst_path = output_dir / rel
        jobs.append((src_path, dst_path))

    iterator = jobs
    if progress and tqdm is not None:
        iterator = tqdm(jobs, desc="Converting to GeoTIFF", unit="file")

    n_success = 0
    n_errors = 0

    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        futures = {
            executor.submit(convert_to_geotiff, src, dst, compression, tile_size): (
                src,
                dst,
            )
            for src, dst in iterator
        }
        for future in as_completed(futures):
            _, success, _ = future.result()
            if success:
                n_success += 1
            else:
                n_errors += 1

    return n_success, n_errors
