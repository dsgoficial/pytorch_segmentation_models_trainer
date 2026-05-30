"""
band_combiner.py — Combine bands from multiple raster sources into a single GeoTIFF.

Finds groups of files with matching names across N directories and combines
their bands into multi-band TIFFs for multi-source dataset preparation.
"""

from __future__ import annotations

import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Tuple


def find_file_groups(
    source_dirs: List[Path],
    glob_pattern: str = "**/*.vrt",
    name_pattern: Optional[str] = None,
) -> List[Tuple[str, List[Path]]]:
    """Find groups of files with the same name across all *source_dirs*.

    Only groups present in **all** directories are returned.

    Args:
        source_dirs: List of directories to scan.
        glob_pattern: Glob pattern relative to each directory.
        name_pattern: Optional pattern like ``"MI_{name}.tif"`` to extract a
            group key from the filename.  The pattern must contain a named
            capture group ``{name}`` which is expanded to ``(?P<name>.+)``.
            If ``None``, the file *stem* is used as the group key.

    Returns:
        Sorted list of ``(group_name, [path_from_dir0, path_from_dir1, ...])``.

    Example::

        groups = find_file_groups(
            source_dirs=[Path("rgb/"), Path("nir/")],
            glob_pattern="**/*.tif",
        )
    """
    if not source_dirs:
        return []

    regex = None
    if name_pattern is not None:
        raw = re.escape(name_pattern).replace(r"\{name\}", "(?P<name>.+)")
        regex = re.compile(raw)

    def _extract_key(path: Path) -> Optional[str]:
        if regex is None:
            return path.stem
        m = regex.match(path.name)
        return m.group("name") if m else None

    # Build a mapping: group_key -> list of one path per source_dir (in order)
    dir_maps: List[Dict[str, Path]] = []
    for d in source_dirs:
        mapping: Dict[str, Path] = {}
        for p in d.glob(glob_pattern):
            key = _extract_key(p)
            if key is not None:
                mapping[key] = p
        dir_maps.append(mapping)

    # Intersection of all keys
    common_keys = set(dir_maps[0].keys())
    for m in dir_maps[1:]:
        common_keys &= set(m.keys())

    result = []
    for key in sorted(common_keys):
        paths = [m[key] for m in dir_maps]
        result.append((key, paths))

    return result


def combine_sources_to_tiff(
    source_paths: List[Path],
    output_path: Path,
    bands_per_source: Optional[List[Optional[List[int]]]] = None,
    skip_alpha: bool = True,
    overwrite: bool = False,
) -> Tuple[Path, bool, Optional[str]]:
    """Combine bands from multiple rasters into a single GeoTIFF.

    All sources must share the same spatial shape and CRS.

    Args:
        source_paths: Ordered list of source raster paths.
        output_path: Destination GeoTIFF path.
        bands_per_source: Per-source list of **1-based** band indices to
            include.  ``None`` for a given source means "read all bands"
            (respecting *skip_alpha*).
        skip_alpha: When a source selection is ``None`` and the source has
            more than 3 bands, skip the last band (assumed to be alpha).
        overwrite: If ``False`` and *output_path* already exists, skip.

    Returns:
        Tuple ``(output_path, success, error_message)``.

    Example::

        combine_sources_to_tiff(
            source_paths=[Path("rgb.vrt"), Path("nir.tif")],
            output_path=Path("combined.tif"),
            bands_per_source=[None, [1]],
        )
    """
    if not overwrite and output_path.exists():
        return output_path, True, None

    try:
        import numpy as np
        import rasterio

        output_path.parent.mkdir(parents=True, exist_ok=True)

        if bands_per_source is None:
            bands_per_source = [None] * len(source_paths)

        arrays = []
        ref_profile = None
        ref_shape = None

        for i, (src_path, band_sel) in enumerate(zip(source_paths, bands_per_source)):
            with rasterio.open(src_path) as src:
                profile = src.profile.copy()
                n_bands = src.count

                if band_sel is not None:
                    data = src.read(band_sel)
                else:
                    if skip_alpha and n_bands > 3:
                        data = src.read(list(range(1, n_bands)))
                    else:
                        data = src.read()

                if data.ndim == 1:  # pragma: no cover
                    data = data[np.newaxis, ...]

                shape = (data.shape[-2], data.shape[-1])

                if ref_shape is None:
                    ref_shape = shape
                    ref_profile = profile
                else:
                    if shape != ref_shape:
                        return (
                            output_path,
                            False,
                            f"Shape mismatch: source 0 has {ref_shape}, "
                            f"source {i} has {shape}.",
                        )

                arrays.append(data)

        combined = np.concatenate(arrays, axis=0)
        out_profile = ref_profile.copy()
        out_profile.update(driver="GTiff", count=combined.shape[0])

        with rasterio.open(output_path, "w", **out_profile) as dst:
            dst.write(combined)

        return output_path, True, None
    except Exception as exc:  # noqa: BLE001
        return output_path, False, str(exc)


def combine_all(
    source_dirs: List[Path],
    output_dir: Path,
    glob_pattern: str = "**/*.vrt",
    bands_per_source: Optional[List[Optional[List[int]]]] = None,
    name_pattern: Optional[str] = None,
    skip_alpha: bool = True,
    overwrite: bool = False,
    n_workers: int = 4,
    progress: bool = True,
) -> List[Path]:
    """Find matching file groups and combine each into a multi-band TIFF.

    Scans *source_dirs* for files matching *glob_pattern*, groups them by
    name (or by *name_pattern* capture), then combines each group with
    :func:`combine_sources_to_tiff`.

    Args:
        source_dirs: Directories to scan for source rasters.
        output_dir: Directory for combined output TIFFs.
        glob_pattern: Glob pattern relative to each directory.
        bands_per_source: Forwarded to :func:`combine_sources_to_tiff`.
        name_pattern: Forwarded to :func:`find_file_groups`.
        skip_alpha: Forwarded to :func:`combine_sources_to_tiff`.
        overwrite: Forwarded to :func:`combine_sources_to_tiff`.
        n_workers: Number of worker threads.
        progress: Whether to display a tqdm progress bar.

    Returns:
        List of successfully written output paths.
    """
    try:
        from tqdm import tqdm
    except ImportError:  # pragma: no cover
        tqdm = None  # type: ignore[assignment]

    groups = find_file_groups(source_dirs, glob_pattern, name_pattern)
    if not groups:
        return []

    jobs = []
    for group_name, paths in groups:
        out_path = output_dir / f"{group_name}.tif"
        jobs.append((group_name, paths, out_path))

    iterator = jobs
    if progress and tqdm is not None:
        iterator = tqdm(jobs, desc="Combining bands", unit="group")

    successful: List[Path] = []

    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        futures = {
            executor.submit(
                combine_sources_to_tiff,
                paths,
                out_path,
                bands_per_source,
                skip_alpha,
                overwrite,
            ): out_path
            for _, paths, out_path in iterator
        }
        for future in as_completed(futures):
            out_path = futures[future]
            _, success, _ = future.result()
            if success:
                successful.append(out_path)

    return sorted(successful)
