"""
tiff_remap.py — Remap pixel class values in raster files.

Provides two complementary APIs:

* Simple dict-based remapping (``remap_raster`` / ``remap_raster_folder``):
  unmapped values are kept as-is.

* DsgTools-compatible JSON remapping (``remap_raster_windowed`` /
  ``remap_raster_folder_from_json``): unmapped values become ``nodata_value``;
  reads are performed in parallel windows with a write lock.

JSON file format (DsgTools-compatible)::

    {
        "description": "MapBiomas -> EDGV mapping",
        "nodata_value": 255,
        "mapping": {
            "3": 1,
            "15": 2,
            "33": 0
        }
    }
"""

from __future__ import annotations

import json
import os
import warnings
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Lock
from typing import Dict, List, Optional, Tuple

import numpy as np

# Maps numpy dtype names to GDAL VRT dataType attribute strings.
_NUMPY_TO_GDAL_DTYPE: Dict[str, str] = {
    "uint8": "Byte",
    "uint16": "UInt16",
    "int16": "Int16",
    "int32": "Int32",
    "float32": "Float32",
    "float64": "Float64",
}


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


# ---------------------------------------------------------------------------
# DsgTools-compatible JSON-based remapping
# ---------------------------------------------------------------------------


def load_remap_json(
    json_path: Path,
) -> Tuple[Dict[int, int], int, str]:
    """Load and validate a DsgTools-compatible remap JSON file.

    The file must contain a ``mapping`` object with ``"source": target`` pairs
    (both converted to ``int``).  The ``nodata_value`` field defaults to 255
    when absent.  Non-integer entries are skipped with a :class:`UserWarning`.

    Args:
        json_path: Path to the JSON mapping file.

    Returns:
        Tuple of ``(mapping, nodata_value, description)``.

    Raises:
        FileNotFoundError: If *json_path* does not exist.
        ValueError: If the JSON is malformed, missing the ``mapping`` field,
            or contains no valid integer mappings.

    Example JSON::

        {
            "description": "MapBiomas -> EDGV",
            "nodata_value": 255,
            "mapping": {"3": 1, "15": 2}
        }
    """
    json_path = Path(json_path)
    if not json_path.exists():
        raise FileNotFoundError(f"Mapping JSON not found: {json_path}")

    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in {json_path}: {exc}") from exc

    if "mapping" not in data:
        raise ValueError('JSON must contain a "mapping" field.')

    nodata_value = int(data.get("nodata_value", 255))
    description = data.get("description", "")

    mapping: Dict[int, int] = {}
    invalid = []
    for k, v in data["mapping"].items():
        try:
            mapping[int(k)] = int(v)
        except (ValueError, TypeError):
            invalid.append(f"{k!r} -> {v!r}")

    if invalid:
        warnings.warn(
            f"Skipped invalid (non-integer) mapping entries: {', '.join(invalid)}",
            stacklevel=2,
        )

    if not mapping:
        raise ValueError("No valid integer mappings found in JSON.")

    return mapping, nodata_value, description


def infer_output_dtype(
    mapping: Dict[int, int],
    nodata_value: int,
) -> np.dtype:
    """Return the smallest numpy dtype that can represent all output values.

    Considers both the mapped target values and *nodata_value*.

    Args:
        mapping: Dict mapping source pixel values to target pixel values.
        nodata_value: Nodata value that will be written to the output raster.

    Returns:
        A numpy :class:`~numpy.dtype` instance.

    Example::

        >>> infer_output_dtype({3: 1, 15: 2}, 255)
        dtype('uint8')
        >>> infer_output_dtype({1: 601, 2: 901}, -9999)
        dtype('int16')
    """
    all_values = list(mapping.values()) + [nodata_value]
    min_val = min(all_values)
    max_val = max(all_values)

    if min_val >= 0:
        if max_val <= 255:
            return np.dtype(np.uint8)
        if max_val <= 65535:
            return np.dtype(np.uint16)
    if min_val >= -32768 and max_val <= 32767:
        return np.dtype(np.int16)
    if min_val >= -(2**31) and max_val <= 2**31 - 1:
        return np.dtype(np.int32)
    return np.dtype(np.float32)


def _apply_remap_array(
    data: np.ndarray,
    mapping: Dict[int, int],
    nodata_value: int,
    src_nodata: Optional[float],
    out_dtype: np.dtype,
) -> np.ndarray:
    """Apply pixel-value remapping to a NumPy array.

    Initialises output with *nodata_value* (DsgTools semantics: unmapped pixels
    become nodata).  Input nodata pixels are forced to *nodata_value* after
    applying the mapping, so a mapping key that equals *src_nodata* does not
    accidentally survive.
    """
    # Cast to int64 for safe comparison across numeric dtypes
    data_int = data.astype(np.int64)
    output = np.full(data.shape, nodata_value, dtype=out_dtype)

    for in_val, out_val in mapping.items():
        output[data_int == in_val] = out_val

    # Input nodata pixels override any mapping result
    if src_nodata is not None:
        output[data_int == int(src_nodata)] = nodata_value

    return output


def remap_raster_windowed(
    input_path: Path,
    output_path: Path,
    mapping: Dict[int, int],
    nodata_value: int = 255,
    n_workers: int = 4,
    compress: str = "lzw",
) -> Tuple[Path, bool, Optional[str]]:
    """Remap pixel values in a single raster using concurrent windowed I/O.

    Reads the raster in parallel windows (one rasterio handle per thread) and
    writes with a shared lock.  The output dtype is inferred automatically from
    the range of target values and *nodata_value*.

    Unmapped pixel values become *nodata_value* (DsgTools semantics).  Input
    nodata pixels also become *nodata_value* even when the input nodata value
    appears as a key in *mapping*.

    Args:
        input_path: Source raster path (any rasterio-readable format).
        output_path: Destination GeoTIFF path.
        mapping: ``{source_value: target_value}`` integer mapping.
        nodata_value: Value written for unmapped / nodata pixels (default 255).
        n_workers: Number of reader threads.
        compress: GeoTIFF compression codec (e.g. ``"lzw"``, ``"deflate"``).

    Returns:
        Tuple of ``(output_path, success, error_message)``.

    Example YAML::

        mapping_json: /path/to/mapping.json
        nodata_value: 255
        n_workers: 4
    """
    try:
        import rasterio

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with rasterio.open(input_path) as src:
            profile = src.profile.copy()
            src_nodata = src.nodata
            windows = list(src.block_windows(1))

        out_dtype = infer_output_dtype(mapping, nodata_value)
        profile.update(
            driver="GTiff",
            dtype=out_dtype,
            nodata=nodata_value,
            compress=compress,
            tiled=True,
            blockxsize=256,
            blockysize=256,
        )

        write_lock = Lock()

        with rasterio.open(output_path, "w", **profile) as dst:

            def _process_window(window):
                with rasterio.open(input_path) as src_local:
                    data = src_local.read(window=window)
                remapped = _apply_remap_array(
                    data, mapping, nodata_value, src_nodata, out_dtype
                )
                with write_lock:
                    dst.write(remapped, window=window)

            with ThreadPoolExecutor(max_workers=n_workers) as executor:
                futures = [executor.submit(_process_window, w) for _, w in windows]
                for future in as_completed(futures):
                    future.result()

        return output_path, True, None
    except Exception as exc:  # noqa: BLE001
        return output_path, False, str(exc)


def remap_raster_folder_from_json(
    input_dir: Path,
    output_dir: Path,
    json_path: Path,
    extensions: Tuple[str, ...] = (".tif", ".tiff"),
    n_workers: Optional[int] = None,
    progress: bool = True,
    create_vrt: bool = False,
    vrt_path: Optional[Path] = None,
) -> Tuple[int, int]:
    """Remap all rasters in a directory using a DsgTools-compatible JSON mapping.

    Walks *input_dir* recursively and calls :func:`remap_raster_windowed` on
    every file whose extension matches *extensions*, using a file-level
    :class:`~concurrent.futures.ThreadPoolExecutor`.

    When *create_vrt* is ``True``, a GDAL VRT mosaic of all successfully
    remapped rasters is written to *vrt_path* (default:
    ``output_dir/mosaic.vrt``).  The VRT carries the same ``nodata_value``
    defined in the JSON file.

    Args:
        input_dir: Root directory to search for raster files.
        output_dir: Root directory for remapped output files (mirrored tree).
        json_path: Path to the DsgTools-compatible JSON mapping file.
        extensions: File extensions to process (case-insensitive).
        n_workers: Number of concurrent file-processing threads.
            Defaults to ``os.cpu_count()``.
        progress: Whether to display a tqdm progress bar.
        create_vrt: If ``True``, build a VRT mosaic of all remapped rasters
            after processing.
        vrt_path: Destination for the VRT file.  Defaults to
            ``output_dir / "mosaic.vrt"`` when *create_vrt* is ``True``.

    Returns:
        Tuple ``(n_success, n_errors)``.

    Example YAML::

        input_dir: /path/to/masks
        output_dir: /path/to/masks_remapped
        json_path: /path/to/mapping.json
        n_workers: 8
        create_vrt: true
    """
    try:
        from tqdm import tqdm
    except ImportError:  # pragma: no cover
        tqdm = None  # type: ignore[assignment]

    mapping, nodata_value, _ = load_remap_json(json_path)

    exts_lower = {e.lower() for e in extensions}
    files = [
        p
        for p in Path(input_dir).rglob("*")
        if p.is_file() and p.suffix.lower() in exts_lower
    ]

    if not files:
        return 0, 0

    workers = n_workers or os.cpu_count() or 1
    n_success = 0
    n_errors = 0
    successful_outputs: List[Path] = []

    jobs = [(p, Path(output_dir) / p.relative_to(input_dir)) for p in files]

    iterator = jobs
    if progress and tqdm is not None:
        iterator = tqdm(jobs, desc="Remapping rasters", unit="file")

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(remap_raster_windowed, src, dst, mapping, nodata_value): dst
            for src, dst in iterator
        }
        for future in as_completed(futures):
            dst_path = futures[future]
            _, success, _ = future.result()
            if success:
                n_success += 1
                successful_outputs.append(dst_path)
            else:
                n_errors += 1

    if create_vrt and successful_outputs:
        out_vrt = (
            Path(vrt_path) if vrt_path is not None else Path(output_dir) / "mosaic.vrt"
        )
        build_vrt(successful_outputs, out_vrt, nodata_value)

    return n_success, n_errors


def build_vrt(
    raster_paths: List[Path],
    output_path: Path,
    nodata_value: int,
) -> Path:
    """Build a GDAL VRT mosaic from a list of raster files.

    Computes the union bounding box of all inputs and positions each raster
    at its correct pixel offset within the VRT.  All inputs must share the
    same CRS, pixel resolution, and band count.

    Uses only ``xml.etree.ElementTree`` and rasterio — no ``osgeo.gdal``
    dependency required.

    Args:
        raster_paths: Source rasters (any rasterio-readable format).
        output_path: Destination ``.vrt`` path.
        nodata_value: Nodata value embedded in every VRT band.

    Returns:
        *output_path* after writing.

    Raises:
        ValueError: If *raster_paths* is empty, or rasters have mismatched
            CRS, pixel resolution, or band count.

    Example::

        from pathlib import Path
        from pytorch_segmentation_models_trainer.tools.raster.tiff_remap import build_vrt

        build_vrt(
            raster_paths=list(Path("masks_remapped").rglob("*.tif")),
            output_path=Path("masks_remapped/mosaic.vrt"),
            nodata_value=255,
        )
    """
    import rasterio

    if not raster_paths:
        raise ValueError("raster_paths must not be empty.")

    output_path = Path(output_path)

    metas = []
    for p in raster_paths:
        with rasterio.open(p) as src:
            metas.append(
                {
                    "path": Path(p),
                    "crs": src.crs,
                    "pixel_w": src.transform.a,
                    "pixel_h": abs(src.transform.e),
                    "width": src.width,
                    "height": src.height,
                    "count": src.count,
                    "dtype": src.dtypes[0],
                    "bounds": src.bounds,
                }
            )

    ref = metas[0]
    for m in metas[1:]:
        if m["crs"] != ref["crs"]:
            raise ValueError(
                f"CRS mismatch: {m['path'].name} has {m['crs']} "
                f"(expected {ref['crs']})"
            )
        if abs(m["pixel_w"] - ref["pixel_w"]) > 1e-9:
            raise ValueError(
                f"Pixel width mismatch: {m['path'].name} "
                f"({m['pixel_w']:.10f} vs {ref['pixel_w']:.10f})"
            )
        if abs(m["pixel_h"] - ref["pixel_h"]) > 1e-9:
            raise ValueError(
                f"Pixel height mismatch: {m['path'].name} "
                f"({m['pixel_h']:.10f} vs {ref['pixel_h']:.10f})"
            )
        if m["count"] != ref["count"]:
            raise ValueError(
                f"Band count mismatch: {m['path'].name} "
                f"({m['count']} vs {ref['count']})"
            )

    x_min = min(m["bounds"].left for m in metas)
    y_max = max(m["bounds"].top for m in metas)
    x_max = max(m["bounds"].right for m in metas)

    pixel_w = ref["pixel_w"]
    pixel_h = ref["pixel_h"]

    vrt_width = round((x_max - x_min) / pixel_w)
    vrt_height = round((y_max - min(m["bounds"].bottom for m in metas)) / pixel_h)

    gdal_dtype = _NUMPY_TO_GDAL_DTYPE.get(str(ref["dtype"]), "Byte")

    root = ET.Element(
        "VRTDataset",
        rasterXSize=str(vrt_width),
        rasterYSize=str(vrt_height),
    )

    srs_elem = ET.SubElement(root, "SRS")
    srs_elem.text = ref["crs"].to_wkt() if ref["crs"] else ""

    geo_elem = ET.SubElement(root, "GeoTransform")
    geo_elem.text = (
        f" {x_min:.16f}, {pixel_w:.16f}, 0.0," f" {y_max:.16f}, 0.0, {-pixel_h:.16f}"
    )

    for band_idx in range(1, ref["count"] + 1):
        band_elem = ET.SubElement(
            root, "VRTRasterBand", dataType=gdal_dtype, band=str(band_idx)
        )
        nd_elem = ET.SubElement(band_elem, "NoDataValue")
        nd_elem.text = str(nodata_value)

        for m in metas:
            x_off = round((m["bounds"].left - x_min) / pixel_w)
            y_off = round((y_max - m["bounds"].top) / pixel_h)

            try:
                rel = str(m["path"].relative_to(output_path.parent))
                relative_to_vrt = "1"
            except ValueError:
                rel = str(m["path"])
                relative_to_vrt = "0"

            src_elem = ET.SubElement(band_elem, "SimpleSource")
            fn = ET.SubElement(
                src_elem, "SourceFilename", relativeToVRT=relative_to_vrt
            )
            fn.text = rel
            ET.SubElement(src_elem, "SourceBand").text = str(band_idx)
            ET.SubElement(
                src_elem,
                "SrcRect",
                xOff="0",
                yOff="0",
                xSize=str(m["width"]),
                ySize=str(m["height"]),
            )
            ET.SubElement(
                src_elem,
                "DstRect",
                xOff=str(x_off),
                yOff=str(y_off),
                xSize=str(m["width"]),
                ySize=str(m["height"]),
            )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    tree = ET.ElementTree(root)
    ET.indent(tree, space="  ")
    tree.write(str(output_path), encoding="unicode", xml_declaration=False)

    return output_path
