# flake8: noqa
"""Build multiclass mask rasters from vector labels on an MBTiles reference grid."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
import math
from pathlib import Path
from typing import Optional

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.features import rasterize
from rasterio.windows import Window, from_bounds, transform as window_transform
from shapely.geometry import box, mapping
from shapely.validation import make_valid


def _sanitize_stem(value: object, fallback: str) -> str:
    """Return a filesystem-friendly stem for an output file."""
    text = fallback if value is None else str(value)
    text = text.strip()
    if not text:
        return fallback
    return "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in text)


def _window_from_bounds(
    bounds: tuple[float, float, float, float],
    transform,
    raster_width: int,
    raster_height: int,
) -> Window:
    """Return an integer window that fully covers *bounds*."""
    float_window = from_bounds(*bounds, transform=transform)
    eps = 1e-6
    col_start = max(0, int(math.floor(float_window.col_off + eps)))
    row_start = max(0, int(math.floor(float_window.row_off + eps)))
    col_stop = min(
        raster_width,
        int(math.ceil(float_window.col_off + float_window.width - eps)),
    )
    row_stop = min(
        raster_height,
        int(math.ceil(float_window.row_off + float_window.height - eps)),
    )
    width = max(0, col_stop - col_start)
    height = max(0, row_stop - row_start)
    return Window(col_start, row_start, width, height)


def _frame_identifier(frame: pd.Series, frame_id_attribute: str, index: int) -> str:
    """Build a stable output stem for a frame record."""
    if frame_id_attribute in frame and not pd.isna(frame[frame_id_attribute]):
        return _sanitize_stem(frame[frame_id_attribute], f"frame_{index:05d}")
    return f"frame_{index:05d}"


def _write_mask(
    output_path: Path,
    mask: np.ndarray,
    crs,
    transform,
    nodata: int = 255,
) -> None:
    """Write a single-band ``uint8`` mask to disk."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    profile = {
        "driver": "GTiff",
        "height": int(mask.shape[0]),
        "width": int(mask.shape[1]),
        "count": 1,
        "dtype": "uint8",
        "crs": crs,
        "transform": transform,
        "nodata": int(nodata),
    }
    with rasterio.open(output_path, "w", **profile) as dst:
        dst.write(mask, 1)


def _ensure_valid_geometry(geometry):
    """Return a valid geometry or ``None`` when it cannot be repaired.

    Args:
        geometry: Any shapely geometry instance.

    Returns:
        A valid geometry, or ``None`` if the input is empty or cannot be
        repaired into a polygonal geometry.
    """
    if geometry is None or geometry.is_empty:
        return None
    if geometry.is_valid:
        return geometry

    repaired = make_valid(geometry)
    if repaired is None or repaired.is_empty:
        return None
    return repaired


def _query_vector_candidates(vector_gdf: gpd.GeoDataFrame, clipped_frame):
    """Return candidate vector rows that may intersect *clipped_frame*.

    Args:
        vector_gdf: Vector polygons already reprojected to the reference CRS.
        clipped_frame: Frame geometry clipped to the reference bounds.

    Returns:
        A GeoDataFrame slice containing only rows whose bounding boxes may
        intersect *clipped_frame*.
    """
    sindex = vector_gdf.sindex
    if hasattr(sindex, "query"):
        try:
            candidates = sindex.query(clipped_frame, predicate="intersects")
        except TypeError:
            candidates = sindex.query(clipped_frame)
    else:  # pragma: no cover - defensive fallback for older spatial indexes.
        candidates = sindex.intersection(clipped_frame.bounds)

    return vector_gdf.iloc[list(candidates)]


def _iter_frame_jobs(
    frames_gdf: gpd.GeoDataFrame,
    vector_gdf: gpd.GeoDataFrame,
    reference_bounds,
    reference_transform,
    reference_width: int,
    reference_height: int,
    frame_id_attribute: str,
    class_attribute: str,
):
    """Yield rasterization jobs one frame at a time.

    Args:
        frames_gdf: Frame geometries already reprojected to the reference CRS.
        vector_gdf: Class polygons already reprojected to the reference CRS.
        reference_bounds: Bounding geometry of the MBTiles reference grid.
        reference_transform: Affine transform of the reference grid.
        reference_width: Raster width in pixels.
        reference_height: Raster height in pixels.
        frame_id_attribute: Column used to name output files.
        class_attribute: Column containing integer class ids.

    Yields:
        Tuples ``(index, frame_id, clipped_frame, shapes, output_transform,
        out_shape)`` for frames that intersect the reference grid.

    Example YAML:
        ```yaml
        frame_id_attribute: rect_id
        class_attribute: tipo
        ```
    """
    for index, frame in frames_gdf.iterrows():
        frame_geom = frame.geometry
        if frame_geom is None or frame_geom.is_empty:
            continue

        clipped_frame = frame_geom.intersection(reference_bounds)
        if clipped_frame.is_empty:
            continue

        window = _window_from_bounds(
            clipped_frame.bounds,
            reference_transform,
            reference_width,
            reference_height,
        )
        if window.width <= 0 or window.height <= 0:
            continue

        subset = _query_vector_candidates(vector_gdf, clipped_frame)
        shapes = []
        if not subset.empty:
            subset = subset[subset.intersects(clipped_frame)]
            for geom, class_id in zip(subset.geometry, subset[class_attribute]):
                if geom is None or geom.is_empty:  # pragma: no cover
                    continue
                repaired_geom = _ensure_valid_geometry(geom)
                if repaired_geom is None or repaired_geom.is_empty:
                    continue
                shapes.append((repaired_geom, int(class_id)))

        frame_id = _frame_identifier(frame, frame_id_attribute, index)
        output_transform = window_transform(window, reference_transform)
        out_shape = (int(window.height), int(window.width))
        yield (
            index,
            frame_id,
            clipped_frame,
            shapes,
            output_transform,
            out_shape,
        )


def _build_single_frame_mask(
    index: int,
    *,
    frame_id: str,
    reference_crs,
    masks_dir: Path,
    output_extension: str,
    background_value: int,
    output_transform,
    clipped_frame,
    shapes,
    out_shape,
) -> Optional[dict]:
    """Build and write the mask for one frame record.

    Args:
        index: Original dataframe row index used for fallback naming.
        reference_crs: CRS inherited from the MBTiles source.
        masks_dir: Directory that receives the mask rasters.
        output_extension: Output raster extension.
        background_value: Background class written outside polygons.
        output_transform: Affine transform for the output mask window.
        clipped_frame: Frame geometry clipped to the reference bounds.
        shapes: Rasterize-ready ``(geometry, class_id)`` pairs.
        out_shape: Output mask shape as ``(height, width)``.

    Returns:
        Dictionary describing the written mask, or ``None`` if the frame does
        not intersect the reference grid or produces an empty output window.
    """
    background_value = int(background_value)
    mask = np.empty(out_shape, dtype=np.uint8)
    mask.fill(background_value)
    if shapes:
        rasterize(
            [(mapping(geom), class_id) for geom, class_id in shapes],
            out=mask,
            out_shape=out_shape,
            transform=output_transform,
            fill=background_value,
            dtype=np.uint8,
        )

    frame_footprint = np.zeros(out_shape, dtype=np.uint8)
    rasterize(
        [(mapping(clipped_frame), 1)],
        out=frame_footprint,
        out_shape=out_shape,
        transform=output_transform,
        fill=0,
        dtype=np.uint8,
    )
    mask[frame_footprint == 0] = background_value

    mask_path = masks_dir / f"{frame_id}.{output_extension.lstrip('.')}"
    _write_mask(mask_path, mask, reference_crs, output_transform)

    return {
        "frame_id": frame_id,
        "mask_path": str(mask_path),
        "width": int(mask.shape[1]),
        "height": int(mask.shape[0]),
        "crs": str(reference_crs),
        "transform": str(output_transform),
        "background_value": int(background_value),
        "reference_mbtiles_path": "",
        "frames_path": "",
        "vector_path": "",
        "_index": int(index),
    }


def build_mbtiles_multiclass_masks(
    reference_mbtiles_path: Path,
    frames_path: Path,
    vector_path: Path,
    output_dir: Path,
    frame_layer: Optional[str] = None,
    vector_layer: Optional[str] = None,
    frame_id_attribute: str = "rect_id",
    class_attribute: str = "tipo",
    output_subdir: str = "masks",
    output_extension: str = "tif",
    background_value: int = 255,
    n_workers: int = 8,
    progress: bool = True,
) -> pd.DataFrame:
    """Rasterize multiclass masks using an MBTiles source as the reference grid.

    The MBTiles file defines the target CRS, transform, resolution, width, and
    height. Each frame geometry in *frames_path* becomes one output mask raster.
    Features from *vector_path* are rasterized into that frame using the
    ``class_attribute`` column as the pixel class index.

    Args:
        reference_mbtiles_path: MBTiles file used as the grid reference.
        frames_path: GeoJSON or other vector file with frame geometries.
        vector_path: GeoPackage or other vector file with class polygons.
        output_dir: Directory that will receive ``dataset.csv`` and masks.
        frame_layer: Optional layer name for *frames_path*.
        vector_layer: Optional layer name for *vector_path*.
        frame_id_attribute: Column used to name each output mask file.
        class_attribute: Column with integer class ids in *vector_path*.
        output_subdir: Subdirectory under *output_dir* for mask rasters.
        output_extension: Output raster extension. ``tif`` is recommended.
        background_value: Background class value written outside polygons.
        n_workers: Number of worker threads used to process frames.
        progress: Whether to display a tqdm progress bar.

    Returns:
        DataFrame with one row per generated mask.

    Example YAML:
        ```yaml
        reference_mbtiles_path: /data/tiles.mbtiles
        frames_path: /data/locais_mascaras.geojson
        vector_path: /data/dsg_masks.gpkg
        output_dir: /data/output
        frame_layer: locais_mascaras
        vector_layer: dsg_masks
        frame_id_attribute: rect_id
        class_attribute: tipo
        background_value: 255
        ```
    """
    if not 0 <= int(background_value) <= 255:
        raise ValueError("background_value must be in the uint8 range [0, 255]")
    if int(n_workers) < 1:
        raise ValueError("n_workers must be at least 1")

    output_dir = Path(output_dir)
    masks_dir = output_dir / output_subdir
    output_dir.mkdir(parents=True, exist_ok=True)
    masks_dir.mkdir(parents=True, exist_ok=True)

    frames_gdf = gpd.read_file(frames_path, layer=frame_layer)
    vector_gdf = gpd.read_file(vector_path, layer=vector_layer)

    with rasterio.open(reference_mbtiles_path) as src:
        reference_crs = src.crs
        reference_transform = src.transform
        reference_bounds = box(*src.bounds)
        reference_width = src.width
        reference_height = src.height

    if frames_gdf.crs is None:
        raise ValueError("frames_path must declare a CRS")
    if vector_gdf.crs is None:
        raise ValueError("vector_path must declare a CRS")
    if reference_crs is None:
        raise ValueError("reference_mbtiles_path must declare a CRS")

    try:
        from tqdm import tqdm
    except ImportError:  # pragma: no cover
        tqdm = None  # type: ignore[assignment]

    if frames_gdf.crs != reference_crs:
        frames_gdf = frames_gdf.to_crs(reference_crs)
    if vector_gdf.crs != reference_crs:
        vector_gdf = vector_gdf.to_crs(reference_crs)

    rows = []
    job_iter = _iter_frame_jobs(
        frames_gdf=frames_gdf,
        vector_gdf=vector_gdf,
        reference_bounds=reference_bounds,
        reference_transform=reference_transform,
        reference_width=reference_width,
        reference_height=reference_height,
        frame_id_attribute=frame_id_attribute,
        class_attribute=class_attribute,
    )
    pbar = (
        tqdm(total=len(frames_gdf), desc="Building masks", unit="frame")
        if progress and tqdm is not None
        else None
    )

    batch_size = max(int(n_workers) * 2, 1)
    pending = set()

    try:
        with ThreadPoolExecutor(max_workers=int(n_workers)) as executor:
            while True:
                while len(pending) < batch_size:
                    try:
                        (
                            index,
                            frame_id,
                            clipped_frame,
                            shapes,
                            output_transform,
                            out_shape,
                        ) = next(job_iter)
                    except StopIteration:
                        break

                    pending.add(
                        executor.submit(
                            _build_single_frame_mask,
                            index,
                            frame_id=frame_id,
                            reference_crs=reference_crs,
                            masks_dir=masks_dir,
                            output_extension=output_extension,
                            background_value=background_value,
                            output_transform=output_transform,
                            clipped_frame=clipped_frame,
                            shapes=shapes,
                            out_shape=out_shape,
                        )
                    )

                if not pending:
                    break

                for future in as_completed(pending):
                    pending.remove(future)
                    row = future.result()
                    if row is not None:
                        rows.append(row)
                    if pbar is not None:
                        pbar.update(1)
                    break
    finally:
        if pbar is not None:
            pbar.close()

    rows.sort(key=lambda row: row["_index"])
    for row in rows:
        row.pop("_index", None)
        row["reference_mbtiles_path"] = str(reference_mbtiles_path)
        row["frames_path"] = str(frames_path)
        row["vector_path"] = str(vector_path)

    df = pd.DataFrame(
        rows,
        columns=[
            "frame_id",
            "mask_path",
            "width",
            "height",
            "crs",
            "transform",
            "background_value",
            "reference_mbtiles_path",
            "frames_path",
            "vector_path",
        ],
    )
    df.to_csv(output_dir / "dataset.csv", index=False)
    return df
