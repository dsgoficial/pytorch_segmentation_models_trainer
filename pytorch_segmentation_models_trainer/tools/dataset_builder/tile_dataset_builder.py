"""
tile_dataset_builder.py — Build a tile dataset from raster images + vector masks.

Generates fixed-size tile patches from raster images and rasterizes polygon
features from a vector file as segmentation masks.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Generator, List, Optional, Tuple

import pandas as pd


def compute_tile_windows(
    image_width: int,
    image_height: int,
    tile_width: int,
    tile_height: int,
    overlap_x: int = 0,
    overlap_y: int = 0,
) -> Generator[Tuple[int, int, int, int, int], None, None]:
    """Yield ``(x_start, y_start, x_end, y_end, tile_id)`` for all fixed-size tiles.

    Tiles at image edges are adjusted so they stay within bounds while
    maintaining a fixed size of ``tile_width × tile_height`` pixels.

    Args:
        image_width: Width of the source image in pixels.
        image_height: Height of the source image in pixels.
        tile_width: Width of each output tile in pixels.
        tile_height: Height of each output tile in pixels.
        overlap_x: Overlap in pixels along the X axis (columns).
        overlap_y: Overlap in pixels along the Y axis (rows).

    Yields:
        Tuples ``(x_start, y_start, x_end, y_end, tile_id)`` where
        ``tile_id`` is a zero-based sequential index.

    Example::

        for x0, y0, x1, y1, tid in compute_tile_windows(512, 512, 256, 256):
            ...
    """
    stride_x = max(1, tile_width - overlap_x)
    stride_y = max(1, tile_height - overlap_y)

    tile_id = 0
    y = 0
    while y < image_height:
        x = 0
        while x < image_width:
            x_start = min(x, max(0, image_width - tile_width))
            y_start = min(y, max(0, image_height - tile_height))
            x_end = x_start + tile_width
            y_end = y_start + tile_height
            yield x_start, y_start, x_end, y_end, tile_id
            tile_id += 1
            x += stride_x
        y += stride_y


def _process_image(
    image_path: Path,
    gdf,
    class_attribute: str,
    output_dir: Path,
    tile_width: int,
    tile_height: int,
    overlap_x: int,
    overlap_y: int,
    min_valid_pixel_ratio: float,
    skip_empty_tiles: bool,
    generate_full_size_masks: bool,
    background_value: int,
) -> List[dict]:
    """Process a single image: tile it and rasterize polygon masks.

    Returns a list of row dicts suitable for building the output DataFrame.
    """
    import numpy as np
    import rasterio
    from rasterio.features import rasterize
    from shapely.geometry import mapping

    if not 0 <= background_value <= 255:
        raise ValueError("background_value must be in the uint8 range [0, 255]")

    background_value = int(background_value)

    rows = []

    with rasterio.open(image_path) as src:
        image_crs = src.crs
        image_transform = src.transform
        image_width = src.width
        image_height = src.height

    # Reproject GDF to image CRS if needed
    if gdf.crs is not None and image_crs is not None:
        gdf_reprojected = gdf.to_crs(image_crs)
    else:
        gdf_reprojected = gdf

    images_dir = output_dir / image_path.stem / "images"
    masks_dir = output_dir / image_path.stem / "masks"
    images_dir.mkdir(parents=True, exist_ok=True)
    masks_dir.mkdir(parents=True, exist_ok=True)

    # Generate full-size mask if requested
    if generate_full_size_masks:
        full_mask_path = output_dir / image_path.stem / "mask_full.tif"
        shapes = [
            (mapping(geom), int(cls_val))
            for geom, cls_val in zip(
                gdf_reprojected.geometry, gdf_reprojected[class_attribute]
            )
            if geom is not None and not geom.is_empty
        ]
        if shapes:
            full_mask = rasterize(
                shapes,
                out_shape=(image_height, image_width),
                transform=image_transform,
                fill=background_value,
                dtype=np.uint8,
            )
        else:
            full_mask = np.full(
                (image_height, image_width), background_value, dtype=np.uint8
            )

        mask_profile = dict(
            driver="GTiff",
            dtype=np.uint8,
            width=image_width,
            height=image_height,
            count=1,
            crs=image_crs,
            transform=image_transform,
        )
        with rasterio.open(full_mask_path, "w", **mask_profile) as dst:
            dst.write(full_mask, 1)

    # Process tiles
    with rasterio.open(image_path) as src:
        for x_start, y_start, x_end, y_end, tile_id in compute_tile_windows(
            image_width,
            image_height,
            tile_width,
            tile_height,
            overlap_x,
            overlap_y,
        ):
            window = rasterio.windows.Window(
                col_off=x_start,
                row_off=y_start,
                width=tile_width,
                height=tile_height,
            )
            tile_transform = rasterio.windows.transform(window, image_transform)
            tile_data = src.read(window=window)

            # Check valid pixel ratio (non-zero pixels in any band)
            valid_ratio = np.count_nonzero(tile_data) / tile_data.size
            if skip_empty_tiles and valid_ratio < min_valid_pixel_ratio:
                continue

            # Rasterize mask for this tile
            shapes = [
                (mapping(geom), int(cls_val))
                for geom, cls_val in zip(
                    gdf_reprojected.geometry, gdf_reprojected[class_attribute]
                )
                if geom is not None and not geom.is_empty
            ]

            if shapes:
                tile_mask = rasterize(
                    shapes,
                    out_shape=(tile_height, tile_width),
                    transform=tile_transform,
                    fill=background_value,
                    dtype=np.uint8,
                )
            else:
                tile_mask = np.full(
                    (tile_height, tile_width), background_value, dtype=np.uint8
                )

            if skip_empty_tiles and np.all(tile_mask == background_value):
                continue

            tile_name = f"{tile_id:06d}.tif"
            img_out = images_dir / tile_name
            mask_out = masks_dir / tile_name

            img_profile = dict(
                driver="GTiff",
                dtype=tile_data.dtype,
                width=tile_width,
                height=tile_height,
                count=tile_data.shape[0],
                crs=image_crs,
                transform=tile_transform,
            )
            with rasterio.open(img_out, "w", **img_profile) as dst:
                dst.write(tile_data)

            mask_profile = dict(
                driver="GTiff",
                dtype=np.uint8,
                width=tile_width,
                height=tile_height,
                count=1,
                crs=image_crs,
                transform=tile_transform,
            )
            with rasterio.open(mask_out, "w", **mask_profile) as dst:
                dst.write(tile_mask, 1)

            rows.append(
                {
                    "image_path": str(img_out),
                    "label_path": str(mask_out),
                    "rows": tile_height,
                    "columns": tile_width,
                }
            )

    return rows


def build_tile_dataset(
    image_paths: List[Path],
    vector_path: Path,
    class_attribute: str,
    output_dir: Path,
    vector_layer: Optional[str] = None,
    tile_width: int = 256,
    tile_height: int = 256,
    overlap_x_percent: int = 0,
    overlap_y_percent: int = 0,
    min_valid_pixel_ratio: float = 0.1,
    skip_empty_tiles: bool = True,
    generate_full_size_masks: bool = False,
    max_workers: int = 8,
    progress: bool = True,
    background_value: int = 255,
) -> pd.DataFrame:
    """Build a tile dataset from raster images and vector polygon masks.

    For each image in *image_paths*, generates fixed-size tile patches and
    rasterizes polygon features as segmentation masks.

    Output structure::

        output_dir/
          {image_stem}/
            images/  ← tile GeoTIFFs
            masks/   ← mask GeoTIFFs (uint8 class indices)
            mask_full.tif  ← only when generate_full_size_masks=True
          dataset.csv

    Args:
        image_paths: List of source raster paths.
        vector_path: Path to vector file (GeoPackage, Shapefile, GeoJSON …).
        class_attribute: Column in the vector file containing integer class IDs.
        output_dir: Root directory for the dataset.
        vector_layer: Layer name within *vector_path* (for GeoPackage).
        tile_width: Width of each tile in pixels.
        tile_height: Height of each tile in pixels.
        overlap_x_percent: Horizontal overlap as a percentage of *tile_width*.
        overlap_y_percent: Vertical overlap as a percentage of *tile_height*.
        min_valid_pixel_ratio: Minimum fraction of non-zero pixels required
            to keep an image tile (ignored for masks).
        skip_empty_tiles: Skip tiles where the mask is all zeros.
        generate_full_size_masks: Also write a full-resolution mask raster
            for each image.
        max_workers: Number of parallel worker threads.
        progress: Whether to show a tqdm progress bar.
        background_value: Pixel value used for nodata/background in the mask.
            Defaults to 255 so background is distinct from class index 0.

    Returns:
        DataFrame with columns ``image_path``, ``label_path``, ``rows``,
        ``columns``.  Also saved to ``output_dir/dataset.csv``.
    """
    try:
        from tqdm import tqdm
    except ImportError:  # pragma: no cover
        tqdm = None  # type: ignore[assignment]

    import geopandas as gpd

    output_dir.mkdir(parents=True, exist_ok=True)

    gdf = gpd.read_file(vector_path, layer=vector_layer)

    overlap_x = int(tile_width * overlap_x_percent / 100)
    overlap_y = int(tile_height * overlap_y_percent / 100)

    iterator = image_paths
    if progress and tqdm is not None:
        iterator = tqdm(image_paths, desc="Building tile dataset", unit="image")

    all_rows: List[dict] = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                _process_image,
                img_path,
                gdf,
                class_attribute,
                output_dir,
                tile_width,
                tile_height,
                overlap_x,
                overlap_y,
                min_valid_pixel_ratio,
                skip_empty_tiles,
                generate_full_size_masks,
                background_value,
            ): img_path
            for img_path in iterator
        }
        for future in as_completed(futures):
            rows = future.result()
            all_rows.extend(rows)

    df = pd.DataFrame(all_rows, columns=["image_path", "label_path", "rows", "columns"])
    csv_path = output_dir / "dataset.csv"
    df.to_csv(csv_path, index=False)

    return df
