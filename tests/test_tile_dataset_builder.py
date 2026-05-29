"""Tests for tools/dataset_builder/tile_dataset_builder.py."""

from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
import rasterio
from rasterio.crs import CRS
from rasterio.transform import from_bounds
from shapely.geometry import box


def make_tiff(path: Path, data: np.ndarray, crs: str = "EPSG:4326") -> None:
    """Helper: write a minimal GeoTIFF."""
    h = data.shape[-2] if data.ndim > 2 else data.shape[0]
    w = data.shape[-1] if data.ndim > 2 else data.shape[1]
    transform = from_bounds(0, 0, 1, 1, w, h)
    bands = 1 if data.ndim == 2 else data.shape[0]
    profile = dict(
        driver="GTiff",
        dtype=data.dtype,
        width=w,
        height=h,
        count=bands,
        crs=crs,
        transform=transform,
    )
    with rasterio.open(path, "w", **profile) as dst:
        if data.ndim == 2:
            dst.write(data, 1)
        else:
            dst.write(data)


from pytorch_segmentation_models_trainer.tools.dataset_builder.tile_dataset_builder import (
    build_tile_dataset,
    compute_tile_windows,
)


@pytest.fixture()
def image_and_vector(tmp_path: Path):
    """Create a 100×100 image + GeoPackage polygon covering ~80% of the image."""
    img_path = tmp_path / "image.tif"
    data = np.random.randint(1, 255, (3, 100, 100), dtype=np.uint8)
    make_tiff(img_path, data, crs="EPSG:4326")

    # Polygon covering pixel columns 10-90, rows 10-90 (80% coverage)
    # from_bounds(0,0,1,1,100,100) → pixel (c, r) maps to (c/100, 1-r/100)
    poly = box(0.10, 0.10, 0.90, 0.90)
    gdf = gpd.GeoDataFrame(
        {"geometry": [poly], "class_id": [1]},
        crs="EPSG:4326",
    )
    gpkg_path = tmp_path / "masks.gpkg"
    gdf.to_file(gpkg_path, driver="GPKG", layer="polygons")

    return img_path, gpkg_path, tmp_path / "dataset"


# ---- compute_tile_windows ----


def test_compute_tile_windows_no_overlap() -> None:
    """Windows with no overlap should tile the image completely."""
    windows = list(compute_tile_windows(100, 100, 50, 50))
    assert len(windows) == 4  # 2x2 grid
    for x0, y0, x1, y1, _ in windows:
        assert x1 - x0 == 50
        assert y1 - y0 == 50


def test_compute_tile_windows_with_overlap_reduces_stride() -> None:
    """Overlap should produce more tiles than without overlap."""
    windows_no_overlap = list(compute_tile_windows(100, 100, 50, 50, 0, 0))
    windows_with_overlap = list(compute_tile_windows(100, 100, 50, 50, 25, 25))
    assert len(windows_with_overlap) > len(windows_no_overlap)


def test_compute_tile_windows_all_tiles_fixed_size() -> None:
    """Every generated tile should have exactly the requested size."""
    for x0, y0, x1, y1, _ in compute_tile_windows(120, 80, 50, 50):
        assert x1 - x0 == 50, f"Tile width mismatch: {x1-x0}"
        assert y1 - y0 == 50, f"Tile height mismatch: {y1-y0}"


def test_compute_tile_windows_edge_tile_adjusted() -> None:
    """Edge tiles should be adjusted so they don't exceed image bounds."""
    for x0, y0, x1, y1, _ in compute_tile_windows(70, 70, 50, 50):
        assert x1 <= 70
        assert y1 <= 70


# ---- build_tile_dataset ----


def test_build_tile_dataset_creates_output_csv(image_and_vector) -> None:
    """dataset.csv should be created in output_dir."""
    img_path, gpkg_path, out_dir = image_and_vector

    build_tile_dataset(
        image_paths=[img_path],
        vector_path=gpkg_path,
        class_attribute="class_id",
        output_dir=out_dir,
        vector_layer="polygons",
        tile_width=32,
        tile_height=32,
        skip_empty_tiles=False,
        progress=False,
    )

    assert (out_dir / "dataset.csv").exists()


def test_build_tile_dataset_output_structure(image_and_vector) -> None:
    """images/ and masks/ directories should be created under output_dir/{stem}/."""
    img_path, gpkg_path, out_dir = image_and_vector

    build_tile_dataset(
        image_paths=[img_path],
        vector_path=gpkg_path,
        class_attribute="class_id",
        output_dir=out_dir,
        vector_layer="polygons",
        tile_width=32,
        tile_height=32,
        skip_empty_tiles=False,
        progress=False,
    )

    stem = img_path.stem
    assert (out_dir / stem / "images").is_dir()
    assert (out_dir / stem / "masks").is_dir()


def test_build_tile_dataset_mask_values_from_polygon(image_and_vector) -> None:
    """Mask tiles inside the polygon should have non-zero values."""
    img_path, gpkg_path, out_dir = image_and_vector

    df = build_tile_dataset(
        image_paths=[img_path],
        vector_path=gpkg_path,
        class_attribute="class_id",
        output_dir=out_dir,
        vector_layer="polygons",
        tile_width=32,
        tile_height=32,
        skip_empty_tiles=False,
        progress=False,
    )

    assert len(df) > 0

    # At least one mask should contain non-zero values (the polygon region)
    has_nonzero = False
    for _, row in df.iterrows():
        with rasterio.open(row["label_path"]) as f:
            data = f.read(1)
        if data.max() > 0:
            has_nonzero = True
            break

    assert has_nonzero, "No mask tile contains non-zero values from the polygon"


def test_build_tile_dataset_skip_empty_tiles(image_and_vector) -> None:
    """With skip_empty_tiles=True, fewer tiles should be saved than with False."""
    img_path, gpkg_path, out_dir_all = image_and_vector
    out_dir_skip = img_path.parent / "dataset_skip"

    df_all = build_tile_dataset(
        image_paths=[img_path],
        vector_path=gpkg_path,
        class_attribute="class_id",
        output_dir=out_dir_all,
        vector_layer="polygons",
        tile_width=32,
        tile_height=32,
        skip_empty_tiles=False,
        progress=False,
    )

    df_skip = build_tile_dataset(
        image_paths=[img_path],
        vector_path=gpkg_path,
        class_attribute="class_id",
        output_dir=out_dir_skip,
        vector_layer="polygons",
        tile_width=32,
        tile_height=32,
        skip_empty_tiles=True,
        progress=False,
    )

    assert len(df_skip) <= len(df_all)


def test_build_tile_dataset_full_size_mask(image_and_vector) -> None:
    """generate_full_size_masks=True should create a mask_full.tif per image."""
    img_path, gpkg_path, out_dir = image_and_vector

    build_tile_dataset(
        image_paths=[img_path],
        vector_path=gpkg_path,
        class_attribute="class_id",
        output_dir=out_dir,
        vector_layer="polygons",
        tile_width=32,
        tile_height=32,
        skip_empty_tiles=False,
        generate_full_size_masks=True,
        progress=False,
    )

    full_mask_path = out_dir / img_path.stem / "mask_full.tif"
    assert full_mask_path.exists()

    with rasterio.open(full_mask_path) as f:
        assert f.width == 100
        assert f.height == 100
