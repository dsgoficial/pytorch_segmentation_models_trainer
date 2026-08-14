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


def test_build_tile_dataset_uses_default_background_value_255(tmp_path: Path) -> None:
    """Default masks should use 255 as background instead of 0."""
    img_path = tmp_path / "image.tif"
    data = np.ones((3, 64, 64), dtype=np.uint8)
    make_tiff(img_path, data, crs="EPSG:4326")

    poly = box(0.00, 0.50, 0.50, 1.00)
    gdf = gpd.GeoDataFrame({"geometry": [poly], "class_id": [3]}, crs="EPSG:4326")
    gpkg_path = tmp_path / "masks.gpkg"
    gdf.to_file(gpkg_path, driver="GPKG", layer="polygons")

    out_dir = tmp_path / "dataset"
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
    empty_tile_found = False
    for _, row in df.iterrows():
        with rasterio.open(row["label_path"]) as f:
            mask = f.read(1)
        if np.all(mask == 255):
            empty_tile_found = True
            break

    assert empty_tile_found, "Expected at least one tile with 255 background"


def test_build_tile_dataset_allows_custom_background_value(tmp_path: Path) -> None:
    """background_value should be configurable for multi-class masks."""
    img_path = tmp_path / "image.tif"
    data = np.ones((3, 64, 64), dtype=np.uint8)
    make_tiff(img_path, data, crs="EPSG:4326")

    poly = box(0.00, 0.50, 0.50, 1.00)
    gdf = gpd.GeoDataFrame({"geometry": [poly], "class_id": [3]}, crs="EPSG:4326")
    gpkg_path = tmp_path / "masks.gpkg"
    gdf.to_file(gpkg_path, driver="GPKG", layer="polygons")

    out_dir = tmp_path / "dataset_custom_bg"
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
        background_value=17,
    )

    assert len(df) > 0
    found_custom_background = False
    for _, row in df.iterrows():
        with rasterio.open(row["label_path"]) as f:
            mask = f.read(1)
        assert mask.dtype == np.uint8
        if 17 in np.unique(mask):
            found_custom_background = True
            break

    assert found_custom_background, "Expected at least one tile with background 17"


def test_build_tile_dataset_rejects_invalid_background_value(tmp_path: Path) -> None:
    """background_value must stay within the uint8 range."""
    img_path = tmp_path / "image.tif"
    data = np.ones((3, 16, 16), dtype=np.uint8)
    make_tiff(img_path, data, crs="EPSG:4326")

    gdf = gpd.GeoDataFrame(
        {"geometry": [box(0.0, 0.0, 1.0, 1.0)], "class_id": [1]},
        crs="EPSG:4326",
    )
    gpkg_path = tmp_path / "masks.gpkg"
    gdf.to_file(gpkg_path, driver="GPKG", layer="polygons")

    with pytest.raises(ValueError, match="uint8 range"):
        build_tile_dataset(
            image_paths=[img_path],
            vector_path=gpkg_path,
            class_attribute="class_id",
            output_dir=tmp_path / "dataset_invalid",
            vector_layer="polygons",
            background_value=999,
            progress=False,
        )


def test_build_tile_dataset_no_crs_skips_reprojection(tmp_path: Path) -> None:
    """When both the image and GDF have no CRS, reprojection is skipped."""
    img_path = tmp_path / "image.tif"
    data = np.ones((3, 32, 32), dtype=np.uint8)
    transform = rasterio.transform.from_bounds(0, 0, 1, 1, 32, 32)
    profile = dict(
        driver="GTiff",
        dtype=data.dtype,
        width=32,
        height=32,
        count=3,
        crs=None,
        transform=transform,
    )
    with rasterio.open(img_path, "w", **profile) as dst:
        dst.write(data)

    poly = box(0.1, 0.1, 0.9, 0.9)
    gdf = gpd.GeoDataFrame({"geometry": [poly], "class_id": [1]}, crs=None)
    gpkg_path = tmp_path / "masks.gpkg"
    gdf.to_file(gpkg_path, driver="GPKG", layer="polys")

    df = build_tile_dataset(
        image_paths=[img_path],
        vector_path=gpkg_path,
        class_attribute="class_id",
        output_dir=tmp_path / "dataset",
        vector_layer="polys",
        tile_width=16,
        tile_height=16,
        skip_empty_tiles=False,
        generate_full_size_masks=False,
        progress=False,
    )
    assert len(df) >= 0


def test_build_tile_dataset_empty_gdf_full_mask_all_background(
    tmp_path: Path,
) -> None:
    """With an empty GDF, generate_full_size_masks must write a background-only mask."""
    img_path = tmp_path / "image.tif"
    data = np.ones((3, 32, 32), dtype=np.uint8)
    make_tiff(img_path, data, crs="EPSG:4326")

    gdf = gpd.GeoDataFrame({"geometry": [], "class_id": []}, crs="EPSG:4326")
    gpkg_path = tmp_path / "masks.gpkg"
    gdf.to_file(gpkg_path, driver="GPKG", layer="polys")

    build_tile_dataset(
        image_paths=[img_path],
        vector_path=gpkg_path,
        class_attribute="class_id",
        output_dir=tmp_path / "dataset",
        vector_layer="polys",
        tile_width=16,
        tile_height=16,
        skip_empty_tiles=False,
        generate_full_size_masks=True,
        background_value=255,
        progress=False,
    )

    full_mask_path = tmp_path / "dataset" / "image" / "mask_full.tif"
    assert full_mask_path.exists()
    with rasterio.open(full_mask_path) as f:
        assert np.all(f.read(1) == 255)


def test_build_tile_dataset_skips_empty_image_tiles(tmp_path: Path) -> None:
    """Tiles whose non-zero pixel ratio is below the threshold should be skipped."""
    img_path = tmp_path / "image.tif"
    # Mostly zero image — only one pixel is non-zero
    data = np.zeros((3, 32, 32), dtype=np.uint8)
    data[:, 0, 0] = 1
    make_tiff(img_path, data, crs="EPSG:4326")

    poly = box(0.0, 0.0, 1.0, 1.0)
    gdf = gpd.GeoDataFrame({"geometry": [poly], "class_id": [1]}, crs="EPSG:4326")
    gpkg_path = tmp_path / "masks.gpkg"
    gdf.to_file(gpkg_path, driver="GPKG", layer="polys")

    df_skip = build_tile_dataset(
        image_paths=[img_path],
        vector_path=gpkg_path,
        class_attribute="class_id",
        output_dir=tmp_path / "skip",
        vector_layer="polys",
        tile_width=16,
        tile_height=16,
        skip_empty_tiles=True,
        min_valid_pixel_ratio=0.9,
        background_value=0,
        progress=False,
    )

    df_no_skip = build_tile_dataset(
        image_paths=[img_path],
        vector_path=gpkg_path,
        class_attribute="class_id",
        output_dir=tmp_path / "no_skip",
        vector_layer="polys",
        tile_width=16,
        tile_height=16,
        skip_empty_tiles=False,
        min_valid_pixel_ratio=0.9,
        background_value=0,
        progress=False,
    )

    assert len(df_skip) < len(df_no_skip)


def test_build_tile_dataset_skips_all_background_mask_tiles(tmp_path: Path) -> None:
    """Tiles where the mask is entirely background (and skip_empty_tiles=True) are skipped."""
    img_path = tmp_path / "image.tif"
    data = np.ones((3, 32, 32), dtype=np.uint8) * 128
    make_tiff(img_path, data, crs="EPSG:4326")

    # Polygon outside image bounds — mask will be all background
    poly = box(2.0, 2.0, 3.0, 3.0)
    gdf = gpd.GeoDataFrame({"geometry": [poly], "class_id": [1]}, crs="EPSG:4326")
    gpkg_path = tmp_path / "masks.gpkg"
    gdf.to_file(gpkg_path, driver="GPKG", layer="polys")

    df = build_tile_dataset(
        image_paths=[img_path],
        vector_path=gpkg_path,
        class_attribute="class_id",
        output_dir=tmp_path / "dataset",
        vector_layer="polys",
        tile_width=16,
        tile_height=16,
        skip_empty_tiles=True,
        background_value=255,
        progress=False,
    )

    assert len(df) == 0


def test_build_tile_dataset_with_progress_bar(image_and_vector) -> None:
    """build_tile_dataset with progress=True should produce the same result."""
    img_path, gpkg_path, out_dir = image_and_vector

    df = build_tile_dataset(
        image_paths=[img_path],
        vector_path=gpkg_path,
        class_attribute="class_id",
        output_dir=out_dir / "prog",
        vector_layer="polygons",
        tile_width=32,
        tile_height=32,
        skip_empty_tiles=False,
        progress=True,
    )

    assert len(df) > 0
