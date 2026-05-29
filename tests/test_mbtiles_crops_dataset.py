# -*- coding: utf-8 -*-
"""Tests for MBTilesCropsGeoTifMaskDataset."""

import numpy as np
import pandas as pd
import pytest
import rasterio
import torch
from rasterio.crs import CRS
from rasterio.transform import from_bounds

from pytorch_segmentation_models_trainer.dataset_loader.mbtiles_crops_dataset import (
    MBTilesCropsGeoTifMaskDataset,
)

# ---------------------------------------------------------------------------
# Raster geometry constants
# ---------------------------------------------------------------------------

EPSG4326 = CRS.from_epsg(4326)
RASTER_W, RASTER_H = 512, 512
LON_MIN, LAT_MIN = -54.0, -15.0
PIXEL_SIZE = 0.0001
LON_MAX = LON_MIN + RASTER_W * PIXEL_SIZE
LAT_MAX = LAT_MIN + RASTER_H * PIXEL_SIZE
PATCH_SIZE = 256

COLOR_MAP = [[255, 0, 0, 1], [0, 255, 0, 2], [0, 0, 255, 3]]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_transform(
    w=RASTER_W,
    h=RASTER_H,
    lon_min=LON_MIN,
    lat_min=LAT_MIN,
    lon_max=LON_MAX,
    lat_max=LAT_MAX,
):
    return from_bounds(lon_min, lat_min, lon_max, lat_max, w, h)


def _write_raster(path, data, crs=EPSG4326, transform=None):
    if transform is None:
        transform = _make_transform()
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=data.shape[1],
        width=data.shape[2],
        count=data.shape[0],
        dtype=data.dtype,
        crs=crs,
        transform=transform,
    ) as dst:
        dst.write(data)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def raster_pair(tmp_path):
    """512×512 image (3-band RGB) + 3-band RGB mask with 4 colour quadrants.

    Quadrant layout (each 256×256):
      Q0 top-left     red   (255,0,0) → class 1
      Q1 top-right  green   (0,255,0) → class 2
      Q2 bottom-left  blue  (0,0,255) → class 3
      Q3 bottom-right black (0,0,0)   → class 0
    """
    transform = _make_transform()
    rng = np.random.default_rng(0)
    img_data = rng.integers(0, 255, (3, RASTER_H, RASTER_W), dtype=np.uint8)
    img_path = tmp_path / "image.tif"
    _write_raster(img_path, img_data, transform=transform)

    mask_data = np.zeros((3, RASTER_H, RASTER_W), dtype=np.uint8)
    mask_data[:, :256, :256] = np.array([[[255]], [[0]], [[0]]])
    mask_data[:, :256, 256:] = np.array([[[0]], [[255]], [[0]]])
    mask_data[:, 256:, :256] = np.array([[[0]], [[0]], [[255]]])
    mask_path = tmp_path / "mask.tif"
    _write_raster(mask_path, mask_data, transform=transform)

    return img_path, mask_path


@pytest.fixture()
def single_band_mask(tmp_path):
    """512×512 image + single-band integer mask (foreground=1 in top half)."""
    transform = _make_transform()
    rng = np.random.default_rng(1)
    img_data = rng.integers(0, 255, (3, RASTER_H, RASTER_W), dtype=np.uint8)
    img_path = tmp_path / "image_sb.tif"
    _write_raster(img_path, img_data, transform=transform)

    mask_data = np.zeros((1, RASTER_H, RASTER_W), dtype=np.uint8)
    mask_data[0, :256, :] = 1
    mask_path = tmp_path / "mask_sb.tif"
    _write_raster(mask_path, mask_data, transform=transform)

    return img_path, mask_path


@pytest.fixture()
def csv_crops(tmp_path):
    """CSV with two 256×256 windows: Q0 (0,0) and Q3 (256,256)."""
    df = pd.DataFrame({"row_off": [0, 256], "col_off": [0, 256]})
    path = tmp_path / "crops.csv"
    df.to_csv(path, index=False)
    return path


@pytest.fixture()
def parquet_crops(tmp_path):
    """Parquet equivalent of csv_crops."""
    df = pd.DataFrame({"row_off": [0, 256], "col_off": [0, 256]})
    path = tmp_path / "crops.parquet"
    df.to_parquet(path, index=False)
    return path


@pytest.fixture()
def vector_crops(tmp_path):
    """GeoJSON with two point-like boxes whose centres map to Q0 and Q3."""
    import geopandas as gpd
    from shapely.geometry import box as shapely_box

    # Q0 centre in geo-coords
    q0_lon = LON_MIN + 128 * PIXEL_SIZE
    q0_lat = LAT_MIN + (RASTER_H - 128) * PIXEL_SIZE
    q0_box = shapely_box(
        q0_lon - PIXEL_SIZE,
        q0_lat - PIXEL_SIZE,
        q0_lon + PIXEL_SIZE,
        q0_lat + PIXEL_SIZE,
    )
    # Q3 centre in geo-coords
    q3_lon = LON_MIN + (256 + 128) * PIXEL_SIZE
    q3_lat = LAT_MIN + (RASTER_H - 256 - 128) * PIXEL_SIZE
    q3_box = shapely_box(
        q3_lon - PIXEL_SIZE,
        q3_lat - PIXEL_SIZE,
        q3_lon + PIXEL_SIZE,
        q3_lat + PIXEL_SIZE,
    )
    path = tmp_path / "crops.geojson"
    gpd.GeoDataFrame(geometry=[q0_box, q3_box], crs="EPSG:4326").to_file(
        path, driver="GeoJSON"
    )
    return path


# ---------------------------------------------------------------------------
# Construction & validation
# ---------------------------------------------------------------------------


class TestConstruction:
    def test_csv_source_two_windows(self, raster_pair, csv_crops):
        img, mask = raster_pair
        ds = MBTilesCropsGeoTifMaskDataset(
            image_mbtiles_path=img,
            mask_path=mask,
            crops_path=csv_crops,
            patch_size=PATCH_SIZE,
            color_map=COLOR_MAP,
        )
        assert len(ds) == 2

    def test_parquet_source(self, raster_pair, parquet_crops):
        img, mask = raster_pair
        ds = MBTilesCropsGeoTifMaskDataset(
            image_mbtiles_path=img,
            mask_path=mask,
            crops_path=parquet_crops,
            patch_size=PATCH_SIZE,
            color_map=COLOR_MAP,
        )
        assert len(ds) == 2

    def test_vector_source_two_windows(self, raster_pair, vector_crops):
        img, mask = raster_pair
        ds = MBTilesCropsGeoTifMaskDataset(
            image_mbtiles_path=img,
            mask_path=mask,
            crops_path=vector_crops,
            patch_size=PATCH_SIZE,
            color_map=COLOR_MAP,
        )
        assert len(ds) == 2

    def test_invalid_image_dtype_raises(self, raster_pair, csv_crops):
        img, mask = raster_pair
        with pytest.raises(ValueError, match="image_dtype"):
            MBTilesCropsGeoTifMaskDataset(
                image_mbtiles_path=img,
                mask_path=mask,
                crops_path=csv_crops,
                patch_size=PATCH_SIZE,
                color_map=COLOR_MAP,
                image_dtype="bad",
            )

    def test_invalid_patch_size_raises(self, raster_pair, csv_crops):
        img, mask = raster_pair
        with pytest.raises(ValueError, match="patch_size"):
            MBTilesCropsGeoTifMaskDataset(
                image_mbtiles_path=img,
                mask_path=mask,
                crops_path=csv_crops,
                patch_size=0,
                color_map=COLOR_MAP,
            )

    def test_multiband_mask_without_color_map_raises(self, raster_pair, csv_crops):
        img, mask = raster_pair  # mask is 3-band
        with pytest.raises(ValueError, match="color_map"):
            MBTilesCropsGeoTifMaskDataset(
                image_mbtiles_path=img,
                mask_path=mask,
                crops_path=csv_crops,
                patch_size=PATCH_SIZE,
            )

    def test_single_band_mask_no_color_map_ok(self, single_band_mask, csv_crops):
        img, mask = single_band_mask
        ds = MBTilesCropsGeoTifMaskDataset(
            image_mbtiles_path=img,
            mask_path=mask,
            crops_path=csv_crops,
            patch_size=PATCH_SIZE,
        )
        assert len(ds) == 2

    def test_missing_csv_columns_raises(self, raster_pair, tmp_path):
        img, mask = raster_pair
        bad_csv = tmp_path / "bad.csv"
        pd.DataFrame({"x": [0, 1]}).to_csv(bad_csv, index=False)
        with pytest.raises(ValueError, match="missing required columns"):
            MBTilesCropsGeoTifMaskDataset(
                image_mbtiles_path=img,
                mask_path=mask,
                crops_path=bad_csv,
                patch_size=PATCH_SIZE,
                color_map=COLOR_MAP,
            )

    def test_kwargs_ignored(self, raster_pair, csv_crops):
        img, mask = raster_pair
        ds = MBTilesCropsGeoTifMaskDataset(
            image_mbtiles_path=img,
            mask_path=mask,
            crops_path=csv_crops,
            patch_size=PATCH_SIZE,
            color_map=COLOR_MAP,
            unknown_param="ignored",
        )
        assert len(ds) == 2


# ---------------------------------------------------------------------------
# __getitem__ — output contract
# ---------------------------------------------------------------------------


class TestGetItem:
    def test_image_shape_and_dtype(self, raster_pair, csv_crops):
        img, mask = raster_pair
        ds = MBTilesCropsGeoTifMaskDataset(
            image_mbtiles_path=img,
            mask_path=mask,
            crops_path=csv_crops,
            patch_size=PATCH_SIZE,
            color_map=COLOR_MAP,
        )
        sample = ds[0]
        assert sample["image"].shape == (3, PATCH_SIZE, PATCH_SIZE)
        assert sample["image"].dtype == torch.float32

    def test_mask_shape_and_dtype(self, raster_pair, csv_crops):
        img, mask = raster_pair
        ds = MBTilesCropsGeoTifMaskDataset(
            image_mbtiles_path=img,
            mask_path=mask,
            crops_path=csv_crops,
            patch_size=PATCH_SIZE,
            color_map=COLOR_MAP,
        )
        sample = ds[0]
        assert sample["mask"].shape == (PATCH_SIZE, PATCH_SIZE)
        assert sample["mask"].dtype == torch.int64

    def test_color_map_q0_is_class1(self, raster_pair, csv_crops):
        """Q0 is solid red → class 1 after color_map."""
        img, mask = raster_pair
        ds = MBTilesCropsGeoTifMaskDataset(
            image_mbtiles_path=img,
            mask_path=mask,
            crops_path=csv_crops,
            patch_size=PATCH_SIZE,
            color_map=COLOR_MAP,
        )
        sample = ds[0]  # Q0 window (row_off=0, col_off=0)
        assert torch.all(sample["mask"] == 1)

    def test_color_map_q3_is_class0(self, raster_pair, csv_crops):
        """Q3 is black → class 0 (background)."""
        img, mask = raster_pair
        ds = MBTilesCropsGeoTifMaskDataset(
            image_mbtiles_path=img,
            mask_path=mask,
            crops_path=csv_crops,
            patch_size=PATCH_SIZE,
            color_map=COLOR_MAP,
        )
        sample = ds[1]  # Q3 window (row_off=256, col_off=256)
        assert torch.all(sample["mask"] == 0)

    def test_single_band_mask_binarized(self, single_band_mask, csv_crops):
        """Single-band mask, n_classes=2: top half → 1, bottom half → 0."""
        img, mask = single_band_mask
        ds = MBTilesCropsGeoTifMaskDataset(
            image_mbtiles_path=img,
            mask_path=mask,
            crops_path=csv_crops,
            patch_size=PATCH_SIZE,
            n_classes=2,
        )
        top = ds[0]  # row_off=0 → foreground
        bottom = ds[1]  # row_off=256 → background
        assert torch.all(top["mask"] == 1)
        assert torch.all(bottom["mask"] == 0)

    def test_index_out_of_range_raises(self, raster_pair, csv_crops):
        img, mask = raster_pair
        ds = MBTilesCropsGeoTifMaskDataset(
            image_mbtiles_path=img,
            mask_path=mask,
            crops_path=csv_crops,
            patch_size=PATCH_SIZE,
            color_map=COLOR_MAP,
        )
        with pytest.raises(IndexError):
            _ = ds[99]

    def test_no_metadata_by_default(self, raster_pair, csv_crops):
        img, mask = raster_pair
        ds = MBTilesCropsGeoTifMaskDataset(
            image_mbtiles_path=img,
            mask_path=mask,
            crops_path=csv_crops,
            patch_size=PATCH_SIZE,
            color_map=COLOR_MAP,
        )
        assert "metadata" not in ds[0]

    def test_return_metadata(self, raster_pair, csv_crops):
        img, mask = raster_pair
        ds = MBTilesCropsGeoTifMaskDataset(
            image_mbtiles_path=img,
            mask_path=mask,
            crops_path=csv_crops,
            patch_size=PATCH_SIZE,
            color_map=COLOR_MAP,
            return_metadata=True,
        )
        meta = ds[0]["metadata"]
        assert meta["row_off"] == 0
        assert meta["col_off"] == 0

    def test_image_normalized_uint8(self, raster_pair, csv_crops):
        """Without augmentations, uint8 image values are in [0, 1]."""
        img, mask = raster_pair
        ds = MBTilesCropsGeoTifMaskDataset(
            image_mbtiles_path=img,
            mask_path=mask,
            crops_path=csv_crops,
            patch_size=PATCH_SIZE,
            color_map=COLOR_MAP,
            image_dtype="uint8",
        )
        t = ds[0]["image"]
        assert t.min() >= 0.0
        assert t.max() <= 1.0

    def test_selected_bands(self, raster_pair, csv_crops):
        """selected_bands=[1] → single-channel image."""
        img, mask = raster_pair
        ds = MBTilesCropsGeoTifMaskDataset(
            image_mbtiles_path=img,
            mask_path=mask,
            crops_path=csv_crops,
            patch_size=PATCH_SIZE,
            color_map=COLOR_MAP,
            selected_bands=[1],
        )
        assert ds[0]["image"].shape == (1, PATCH_SIZE, PATCH_SIZE)


# ---------------------------------------------------------------------------
# CRS and resolution mismatch
# ---------------------------------------------------------------------------


class TestCRSAndResolution:
    def test_mask_different_crs(self, tmp_path, csv_crops):
        """Mask in EPSG:3857 is automatically reprojected to image CRS."""
        import pyproj
        from rasterio.crs import CRS as RioCRS

        transform_4326 = _make_transform()
        rng = np.random.default_rng(2)
        img_data = rng.integers(0, 255, (3, RASTER_H, RASTER_W), dtype=np.uint8)
        img_path = tmp_path / "img_crs.tif"
        _write_raster(img_path, img_data, crs=EPSG4326, transform=transform_4326)

        # Build a mask in EPSG:3857 covering the same geographic area
        transformer = pyproj.Transformer.from_crs(
            "EPSG:4326", "EPSG:3857", always_xy=True
        )
        x_min, y_min = transformer.transform(LON_MIN, LAT_MIN)
        x_max, y_max = transformer.transform(LON_MAX, LAT_MAX)
        transform_3857 = from_bounds(x_min, y_min, x_max, y_max, RASTER_W, RASTER_H)
        mask_data = np.zeros((1, RASTER_H, RASTER_W), dtype=np.uint8)
        mask_data[0, :256, :] = 1
        mask_path = tmp_path / "mask_3857.tif"
        _write_raster(
            mask_path, mask_data, crs=RioCRS.from_epsg(3857), transform=transform_3857
        )

        ds = MBTilesCropsGeoTifMaskDataset(
            image_mbtiles_path=img_path,
            mask_path=mask_path,
            crops_path=csv_crops,
            patch_size=PATCH_SIZE,
        )
        sample = ds[0]
        assert sample["image"].shape == (3, PATCH_SIZE, PATCH_SIZE)
        assert sample["mask"].shape == (PATCH_SIZE, PATCH_SIZE)

    def test_mask_different_resolution(self, tmp_path, csv_crops):
        """Mask at half resolution is resampled to patch_size output."""
        transform_img = _make_transform()
        rng = np.random.default_rng(3)
        img_data = rng.integers(0, 255, (3, RASTER_H, RASTER_W), dtype=np.uint8)
        img_path = tmp_path / "img_res.tif"
        _write_raster(img_path, img_data, transform=transform_img)

        # Mask at half the spatial resolution (256×256 pixels same extent)
        transform_half = _make_transform(w=256, h=256)
        mask_data = np.zeros((1, 256, 256), dtype=np.uint8)
        mask_data[0, :128, :] = 1
        mask_path = tmp_path / "mask_half_res.tif"
        _write_raster(mask_path, mask_data, transform=transform_half)

        ds = MBTilesCropsGeoTifMaskDataset(
            image_mbtiles_path=img_path,
            mask_path=mask_path,
            crops_path=csv_crops,
            patch_size=PATCH_SIZE,
        )
        sample = ds[0]
        assert sample["mask"].shape == (PATCH_SIZE, PATCH_SIZE)


# ---------------------------------------------------------------------------
# Vector crops
# ---------------------------------------------------------------------------


class TestVectorCrops:
    def test_vector_crops_returns_fixed_patch(self, raster_pair, vector_crops):
        img, mask = raster_pair
        ds = MBTilesCropsGeoTifMaskDataset(
            image_mbtiles_path=img,
            mask_path=mask,
            crops_path=vector_crops,
            patch_size=PATCH_SIZE,
            color_map=COLOR_MAP,
        )
        sample = ds[0]
        assert sample["image"].shape == (3, PATCH_SIZE, PATCH_SIZE)
        assert sample["mask"].shape == (PATCH_SIZE, PATCH_SIZE)

    def test_vector_crops_different_crs(self, tmp_path, raster_pair):
        """Vector file in EPSG:3857 is reprojected to image CRS."""
        import geopandas as gpd
        from shapely.geometry import box as shapely_box
        from rasterio.crs import CRS as RioCRS

        img, mask = raster_pair
        # Build box in EPSG:3857 at Q0 centre
        import pyproj

        transformer = pyproj.Transformer.from_crs(
            "EPSG:4326", "EPSG:3857", always_xy=True
        )
        cx, cy = transformer.transform(
            LON_MIN + 128 * PIXEL_SIZE, LAT_MIN + (RASTER_H - 128) * PIXEL_SIZE
        )
        box_3857 = shapely_box(cx - 1, cy - 1, cx + 1, cy + 1)
        path = tmp_path / "crops_3857.geojson"
        gpd.GeoDataFrame(geometry=[box_3857], crs="EPSG:3857").to_file(
            path, driver="GeoJSON"
        )

        ds = MBTilesCropsGeoTifMaskDataset(
            image_mbtiles_path=img,
            mask_path=mask,
            crops_path=path,
            patch_size=PATCH_SIZE,
            color_map=COLOR_MAP,
        )
        assert len(ds) == 1
        sample = ds[0]
        assert sample["image"].shape == (3, PATCH_SIZE, PATCH_SIZE)

    def test_vector_clamp_to_raster_bounds(self, tmp_path, raster_pair):
        """Feature centroid near raster edge is clamped so window stays inside."""
        import geopandas as gpd
        from shapely.geometry import box as shapely_box

        img, mask = raster_pair
        # Feature centred very close to the top-left corner (would go negative)
        cx = LON_MIN + 2 * PIXEL_SIZE
        cy = LAT_MAX - 2 * PIXEL_SIZE
        edge_box = shapely_box(
            cx - PIXEL_SIZE, cy - PIXEL_SIZE, cx + PIXEL_SIZE, cy + PIXEL_SIZE
        )
        path = tmp_path / "crops_edge.geojson"
        gpd.GeoDataFrame(geometry=[edge_box], crs="EPSG:4326").to_file(
            path, driver="GeoJSON"
        )

        ds = MBTilesCropsGeoTifMaskDataset(
            image_mbtiles_path=img,
            mask_path=mask,
            crops_path=path,
            patch_size=PATCH_SIZE,
            color_map=COLOR_MAP,
        )
        rec = ds._index[0]
        assert rec["row_off"] >= 0
        assert rec["col_off"] >= 0
        assert rec["row_off"] + PATCH_SIZE <= RASTER_H
        assert rec["col_off"] + PATCH_SIZE <= RASTER_W


# ---------------------------------------------------------------------------
# Window index cache
# ---------------------------------------------------------------------------


class TestWindowIndexCache:
    def test_cache_written_and_reloaded(self, raster_pair, csv_crops, tmp_path):
        img, mask = raster_pair
        cache = tmp_path / "cache.csv"
        ds1 = MBTilesCropsGeoTifMaskDataset(
            image_mbtiles_path=img,
            mask_path=mask,
            crops_path=csv_crops,
            patch_size=PATCH_SIZE,
            color_map=COLOR_MAP,
            window_index_cache=cache,
        )
        assert cache.exists()
        ds2 = MBTilesCropsGeoTifMaskDataset(
            image_mbtiles_path=img,
            mask_path=mask,
            crops_path=csv_crops,
            patch_size=PATCH_SIZE,
            color_map=COLOR_MAP,
            window_index_cache=cache,
        )
        assert ds1._index == ds2._index

    def test_invalid_cache_extension_raises(self, raster_pair, csv_crops, tmp_path):
        img, mask = raster_pair
        with pytest.raises(ValueError):
            MBTilesCropsGeoTifMaskDataset(
                image_mbtiles_path=img,
                mask_path=mask,
                crops_path=csv_crops,
                patch_size=PATCH_SIZE,
                color_map=COLOR_MAP,
                window_index_cache=tmp_path / "bad.txt",
            )


# ---------------------------------------------------------------------------
# Augmentation pipeline
# ---------------------------------------------------------------------------


class TestAugmentations:
    def test_augmentation_list_applied(self, raster_pair, csv_crops):
        """Augmentation pipeline runs and output tensors have correct shapes."""
        img, mask = raster_pair
        aug = [
            {"_target_": "albumentations.HorizontalFlip", "p": 0.0},
            {"_target_": "albumentations.pytorch.ToTensorV2"},
        ]
        ds = MBTilesCropsGeoTifMaskDataset(
            image_mbtiles_path=img,
            mask_path=mask,
            crops_path=csv_crops,
            patch_size=PATCH_SIZE,
            color_map=COLOR_MAP,
            augmentation_list=aug,
        )
        sample = ds[0]
        assert sample["image"].shape == (3, PATCH_SIZE, PATCH_SIZE)
        assert sample["mask"].dtype == torch.int64

    def test_augmentation_without_to_tensor_v2(self, raster_pair, csv_crops):
        """Augmentation that returns numpy mask is converted to tensor (defensive branch)."""
        img, mask = raster_pair
        aug = [{"_target_": "albumentations.HorizontalFlip", "p": 0.0}]
        ds = MBTilesCropsGeoTifMaskDataset(
            image_mbtiles_path=img,
            mask_path=mask,
            crops_path=csv_crops,
            patch_size=PATCH_SIZE,
            color_map=COLOR_MAP,
            augmentation_list=aug,
        )
        sample = ds[0]
        assert isinstance(sample["mask"], torch.Tensor)
        assert sample["mask"].dtype == torch.int64


# ---------------------------------------------------------------------------
# Edge cases in _load_window_index
# ---------------------------------------------------------------------------


class TestWindowIndexEdgeCases:
    def test_unknown_extension_reads_as_tabular(self, raster_pair, tmp_path):
        """File with unknown extension that is valid CSV is read correctly."""
        img, mask = raster_pair
        # Write a valid CSV with a non-standard extension
        df = pd.DataFrame({"row_off": [0], "col_off": [0]})
        unknown_path = tmp_path / "crops.dat"
        df.to_csv(unknown_path, index=False)
        ds = MBTilesCropsGeoTifMaskDataset(
            image_mbtiles_path=img,
            mask_path=mask,
            crops_path=unknown_path,
            patch_size=PATCH_SIZE,
            color_map=COLOR_MAP,
        )
        assert len(ds) == 1

    def test_vector_no_crs_raises(self, raster_pair, tmp_path):
        """Shapefile without .prj (CRS=None) raises ValueError."""
        import geopandas as gpd
        from shapely.geometry import box as shapely_box

        img, mask = raster_pair
        shp_path = tmp_path / "no_crs.shp"
        gdf = gpd.GeoDataFrame(
            geometry=[shapely_box(LON_MIN, LAT_MIN, LON_MAX, LAT_MAX)]
        )
        # Write without CRS so .prj is absent
        gdf.to_file(shp_path, driver="ESRI Shapefile")
        prj = tmp_path / "no_crs.prj"
        if prj.exists():
            prj.unlink()

        with pytest.raises(ValueError, match="no CRS"):
            MBTilesCropsGeoTifMaskDataset(
                image_mbtiles_path=img,
                mask_path=mask,
                crops_path=shp_path,
                patch_size=PATCH_SIZE,
                color_map=COLOR_MAP,
            )

    def test_vector_empty_geometry_skipped(self, raster_pair, tmp_path):
        """Features with empty geometry are silently skipped."""
        import geopandas as gpd
        from shapely.geometry import box as shapely_box
        from shapely import wkt

        img, mask = raster_pair
        valid_box = shapely_box(
            LON_MIN + 128 * PIXEL_SIZE - PIXEL_SIZE,
            LAT_MIN + (RASTER_H - 128) * PIXEL_SIZE - PIXEL_SIZE,
            LON_MIN + 128 * PIXEL_SIZE + PIXEL_SIZE,
            LAT_MIN + (RASTER_H - 128) * PIXEL_SIZE + PIXEL_SIZE,
        )
        empty_geom = wkt.loads("POLYGON EMPTY")
        path = tmp_path / "mixed.geojson"
        gpd.GeoDataFrame(geometry=[valid_box, empty_geom], crs="EPSG:4326").to_file(
            path, driver="GeoJSON"
        )

        ds = MBTilesCropsGeoTifMaskDataset(
            image_mbtiles_path=img,
            mask_path=mask,
            crops_path=path,
            patch_size=PATCH_SIZE,
            color_map=COLOR_MAP,
        )
        assert len(ds) == 1

    def test_vector_no_valid_features_logs_warning(self, raster_pair, tmp_path, caplog):
        """Vector file with only empty geometries logs a warning."""
        import geopandas as gpd
        from shapely import wkt

        img, mask = raster_pair
        empty_geom = wkt.loads("POLYGON EMPTY")
        path = tmp_path / "all_empty.geojson"
        gpd.GeoDataFrame(geometry=[empty_geom], crs="EPSG:4326").to_file(
            path, driver="GeoJSON"
        )

        import logging

        with caplog.at_level(logging.WARNING):
            ds = MBTilesCropsGeoTifMaskDataset(
                image_mbtiles_path=img,
                mask_path=mask,
                crops_path=path,
                patch_size=PATCH_SIZE,
                color_map=COLOR_MAP,
            )
        assert len(ds) == 0
        assert "no valid windows" in caplog.text.lower()

    def test_unknown_extension_csv_missing_columns_fallback(
        self, raster_pair, tmp_path
    ):
        """Unknown-ext CSV with wrong columns triggers fallback to vector path."""
        import geopandas as gpd
        from shapely.geometry import box as shapely_box
        import shutil

        img, mask = raster_pair
        # Write a valid GeoJSON so vector fallback succeeds
        geojson_src = tmp_path / "valid.geojson"
        box = shapely_box(
            LON_MIN + 10 * PIXEL_SIZE,
            LAT_MIN + (RASTER_H - 10) * PIXEL_SIZE - PIXEL_SIZE,
            LON_MIN + 12 * PIXEL_SIZE,
            LAT_MIN + (RASTER_H - 10) * PIXEL_SIZE + PIXEL_SIZE,
        )
        gpd.GeoDataFrame(geometry=[box], crs="EPSG:4326").to_file(
            geojson_src, driver="GeoJSON"
        )
        unknown = tmp_path / "wrong_cols.xyz"
        shutil.copy(geojson_src, unknown)

        # The CSV parser will succeed but column check will raise ValueError
        # (GeoJSON as CSV has no row_off/col_off columns), triggering vector fallback.
        ds = MBTilesCropsGeoTifMaskDataset(
            image_mbtiles_path=img,
            mask_path=mask,
            crops_path=unknown,
            patch_size=PATCH_SIZE,
            color_map=COLOR_MAP,
        )
        assert len(ds) == 1

    def test_unknown_extension_fallback_to_vector(self, raster_pair, tmp_path):
        """Unknown extension that is valid GeoJSON falls back to vector reading."""
        import geopandas as gpd
        from shapely.geometry import box as shapely_box
        import shutil

        img, mask = raster_pair
        # Write a valid GeoJSON to an unknown extension
        geojson_src = tmp_path / "crops_src.geojson"
        box = shapely_box(
            LON_MIN + 10 * PIXEL_SIZE,
            LAT_MIN + (RASTER_H - 10) * PIXEL_SIZE - PIXEL_SIZE,
            LON_MIN + 12 * PIXEL_SIZE,
            LAT_MIN + (RASTER_H - 10) * PIXEL_SIZE + PIXEL_SIZE,
        )
        gpd.GeoDataFrame(geometry=[box], crs="EPSG:4326").to_file(
            geojson_src, driver="GeoJSON"
        )
        # Copy to unknown extension so neither tabular nor vector path is taken
        unknown = tmp_path / "crops.xyz"
        shutil.copy(geojson_src, unknown)

        ds = MBTilesCropsGeoTifMaskDataset(
            image_mbtiles_path=img,
            mask_path=mask,
            crops_path=unknown,
            patch_size=PATCH_SIZE,
            color_map=COLOR_MAP,
        )
        assert len(ds) == 1

    def test_csv_fallback_missing_columns_raises(
        self, raster_pair, csv_crops, tmp_path
    ):
        """_windows_from_csv_fallback raises ValueError when columns are absent."""
        img, mask = raster_pair
        bad = tmp_path / "wrong_cols.dat"
        pd.DataFrame({"x": [0], "y": [1]}).to_csv(bad, index=False)
        ds = MBTilesCropsGeoTifMaskDataset(
            image_mbtiles_path=img,
            mask_path=mask,
            crops_path=csv_crops,
            patch_size=PATCH_SIZE,
            color_map=COLOR_MAP,
        )
        with pytest.raises(ValueError, match="missing required columns"):
            ds._windows_from_csv_fallback(bad)

    def test_crops_layer_parameter(self, raster_pair, tmp_path):
        """crops_layer selects the correct layer from a multi-layer GPKG."""
        import geopandas as gpd
        from shapely.geometry import box as shapely_box

        img, mask = raster_pair
        box = shapely_box(
            LON_MIN + 10 * PIXEL_SIZE,
            LAT_MIN + (RASTER_H - 10) * PIXEL_SIZE - PIXEL_SIZE,
            LON_MIN + 12 * PIXEL_SIZE,
            LAT_MIN + (RASTER_H - 10) * PIXEL_SIZE + PIXEL_SIZE,
        )
        gpkg_path = tmp_path / "multi.gpkg"
        gpd.GeoDataFrame(geometry=[box], crs="EPSG:4326").to_file(
            gpkg_path, driver="GPKG", layer="crops"
        )
        ds = MBTilesCropsGeoTifMaskDataset(
            image_mbtiles_path=img,
            mask_path=mask,
            crops_path=gpkg_path,
            patch_size=PATCH_SIZE,
            color_map=COLOR_MAP,
            crops_layer="crops",
        )
        assert len(ds) == 1
