# -*- coding: utf-8 -*-
"""Tests for MBTilesPolygonDataset."""

import pickle

import numpy as np
import pytest
import rasterio
import torch
from rasterio.crs import CRS
from rasterio.transform import from_bounds
from rasterio.windows import Window
from rasterio.windows import bounds as window_bounds

from pytorch_segmentation_models_trainer.dataset_loader.mbtiles_dataset import (
    MBTilesPolygonDataset,
    _apply_color_map,
    _parse_color_map,
)

# ---------------------------------------------------------------------------
# Raster geometry constants
# ---------------------------------------------------------------------------

EPSG4326 = CRS.from_epsg(4326)
RASTER_W, RASTER_H = 512, 512
LON_MIN, LAT_MIN = -54.0, -15.0
PIXEL_SIZE = 0.0001  # degrees per pixel
LON_MAX = LON_MIN + RASTER_W * PIXEL_SIZE
LAT_MAX = LAT_MIN + RASTER_H * PIXEL_SIZE

PATCH_SIZE = 256
STRIDE = 256


# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------


def _make_transform():
    return from_bounds(LON_MIN, LAT_MIN, LON_MAX, LAT_MAX, RASTER_W, RASTER_H)


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


def _window_bbox(row_off, col_off, size, transform):
    """Return (left, bottom, right, top) in raster CRS for a square window."""
    return window_bounds(Window(col_off, row_off, size, size), transform)


@pytest.fixture()
def raster_pair(tmp_path):
    """
    512×512 image + mask pair in EPSG:4326.

    Spatial layout (4 equal quadrants of 256×256):
    ┌──────────┬──────────┐
    │ Q0 (TL)  │ Q1 (TR)  │
    │ row 0-255│ row 0-255│
    │ col 0-255│col256-511│
    ├──────────┼──────────┤
    │ Q2 (BL)  │ Q3 (BR)  │
    │row256-511│row256-511│
    │ col 0-255│col256-511│
    └──────────┴──────────┘

    Mask colours:
    - Q0: red  (255, 0, 0) → class 1
    - Q1: green(0, 255, 0) → class 2
    - Q2: blue (0, 0, 255) → class 3
    - Q3: black(0, 0,   0) → class 0 (background)

    Regions polygon: covers the full raster extent + ε so all 4 quadrant
    windows are fully contained.
    """
    import geopandas as gpd
    from shapely.geometry import box as shapely_box

    transform = _make_transform()

    # --- image raster (random RGB) ---
    rng = np.random.default_rng(42)
    img_data = rng.integers(0, 255, (3, RASTER_H, RASTER_W), dtype=np.uint8)
    img_path = tmp_path / "image.tif"
    _write_raster(img_path, img_data, transform=transform)

    # --- mask raster (4 solid-color quadrants) ---
    mask_data = np.zeros((3, RASTER_H, RASTER_W), dtype=np.uint8)
    mask_data[:, :256, :256] = np.array([[[255]], [[0]], [[0]]])  # Q0 red
    mask_data[:, :256, 256:] = np.array([[[0]], [[255]], [[0]]])  # Q1 green
    mask_data[:, 256:, :256] = np.array([[[0]], [[0]], [[255]]])  # Q2 blue
    # Q3 stays black (0,0,0)
    mask_path = tmp_path / "mask.tif"
    _write_raster(mask_path, mask_data, transform=transform)

    # --- region polygon covering the full raster ---
    region = shapely_box(
        LON_MIN - 0.001, LAT_MIN - 0.001, LON_MAX + 0.001, LAT_MAX + 0.001
    )
    regions_path = tmp_path / "regions.geojson"
    gpd.GeoDataFrame(geometry=[region], crs="EPSG:4326").to_file(
        regions_path, driver="GeoJSON"
    )

    return img_path, mask_path, regions_path


@pytest.fixture()
def color_map_rgb():
    return [
        [255, 0, 0, 1],
        [0, 255, 0, 2],
        [0, 0, 255, 3],
    ]


# ---------------------------------------------------------------------------
# Color-map unit tests
# ---------------------------------------------------------------------------


class TestParseColorMap:
    def test_valid_entries(self):
        cm = _parse_color_map([[255, 0, 0, 1], [0, 255, 0, 2]])
        assert cm == {(255, 0, 0): 1, (0, 255, 0): 2}

    def test_entry_too_short_raises(self):
        with pytest.raises(ValueError, match="class_idx"):
            _parse_color_map([[255, 0, 0]])

    def test_single_entry(self):
        cm = _parse_color_map([[0, 128, 64, 5]])
        assert cm[(0, 128, 64)] == 5


class TestApplyColorMap:
    def test_solid_red_becomes_class1(self):
        arr = np.full((8, 8, 3), (255, 0, 0), dtype=np.uint8)
        result = _apply_color_map(arr, {(255, 0, 0): 1})
        assert result.shape == (8, 8)
        assert np.all(result == 1)

    def test_background_stays_zero(self):
        arr = np.zeros((8, 8, 3), dtype=np.uint8)
        result = _apply_color_map(arr, {(255, 0, 0): 1})
        assert np.all(result == 0)

    def test_rgba_ignores_alpha(self):
        arr = np.zeros((4, 4, 4), dtype=np.uint8)
        arr[:, :, :3] = [255, 0, 0]
        arr[:, :, 3] = 200
        assert np.all(_apply_color_map(arr, {(255, 0, 0): 1}) == 1)

    def test_mixed_colors(self):
        arr = np.zeros((4, 8, 3), dtype=np.uint8)
        arr[:, :4] = [255, 0, 0]
        arr[:, 4:] = [0, 255, 0]
        result = _apply_color_map(arr, {(255, 0, 0): 1, (0, 255, 0): 2})
        assert np.all(result[:, :4] == 1)
        assert np.all(result[:, 4:] == 2)


# ---------------------------------------------------------------------------
# Dataset construction
# ---------------------------------------------------------------------------


class TestDatasetConstruction:
    def test_len_all_quadrants_contained(self, raster_pair, color_map_rgb):
        img, mask, regions = raster_pair
        ds = MBTilesPolygonDataset(
            image_mbtiles_path=img,
            mask_mbtiles_path=mask,
            regions_path=regions,
            patch_size=PATCH_SIZE,
            stride=STRIDE,
            color_map=color_map_rgb,
        )
        # 4 non-overlapping 256×256 patches over a 512×512 raster
        assert len(ds) == 4

    def test_len_overlapping_stride(self, raster_pair, color_map_rgb):
        img, mask, regions = raster_pair
        ds = MBTilesPolygonDataset(
            image_mbtiles_path=img,
            mask_mbtiles_path=mask,
            regions_path=regions,
            patch_size=256,
            stride=128,
            color_map=color_map_rgb,
        )
        # stride=128 gives a 3×3 grid over a 512×512 raster
        assert len(ds) == 9

    def test_invalid_image_dtype_raises(self, raster_pair, color_map_rgb):
        img, mask, regions = raster_pair
        with pytest.raises(ValueError, match="image_dtype"):
            MBTilesPolygonDataset(
                image_mbtiles_path=img,
                mask_mbtiles_path=mask,
                regions_path=regions,
                patch_size=PATCH_SIZE,
                color_map=color_map_rgb,
                image_dtype="bad",
            )

    def test_invalid_patch_size_raises(self, raster_pair, color_map_rgb):
        img, mask, regions = raster_pair
        with pytest.raises(ValueError, match="patch_size"):
            MBTilesPolygonDataset(
                image_mbtiles_path=img,
                mask_mbtiles_path=mask,
                regions_path=regions,
                patch_size=0,
                color_map=color_map_rgb,
            )

    def test_invalid_color_map_raises(self, raster_pair):
        img, mask, regions = raster_pair
        with pytest.raises(ValueError):
            MBTilesPolygonDataset(
                image_mbtiles_path=img,
                mask_mbtiles_path=mask,
                regions_path=regions,
                patch_size=PATCH_SIZE,
                color_map=[[255, 0, 0]],  # missing class_idx
            )

    def test_invalid_cache_extension_raises(self, raster_pair, color_map_rgb):
        img, mask, regions = raster_pair
        with pytest.raises(ValueError, match="window_index_cache"):
            MBTilesPolygonDataset(
                image_mbtiles_path=img,
                mask_mbtiles_path=mask,
                regions_path=regions,
                patch_size=PATCH_SIZE,
                color_map=color_map_rgb,
                window_index_cache="/tmp/idx.txt",
            )

    def test_empty_index_when_no_windows_contained(self, tmp_path, color_map_rgb):
        """Tiny polygon outside the raster → empty dataset, no exception."""
        import geopandas as gpd
        from shapely.geometry import box as shapely_box

        transform = _make_transform()
        img_data = np.zeros((3, RASTER_H, RASTER_W), dtype=np.uint8)
        mask_data = np.zeros((3, RASTER_H, RASTER_W), dtype=np.uint8)
        img_path = tmp_path / "img.tif"
        mask_path = tmp_path / "mask.tif"
        _write_raster(img_path, img_data, transform=transform)
        _write_raster(mask_path, mask_data, transform=transform)

        # Polygon far outside the raster extent
        tiny = shapely_box(0, 0, 0.001, 0.001)
        regions_path = tmp_path / "r.geojson"
        gpd.GeoDataFrame(geometry=[tiny], crs="EPSG:4326").to_file(
            regions_path, driver="GeoJSON"
        )

        ds = MBTilesPolygonDataset(
            image_mbtiles_path=img_path,
            mask_mbtiles_path=mask_path,
            regions_path=regions_path,
            patch_size=PATCH_SIZE,
            color_map=color_map_rgb,
        )
        assert len(ds) == 0

    def test_partial_polygon_excludes_border_windows(self, tmp_path, color_map_rgb):
        """Polygon covering only Q0 → only 1 of 4 quadrant windows is kept."""
        import geopandas as gpd
        from shapely.geometry import box as shapely_box

        transform = _make_transform()
        img_data = np.zeros((3, RASTER_H, RASTER_W), dtype=np.uint8)
        mask_data = np.zeros((3, RASTER_H, RASTER_W), dtype=np.uint8)
        img_path = tmp_path / "img.tif"
        mask_path = tmp_path / "mask.tif"
        _write_raster(img_path, img_data, transform=transform)
        _write_raster(mask_path, mask_data, transform=transform)

        # Geographic bounds of Q0 (top-left 256×256 window)
        left, bottom, right, top = _window_bbox(0, 0, PATCH_SIZE, transform)
        region = shapely_box(
            left - 0.00001, bottom - 0.00001, right + 0.00001, top + 0.00001
        )
        regions_path = tmp_path / "r.geojson"
        gpd.GeoDataFrame(geometry=[region], crs="EPSG:4326").to_file(
            regions_path, driver="GeoJSON"
        )

        ds = MBTilesPolygonDataset(
            image_mbtiles_path=img_path,
            mask_mbtiles_path=mask_path,
            regions_path=regions_path,
            patch_size=PATCH_SIZE,
            color_map=color_map_rgb,
        )
        assert len(ds) == 1


# ---------------------------------------------------------------------------
# __getitem__ contract
# ---------------------------------------------------------------------------


class TestGetItem:
    def test_returns_image_and_mask_keys(self, raster_pair, color_map_rgb):
        img, mask, regions = raster_pair
        ds = MBTilesPolygonDataset(
            image_mbtiles_path=img,
            mask_mbtiles_path=mask,
            regions_path=regions,
            patch_size=PATCH_SIZE,
            color_map=color_map_rgb,
        )
        sample = ds[0]
        assert "image" in sample
        assert "mask" in sample

    def test_image_shape_chw(self, raster_pair, color_map_rgb):
        img, mask, regions = raster_pair
        ds = MBTilesPolygonDataset(
            image_mbtiles_path=img,
            mask_mbtiles_path=mask,
            regions_path=regions,
            patch_size=PATCH_SIZE,
            color_map=color_map_rgb,
        )
        assert ds[0]["image"].shape == (3, PATCH_SIZE, PATCH_SIZE)

    def test_mask_shape_hw(self, raster_pair, color_map_rgb):
        img, mask, regions = raster_pair
        ds = MBTilesPolygonDataset(
            image_mbtiles_path=img,
            mask_mbtiles_path=mask,
            regions_path=regions,
            patch_size=PATCH_SIZE,
            color_map=color_map_rgb,
        )
        assert ds[0]["mask"].shape == (PATCH_SIZE, PATCH_SIZE)

    def test_image_dtype_float32(self, raster_pair, color_map_rgb):
        img, mask, regions = raster_pair
        ds = MBTilesPolygonDataset(
            image_mbtiles_path=img,
            mask_mbtiles_path=mask,
            regions_path=regions,
            patch_size=PATCH_SIZE,
            color_map=color_map_rgb,
        )
        assert ds[0]["image"].dtype == torch.float32

    def test_mask_dtype_int64(self, raster_pair, color_map_rgb):
        img, mask, regions = raster_pair
        ds = MBTilesPolygonDataset(
            image_mbtiles_path=img,
            mask_mbtiles_path=mask,
            regions_path=regions,
            patch_size=PATCH_SIZE,
            color_map=color_map_rgb,
        )
        assert ds[0]["mask"].dtype == torch.int64

    def test_uint8_image_normalized_to_unit_range(self, raster_pair, color_map_rgb):
        img, mask, regions = raster_pair
        ds = MBTilesPolygonDataset(
            image_mbtiles_path=img,
            mask_mbtiles_path=mask,
            regions_path=regions,
            patch_size=PATCH_SIZE,
            color_map=color_map_rgb,
            image_dtype="uint8",
        )
        t = ds[0]["image"]
        assert float(t.min()) >= 0.0
        assert float(t.max()) <= 1.0

    def test_color_map_decoding_per_quadrant(self, raster_pair, color_map_rgb):
        """Each quadrant's mask matches the expected class from the color map."""
        img, mask, regions = raster_pair
        ds = MBTilesPolygonDataset(
            image_mbtiles_path=img,
            mask_mbtiles_path=mask,
            regions_path=regions,
            patch_size=PATCH_SIZE,
            stride=PATCH_SIZE,
            color_map=color_map_rgb,
            return_metadata=True,
        )
        # Build a mapping from (row_off, col_off) → expected class
        expected = {
            (0, 0): 1,  # Q0 red
            (0, 256): 2,  # Q1 green
            (256, 0): 3,  # Q2 blue
            (256, 256): 0,  # Q3 black = background
        }
        for i in range(len(ds)):
            sample = ds[i]
            meta = sample["metadata"]
            key = (meta["row_off"], meta["col_off"])
            cls = int(sample["mask"].unique().item())
            assert (
                cls == expected[key]
            ), f"window {key}: got class {cls}, expected {expected[key]}"

    def test_out_of_bounds_raises_index_error(self, raster_pair, color_map_rgb):
        img, mask, regions = raster_pair
        ds = MBTilesPolygonDataset(
            image_mbtiles_path=img,
            mask_mbtiles_path=mask,
            regions_path=regions,
            patch_size=PATCH_SIZE,
            color_map=color_map_rgb,
        )
        with pytest.raises(IndexError):
            ds[len(ds)]

    def test_selected_bands_reduces_channels(self, raster_pair, color_map_rgb):
        img, mask, regions = raster_pair
        ds = MBTilesPolygonDataset(
            image_mbtiles_path=img,
            mask_mbtiles_path=mask,
            regions_path=regions,
            patch_size=PATCH_SIZE,
            color_map=color_map_rgb,
            selected_bands=[1, 2],
        )
        assert ds[0]["image"].shape[0] == 2

    def test_return_metadata_keys(self, raster_pair, color_map_rgb):
        img, mask, regions = raster_pair
        ds = MBTilesPolygonDataset(
            image_mbtiles_path=img,
            mask_mbtiles_path=mask,
            regions_path=regions,
            patch_size=PATCH_SIZE,
            color_map=color_map_rgb,
            return_metadata=True,
        )
        meta = ds[0]["metadata"]
        assert set(meta.keys()) == {"row_off", "col_off"}

    def test_no_metadata_by_default(self, raster_pair, color_map_rgb):
        img, mask, regions = raster_pair
        ds = MBTilesPolygonDataset(
            image_mbtiles_path=img,
            mask_mbtiles_path=mask,
            regions_path=regions,
            patch_size=PATCH_SIZE,
            color_map=color_map_rgb,
        )
        assert "metadata" not in ds[0]

    def test_no_color_map_uses_first_band(self, tmp_path):
        """When color_map is None the first mask band is used directly."""
        import geopandas as gpd
        from shapely.geometry import box as shapely_box

        transform = _make_transform()
        img_data = np.zeros((3, RASTER_H, RASTER_W), dtype=np.uint8)
        # Single-channel mask with value 7 in Q0
        mask_single = np.zeros((1, RASTER_H, RASTER_W), dtype=np.uint8)
        mask_single[0, :256, :256] = 7

        img_path = tmp_path / "img.tif"
        mask_path = tmp_path / "mask.tif"
        _write_raster(img_path, img_data, transform=transform)
        _write_raster(mask_path, mask_single, transform=transform)

        left, bottom, right, top = _window_bbox(0, 0, PATCH_SIZE, transform)
        region = shapely_box(
            left - 0.00001, bottom - 0.00001, right + 0.00001, top + 0.00001
        )
        regions_path = tmp_path / "r.geojson"
        gpd.GeoDataFrame(geometry=[region], crs="EPSG:4326").to_file(
            regions_path, driver="GeoJSON"
        )

        ds = MBTilesPolygonDataset(
            image_mbtiles_path=img_path,
            mask_mbtiles_path=mask_path,
            regions_path=regions_path,
            patch_size=PATCH_SIZE,
            color_map=None,
        )
        assert len(ds) == 1
        assert int(ds[0]["mask"].unique().item()) == 7

    def test_augmentation_pipeline_applied(self, raster_pair, color_map_rgb):
        augmentation_list = [
            {"_target_": "albumentations.HorizontalFlip", "p": 1.0},
            {"_target_": "albumentations.pytorch.ToTensorV2"},
        ]
        img, mask, regions = raster_pair
        ds = MBTilesPolygonDataset(
            image_mbtiles_path=img,
            mask_mbtiles_path=mask,
            regions_path=regions,
            patch_size=PATCH_SIZE,
            color_map=color_map_rgb,
            augmentation_list=augmentation_list,
        )
        sample = ds[0]
        assert isinstance(sample["image"], torch.Tensor)
        assert isinstance(sample["mask"], torch.Tensor)
        assert sample["mask"].dtype == torch.int64


# ---------------------------------------------------------------------------
# Persistence and cache
# ---------------------------------------------------------------------------


@pytest.fixture()
def two_polygon_pair(tmp_path):
    """
    512×512 raster pair with two non-overlapping region polygons.

    Polygon A covers Q0 (top-left 256×256).
    Polygon B covers Q1 (top-right 256×256).
    Expected valid windows: (row=0,col=0) and (row=0,col=256) only.
    """
    import geopandas as gpd
    from shapely.geometry import box as shapely_box

    transform = _make_transform()
    rng = np.random.default_rng(7)
    img_data = rng.integers(0, 255, (3, RASTER_H, RASTER_W), dtype=np.uint8)
    mask_data = np.zeros((3, RASTER_H, RASTER_W), dtype=np.uint8)
    img_path = tmp_path / "image.tif"
    mask_path = tmp_path / "mask.tif"
    _write_raster(img_path, img_data, transform=transform)
    _write_raster(mask_path, mask_data, transform=transform)

    buf = 0.00001
    q0_bbox = _window_bbox(0, 0, PATCH_SIZE, transform)
    q1_bbox = _window_bbox(0, PATCH_SIZE, PATCH_SIZE, transform)
    poly_a = shapely_box(
        q0_bbox[0] - buf, q0_bbox[1] - buf, q0_bbox[2] + buf, q0_bbox[3] + buf
    )
    poly_b = shapely_box(
        q1_bbox[0] - buf, q1_bbox[1] - buf, q1_bbox[2] + buf, q1_bbox[3] + buf
    )
    regions_path = tmp_path / "regions.geojson"
    gpd.GeoDataFrame(geometry=[poly_a, poly_b], crs="EPSG:4326").to_file(
        regions_path, driver="GeoJSON"
    )
    return img_path, mask_path, regions_path


# ---------------------------------------------------------------------------
# Polygon-based index building
# ---------------------------------------------------------------------------


class TestPolygonBasedIndex:
    def test_multiple_non_overlapping_polygons(self, two_polygon_pair, color_map_rgb):
        img, mask, regions = two_polygon_pair
        ds = MBTilesPolygonDataset(
            image_mbtiles_path=img,
            mask_mbtiles_path=mask,
            regions_path=regions,
            patch_size=PATCH_SIZE,
            stride=STRIDE,
            color_map=color_map_rgb,
            return_metadata=True,
        )
        assert len(ds) == 2
        coords = {(s["metadata"]["row_off"], s["metadata"]["col_off"]) for s in ds}
        assert coords == {(0, 0), (0, PATCH_SIZE)}

    def test_overlapping_polygons_no_duplicates(self, raster_pair, color_map_rgb):
        """Two full-coverage polygons → same 4 windows, no duplicates."""
        import geopandas as gpd
        from shapely.geometry import box as shapely_box

        img, mask, _ = raster_pair
        buf = 0.001
        full = shapely_box(LON_MIN - buf, LAT_MIN - buf, LON_MAX + buf, LAT_MAX + buf)
        regions_path = img.parent / "dup_regions.geojson"
        gpd.GeoDataFrame(geometry=[full, full], crs="EPSG:4326").to_file(
            regions_path, driver="GeoJSON"
        )
        ds = MBTilesPolygonDataset(
            image_mbtiles_path=img,
            mask_mbtiles_path=mask,
            regions_path=regions_path,
            patch_size=PATCH_SIZE,
            stride=STRIDE,
            color_map=color_map_rgb,
        )
        assert len(ds) == 4

    def test_index_sorted_row_major(self, raster_pair, color_map_rgb):
        img, mask, regions = raster_pair
        ds = MBTilesPolygonDataset(
            image_mbtiles_path=img,
            mask_mbtiles_path=mask,
            regions_path=regions,
            patch_size=PATCH_SIZE,
            stride=STRIDE,
            color_map=color_map_rgb,
            return_metadata=True,
        )
        coords = [(s["metadata"]["row_off"], s["metadata"]["col_off"]) for s in ds]
        assert coords == sorted(coords)

    def test_index_build_workers_serial_same_result(self, raster_pair, color_map_rgb):
        img, mask, regions = raster_pair
        ds_default = MBTilesPolygonDataset(
            image_mbtiles_path=img,
            mask_mbtiles_path=mask,
            regions_path=regions,
            patch_size=PATCH_SIZE,
            stride=STRIDE,
            color_map=color_map_rgb,
            return_metadata=True,
        )
        ds_serial = MBTilesPolygonDataset(
            image_mbtiles_path=img,
            mask_mbtiles_path=mask,
            regions_path=regions,
            patch_size=PATCH_SIZE,
            stride=STRIDE,
            color_map=color_map_rgb,
            return_metadata=True,
            index_build_workers=1,
        )
        assert len(ds_serial) == len(ds_default)
        for i in range(len(ds_default)):
            m1 = ds_default[i]["metadata"]
            m2 = ds_serial[i]["metadata"]
            assert m1 == m2

    def test_index_build_workers_parallel_same_result(
        self, two_polygon_pair, color_map_rgb
    ):
        img, mask, regions = two_polygon_pair
        ds_serial = MBTilesPolygonDataset(
            image_mbtiles_path=img,
            mask_mbtiles_path=mask,
            regions_path=regions,
            patch_size=PATCH_SIZE,
            stride=STRIDE,
            color_map=color_map_rgb,
            return_metadata=True,
            index_build_workers=1,
        )
        ds_parallel = MBTilesPolygonDataset(
            image_mbtiles_path=img,
            mask_mbtiles_path=mask,
            regions_path=regions,
            patch_size=PATCH_SIZE,
            stride=STRIDE,
            color_map=color_map_rgb,
            return_metadata=True,
            index_build_workers=2,
        )
        assert len(ds_parallel) == len(ds_serial)
        for i in range(len(ds_serial)):
            assert ds_serial[i]["metadata"] == ds_parallel[i]["metadata"]

    def test_multipolygon_geometry_handled(self, tmp_path, color_map_rgb):
        """A MultiPolygon covering Q0 and Q2 → 2 windows."""
        import geopandas as gpd
        from shapely.geometry import MultiPolygon, box as shapely_box

        transform = _make_transform()
        img_data = np.zeros((3, RASTER_H, RASTER_W), dtype=np.uint8)
        mask_data = np.zeros((3, RASTER_H, RASTER_W), dtype=np.uint8)
        img_path = tmp_path / "img.tif"
        mask_path = tmp_path / "mask.tif"
        _write_raster(img_path, img_data, transform=transform)
        _write_raster(mask_path, mask_data, transform=transform)

        buf = 0.00001
        q0 = _window_bbox(0, 0, PATCH_SIZE, transform)
        q2 = _window_bbox(PATCH_SIZE, 0, PATCH_SIZE, transform)
        poly_q0 = shapely_box(q0[0] - buf, q0[1] - buf, q0[2] + buf, q0[3] + buf)
        poly_q2 = shapely_box(q2[0] - buf, q2[1] - buf, q2[2] + buf, q2[3] + buf)
        multi = MultiPolygon([poly_q0, poly_q2])
        regions_path = tmp_path / "r.geojson"
        gpd.GeoDataFrame(geometry=[multi], crs="EPSG:4326").to_file(
            regions_path, driver="GeoJSON"
        )

        ds = MBTilesPolygonDataset(
            image_mbtiles_path=img_path,
            mask_mbtiles_path=mask_path,
            regions_path=regions_path,
            patch_size=PATCH_SIZE,
            stride=STRIDE,
            color_map=color_map_rgb,
            return_metadata=True,
        )
        assert len(ds) == 2
        coords = {(s["metadata"]["row_off"], s["metadata"]["col_off"]) for s in ds}
        assert coords == {(0, 0), (PATCH_SIZE, 0)}


class TestPersistenceAndCache:
    def test_pickle_round_trip(self, raster_pair, color_map_rgb):
        img, mask, regions = raster_pair
        ds = MBTilesPolygonDataset(
            image_mbtiles_path=img,
            mask_mbtiles_path=mask,
            regions_path=regions,
            patch_size=PATCH_SIZE,
            color_map=color_map_rgb,
        )
        ds2 = pickle.loads(pickle.dumps(ds))
        assert len(ds2) == len(ds)
        sample = ds2[0]
        assert "image" in sample and "mask" in sample

    def test_window_index_cache_csv_written(self, raster_pair, color_map_rgb, tmp_path):
        img, mask, regions = raster_pair
        cache = tmp_path / "idx.csv"
        MBTilesPolygonDataset(
            image_mbtiles_path=img,
            mask_mbtiles_path=mask,
            regions_path=regions,
            patch_size=PATCH_SIZE,
            color_map=color_map_rgb,
            window_index_cache=cache,
        )
        assert cache.exists()

    def test_window_index_cache_reloaded(self, raster_pair, color_map_rgb, tmp_path):
        img, mask, regions = raster_pair
        cache = tmp_path / "idx.csv"
        ds1 = MBTilesPolygonDataset(
            image_mbtiles_path=img,
            mask_mbtiles_path=mask,
            regions_path=regions,
            patch_size=PATCH_SIZE,
            color_map=color_map_rgb,
            window_index_cache=cache,
        )
        ds2 = MBTilesPolygonDataset(
            image_mbtiles_path=img,
            mask_mbtiles_path=mask,
            regions_path=regions,
            patch_size=PATCH_SIZE,
            color_map=color_map_rgb,
            window_index_cache=cache,
        )
        assert len(ds1) == len(ds2)
        # Items should be identical
        s1, s2 = ds1[0], ds2[0]
        assert torch.equal(s1["mask"], s2["mask"])

    def test_window_index_cache_parquet(self, raster_pair, color_map_rgb, tmp_path):
        img, mask, regions = raster_pair
        cache = tmp_path / "idx.parquet"
        ds = MBTilesPolygonDataset(
            image_mbtiles_path=img,
            mask_mbtiles_path=mask,
            regions_path=regions,
            patch_size=PATCH_SIZE,
            color_map=color_map_rgb,
            window_index_cache=cache,
        )
        assert cache.exists()
        assert len(ds) > 0
