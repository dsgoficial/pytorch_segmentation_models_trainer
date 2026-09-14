# -*- coding: utf-8 -*-
"""Tests for the TiledImageSource dispatch path in tools.mbtiles.alignment."""

from pathlib import Path

import numpy as np
import pytest

try:
    import rasterio
    from rasterio.transform import from_origin
    from rasterio.windows import Window

    HAS_RASTERIO = True
except ImportError:
    HAS_RASTERIO = False

pytestmark = pytest.mark.skipif(not HAS_RASTERIO, reason="rasterio not installed")

from pytorch_segmentation_models_trainer.tools.mbtiles.alignment import (
    read_source_aligned_to_mask_window,
)
from pytorch_segmentation_models_trainer.tools.mbtiles.image_source import (
    DirectoryImageSource,
    TiledImageSource,
)


def _write_raster(path: Path, minx: float, maxy: float, width, height, fill):
    path.parent.mkdir(parents=True, exist_ok=True)
    profile = {
        "driver": "GTiff",
        "height": height,
        "width": width,
        "count": 1,
        "dtype": "uint8",
        "crs": "EPSG:3857",
        "transform": from_origin(minx, maxy, 1, 1),
    }
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(np.full((height, width), fill, dtype=np.uint8), 1)


def _write_mask(path: Path, width=16, height=8):
    path.parent.mkdir(parents=True, exist_ok=True)
    profile = {
        "driver": "GTiff",
        "height": height,
        "width": width,
        "count": 1,
        "dtype": "uint8",
        "crs": "EPSG:3857",
        "transform": from_origin(0, 8, 1, 1),
    }
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(np.zeros((height, width), dtype=np.uint8), 1)


class TestReadTiledSourceAligned:
    def test_composites_two_adjacent_tiles(self, tmp_path):
        # Two 8x8 tiles side by side: left=[0,8), right=[8,16), covering the mask's
        # full 16x8 extent between them. Tiles live in their own subdir, separate
        # from the mask, so the mask itself is never a candidate.
        tiles_dir = tmp_path / "tiles"
        _write_raster(tiles_dir / "left.tif", 0, 8, 8, 8, fill=10)
        _write_raster(tiles_dir / "right.tif", 8, 8, 8, 8, fill=20)
        mask_path = tmp_path / "mask.tif"
        _write_mask(mask_path, width=16, height=8)

        source = TiledImageSource(DirectoryImageSource(directory=str(tiles_dir)))

        with rasterio.open(mask_path) as mask_src:
            window = Window(0, 0, 16, 8)
            data = read_source_aligned_to_mask_window(
                source_path=source,
                mask_src=mask_src,
                window=window,
                selected_bands=[1],
                image_dtype="uint8",
                image_resampling="nearest",
            )

        assert data.shape == (1, 8, 16)
        assert np.all(data[0, :, :8] == 10)
        assert np.all(data[0, :, 8:] == 20)

    def test_no_overlapping_candidates_returns_zeros(self, tmp_path):
        tiles_dir = tmp_path / "tiles"
        _write_raster(tiles_dir / "far_away.tif", 10_000, 10_008, 8, 8, fill=42)
        mask_path = tmp_path / "mask.tif"
        _write_mask(mask_path, width=16, height=8)

        source = TiledImageSource(DirectoryImageSource(directory=str(tiles_dir)))

        with rasterio.open(mask_path) as mask_src:
            window = Window(0, 0, 16, 8)
            data = read_source_aligned_to_mask_window(
                source_path=source,
                mask_src=mask_src,
                window=window,
                selected_bands=[1],
                image_dtype="uint8",
                image_resampling="nearest",
            )

        assert data.shape == (1, 8, 16)
        assert np.all(data == 0)

    def test_native_dtype_preserved(self, tmp_path):
        tiles_dir = tmp_path / "tiles"
        _write_raster(tiles_dir / "a.tif", 0, 8, 16, 8, fill=5)
        mask_path = tmp_path / "mask.tif"
        _write_mask(mask_path, width=16, height=8)

        source = TiledImageSource(DirectoryImageSource(directory=str(tiles_dir)))

        with rasterio.open(mask_path) as mask_src:
            window = Window(0, 0, 16, 8)
            data = read_source_aligned_to_mask_window(
                source_path=source,
                mask_src=mask_src,
                window=window,
                selected_bands=[1],
                image_dtype="native",
                image_resampling="nearest",
            )
        assert data.dtype == np.uint8
