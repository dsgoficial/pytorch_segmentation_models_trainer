"""Tests for tools/raster/vrt2tif.py."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_bounds


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


from pytorch_segmentation_models_trainer.tools.raster.vrt2tif import (
    convert_folder,
    convert_to_geotiff,
)


def test_convert_to_geotiff_basic(tmp_path: Path) -> None:
    """Output file should be created and dtype should be preserved."""
    data = np.array([[1, 2], [3, 4]], dtype=np.uint16)
    src = tmp_path / "src.tif"
    dst = tmp_path / "dst.tif"
    make_tiff(src, data)

    out, success, err = convert_to_geotiff(src, dst)

    assert success is True
    assert err is None
    assert dst.exists()
    with rasterio.open(dst) as f:
        assert f.read(1).dtype == np.uint16


def test_convert_to_geotiff_preserves_band_count(tmp_path: Path) -> None:
    """Number of output bands should match the source."""
    data = np.ones((3, 8, 8), dtype=np.uint8)
    src = tmp_path / "src.tif"
    dst = tmp_path / "dst.tif"
    make_tiff(src, data)

    convert_to_geotiff(src, dst)

    with rasterio.open(dst) as f:
        assert f.count == 3


def test_convert_to_geotiff_compression_set(tmp_path: Path) -> None:
    """Output profile should show GTiff driver."""
    data = np.zeros((4, 4), dtype=np.uint8)
    src = tmp_path / "src.tif"
    dst = tmp_path / "dst.tif"
    make_tiff(src, data)

    convert_to_geotiff(src, dst, compression="DEFLATE")

    with rasterio.open(dst) as f:
        assert f.driver == "GTiff"


def test_convert_to_geotiff_error_bad_input(tmp_path: Path) -> None:
    """A non-existent input path should return success=False."""
    src = tmp_path / "does_not_exist.vrt"
    dst = tmp_path / "dst.tif"

    out, success, err = convert_to_geotiff(src, dst)

    assert success is False
    assert err is not None


def test_convert_folder_basic(tmp_path: Path) -> None:
    """Files matching glob are converted to .tif."""
    in_dir = tmp_path / "input"
    out_dir = tmp_path / "output"
    in_dir.mkdir()

    for i in range(2):
        data = np.ones((4, 4), dtype=np.uint8)
        make_tiff(in_dir / f"image_{i}.tif", data)

    n_success, n_errors = convert_folder(
        in_dir, out_dir, glob_pattern="**/*.tif", progress=False
    )

    assert n_success == 2
    assert n_errors == 0


def test_convert_folder_empty_dir_returns_zeros(tmp_path: Path) -> None:
    """Empty directory should return (0, 0)."""
    in_dir = tmp_path / "empty"
    in_dir.mkdir()
    out_dir = tmp_path / "output"

    n_success, n_errors = convert_folder(
        in_dir, out_dir, glob_pattern="**/*.vrt", progress=False
    )

    assert n_success == 0
    assert n_errors == 0


def test_convert_folder_preserves_structure(tmp_path: Path) -> None:
    """Subdirectory structure should be mirrored in output."""
    in_dir = tmp_path / "input"
    sub = in_dir / "sub"
    sub.mkdir(parents=True)
    out_dir = tmp_path / "output"

    data = np.zeros((4, 4), dtype=np.uint8)
    make_tiff(in_dir / "top.tif", data)
    make_tiff(sub / "nested.tif", data)

    convert_folder(in_dir, out_dir, glob_pattern="**/*.tif", progress=False)

    assert (out_dir / "top.tif").exists()
    assert (out_dir / "sub" / "nested.tif").exists()
