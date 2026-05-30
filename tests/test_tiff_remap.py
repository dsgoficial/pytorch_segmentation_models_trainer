"""Tests for tools/raster/tiff_remap.py."""

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


from pytorch_segmentation_models_trainer.tools.raster.tiff_remap import (
    remap_raster,
    remap_raster_folder,
)


def test_remap_raster_basic(tmp_path: Path) -> None:
    """Pixel values should be correctly remapped."""
    data = np.array([[1, 2], [3, 4]], dtype=np.uint8)
    src = tmp_path / "src.tif"
    dst = tmp_path / "dst.tif"
    make_tiff(src, data)

    remap_raster(src, dst, pixel_mapping={1: 10, 2: 20, 3: 30})

    with rasterio.open(dst) as f:
        result = f.read(1)

    assert result[0, 0] == 10
    assert result[0, 1] == 20
    assert result[1, 0] == 30
    assert result[1, 1] == 4  # unmapped stays unchanged


def test_remap_raster_output_created(tmp_path: Path) -> None:
    """Output file should exist after the call."""
    data = np.zeros((5, 5), dtype=np.uint8)
    src = tmp_path / "src.tif"
    dst = tmp_path / "subdir" / "dst.tif"
    make_tiff(src, data)

    remap_raster(src, dst, pixel_mapping={})

    assert dst.exists()


def test_remap_raster_returns_success_true(tmp_path: Path) -> None:
    """Success flag should be True for a valid file."""
    data = np.ones((4, 4), dtype=np.uint8)
    src = tmp_path / "src.tif"
    dst = tmp_path / "dst.tif"
    make_tiff(src, data)

    out, success, err = remap_raster(src, dst, pixel_mapping={1: 5})

    assert success is True
    assert err is None
    assert out == dst


def test_remap_raster_nonexistent_returns_error(tmp_path: Path) -> None:
    """A non-existent input should return success=False and an error message."""
    src = tmp_path / "nonexistent.tif"
    dst = tmp_path / "dst.tif"

    out, success, err = remap_raster(src, dst, pixel_mapping={})

    assert success is False
    assert err is not None
    assert isinstance(err, str)


def test_remap_raster_folder_basic(tmp_path: Path) -> None:
    """All .tif files in the folder should be remapped."""
    in_dir = tmp_path / "input"
    out_dir = tmp_path / "output"
    in_dir.mkdir()

    for i in range(3):
        data = np.full((4, 4), fill_value=1, dtype=np.uint8)
        make_tiff(in_dir / f"mask_{i}.tif", data)

    n_success, n_errors = remap_raster_folder(
        in_dir, out_dir, pixel_mapping={1: 7}, progress=False
    )

    assert n_success == 3
    assert n_errors == 0

    for i in range(3):
        out_file = out_dir / f"mask_{i}.tif"
        assert out_file.exists()
        with rasterio.open(out_file) as f:
            assert f.read(1).max() == 7


def test_remap_raster_folder_preserves_structure(tmp_path: Path) -> None:
    """Subdirectory structure should be mirrored in output."""
    in_dir = tmp_path / "input"
    out_dir = tmp_path / "output"
    sub = in_dir / "sub"
    sub.mkdir(parents=True)

    data = np.ones((4, 4), dtype=np.uint8)
    make_tiff(in_dir / "top.tif", data)
    make_tiff(sub / "nested.tif", data)

    remap_raster_folder(in_dir, out_dir, pixel_mapping={1: 2}, progress=False)

    assert (out_dir / "top.tif").exists()
    assert (out_dir / "sub" / "nested.tif").exists()


def test_remap_raster_folder_empty_dir(tmp_path: Path) -> None:
    """Empty directory should return (0, 0)."""
    in_dir = tmp_path / "empty"
    in_dir.mkdir()
    out_dir = tmp_path / "output"

    n_success, n_errors = remap_raster_folder(
        in_dir, out_dir, pixel_mapping={1: 2}, progress=False
    )

    assert n_success == 0
    assert n_errors == 0


def test_remap_raster_folder_returns_counts(tmp_path: Path) -> None:
    """Return values should correctly count successes and errors."""
    in_dir = tmp_path / "input"
    out_dir = tmp_path / "output"
    in_dir.mkdir()

    # 2 valid files
    for i in range(2):
        data = np.ones((4, 4), dtype=np.uint8)
        make_tiff(in_dir / f"ok_{i}.tif", data)

    n_success, n_errors = remap_raster_folder(
        in_dir, out_dir, pixel_mapping={1: 9}, progress=False
    )

    assert n_success == 2
    assert n_errors == 0


def test_remap_raster_folder_with_progress_bar(tmp_path: Path) -> None:
    """remap_raster_folder with progress=True should wrap the iterator in tqdm."""
    in_dir = tmp_path / "input"
    out_dir = tmp_path / "output"
    in_dir.mkdir()
    make_tiff(in_dir / "f.tif", np.ones((4, 4), dtype=np.uint8))

    n_success, n_errors = remap_raster_folder(
        in_dir, out_dir, pixel_mapping={1: 2}, progress=True
    )

    assert n_success == 1
    assert n_errors == 0


def test_remap_raster_folder_error_increments_counter(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A failure from remap_raster should be counted in n_errors."""
    import pytorch_segmentation_models_trainer.tools.raster.tiff_remap as tiff_remap

    in_dir = tmp_path / "input"
    out_dir = tmp_path / "output"
    in_dir.mkdir()
    make_tiff(in_dir / "f.tif", np.ones((4, 4), dtype=np.uint8))

    monkeypatch.setattr(
        tiff_remap, "remap_raster", lambda *a, **kw: (a[1], False, "forced error")
    )

    n_success, n_errors = tiff_remap.remap_raster_folder(
        in_dir, out_dir, pixel_mapping={1: 2}, progress=False
    )

    assert n_errors == 1
    assert n_success == 0
