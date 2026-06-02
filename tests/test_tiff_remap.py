"""Tests for tools/raster/tiff_remap.py."""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_bounds


def make_tiff(
    path: Path,
    data: np.ndarray,
    crs: str = "EPSG:4326",
    nodata=None,
) -> None:
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
    if nodata is not None:
        profile["nodata"] = nodata
    with rasterio.open(path, "w", **profile) as dst:
        if data.ndim == 2:
            dst.write(data, 1)
        else:
            dst.write(data)


def make_remap_json(
    path: Path, mapping: dict, nodata_value=255, description="test"
) -> None:
    """Helper: write a DsgTools-compatible remap JSON file."""
    payload = {
        "description": description,
        "nodata_value": nodata_value,
        "mapping": mapping,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f)


from pytorch_segmentation_models_trainer.tools.raster.tiff_remap import (
    build_vrt,
    infer_output_dtype,
    load_remap_json,
    remap_raster,
    remap_raster_folder,
    remap_raster_folder_from_json,
    remap_raster_windowed,
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


# ---------------------------------------------------------------------------
# load_remap_json
# ---------------------------------------------------------------------------


def test_load_remap_json_valid(tmp_path: Path) -> None:
    """Valid JSON returns correct (mapping, nodata_value, description) tuple."""
    json_path = tmp_path / "map.json"
    make_remap_json(
        json_path, {"3": 1, "15": 2}, nodata_value=255, description="test map"
    )

    mapping, nodata, desc = load_remap_json(json_path)

    assert mapping == {3: 1, 15: 2}
    assert nodata == 255
    assert desc == "test map"


def test_load_remap_json_default_nodata(tmp_path: Path) -> None:
    """Missing nodata_value in JSON defaults to 255."""
    json_path = tmp_path / "map.json"
    payload = {"mapping": {"1": 10}}
    with open(json_path, "w") as f:
        json.dump(payload, f)

    _, nodata, _ = load_remap_json(json_path)

    assert nodata == 255


def test_load_remap_json_missing_mapping_key(tmp_path: Path) -> None:
    """JSON without 'mapping' field raises ValueError."""
    json_path = tmp_path / "map.json"
    with open(json_path, "w") as f:
        json.dump({"nodata_value": 255}, f)

    with pytest.raises(ValueError, match="mapping"):
        load_remap_json(json_path)


def test_load_remap_json_all_invalid_values(tmp_path: Path) -> None:
    """All non-integer values raise ValueError after warnings."""
    json_path = tmp_path / "map.json"
    with open(json_path, "w") as f:
        json.dump({"mapping": {"a": "b", "x": "y"}}, f)

    with pytest.raises(ValueError, match="No valid"):
        load_remap_json(json_path)


def test_load_remap_json_partial_invalid_warns(tmp_path: Path) -> None:
    """Partially invalid entries emit a warning; valid entries are returned."""
    json_path = tmp_path / "map.json"
    with open(json_path, "w") as f:
        json.dump({"mapping": {"3": 1, "bad": "val"}}, f)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        mapping, _, _ = load_remap_json(json_path)

    assert mapping == {3: 1}
    assert any("bad" in str(w.message) for w in caught)


def test_load_remap_json_not_found(tmp_path: Path) -> None:
    """Non-existent path raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        load_remap_json(tmp_path / "missing.json")


def test_load_remap_json_bad_json(tmp_path: Path) -> None:
    """Malformed JSON raises ValueError."""
    json_path = tmp_path / "bad.json"
    json_path.write_text("{ not valid json }")

    with pytest.raises(ValueError, match="[Ii]nvalid JSON"):
        load_remap_json(json_path)


# ---------------------------------------------------------------------------
# infer_output_dtype
# ---------------------------------------------------------------------------


def test_infer_output_dtype_uint8() -> None:
    """Values in [0, 255] produce uint8 output dtype."""
    assert infer_output_dtype({3: 1, 15: 100}, 255) == np.dtype(np.uint8)


def test_infer_output_dtype_uint16() -> None:
    """Values in [0, 65535] produce uint16 output dtype."""
    assert infer_output_dtype({3: 601, 15: 60000}, 255) == np.dtype(np.uint16)


def test_infer_output_dtype_int16() -> None:
    """Negative values in int16 range produce int16 output dtype."""
    assert infer_output_dtype({1: -100, 2: 100}, -9999) == np.dtype(np.int16)


def test_infer_output_dtype_int32() -> None:
    """Values outside int16 range produce int32 output dtype."""
    assert infer_output_dtype({1: 100000, 2: -100000}, -9999) == np.dtype(np.int32)


def test_infer_output_dtype_float32() -> None:
    """Extreme values produce float32 output dtype."""
    extreme = 2**33
    assert infer_output_dtype({1: extreme}, 255) == np.dtype(np.float32)


# ---------------------------------------------------------------------------
# remap_raster_windowed
# ---------------------------------------------------------------------------


def test_remap_raster_windowed_basic(tmp_path: Path) -> None:
    """Mapped pixel values are correctly replaced in the output."""
    data = np.array([[1, 2], [3, 4]], dtype=np.uint8)
    src = tmp_path / "src.tif"
    dst = tmp_path / "dst.tif"
    make_tiff(src, data)

    remap_raster_windowed(src, dst, {1: 10, 2: 20, 3: 30, 4: 40}, nodata_value=255)

    with rasterio.open(dst) as f:
        result = f.read(1)

    assert result[0, 0] == 10
    assert result[0, 1] == 20
    assert result[1, 0] == 30
    assert result[1, 1] == 40


def test_remap_raster_windowed_unmapped_to_nodata(tmp_path: Path) -> None:
    """Pixel values not in the mapping become nodata_value (DsgTools semantics)."""
    data = np.array([[1, 2], [3, 4]], dtype=np.uint8)
    src = tmp_path / "src.tif"
    dst = tmp_path / "dst.tif"
    make_tiff(src, data)

    remap_raster_windowed(src, dst, {1: 10, 2: 20}, nodata_value=255)

    with rasterio.open(dst) as f:
        result = f.read(1)

    assert result[0, 0] == 10
    assert result[0, 1] == 20
    assert result[1, 0] == 255  # unmapped → nodata
    assert result[1, 1] == 255  # unmapped → nodata


def test_remap_raster_windowed_input_nodata_to_output_nodata(tmp_path: Path) -> None:
    """Pixels equal to src nodata become nodata_value even when src nodata matches a mapping key."""
    data = np.array([[0, 1], [2, 1]], dtype=np.uint8)
    src = tmp_path / "src.tif"
    dst = tmp_path / "dst.tif"
    make_tiff(src, data, nodata=0)

    remap_raster_windowed(src, dst, {0: 5, 1: 10, 2: 20}, nodata_value=255)

    with rasterio.open(dst) as f:
        result = f.read(1)

    assert result[0, 0] == 255  # src nodata → output nodata (overrides mapping)
    assert result[0, 1] == 10
    assert result[1, 0] == 20
    assert result[1, 1] == 10


def test_remap_raster_windowed_output_created(tmp_path: Path) -> None:
    """Output file is created at the specified path."""
    src = tmp_path / "src.tif"
    dst = tmp_path / "subdir" / "dst.tif"
    make_tiff(src, np.zeros((4, 4), dtype=np.uint8))

    remap_raster_windowed(src, dst, {0: 1}, nodata_value=255)

    assert dst.exists()


def test_remap_raster_windowed_preserves_geo(tmp_path: Path) -> None:
    """Output preserves the CRS and geotransform of the input."""
    data = np.ones((8, 8), dtype=np.uint8)
    src = tmp_path / "src.tif"
    dst = tmp_path / "dst.tif"
    make_tiff(src, data, crs="EPSG:32633")

    remap_raster_windowed(src, dst, {1: 2}, nodata_value=255)

    with rasterio.open(src) as s, rasterio.open(dst) as d:
        assert d.crs == s.crs
        assert d.transform == s.transform


def test_remap_raster_windowed_success_return(tmp_path: Path) -> None:
    """Returns (output_path, True, None) on success."""
    src = tmp_path / "src.tif"
    dst = tmp_path / "dst.tif"
    make_tiff(src, np.ones((4, 4), dtype=np.uint8))

    out, success, err = remap_raster_windowed(src, dst, {1: 2}, nodata_value=255)

    assert out == dst
    assert success is True
    assert err is None


def test_remap_raster_windowed_error_return(tmp_path: Path) -> None:
    """Returns (output_path, False, str) when the input raster is missing."""
    src = tmp_path / "missing.tif"
    dst = tmp_path / "dst.tif"

    out, success, err = remap_raster_windowed(src, dst, {1: 2}, nodata_value=255)

    assert out == dst
    assert success is False
    assert isinstance(err, str)


def test_remap_raster_windowed_dtype_inference(tmp_path: Path) -> None:
    """Output dtype matches infer_output_dtype result for the given mapping."""
    data = np.array([[1, 2], [3, 4]], dtype=np.uint8)
    src = tmp_path / "src.tif"
    dst = tmp_path / "dst.tif"
    make_tiff(src, data)

    # mapping values fit in uint8; nodata_value=255 also fits
    remap_raster_windowed(src, dst, {1: 10, 2: 20, 3: 30, 4: 40}, nodata_value=255)

    with rasterio.open(dst) as f:
        assert f.dtypes[0] == "uint8"


# ---------------------------------------------------------------------------
# remap_raster_folder_from_json
# ---------------------------------------------------------------------------


def test_remap_raster_folder_from_json_basic(tmp_path: Path) -> None:
    """All .tif files in the input directory are remapped using the JSON mapping."""
    in_dir = tmp_path / "input"
    out_dir = tmp_path / "output"
    in_dir.mkdir()
    json_path = tmp_path / "map.json"
    make_remap_json(json_path, {"1": 9}, nodata_value=255)

    for i in range(3):
        make_tiff(in_dir / f"mask_{i}.tif", np.full((4, 4), 1, dtype=np.uint8))

    n_success, n_errors = remap_raster_folder_from_json(
        in_dir, out_dir, json_path, progress=False
    )

    assert n_success == 3
    assert n_errors == 0
    for i in range(3):
        out_file = out_dir / f"mask_{i}.tif"
        assert out_file.exists()
        with rasterio.open(out_file) as f:
            assert f.read(1).max() == 9


def test_remap_raster_folder_from_json_empty(tmp_path: Path) -> None:
    """Empty directory returns (0, 0)."""
    in_dir = tmp_path / "empty"
    in_dir.mkdir()
    out_dir = tmp_path / "output"
    json_path = tmp_path / "map.json"
    make_remap_json(json_path, {"1": 2})

    n_success, n_errors = remap_raster_folder_from_json(
        in_dir, out_dir, json_path, progress=False
    )

    assert n_success == 0
    assert n_errors == 0


def test_remap_raster_folder_from_json_preserves_structure(tmp_path: Path) -> None:
    """Subdirectory structure is mirrored in the output directory."""
    in_dir = tmp_path / "input"
    out_dir = tmp_path / "output"
    sub = in_dir / "sub"
    sub.mkdir(parents=True)
    json_path = tmp_path / "map.json"
    make_remap_json(json_path, {"1": 2})

    make_tiff(in_dir / "top.tif", np.ones((4, 4), dtype=np.uint8))
    make_tiff(sub / "nested.tif", np.ones((4, 4), dtype=np.uint8))

    remap_raster_folder_from_json(in_dir, out_dir, json_path, progress=False)

    assert (out_dir / "top.tif").exists()
    assert (out_dir / "sub" / "nested.tif").exists()


def test_remap_raster_folder_from_json_with_progress(tmp_path: Path) -> None:
    """progress=True wraps the iterator in tqdm without changing the result."""
    in_dir = tmp_path / "input"
    out_dir = tmp_path / "output"
    in_dir.mkdir()
    json_path = tmp_path / "map.json"
    make_remap_json(json_path, {"1": 2})
    make_tiff(in_dir / "f.tif", np.ones((4, 4), dtype=np.uint8))

    n_success, n_errors = remap_raster_folder_from_json(
        in_dir, out_dir, json_path, progress=True
    )

    assert n_success == 1
    assert n_errors == 0


def test_remap_raster_folder_from_json_error_counting(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A failure from remap_raster_windowed is counted in n_errors."""
    import pytorch_segmentation_models_trainer.tools.raster.tiff_remap as tiff_remap

    in_dir = tmp_path / "input"
    out_dir = tmp_path / "output"
    in_dir.mkdir()
    json_path = tmp_path / "map.json"
    make_remap_json(json_path, {"1": 2})
    make_tiff(in_dir / "f.tif", np.ones((4, 4), dtype=np.uint8))

    monkeypatch.setattr(
        tiff_remap,
        "remap_raster_windowed",
        lambda *a, **kw: (a[1], False, "forced error"),
    )

    n_success, n_errors = tiff_remap.remap_raster_folder_from_json(
        in_dir, out_dir, json_path, progress=False
    )

    assert n_errors == 1
    assert n_success == 0


# ---------------------------------------------------------------------------
# build_vrt
# ---------------------------------------------------------------------------


def _make_tiff_at(path: Path, data: np.ndarray, bounds, crs: str = "EPSG:4326") -> None:
    """Helper: write a minimal GeoTIFF with explicit geographic bounds."""
    left, bottom, right, top = bounds
    h = data.shape[0]
    w = data.shape[1]
    transform = from_bounds(left, bottom, right, top, w, h)
    profile = dict(
        driver="GTiff",
        dtype=data.dtype,
        width=w,
        height=h,
        count=1,
        crs=crs,
        transform=transform,
    )
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(data, 1)


def test_build_vrt_creates_file(tmp_path: Path) -> None:
    """build_vrt writes a .vrt file to the given path."""
    r = tmp_path / "r.tif"
    make_tiff(r, np.ones((4, 4), dtype=np.uint8))
    vrt = tmp_path / "out.vrt"

    build_vrt([r], vrt, nodata_value=255)

    assert vrt.exists()


def test_build_vrt_returns_path(tmp_path: Path) -> None:
    """build_vrt returns the output_path."""
    r = tmp_path / "r.tif"
    make_tiff(r, np.ones((4, 4), dtype=np.uint8))
    vrt = tmp_path / "out.vrt"

    result = build_vrt([r], vrt, nodata_value=255)

    assert result == vrt


def test_build_vrt_readable_by_rasterio(tmp_path: Path) -> None:
    """The created VRT can be opened and read by rasterio."""
    r = tmp_path / "r.tif"
    data = np.array([[1, 2], [3, 4]], dtype=np.uint8)
    make_tiff(r, data)
    vrt = tmp_path / "out.vrt"

    build_vrt([r], vrt, nodata_value=255)

    with rasterio.open(vrt) as f:
        result = f.read(1)

    np.testing.assert_array_equal(result, data)


def test_build_vrt_nodata_value(tmp_path: Path) -> None:
    """VRT band carries the specified nodata value."""
    r = tmp_path / "r.tif"
    make_tiff(r, np.ones((4, 4), dtype=np.uint8))
    vrt = tmp_path / "out.vrt"

    build_vrt([r], vrt, nodata_value=42)

    with rasterio.open(vrt) as f:
        assert f.nodata == 42


def test_build_vrt_covers_union_extent(tmp_path: Path) -> None:
    """VRT bounds cover the union of all input rasters."""
    r1 = tmp_path / "r1.tif"
    r2 = tmp_path / "r2.tif"
    _make_tiff_at(r1, np.ones((4, 4), dtype=np.uint8), bounds=(0, 0, 1, 1))
    _make_tiff_at(r2, np.ones((4, 4), dtype=np.uint8), bounds=(1, 0, 2, 1))
    vrt = tmp_path / "out.vrt"

    build_vrt([r1, r2], vrt, nodata_value=255)

    with rasterio.open(vrt) as f:
        assert abs(f.bounds.left - 0.0) < 1e-6
        assert abs(f.bounds.right - 2.0) < 1e-6
        assert f.width == 8  # two 4-pixel rasters side by side


def test_build_vrt_empty_list_raises(tmp_path: Path) -> None:
    """Empty raster list raises ValueError."""
    with pytest.raises(ValueError, match="empty"):
        build_vrt([], tmp_path / "out.vrt", nodata_value=255)


def test_build_vrt_crs_mismatch_raises(tmp_path: Path) -> None:
    """Rasters with different CRS raise ValueError."""
    r1 = tmp_path / "r1.tif"
    r2 = tmp_path / "r2.tif"
    make_tiff(r1, np.ones((4, 4), dtype=np.uint8), crs="EPSG:4326")
    make_tiff(r2, np.ones((4, 4), dtype=np.uint8), crs="EPSG:32633")

    with pytest.raises(ValueError, match="[Cc][Rr][Ss]"):
        build_vrt([r1, r2], tmp_path / "out.vrt", nodata_value=255)


def test_build_vrt_pixel_width_mismatch_raises(tmp_path: Path) -> None:
    """Rasters with different pixel widths raise ValueError."""
    r1 = tmp_path / "r1.tif"
    r2 = tmp_path / "r2.tif"
    # same x-extent, different column count → different pixel width
    _make_tiff_at(r1, np.ones((4, 4), dtype=np.uint8), bounds=(0, 0, 1, 1))
    _make_tiff_at(r2, np.ones((4, 8), dtype=np.uint8), bounds=(0, 0, 1, 1))

    with pytest.raises(ValueError, match="[Pp]ixel"):
        build_vrt([r1, r2], tmp_path / "out.vrt", nodata_value=255)


def test_build_vrt_pixel_height_mismatch_raises(tmp_path: Path) -> None:
    """Rasters with different pixel heights raise ValueError."""
    r1 = tmp_path / "r1.tif"
    r2 = tmp_path / "r2.tif"
    # same pixel width, different pixel height: width=(1/4)=0.25 for both,
    # height=(1/4)=0.25 for r1 vs (2/8)=0.25 ... let's make them really differ:
    # r1: x-extent=1, 4 cols → pixel_w=0.25; y-extent=1, 4 rows → pixel_h=0.25
    # r2: x-extent=1, 4 cols → pixel_w=0.25; y-extent=2, 4 rows → pixel_h=0.5
    _make_tiff_at(r1, np.ones((4, 4), dtype=np.uint8), bounds=(0, 0, 1, 1))
    _make_tiff_at(r2, np.ones((4, 4), dtype=np.uint8), bounds=(0, 0, 1, 2))

    with pytest.raises(ValueError, match="[Pp]ixel"):
        build_vrt([r1, r2], tmp_path / "out.vrt", nodata_value=255)


def test_build_vrt_band_count_mismatch_raises(tmp_path: Path) -> None:
    """Rasters with different band counts raise ValueError."""
    r1 = tmp_path / "r1.tif"
    r2 = tmp_path / "r2.tif"
    make_tiff(r1, np.ones((4, 4), dtype=np.uint8))
    make_tiff(r2, np.ones((3, 4, 4), dtype=np.uint8))  # 3 bands

    with pytest.raises(ValueError, match="[Bb]and"):
        build_vrt([r1, r2], tmp_path / "out.vrt", nodata_value=255)


# ---------------------------------------------------------------------------
# remap_raster_folder_from_json + create_vrt integration
# ---------------------------------------------------------------------------


def test_remap_folder_from_json_creates_vrt(tmp_path: Path) -> None:
    """create_vrt=True builds a mosaic.vrt in the output directory."""
    in_dir = tmp_path / "input"
    out_dir = tmp_path / "output"
    in_dir.mkdir()
    json_path = tmp_path / "map.json"
    make_remap_json(json_path, {"1": 2})
    for i in range(2):
        make_tiff(in_dir / f"m{i}.tif", np.ones((4, 4), dtype=np.uint8))

    remap_raster_folder_from_json(
        in_dir, out_dir, json_path, progress=False, create_vrt=True
    )

    assert (out_dir / "mosaic.vrt").exists()


def test_remap_folder_from_json_vrt_custom_path(tmp_path: Path) -> None:
    """vrt_path overrides the default VRT location."""
    in_dir = tmp_path / "input"
    out_dir = tmp_path / "output"
    in_dir.mkdir()
    json_path = tmp_path / "map.json"
    make_remap_json(json_path, {"1": 2})
    make_tiff(in_dir / "m.tif", np.ones((4, 4), dtype=np.uint8))

    custom_vrt = tmp_path / "custom" / "result.vrt"
    remap_raster_folder_from_json(
        in_dir,
        out_dir,
        json_path,
        progress=False,
        create_vrt=True,
        vrt_path=custom_vrt,
    )

    assert custom_vrt.exists()


def test_remap_folder_from_json_no_vrt_by_default(tmp_path: Path) -> None:
    """Default behaviour (create_vrt=False) creates no VRT file."""
    in_dir = tmp_path / "input"
    out_dir = tmp_path / "output"
    in_dir.mkdir()
    json_path = tmp_path / "map.json"
    make_remap_json(json_path, {"1": 2})
    make_tiff(in_dir / "m.tif", np.ones((4, 4), dtype=np.uint8))

    remap_raster_folder_from_json(in_dir, out_dir, json_path, progress=False)

    assert not (out_dir / "mosaic.vrt").exists()
