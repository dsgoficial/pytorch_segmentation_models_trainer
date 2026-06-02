# flake8: noqa
"""Tests for the MBTiles multiclass mask builder tool."""

from __future__ import annotations

import sqlite3
import sys
from concurrent.futures import ThreadPoolExecutor as RealThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
import pytest
from click.testing import CliRunner
from shapely.affinity import translate
from shapely.geometry import Polygon, box

from pytorch_segmentation_models_trainer.tools.cli import cli
import pytorch_segmentation_models_trainer.tools.mbtiles.multiclass_mask_builder as multiclass_mask_builder
from pytorch_segmentation_models_trainer.tools.mbtiles.multiclass_mask_builder import (
    build_mbtiles_multiclass_masks,
)


def _write_minimal_mbtiles(path: Path) -> None:
    """Create a small valid MBTiles file with a world grid at zoom 1."""
    if path.exists():
        path.unlink()
    with sqlite3.connect(path) as conn:
        conn.execute("CREATE TABLE metadata (name text, value text)")
        conn.execute(
            "CREATE TABLE tiles "
            "(zoom_level INTEGER, tile_column INTEGER, tile_row INTEGER, tile_data BLOB)"
        )
        conn.executemany(
            "INSERT INTO metadata VALUES (?,?)",
            [
                ("name", "test"),
                ("format", "png"),
                ("bounds", "-180,-85.0511287798066,180,85.0511287798066"),
                ("minzoom", "1"),
                ("maxzoom", "1"),
            ],
        )
        conn.executemany(
            "INSERT INTO tiles VALUES (?,?,?,?)",
            [
                (1, 0, 0, b""),
                (1, 1, 0, b""),
                (1, 0, 1, b""),
                (1, 1, 1, b""),
            ],
        )


def _build_fixture_data(tmp_path: Path):
    """Create a reference MBTiles, frame GeoJSON, and multiclass GPKG."""
    reference_path = tmp_path / "tiles.mbtiles"
    _write_minimal_mbtiles(reference_path)

    with rasterio.open(reference_path) as src:
        left, bottom, right, top = src.bounds
        midx = (left + right) / 2.0
        midy = (bottom + top) / 2.0

    frames = gpd.GeoDataFrame(
        {
            "rect_id": [10, 11],
            "coverage_pct": [100.0, 100.0],
            "tile_count": [1, 1],
            "geometry": [box(left, bottom, midx, top), box(midx, bottom, right, top)],
        },
        crs="EPSG:3857",
    )
    frames_path = tmp_path / "locais_mascaras.geojson"
    frames.to_file(frames_path, driver="GeoJSON")

    vectors = gpd.GeoDataFrame(
        {
            "tipo": [1, 2, 3],
            "class_name": ["builtup_area", "bareland", "grassland"],
            "geometry": [
                box(
                    left + (midx - left) * 0.10,
                    bottom + (top - bottom) * 0.10,
                    midx,
                    top,
                ),
                box(midx, bottom + (top - bottom) * 0.25, right, top),
                box(
                    left + (right - left) * 0.25,
                    bottom + (top - bottom) * 0.30,
                    right,
                    top,
                ),
            ],
        },
        crs="EPSG:3857",
    )
    vector_path = tmp_path / "dsg_masks.gpkg"
    vectors.to_file(vector_path, driver="GPKG", layer="dsg_masks")

    return reference_path, frames_path, vector_path


def test_build_mbtiles_multiclass_masks_writes_masks_and_dataset_csv(tmp_path):
    """The builder should write one uint8 mask per frame plus a dataset CSV."""
    reference_path, frames_path, vector_path = _build_fixture_data(tmp_path)
    output_dir = tmp_path / "output"

    df = build_mbtiles_multiclass_masks(
        reference_mbtiles_path=reference_path,
        frames_path=frames_path,
        vector_path=vector_path,
        output_dir=output_dir,
        frame_layer="locais_mascaras",
        vector_layer="dsg_masks",
        frame_id_attribute="rect_id",
        class_attribute="tipo",
        background_value=255,
    )

    assert len(df) == 2
    assert (output_dir / "dataset.csv").exists()
    assert set(df["frame_id"]) == {"10", "11"}
    assert set(df["background_value"]) == {255}

    for row in df.itertuples(index=False):
        with rasterio.open(row.mask_path) as src:
            mask = src.read(1)
            assert mask.dtype == np.uint8
            assert src.crs.to_string() == "EPSG:3857"
            assert src.nodata == 255
            assert mask.shape == (512, 256)
            assert 255 in np.unique(mask)

    with rasterio.open(df.iloc[0]["mask_path"]) as left_src:
        left_mask = left_src.read(1)
    with rasterio.open(df.iloc[1]["mask_path"]) as right_src:
        right_mask = right_src.read(1)
    assert 1 in np.unique(left_mask)
    assert 2 in np.unique(right_mask)


def test_mbtiles_multiclass_mask_helpers_handle_stems_and_empty_geometry():
    """Helper utilities should sanitize stems and ignore unusable geometries."""
    assert (
        multiclass_mask_builder._sanitize_stem("  abc/def  ", "fallback") == "abc_def"
    )
    assert multiclass_mask_builder._sanitize_stem(None, "fallback") == "fallback"

    frame = pd.Series({"rect_id": "  A-1 / 2  "})
    assert multiclass_mask_builder._frame_identifier(frame, "rect_id", 7) == "A-1___2"

    empty_geom = Polygon()
    assert multiclass_mask_builder._ensure_valid_geometry(None) is None
    assert multiclass_mask_builder._ensure_valid_geometry(empty_geom) is None


def test_write_mask_creates_uint8_geotiff(tmp_path):
    """The low-level writer should persist a uint8 single-band GeoTIFF."""
    mask = np.array([[1, 2], [3, 4]], dtype=np.uint8)
    output_path = tmp_path / "nested" / "mask.tif"

    multiclass_mask_builder._write_mask(
        output_path=output_path,
        mask=mask,
        crs="EPSG:3857",
        transform=rasterio.transform.from_origin(0, 2, 1, 1),
    )

    with rasterio.open(output_path) as src:
        assert src.count == 1
        assert src.dtypes[0] == "uint8"
        assert src.crs.to_string() == "EPSG:3857"
        assert src.nodata == 255
        assert src.read(1).dtype == np.uint8
        np.testing.assert_array_equal(src.read(1), mask)


def test_build_mbtiles_multiclass_masks_rejects_invalid_arguments(tmp_path):
    """Invalid background values and worker counts should fail fast."""
    reference_path, frames_path, vector_path = _build_fixture_data(tmp_path)

    with pytest.raises(ValueError, match="uint8 range"):
        build_mbtiles_multiclass_masks(
            reference_mbtiles_path=reference_path,
            frames_path=frames_path,
            vector_path=vector_path,
            output_dir=tmp_path / "invalid_bg",
            background_value=256,
        )

    with pytest.raises(ValueError, match="at least 1"):
        build_mbtiles_multiclass_masks(
            reference_mbtiles_path=reference_path,
            frames_path=frames_path,
            vector_path=vector_path,
            output_dir=tmp_path / "invalid_workers",
            n_workers=0,
        )


def test_build_mbtiles_multiclass_masks_cli_converts_paths_and_passes_yaml(tmp_path):
    """The CLI should convert YAML paths to Path objects before dispatching."""
    yaml_path = tmp_path / "config.yaml"
    yaml_path.write_text(
        "\n".join(
            [
                "reference_mbtiles_path: /tmp/ref.mbtiles",
                "frames_path: /tmp/frames.geojson",
                "vector_path: /tmp/vector.gpkg",
                f"output_dir: {tmp_path / 'out'}",
                "background_value: 17",
            ]
        ),
        encoding="utf-8",
    )

    captured = {}

    def _fake_builder(**kwargs):
        captured.update(kwargs)
        return pd.DataFrame([{"frame_id": "x"}])

    runner = CliRunner()
    with patch(
        "pytorch_segmentation_models_trainer.tools.mbtiles.multiclass_mask_builder.build_mbtiles_multiclass_masks",
        side_effect=_fake_builder,
    ):
        result = runner.invoke(cli, ["build-mbtiles-multiclass-masks", str(yaml_path)])

    assert result.exit_code == 0, result.output
    assert isinstance(captured["reference_mbtiles_path"], Path)
    assert isinstance(captured["frames_path"], Path)
    assert isinstance(captured["vector_path"], Path)
    assert isinstance(captured["output_dir"], Path)
    assert captured["background_value"] == 17


def test_build_mbtiles_multiclass_masks_supports_empty_frame_regions(tmp_path):
    """Frames without intersecting polygons should remain background-only."""
    reference_path, frames_path, vector_path = _build_fixture_data(tmp_path)
    output_dir = tmp_path / "output_empty"

    gdf = gpd.read_file(vector_path, layer="dsg_masks")
    gdf = gdf.iloc[[0]].copy()
    gdf["geometry"] = gdf.geometry.apply(lambda geom: translate(geom, xoff=1e9))
    gdf.to_file(vector_path, driver="GPKG", layer="dsg_masks")

    df = build_mbtiles_multiclass_masks(
        reference_mbtiles_path=reference_path,
        frames_path=frames_path,
        vector_path=vector_path,
        output_dir=output_dir,
        frame_layer="locais_mascaras",
        vector_layer="dsg_masks",
    )

    assert len(df) == 2
    for row in df.itertuples(index=False):
        with rasterio.open(row.mask_path) as src:
            mask = src.read(1)
        assert np.all(mask == 255)


def test_build_mbtiles_multiclass_masks_repairs_invalid_vector_geometries(tmp_path):
    """Invalid polygons should be repaired instead of crashing the builder."""
    reference_path, frames_path, vector_path = _build_fixture_data(tmp_path)
    output_dir = tmp_path / "output_invalid"

    gdf = gpd.read_file(vector_path, layer="dsg_masks")
    gdf = gdf.iloc[[0]].copy()
    minx, miny, maxx, maxy = gdf.geometry.iloc[0].bounds
    gdf.loc[:, "geometry"] = [
        Polygon(
            [
                (minx, miny),
                (maxx, maxy),
                (minx, maxy),
                (maxx, miny),
                (minx, miny),
            ]
        )
    ]
    assert not gdf.geometry.iloc[0].is_valid
    gdf.to_file(vector_path, driver="GPKG", layer="dsg_masks")

    df = build_mbtiles_multiclass_masks(
        reference_mbtiles_path=reference_path,
        frames_path=frames_path,
        vector_path=vector_path,
        output_dir=output_dir,
        frame_layer="locais_mascaras",
        vector_layer="dsg_masks",
    )

    assert len(df) == 2
    with rasterio.open(df.iloc[0]["mask_path"]) as src:
        mask = src.read(1)
    assert 1 in np.unique(mask)


def test_build_mbtiles_multiclass_masks_cli(tmp_path):
    """The CLI should load a YAML config and run the builder."""
    reference_path, frames_path, vector_path = _build_fixture_data(tmp_path)
    output_dir = tmp_path / "cli_output"
    yaml_path = tmp_path / "config.yaml"
    yaml_path.write_text(
        "\n".join(
            [
                f"reference_mbtiles_path: {reference_path}",
                f"frames_path: {frames_path}",
                f"vector_path: {vector_path}",
                f"output_dir: {output_dir}",
                "frame_layer: locais_mascaras",
                "vector_layer: dsg_masks",
                "frame_id_attribute: rect_id",
                "class_attribute: tipo",
                "background_value: 255",
            ]
        ),
        encoding="utf-8",
    )

    runner = CliRunner()
    result = runner.invoke(cli, ["build-mbtiles-multiclass-masks", str(yaml_path)])

    assert result.exit_code == 0, result.output
    assert (output_dir / "dataset.csv").exists()
    manifest = pd.read_csv(output_dir / "dataset.csv")
    assert len(manifest) == 2


def test_build_mbtiles_multiclass_masks_uses_workers_and_progress(tmp_path):
    """The builder should accept worker count and emit tqdm progress."""
    reference_path, frames_path, vector_path = _build_fixture_data(tmp_path)
    output_dir = tmp_path / "parallel_output"
    tqdm_calls = []

    class _FakeTqdm:
        def __init__(self, *args, **kwargs):
            tqdm_calls.append(kwargs)
            self.total = kwargs.get("total")
            self.count = 0

        def update(self, n=1):
            self.count += n

        def close(self):
            return None

    fake_tqdm_module = SimpleNamespace(tqdm=_FakeTqdm)
    submitted_workers = []

    class _RecordingExecutor:
        def __init__(self, max_workers=None):
            submitted_workers.append(max_workers)
            self._executor = RealThreadPoolExecutor(max_workers=max_workers)

        def __enter__(self):
            self._executor.__enter__()
            return self

        def __exit__(self, exc_type, exc, tb):
            return self._executor.__exit__(exc_type, exc, tb)

        def submit(self, *args, **kwargs):
            return self._executor.submit(*args, **kwargs)

    with patch.object(
        multiclass_mask_builder, "ThreadPoolExecutor", _RecordingExecutor
    ):
        with patch.dict(sys.modules, {"tqdm": fake_tqdm_module}):
            df = build_mbtiles_multiclass_masks(
                reference_mbtiles_path=reference_path,
                frames_path=frames_path,
                vector_path=vector_path,
                output_dir=output_dir,
                frame_layer="locais_mascaras",
                vector_layer="dsg_masks",
                n_workers=2,
                progress=True,
            )

    assert len(df) == 2
    assert submitted_workers == [2]
    assert tqdm_calls and tqdm_calls[0]["desc"] == "Building masks"
    assert tqdm_calls[0]["total"] == 2


def test_iter_frame_jobs_skips_invalid_and_empty_frames(tmp_path):
    """The job generator should yield only valid frame/vector intersections."""
    reference_path, frames_path, vector_path = _build_fixture_data(tmp_path)
    frames_gdf = gpd.read_file(frames_path, layer="locais_mascaras")
    vector_gdf = gpd.read_file(vector_path, layer="dsg_masks")

    frames_gdf.loc[len(frames_gdf)] = {
        "rect_id": 99,
        "coverage_pct": 0.0,
        "tile_count": 0,
        "geometry": None,
    }
    frames_gdf.loc[len(frames_gdf)] = {
        "rect_id": 100,
        "coverage_pct": 0.0,
        "tile_count": 0,
        "geometry": box(1e9, 1e9, 1e9 + 1, 1e9 + 1),
    }

    with rasterio.open(reference_path) as src:
        reference_bounds = box(*src.bounds)
        reference_transform = src.transform
        reference_width = src.width
        reference_height = src.height

    jobs = list(
        multiclass_mask_builder._iter_frame_jobs(
            frames_gdf=frames_gdf,
            vector_gdf=vector_gdf,
            reference_bounds=reference_bounds,
            reference_transform=reference_transform,
            reference_width=reference_width,
            reference_height=reference_height,
            frame_id_attribute="rect_id",
            class_attribute="tipo",
        )
    )

    assert [job[1] for job in jobs] == ["10", "11"]
    assert all(isinstance(job[3], list) for job in jobs)
    assert any(job[3] for job in jobs)


def test_query_vector_candidates_uses_spatial_index(tmp_path, monkeypatch):
    """The candidate lookup should consult the spatial index first."""
    reference_path, frames_path, vector_path = _build_fixture_data(tmp_path)
    vector_gdf = gpd.read_file(vector_path, layer="dsg_masks")

    calls = []

    class _FakeSIndex:
        def query(self, bounds, predicate=None):
            calls.append((bounds, predicate))
            return np.array([0, 2], dtype=int)

    monkeypatch.setattr(
        type(vector_gdf), "sindex", property(lambda self: _FakeSIndex())
    )
    clipped_frame = box(0, 0, 1, 1)

    subset = multiclass_mask_builder._query_vector_candidates(vector_gdf, clipped_frame)

    assert calls and calls[0][1] == "intersects"
    assert len(subset) == 2
    assert subset.index.tolist() == [0, 2]


def test_build_mbtiles_multiclass_masks_bounds_pending_futures(tmp_path):
    """The builder should keep only a bounded number of futures in flight."""
    reference_path, frames_path, vector_path = _build_fixture_data(tmp_path)
    output_dir = tmp_path / "bounded_output"

    frames_gdf = gpd.read_file(frames_path, layer="locais_mascaras")
    extra = frames_gdf.iloc[[0]].copy()
    extra.loc[:, "rect_id"] = np.array([12], dtype=np.int32)
    extra.loc[:, "geometry"] = [translate(extra.geometry.iloc[0], xoff=0.0)]
    frames_gdf = pd.concat([frames_gdf, extra], ignore_index=True)
    frames_gdf.to_file(frames_path, driver="GeoJSON")

    submitted = []
    pending_sizes = []

    class _FakeFuture:
        def __init__(self, result):
            self._result = result

        def result(self):
            return self._result

    class _FakeExecutor:
        def __init__(self, max_workers=None):
            self.max_workers = max_workers

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def submit(self, *args, **kwargs):
            submitted.append(kwargs["frame_id"])
            return _FakeFuture(
                {
                    "frame_id": kwargs["frame_id"],
                    "mask_path": str(
                        output_dir / "masks" / f'{kwargs["frame_id"]}.tif'
                    ),
                    "width": 1,
                    "height": 1,
                    "crs": "EPSG:3857",
                    "transform": "transform",
                    "background_value": 255,
                    "reference_mbtiles_path": "",
                    "frames_path": "",
                    "vector_path": "",
                    "_index": kwargs["clipped_frame"].bounds[0],
                }
            )

    def _fake_as_completed(futures):
        pending_sizes.append(len(futures))
        return iter(list(futures)[:1])

    with patch.object(multiclass_mask_builder, "ThreadPoolExecutor", _FakeExecutor):
        with patch.object(multiclass_mask_builder, "as_completed", _fake_as_completed):
            with patch.object(
                multiclass_mask_builder, "_build_single_frame_mask"
            ) as mocked:
                mocked.side_effect = AssertionError("worker should not be called")
                df = build_mbtiles_multiclass_masks(
                    reference_mbtiles_path=reference_path,
                    frames_path=frames_path,
                    vector_path=vector_path,
                    output_dir=output_dir,
                    frame_layer="locais_mascaras",
                    vector_layer="dsg_masks",
                    n_workers=1,
                    progress=False,
                )

    assert len(df) == 3
    assert submitted == ["10", "11", "12"]
    assert pending_sizes[0] == 2


def test_sanitize_stem_empty_string_returns_fallback():
    """_sanitize_stem should fall back to the default when the result is empty."""
    assert multiclass_mask_builder._sanitize_stem("   ", "fb") == "fb"
    assert multiclass_mask_builder._sanitize_stem("", "fb") == "fb"


def test_frame_identifier_missing_attribute_returns_index_default():
    """_frame_identifier should use the default when the attribute is absent or NaN."""
    frame_no_attr = pd.Series({"other_col": 99})
    assert (
        multiclass_mask_builder._frame_identifier(frame_no_attr, "rect_id", 3)
        == "frame_00003"
    )

    import math

    frame_nan = pd.Series({"rect_id": float("nan")})
    assert (
        multiclass_mask_builder._frame_identifier(frame_nan, "rect_id", 5)
        == "frame_00005"
    )


def test_ensure_valid_geometry_returns_none_for_degenerate_repair(monkeypatch):
    """_ensure_valid_geometry should return None when make_valid gives an empty result."""
    from shapely.geometry import GeometryCollection

    monkeypatch.setattr(
        "pytorch_segmentation_models_trainer.tools.mbtiles.multiclass_mask_builder.make_valid",
        lambda g: GeometryCollection(),  # empty → .is_empty == True
    )

    class _FakeInvalid:
        is_valid = False
        is_empty = False

    result = multiclass_mask_builder._ensure_valid_geometry(_FakeInvalid())
    assert result is None


def test_query_vector_candidates_typeerror_fallback(monkeypatch):
    """When sindex.query raises TypeError, the no-predicate fallback is used."""
    gdf = gpd.GeoDataFrame(
        {"geometry": [box(0, 0, 1, 1), box(2, 2, 3, 3)], "tipo": [1, 2]},
        crs="EPSG:4326",
    )
    clipped = box(0, 0, 1, 1)

    call_log = []

    class _FakeSIndexTypeError:
        def query(self, geom, predicate=None):
            if predicate is not None:
                call_log.append("with_predicate")
                raise TypeError("old API")
            call_log.append("no_predicate")
            return np.array([0], dtype=int)

    monkeypatch.setattr(
        type(gdf), "sindex", property(lambda self: _FakeSIndexTypeError())
    )
    result = multiclass_mask_builder._query_vector_candidates(gdf, clipped)
    assert "with_predicate" in call_log
    assert "no_predicate" in call_log
    assert len(result) == 1


def test_iter_frame_jobs_skips_zero_size_window(tmp_path):
    """Frames that map to a zero-width or zero-height window should be skipped."""
    reference_path, frames_path, vector_path = _build_fixture_data(tmp_path)

    with rasterio.open(reference_path) as src:
        reference_crs = src.crs
        reference_transform = src.transform
        bounds = src.bounds
        reference_bounds = box(*bounds)
        reference_width = src.width
        reference_height = src.height

    vector_gdf = gpd.read_file(vector_path, layer="dsg_masks")
    # Frame that is within reference_bounds but has negligible height → zero window
    left, bottom, right, top = bounds
    tiny_frame = box(left, bottom, right, bottom + 1e-15)
    frames_gdf = gpd.GeoDataFrame(
        {"geometry": [tiny_frame], "rect_id": [99]}, crs=reference_crs
    )

    jobs = list(
        multiclass_mask_builder._iter_frame_jobs(
            frames_gdf=frames_gdf,
            vector_gdf=vector_gdf,
            reference_bounds=reference_bounds,
            reference_transform=reference_transform,
            reference_width=reference_width,
            reference_height=reference_height,
            frame_id_attribute="rect_id",
            class_attribute="tipo",
        )
    )
    assert jobs == []


def test_iter_frame_jobs_skips_none_and_empty_vector_geoms(tmp_path, monkeypatch):
    """None and empty geometries in the vector GDF should be skipped silently."""
    reference_path, frames_path, vector_path = _build_fixture_data(tmp_path)

    with rasterio.open(reference_path) as src:
        reference_crs = src.crs
        reference_transform = src.transform
        bounds2 = src.bounds
        reference_bounds = box(*bounds2)
        reference_width = src.width
        reference_height = src.height

    left, bottom, right, top = bounds2
    full_frame = box(left, bottom, right, top)
    frames_gdf = gpd.GeoDataFrame(
        {"geometry": [full_frame], "rect_id": [1]}, crs=reference_crs
    )

    valid_geom = box(left, bottom, right, top)
    valid_gdf = gpd.GeoDataFrame(
        {"geometry": [valid_geom], "tipo": [1]}, crs=reference_crs
    )

    class _FakeEmptyGeom:
        is_empty = True
        is_valid = True

    # Monkeypatch _query_vector_candidates to return GDF with None and empty geometries
    def _fake_candidates(vector_gdf, clipped_frame):
        from shapely.geometry import GeometryCollection

        # Return a GDF with: None geom, empty geom, and valid geom
        rows = gpd.GeoDataFrame(
            {
                "geometry": [None, GeometryCollection(), valid_geom],
                "tipo": [1, 2, 3],
            },
            crs=reference_crs,
        )
        return rows

    monkeypatch.setattr(
        multiclass_mask_builder, "_query_vector_candidates", _fake_candidates
    )
    # Also monkeypatch _ensure_valid_geometry so the last entry (valid_geom) repairs to None,
    # exercising line 187 as well.
    orig_ensure = multiclass_mask_builder._ensure_valid_geometry

    def _fake_ensure(geom):
        if geom is valid_geom:
            return None
        return orig_ensure(geom)

    monkeypatch.setattr(multiclass_mask_builder, "_ensure_valid_geometry", _fake_ensure)

    jobs = list(
        multiclass_mask_builder._iter_frame_jobs(
            frames_gdf=frames_gdf,
            vector_gdf=valid_gdf,
            reference_bounds=reference_bounds,
            reference_transform=reference_transform,
            reference_width=reference_width,
            reference_height=reference_height,
            frame_id_attribute="rect_id",
            class_attribute="tipo",
        )
    )
    # One job is yielded (the frame itself) but shapes list will be empty
    assert len(jobs) == 1
    _, _, _, shapes, _, _ = jobs[0]
    assert shapes == []


def test_build_mbtiles_raises_when_frames_crs_is_none(tmp_path, monkeypatch):
    """build_mbtiles_multiclass_masks must raise when the frames GDF has no CRS."""
    reference_path, frames_path, vector_path = _build_fixture_data(tmp_path)

    import geopandas as gpd_inner

    orig_read = gpd_inner.read_file

    def patched_read(path, **kw):
        gdf = orig_read(path, **kw)
        if "locais_mascaras" in str(kw.get("layer", "")):
            gdf = gdf.set_crs(None, allow_override=True)
        return gdf

    monkeypatch.setattr(
        "pytorch_segmentation_models_trainer.tools.mbtiles.multiclass_mask_builder.gpd.read_file",
        patched_read,
    )
    with pytest.raises(ValueError, match="frames_path must declare a CRS"):
        build_mbtiles_multiclass_masks(
            reference_mbtiles_path=reference_path,
            frames_path=frames_path,
            vector_path=vector_path,
            output_dir=tmp_path / "out",
            frame_layer="locais_mascaras",
            vector_layer="dsg_masks",
            progress=False,
        )


def test_build_mbtiles_raises_when_vector_crs_is_none(tmp_path, monkeypatch):
    """build_mbtiles_multiclass_masks must raise when the vector GDF has no CRS."""
    reference_path, frames_path, vector_path = _build_fixture_data(tmp_path)

    import geopandas as gpd_inner

    orig_read = gpd_inner.read_file

    def patched_read(path, **kw):
        gdf = orig_read(path, **kw)
        if "dsg_masks" in str(kw.get("layer", "")):
            gdf = gdf.set_crs(None, allow_override=True)
        return gdf

    monkeypatch.setattr(
        "pytorch_segmentation_models_trainer.tools.mbtiles.multiclass_mask_builder.gpd.read_file",
        patched_read,
    )
    with pytest.raises(ValueError, match="vector_path must declare a CRS"):
        build_mbtiles_multiclass_masks(
            reference_mbtiles_path=reference_path,
            frames_path=frames_path,
            vector_path=vector_path,
            output_dir=tmp_path / "out",
            frame_layer="locais_mascaras",
            vector_layer="dsg_masks",
            progress=False,
        )


def test_build_mbtiles_raises_when_reference_has_no_crs(tmp_path, monkeypatch):
    """build_mbtiles_multiclass_masks must raise when the reference has no CRS."""
    reference_path, frames_path, vector_path = _build_fixture_data(tmp_path)

    import rasterio as _rasterio
    from unittest.mock import MagicMock

    orig_open = _rasterio.open
    _open_calls = []

    _t = _rasterio.transform.from_bounds(-180, -85, 180, 85, 512, 512)

    class _NoCrsSrc:
        crs = None
        transform = _t
        bounds = _rasterio.coords.BoundingBox(-180, -85, 180, 85)
        width = 512
        height = 512

        def __enter__(self):
            return self

        def __exit__(self, *a):
            pass

    def patched_open(path, *a, **kw):
        if str(path).endswith(".mbtiles"):
            return _NoCrsSrc()
        return orig_open(path, *a, **kw)

    monkeypatch.setattr(
        "pytorch_segmentation_models_trainer.tools.mbtiles.multiclass_mask_builder.rasterio.open",
        patched_open,
    )
    with pytest.raises(ValueError, match="reference_mbtiles_path must declare a CRS"):
        build_mbtiles_multiclass_masks(
            reference_mbtiles_path=reference_path,
            frames_path=frames_path,
            vector_path=vector_path,
            output_dir=tmp_path / "out",
            frame_layer="locais_mascaras",
            vector_layer="dsg_masks",
            progress=False,
        )


def test_build_mbtiles_reprojects_when_crs_differs(tmp_path):
    """When frames or vector GDF CRS differs from reference, they should be reprojected."""
    reference_path, frames_path, vector_path = _build_fixture_data(tmp_path)
    output_dir = tmp_path / "reprojected_output"

    # Rewrite both frames and vector in EPSG:4326 (reference is EPSG:3857)
    frames_gdf = gpd.read_file(frames_path, layer="locais_mascaras")
    frames_4326 = frames_gdf.to_crs("EPSG:4326")
    frames_path_4326 = tmp_path / "frames_4326.geojson"
    frames_4326.to_file(frames_path_4326, driver="GeoJSON")

    vector_gdf = gpd.read_file(vector_path, layer="dsg_masks")
    vector_gdf_4326 = vector_gdf.to_crs("EPSG:4326")
    vector_path_4326 = tmp_path / "vector_4326.gpkg"
    vector_gdf_4326.to_file(vector_path_4326, driver="GPKG", layer="dsg_masks")

    df = build_mbtiles_multiclass_masks(
        reference_mbtiles_path=reference_path,
        frames_path=frames_path_4326,
        vector_path=vector_path_4326,
        output_dir=output_dir,
        frame_layer=None,
        vector_layer="dsg_masks",
        frame_id_attribute="rect_id",
        class_attribute="tipo",
        background_value=255,
        progress=False,
    )

    assert len(df) == 2
