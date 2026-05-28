# -*- coding: utf-8 -*-
"""Tests for tools/mbtiles/scan_unique_colors.py and the scan-mask-colors CLI command."""

import json
import sqlite3

import numpy as np
import pytest
import rasterio
from click.testing import CliRunner
from rasterio.crs import CRS
from rasterio.transform import from_bounds
from rasterio.windows import Window

from pytorch_segmentation_models_trainer.tools.cli import cli
from pytorch_segmentation_models_trainer.tools.mbtiles.scan_unique_colors import (
    _get_mbtiles_tile_windows,
    _unique_colors_in_window,
    build_color_map,
    scan_and_report,
    scan_unique_colors,
)

# ---------------------------------------------------------------------------
# Constants and shared fixture helpers
# ---------------------------------------------------------------------------

EPSG4326 = CRS.from_epsg(4326)
RASTER_H, RASTER_W = 256, 256
LON_MIN, LAT_MIN = -54.0, -15.0
PIXEL_SIZE = 0.001
LON_MAX = LON_MIN + RASTER_W * PIXEL_SIZE
LAT_MAX = LAT_MIN + RASTER_H * PIXEL_SIZE

RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
BLACK = (0, 0, 0)


def _make_transform():
    return from_bounds(LON_MIN, LAT_MIN, LON_MAX, LAT_MAX, RASTER_W, RASTER_H)


def _write_rgb_raster(path, data_chw, crs=EPSG4326):
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=data_chw.shape[1],
        width=data_chw.shape[2],
        count=data_chw.shape[0],
        dtype=data_chw.dtype,
        crs=crs,
        transform=_make_transform(),
    ) as dst:
        dst.write(data_chw)


@pytest.fixture()
def quad_mask(tmp_path):
    """
    256×256 RGB mask with 4 solid-colour quadrants (128×128 each):
      Q0 (TL): red   (255,   0,   0)
      Q1 (TR): green (  0, 255,   0)
      Q2 (BL): blue  (  0,   0, 255)
      Q3 (BR): black (  0,   0,   0)
    """
    data = np.zeros((3, RASTER_H, RASTER_W), dtype=np.uint8)
    data[:, :128, :128] = np.array([[[255]], [[0]], [[0]]])  # Q0 red
    data[:, :128, 128:] = np.array([[[0]], [[255]], [[0]]])  # Q1 green
    data[:, 128:, :128] = np.array([[[0]], [[0]], [[255]]])  # Q2 blue
    # Q3 stays black
    path = tmp_path / "quad_mask.tif"
    _write_rgb_raster(path, data)
    return path


@pytest.fixture()
def solid_red_mask(tmp_path):
    """256×256 mask with a single colour (255, 0, 0)."""
    data = np.full((3, RASTER_H, RASTER_W), 0, dtype=np.uint8)
    data[0] = 255
    path = tmp_path / "solid_red.tif"
    _write_rgb_raster(path, data)
    return path


@pytest.fixture()
def tiny_mbtiles(tmp_path):
    """Minimal SQLite MBTiles with a 2×2 tile grid at zoom 1.

    TMS tile coordinates:
      (col=0, row=1) → top-left    → rasterio (col_off=0,   row_off=0)
      (col=1, row=1) → top-right   → rasterio (col_off=256, row_off=0)
      (col=0, row=0) → bottom-left → rasterio (col_off=0,   row_off=256)
      (col=1, row=0) → bottom-right→ rasterio (col_off=256, row_off=256)
    """
    path = tmp_path / "tiny.mbtiles"
    with sqlite3.connect(str(path)) as conn:
        conn.execute(
            "CREATE TABLE tiles "
            "(zoom_level INTEGER, tile_column INTEGER, tile_row INTEGER, tile_data BLOB)"
        )
        conn.executemany(
            "INSERT INTO tiles VALUES (?, ?, ?, ?)",
            [
                (1, 0, 0, b""),  # bottom-left (TMS)
                (1, 0, 1, b""),  # top-left    (TMS)
                (1, 1, 0, b""),  # bottom-right(TMS)
                (1, 1, 1, b""),  # top-right   (TMS)
            ],
        )
    return path


# ---------------------------------------------------------------------------
# _get_mbtiles_tile_windows (unit)
# ---------------------------------------------------------------------------


class TestGetMbtilesTileWindows:
    def test_returns_list_for_valid_mbtiles(self, tiny_mbtiles):
        result = _get_mbtiles_tile_windows(str(tiny_mbtiles))
        assert isinstance(result, list)

    def test_returns_four_windows_for_2x2_grid(self, tiny_mbtiles):
        result = _get_mbtiles_tile_windows(str(tiny_mbtiles))
        assert len(result) == 4

    def test_all_windows_are_256x256(self, tiny_mbtiles):
        result = _get_mbtiles_tile_windows(str(tiny_mbtiles))
        for win in result:
            assert win.width == 256
            assert win.height == 256

    def test_tms_y_inversion_top_tile_maps_to_row_off_zero(self, tiny_mbtiles):
        """TMS row=max_row (topmost tile) must map to rasterio row_off=0."""
        result = _get_mbtiles_tile_windows(str(tiny_mbtiles))
        top_left = Window(0, 0, 256, 256)
        assert top_left in result

    def test_tms_y_inversion_bottom_tile_maps_to_row_off_256(self, tiny_mbtiles):
        """TMS row=0 (bottom tile) must map to rasterio row_off=256."""
        result = _get_mbtiles_tile_windows(str(tiny_mbtiles))
        bottom_left = Window(0, 256, 256, 256)
        assert bottom_left in result

    def test_returns_none_for_geotiff(self, quad_mask):
        result = _get_mbtiles_tile_windows(str(quad_mask))
        assert result is None

    def test_returns_none_for_non_sqlite_binary(self, tmp_path):
        path = tmp_path / "notadb.bin"
        path.write_bytes(b"\x00\x01\x02\x03\xff\xfe\xfd\xfc")
        result = _get_mbtiles_tile_windows(str(path))
        assert result is None

    def test_returns_none_for_empty_tiles_table(self, tmp_path):
        """SQLite with tiles table but no rows → MAX(zoom_level) is NULL → None."""
        path = tmp_path / "empty.mbtiles"
        with sqlite3.connect(str(path)) as conn:
            conn.execute(
                "CREATE TABLE tiles "
                "(zoom_level INTEGER, tile_column INTEGER, tile_row INTEGER, tile_data BLOB)"
            )
        result = _get_mbtiles_tile_windows(str(path))
        assert result is None

    def test_returns_none_for_sqlite_without_tiles_table(self, tmp_path):
        """Plain SQLite with no tiles table → OperationalError → None."""
        path = tmp_path / "notiles.db"
        with sqlite3.connect(str(path)) as conn:
            conn.execute("CREATE TABLE unrelated (id INTEGER)")
        result = _get_mbtiles_tile_windows(str(path))
        assert result is None


# ---------------------------------------------------------------------------
# _unique_colors_in_window (unit)
# ---------------------------------------------------------------------------


class TestUniqueColorsInWindow:
    def test_solid_colour_returns_one_tuple(self, solid_red_mask):
        from rasterio.windows import Window

        win = Window(0, 0, RASTER_W, RASTER_H)
        result = _unique_colors_in_window(str(solid_red_mask), win, [1, 2, 3])
        assert result == {RED}

    def test_four_quadrant_mask_returns_four_tuples(self, quad_mask):
        from rasterio.windows import Window

        win = Window(0, 0, RASTER_W, RASTER_H)
        result = _unique_colors_in_window(str(quad_mask), win, [1, 2, 3])
        assert result == {RED, GREEN, BLUE, BLACK}

    def test_partial_window_single_quadrant(self, quad_mask):
        from rasterio.windows import Window

        # Top-left 128×128 → only red
        win = Window(0, 0, 128, 128)
        result = _unique_colors_in_window(str(quad_mask), win, [1, 2, 3])
        assert result == {RED}

    def test_returns_set_type(self, solid_red_mask):
        from rasterio.windows import Window

        win = Window(0, 0, 32, 32)
        result = _unique_colors_in_window(str(solid_red_mask), win, [1, 2, 3])
        assert isinstance(result, set)


# ---------------------------------------------------------------------------
# scan_unique_colors
# ---------------------------------------------------------------------------


class TestScanUniqueColors:
    def test_four_colours_found(self, quad_mask):
        result = scan_unique_colors(quad_mask, progress=False)
        assert set(result) == {RED, GREEN, BLUE, BLACK}

    def test_result_is_sorted(self, quad_mask):
        result = scan_unique_colors(quad_mask, progress=False)
        assert result == sorted(result)

    def test_single_colour(self, solid_red_mask):
        result = scan_unique_colors(solid_red_mask, progress=False)
        assert result == [RED]

    def test_workers_1_same_as_auto(self, quad_mask):
        result_serial = scan_unique_colors(quad_mask, workers=1, progress=False)
        result_auto = scan_unique_colors(quad_mask, progress=False)
        assert result_serial == result_auto

    def test_workers_4_same_as_auto(self, quad_mask):
        result_parallel = scan_unique_colors(quad_mask, workers=4, progress=False)
        result_auto = scan_unique_colors(quad_mask, progress=False)
        assert result_parallel == result_auto

    def test_small_window_size_same_result(self, quad_mask):
        result_small = scan_unique_colors(quad_mask, window_size=64, progress=False)
        result_large = scan_unique_colors(quad_mask, window_size=1024, progress=False)
        assert result_small == result_large

    def test_invalid_band_raises(self, quad_mask):
        with pytest.raises(ValueError, match="Band index"):
            scan_unique_colors(quad_mask, bands=[1, 2, 10], progress=False)

    def test_wrong_band_count_raises(self, quad_mask):
        with pytest.raises(ValueError, match="Exactly 3"):
            scan_unique_colors(quad_mask, bands=[1, 2], progress=False)

    def test_duplicate_band_allowed(self, solid_red_mask):
        # Duplicate band 1 as R=G=B
        result = scan_unique_colors(solid_red_mask, bands=[1, 1, 1], progress=False)
        # Band 1 of solid_red is all 255; packed as (255,255,255)
        assert result == [(255, 255, 255)]

    def test_progress_bar_runs_without_error(self, quad_mask):
        result = scan_unique_colors(quad_mask, progress=True)
        assert len(result) == 4


# ---------------------------------------------------------------------------
# build_color_map
# ---------------------------------------------------------------------------


class TestBuildColorMap:
    def test_black_gets_class_0(self):
        colors = [BLACK, BLUE, GREEN, RED]
        cm = build_color_map(colors)
        black_entry = next(e for e in cm if e[:3] == list(BLACK))
        assert black_entry[3] == 0

    def test_others_sequential_from_1(self):
        colors = [BLACK, BLUE, GREEN, RED]
        cm = build_color_map(colors)
        non_black = [e for e in cm if e[:3] != [0, 0, 0]]
        indices = [e[3] for e in non_black]
        assert indices == list(range(1, len(non_black) + 1))

    def test_no_black_still_sequential(self):
        colors = [BLUE, GREEN, RED]
        cm = build_color_map(colors)
        indices = [e[3] for e in cm]
        assert indices == list(range(1, len(cm) + 1))

    def test_output_length_matches_input(self):
        colors = [BLACK, BLUE, GREEN, RED]
        assert len(build_color_map(colors)) == 4

    def test_entries_are_four_elements(self):
        cm = build_color_map([RED, GREEN])
        assert all(len(e) == 4 for e in cm)


# ---------------------------------------------------------------------------
# scan_and_report
# ---------------------------------------------------------------------------


class TestScanAndReport:
    def test_report_keys(self, quad_mask):
        report = scan_and_report(quad_mask, progress=False)
        assert set(report.keys()) == {"raster", "n_unique_colors", "color_map"}

    def test_n_unique_colors(self, quad_mask):
        report = scan_and_report(quad_mask, progress=False)
        assert report["n_unique_colors"] == 4

    def test_color_map_format(self, quad_mask):
        report = scan_and_report(quad_mask, progress=False)
        for entry in report["color_map"]:
            assert len(entry) == 4
            assert all(isinstance(v, int) for v in entry)

    def test_raster_field_is_absolute(self, quad_mask):
        report = scan_and_report(quad_mask, progress=False)
        from pathlib import Path

        assert Path(report["raster"]).is_absolute()


# ---------------------------------------------------------------------------
# CLI integration: scan-mask-colors
# ---------------------------------------------------------------------------


class TestScanMaskColorsCLI:
    def test_valid_mask_outputs_json(self, quad_mask):
        runner = CliRunner()
        result = runner.invoke(
            cli, ["scan-mask-colors", str(quad_mask), "--no-progress"]
        )
        assert result.exit_code == 0, result.output
        parsed = json.loads(result.output)
        assert "color_map" in parsed
        assert parsed["n_unique_colors"] == 4

    def test_output_json_is_valid(self, quad_mask):
        runner = CliRunner()
        result = runner.invoke(
            cli, ["scan-mask-colors", str(quad_mask), "--no-progress"]
        )
        assert result.exit_code == 0
        # Must be parseable JSON
        json.loads(result.output)

    def test_custom_window_size(self, quad_mask):
        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "scan-mask-colors",
                str(quad_mask),
                "--window-size",
                "64",
                "--no-progress",
            ],
        )
        assert result.exit_code == 0
        parsed = json.loads(result.output)
        assert parsed["n_unique_colors"] == 4

    def test_custom_workers(self, quad_mask):
        runner = CliRunner()
        result = runner.invoke(
            cli,
            ["scan-mask-colors", str(quad_mask), "--workers", "2", "--no-progress"],
        )
        assert result.exit_code == 0
        parsed = json.loads(result.output)
        assert parsed["n_unique_colors"] == 4

    def test_invalid_bands_error(self, quad_mask):
        runner = CliRunner()
        result = runner.invoke(
            cli,
            ["scan-mask-colors", str(quad_mask), "--bands", "1,2,99", "--no-progress"],
        )
        assert result.exit_code != 0

    def test_output_file_written(self, quad_mask, tmp_path):
        out = tmp_path / "report.json"
        runner = CliRunner()
        result = runner.invoke(
            cli,
            ["scan-mask-colors", str(quad_mask), "--no-progress", "-o", str(out)],
        )
        assert result.exit_code == 0
        assert out.exists()
        parsed = json.loads(out.read_text())
        assert "color_map" in parsed

    def test_nonexistent_mask_error(self, tmp_path):
        runner = CliRunner()
        result = runner.invoke(
            cli,
            ["scan-mask-colors", str(tmp_path / "nonexistent.tif"), "--no-progress"],
        )
        assert result.exit_code != 0

    def test_color_map_black_is_class_0(self, quad_mask):
        runner = CliRunner()
        result = runner.invoke(
            cli, ["scan-mask-colors", str(quad_mask), "--no-progress"]
        )
        parsed = json.loads(result.output)
        black_entries = [e for e in parsed["color_map"] if e[:3] == [0, 0, 0]]
        assert len(black_entries) == 1
        assert black_entries[0][3] == 0
