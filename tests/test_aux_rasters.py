# -*- coding: utf-8 -*-
"""Tests for tools/aux_rasters (GEM-15 and Forest Typology per-tile caches).

Offline: synthetic UTM GeoTIFFs stand in for the public COGs; the STAC
lookup (GEM-15) and the GCS listing (ForTy) are monkeypatched.
"""

import json
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest
import rasterio
import torch
from click.testing import CliRunner
from rasterio.transform import from_origin
from rasterio.warp import transform_bounds

from pytorch_segmentation_models_trainer.tools.aux_rasters import common
from pytorch_segmentation_models_trainer.tools.aux_rasters import forest_typology as ft
from pytorch_segmentation_models_trainer.tools.aux_rasters import gem15

UTM = "EPSG:32722"
# 1.2 km x 1.2 km square around (-50.5, -25.5) in UTM 22S.
X0, Y0 = 550000.0, 7180000.0
TILE = transform_bounds(UTM, "EPSG:4326", X0 + 300, Y0 - 900, X0 + 900, Y0 - 300)


def _write(path, data, res, nodata, x0=X0, y0=Y0, south_up=False):
    count, h, w = data.shape
    transform = from_origin(x0, y0, res, res)
    if south_up:  # ForTy files are stored with a positive y pixel size
        transform = rasterio.Affine(res, 0, x0, 0, res, y0 - h * res)
        data = data[:, ::-1, :]
    with rasterio.open(
        path, "w", driver="GTiff", height=h, width=w, count=count,
        dtype=data.dtype, crs=UTM, transform=transform, nodata=nodata,
    ) as dst:
        dst.write(data)
    return str(path)


def _gem_files(tmp_path, dtm_value=None, nan_block=False):
    n = 80  # 80 x 15 m = 1.2 km
    rows = np.arange(n, dtype=np.float32)[:, None].repeat(n, 1)
    dtm = (100.0 + rows * 1.5) if dtm_value is None else np.full((n, n), dtm_value, np.float32)
    dsm = np.stack([dtm + 5, dtm + 1, dtm + 4, dtm + 9, np.full_like(dtm, 2.0)])
    if nan_block:
        dtm[:, :] = np.nan
        dsm[:] = np.nan
    return (
        _write(tmp_path / "dsm.tif", dsm.astype(np.float32), 15.0, np.nan),
        _write(tmp_path / "dtm.tif", dtm[None].astype(np.float32), 15.0, np.nan),
    )


# --------------------------------------------------------------------- common
class TestCommon:
    def test_gdal_path(self):
        assert common.gdal_path("https://x/y.tif") == "/vsicurl/https://x/y.tif"
        assert common.gdal_path("/local/y.tif") == "/local/y.tif"

    def test_read_to_tile_grid_contract(self, tmp_path):
        dsm, _ = _gem_files(tmp_path)
        out = common.read_to_tile_grid([dsm], TILE, (16, 16), np.float32, np.nan, "bilinear")
        assert out.shape == (5, 16, 16) and out.dtype == np.float32
        assert np.isfinite(out).all()

    def test_band_selection(self, tmp_path):
        dsm, _ = _gem_files(tmp_path)
        out = common.read_to_tile_grid([dsm], TILE, (8, 8), np.float32, np.nan, bands=[5])
        assert out.shape == (1, 8, 8)
        np.testing.assert_allclose(out, 2.0)

    def test_second_source_fills_only_nodata(self, tmp_path):
        a = np.full((1, 80, 80), 7, np.uint8)
        a[0, :, :40] = 255  # west half missing in the first source
        b = np.full((1, 80, 80), 9, np.uint8)
        fa = _write(tmp_path / "a.tif", a, 15.0, 255)
        fb = _write(tmp_path / "b.tif", b, 15.0, 255)
        out = common.read_to_tile_grid([fa, fb], TILE, (8, 8), np.uint8, 255)
        assert set(np.unique(out)) <= {7, 9} and (out == 7).any() and (out == 9).any()

    def test_no_source_raises(self):
        with pytest.raises(ValueError):
            common.read_to_tile_grid([], TILE, (8, 8), np.float32, np.nan)

    def test_source_without_nodata_uses_given_fill(self, tmp_path):
        data = np.full((1, 80, 80), 3, np.uint8)
        f = _write(tmp_path / "n.tif", data, 15.0, None)
        out = common.read_to_tile_grid([f], TILE, (4, 4), np.uint8, 255)
        assert (out == 3).all()

    def test_is_nodata_nan_and_value(self):
        assert common._is_nodata(np.array([np.nan, 1.0]), np.nan).tolist() == [True, False]
        assert common._is_nodata(np.array([255, 1]), 255).tolist() == [True, False]

    def test_tiles_from_manifest(self):
        df = pd.DataFrame(
            {"id": [1, 2], "bbox_west": [0, 1], "bbox_south": [0, 1],
             "bbox_east": [1, 2], "bbox_north": [1, 2]}
        )
        tiles = common.tiles_from_manifest(df)
        assert tiles == {"1": (0.0, 0.0, 1.0, 1.0), "2": (1.0, 1.0, 2.0, 2.0)}
        with pytest.raises(ValueError):
            common.tiles_from_manifest(df.drop(columns="bbox_east"))

    def test_int_id_with_float_bounds_stays_int_string(self):
        """Regression: iterrows upcast the id to 1.0 -> cache '1.0.npz'."""
        df = pd.DataFrame(
            {"id": [100], "bbox_west": [-50.5], "bbox_south": [-25.5],
             "bbox_east": [-50.4], "bbox_north": [-25.4]}
        )
        assert list(common.tiles_from_manifest(df)) == ["100"]

    def test_download_tiles_report(self, tmp_path):
        (tmp_path / "old.npz").write_bytes(b"x")

        def one(bounds, path):
            if bounds[0] == 1:
                return False
            if bounds[0] == 2:
                raise RuntimeError("boom")
            Path(path).write_bytes(b"ok")
            return True

        tiles = {"old": (0, 0, 1, 1), "a": (0, 0, 1, 1), "b": (1, 0, 2, 1), "c": (2, 0, 3, 1)}
        rep = common.download_tiles(tiles, tmp_path, one, max_workers=2)
        assert rep["skipped"] == ["old"] and rep["written"] == ["a"]
        assert rep["no_data"] == ["b"] and rep["failed"][0][0] == "c"
        rep2 = common.download_tiles({"old": (0, 0, 1, 1)}, tmp_path, one, overwrite=True)
        assert rep2["written"] == ["old"]


# --------------------------------------------------------------------- GEM-15
class TestGem15Naming:
    def test_cell_id_matches_catalog_examples(self):
        assert gem15.cell_id(-50.5, -20.5) == ("m30_m60", "wm51sm21")
        assert gem15.cell_id(-90.5, -0.5) == ("m10_m100", "wm91sm1")
        assert gem15.cell_id(5.5, 12.5) == ("p10_p0", "wp5sp12")

    def test_cells_for_bounds(self):
        assert gem15.cells_for_bounds((-50.6, -20.6, -50.4, -20.4)) == [("m30_m60", "wm51sm21")]
        assert len(gem15.cells_for_bounds((-50.1, -20.1, -49.9, -19.9))) == 4
        # ending exactly on a degree line does not touch the next cell
        assert len(gem15.cells_for_bounds((-51.0, -21.0, -50.0, -20.0))) == 1

    def test_pixel_size(self):
        dy, dx = gem15.pixel_size_m((0.0, 0.0, 0.001, 0.001), (1, 1))
        assert dy == pytest.approx(111.32, rel=1e-3) and dx == pytest.approx(111.32, rel=1e-3)


class TestGem15ItemAssets:
    def setup_method(self):
        gem15.item_assets.cache_clear()

    def test_found(self, monkeypatch):
        resp = MagicMock(status_code=200)
        resp.json.return_value = {"assets": {"dsm": {"href": "D"}, "dtm": {"href": "T"}, "x": {}}}
        monkeypatch.setattr("requests.get", lambda url, timeout: resp)
        assert gem15.item_assets("g", "c") == {"dsm": "D", "dtm": "T"}

    def test_missing_cell(self, monkeypatch):
        monkeypatch.setattr("requests.get", lambda url, timeout: MagicMock(status_code=404))
        assert gem15.item_assets("g", "c404") is None


class TestGem15Download:
    def test_writes_cache(self, tmp_path, monkeypatch):
        dsm, dtm = _gem_files(tmp_path)
        monkeypatch.setattr(gem15, "item_assets", lambda g, c: {"dsm": dsm, "dtm": dtm})
        out = tmp_path / "c" / "7.npz"
        assert gem15.download_tile(TILE, out, size=(16, 16)) is True
        z = np.load(out)
        assert z["dsm"].shape == (5, 16, 16) and z["dtm"].shape == (16, 16)
        assert z["dsm"].dtype == np.float32 and z["pixel_size_m"].shape == (2,)

    def test_no_cell(self, tmp_path, monkeypatch):
        monkeypatch.setattr(gem15, "item_assets", lambda g, c: None)
        assert gem15.download_tile(TILE, tmp_path / "x.npz") is False
        assert not (tmp_path / "x.npz").exists()

    def test_all_nodata(self, tmp_path, monkeypatch):
        dsm, dtm = _gem_files(tmp_path, nan_block=True)
        monkeypatch.setattr(gem15, "item_assets", lambda g, c: {"dsm": dsm, "dtm": dtm})
        assert gem15.download_tile(TILE, tmp_path / "x.npz") is False


class TestTerrainFeatures:
    def _cache(self, dtm):
        dsm = np.stack([dtm + 5, dtm + 1, dtm + 4, dtm + 9, np.full_like(dtm, 2.0)])
        return dsm.astype(np.float32), dtm.astype(np.float32)

    def test_contract_and_values(self):
        dsm, dtm = self._cache(np.full((8, 8), 50.0))
        f = gem15.terrain_features(dsm, dtm, (10.0, 10.0), tpi_radius=1)
        assert f.shape == (len(gem15.FEATURE_NAMES), 8, 8) and f.dtype == np.float32
        named = dict(zip(gem15.FEATURE_NAMES, f))
        assert np.allclose(named["ndsm"], 5) and np.allclose(named["ndsm_p95"], 9)
        assert np.allclose(named["dsm_std"], 2) and np.allclose(named["relief_p95_p5"], 8)
        assert np.allclose(named["slope_deg"], 0) and np.allclose(named["tpi"], 0)
        assert np.allclose(named["valid"], 1)

    def test_slope_of_known_ramp(self):
        dtm = np.arange(8, dtype=np.float32)[:, None].repeat(8, 1) * 10.0  # 10 m rise per 10 m
        dsm, dtm = self._cache(dtm)
        f = gem15.terrain_features(dsm, dtm, (10.0, 10.0))
        assert np.allclose(f[4], 45.0, atol=1e-4)

    def test_tpi_sign(self):
        dtm = np.zeros((9, 9), np.float32)
        dtm[4, 4] = 9.0  # a peak
        dsm, dtm = self._cache(dtm)
        f = gem15.terrain_features(dsm, dtm, (10.0, 10.0), tpi_radius=1)
        assert f[5, 4, 4] > 0 and f[5, 4, 3] < 0

    def test_invalid_pixels_zeroed(self):
        dsm, dtm = self._cache(np.full((4, 4), 10.0))
        dtm[0, 0] = np.nan
        dsm[2, 1, 1] = np.nan
        f = gem15.terrain_features(dsm, dtm, (10.0, 10.0), tpi_radius=1)
        assert f[-1, 0, 0] == 0 and f[-1, 1, 1] == 0 and f[-1, 2, 2] == 1
        assert (f[:-1, 0, 0] == 0).all() and np.isfinite(f).all()

    def test_all_invalid(self):
        dsm, dtm = self._cache(np.full((4, 4), np.nan))
        f = gem15.terrain_features(dsm, dtm, (10.0, 10.0), tpi_radius=1)
        assert (f == 0).all()


class TestGem15Loader:
    def _save(self, path, n=8):
        dtm = np.full((n, n), 10.0, np.float32)
        dsm = np.stack([dtm + 5, dtm, dtm, dtm + 9, dtm * 0 + 1]).astype(np.float32)
        np.savez_compressed(path, dsm=dsm, dtm=dtm, pixel_size_m=np.array([10, 10], np.float32))

    def test_batch_with_missing_and_resize(self, tmp_path):
        self._save(tmp_path / "1.npz", n=4)
        loader = gem15.Gem15CacheLoader(str(tmp_path), tpi_radius=1, unused="ok")
        out = loader.load_batch(["a/1.tif", "2.tif"], "cpu", (8, 8))
        assert out.shape == (2, 7, 8, 8) and out.dtype == torch.float32
        assert torch.allclose(out[0, 0], torch.full((8, 8), 5.0))
        assert torch.equal(out[1], torch.zeros(7, 8, 8))

    def test_same_size_and_empty(self, tmp_path):
        self._save(tmp_path / "3.npz")
        loader = gem15.Gem15CacheLoader(str(tmp_path))
        assert loader.load_batch(["3.tif"], "cpu", (8, 8)).shape == (1, 7, 8, 8)
        assert loader.load_batch([], "cpu", (8, 8)) is None


# ---------------------------------------------------------------------- ForTy
def _forty_file(tmp_path, name="raster_22J_20_0_0.tif", value=50, nodata_all=False):
    data = np.zeros((5, 120, 120), np.uint8)
    data[1] = value  # NaturallyRegenerating
    data[3] = value // 2  # Plantation
    if nodata_all:
        data[:] = 255
    return _write(tmp_path / name, data, 10.0, 255, south_up=True)


class TestForTyNaming:
    def test_gzd(self):
        assert ft.gzd(-50.5, -25.5) == "22J"
        assert ft.gzd(-57.1, -29.8) == "21J"
        assert ft.gzd(-45.0, -20.0) == "23K"
        assert ft.gzd(0.0, -90.0) == "31C" and ft.gzd(0.0, 89.0) == "31X"

    def test_gzds_for_bounds(self):
        assert ft.gzds_for_bounds([(-51.1, -24.1, -50.9, -23.9)]) == ["22J", "22K"]


class TestForTyIndex:
    def test_list_files_paginates(self, monkeypatch):
        pages = [
            {"items": [{"name": "p/raster_22J_a.tif"}, {"name": "p/"}], "nextPageToken": "t"},
            {"items": [{"name": "p/raster_22J_b.tif"}]},
        ]
        calls = []

        def fake_get(url, params, timeout):
            calls.append(params)
            r = MagicMock()
            r.json.return_value = pages[len(calls) - 1]
            return r

        monkeypatch.setattr("requests.get", fake_get)
        assert ft.list_files("22J") == ["p/raster_22J_a.tif", "p/raster_22J_b.tif"]
        assert calls[1]["pageToken"] == "t"

    def test_file_bounds_south_up(self, tmp_path):
        f = _forty_file(tmp_path)
        w, s, e, n = ft._file_bounds(f)
        assert w < e and s < n

    def test_build_index_and_cache(self, tmp_path, monkeypatch):
        f = _forty_file(tmp_path)
        monkeypatch.setattr(ft, "PUBLIC_URL", str(tmp_path) + "/")
        monkeypatch.setattr(ft, "list_files", lambda z: [Path(f).name])
        cache = tmp_path / "idx" / "index.json"
        idx = ft.build_index(["22J"], cache_path=cache, max_workers=2)
        assert list(idx) == [f] and cache.exists()
        monkeypatch.setattr(ft, "list_files", lambda z: pytest.fail("must use cache"))
        assert ft.build_index(["22J"], cache_path=cache) == idx
        assert json.loads(cache.read_text())["zones"] == ["22J"]

    def test_build_index_without_cache(self, tmp_path, monkeypatch):
        monkeypatch.setattr(ft, "list_files", lambda z: [])
        assert ft.build_index(["21J"]) == {}

    def test_files_for_bounds(self):
        idx = {"a": [0, 0, 1, 1], "b": [5, 5, 6, 6]}
        assert ft.files_for_bounds(idx, (0.5, 0.5, 2, 2)) == ["a"]


class TestForTyDownloadAndProbs:
    def test_writes_cache(self, tmp_path):
        f = _forty_file(tmp_path)
        idx = {f: ft._file_bounds(f)}
        out = tmp_path / "c" / "5.npz"
        assert ft.download_tile(TILE, out, idx, size=(16, 16)) is True
        raw = np.load(out)["forty"]
        assert raw.shape == (5, 16, 16) and raw.dtype == np.uint8 and (raw[1] == 50).all()

    def test_no_file_and_all_nodata(self, tmp_path):
        assert ft.download_tile(TILE, tmp_path / "x.npz", {}) is False
        f = _forty_file(tmp_path, nodata_all=True)
        assert ft.download_tile(TILE, tmp_path / "x.npz", {f: ft._file_bounds(f)}) is False
        assert not (tmp_path / "x.npz").exists()

    def test_probabilities(self):
        raw = np.zeros((5, 2, 2), np.uint8)
        raw[0] = 100
        raw[4] = 50
        raw[:, 1, 1] = 255
        p = ft.forty_probabilities(raw)
        assert p.shape == (6, 2, 2) and p.dtype == np.float32
        assert p[0, 0, 0] == pytest.approx(0.4) and p[5, 0, 0] == pytest.approx(0.4)
        assert (p[:, 1, 1] == 0).all()

    def test_other_land_clipped(self):
        raw = np.full((5, 1, 1), 100, np.uint8)  # sums to 2.0
        assert ft.forty_probabilities(raw)[5, 0, 0] == 0.0


class TestForTyLoader:
    def test_batch(self, tmp_path):
        raw = np.zeros((5, 4, 4), np.uint8)
        raw[2] = 250
        np.savez_compressed(tmp_path / "9.npz", forty=raw)
        loader = ft.ForTyCacheLoader(str(tmp_path), extra=1)
        out = loader.load_batch(["x/9.tif", "8.tif"], "cpu", (8, 8))
        assert out.shape == (2, 6, 8, 8)
        assert torch.allclose(out[0, 2], torch.ones(8, 8))
        assert torch.equal(out[1], torch.zeros(6, 8, 8))
        assert loader.load_batch(["9.tif"], "cpu", (4, 4)).shape == (1, 6, 4, 4)
        assert loader.load_batch([], "cpu", (4, 4)) is None


# ------------------------------------------------------------------------ CLI
class TestCli:
    def _manifest(self, tmp_path):
        p = tmp_path / "m.csv"
        pd.DataFrame(
            {"id": [1], "bbox_west": [TILE[0]], "bbox_south": [TILE[1]],
             "bbox_east": [TILE[2]], "bbox_north": [TILE[3]]}
        ).to_csv(p, index=False)
        return p

    def test_download_gem15(self, tmp_path, monkeypatch):
        from pytorch_segmentation_models_trainer.tools.cli import cli

        dsm, dtm = _gem_files(tmp_path)
        monkeypatch.setattr(gem15, "item_assets", lambda g, c: {"dsm": dsm, "dtm": dtm})
        res = CliRunner().invoke(
            cli, ["download-gem15", "--manifest", str(self._manifest(tmp_path)),
                  "--output-dir", str(tmp_path / "out"), "--size", "16"],
        )
        assert res.exit_code == 0, res.output
        assert "written: 1" in res.output, res.output
        assert (tmp_path / "out" / "1.npz").exists()

    def test_download_forest_typology(self, tmp_path, monkeypatch):
        from pytorch_segmentation_models_trainer.tools.cli import cli

        f = _forty_file(tmp_path)
        monkeypatch.setattr(ft, "build_index", lambda zones, cache_path, max_workers: {f: ft._file_bounds(f)})
        res = CliRunner().invoke(
            cli, ["download-forest-typology", "--manifest", str(self._manifest(tmp_path)),
                  "--output-dir", str(tmp_path / "out")],
        )
        assert res.exit_code == 0, res.output
        assert "written: 1" in res.output, res.output
        assert (tmp_path / "out" / "1.npz").exists()

    def test_failed_tiles_are_listed(self, tmp_path, monkeypatch):
        from pytorch_segmentation_models_trainer.tools.cli import cli

        def boom(g, c):
            raise RuntimeError("no COG")

        monkeypatch.setattr(gem15, "item_assets", boom)
        res = CliRunner().invoke(
            cli, ["download-gem15", "--manifest", str(self._manifest(tmp_path)),
                  "--output-dir", str(tmp_path / "out"), "--size", "16"],
        )
        assert res.exit_code == 0, res.output
        assert "failed: 1" in res.output, res.output
        assert "failed 1: " in res.output and "no COG" in res.output, res.output
