# -*- coding: utf-8 -*-
"""Tests for TesseraCacheLoader / dequantize_tessera / _cell_center
(few_shot/tessera_cache.py).

Only the pure cache-reading/crop-math path is tested here (no network) --
download_tile/fetch_cell/open_manifest themselves talk to the live Tessera
HTTPS store and are exercised manually (see the companion research
project's analysis doc for the live coverage probe), not in this offline
suite -- same convention as test_olmoearth_cache.py.
"""

import numpy as np
import pytest
import torch

from pytorch_segmentation_models_trainer.few_shot.tessera_cache import (
    TesseraCacheLoader,
    _cell_center,
    dequantize_tessera,
)


class TestDequantizeTessera:
    def test_matches_the_dataset_s_own_formula(self):
        """v = quantized * scale, per geotessera's own dequantize_embedding."""
        q = np.array([0, 1, -1, 64, -64, 127, -127], dtype=np.int8)
        scale = np.full(7, 0.05, dtype=np.float32)
        expected = q.astype(np.float32) * scale
        np.testing.assert_allclose(dequantize_tessera(q, scale), expected, rtol=1e-6)

    def test_scale_broadcasts_over_the_band_axis(self):
        """scale is (H, W) but quantized is (H, W, n_bands) -- scale must
        broadcast over the last (band) axis, not require a matching shape."""
        q = np.full((2, 2, 4), 10, dtype=np.int8)
        scale = np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32)
        out = dequantize_tessera(q, scale)
        assert out.shape == (2, 2, 4)
        np.testing.assert_allclose(out[0, 0], np.full(4, 1.0), rtol=1e-5)
        np.testing.assert_allclose(out[1, 1], np.full(4, 4.0), rtol=1e-5)

    def test_nonfinite_scale_becomes_zero_not_nan(self):
        """Nodata (e.g. ocean/no-coverage pixels) surfaces as a non-finite
        scale in the live dataset -- must become 0, matching
        AefCacheLoader's zero-fallback convention, never NaN propagating
        into the model."""
        q = np.array([5, 5, 5], dtype=np.int8)
        scale = np.array([0.1, np.nan, np.inf], dtype=np.float32)
        out = dequantize_tessera(q, scale)
        assert out[0] != 0.0
        assert out[1] == 0.0
        assert out[2] == 0.0
        assert np.isfinite(out).all()

    def test_not_a_fixed_linear_scheme(self):
        """Sanity: unlike AlphaEarth's fixed q/127, the same quantized
        value dequantizes differently depending on the (per-pixel) scale."""
        q = np.array([64, 64], dtype=np.int8)
        scale = np.array([0.01, 0.5], dtype=np.float32)
        out = dequantize_tessera(q, scale)
        assert out[0] != out[1]


class TestCellCenter:
    def test_snaps_to_the_dataset_s_own_tiling_convention(self):
        """Cell centers sit at .x5 offsets (0.1x0.1 degree cells) --
        confirmed against the live manifest (e.g. -51.35, -30.85 is a real
        published cell; the project tile at (-51.367011, -30.842242)
        snaps into it)."""
        assert _cell_center(-51.367011, -30.842242) == (-51.35, -30.85)
        assert _cell_center(0.0, 0.0) == (0.05, 0.05)

    def test_negative_coordinates_dont_round_toward_zero(self):
        """A naive int()-based floor would round -51.001 up to -51.0
        instead of down to -51.1 -- must use math.floor, not truncation."""
        assert _cell_center(-51.001, -30.001) == (-51.05, -30.05)


class TestTesseraCacheLoader:
    def test_missing_tile_falls_back_to_zeros(self, tmp_path):
        loader = TesseraCacheLoader(str(tmp_path), n_bands=128)
        out = loader.load_batch(["999.tif"], "cpu", (8, 8))
        assert out.shape == (1, 128, 8, 8)
        assert torch.equal(out, torch.zeros(1, 128, 8, 8))

    def test_empty_mask_paths_returns_none(self, tmp_path):
        loader = TesseraCacheLoader(str(tmp_path))
        assert loader.load_batch([], "cpu", (8, 8)) is None

    def test_loads_and_dequantizes_a_real_cache_file(self, tmp_path):
        raw = np.zeros((8, 8, 4), dtype=np.int8)  # (h, w, n_bands) -- channel-last cache
        raw[..., 0] = 100  # one distinctive band
        scale = np.full((8, 8), 0.01, dtype=np.float32)
        np.savez_compressed(tmp_path / "42.npz", tessera=raw, tessera_scale=scale)

        loader = TesseraCacheLoader(str(tmp_path), n_bands=4)
        out = loader.load_batch(["some/path/42.tif"], "cpu", (8, 8))
        assert out.shape == (1, 4, 8, 8)  # returned tensor is channel-first
        assert torch.allclose(out[0, 0], torch.full((8, 8), 1.0), atol=1e-5)
        assert torch.equal(out[0, 1], torch.zeros(8, 8))

    def test_resizes_when_cache_resolution_differs_from_requested_size(self, tmp_path):
        raw = np.full((4, 4, 2), 100, dtype=np.int8)
        scale = np.full((4, 4), 0.01, dtype=np.float32)
        np.savez_compressed(tmp_path / "7.npz", tessera=raw, tessera_scale=scale)
        loader = TesseraCacheLoader(str(tmp_path), n_bands=2)
        out = loader.load_batch(["7.tif"], "cpu", (8, 8))
        assert out.shape == (1, 2, 8, 8)

    def test_batch_of_multiple_tiles_some_missing(self, tmp_path):
        raw = np.ones((4, 4, 2), dtype=np.int8)
        scale = np.ones((4, 4), dtype=np.float32)
        np.savez_compressed(tmp_path / "1.npz", tessera=raw, tessera_scale=scale)
        loader = TesseraCacheLoader(str(tmp_path), n_bands=2)
        out = loader.load_batch(["1.tif", "2.tif"], "cpu", (4, 4))
        assert out.shape == (2, 2, 4, 4)
        assert not torch.equal(out[0], torch.zeros(2, 4, 4))
        assert torch.equal(out[1], torch.zeros(2, 4, 4))  # "2.tif" has no cache file


class TestDownloadTileManifestReuse:
    """download_tile's manifest param lets a caller downloading many tiles
    load the manifest once (open_manifest()) instead of re-fetching it per
    call -- mirrors OlmoEarthCacheLoader's download_tile ``root`` reuse
    pattern. No network here: confirms the control flow calls
    open_manifest only when manifest is omitted."""

    def test_omitting_manifest_calls_open_manifest(self, monkeypatch):
        import pandas as pd

        import pytorch_segmentation_models_trainer.few_shot.tessera_cache as mod

        calls = {"open_manifest": 0}
        empty = pd.DataFrame(columns=["year", "lon", "lat", "grid_path", "scales_path"])

        def fake_open_manifest(year=None):
            calls["open_manifest"] += 1
            return empty

        monkeypatch.setattr(mod, "open_manifest", fake_open_manifest)
        result = mod.download_tile((-1, -1, 1, 1), 2024, "unused.npz")
        assert result is False  # empty manifest -> no matching cell -> no file written
        assert calls["open_manifest"] == 1

    def test_passing_manifest_skips_open_manifest(self, monkeypatch):
        import pandas as pd

        import pytorch_segmentation_models_trainer.few_shot.tessera_cache as mod

        calls = {"open_manifest": 0}
        monkeypatch.setattr(
            mod, "open_manifest", lambda **kw: calls.__setitem__("open_manifest", calls["open_manifest"] + 1)
        )
        empty = pd.DataFrame(columns=["year", "lon", "lat", "grid_path", "scales_path"])
        result = mod.download_tile((-1, -1, 1, 1), 2024, "unused.npz", manifest=empty)
        assert result is False
        assert calls["open_manifest"] == 0
