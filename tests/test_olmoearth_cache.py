# -*- coding: utf-8 -*-
"""Tests for OlmoEarthCacheLoader / dequantize_olmoearth (few_shot/olmoearth_cache.py).

Only the pure cache-reading path is tested here (no network) -- download_tile
itself talks to the live OlmoEarth S3-compatible store and is exercised
manually (see the companion research project's analysis doc for the live
probe that confirmed coverage), not in this offline suite.
"""

import numpy as np
import pytest
import torch

from pytorch_segmentation_models_trainer.few_shot.olmoearth_cache import (
    OlmoEarthCacheLoader,
    dequantize_olmoearth,
)


class TestDequantizeOlmoearth:
    def test_matches_the_dataset_s_own_formula(self):
        """v = sign(q) * (|q| / 127.5) ** 2, per the dataset's README."""
        q = np.array([0, 1, -1, 64, -64, 127, -127], dtype=np.int8)
        expected = np.sign(q.astype(np.float32)) * (np.abs(q.astype(np.float32)) / 127.5) ** 2
        np.testing.assert_allclose(dequantize_olmoearth(q), expected, rtol=1e-6)

    def test_nodata_becomes_zero_not_nan(self):
        """Unlike the dataset's own example (-> NaN), this loader's
        zero-fallback matches AefCacheLoader's convention -- a neutral
        value the gate's nn.Linear can learn to discount, never NaN
        propagating into the model."""
        q = np.array([-128, 0, 5], dtype=np.int8)
        out = dequantize_olmoearth(q)
        assert out[0] == 0.0
        assert np.isfinite(out).all()

    def test_not_the_alphaearth_linear_scheme(self):
        """Sanity: this must NOT equal AefCacheLoader's q/127 formula --
        the two products use different quantization schemes."""
        q = np.array([64], dtype=np.int8)
        olmo = dequantize_olmoearth(q)[0]
        aef_style = 64 / 127.0
        assert abs(olmo - aef_style) > 1e-3

    def test_output_range_matches_reported_embedding_norms(self):
        """Per-value magnitude stays in [-1, 1]; the dataset's README
        reports per-pixel L2 norms over 128 dims around 3.0-3.8, consistent
        with individual dequantized values being well under 1."""
        q = np.full(128, 127, dtype=np.int8)
        out = dequantize_olmoearth(q)
        assert (np.abs(out) <= 1.0 + 1e-6).all()


class TestOlmoEarthCacheLoader:
    def test_missing_tile_falls_back_to_zeros(self, tmp_path):
        loader = OlmoEarthCacheLoader(str(tmp_path), n_bands=64)
        out = loader.load_batch(["999.tif"], "cpu", (8, 8))
        assert out.shape == (1, 64, 8, 8)
        assert torch.equal(out, torch.zeros(1, 64, 8, 8))

    def test_empty_mask_paths_returns_none(self, tmp_path):
        loader = OlmoEarthCacheLoader(str(tmp_path))
        assert loader.load_batch([], "cpu", (8, 8)) is None

    def test_loads_and_dequantizes_a_real_cache_file(self, tmp_path):
        raw = np.zeros((4, 8, 8), dtype=np.int8)
        raw[0] = 64  # one distinctive band
        np.savez_compressed(tmp_path / "42.npz", olmoearth=raw)

        loader = OlmoEarthCacheLoader(str(tmp_path), n_bands=4)
        out = loader.load_batch(["some/path/42.tif"], "cpu", (8, 8))
        assert out.shape == (1, 4, 8, 8)
        expected_band0 = dequantize_olmoearth(np.array([64], dtype=np.int8))[0]
        assert torch.allclose(out[0, 0], torch.full((8, 8), float(expected_band0)), atol=1e-5)
        assert torch.equal(out[0, 1], torch.zeros(8, 8))  # band quantized to 0 -> dequant 0

    def test_resizes_when_cache_resolution_differs_from_requested_size(self, tmp_path):
        raw = np.full((2, 4, 4), 100, dtype=np.int8)
        np.savez_compressed(tmp_path / "7.npz", olmoearth=raw)
        loader = OlmoEarthCacheLoader(str(tmp_path), n_bands=2)
        out = loader.load_batch(["7.tif"], "cpu", (8, 8))
        assert out.shape == (1, 2, 8, 8)

    def test_batch_of_multiple_tiles_some_missing(self, tmp_path):
        raw = np.ones((2, 4, 4), dtype=np.int8)
        np.savez_compressed(tmp_path / "1.npz", olmoearth=raw)
        loader = OlmoEarthCacheLoader(str(tmp_path), n_bands=2)
        out = loader.load_batch(["1.tif", "2.tif"], "cpu", (4, 4))
        assert out.shape == (2, 2, 4, 4)
        assert not torch.equal(out[0], torch.zeros(2, 4, 4))
        assert torch.equal(out[1], torch.zeros(2, 4, 4))  # "2.tif" has no cache file
