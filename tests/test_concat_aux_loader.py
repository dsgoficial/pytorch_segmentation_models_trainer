# -*- coding: utf-8 -*-
"""Tests for ConcatAuxLoader (few_shot/concat_aux_loader.py)."""

import pytest
import torch

from pytorch_segmentation_models_trainer.few_shot.aef_cache import AefCacheLoader
from pytorch_segmentation_models_trainer.few_shot.concat_aux_loader import (
    ConcatAuxLoader,
)


class _FakeLoader:
    """Returns a constant-valued batch of the given channel count --
    isolates ConcatAuxLoader's own concatenation logic from any real
    cache-reading loader's behaviour."""

    def __init__(self, n_bands: int, value: float) -> None:
        self.n_bands = n_bands
        self.value = value

    def load_batch(self, mask_paths, device, size):
        if not mask_paths:
            return None
        return torch.full((len(mask_paths), self.n_bands, *size), self.value, device=device)


class TestConstruction:
    def test_accepts_already_built_loaders(self):
        loader = ConcatAuxLoader([_FakeLoader(4, 1.0), _FakeLoader(3, 2.0)])
        assert len(loader.loaders) == 2

    def test_bare_string_resolves_to_aef_cache_loader(self, tmp_path):
        loader = ConcatAuxLoader([str(tmp_path), _FakeLoader(3, 2.0)])
        assert isinstance(loader.loaders[0], AefCacheLoader)

    def test_hydra_subconfig_resolves_via_instantiate(self, tmp_path):
        loader = ConcatAuxLoader(
            [
                {
                    "_target_": "pytorch_segmentation_models_trainer.few_shot.aef_cache.AefCacheLoader",
                    "cache_dir": str(tmp_path),
                    "n_bands": 8,
                },
                _FakeLoader(3, 2.0),
            ]
        )
        assert isinstance(loader.loaders[0], AefCacheLoader)
        assert loader.loaders[0].n_bands == 8

    def test_fewer_than_2_loaders_raises(self):
        with pytest.raises(ValueError, match="at least 2"):
            ConcatAuxLoader([_FakeLoader(4, 1.0)])

    def test_zero_loaders_raises(self):
        with pytest.raises(ValueError, match="at least 2"):
            ConcatAuxLoader([])


class TestLoadBatch:
    def test_concatenates_along_channel_dim_in_order(self):
        loader = ConcatAuxLoader([_FakeLoader(4, 1.0), _FakeLoader(3, 2.0)])
        out = loader.load_batch(["0.tif", "1.tif"], "cpu", (8, 8))
        assert out.shape == (2, 7, 8, 8)
        torch.testing.assert_close(out[:, :4], torch.full((2, 4, 8, 8), 1.0))
        torch.testing.assert_close(out[:, 4:], torch.full((2, 3, 8, 8), 2.0))

    def test_more_than_2_sub_loaders(self):
        loader = ConcatAuxLoader([_FakeLoader(2, 1.0), _FakeLoader(3, 2.0), _FakeLoader(1, 3.0)])
        out = loader.load_batch(["0.tif"], "cpu", (4, 4))
        assert out.shape == (1, 6, 4, 4)

    def test_empty_mask_paths_returns_none(self):
        loader = ConcatAuxLoader([_FakeLoader(4, 1.0), _FakeLoader(3, 2.0)])
        assert loader.load_batch([], "cpu", (8, 8)) is None

    def test_real_aef_cache_loader_as_one_of_the_sub_loaders(self, tmp_path):
        """End-to-end with a real AefCacheLoader (missing-tile fallback
        path, no fixture .npz needed) composed with a fake second
        product -- confirms the resolved sub-loader's own load_batch
        contract (zeros on a missing tile) flows through unchanged."""
        loader = ConcatAuxLoader([AefCacheLoader(str(tmp_path), n_bands=64), _FakeLoader(128, 5.0)])
        out = loader.load_batch(["999.tif"], "cpu", (8, 8))
        assert out.shape == (1, 192, 8, 8)
        torch.testing.assert_close(out[:, :64], torch.zeros(1, 64, 8, 8))
        torch.testing.assert_close(out[:, 64:], torch.full((1, 128, 8, 8), 5.0))

    def test_device_is_respected(self):
        loader = ConcatAuxLoader([_FakeLoader(2, 1.0), _FakeLoader(2, 2.0)])
        out = loader.load_batch(["0.tif"], "cpu", (4, 4))
        assert out.device.type == "cpu"
