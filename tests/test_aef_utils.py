# -*- coding: utf-8 -*-
"""Tests for local AlphaEarth Foundation embedding utilities."""

import numpy as np
import pytest
import rasterio
from pathlib import Path
from rasterio.crs import CRS
from rasterio.transform import from_bounds

from pytorch_segmentation_models_trainer.tools.soft_labels.aef_utils import (
    AEF_NODATA_VALUE,
    choose_aef_resampling_strategy,
    dequantize_aef,
    nearest_aef_raster_to_grid,
    normalize_aef_vectors,
    quantize_aef,
    resample_aef_raster_to_grid,
    valid_aef_vector_mask,
)

EPSG4326 = CRS.from_epsg(4326)
BOUNDS = (0.0, 0.0, 1.0, 1.0)


def _write_aef_tif(path: Path, data: np.ndarray, transform, crs=EPSG4326) -> None:
    """Write a multi-band AEF-like raster."""
    d, h, w = data.shape
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=h,
        width=w,
        count=d,
        dtype=str(data.dtype),
        crs=crs,
        transform=transform,
    ) as dst:
        dst.write(data)


class TestAefNumericHelpers:
    def test_dequantize_contract_and_nodata(self):
        raw = np.array([127, -127, 0, AEF_NODATA_VALUE], dtype=np.int8)
        result = dequantize_aef(raw)
        assert result.dtype == np.float32
        assert result.shape == raw.shape
        assert np.isclose(result[0], (127 / 127.5) ** 2)
        assert np.isclose(result[1], -((127 / 127.5) ** 2))
        assert result[2] == 0.0
        assert np.isnan(result[3])

    def test_quantize_roundtrip_and_nodata(self):
        values = np.array([1.0, -1.0, 0.0, np.nan], dtype=np.float32)
        result = quantize_aef(values)
        assert result.dtype == np.int8
        assert result.tolist() == [127, -127, 0, AEF_NODATA_VALUE]

    def test_normalize_vectors_handles_zero_and_nan(self):
        data = np.array(
            [
                [[3.0, 0.0], [np.nan, 1.0]],
                [[4.0, 0.0], [1.0, 1.0]],
            ],
            dtype=np.float32,
        )
        result = normalize_aef_vectors(data, axis=0)
        norms = np.linalg.norm(result, axis=0)
        assert np.isclose(norms[0, 0], 1.0)
        assert norms[0, 1] == 0.0
        assert norms[1, 0] == 0.0

    def test_valid_vector_mask_rejects_nan_and_zero(self):
        data = np.array([[[1.0, 0.0], [np.nan, 1.0]], [[0.0, 0.0], [1.0, 1.0]]])
        mask = valid_aef_vector_mask(data.astype(np.float32), axis=0)
        assert mask.tolist() == [[True, False], [False, True]]


class TestAefRasterResampling:
    def test_choose_auto_aggregate_when_target_coarser(self):
        src = from_bounds(*BOUNDS, 8, 8)
        dst = from_bounds(*BOUNDS, 4, 4)
        assert choose_aef_resampling_strategy(src, dst, "auto") == "aggregate"

    def test_choose_auto_nearest_when_target_finer(self):
        src = from_bounds(*BOUNDS, 4, 4)
        dst = from_bounds(*BOUNDS, 8, 8)
        assert choose_aef_resampling_strategy(src, dst, "auto") == "nearest"

    def test_nearest_upsample_contract(self, tmp_path):
        src_h, src_w, d = 4, 4, 3
        dst_h, dst_w = 8, 8
        src_transform = from_bounds(*BOUNDS, src_w, src_h)
        dst_transform = from_bounds(*BOUNDS, dst_w, dst_h)
        data = np.ones((d, src_h, src_w), dtype=np.int8) * 64
        path = tmp_path / "aef.tif"
        _write_aef_tif(path, data, src_transform)

        result = nearest_aef_raster_to_grid(
            str(path), dst_h, dst_w, dst_transform, EPSG4326
        )

        assert result.shape == (dst_h, dst_w, d)
        assert result.dtype == np.float32
        assert np.allclose(np.linalg.norm(result, axis=2), 1.0, atol=1e-5)

    def test_auto_upsample_uses_nearest(self, tmp_path):
        src_h, src_w, d = 4, 4, 3
        dst_h, dst_w = 8, 8
        src_transform = from_bounds(*BOUNDS, src_w, src_h)
        dst_transform = from_bounds(*BOUNDS, dst_w, dst_h)
        data = np.ones((d, src_h, src_w), dtype=np.int8) * 64
        path = tmp_path / "aef.tif"
        _write_aef_tif(path, data, src_transform)

        result = resample_aef_raster_to_grid(
            str(path),
            dst_h,
            dst_w,
            dst_transform,
            EPSG4326,
            strategy="auto",
        )

        assert result.shape == (dst_h, dst_w, d)

    def test_none_with_target_grid_raises(self, tmp_path):
        transform = from_bounds(*BOUNDS, 4, 4)
        data = np.ones((3, 4, 4), dtype=np.int8)
        path = tmp_path / "aef.tif"
        _write_aef_tif(path, data, transform)

        with pytest.raises(ValueError, match="none"):
            resample_aef_raster_to_grid(
                str(path), 4, 4, transform, EPSG4326, strategy="none"
            )
