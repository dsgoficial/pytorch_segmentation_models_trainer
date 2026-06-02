# -*- coding: utf-8 -*-
"""Local AlphaEarth Foundation embedding utilities.

These helpers mirror the small numerical subset needed from ``aef-loader``
without depending on its remote COG, Dask, Xarray, and Virtualizarr stack.
"""

from __future__ import annotations

from typing import Literal, Tuple

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.transform import Affine
from rasterio.warp import reproject as warp_reproject

AEF_NODATA_VALUE = -128
AEF_DEQUANT_DIVISOR = 127.5

AEFResamplingStrategy = Literal["auto", "aggregate", "nearest", "none"]


def mask_aef_nodata(
    data: np.ndarray,
    nodata_value: int = AEF_NODATA_VALUE,
) -> np.ndarray:
    """Convert AEF NoData sentinel values to ``NaN``.

    Args:
        data: Raw AEF array.
        nodata_value: Integer sentinel used by AEF COGs. Defaults to ``-128``.

    Returns:
        Float32 array with sentinel values replaced by ``NaN``.
    """
    out = data.astype(np.float32, copy=True)
    out[data == nodata_value] = np.nan
    return out


def dequantize_aef(
    data: np.ndarray,
    nodata_value: int = AEF_NODATA_VALUE,
    divisor: float = AEF_DEQUANT_DIVISOR,
) -> np.ndarray:
    """Dequantize raw int8 AEF embeddings to float32.

    Args:
        data: Raw quantized AEF array.
        nodata_value: NoData sentinel. Defaults to ``-128``.
        divisor: Dequantization divisor. Defaults to ``127.5``.

    Returns:
        Float32 array where valid values follow
        ``sign(v) * (abs(v) / divisor) ** 2`` and NoData values are ``NaN``.
    """
    masked = mask_aef_nodata(data, nodata_value=nodata_value)
    return (np.sign(masked) * (np.abs(masked) / divisor) ** 2).astype(np.float32)


def quantize_aef(
    data: np.ndarray,
    divisor: float = AEF_DEQUANT_DIVISOR,
    nodata_value: int = AEF_NODATA_VALUE,
) -> np.ndarray:
    """Quantize float32 AEF embeddings back to int8 storage values.

    Args:
        data: Float embedding array, usually in ``[-1, 1]``.
        divisor: Quantization divisor. Defaults to ``127.5``.
        nodata_value: NoData sentinel written where input is ``NaN``.

    Returns:
        Int8 array with ``NaN`` values encoded as ``-128``.
    """
    clipped = np.clip(data.astype(np.float32), -1.0, 1.0)
    nan_mask = np.isnan(clipped)
    clipped = np.nan_to_num(clipped, nan=0.0)
    quantized = np.sign(clipped) * np.sqrt(np.abs(clipped)) * divisor
    quantized = np.rint(quantized).clip(-127, 127).astype(np.int16)
    quantized[nan_mask] = nodata_value
    return quantized.astype(np.int8)


def valid_aef_vector_mask(data: np.ndarray, axis: int, eps: float = 1e-8) -> np.ndarray:
    """Return mask for vectors with finite values and non-zero norm.

    Args:
        data: Embedding array.
        axis: Vector axis.
        eps: Minimum norm considered valid.

    Returns:
        Boolean mask with vector axis removed.
    """
    finite = np.all(np.isfinite(data), axis=axis)
    norm = np.linalg.norm(np.nan_to_num(data, nan=0.0), axis=axis)
    return finite & (norm > eps)


def normalize_aef_vectors(
    data: np.ndarray,
    axis: int,
    eps: float = 1e-8,
    invalid_value: float = 0.0,
) -> np.ndarray:
    """L2-normalize AEF vectors along one axis.

    Args:
        data: Embedding array.
        axis: Vector axis.
        eps: Small denominator guard.
        invalid_value: Fill value for invalid or zero-norm vectors.

    Returns:
        Float32 array with unit-norm valid vectors.
    """
    values = np.nan_to_num(data.astype(np.float32), nan=0.0)
    norm = np.linalg.norm(values, axis=axis, keepdims=True)
    out = np.divide(values, norm + eps, out=np.zeros_like(values), where=norm > eps)
    valid = valid_aef_vector_mask(data, axis=axis, eps=eps)
    out = np.where(np.expand_dims(valid, axis=axis), out, invalid_value)
    return out.astype(np.float32)


def target_pixel_area(transform: Affine) -> float:
    """Return absolute pixel area for an affine transform."""
    return abs(transform.a * transform.e)


def choose_aef_resampling_strategy(
    src_transform: Affine,
    dst_transform: Affine,
    requested: AEFResamplingStrategy = "auto",
    tolerance: float = 0.99,
) -> AEFResamplingStrategy:
    """Resolve AEF resampling strategy for source and destination grids.

    Args:
        src_transform: Source AEF transform.
        dst_transform: Target image-grid transform.
        requested: ``"auto"``, ``"aggregate"``, ``"nearest"``, or ``"none"``.
        tolerance: Pixel-area tolerance for same/downsample checks.

    Returns:
        Concrete strategy: ``"aggregate"``, ``"nearest"``, or ``"none"``.

    Raises:
        ValueError: If *requested* is unsupported.
    """
    if requested not in {"auto", "aggregate", "nearest", "none"}:
        raise ValueError(
            "aef_resampling must be one of 'auto', 'aggregate', 'nearest', or 'none'."
        )
    if requested != "auto":
        return requested

    src_area = target_pixel_area(src_transform)
    dst_area = target_pixel_area(dst_transform)
    return "aggregate" if dst_area >= src_area * tolerance else "nearest"


def read_aef_raster(path: str) -> Tuple[np.ndarray, Affine, object]:
    """Read local AEF GeoTIFF as ``(D, H, W)`` float32 embeddings.

    Args:
        path: Local AEF GeoTIFF path.

    Returns:
        Tuple ``(data, transform, crs)``. Raw integer rasters are dequantized;
        floating-point rasters are returned as float32.
    """
    with rasterio.open(path) as src:
        data = src.read()
        transform = src.transform
        crs = src.crs

    if np.issubdtype(data.dtype, np.integer):
        data = dequantize_aef(data)
    else:
        data = data.astype(np.float32)
    return data, transform, crs


def aggregate_aef_raster_to_grid(
    path: str,
    dst_height: int,
    dst_width: int,
    dst_transform: Affine,
    dst_crs,
    tolerance: float = 0.99,
) -> np.ndarray:
    """Downsample AEF embeddings with vector-sum aggregation.

    Args:
        path: Local raw AEF GeoTIFF.
        dst_height: Target grid height.
        dst_width: Target grid width.
        dst_transform: Target affine transform.
        dst_crs: Target CRS.
        tolerance: Pixel-area tolerance for downsample validation.

    Returns:
        ``(H, W, D)`` float32 L2-normalized embedding array.

    Raises:
        ValueError: If target pixels are finer than source pixels.
    """
    with rasterio.open(path) as src:
        src_area = target_pixel_area(src.transform)
        dst_area = target_pixel_area(dst_transform)
        if dst_area < src_area * tolerance:
            raise ValueError(
                f"AEF aggregate resampling only supports downscaling "
                f"(target pixel area must be >= source pixel area). "
                f"Source: {src_area:.6g}, target: {dst_area:.6g}."
            )

        raw = src.read()
        if np.issubdtype(raw.dtype, np.integer):
            data = dequantize_aef(raw)
        else:
            data = raw.astype(np.float32)
        valid = np.isfinite(data).astype(np.float32)
        data = np.nan_to_num(data, nan=0.0)
        dst_sum = np.zeros((src.count, dst_height, dst_width), dtype=np.float32)
        dst_valid = np.zeros((src.count, dst_height, dst_width), dtype=np.float32)

        for band_idx in range(src.count):
            warp_reproject(
                source=data[band_idx],
                destination=dst_sum[band_idx],
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=dst_transform,
                dst_crs=dst_crs,
                resampling=Resampling.sum,
            )
            warp_reproject(
                source=valid[band_idx],
                destination=dst_valid[band_idx],
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=dst_transform,
                dst_crs=dst_crs,
                resampling=Resampling.sum,
            )

    any_valid = dst_valid.sum(axis=0, keepdims=True) > 0
    dst_sum = np.where(any_valid, dst_sum, np.nan)
    normalised = normalize_aef_vectors(dst_sum, axis=0)
    return np.transpose(normalised, (1, 2, 0)).astype(np.float32)


def nearest_aef_raster_to_grid(
    path: str,
    dst_height: int,
    dst_width: int,
    dst_transform: Affine,
    dst_crs,
) -> np.ndarray:
    """Resample AEF embeddings to a target grid using nearest neighbor.

    Args:
        path: Local AEF GeoTIFF.
        dst_height: Target grid height.
        dst_width: Target grid width.
        dst_transform: Target affine transform.
        dst_crs: Target CRS.

    Returns:
        ``(H, W, D)`` float32 L2-normalized embedding array.
    """
    with rasterio.open(path) as src:
        raw = src.read()
        if np.issubdtype(raw.dtype, np.integer):
            data = dequantize_aef(raw)
        else:
            data = raw.astype(np.float32)
        dst = np.full((src.count, dst_height, dst_width), np.nan, dtype=np.float32)
        for band_idx in range(src.count):
            warp_reproject(
                source=data[band_idx],
                destination=dst[band_idx],
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=dst_transform,
                dst_crs=dst_crs,
                resampling=Resampling.nearest,
                src_nodata=np.nan,
                dst_nodata=np.nan,
            )

    normalised = normalize_aef_vectors(dst, axis=0)
    return np.transpose(normalised, (1, 2, 0)).astype(np.float32)


def resample_aef_raster_to_grid(
    path: str,
    dst_height: int,
    dst_width: int,
    dst_transform: Affine,
    dst_crs,
    strategy: AEFResamplingStrategy = "auto",
) -> np.ndarray:
    """Resample local AEF raster to target image grid.

    Args:
        path: Local AEF GeoTIFF.
        dst_height: Target grid height.
        dst_width: Target grid width.
        dst_transform: Target affine transform.
        dst_crs: Target CRS.
        strategy: ``"auto"``, ``"aggregate"``, ``"nearest"``, or ``"none"``.

    Returns:
        ``(H, W, D)`` float32 embedding array.

    Raises:
        ValueError: If ``strategy="none"`` is requested with a target grid.
    """
    with rasterio.open(path) as src:
        resolved = choose_aef_resampling_strategy(
            src.transform, dst_transform, requested=strategy
        )

    if resolved == "none":
        raise ValueError("aef_resampling='none' cannot be used with dst_* parameters.")
    if resolved == "aggregate":
        return aggregate_aef_raster_to_grid(
            path, dst_height, dst_width, dst_transform, dst_crs
        )
    if resolved == "nearest":
        return nearest_aef_raster_to_grid(
            path, dst_height, dst_width, dst_transform, dst_crs
        )
    raise ValueError(f"Unsupported AEF resampling strategy: {strategy!r}")
