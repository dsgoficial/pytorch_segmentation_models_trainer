# -*- coding: utf-8 -*-
"""Raster alignment helpers for MBTiles imagery and GeoTIFF masks."""

from pathlib import Path
from typing import Iterable, Optional, Sequence

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.vrt import WarpedVRT
from rasterio.windows import Window

_RESAMPLING = {
    "nearest": Resampling.nearest,
    "bilinear": Resampling.bilinear,
    "cubic": Resampling.cubic,
    "average": Resampling.average,
}


def resolve_resampling(name: str) -> Resampling:
    """Resolve a rasterio resampling method name.

    Args:
        name: Resampling method name. Supported values are ``"nearest"``,
            ``"bilinear"``, ``"cubic"``, and ``"average"``.

    Returns:
        Matching :class:`rasterio.enums.Resampling` enum value.

    Raises:
        ValueError: If *name* is unknown.
    """
    key = name.lower()
    if key not in _RESAMPLING:
        raise ValueError(
            f"Unknown resampling method '{name}'. "
            f"Accepted values: {sorted(_RESAMPLING)}"
        )
    return _RESAMPLING[key]


def normalize_selected_bands(
    selected_bands: Optional[Sequence[int]], band_count: int
) -> Optional[Sequence[int]]:
    """Validate selected raster bands.

    Args:
        selected_bands: Optional 1-based band indexes.
        band_count: Number of bands in the source raster.

    Returns:
        ``None`` when all bands should be read, otherwise band indexes.

    Raises:
        ValueError: If a band index is outside ``[1, band_count]``.
    """
    if selected_bands is None:
        return None
    if not all(isinstance(b, int) and 1 <= b <= band_count for b in selected_bands):
        raise ValueError(
            f"selected_bands must contain 1-based indexes in [1, {band_count}]"
        )
    return selected_bands


def read_source_aligned_to_mask_window(
    source_path: Path,
    mask_src: rasterio.io.DatasetReader,
    window: Window,
    selected_bands: Optional[Sequence[int]] = None,
    image_dtype: str = "uint8",
    image_resampling: str = "bilinear",
) -> np.ndarray:
    """Read source imagery warped onto the exact grid of a mask window.

    The destination CRS, transform, width, and height are derived from the mask
    raster window. This keeps the mask as the training reference grid while the
    MBTiles/source imagery is resampled into that grid.

    Args:
        source_path: Path to the MBTiles or any raster readable by rasterio.
        mask_src: Open mask raster dataset.
        window: Mask pixel window to use as destination grid.
        selected_bands: Optional 1-based source band indexes.
        image_dtype: Output numpy dtype, or ``"native"`` to preserve source
            dtype.
        image_resampling: Rasterio resampling method for imagery.

    Returns:
        Array with shape ``(C, H, W)`` aligned to *window*.

    Example YAML:
        ```yaml
        mbtiles_export:
          mbtiles_path: /data/source.mbtiles
          mask_dir: /data/masks
          output_dir: /data/qa
          patch_size: 512
          stride: 512
        ```
    """
    dst_width = int(window.width)
    dst_height = int(window.height)
    dst_transform = mask_src.window_transform(window)
    dst_crs = mask_src.crs
    resampling = resolve_resampling(image_resampling)

    with rasterio.open(source_path) as source_src:
        bands = normalize_selected_bands(selected_bands, source_src.count)
        indexes: Optional[Iterable[int]] = bands
        with WarpedVRT(
            source_src,
            crs=dst_crs,
            transform=dst_transform,
            width=dst_width,
            height=dst_height,
            resampling=resampling,
        ) as vrt:
            data = vrt.read(indexes=indexes)

    if image_dtype == "native":
        return data
    return data.astype(np.dtype(image_dtype), copy=False)


def read_mask_window(
    mask_src: rasterio.io.DatasetReader,
    window: Window,
    n_classes: int = 2,
) -> np.ndarray:
    """Read a single-band mask window using class-index conventions.

    Args:
        mask_src: Open mask raster dataset.
        window: Pixel window to read.
        n_classes: Number of classes. When ``2``, all values greater than zero
            are mapped to foreground class ``1``.

    Returns:
        ``uint8`` mask array with shape ``(H, W)``.
    """
    mask = mask_src.read(1, window=window).astype(np.uint8, copy=False)
    if n_classes == 2:
        mask = (mask > 0).astype(np.uint8)
    return mask
