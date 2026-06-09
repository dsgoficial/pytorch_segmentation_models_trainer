# -*- coding: utf-8 -*-
"""Per-patch mask statistics: class distribution, entropy, nodata ratio, border checks."""

from typing import Dict

import numpy as np
import rasterio
import rasterio.windows


def has_border_nodata_mask(mask: np.ndarray, nodata_class: int = 0) -> bool:
    """Return True if any outer-border pixel of mask equals nodata_class.

    Args:
        mask: 2-D array of shape (H, W).
        nodata_class: pixel value considered as no-data.

    Returns:
        True when at least one border pixel matches nodata_class.
    """
    border = np.concatenate([mask[0, :], mask[-1, :], mask[:, 0], mask[:, -1]])
    return bool(np.any(border == nodata_class))


def has_black_border_image(image: np.ndarray, tolerance: int = 0) -> bool:
    """Return True if any outer-border pixel of image is black (all channels ≤ tolerance).

    Args:
        image: array of shape (H, W) or (H, W, C).
        tolerance: pixel value at or below this threshold is considered black.

    Returns:
        True when at least one border pixel is fully black.
    """
    if image.ndim == 2:
        border = np.concatenate([image[0, :], image[-1, :], image[:, 0], image[:, -1]])
        return bool(np.any(border <= tolerance))
    border = np.concatenate(
        [image[0, :, :], image[-1, :, :], image[:, 0, :], image[:, -1, :]]
    )
    return bool(np.any(np.all(border <= tolerance, axis=-1)))


def compute_mask_stats(
    mask_path: str,
    row_off: int,
    col_off: int,
    patch_size: int,
    nodata_class: int = 0,
) -> Dict:
    """Compute class distribution, entropy, and nodata ratio for a mask patch.

    Args:
        mask_path: path to a rasterio-readable single-band mask raster.
        row_off: top-left row offset of the patch window.
        col_off: top-left column offset of the patch window.
        patch_size: height and width of the patch in pixels.
        nodata_class: pixel value excluded from class distribution.

    Returns:
        dict with keys:
            class_dist (dict[int, float]): fraction per valid class (sums to 1).
            class_entropy (float | None): Shannon entropy; None if all pixels are nodata.
            nodata_ratio (float): fraction of nodata pixels in the patch.
    """
    window = rasterio.windows.Window(col_off, row_off, patch_size, patch_size)
    with rasterio.open(mask_path) as src:
        data = src.read(1, window=window)

    total = data.size
    nodata_mask = data == nodata_class
    nodata_ratio = float(nodata_mask.sum()) / total

    valid = data[~nodata_mask]
    if valid.size == 0:
        return {"class_dist": {}, "class_entropy": None, "nodata_ratio": nodata_ratio}

    classes, counts = np.unique(valid, return_counts=True)
    proportions = counts / counts.sum()
    class_dist = {int(c): float(p) for c, p in zip(classes, proportions)}
    entropy = float(-np.sum(proportions * np.log(proportions + 1e-12)))
    return {
        "class_dist": class_dist,
        "class_entropy": entropy,
        "nodata_ratio": nodata_ratio,
    }
