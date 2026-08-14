# -*- coding: utf-8 -*-
"""Tests for tools/sampling/mask_stats.py."""

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_bounds

from pytorch_segmentation_models_trainer.tools.sampling.mask_stats import (
    compute_mask_stats,
    has_black_border_image,
    has_border_nodata_mask,
)


def _write_mask(path, data: np.ndarray):
    h, w = data.shape
    transform = from_bounds(0, 0, 1, 1, w, h)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=h,
        width=w,
        count=1,
        dtype=data.dtype.name,
        crs="EPSG:4326",
        transform=transform,
    ) as dst:
        dst.write(data[np.newaxis])


# ---------------------------------------------------------------------------
# has_border_nodata_mask
# ---------------------------------------------------------------------------


def test_has_border_nodata_mask_true_top():
    mask = np.ones((8, 8), dtype=np.uint8)
    mask[0, 3] = 0
    assert has_border_nodata_mask(mask, nodata_class=0) is True


def test_has_border_nodata_mask_true_bottom():
    mask = np.ones((8, 8), dtype=np.uint8)
    mask[-1, 3] = 0
    assert has_border_nodata_mask(mask, nodata_class=0) is True


def test_has_border_nodata_mask_true_left():
    mask = np.ones((8, 8), dtype=np.uint8)
    mask[3, 0] = 0
    assert has_border_nodata_mask(mask, nodata_class=0) is True


def test_has_border_nodata_mask_true_right():
    mask = np.ones((8, 8), dtype=np.uint8)
    mask[3, -1] = 0
    assert has_border_nodata_mask(mask, nodata_class=0) is True


def test_has_border_nodata_mask_interior_only():
    mask = np.ones((8, 8), dtype=np.uint8)
    mask[3, 3] = 0
    assert has_border_nodata_mask(mask, nodata_class=0) is False


def test_has_border_nodata_mask_no_nodata():
    mask = np.ones((8, 8), dtype=np.uint8) * 2
    assert has_border_nodata_mask(mask, nodata_class=0) is False


def test_has_border_nodata_mask_custom_nodata_class():
    mask = np.ones((8, 8), dtype=np.uint8)
    mask[0, 0] = 5
    assert has_border_nodata_mask(mask, nodata_class=5) is True
    assert has_border_nodata_mask(mask, nodata_class=0) is False


# ---------------------------------------------------------------------------
# has_black_border_image
# ---------------------------------------------------------------------------


def test_has_black_border_image_true_2d():
    img = np.ones((8, 8), dtype=np.uint8) * 100
    img[0, 4] = 0
    assert has_black_border_image(img, tolerance=0) is True


def test_has_black_border_image_false_2d():
    img = np.ones((8, 8), dtype=np.uint8) * 100
    assert has_black_border_image(img, tolerance=0) is False


def test_has_black_border_image_3d_true():
    img = np.ones((8, 8, 3), dtype=np.uint8) * 100
    img[0, 4, :] = 0
    assert has_black_border_image(img, tolerance=0) is True


def test_has_black_border_image_3d_partial_channel_false():
    img = np.ones((8, 8, 3), dtype=np.uint8) * 100
    img[0, 4, 0] = 0  # only channel 0, not all → not black
    assert has_black_border_image(img, tolerance=0) is False


def test_has_black_border_tolerance_true():
    img = np.ones((8, 8), dtype=np.uint8) * 5
    img[0, 0] = 3
    assert has_black_border_image(img, tolerance=3) is True


def test_has_black_border_tolerance_false():
    img = np.ones((8, 8), dtype=np.uint8) * 5
    img[0, 0] = 3
    assert has_black_border_image(img, tolerance=2) is False


def test_has_black_border_interior_only():
    img = np.ones((8, 8), dtype=np.uint8) * 100
    img[4, 4] = 0
    assert has_black_border_image(img, tolerance=0) is False


def test_has_black_border_image_bottom_right():
    img = np.ones((8, 8), dtype=np.uint8) * 100
    img[-1, -1] = 0
    assert has_black_border_image(img, tolerance=0) is True


# ---------------------------------------------------------------------------
# compute_mask_stats
# ---------------------------------------------------------------------------


def test_compute_mask_stats_full_image(tmp_path):
    mask = np.array(
        [[1, 1, 2, 2], [1, 2, 2, 3], [3, 3, 3, 3], [1, 2, 3, 1]], dtype=np.uint8
    )
    path = tmp_path / "mask.tif"
    _write_mask(path, mask)
    stats = compute_mask_stats(
        str(path), row_off=0, col_off=0, patch_size=4, nodata_class=0
    )
    assert "class_dist" in stats
    assert "class_entropy" in stats
    assert "nodata_ratio" in stats
    dist_values = list(stats["class_dist"].values())
    assert abs(sum(dist_values) - 1.0) < 1e-6


def test_compute_mask_stats_nodata_only(tmp_path):
    mask = np.zeros((4, 4), dtype=np.uint8)
    path = tmp_path / "mask.tif"
    _write_mask(path, mask)
    stats = compute_mask_stats(
        str(path), row_off=0, col_off=0, patch_size=4, nodata_class=0
    )
    assert stats["class_dist"] == {}
    assert stats["class_entropy"] is None
    assert stats["nodata_ratio"] == pytest.approx(1.0)


def test_compute_mask_stats_excludes_nodata(tmp_path):
    mask = np.array(
        [[0, 0, 1, 1], [0, 0, 1, 1], [0, 0, 2, 2], [0, 0, 2, 2]], dtype=np.uint8
    )
    path = tmp_path / "mask.tif"
    _write_mask(path, mask)
    stats = compute_mask_stats(
        str(path), row_off=0, col_off=0, patch_size=4, nodata_class=0
    )
    assert 0 not in stats["class_dist"]
    assert stats["nodata_ratio"] == pytest.approx(0.5)
    assert stats["class_dist"][1] == pytest.approx(0.5)
    assert stats["class_dist"][2] == pytest.approx(0.5)


def test_compute_mask_stats_window(tmp_path):
    mask = np.zeros((8, 8), dtype=np.uint8)
    mask[:4, :4] = 1
    mask[4:, 4:] = 2
    path = tmp_path / "mask.tif"
    _write_mask(path, mask)
    stats = compute_mask_stats(
        str(path), row_off=0, col_off=0, patch_size=4, nodata_class=0
    )
    assert stats["class_dist"] == {1: pytest.approx(1.0)}


def test_compute_mask_stats_entropy_positive(tmp_path):
    mask = np.array([[1, 2], [3, 4]], dtype=np.uint8)
    path = tmp_path / "mask.tif"
    _write_mask(path, mask)
    stats = compute_mask_stats(
        str(path), row_off=0, col_off=0, patch_size=2, nodata_class=0
    )
    assert stats["class_entropy"] is not None
    assert stats["class_entropy"] > 0
