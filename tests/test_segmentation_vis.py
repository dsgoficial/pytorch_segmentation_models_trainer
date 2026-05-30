# flake8: noqa
"""Tests for tools/visualization/segmentation_vis.py."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
import rasterio
from rasterio.transform import from_bounds


def make_tiff(path: Path, data: np.ndarray, crs: str = "EPSG:4326") -> None:
    """Helper: write a minimal GeoTIFF."""
    h = data.shape[-2] if data.ndim > 2 else data.shape[0]
    w = data.shape[-1] if data.ndim > 2 else data.shape[1]
    transform = from_bounds(0, 0, 1, 1, w, h)
    bands = 1 if data.ndim == 2 else data.shape[0]
    profile = dict(
        driver="GTiff",
        dtype=data.dtype,
        width=w,
        height=h,
        count=bands,
        crs=crs,
        transform=transform,
    )
    with rasterio.open(path, "w", **profile) as dst:
        if data.ndim == 2:
            dst.write(data, 1)
        else:
            dst.write(data)


from pytorch_segmentation_models_trainer.tools.visualization.segmentation_vis import (
    ColorMap,
    colorize_mask,
    create_segmentation_grid,
    prepare_image_for_display,
)

COLOR_MAP: ColorMap = {
    0: (0, 0, 0),
    1: (255, 0, 0),
    2: (0, 255, 0),
}


# ---- colorize_mask ----


def test_colorize_mask_maps_class_to_color() -> None:
    """Class pixels should map to their corresponding RGB color."""
    mask = np.array([[0, 1], [2, 1]], dtype=np.uint8)
    rgb = colorize_mask(mask, COLOR_MAP)

    assert rgb.shape == (2, 2, 3)
    np.testing.assert_array_equal(rgb[0, 0], [0, 0, 0])
    np.testing.assert_array_equal(rgb[0, 1], [255, 0, 0])
    np.testing.assert_array_equal(rgb[1, 0], [0, 255, 0])


def test_colorize_mask_nodata_becomes_white() -> None:
    """Nodata value should map to white."""
    mask = np.array([[255, 0]], dtype=np.uint8)
    rgb = colorize_mask(mask, COLOR_MAP, nodata_value=255)

    np.testing.assert_array_equal(rgb[0, 0], [255, 255, 255])


def test_colorize_mask_unknown_class_remains_black() -> None:
    """Unknown class IDs (not in color_map, not nodata) should be black."""
    mask = np.array([[99]], dtype=np.uint8)
    rgb = colorize_mask(mask, COLOR_MAP, nodata_value=255)

    np.testing.assert_array_equal(rgb[0, 0], [0, 0, 0])


# ---- prepare_image_for_display ----


def test_prepare_image_for_display_uint8_normalized() -> None:
    """uint8 input should be divided by 255."""
    img = np.full((3, 4, 4), fill_value=128, dtype=np.uint8)
    result = prepare_image_for_display(img)

    assert result.dtype == np.float32
    assert abs(result.mean() - 128 / 255) < 1e-4


def test_prepare_image_for_display_uint16_normalized() -> None:
    """uint16 input should be divided by 65535."""
    img = np.full((3, 4, 4), fill_value=32768, dtype=np.uint16)
    result = prepare_image_for_display(img)

    assert result.dtype == np.float32
    assert abs(result.mean() - 32768 / 65535) < 1e-4


def test_prepare_image_for_display_float_already_unit() -> None:
    """float input already in [0,1] should be returned clipped."""
    img = np.full((3, 4, 4), fill_value=0.5, dtype=np.float32)
    result = prepare_image_for_display(img)

    assert result.min() >= 0.0
    assert result.max() <= 1.0
    assert abs(result.mean() - 0.5) < 1e-4


def test_prepare_image_for_display_float_surface_reflectance() -> None:
    """float input with max > 1.0 should be divided by 10000."""
    img = np.full((3, 4, 4), fill_value=5000.0, dtype=np.float32)
    result = prepare_image_for_display(img)

    assert abs(result.mean() - 0.5) < 1e-4


def test_prepare_image_for_display_chw_to_hwc() -> None:
    """(C, H, W) input should be transposed to (H, W, C)."""
    img = np.zeros((4, 16, 8), dtype=np.uint8)  # 4 bands, H=16, W=8
    result = prepare_image_for_display(img)

    assert result.shape == (16, 8, 3)


def test_prepare_image_for_display_clips_to_3_bands() -> None:
    """Only the first 3 bands should be kept."""
    img = np.zeros((6, 8, 8), dtype=np.uint8)
    result = prepare_image_for_display(img)

    assert result.shape[2] == 3


# ---- create_segmentation_grid ----


@pytest.fixture()
def grid_fixtures(tmp_path: Path):
    """Create minimal GT and prediction rasters + records DataFrame."""
    gt_dir = tmp_path / "gt"
    pred_dir = tmp_path / "pred"
    mi = "mi_001"
    tile_id = "tile_001"

    (gt_dir / mi).mkdir(parents=True)
    (pred_dir / mi).mkdir(parents=True)

    gt_mask = np.zeros((32, 32), dtype=np.uint8)
    gt_mask[8:24, 8:24] = 1
    make_tiff(gt_dir / mi / f"{tile_id}.tif", gt_mask)

    pred_mask = np.zeros((32, 32), dtype=np.uint8)
    pred_mask[10:22, 10:22] = 1
    make_tiff(pred_dir / mi / f"{tile_id}.tif", pred_mask)

    records = pd.DataFrame({"tile_id": [tile_id], "mi": [mi], "mean_iou": [0.75]})

    return records, gt_dir, [pred_dir], ["model_A"], tmp_path


def test_create_segmentation_grid_returns_figure(grid_fixtures) -> None:
    """create_segmentation_grid should return a matplotlib Figure."""
    records, gt_dir, pred_dirs, labels, tmp_path = grid_fixtures

    fig = create_segmentation_grid(
        records=records,
        gt_dir=gt_dir,
        pred_dirs=pred_dirs,
        pred_labels=labels,
        color_map=COLOR_MAP,
        n_samples=1,
        mode="best",
    )

    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_create_segmentation_grid_invalid_mode_raises(grid_fixtures) -> None:
    """An invalid mode should raise ValueError."""
    records, gt_dir, pred_dirs, labels, tmp_path = grid_fixtures

    with pytest.raises(ValueError, match="mode must be one of"):
        create_segmentation_grid(
            records=records,
            gt_dir=gt_dir,
            pred_dirs=pred_dirs,
            pred_labels=labels,
            color_map=COLOR_MAP,
            mode="invalid_mode",
        )


def test_create_segmentation_grid_saves_file(grid_fixtures) -> None:
    """Providing output_path should save the figure to disk."""
    records, gt_dir, pred_dirs, labels, tmp_path = grid_fixtures
    out_path = tmp_path / "grid.png"

    fig = create_segmentation_grid(
        records=records,
        gt_dir=gt_dir,
        pred_dirs=pred_dirs,
        pred_labels=labels,
        color_map=COLOR_MAP,
        output_path=out_path,
        n_samples=1,
        mode="best",
    )

    assert out_path.exists()
    assert out_path.stat().st_size > 0
    plt.close(fig)


def test_create_segmentation_grid_random_mode_uses_sample_and_legend(
    grid_fixtures,
) -> None:
    """Random mode should sample rows and render the optional legend."""
    records, gt_dir, pred_dirs, labels, tmp_path = grid_fixtures
    class_labels = {0: "background", 1: "foreground"}

    fig = create_segmentation_grid(
        records=pd.concat([records, records.assign(mean_iou=0.9)]),
        gt_dir=gt_dir,
        pred_dirs=pred_dirs,
        pred_labels=labels,
        color_map=COLOR_MAP,
        class_labels=class_labels,
        n_samples=1,
        mode="random",
    )

    assert len(fig.axes) == 2
    assert fig.legends, "Expected a legend when class_labels is provided"
    plt.close(fig)


def test_create_segmentation_grid_with_image_dir_and_worst_mode(grid_fixtures) -> None:
    """The image column and worst-mode path should both be exercised."""
    records, gt_dir, pred_dirs, labels, tmp_path = grid_fixtures
    image_dir = tmp_path / "images"
    (image_dir / "mi_001").mkdir(parents=True)
    make_tiff(
        image_dir / "mi_001" / "tile_001.tif", np.ones((3, 32, 32), dtype=np.uint8)
    )

    fig = create_segmentation_grid(
        records=records.assign(mean_iou=[0.1]),
        gt_dir=gt_dir,
        pred_dirs=pred_dirs,
        pred_labels=labels,
        color_map=COLOR_MAP,
        image_dir=image_dir,
        n_samples=1,
        mode="worst",
    )

    assert len(fig.axes) == 3
    plt.close(fig)


def test_prepare_image_for_display_2d_input_padded_to_3_channels() -> None:
    """A 2D (H, W) array should become (H, W, 3) after padding."""
    img = np.full((8, 8), fill_value=200, dtype=np.uint8)
    result = prepare_image_for_display(img)

    assert result.shape == (8, 8, 3)
    assert result.dtype == np.float32


def test_create_segmentation_grid_missing_files_display_na(tmp_path: Path) -> None:
    """When raster files do not exist, cells should render N/A text (no crash)."""
    gt_dir = tmp_path / "gt"
    pred_dir = tmp_path / "pred"
    image_dir = tmp_path / "images"
    gt_dir.mkdir()
    pred_dir.mkdir()
    image_dir.mkdir()

    records = pd.DataFrame({"tile_id": ["missing"], "mi": ["mi_x"], "mean_iou": [0.0]})

    fig = create_segmentation_grid(
        records=records,
        gt_dir=gt_dir,
        pred_dirs=[pred_dir],
        pred_labels=["model"],
        color_map=COLOR_MAP,
        image_dir=image_dir,
        n_samples=1,
        mode="best",
    )

    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_create_segmentation_grid_pred_shape_mismatch_is_reprojected(
    grid_fixtures,
) -> None:
    """When pred shape differs from gt shape, _reproject_to_gt should resize it."""
    records, gt_dir, _, labels, tmp_path = grid_fixtures

    pred_dir_small = tmp_path / "pred_small"
    (pred_dir_small / "mi_001").mkdir(parents=True)
    small_mask = np.zeros((16, 16), dtype=np.uint8)
    small_mask[4:12, 4:12] = 1
    make_tiff(pred_dir_small / "mi_001" / "tile_001.tif", small_mask)

    fig = create_segmentation_grid(
        records=records,
        gt_dir=gt_dir,
        pred_dirs=[pred_dir_small],
        pred_labels=["small_model"],
        color_map=COLOR_MAP,
        n_samples=1,
        mode="best",
    )

    assert isinstance(fig, plt.Figure)
    plt.close(fig)
