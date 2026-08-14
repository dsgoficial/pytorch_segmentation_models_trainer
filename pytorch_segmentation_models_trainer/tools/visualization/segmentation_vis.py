# flake8: noqa
"""
segmentation_vis.py — Colorization and comparison grid utilities for segmentation results.

Provides helpers to colorize integer class masks, normalize raster arrays for
display, and build publication-quality comparison grids comparing GT against
one or more prediction sets.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

from matplotlib.figure import Figure
import numpy as np
import pandas as pd
from concurrent.futures import ThreadPoolExecutor

# Type alias exported for callers
ColorMap = Dict[int, Tuple[int, int, int]]  # {class_id: (R, G, B)}


def colorize_mask(
    mask: np.ndarray,
    color_map: ColorMap,
    nodata_value: int = 255,
) -> np.ndarray:
    """Convert a 2-D integer class mask to an ``(H, W, 3)`` uint8 RGB array.

    Unknown class IDs that are not in *color_map* and are not the nodata value
    are mapped to black ``(0, 0, 0)``.  The *nodata_value* maps to white
    ``(255, 255, 255)``.

    Args:
        mask: 2-D integer array of class IDs.
        color_map: Mapping ``{class_id: (R, G, B)}`` where each channel is in
            ``[0, 255]``.
        nodata_value: Class ID treated as nodata; mapped to white.

    Returns:
        ``(H, W, 3)`` uint8 array.

    Example::

        rgb = colorize_mask(mask, color_map={0: (0, 0, 0), 1: (255, 0, 0)})
    """
    h, w = mask.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)

    for cls_id, color in color_map.items():
        rgb[mask == cls_id] = color

    rgb[mask == nodata_value] = (255, 255, 255)

    return rgb


def prepare_image_for_display(image: np.ndarray) -> np.ndarray:
    """Normalize a raster array to float32 in ``[0, 1]`` for matplotlib.

    Handles ``(bands, H, W)`` → ``(H, W, bands)`` transposition.
    Uses only the first 3 bands for RGB display.

    Scaling rules:

    - ``uint8``: divided by 255.
    - ``uint16``: divided by 65535.
    - ``float``: clipped to ``[0, 1]``; if ``max > 1.0`` the array is first
      divided by 10000 (assumes surface reflectance in the 0–10000 range).

    Args:
        image: Input array.  Can be ``(H, W)``, ``(H, W, C)``, or
            ``(C, H, W)``.

    Returns:
        ``(H, W, 3)`` float32 array with values in ``[0, 1]``.

    Example::

        display = prepare_image_for_display(data)
    """
    arr = image.copy()

    # Handle (C, H, W) → (H, W, C)
    if arr.ndim == 3 and arr.shape[0] < arr.shape[-1]:
        arr = np.transpose(arr, (1, 2, 0))

    # Ensure (H, W, C)
    if arr.ndim == 2:
        arr = arr[:, :, np.newaxis]

    # Keep only first 3 bands
    arr = arr[:, :, :3]

    # Normalize by dtype
    if arr.dtype == np.uint8:
        result = arr.astype(np.float32) / 255.0
    elif arr.dtype == np.uint16:
        result = arr.astype(np.float32) / 65535.0
    else:
        result = arr.astype(np.float32)
        if result.max() > 1.0:
            result = result / 10000.0
        result = np.clip(result, 0.0, 1.0)

    # Pad to 3 channels if fewer
    if result.shape[2] < 3:
        pad = np.zeros(
            (result.shape[0], result.shape[1], 3 - result.shape[2]), dtype=np.float32
        )
        result = np.concatenate([result, pad], axis=2)

    return result


def create_segmentation_grid(
    records: pd.DataFrame,
    gt_dir: Path,
    pred_dirs: List[Path],
    pred_labels: List[str],
    color_map: ColorMap,
    class_labels: Optional[Dict[int, str]] = None,
    output_path: Optional[Path] = None,
    image_dir: Optional[Path] = None,
    sort_by: str = "mean_iou",
    n_samples: int = 5,
    mode: str = "best",
    figsize_per_tile: Tuple[float, float] = (3.0, 3.0),
    dpi: int = 150,
    nodata: int = 255,
) -> Figure:
    """Build a comparison grid figure: image | GT | pred_1 ... pred_N per row.

    Args:
        records: DataFrame with columns ``tile_id``, ``mi``, and the column
            named by *sort_by*.
        gt_dir: Root directory for GT masks.  Layout: ``{gt_dir}/{mi}/{tile_id}.tif``
            with ``{gt_dir}/{tile_id}.tif`` as flat fallback.
        pred_dirs: List of directories for each prediction experiment.
            Same layout as *gt_dir*.
        pred_labels: Display labels for each prediction directory.
        color_map: Mapping used by :func:`colorize_mask`.
        class_labels: Optional ``{class_id: name}`` for the legend.
        output_path: If provided, saves the figure to this path.
        image_dir: If provided, the first column shows the source image.
            Same layout as *gt_dir*.
        sort_by: Column in *records* used for sorting.
        n_samples: Number of rows in the grid.
        mode: ``"best"``, ``"worst"``, or ``"random"``.
        figsize_per_tile: Figure size per tile cell ``(width, height)`` in
            inches.
        dpi: Figure DPI.
        nodata: Nodata value forwarded to :func:`colorize_mask`.

    Returns:
        The ``matplotlib.figure.Figure`` object.

    Raises:
        ValueError: If *mode* is not one of ``"best"``, ``"worst"``,
            ``"random"``.
    """
    import matplotlib.patches as mpatches
    import matplotlib.pyplot as plt
    import rasterio

    valid_modes = {"best", "worst", "random"}
    if mode not in valid_modes:
        raise ValueError(f"mode must be one of {sorted(valid_modes)}, got '{mode}'.")

    # Sort / sample records
    df = records.copy()
    if mode == "best":
        df = df.sort_values(sort_by, ascending=False).head(n_samples)
    elif mode == "worst":
        df = df.sort_values(sort_by, ascending=True).head(n_samples)
    else:  # random
        n = min(n_samples, len(df))
        df = df.sample(n=n, random_state=0)

    df = df.reset_index(drop=True)
    actual_samples = len(df)

    def _load_mask(base_dir: Path, mi: str, tile_id: str) -> Optional[np.ndarray]:
        """Try {base_dir}/{mi}/{tile_id}.tif then {base_dir}/{tile_id}.tif."""
        candidates = [
            base_dir / str(mi) / f"{tile_id}.tif",
            base_dir / f"{tile_id}.tif",
        ]
        for p in candidates:
            if p.exists():
                with rasterio.open(p) as src:
                    data = src.read(1)
                return data
        return None

    def _load_image(base_dir: Path, mi: str, tile_id: str) -> Optional[np.ndarray]:
        candidates = [
            base_dir / str(mi) / f"{tile_id}.tif",
            base_dir / f"{tile_id}.tif",
        ]
        for p in candidates:
            if p.exists():
                with rasterio.open(p) as src:
                    data = src.read()
                return data
        return None

    def _reproject_to_gt(pred_arr: np.ndarray, gt_arr: np.ndarray) -> np.ndarray:
        """Resize pred to match gt shape if needed (simple nearest-neighbor)."""
        if pred_arr.shape == gt_arr.shape:
            return pred_arr
        from rasterio.enums import Resampling
        from rasterio.transform import from_bounds
        import rasterio.warp as warp

        # Simple resize via zoom
        from scipy.ndimage import zoom

        zy = gt_arr.shape[0] / pred_arr.shape[0]
        zx = gt_arr.shape[1] / pred_arr.shape[1]
        return zoom(pred_arr, (zy, zx), order=0).astype(pred_arr.dtype)

    def _load_record_payload(record: pd.Series) -> dict:
        """Load all raster payloads needed to render a single row."""
        tile_id = str(record["tile_id"])
        mi = str(record["mi"])

        image_data = _load_image(image_dir, mi, tile_id) if image_dir else None
        gt_data = _load_mask(gt_dir, mi, tile_id)
        pred_data = [_load_mask(pred_dir, mi, tile_id) for pred_dir in pred_dirs]

        return {
            "tile_id": tile_id,
            "mi": mi,
            "image_data": image_data,
            "gt_data": gt_data,
            "pred_data": pred_data,
        }

    # Determine number of columns
    n_cols = 1  # GT
    if image_dir is not None:
        n_cols += 1  # image
    n_cols += len(pred_dirs)

    fig_w = n_cols * figsize_per_tile[0]
    fig_h = actual_samples * figsize_per_tile[1]

    if class_labels:
        fig_h += figsize_per_tile[1] * 0.5  # extra space for legend

    fig, axes = plt.subplots(
        actual_samples,
        n_cols,
        figsize=(fig_w, fig_h),
        squeeze=False,
    )

    col_titles = []
    if image_dir is not None:
        col_titles.append("Image")
    col_titles.append("GT")
    col_titles += pred_labels

    for col_idx, title in enumerate(col_titles):
        axes[0, col_idx].set_title(title, fontsize=9, pad=3)

    with ThreadPoolExecutor(max_workers=min(len(df), 8) or 1) as executor:
        payloads = list(
            executor.map(_load_record_payload, [row for _, row in df.iterrows()])
        )

    for row_idx, payload in enumerate(payloads):
        tile_id = payload["tile_id"]
        mi = payload["mi"]
        gt_data = payload["gt_data"]
        pred_payloads = payload["pred_data"]

        col = 0

        if image_dir is not None:
            ax = axes[row_idx, col]
            if payload["image_data"] is not None:
                ax.imshow(prepare_image_for_display(payload["image_data"]))
            else:
                ax.text(
                    0.5, 0.5, "N/A", ha="center", va="center", transform=ax.transAxes
                )
            ax.axis("off")
            col += 1

        ax = axes[row_idx, col]
        if gt_data is not None:
            ax.imshow(colorize_mask(gt_data, color_map, nodata_value=nodata))
        else:
            ax.text(0.5, 0.5, "N/A", ha="center", va="center", transform=ax.transAxes)
        ax.axis("off")
        col += 1

        for pred_data in pred_payloads:
            ax = axes[row_idx, col]
            if pred_data is not None:
                if gt_data is not None:
                    pred_data = _reproject_to_gt(pred_data, gt_data)
                ax.imshow(colorize_mask(pred_data, color_map, nodata_value=nodata))
            else:
                ax.text(
                    0.5, 0.5, "N/A", ha="center", va="center", transform=ax.transAxes
                )
            ax.axis("off")
            col += 1

    if class_labels:
        patches = [
            mpatches.Patch(
                color=tuple(v / 255 for v in color_map.get(cls_id, (128, 128, 128))),
                label=label,
            )
            for cls_id, label in class_labels.items()
        ]
        fig.legend(
            handles=patches,
            loc="lower center",
            ncol=min(len(patches), 6),
            fontsize=8,
            bbox_to_anchor=(0.5, 0.0),
        )

    fig.tight_layout()

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=dpi, bbox_inches="tight")

    return fig
