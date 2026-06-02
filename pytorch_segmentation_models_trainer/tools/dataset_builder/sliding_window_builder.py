"""
sliding_window_builder.py — Crop existing image/mask pairs into sliding-window patches.

Reads a CSV of image/mask pairs, cuts each pair into fixed-size overlapping
patches, and writes an output CSV pointing to the new patches.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd


def compute_sliding_windows(
    height: int,
    width: int,
    window_size: int,
    overlap: float,
) -> List[Tuple[int, int, int, int]]:
    """Return ``(row_start, row_end, col_start, col_end)`` for each patch.

    Edge patches are adjusted so they stay within bounds while maintaining a
    fixed size of ``window_size × window_size`` pixels.

    Args:
        height: Image height in pixels.
        width: Image width in pixels.
        window_size: Patch size in pixels (square).
        overlap: Overlap fraction in ``[0, 1)``.  Stride is
            ``window_size * (1 - overlap)``.

    Returns:
        List of ``(row_start, row_end, col_start, col_end)`` tuples.

    Example::

        windows = compute_sliding_windows(512, 512, 256, 0.5)
    """
    if overlap < 0 or overlap >= 1:
        raise ValueError(f"overlap must be in [0, 1), got {overlap}")

    stride = max(1, int(window_size * (1 - overlap)))
    windows = []

    row = 0
    while row < height:
        col = 0
        while col < width:
            r0 = min(row, max(0, height - window_size))
            c0 = min(col, max(0, width - window_size))
            r1 = r0 + window_size
            c1 = c0 + window_size
            windows.append((r0, r1, c0, c1))
            col += stride
        row += stride

    return windows


def _process_pair(
    image_path: Path,
    mask_path: Path,
    output_dir: Path,
    window_size: int,
    overlap: float,
    class_remap: Optional[Dict[int, int]],
    mi: Optional[str],
) -> List[dict]:
    """Crop a single image/mask pair into sliding-window patches."""
    import numpy as np
    import rasterio
    from rasterio.windows import Window, transform as window_transform

    rows = []

    with rasterio.open(image_path) as img_src:
        img_profile = img_src.profile.copy()
        img_transform = img_src.transform
        height = img_src.height
        width = img_src.width

    with rasterio.open(mask_path) as mask_src:
        mask_profile = mask_src.profile.copy()

    windows = compute_sliding_windows(height, width, window_size, overlap)

    # Preserve relative structure inside output_dir
    rel_stem = image_path.parent.name
    img_out_dir = output_dir / rel_stem / "images"
    mask_out_dir = output_dir / rel_stem / "masks"
    img_out_dir.mkdir(parents=True, exist_ok=True)
    mask_out_dir.mkdir(parents=True, exist_ok=True)

    for idx, (r0, r1, c0, c1) in enumerate(windows):
        win = Window(col_off=c0, row_off=r0, width=window_size, height=window_size)
        patch_transform = window_transform(win, img_transform)

        with rasterio.open(image_path) as img_src:
            tile_data = img_src.read(window=win)

        with rasterio.open(mask_path) as mask_src:
            mask_data = mask_src.read(window=win)

        if class_remap:
            for old_cls, new_cls in class_remap.items():
                mask_data[mask_data == old_cls] = new_cls

        tile_name = f"{image_path.stem}_r{r0}_c{c0}.tif"

        img_out = img_out_dir / tile_name
        mask_out = mask_out_dir / tile_name

        out_img_profile = img_profile.copy()
        out_img_profile.update(
            driver="GTiff",
            width=window_size,
            height=window_size,
            transform=patch_transform,
        )
        with rasterio.open(img_out, "w", **out_img_profile) as dst:
            dst.write(tile_data)

        out_mask_profile = mask_profile.copy()
        out_mask_profile.update(
            driver="GTiff",
            width=window_size,
            height=window_size,
            transform=patch_transform,
        )
        with rasterio.open(mask_out, "w", **out_mask_profile) as dst:
            dst.write(mask_data)

        row_dict = {
            "image": str(img_out),
            "mask": str(mask_out),
            "rows": window_size,
            "columns": window_size,
        }
        if mi is not None:
            row_dict["mi"] = mi

        rows.append(row_dict)

    return rows


def build_sliding_window_dataset(
    input_csv: Path,
    output_dir: Path,
    window_size: int = 256,
    overlap: float = 0.0,
    class_remap: Optional[Dict[int, int]] = None,
    blacklist: Optional[List[str]] = None,
    n_workers: int = 8,
    progress: bool = True,
) -> pd.DataFrame:
    """Crop image/mask pairs from a CSV into sliding-window patches.

    *input_csv* must have columns ``image`` and ``mask``.  The optional
    ``mi`` column (map index) is preserved in the output CSV.

    Args:
        input_csv: CSV with at least ``image`` and ``mask`` columns.
        output_dir: Root output directory for patches.
        window_size: Patch size in pixels (square).
        overlap: Overlap fraction in ``[0, 1)``.
        class_remap: Mapping ``{old_class: new_class}`` applied to mask
            patches before saving.
        blacklist: Directory name segments to skip.  Any row whose image path
            contains a blacklisted segment is ignored.
        n_workers: Number of worker threads.
        progress: Whether to show a tqdm progress bar.

    Returns:
        DataFrame with columns ``image``, ``mask``, ``rows``, ``columns``
        (plus ``mi`` if present in *input_csv*).  Also saved to
        ``output_dir / {input_csv.stem}.csv``.
    """
    try:
        from tqdm import tqdm
    except ImportError:  # pragma: no cover
        tqdm = None  # type: ignore[assignment]

    output_dir.mkdir(parents=True, exist_ok=True)

    df_in = pd.read_csv(input_csv)
    has_mi = "mi" in df_in.columns

    if blacklist:
        mask_bl = df_in["image"].apply(
            lambda p: any(seg in str(p) for seg in blacklist)
        )
        df_in = df_in[~mask_bl].reset_index(drop=True)

    iterator = df_in.iterrows()
    if progress and tqdm is not None:
        iterator = tqdm(
            df_in.iterrows(), total=len(df_in), desc="Sliding window", unit="pair"
        )

    all_rows: List[dict] = []

    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        futures = {}
        for _, row in iterator:
            img_path = Path(row["image"])
            msk_path = Path(row["mask"])
            mi_val = str(row["mi"]) if has_mi else None
            fut = executor.submit(
                _process_pair,
                img_path,
                msk_path,
                output_dir,
                window_size,
                overlap,
                class_remap,
                mi_val,
            )
            futures[fut] = (img_path, msk_path)

        for future in as_completed(futures):
            rows = future.result()
            all_rows.extend(rows)

    cols = ["image", "mask", "rows", "columns"]
    if has_mi:
        cols = ["image", "mask", "rows", "columns", "mi"]

    df_out = pd.DataFrame(all_rows, columns=cols)
    out_csv = output_dir / f"{input_csv.stem}.csv"
    df_out.to_csv(out_csv, index=False)

    return df_out
