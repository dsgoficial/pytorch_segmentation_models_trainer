# -*- coding: utf-8 -*-
"""H3-based spatial validation/test split for manually labeled patches.

Implements the Dataset Partitions protocol from LaTeX §5.1.2:
- Assign patch centroids to H3 cells at resolution 7 (membership by centroid).
- Allocate entire H3 cells to val (~20%) or test (~80%).
- Selection criterion: val class distribution closest to the full set.
- Optionally filter weak-training tiles with a 1 km exclusion buffer.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import pyproj
import rasterio


def get_patch_centroid(image_path: Union[str, Path]) -> Tuple[float, float]:
    """Return the ``(lat, lon)`` centroid of a GeoTIFF in WGS84.

    Args:
        image_path: Path to a georeferenced raster readable by rasterio.

    Returns:
        Tuple ``(lat, lon)`` in decimal degrees (WGS84 / EPSG:4326).
    """
    with rasterio.open(image_path) as src:
        b = src.bounds
        src_crs = src.crs

    cx = (b.left + b.right) / 2.0
    cy = (b.bottom + b.top) / 2.0

    transformer = pyproj.Transformer.from_crs(src_crs, "EPSG:4326", always_xy=True)
    lon, lat = transformer.transform(cx, cy)
    return float(lat), float(lon)


def assign_h3_cells(
    df: pd.DataFrame,
    image_col: str = "image_path",
    resolution: int = 7,
) -> pd.DataFrame:
    """Add ``h3_cell``, ``centroid_lat``, ``centroid_lon`` columns to *df*.

    Args:
        df: DataFrame containing a column of GeoTIFF paths.
        image_col: Name of the column holding GeoTIFF paths.
        resolution: H3 resolution (0–15). Default 7 per §5.1.2.

    Returns:
        Copy of *df* with three new columns appended.

    Raises:
        ImportError: If the ``h3`` package is not installed.
    """
    try:
        import h3
    except ImportError as exc:
        raise ImportError(
            "h3 is required for H3 cell assignment. "
            "Install it with: pip install 'h3>=3.7'"
        ) from exc

    df = df.copy()
    lats: List[float] = []
    lons: List[float] = []
    cells: List[str] = []

    for path in df[image_col]:
        lat, lon = get_patch_centroid(path)
        cell = h3.latlng_to_cell(lat, lon, resolution)
        lats.append(lat)
        lons.append(lon)
        cells.append(cell)

    df["centroid_lat"] = lats
    df["centroid_lon"] = lons
    df["h3_cell"] = cells
    return df


def _class_distribution(df: pd.DataFrame, class_col: str) -> Dict[Any, float]:
    """Return normalized class-frequency dict for *df*.

    Args:
        df: Input DataFrame.
        class_col: Column holding class labels.

    Returns:
        Dict mapping each class label to its relative frequency.
        Returns an empty dict when *df* is empty.
    """
    if df.empty:
        return {}
    total = len(df)
    counts = df[class_col].value_counts()
    return {cls: int(cnt) / total for cls, cnt in counts.items()}


def _score_split(
    val_df: pd.DataFrame,
    full_df: pd.DataFrame,
    class_col: Optional[str],
    val_fraction: float,
) -> float:
    """Compute split quality score (lower is better).

    When *class_col* is ``None``, returns the squared deviation from the
    target *val_fraction*.  Otherwise returns the sum of squared differences
    between val and full class proportions (LaTeX §5.1.2 criterion).

    Args:
        val_df: Candidate validation DataFrame.
        full_df: Full (combined) DataFrame.
        class_col: Column with class labels, or ``None``.
        val_fraction: Target validation fraction.

    Returns:
        Non-negative float score; lower means better split.
    """
    n_full = len(full_df)
    if n_full == 0:
        return np.inf

    if class_col is None:
        frac = len(val_df) / n_full
        return (frac - val_fraction) ** 2

    # Class-distribution distance using all classes present in full_df.
    all_classes = sorted(full_df[class_col].unique())
    full_total = max(len(full_df), 1)
    val_total = max(len(val_df), 1) if len(val_df) > 0 else 1

    full_counts = full_df[class_col].value_counts()
    val_counts = val_df[class_col].value_counts() if len(val_df) > 0 else {}

    score = 0.0
    for cls in all_classes:
        fp = full_counts.get(cls, 0) / full_total
        vp = val_counts.get(cls, 0) / val_total if len(val_df) > 0 else 0.0
        score += (vp - fp) ** 2
    return score


def split_val_test(
    df: pd.DataFrame,
    val_fraction: float = 0.20,
    class_col: Optional[str] = None,
    seed: int = 42,
    n_trials: int = 20_000,
    min_fraction: float = 0.10,
    max_fraction: float = 0.30,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split *df* into ``(val, test)`` keeping H3 cells intact.

    Implements the §5.1.2 criterion: among feasible splits (val fraction within
    ``[min_fraction, max_fraction]``), return the one whose val class
    distribution is closest to the full set.  Uses random search over cell
    orderings.

    When only one H3 cell is present, returns ``(empty_val, full_test)``
    because no two-way cell split is possible.

    Args:
        df: DataFrame with an ``h3_cell`` column (from :func:`assign_h3_cells`).
        val_fraction: Target fraction of patches for validation. Default 0.20.
        class_col: Column name for class labels used to rank candidate splits
            by class balance.  When ``None``, splits are ranked by fraction
            deviation only.
        seed: NumPy random seed for reproducibility.
        n_trials: Number of random cell-order shuffles to evaluate.
        min_fraction: Minimum acceptable val fraction. Default 0.10.
        max_fraction: Maximum acceptable val fraction. Default 0.30.

    Returns:
        Tuple ``(val_df, test_df)`` — non-overlapping DataFrames whose union
        equals *df*.

    Raises:
        ValueError: If *df* has no ``h3_cell`` column.

    Example YAML (for documentation reference)::

        # After running assign_h3_cells, call:
        val_df, test_df = split_val_test(
            df,
            val_fraction=0.20,
            class_col="majority_class",
            seed=42,
        )
    """
    if "h3_cell" not in df.columns:
        raise ValueError(
            "DataFrame must have an 'h3_cell' column. " "Call assign_h3_cells() first."
        )

    cells = list(df["h3_cell"].unique())
    n_cells = len(cells)
    n_patches = len(df)

    if n_cells <= 1:
        return df.iloc[0:0].copy(), df.copy()

    cell_sizes: Dict[str, int] = df.groupby("h3_cell").size().to_dict()
    target_val = val_fraction * n_patches
    rng = np.random.default_rng(seed)

    seen: set = set()
    best_val_cells: Optional[frozenset] = None
    best_score = np.inf

    for _ in range(n_trials):
        shuffled = cells.copy()
        rng.shuffle(shuffled)

        cumulative = 0
        breakpoint_idx = n_cells - 1
        for i, cell in enumerate(shuffled):
            cumulative += cell_sizes[cell]
            if cumulative >= target_val:
                breakpoint_idx = i
                break

        for n_val in [breakpoint_idx, breakpoint_idx + 1]:
            n_val = max(0, min(n_val, n_cells - 1))
            val_cells = frozenset(shuffled[:n_val])
            if val_cells in seen:
                continue
            seen.add(val_cells)

            val_count = sum(cell_sizes[c] for c in val_cells)
            frac = val_count / n_patches
            if not (min_fraction <= frac <= max_fraction):
                continue

            val_candidate = df[df["h3_cell"].isin(val_cells)]
            score = _score_split(val_candidate, df, class_col, val_fraction)
            if score < best_score:
                best_score = score
                best_val_cells = val_cells

    if best_val_cells is None:
        # Fallback: no split found within [min_fraction, max_fraction].
        # Pick the single cell whose count is closest to target_val.
        best_cell = min(cells, key=lambda c: abs(cell_sizes[c] - target_val))
        best_val_cells = frozenset([best_cell])

    val_df = df[df["h3_cell"].isin(best_val_cells)].copy()
    test_df = df[~df["h3_cell"].isin(best_val_cells)].copy()
    return val_df, test_df


def build_exclusion_buffer_gdf(
    patches_df: pd.DataFrame,
    image_col: str = "image_path",
    buffer_m: float = 1000.0,
    native_crs: str = "EPSG:3857",
):
    """Return a GeoDataFrame with a *buffer_m*-metre buffer around each patch.

    Args:
        patches_df: DataFrame whose *image_col* column lists GeoTIFF paths.
        image_col: Column name holding GeoTIFF paths.
        buffer_m: Buffer radius in metres.  The *native_crs* must use metric
            units (e.g. EPSG:3857).
        native_crs: CRS for the output geometries.  Must match the rasters.

    Returns:
        GeoDataFrame with ``geometry`` = buffered patch bounding boxes.
    """
    import geopandas as gpd
    from shapely.geometry import box

    geoms = []
    for path in patches_df[image_col]:
        with rasterio.open(path) as src:
            b = src.bounds
        geoms.append(box(b.left, b.bottom, b.right, b.top))

    gdf = gpd.GeoDataFrame(patches_df.copy(), geometry=geoms, crs=native_crs)
    gdf["geometry"] = gdf.geometry.buffer(buffer_m)
    return gdf


def filter_weak_training_tiles(
    train_df: pd.DataFrame,
    val_test_df: pd.DataFrame,
    image_col: str = "image_path",
    buffer_m: float = 1000.0,
    native_crs: str = "EPSG:3857",
) -> pd.DataFrame:
    """Remove training tiles that intersect the val/test exclusion buffer.

    Args:
        train_df: DataFrame of weak-training tiles with a column of GeoTIFF
            paths.
        val_test_df: Combined val + test DataFrame used to build the buffer.
            Pass an empty DataFrame to skip filtering.
        image_col: Column holding GeoTIFF paths in both DataFrames.
        buffer_m: Exclusion buffer radius in metres (must match *native_crs*
            units).
        native_crs: CRS of all rasters.

    Returns:
        Filtered copy of *train_df* with intersecting tiles removed.
    """
    import geopandas as gpd
    from shapely.geometry import box

    if val_test_df.empty:
        return train_df.copy()

    exclusion_gdf = build_exclusion_buffer_gdf(
        val_test_df,
        image_col=image_col,
        buffer_m=buffer_m,
        native_crs=native_crs,
    )

    train_geoms = []
    for path in train_df[image_col]:
        with rasterio.open(path) as src:
            b = src.bounds
        train_geoms.append(box(b.left, b.bottom, b.right, b.top))

    train_gdf = gpd.GeoDataFrame(train_df.copy(), geometry=train_geoms, crs=native_crs)

    joined = gpd.sjoin(
        train_gdf,
        exclusion_gdf[["geometry"]],
        how="left",
        predicate="intersects",
    )
    keep_idx = joined[joined["index_right"].isna()].index.drop_duplicates()
    return train_df.loc[keep_idx].copy()


def run(
    patches_csv: Union[str, Path],
    output_dir: Union[str, Path],
    image_col: str = "image_path",
    class_col: Optional[str] = None,
    val_fraction: float = 0.20,
    h3_resolution: int = 7,
    seed: int = 42,
    n_trials: int = 20_000,
    train_csv: Optional[Union[str, Path]] = None,
    buffer_m: float = 1000.0,
) -> Dict:
    """Full pipeline: assign H3 cells, split val/test, optionally filter training.

    Args:
        patches_csv: Path to CSV of manually labeled patches.  Must contain
            the column *image_col* with GeoTIFF paths.
        output_dir: Directory where ``val.csv`` and ``test.csv`` are written.
            Created automatically if absent.
        image_col: Column holding GeoTIFF paths. Default ``"image_path"``.
        class_col: Column with class labels for distribution-aware splitting.
            When ``None``, splits are ranked by fraction deviation only.
        val_fraction: Target validation fraction. Default 0.20.
        h3_resolution: H3 resolution for cell assignment. Default 7.
        seed: Random seed. Default 42.
        n_trials: Random-search iterations. Default 20 000.
        train_csv: Optional path to weak-training tile CSV.  When provided,
            tiles overlapping the 1 km exclusion buffer are removed and the
            result is written to ``train_filtered.csv``.  When ``None``, the
            filtering step is skipped entirely.
        buffer_m: Exclusion buffer radius in metres. Default 1000.

    Returns:
        Dict with keys ``n_val``, ``n_test``, ``val_fraction``,
        ``val_h3_cells``, ``test_h3_cells``.  When *train_csv* is provided,
        also includes ``n_train_before`` and ``n_train_after``.
    """
    patches_csv = Path(patches_csv)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(patches_csv)
    df = assign_h3_cells(df, image_col=image_col, resolution=h3_resolution)

    val_df, test_df = split_val_test(
        df,
        val_fraction=val_fraction,
        class_col=class_col,
        seed=seed,
        n_trials=n_trials,
    )

    val_df.to_csv(output_dir / "val.csv", index=False)
    test_df.to_csv(output_dir / "test.csv", index=False)

    stats: Dict = {
        "n_val": len(val_df),
        "n_test": len(test_df),
        "val_fraction": len(val_df) / max(len(df), 1),
        "val_h3_cells": sorted(val_df["h3_cell"].unique().tolist()),
        "test_h3_cells": sorted(test_df["h3_cell"].unique().tolist()),
    }

    if train_csv is not None:
        train_df = pd.read_csv(train_csv)
        stats["n_train_before"] = len(train_df)
        val_test_combined = pd.concat([val_df, test_df], ignore_index=True)
        filtered = filter_weak_training_tiles(
            train_df,
            val_test_combined,
            image_col=image_col,
            buffer_m=buffer_m,
        )
        filtered.to_csv(output_dir / "train_filtered.csv", index=False)
        stats["n_train_after"] = len(filtered)

    return stats
