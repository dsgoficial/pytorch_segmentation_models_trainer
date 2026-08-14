---
sidebar_position: 8
title: H3 Spatial Val/Test Split
---

# H3 Spatial Val/Test Split

Standard random train/val/test splits leak spatial autocorrelation: patches from the same area
appear in both training and validation, inflating metrics. H3-based splitting assigns each patch
to an H3 hexagonal cell (≈1.4 km edge at resolution 7) and allocates **entire cells** to val or
test, ensuring geographic separation between splits.

This implements the §5.1.2 Dataset Partitions protocol: a 20 000-trial random search finds the
H3-cell-intact split whose validation class distribution is closest to the full set (target
val ≈ 20%).

---

## Installation

```bash
pip install 'h3>=3.7.0,<5' geopandas shapely pyproj
```

`h3` is required for cell assignment. `geopandas` and `shapely` are needed only for the optional
1 km training exclusion buffer.

---

## Quick Start

### CLI

```bash
pytorch-smt-tools split-manual-patches \
    --input /path/to/patches.csv \
    --output /path/to/splits/ \
    --val-fraction 0.20 \
    --class-col majority_class \
    --seed 42
```

Outputs `val.csv` and `test.csv` in the output directory. When `--train-csv` is provided, a
`train_filtered.csv` is also written with training tiles that overlap the 1 km exclusion buffer
removed.

### CLI options

| Option | Default | Description |
|---|---|---|
| `--input` | required | CSV of manually labeled patches with `image_path` column |
| `--output` | required | Directory for output CSVs |
| `--val-fraction` | `0.20` | Target validation fraction |
| `--class-col` | `None` | Column with class labels for distribution-aware splitting |
| `--h3-resolution` | `7` | H3 resolution (0–15); resolution 7 ≈ 1.4 km edge |
| `--seed` | `42` | Random seed for reproducibility |
| `--n-trials` | `20000` | Random-search iterations for optimal split |
| `--train-csv` | `None` | Weak-training CSV; overlapping tiles removed |
| `--buffer-m` | `1000.0` | Exclusion buffer radius in metres |

---

## How It Works

1. **H3 cell assignment** — each patch centroid (read from the GeoTIFF) is assigned to the H3
   cell at the configured resolution. All patches sharing a cell are kept together.

2. **Random-search split** — 20 000 random cell-order shuffles are evaluated. For each
   candidate split, the score is the sum of squared differences between val and full-set class
   distributions. The best split within `[min_fraction, max_fraction]` is kept.

3. **Training exclusion buffer** (optional) — builds a 1 km buffer around all val+test patches
   and removes training tiles that intersect it, preventing contamination from spatially adjacent
   weak-label tiles.

---

## Python API

### Full pipeline

```python
from pytorch_segmentation_models_trainer.utils.h3_val_test_split import run

stats = run(
    patches_csv="/path/to/patches.csv",
    output_dir="/path/to/splits/",
    image_col="image_path",
    class_col="majority_class",
    val_fraction=0.20,
    h3_resolution=7,
    seed=42,
    n_trials=20_000,
    train_csv="/path/to/weak_train.csv",  # optional
    buffer_m=1000.0,
)

print(stats)
# {
#   "n_val": 312, "n_test": 1245,
#   "val_fraction": 0.198,
#   "val_h3_cells": [...], "test_h3_cells": [...],
#   "n_train_before": 8000, "n_train_after": 7650,
# }
```

### Step-by-step

```python
import pandas as pd
from pytorch_segmentation_models_trainer.utils.h3_val_test_split import (
    assign_h3_cells,
    split_val_test,
    filter_weak_training_tiles,
)

# 1. Assign H3 cells (reads GeoTIFF centroid for each row)
df = pd.read_csv("patches.csv")
df = assign_h3_cells(df, image_col="image_path", resolution=7)
# Adds: centroid_lat, centroid_lon, h3_cell columns

# 2. Split val / test keeping cells intact
val_df, test_df = split_val_test(
    df,
    val_fraction=0.20,
    class_col="majority_class",  # None to skip class-aware ranking
    seed=42,
    n_trials=20_000,
)

val_df.to_csv("val.csv", index=False)
test_df.to_csv("test.csv", index=False)

# 3. (Optional) Filter weak-training tiles near val/test
train_df = pd.read_csv("weak_train.csv")
val_test = pd.concat([val_df, test_df], ignore_index=True)
filtered = filter_weak_training_tiles(
    train_df,
    val_test,
    image_col="image_path",
    buffer_m=1000.0,
)
filtered.to_csv("train_filtered.csv", index=False)
```

---

## Input CSV Format

The patches CSV must contain at minimum a column of GeoTIFF paths (default: `image_path`).
Each GeoTIFF must be georeferenced — the centroid is computed from rasterio bounds.

| Column | Required | Description |
|---|---|---|
| `image_path` | Yes (configurable) | Path to a georeferenced GeoTIFF |
| `majority_class` | Optional | Class label for distribution-aware split ranking |
| Any other columns | — | Preserved unchanged in output CSVs |

---

## H3 Resolution Guide

| Resolution | Avg edge | Patches per cell (typical) | Use case |
|---|---|---|---|
| 6 | ≈ 5.7 km | Many — coarse blocks | Very large datasets |
| 7 | ≈ 2.1 km | Medium — recommended | Standard annotation campaigns |
| 8 | ≈ 0.8 km | Few — fine-grained | Dense annotation grids |

Resolution 7 is the default, matching the §5.1.2 protocol. Increase resolution if patches are
small and numerous; decrease if patches are sparse and large.

---

## API Reference

### `assign_h3_cells(df, image_col, resolution)`

Reads each GeoTIFF to get its centroid and assigns it to an H3 cell.

| Arg | Default | Description |
|---|---|---|
| `df` | — | DataFrame with GeoTIFF paths |
| `image_col` | `"image_path"` | Column holding GeoTIFF paths |
| `resolution` | `7` | H3 resolution (0–15) |

Returns: copy of `df` with `centroid_lat`, `centroid_lon`, `h3_cell` columns added.

### `split_val_test(df, val_fraction, class_col, seed, n_trials, min_fraction, max_fraction)`

| Arg | Default | Description |
|---|---|---|
| `val_fraction` | `0.20` | Target validation fraction |
| `class_col` | `None` | Class label column for distribution-aware ranking |
| `seed` | `42` | NumPy random seed |
| `n_trials` | `20_000` | Random-search iterations |
| `min_fraction` | `0.10` | Minimum acceptable val fraction |
| `max_fraction` | `0.30` | Maximum acceptable val fraction |

Returns: `(val_df, test_df)` — non-overlapping DataFrames whose union equals the input.

### `filter_weak_training_tiles(train_df, val_test_df, image_col, buffer_m, native_crs)`

| Arg | Default | Description |
|---|---|---|
| `buffer_m` | `1000.0` | Exclusion radius in metres |
| `native_crs` | `"EPSG:3857"` | CRS for buffer geometry (must use metric units) |

Returns: filtered copy of `train_df` with intersecting tiles removed.

---

## Related

- [K-Fold Cross-Validation](kfold.md) — spatial k-fold for model selection
- [Balanced Dataset Sampling](balanced-dataset-sampling.md) — class-balanced patch weighting
