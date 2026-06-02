---
sidebar_position: 14
title: K-Fold Cross-Validation
---

# K-Fold Cross-Validation

This guide explains how to run spatially-correct k-fold cross-validation with
the `ExperimentsRunner`.  The implementation avoids data leakage from
overlapping patches and spatial autocorrelation at fold boundaries.

## The Problem with Standard K-Fold

When using sliding-window datasets (`CSVWindowedSegmentationDataset`), patches
from the same source image overlap in pixels whenever `stride < patch_size`.
Splitting patches randomly between folds causes **direct data leakage**: the
model sees the same pixels in both training and validation.

Even without pixel overlap, spatially adjacent patches share correlated
information (buildings, roads, vegetation patterns repeat across neighbors).
This inflates validation metrics and leads to overly optimistic generalization
estimates.

## Solution: Group K-Fold by Source Image

The `SpatialKFoldSplitter` uses `sklearn.model_selection.GroupKFold` with the
source image path as the group key.  All patches derived from a single image
land in the **same fold**, completely eliminating pixel-level leakage.

For intra-image splits, an optional **buffer zone** (`buffer_px`) can exclude
patches near fold boundaries, reducing residual spatial autocorrelation.

## Basic Usage

Add a `kfold` block to the `experiments_runner` section of your Hydra config:

```yaml
experiments_runner:
  seeds: [42]
  output_base_dir: outputs/kfold_study
  save_summary: true
  resume: true
  kfold:
    n_splits: 5
    seed: 42
    split_strategy: by_image
    group_col: image
    input_csv_path: /data/all_patches.csv
    save_fold_csvs_dir: fold_splits
```

The runner will:
1. Generate `fold_0_train.csv` … `fold_4_val.csv` in `outputs/kfold_study/fold_splits/`.
2. Iterate over all `seed × fold` combinations (1 seed × 5 folds = 5 runs here).
3. Override `train_dataset.input_csv_path` and `val_dataset.input_csv_path`
   automatically for each run.
4. Write `summary.csv` with a `fold_idx` column alongside the usual metrics.

## Split Strategies

### `by_image` (recommended)

Groups all patches by their source image column (`group_col`, default `"image"`).
Uses `sklearn.model_selection.GroupKFold` — no random state needed, fully
deterministic.

**When to use:** Always, when patches come from multiple source images.  This
is the only strategy that eliminates data leakage from patch overlap.

```yaml
kfold:
  split_strategy: by_image
  group_col: image   # column in the CSV that identifies the source image
  n_splits: 5
```

### `by_spatial_region`

Divides the image height into `n_splits` horizontal bands.  For each fold, one
band is validation and the rest are training.  An optional buffer zone
(`buffer_px`) around each boundary excludes patches whose footprint intersects
the zone.

**When to use:** When all patches come from a single large image (e.g., a
city-wide mosaic) and you want geographically separated train/val regions.

```yaml
kfold:
  split_strategy: by_spatial_region
  n_splits: 5
  buffer_px: 256   # pixels to exclude on each side of each boundary
  row_col: row_off
  col_col: col_off
```

## Buffer Zone for Spatial Autocorrelation

For `by_spatial_region`, patches near fold boundaries share information with
the opposite side.  Setting `buffer_px >= patch_size` guarantees **zero pixel
overlap** between adjacent folds.  Setting `buffer_px >= 2 * patch_size`
further reduces residual autocorrelation from neighborhood context.

```
image height (1000 px)
├── Band 0: [0, 500)  ──── validation for fold 0
│   └── buffer: [244, 500)  ← excluded (buffer_px=256)
├── boundary at 500
│   └── buffer: [500, 756)  ← excluded (buffer_px=256)
└── Band 1: [500, 1000) ── training for fold 0
```

The excluded patches are dropped from **both** train and val sets.

## Combining with Multiple Seeds

Use multiple seeds to estimate variance across both folds and random
initialization:

```yaml
experiments_runner:
  seeds: [42, 101, 28]   # 3 seeds × 5 folds = 15 total runs
  kfold:
    n_splits: 5
    input_csv_path: /data/all_patches.csv
```

The total number of runs is `len(seeds) × n_splits`.  Fold CSVs are generated
**once** and reused across seeds.

## Resuming Interrupted Runs

Set `resume: true` to skip already-completed runs.  State is persisted in
`runner_state.json` after every run:

```yaml
experiments_runner:
  resume: true
  kfold:
    n_splits: 5
    input_csv_path: /data/all_patches.csv
```

Each run has a unique `run_idx` (0 … `n_seeds × n_folds − 1`).  On restart,
completed indices are loaded from `runner_state.json` and skipped.

## Output Structure

```
outputs/kfold_study/
├── fold_splits/
│   ├── fold_0_train.csv
│   ├── fold_0_val.csv
│   ├── fold_1_train.csv
│   ├── fold_1_val.csv
│   └── ...
├── fold_00_seed42/      # per-run Lightning output
├── fold_01_seed42/
├── fold_02_seed42/
├── ...
├── runner_state.json
└── summary.csv          # includes fold_idx column
```

`summary.csv` contains one row per run plus `mean` and `std` aggregation rows:

| run | fold_idx | seed | duration_s | val/loss | val/F1Score |
|-----|----------|------|------------|----------|-------------|
| 0   | 0        | 42   | 120.30     | 0.182    | 0.841       |
| 1   | 1        | 42   | 118.90     | 0.195    | 0.829       |
| … | … | … | … | … | … |
| mean | | - | 119.60 | 0.188 | 0.835 |
| std  | | - | 0.70   | 0.006 | 0.006 |

## Parameter Reference

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `n_splits` | int | 5 | Number of folds. |
| `seed` | int | 42 | Reproducibility seed (used by `by_spatial_region`; `by_image` is fully deterministic). |
| `split_strategy` | str | `"by_image"` | `"by_image"` or `"by_spatial_region"`. |
| `group_col` | str | `"image"` | CSV column used as group key (`by_image` only). |
| `buffer_px` | int | 0 | Exclusion buffer in pixels around boundaries (`by_spatial_region` only). |
| `row_col` | str | `"row_off"` | CSV column for row offset (`by_spatial_region` only). |
| `col_col` | str | `"col_off"` | CSV column for column offset (`by_spatial_region` only). |
| `input_csv_path` | str | `""` | Path to the master CSV with all patches. |
| `save_fold_csvs_dir` | str | `"fold_splits"` | Subdirectory (under `output_base_dir`) for fold CSVs. |

## Full Example

See `conf/examples/kfold_segmentation.yaml` for a complete working configuration.

Also see the [ExperimentsRunner guide](experiments_runner.md) for seed-only
reproducibility studies.
