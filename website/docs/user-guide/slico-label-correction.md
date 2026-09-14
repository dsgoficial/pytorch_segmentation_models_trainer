---
sidebar_position: 27
title: SLICO Label Correction
---

# SLICO Label Correction

`build-slico-corrected-masks` corrects noisy segmentation masks the same way
[SAM Label Correction](sam-label-correction.md) does — majority vote *within each
region* across the original topographic vector source and optional auxiliary LULC
rasters — but generates the regions with **SLICO** (zero-parameter SLIC superpixels,
Achanta & Susstrunk 2012, `skimage.segmentation.slic`) instead of SAM AMG.

It exists to answer one question: is a correction gain coming from SAM specifically,
or from region-level label aggregation in general? Stages 2 (majority-vote consensus)
and 3 (per-class eligibility rule) are byte-identical between the two tools — both
call the same `apply_region_correction` function — only stage 1 (region generation)
differs, so any difference in outcome between a SAM-corrected and a SLICO-corrected
dataset is attributable to the region generator alone.

---

## How it works

1. For each tile in the coreset CSV, the tool loads the original GeoTIFF mask and optional
   auxiliary LULC rasters (e.g. MapBiomas, ESRI, Dynamic World).
2. SLICO partitions the RGB image chip (read from an MBTiles file) into superpixels —
   a dense, non-overlapping label map covering every pixel.
3. For each superpixel that contains **at least one pixel from the target class set**, a
   majority vote is computed across all sources (original mask + LULC rasters).
4. The winning class label is written to every pixel in the superpixel.
5. Pixels belonging to **non-target classes are never modified**.

Unlike SAM AMG masks (which can overlap and are resolved by processing order), SLICO
superpixels form a strict partition, so voting is order-independent and vectorized —
no per-region loop.

An **NPZ cache** can be configured to store SLICO label maps on disk, avoiding
redundant SLIC computation across multiple experiments with different target class sets.

---

## Granularity: a density anchor, not an absolute count

`n_segments` (default **1000**) is the target superpixel count **per 256×256 patch**
(~65 px/segment — a deliberately fine oversegmentation). It is scaled proportionally
to whatever chunk area is actually processed (`chunk_size`, default 1024×1024 → 16×
the reference patch area → ~16000 segments/chunk at the default), so per-pixel
superpixel density stays constant regardless of `chunk_size`.

This granularity is fixed and independent of SAM on purpose: matching `n_segments` to
however many masks SAM happened to produce for the same patch would couple the two
region generators at runtime (SLICO would depend on SAM having already run and cached
that patch), which defeats using SLICO as an independent baseline. The optional
`match_sam_cache_dir` (below) exists only for a secondary sensitivity analysis, not
the primary comparison.

---

## Installation

`scikit-image` is a base dependency of the framework — no separate install needed.

---

## Imagery source: four ways to point at it

`mbtiles_path` and each entry of `lulc_paths` accept a single file path, a directory (matched by
spatial bounds or by basename), or a CSV manifest — same mechanism as
[SAM Label Correction](sam-label-correction.md#imagery-source-four-ways-to-point-at-it), which has
the full reference. Quick example (directory matched by basename):

```yaml
mbtiles_path:
  directory: /data/imagery/per_tile
  match_by: basename
```

---

## Parallelism

SLICO is CPU-bound and thread-safe at **tile** granularity: set `n_workers` above 1 to process
tiles concurrently via a thread pool.

```yaml
n_workers: 8
```

`skimage.segmentation.slic`'s inner loop is Cython and releases the GIL, as does rasterio I/O, so
this gives real wall-clock parallelism, not just concurrency. Each tile writes to its own output
file — no shared-file write contention, no locking needed. Tiles are ordered
largest-estimated-chunk-count first before dispatch, so a handful of big tiles don't leave workers
idle at the end of a run. Verified to produce byte-identical output to the sequential (`n_workers: 1`)
path (`tests/test_slico_label_corrector_integration.py`).

Default is `1` (sequential) — safe to raise up to your CPU core count; each worker's peak memory is
roughly one chunk's imagery + label-map arrays, so very large `chunk_size` × high `n_workers`
combinations can add up.

---

## CLI usage

```bash
pytorch-smt-tools build-slico-corrected-masks path/to/config.yaml
```

---

## Configuration

```yaml
# Required
coreset_csv: /data/coreset.csv        # CSV with mask_path, row_off, col_off, patch_size columns
masks_dir: /data/masks                # Directory containing original GeoTIFF masks
mbtiles_path: /data/imagery/tiles.mbtiles

# One or more output targets (different class sets)
targets:
  - classes: [3, 5]                   # grassland + cropland
    output_dir: /data/masks_slico_gc
  - classes: [1, 3, 5]               # forest + grassland + cropland
    output_dir: /data/masks_slico_gcf

# Optional auxiliary LULC rasters included in the majority vote
lulc_paths:
  - /data/lulc/mapbiomas.vrt
  - /data/lulc/esri.vrt
  - /data/lulc/dynamic_world.vrt

include_base_mask: true   # include the mask being corrected as one vote source (default: true)

# Processing
num_classes: 6
nodata_val: 255
chunk_size: 1024          # tile processing chunk size in pixels

# SLICO granularity — target superpixels per 256x256 patch (density anchor)
n_segments: 1000

# Optional: match an existing SAM cache's per-chunk mask count instead of the
# density default (secondary sensitivity analysis, not the primary comparison)
match_sam_cache_dir: ""

# NPZ label-map cache (set to "" to disable)
cache_dir: /data/slico_cache

# Thread-pool parallelism (tile granularity, default 1 = sequential)
n_workers: 1

# Multi-process/multi-machine splits (process tiles [start_idx, end_idx) on each worker)
start_idx: 0
end_idx: 999999
```

---

## Multi-worker parallelism

SLIC runs on CPU, so parallelism here means multiple CPU workers rather than
multiple GPUs. Run multiple processes with non-overlapping `start_idx`/`end_idx`
slices and point each to a shared `cache_dir`.

```bash
pytorch-smt-tools build-slico-corrected-masks config_worker0.yaml &
pytorch-smt-tools build-slico-corrected-masks config_worker1.yaml &
```

---

## Output

For each target, a copy of every tile is written to `output_dir` with SLICO-corrected
pixels. The original files in `masks_dir` are never modified. The `run()` method
returns the same summary dict shape as `SamLabelCorrector.run()` (see
[SAM Label Correction — Output](sam-label-correction.md#output)).

---

## Python API

```python
from pytorch_segmentation_models_trainer.tools.slico_correction import (
    SLICOLabelCorrectionConfig,
    SlicoLabelCorrector,
)

config = SLICOLabelCorrectionConfig(
    coreset_csv="/data/coreset.csv",
    masks_dir="/data/masks",
    targets=[{"classes": [3, 5], "output_dir": "/data/out"}],
    mbtiles_path="/data/tiles.mbtiles",
    lulc_paths=["/data/lulc/mapbiomas.vrt"],
    cache_dir="/data/slico_cache",
)
stats = SlicoLabelCorrector(config).run()
print(f"Processed {stats['n_tiles']} tiles in {stats['elapsed_s']}s")
```

The shared correction core is also available directly, independent of SAM or SLICO:

```python
import numpy as np
from pytorch_segmentation_models_trainer.tools.region_correction import apply_region_correction

# label_map: dense int array (H, W), e.g. skimage.segmentation.slic output
corrected = apply_region_correction(
    base_mask=original_mask,        # (H, W) uint8 — the mask being corrected
    segments=label_map,             # dense label map OR a SAM-style mask-dict list
    lulc_maps=[lulc_array],         # list of (H, W) uint8 arrays
    classes_to_correct=frozenset([3, 5]),
    num_classes=6,
    include_base_mask=True,
)
```
