---
sidebar_position: 24
title: SAM Label Correction
---

# SAM Label Correction

`build-sam-corrected-masks` uses the **Segment Anything Model** (SAM, AMG mode) to correct noisy
segmentation masks produced from cartographic sources. Instead of training with hard or soft labels
derived from imperfect map databases, this tool replaces them with spatially coherent labels
obtained by majority vote *within each SAM segment*.

---

## How it works

1. For each tile in the coreset CSV, the tool loads the original GeoTIFF mask and optional auxiliary
   LULC rasters (e.g. MapBiomas, ESRI, Dynamic World).
2. SAM AMG generates segments for the RGB image chip read from an MBTiles file.
3. For each SAM segment that contains **at least one pixel from the target class set**, a majority
   vote is computed across all sources (original mask + LULC rasters).
4. The winning class label is written to every pixel in the segment.
5. Pixels belonging to **non-target classes are never modified**.
6. SAM masks are processed ascending by `(predicted_iou, area)` so high-confidence, larger segments
   overwrite smaller ones in overlapping regions.

An **NPZ cache** can be configured to store SAM segments on disk, avoiding redundant SAM inference
across multiple experiments with different target class sets.

---

## Installation

SAM is not a dependency of the framework itself. Install it separately:

```bash
pip install git+https://github.com/facebookresearch/segment-anything.git
```

Download a SAM checkpoint (ViT-B recommended for speed):

```bash
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth
```

---

## CLI usage

```bash
pytorch-smt-tools build-sam-corrected-masks path/to/config.yaml
```

---

## Configuration

```yaml
# Required
coreset_csv: /data/coreset.csv        # CSV with mask_path, row_off, col_off, patch_size columns
masks_dir: /data/masks                # Directory containing original GeoTIFF masks
sam_checkpoint: /models/sam_vit_b_01ec64.pth
mbtiles_path: /data/imagery/tiles.mbtiles

# One or more output targets (different class sets)
targets:
  - classes: [3, 5]                   # grassland + cropland
    output_dir: /data/masks_sam_gc
  - classes: [1, 3, 5]               # forest + grassland + cropland
    output_dir: /data/masks_sam_gcf
  - classes: [0, 1, 2, 3, 4, 5]     # all classes
    output_dir: /data/masks_sam_all

# Optional auxiliary LULC rasters included in the majority vote
lulc_paths:
  - /data/lulc/mapbiomas.vrt
  - /data/lulc/esri.vrt
  - /data/lulc/dynamic_world.vrt

include_bags: true        # include original mask as one vote source (default: true)

# SAM AMG parameters
sam_model_type: vit_b
device: cuda:0
points_per_side: 32
pred_iou_thresh: 0.80
stability_score_thresh: 0.90
min_mask_region_area: 200

# Processing
num_classes: 6
nodata_val: 255
chunk_size: 1024          # tile processing chunk size in pixels

# NPZ segment cache (set to "" to disable)
cache_dir: /data/sam_cache

# Multi-GPU splits (process tiles [start_idx, end_idx) on each GPU)
start_idx: 0
end_idx: 999999
```

---

## Multi-GPU parallelism

Run multiple processes with non-overlapping `start_idx`/`end_idx` slices and point each to
a shared `cache_dir`. The first process to process a chunk writes the cache; subsequent
processes with overlapping class sets load from cache instead of running SAM again.

```bash
# GPU 0 — tiles 0..9999
pytorch-smt-tools build-sam-corrected-masks config_gpu0.yaml &

# GPU 1 — tiles 10000..19999
pytorch-smt-tools build-sam-corrected-masks config_gpu1.yaml &
```

---

## Output

For each target, a copy of every tile is written to `output_dir` with SAM-corrected pixels. The
original files in `masks_dir` are never modified.

The `run()` method returns a summary dict:

```json
{
  "n_tiles": 1234,
  "elapsed_s": 5432.1,
  "tiles": [
    {
      "tile": "tile_001.tif",
      "n_chunks": 4,
      "n_skipped": 1,
      "per_target": [
        {
          "output_dir": "/data/masks_sam_gc",
          "classes": [3, 5],
          "n_target": 89432,
          "n_changed": 12340,
          "pct_changed": 13.8
        }
      ]
    }
  ]
}
```

---

## Python API

```python
from pytorch_segmentation_models_trainer.tools.sam_correction import (
    SAMLabelCorrectionConfig,
    SamLabelCorrector,
    apply_sam_correction,
)

config = SAMLabelCorrectionConfig(
    coreset_csv="/data/coreset.csv",
    masks_dir="/data/masks",
    targets=[{"classes": [3, 5], "output_dir": "/data/out"}],
    sam_checkpoint="/models/sam_vit_b_01ec64.pth",
    mbtiles_path="/data/tiles.mbtiles",
    lulc_paths=["/data/lulc/mapbiomas.vrt"],
    cache_dir="/data/sam_cache",
)
stats = SamLabelCorrector(config).run()
print(f"Processed {stats['n_tiles']} tiles in {stats['elapsed_s']}s")
```

The `apply_sam_correction` function is also available as a pure, dependency-free utility
(no SAM or rasterio needed) for unit testing or custom integration:

```python
import numpy as np
from pytorch_segmentation_models_trainer.tools.sam_correction import apply_sam_correction

corrected = apply_sam_correction(
    bags_raw=original_mask,         # (H, W) uint8
    sam_masks=sam_output,           # list of SAM mask dicts
    lulc_maps=[lulc_array],         # list of (H, W) uint8 arrays
    classes_to_correct=frozenset([3, 5]),
    num_classes=6,
    include_bags=True,
)
```
