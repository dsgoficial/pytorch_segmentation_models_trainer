---
sidebar_position: 10
title: "Dataset Builder Tools"
---

# Dataset Builder Tools

The dataset builder tools help you prepare segmentation datasets from raw rasters and vector annotations.  They are accessible via the `pytorch-smt-tools` CLI.

## Overview

| Tool | CLI command | Description |
|------|-------------|-------------|
| Band Combiner | `combine-bands` | Merge bands from N matching rasters into a single multi-band GeoTIFF |
| Tile Dataset Builder | `build-tile-dataset` | Tile rasters and rasterize vector polygons as masks |
| Sliding Window Builder | `build-sliding-window-dataset` | Crop existing image/mask pairs into fixed-size patches |
| MBTiles Multiclass Mask Builder | `build-mbtiles-multiclass-masks` | Rasterize class polygons onto frame extents using an MBTiles reference grid |

---

## Band Combiner (`combine-bands`)

Finds files with matching names across multiple source directories and combines their bands into a single GeoTIFF per group.

### CLI usage

```bash
pytorch-smt-tools combine-bands \
  --source-dir /data/rgb \
  --source-dir /data/nir \
  --output-dir /data/combined \
  --glob "**/*.tif" \
  --workers 4
```

### Options

| Option | Default | Description |
|--------|---------|-------------|
| `--source-dir` | (required) | Source directory. Repeat for each directory. |
| `--output-dir` | (required) | Output directory for combined TIFFs. |
| `--glob` | `**/*.vrt` | Glob pattern inside each source directory. |
| `--name-pattern` | None | Pattern to extract group key, e.g. `MI_{name}.tif`. |
| `--skip-alpha / --no-skip-alpha` | `--skip-alpha` | Skip the last band when a source has >3 bands. |
| `--overwrite` | False | Overwrite existing output files. |
| `--workers` | 4 | Number of worker threads. |

### Python API

```python
from pathlib import Path
from pytorch_segmentation_models_trainer.tools.dataset_builder.band_combiner import (
    find_file_groups,
    combine_sources_to_tiff,
    combine_all,
)

# Find groups common to all directories
groups = find_file_groups(
    source_dirs=[Path("rgb/"), Path("nir/")],
    glob_pattern="**/*.tif",
)

# Combine all groups
results = combine_all(
    source_dirs=[Path("rgb/"), Path("nir/")],
    output_dir=Path("combined/"),
    glob_pattern="**/*.tif",
)
```

---

## Tile Dataset Builder (`build-tile-dataset`)

Generates fixed-size tile patches from raster images and rasterizes vector polygon annotations as segmentation masks.

### CLI usage

```bash
pytorch-smt-tools build-tile-dataset conf/examples/build_tile_dataset.yaml
```

### Config YAML

```yaml
# conf/examples/build_tile_dataset.yaml
image_paths:
  - /path/to/image1.tif
  - /path/to/image2.tif
vector_path: /path/to/masks.gpkg
vector_layer: cobertura_terrestre
class_attribute: tipo
output_dir: /path/to/output/dataset
tile_width: 256
tile_height: 256
overlap_x_percent: 20
overlap_y_percent: 20
min_valid_pixel_ratio: 0.1
skip_empty_tiles: true
generate_full_size_masks: false
max_workers: 8
background_value: 255
```

Use a `background_value` that does not collide with any valid class ID. The
default `255` is a good fit for masks stored as `uint8` when classes start at
`0` or `1`.

### Output structure

```
output_dir/
  {image_stem}/
    images/       ← tile GeoTIFFs
    masks/        ← uint8 class-index masks
    mask_full.tif ← only when generate_full_size_masks=true
  dataset.csv     ← columns: image_path, label_path, rows, columns
```

### Python API

```python
from pathlib import Path
from pytorch_segmentation_models_trainer.tools.dataset_builder.tile_dataset_builder import (
    build_tile_dataset,
)

df = build_tile_dataset(
    image_paths=[Path("image1.tif"), Path("image2.tif")],
    vector_path=Path("masks.gpkg"),
    class_attribute="tipo",
    output_dir=Path("dataset/"),
    tile_width=256,
    tile_height=256,
    background_value=255,
)
print(f"{len(df)} tiles saved")
```

---

## Sliding Window Builder (`build-sliding-window-dataset`)

Crops existing image/mask pairs from a CSV into overlapping sliding-window patches.

### CLI usage

```bash
pytorch-smt-tools build-sliding-window-dataset dataset.csv \
  --output-dir /data/patches \
  --window-size 256 \
  --overlap 0.25 \
  --remap "6:4,7:5" \
  --workers 8
```

### Options

| Option | Default | Description |
|--------|---------|-------------|
| `INPUT_CSV` | (required) | CSV with `image` and `mask` columns. |
| `--output-dir` | (required) | Root output directory. |
| `--window-size` | 256 | Patch size in pixels. |
| `--overlap` | 0.0 | Overlap fraction in [0, 1). |
| `--remap` | None | Class remapping, e.g. `7:5,6:4`. |
| `--blacklist` | None | Comma-separated directory segments to skip. |
| `--workers` | 8 | Number of worker threads. |

### Config YAML example

```yaml
# conf/examples/build_sliding_window_dataset.yaml
input_csv: /path/to/dataset.csv
output_dir: /path/to/output
window_size: 256
overlap: 0.0
class_remap:
  6: 4
  7: 5
blacklist:
  - masks_old
  - backup
n_workers: 8
```

### Python API

```python
from pathlib import Path
from pytorch_segmentation_models_trainer.tools.dataset_builder.sliding_window_builder import (
    build_sliding_window_dataset,
)

df = build_sliding_window_dataset(
    input_csv=Path("dataset.csv"),
    output_dir=Path("patches/"),
    window_size=256,
    overlap=0.25,
    class_remap={6: 4, 7: 5},
)
print(f"{len(df)} patches saved")
```

### Tutorial notebook

See [Notebook Tutorials](../tutorials/notebooks.md) for an end-to-end Potsdam walkthrough that starts from image/mask folders, generates a CSV, and creates a small windowed training set.
