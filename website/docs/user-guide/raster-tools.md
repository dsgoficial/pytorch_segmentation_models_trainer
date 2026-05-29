---
sidebar_position: 11
title: "Raster Utilities"
---

# Raster Utilities

Utility tools for preprocessing raster files.  Accessible via the `pytorch-smt-tools` CLI.

## Overview

| Tool | CLI command | Description |
|------|-------------|-------------|
| Mask Class Remapper | `remap-mask-classes` | Remap pixel class values across all TIFFs in a directory tree |
| VRT to GeoTIFF converter | `convert-to-tiff` | Convert VRT (or any rasterio-readable file) to compressed GeoTIFF |

---

## Mask Class Remapper (`remap-mask-classes`)

Walks a directory tree and remaps pixel class values in all raster files, writing results to a mirrored output tree.

### CLI usage

```bash
pytorch-smt-tools remap-mask-classes \
  --input-dir /data/masks \
  --output-dir /data/masks_remapped \
  --mapping "8:5,6:4" \
  --workers 8
```

### Options

| Option | Default | Description |
|--------|---------|-------------|
| `--input-dir` | (required) | Directory tree of mask rasters. |
| `--output-dir` | (required) | Output directory (mirrors structure of input). |
| `--mapping` | (required) | Pixel value remapping in `old:new` format. |
| `--workers` | None (cpu_count) | Number of worker threads. |

### Config YAML example

```yaml
# conf/examples/remap_mask_classes.yaml
input_dir: /path/to/masks
output_dir: /path/to/masks_remapped
pixel_mapping:
  8: 5   # remap class 8 -> 5
  6: 4   # remap class 6 -> 4
n_workers: 8
```

### Python API

```python
from pathlib import Path
from pytorch_segmentation_models_trainer.tools.raster.tiff_remap import (
    remap_raster,
    remap_raster_folder,
)

# Single file
out_path, success, err = remap_raster(
    input_path=Path("mask.tif"),
    output_path=Path("mask_remapped.tif"),
    pixel_mapping={8: 5, 6: 4},
)

# Entire directory tree
n_success, n_errors = remap_raster_folder(
    input_dir=Path("masks/"),
    output_dir=Path("masks_remapped/"),
    pixel_mapping={8: 5, 6: 4},
)
```

---

## VRT / Raster to GeoTIFF (`convert-to-tiff`)

Converts VRT files (or any rasterio-readable raster) to compressed, tiled GeoTIFFs.

### CLI usage

```bash
pytorch-smt-tools convert-to-tiff \
  --input-dir /data/vrts \
  --output-dir /data/tiffs \
  --glob "**/*.vrt" \
  --compression LZW \
  --workers 4
```

### Options

| Option | Default | Description |
|--------|---------|-------------|
| `--input-dir` | (required) | Root directory to scan. |
| `--output-dir` | (required) | Output directory. |
| `--glob` | `**/*.vrt` | Glob pattern for files to convert. |
| `--compression` | `LZW` | Codec: `LZW`, `DEFLATE`, `JPEG`, `NONE`. |
| `--workers` | 4 | Number of worker threads. |

### Python API

```python
from pathlib import Path
from pytorch_segmentation_models_trainer.tools.raster.vrt2tif import (
    convert_to_geotiff,
    convert_folder,
)

# Single file
out_path, success, err = convert_to_geotiff(
    input_path=Path("mosaic.vrt"),
    output_path=Path("mosaic.tif"),
    compression="LZW",
)

# Entire directory
n_success, n_errors = convert_folder(
    input_dir=Path("vrts/"),
    output_dir=Path("tiffs/"),
    glob_pattern="**/*.vrt",
    compression="LZW",
)
```
