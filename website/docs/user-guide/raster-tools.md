---
sidebar_position: 11
title: "Raster Utilities"
---

# Raster Utilities

Utility tools for preprocessing raster files.  Accessible via the `pytorch-smt-tools` CLI.

## Overview

| Tool | CLI command | Description |
|------|-------------|-------------|
| JSON Raster Remapper | `remap-raster-from-json` | Remap pixel values using a DsgTools-compatible JSON file (single file or folder) |
| Mask Class Remapper | `remap-mask-classes` | Remap pixel class values across all TIFFs in a directory tree (inline dict) |
| VRT to GeoTIFF converter | `convert-to-tiff` | Convert VRT (or any rasterio-readable file) to compressed GeoTIFF |

---

## JSON Raster Remapper (`remap-raster-from-json`)

Remaps pixel values using a **DsgTools-compatible JSON mapping file**.  Key
behaviours:

* **Unmapped values become nodata** — only pixels explicitly listed in
  `mapping` survive; everything else becomes `nodata_value`.
* **Concurrent windowed I/O** — reads are parallelised across rasterio windows
  (one file handle per thread); writes are serialised with a thread lock.
* **Automatic dtype inference** — the smallest integer dtype that can represent
  all target values and `nodata_value` is chosen for the output.
* Supports both single-file and full folder (recursive) processing.

### JSON format

```json
{
  "description": "MapBiomas collection 9 -> EDGV 3.0",
  "nodata_value": 255,
  "mapping": {
    "3":  1,
    "15": 2,
    "33": 3
  }
}
```

| Field | Required | Default | Description |
|-------|----------|---------|-------------|
| `mapping` | yes | — | `"source": target` integer pairs |
| `nodata_value` | no | `255` | Value for unmapped / nodata pixels |
| `description` | no | `""` | Free-text annotation |

### CLI usage — single file

```bash
pytorch-smt-tools remap-raster-from-json \
  --input  /data/mapbiomas.tif \
  --output /data/edgv.tif \
  --mapping-json /data/mapbiomas_to_edgv.json
```

### CLI usage — folder

```bash
pytorch-smt-tools remap-raster-from-json \
  --input-dir  /data/masks \
  --output-dir /data/masks_remapped \
  --mapping-json /data/mapbiomas_to_edgv.json \
  --n-workers 8
```

### Options

| Option | Default | Description |
|--------|---------|-------------|
| `--input` | — | [Single-file] Source raster path |
| `--output` | — | [Single-file] Output raster path |
| `--input-dir` | — | [Folder] Root directory of input rasters |
| `--output-dir` | — | [Folder] Output root directory (mirrors input tree) |
| `--mapping-json` | (required) | Path to the JSON mapping file |
| `--n-workers` | `4` | Number of parallel threads |
| `--compress` | `lzw` | GeoTIFF compression: `lzw`, `deflate`, `zstd`, `none` |
| `--no-progress` | flag | Suppress tqdm progress bar (folder mode) |
| `--build-vrt` | flag | [Folder] Build a GDAL VRT mosaic after remapping |
| `--vrt-path` | `<output-dir>/mosaic.vrt` | [Folder] Custom VRT output path |

### Config YAML example

```yaml
# conf/examples/remap_raster_from_json.yaml
input_dir: /path/to/masks
output_dir: /path/to/masks_remapped
json_path: /path/to/mapbiomas_to_edgv.json
n_workers: 8
create_vrt: true          # optional: build mosaic.vrt in output_dir
# vrt_path: /custom/mosaic.vrt  # optional: override VRT location
```

### Python API

```python
from pathlib import Path
from pytorch_segmentation_models_trainer.tools.raster.tiff_remap import (
    build_vrt,
    load_remap_json,
    remap_raster_windowed,
    remap_raster_folder_from_json,
)

# Load and validate the JSON mapping
mapping, nodata_value, description = load_remap_json(Path("mapbiomas_to_edgv.json"))

# Single file
out_path, success, err = remap_raster_windowed(
    input_path=Path("mapbiomas.tif"),
    output_path=Path("edgv.tif"),
    mapping=mapping,
    nodata_value=nodata_value,
    n_workers=4,
)

# Entire directory tree
n_success, n_errors = remap_raster_folder_from_json(
    input_dir=Path("masks/"),
    output_dir=Path("masks_remapped/"),
    json_path=Path("mapbiomas_to_edgv.json"),
    n_workers=8,
)

# Directory tree + VRT mosaic
n_success, n_errors = remap_raster_folder_from_json(
    input_dir=Path("masks/"),
    output_dir=Path("masks_remapped/"),
    json_path=Path("mapbiomas_to_edgv.json"),
    n_workers=8,
    create_vrt=True,                              # builds masks_remapped/mosaic.vrt
    vrt_path=Path("masks_remapped/mosaic.vrt"),   # optional override
)

# Standalone VRT builder (any list of co-registered rasters)
build_vrt(
    raster_paths=list(Path("masks_remapped").rglob("*.tif")),
    output_path=Path("masks_remapped/mosaic.vrt"),
    nodata_value=255,
)
```

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
