---
sidebar_position: 12
title: Dataset Conversion
---

# Dataset Conversion

This guide covers converting segmentation datasets into formats required by specialized models. Currently the primary supported conversion target is the **Polygon-RNN** format, which requires cropped per-object images, normalized polygon files, and a generated CSV index.

---

## Overview

Standard segmentation datasets store full images alongside polygon annotations. Polygon-RNN, however, operates on individual object crops: each training sample is a tightly cropped image patch containing a single polygon, with the polygon coordinates rescaled to fit the crop. The dataset conversion pipeline automates this transformation:

1. Reads a source `InstanceSegmentationDataset` CSV (full images + per-image JSON annotations).
2. For each annotated polygon, computes a bounding crop with a 10% margin.
3. Resizes the crop to a fixed square (`image_size x image_size` pixels).
4. Rescales and normalizes the polygon coordinates to match the resized crop.
5. Writes the cropped image as a PNG and the polygon as a JSON file.
6. Produces a new CSV index that maps each crop to its polygon, scale factors, and origin coordinates.

---

## Running the Conversion via CLI

Use the `pytorch-smt` CLI with `+mode=convert-dataset`:

```bash
pytorch-smt \
  --config-dir ./configs \
  --config-name convert_config \
  +mode=convert-dataset
```

Hydra will instantiate the `ConversionProcessor`, which in turn calls `conversion_strategy.convert(input_dataset)`.

---

## Classes

### `PolygonRNNDatasetConversionStrategy`

**Import path:**
```python
from pytorch_segmentation_models_trainer.tools.dataset_handlers.convert_dataset import PolygonRNNDatasetConversionStrategy
```

The main conversion strategy for producing Polygon-RNN datasets. Implemented as a Python `dataclass`.

#### Constructor Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `output_dir` | `str` | **required** | Root directory where all output files will be written. Created automatically if it does not exist. |
| `output_file_name` | `str` | **required** | Base name for the generated CSV index (`.csv` extension is appended automatically). Written inside `output_dir`. |
| `output_images_folder` | `str` | `"images_croped"` | Subdirectory under `output_dir` for cropped image PNG files. Created automatically. |
| `output_polygons_folder` | `str` | `"polygons_croped"` | Subdirectory under `output_dir` for polygon JSON files. Created automatically. |
| `write_output_files` | `bool` | `True` | When `False`, generates only the CSV entries without writing image or polygon files. Useful for dry-run inspection. |
| `original_images_folder_name` | `str` | `"images"` | Name of the folder segment used to reconstruct the `original_image_path` column in the output CSV. |
| `simultaneous_tasks` | `int` | `1` | Number of parallel worker processes. Values greater than `1` use a `ProcessPoolExecutor`. |
| `image_size` | `int` | `224` | Width and height in pixels for each cropped output image. All crops are resized to `image_size x image_size` using bilinear interpolation. |

#### Output CSV Columns

Each row in the generated CSV corresponds to one polygon crop:

| Column | Description |
|--------|-------------|
| `image` | Relative path to the cropped PNG, e.g. `images_croped/<stem>/<i>.png` |
| `mask` | Relative path to the normalized polygon JSON, e.g. `polygons_croped/<stem>/<i>.json` |
| `scale_h` | Vertical scale factor applied to polygon coordinates |
| `scale_w` | Horizontal scale factor applied to polygon coordinates |
| `min_col` | Left boundary (column) of the crop in the original image |
| `min_row` | Top boundary (row) of the crop in the original image |
| `original_image_path` | Relative path to the source full image |
| `original_polygon_wkt` | WKT representation of the original (unscaled) polygon |

---

### `ConversionProcessor`

**Import path:**
```python
from pytorch_segmentation_models_trainer.tools.dataset_handlers.convert_dataset import ConversionProcessor
```

A thin orchestrator that connects a source dataset to a conversion strategy. Implemented as a Python `dataclass`.

#### Constructor Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `input_dataset` | `AbstractDataset` | **required** | The source dataset to convert. Must be an `InstanceSegmentationDataset` when using `PolygonRNNDatasetConversionStrategy`. |
| `conversion_strategy` | `AbstractConversionStrategy` | **required** | The strategy object that performs the conversion. |

#### Usage

```python
processor = ConversionProcessor(
    input_dataset=my_instance_seg_dataset,
    conversion_strategy=my_strategy,
)
processor.process()
```

Calling `process()` delegates to `conversion_strategy.convert(input_dataset)`.

---

## Input Dataset Requirements

The source dataset must be an `InstanceSegmentationDataset` (see [Dataset Classes](../api/dataset.md)). Its CSV must contain at minimum:

- An image path column (default key: `image`)
- A keypoint/polygon annotation path column (default key: `keypoints`) pointing to JSON files with the structure:

```json
{
  "imgHeight": 512,
  "imgWidth": 512,
  "objects": [
    {
      "polygon": [[x1, y1], [x2, y2], ...]
    }
  ]
}
```

---

## Example YAML Configuration

Place this file at `configs/convert_config.yaml`:

```yaml
# @package _global_

defaults:
  - _self_

mode: convert-dataset

input_dataset:
  _target_: pytorch_segmentation_models_trainer.dataset_loader.dataset.InstanceSegmentationDataset
  input_csv_path: /data/my_dataset/train.csv
  root_dir: /data/my_dataset
  image_key: image
  keypoint_key: keypoints
  return_mask: false
  return_keypoints: true

conversion_strategy:
  _target_: pytorch_segmentation_models_trainer.tools.dataset_handlers.convert_dataset.PolygonRNNDatasetConversionStrategy
  output_dir: /data/polygonrnn_dataset
  output_file_name: polygonrnn_train
  output_images_folder: images_croped
  output_polygons_folder: polygons_croped
  write_output_files: true
  original_images_folder_name: images
  simultaneous_tasks: 4
  image_size: 224
```

Run with:

```bash
pytorch-smt \
  --config-dir ./configs \
  --config-name convert_config \
  +mode=convert-dataset
```

---

## Output Directory Layout

After a successful conversion run the output directory will contain:

```
/data/polygonrnn_dataset/
├── polygonrnn_train.csv          # Generated index CSV
├── images_croped/
│   ├── image_stem_001/
│   │   ├── 0.png
│   │   ├── 1.png
│   │   └── ...
│   └── image_stem_002/
│       └── ...
└── polygons_croped/
    ├── image_stem_001/
    │   ├── 0.json
    │   ├── 1.json
    │   └── ...
    └── image_stem_002/
        └── ...
```

Each polygon JSON file contains a single key `"polygon"` with coordinates normalized to the `[0, 223]` range of the resized crop:

```json
{
  "polygon": [[x1, y1], [x2, y2], ...]
}
```

---

## Notes

- Polygons whose bounding box has zero height or zero width are skipped silently.
- The crop bounding box is expanded by 10% in each direction before clamping to the image boundary.
- When `simultaneous_tasks > 1` a `ProcessPoolExecutor` is used; set it to match the number of available CPU cores for best throughput.
- The generated CSV is suitable for direct use as the `input_csv_path` of `PolygonRNNDataset` during training.
