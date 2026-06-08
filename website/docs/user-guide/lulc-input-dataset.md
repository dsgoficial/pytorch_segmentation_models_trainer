---
sidebar_position: 23
title: LULC Input Dataset
---

# LULC Input Dataset

`LulcInputDataset` and `LulcInputWindowedDataset` feed multi-source LULC
classification rasters as extra input channels alongside the RGB image.

Each LULC raster is **one-hot encoded** and concatenated with the RGB bands,
producing a `(3 + N × C, H, W)` tensor where `N` is the number of LULC
sources and `C` is `num_classes`.

## Motivation

LULC products (MapBiomas, Dynamic World, ESRI LULC) provide complementary
land-cover signals at different resolutions and with different biases.
Rather than picking one, this dataset lets the model learn which source to
trust per-pixel by exposing all of them as extra input channels.

## Output format

| Key | Shape | dtype | Description |
|-----|-------|-------|-------------|
| `"image"` | `(3 + N*C, H, W)` | `float32` | RGB (normalised) + one-hot LULC blocks |
| `"mask"` | `(H, W)` | `int64` | Hard integer class labels (training target) |
| `"path"` | scalar | `str` | Path of the source image file |

The LULC blocks follow the `lulc_keys` order: channels `[3, 3+C)` for the
first source, `[3+C, 3+2C)` for the second, and so on.

Pixels where the LULC value equals `ignore_lulc_index` (default `255`) or
falls outside `[0, num_classes)` produce **all-zero** one-hot vectors — no
class signal, no gradient.

## CSV format

### Tile-based (`LulcInputDataset`)

Each row is one pre-cut tile:

```
image_path,mask_path,mapbiomas_path,esri_path,dw_path
/data/tiles/img_001.tif,/data/tiles/mask_001.tif,/data/mb/001.tif,/data/esri/001.tif,/data/dw/001.tif
```

### Windowed (`LulcInputWindowedDataset`)

Each row is a patch specification over full-scene rasters:

```
image_path,mask_path,mapbiomas_path,esri_path,dw_path,row_off,col_off,patch_size
/data/scenes/img.tif,/data/scenes/mask.tif,...,0,0,256
/data/scenes/img.tif,/data/scenes/mask.tif,...,256,256,256
```

The same scene file can appear in multiple rows (one row per patch).

## Configuration

### Tile-based

```yaml
train_dataset:
  _target_: pytorch_segmentation_models_trainer.dataset_loader.lulc_input_dataset.LulcInputDataset
  input_csv_path: /data/splits/train.csv
  image_key: image_path
  mask_key: mask_path
  lulc_keys:
    - mapbiomas_path
    - esri_path
    - dw_path
  num_classes: 6
  ignore_lulc_index: 255
  augmentation_list:
    - _target_: albumentations.HorizontalFlip
      p: 0.5
    - _target_: albumentations.VerticalFlip
      p: 0.5
    - _target_: albumentations.RandomRotate90
      p: 0.5
    - _target_: albumentations.Normalize
      mean: [0.485, 0.456, 0.406]
      std: [0.229, 0.224, 0.225]
    - _target_: albumentations.pytorch.ToTensorV2
  data_loader:
    batch_size: 16
    num_workers: 8
    shuffle: true
    pin_memory: true
```

### Windowed

```yaml
train_dataset:
  _target_: pytorch_segmentation_models_trainer.dataset_loader.lulc_input_dataset.LulcInputWindowedDataset
  input_csv_path: /data/splits/train_windows.csv
  image_key: image_path
  mask_key: mask_path
  lulc_keys:
    - mapbiomas_path
    - esri_path
    - dw_path
  num_classes: 6
  row_off_key: row_off
  col_off_key: col_off
  patch_size_key: patch_size
  augmentation_list:
    - _target_: albumentations.HorizontalFlip
      p: 0.5
    - _target_: albumentations.Normalize
      mean: [0.485, 0.456, 0.406]
      std: [0.229, 0.224, 0.225]
    - _target_: albumentations.pytorch.ToTensorV2
  data_loader:
    batch_size: 8
    num_workers: 8
    shuffle: true
    pin_memory: true
```

## Parameters

### Shared (`LulcInputDataset`)

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `input_csv_path` | `str` | `None` | Path to the CSV file |
| `lulc_keys` | `list[str]` | `[]` | CSV columns for LULC raster paths |
| `num_classes` | `int` | `6` | Number of LULC classes (same for all sources) |
| `ignore_lulc_index` | `int` | `255` | Value treated as "no valid class" |
| `image_key` | `str` | `"image_path"` | CSV column for RGB image paths |
| `mask_key` | `str` | `"mask_path"` | CSV column for training target mask |
| `root_dir` | `str` | `None` | Prepended to relative CSV paths |
| `augmentation_list` | `list` | `[]` | Albumentations transform list |

### Extra (`LulcInputWindowedDataset`)

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `row_off_key` | `str` | `"row_off"` | CSV column for patch top-left row offset |
| `col_off_key` | `str` | `"col_off"` | CSV column for patch top-left col offset |
| `patch_size_key` | `str` | `"patch_size"` | CSV column for patch size (square) |

## Python API

```python
from pytorch_segmentation_models_trainer.dataset_loader.lulc_input_dataset import (
    LulcInputDataset,
    LulcInputWindowedDataset,
)
import pandas as pd

# Tile-based
ds = LulcInputDataset(
    input_csv_path="/data/splits/train.csv",
    lulc_keys=["mapbiomas_path", "esri_path", "dw_path"],
    num_classes=6,
)
sample = ds[0]
print(sample["image"].shape)  # torch.Size([21, H, W]) — 3 + 3*6
print(sample["mask"].shape)   # torch.Size([H, W])

# Windowed — from a DataFrame
import pandas as pd
df = pd.DataFrame([
    {"image_path": "/data/scene.tif", "mask_path": "/data/mask.tif",
     "mapbiomas_path": "/data/mb.tif", "esri_path": "/data/esri.tif",
     "dw_path": "/data/dw.tif",
     "row_off": 0, "col_off": 0, "patch_size": 256},
])
ds_win = LulcInputWindowedDataset(df=df, lulc_keys=["mapbiomas_path", "esri_path", "dw_path"])
```

## Adding a new LULC source

Add the raster path column to the CSV and append the column name to `lulc_keys`:

```yaml
lulc_keys:
  - mapbiomas_path
  - esri_path
  - dw_path
  - new_source_path     # new source — no other changes required
```

Output channels increase by `num_classes` automatically.

## Augmentation notes

- Geometric transforms (flip, rotate, …) are applied **consistently** to the
  RGB image, the training mask, and all LULC rasters via albumentations
  `additional_targets`.
- `Normalize` in the pipeline affects **only the `image` target**; the one-hot
  LULC channels remain in `{0, 1}` after augmentation.
- `ToTensorV2` handles the image; LULC tensors are built from one-hot encoded
  numpy arrays after augmentation.
