---
sidebar_position: 1
---

# Building a Segmentation Dataset

This guide explains how to prepare data and configure the `SegmentationDataset` class for semantic segmentation training. The dataset system is built around a CSV index file that points to image and mask pairs on disk.

## CSV-Based Dataset Format

All dataset classes in this project use a CSV file as their index. Each row describes one training sample. The minimal required columns for segmentation are `image` and `mask`.

| Column  | Description                                             |
|---------|---------------------------------------------------------|
| `image` | Path to the input image (absolute or relative to `root_dir`) |
| `mask`  | Path to the corresponding segmentation mask             |

Paths may be absolute or relative. When `root_dir` is provided in the dataset config, relative paths are resolved against it.

### Example CSV

```csv
image,mask
images/tile_001.tif,polygon_masks/tile_001.png
images/tile_002.tif,polygon_masks/tile_002.png
images/tile_003.tif,polygon_masks/tile_003.png
```

### Generating a CSV with Python

```python
import os
import pandas as pd
from pathlib import Path

images_dir = Path("/data/my_dataset/images")
masks_dir  = Path("/data/my_dataset/polygon_masks")

rows = []
for img_path in sorted(images_dir.glob("*.tif")):
    mask_path = masks_dir / img_path.with_suffix(".png").name
    if mask_path.exists():
        rows.append({"image": str(img_path), "mask": str(mask_path)})

df = pd.DataFrame(rows)
df.to_csv("/data/my_dataset/train.csv", index=False)
print(f"Wrote {len(df)} rows")
```

## Supported Image Formats

The dataset supports any format readable by PIL (default) or rasterio (when `use_rasterio: true`):

- **JPG / JPEG** — standard RGB photographs
- **PNG** — lossless RGB or RGBA images
- **TIFF / GeoTIFF** — remote sensing imagery, including multispectral and hyperspectral data

:::tip
For GeoTIFF files with more than three bands or with a specific band ordering, always set `use_rasterio: true` and specify `selected_bands`.
:::

## Mask Format

Masks must be **single-channel (grayscale) PNG** files where each pixel value encodes a class:

| Pixel value | Meaning                          |
|-------------|----------------------------------|
| `0`         | Background                       |
| `1`         | Class 1                          |
| `2`         | Class 2                          |
| `…`         | Additional classes               |
| `255`       | Foreground (binary segmentation) |

When `n_classes=2` (the default), the dataset automatically binarises the mask: any non-zero value becomes `1`, and zero stays `0`. For multi-class masks, set `n_classes` to the actual number of classes and pixel values should match class indices directly.

## The `SegmentationDataset` Class

`SegmentationDataset` extends the abstract base class and handles image/mask loading, optional rasterio-based reading, band selection, and augmentation.

### Constructor Parameters

| Parameter                    | Type                   | Default  | Description                                                                 |
|------------------------------|------------------------|----------|-----------------------------------------------------------------------------|
| `input_csv_path`             | `Path`                 | required | Path to the CSV index file                                                  |
| `root_dir`                   | `str`                  | `None`   | Root directory prepended to relative paths in the CSV                       |
| `augmentation_list`          | `list`                 | `None`   | List of albumentations transforms                                           |
| `data_loader`                | config object          | `None`   | DataLoader keyword arguments                                                |
| `image_key`                  | `str`                  | `"image"` | CSV column name for the image path                                         |
| `mask_key`                   | `str`                  | `"mask"` | CSV column name for the mask path                                           |
| `n_first_rows_to_read`       | `int`                  | `None`   | Limit the number of CSV rows read (useful for quick experiments)            |
| `n_classes`                  | `int`                  | `2`      | Number of segmentation classes                                              |
| `selected_bands`             | `List[int]`            | `None`   | 1-based list of band indices to read (e.g. `[1, 2, 3]`)                    |
| `use_rasterio`               | `bool`                 | `False`  | Use rasterio instead of PIL for image loading                               |
| `image_dtype`                | `str`                  | `"uint8"` | Data type for image interpretation when using rasterio. Accepted values: `"uint8"`, `"uint16"`, `"float32"`, `"native"`. See [Image Dtype](#image-dtype) below. |
| `reset_augmentation_function`| `bool`                 | `False`  | Deep-copy the transform on each call to prevent albumentations memory leaks |

### `data_loader` Sub-Config

The `data_loader` config block is passed directly as keyword arguments to PyTorch's `DataLoader`. Supported fields:

| Field         | Type   | Default | Description                                          |
|---------------|--------|---------|------------------------------------------------------|
| `shuffle`     | `bool` | `true`  | Shuffle samples every epoch                          |
| `num_workers` | `int`  | `4`     | Number of parallel data loading workers              |
| `pin_memory`  | `bool` | `true`  | Pin memory for faster GPU transfer                   |
| `batch_size`  | `int`  | `8`     | Number of samples per batch                          |
| `drop_last`   | `bool` | `false` | Drop the last incomplete batch                       |

## Augmentation Pipeline

The project uses [albumentations](https://albumentations.ai/) for augmentation. Each transform in `augmentation_list` is instantiated via Hydra's `_target_` mechanism.

### Example Augmentation Transforms

```yaml
augmentation_list:
  - _target_: albumentations.RandomCrop
    height: 512
    width: 512
  - _target_: albumentations.HorizontalFlip
    p: 0.5
  - _target_: albumentations.VerticalFlip
    p: 0.5
  - _target_: albumentations.RandomBrightnessContrast
    p: 0.2
  - _target_: albumentations.Normalize
    mean: [0.485, 0.456, 0.406]
    std: [0.229, 0.224, 0.225]
  - _target_: albumentations.pytorch.ToTensorV2
```

:::note
`ToTensorV2` converts the image to a `(C, H, W)` float tensor and the mask to a `(H, W)` long tensor. It must be the last transform in the list.
:::

## Multispectral Support

For multispectral images (e.g. 4-band or 8-band GeoTIFFs), set `use_rasterio: true` and specify which bands to load with `selected_bands`. Band indices are **1-based**, matching rasterio's convention.

```yaml
use_rasterio: true
selected_bands: [1, 2, 3, 4]   # load the first four bands
```

If `selected_bands` is omitted while `use_rasterio` is `true`, all available bands are loaded.

:::tip
Setting `use_rasterio: true` also works for standard 3-band GeoTIFFs when you need correct handling of geospatial metadata or when PIL cannot open the format.
:::

## Image Dtype

The `image_dtype` parameter controls how the pixel values are interpreted after loading via rasterio. It only affects the rasterio code path (i.e. when `use_rasterio: true` or `selected_bands` is set).

| Value | numpy dtype | Automatic normalization (no-transform path) | Typical use case |
|-------|-------------|---------------------------------------------|------------------|
| `"uint8"` (default) | `np.uint8` | divided by `255.0` | Standard RGB / 8-bit GeoTIFF |
| `"uint16"` | `np.uint16` | divided by `65535.0` | 16-bit satellite imagery (Sentinel-2, Landsat, WorldView) |
| `"float32"` | `np.float32` | no division | Imagery already stored as normalized floats |
| `"native"` | unchanged | no division | Preserves the file's original dtype; useful for inspection |

The default `"uint8"` preserves the original behaviour, so **existing configuration files do not need any change**.

:::note Transform vs no-transform path
When an `augmentation_list` is provided, normalization is fully handled by Albumentations (e.g. `A.Normalize`, `A.ToFloat`). In that case `image_dtype` only controls the cast applied to the numpy array before it enters the transform pipeline. The automatic division described in the table above applies only when `augmentation_list` is `null`.
:::

:::warning Albumentations and uint16
Albumentations does not natively support `uint16` arrays. When using `image_dtype: uint16` with an augmentation pipeline, add `A.ToFloat` as the **first** transform to convert values to `float32` in `[0, 1]`:

```yaml
augmentation_list:
  - _target_: albumentations.ToFloat
    max_value: 65535.0
  - _target_: albumentations.RandomCrop
    height: 256
    width: 256
  - _target_: albumentations.Normalize
    mean: [0.5, 0.5, 0.5, 0.4]
    std: [0.2, 0.2, 0.2, 0.15]
    max_pixel_value: 1.0   # values are already in [0, 1] after ToFloat
  - _target_: albumentations.pytorch.ToTensorV2
```
:::

## Full YAML Configuration Example

The following example shows a complete train/val dataset configuration for a binary building segmentation task:

```yaml
# configs/dataset/train_val.yaml

train_dataset:
  _target_: pytorch_segmentation_models_trainer.dataset_loader.dataset.SegmentationDataset
  input_csv_path: /data/my_dataset/train.csv
  root_dir: /data/my_dataset
  n_classes: 2
  use_rasterio: false
  selected_bands: null
  augmentation_list:
    - _target_: albumentations.RandomCrop
      height: 512
      width: 512
    - _target_: albumentations.HorizontalFlip
      p: 0.5
    - _target_: albumentations.Normalize
      mean: [0.485, 0.456, 0.406]
      std: [0.229, 0.224, 0.225]
    - _target_: albumentations.pytorch.ToTensorV2
  data_loader:
    shuffle: true
    num_workers: 4
    pin_memory: true
    batch_size: 8
    drop_last: true

val_dataset:
  _target_: pytorch_segmentation_models_trainer.dataset_loader.dataset.SegmentationDataset
  input_csv_path: /data/my_dataset/val.csv
  root_dir: /data/my_dataset
  n_classes: 2
  use_rasterio: false
  selected_bands: null
  augmentation_list:
    - _target_: albumentations.Resize
      height: 512
      width: 512
    - _target_: albumentations.Normalize
      mean: [0.485, 0.456, 0.406]
      std: [0.229, 0.224, 0.225]
    - _target_: albumentations.pytorch.ToTensorV2
  data_loader:
    shuffle: false
    num_workers: 4
    pin_memory: true
    batch_size: 8
    drop_last: false
```

### Multispectral Example — 8-bit GeoTIFF

```yaml
train_dataset:
  _target_: pytorch_segmentation_models_trainer.dataset_loader.dataset.SegmentationDataset
  input_csv_path: /data/multispectral/train.csv
  root_dir: /data/multispectral
  n_classes: 5
  use_rasterio: true
  selected_bands: [1, 2, 3, 4]   # RGB + NIR
  image_dtype: uint8              # default; explicit for clarity
  augmentation_list:
    - _target_: albumentations.RandomCrop
      height: 256
      width: 256
    - _target_: albumentations.HorizontalFlip
      p: 0.5
    - _target_: albumentations.Normalize
      mean: [0.485, 0.456, 0.406, 0.4]
      std: [0.229, 0.224, 0.225, 0.2]
    - _target_: albumentations.pytorch.ToTensorV2
  data_loader:
    shuffle: true
    num_workers: 4
    pin_memory: true
    batch_size: 16
    drop_last: true
```

### Multispectral Example — 16-bit GeoTIFF (Sentinel-2, Landsat)

```yaml
train_dataset:
  _target_: pytorch_segmentation_models_trainer.dataset_loader.dataset.SegmentationDataset
  input_csv_path: /data/sentinel2/train.csv
  root_dir: /data/sentinel2
  n_classes: 2
  use_rasterio: true
  selected_bands: [1, 2, 3, 4]
  image_dtype: uint16             # preserves 16-bit precision
  augmentation_list:
    - _target_: albumentations.ToFloat
      max_value: 65535.0          # converts uint16 → float32 in [0, 1]
    - _target_: albumentations.RandomCrop
      height: 256
      width: 256
    - _target_: albumentations.HorizontalFlip
      p: 0.5
    - _target_: albumentations.Normalize
      mean: [0.5, 0.5, 0.5, 0.4]
      std: [0.2, 0.2, 0.2, 0.15]
      max_pixel_value: 1.0        # values are already in [0, 1] after ToFloat
    - _target_: albumentations.pytorch.ToTensorV2
  data_loader:
    shuffle: true
    num_workers: 4
    pin_memory: true
    batch_size: 8
    drop_last: true
```

## Dataset Output

`__getitem__` returns a dictionary with the following keys:

| Key     | Shape           | dtype            | Description                              |
|---------|-----------------|------------------|------------------------------------------|
| `image` | `(C, H, W)`     | `torch.float32`  | Normalised image tensor                  |
| `mask`  | `(H, W)`        | `torch.int64`    | Class-index mask                         |

When `augmentation_list` is `null`, the image is automatically converted to a `(C, H, W)` float tensor. The normalization factor depends on `image_dtype`:

| `image_dtype` | Automatic scaling |
|---------------|-------------------|
| `"uint8"` | divided by `255.0` → `[0, 1]` |
| `"uint16"` | divided by `65535.0` → `[0, 1]` |
| `"float32"` | no scaling applied |
| `"native"` | no scaling applied |

The mask is always cast to `torch.int64`.
