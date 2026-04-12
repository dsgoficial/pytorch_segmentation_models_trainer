---
sidebar_position: 2
---

# Sliding-Window Patch Dataset

`RasterPatchDataset` provides **systematic, deterministic sliding-window training** directly from full-size raster images (GeoTIFF, etc.) — no pre-generated tiles on disk required.

## When to use

| Scenario | Recommended dataset |
|---|---|
| Large GeoTIFFs, systematic coverage, reproducible splits | **`RasterPatchDataset`** |
| Large GeoTIFFs, random crops with class-based filtering | `RandomCropSegmentationDataset` |
| Pre-tiled images listed in a CSV file | `SegmentationDataset` |
| Pre-tiled images discovered from folders, no CSV | `SegmentationDatasetFromFolder` |

Use `RasterPatchDataset` when you want every pixel of every image to appear in training exactly (or proportionally) the same number of times, controlled entirely by `patch_size` and `stride`.

## How it works

The dataset scans `image_dir` and `mask_dir` recursively for files matching `extension`. Images and masks are matched by **relative path** — a file at `image_dir/area_a/scene_001.tif` is paired with `mask_dir/area_a/scene_001.tif`.

For each valid pair the dataset computes how many `patch_size × patch_size` windows fit with the given `stride`:

```
patches_per_row = (width  - patch_size) // stride + 1
patches_per_col = (height - patch_size) // stride + 1
patches_per_image = patches_per_row × patches_per_col
```

The global `__len__` is the **sum of patches across all images**, not the number of images. A `DataLoader` with `shuffle=True` will draw patches from different images and locations in each batch.

### Index mapping

`__getitem__(idx)` maps the global index to the right image and window position in O(log N) using binary search over a cumulative-count list — no iteration, no pre-loading of images.

```
Given idx:
  1. bisect → find which image
  2. local_idx = idx - cumulative[img_idx]
  3. grid_row  = local_idx // patches_per_row
     grid_col  = local_idx % patches_per_row
  4. rasterio.Window(col_off, row_off, patch_size, patch_size)
  5. read only that window — the full image never enters RAM
```

## Directory structure

```
images_root/
    area_a/
        scene_001.tif
        scene_002.tif
    area_b/
        scene_003.tif

masks_root/
    area_a/
        scene_001.tif   ← matched by relative path
        scene_002.tif
    area_b/
        scene_003.tif
```

Subdirectories are matched automatically. The mask extension may differ from the image extension via `mask_extension`.

## Patch count example

A **1024 × 1024** image with `patch_size=256` and `stride=128` (50 % overlap):

```
patches_per_row = (1024 - 256) // 128 + 1 = 7
patches_per_col = (1024 - 256) // 128 + 1 = 7
patches_per_image = 7 × 7 = 49
```

Setting `stride = patch_size` removes overlap and gives the minimum patch count:

```
patches_per_row = (1024 - 256) // 256 + 1 = 4
patches_per_image = 4 × 4 = 16
```

## Quick-start Python

```python
from pytorch_segmentation_models_trainer.dataset_loader.raster_patch_dataset import (
    RasterPatchDataset,
)
from torch.utils.data import DataLoader

ds = RasterPatchDataset(
    image_dir="/data/images",
    mask_dir="/data/masks",
    extension=".tif",
    patch_size=256,
    stride=128,      # 50 % overlap
)

print(f"Images:  {len(ds.image_info)}")
print(f"Patches: {len(ds)}")

loader = DataLoader(ds, batch_size=16, shuffle=True, num_workers=4, pin_memory=True)
for batch in loader:
    imgs  = batch["image"]   # (16, C, 256, 256)  float32
    masks = batch["mask"]    # (16, 256, 256)      int64
    break
```

## YAML configuration

```yaml
train_dataset:
  _target_: pytorch_segmentation_models_trainer.dataset_loader.raster_patch_dataset.RasterPatchDataset
  image_dir: /data/train/images
  mask_dir: /data/train/masks
  extension: .tif
  patch_size: 256
  stride: 128
  image_dtype: uint8      # uint8 | uint16 | float32 | native
  augmentation_list:
    - _target_: albumentations.HorizontalFlip
      p: 0.5
    - _target_: albumentations.Normalize
      mean: [0.485, 0.456, 0.406]
      std:  [0.229, 0.224, 0.225]
    - _target_: albumentations.pytorch.ToTensorV2
  data_loader:
    batch_size: 16
    num_workers: 8
    shuffle: true
    pin_memory: true
    drop_last: true

val_dataset:
  _target_: pytorch_segmentation_models_trainer.dataset_loader.raster_patch_dataset.RasterPatchDataset
  image_dir: /data/val/images
  mask_dir: /data/val/masks
  extension: .tif
  patch_size: 256
  stride: 256             # no overlap in validation → each pixel seen once
  augmentation_list:
    - _target_: albumentations.Normalize
      mean: [0.485, 0.456, 0.406]
      std:  [0.229, 0.224, 0.225]
    - _target_: albumentations.pytorch.ToTensorV2
  data_loader:
    batch_size: 16
    num_workers: 8
    shuffle: false
    pin_memory: true
    drop_last: false
```

A ready-to-run full example is available at `conf/examples/raster_patch_segmentation.yaml`.

## Constructor parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `image_dir` | `str \| Path` | **required** | Root directory of input images. |
| `mask_dir` | `str \| Path` | **required** | Root directory of segmentation masks. |
| `extension` | `str` | `".tif"` | Image file extension. Leading dot optional. |
| `patch_size` | `int` | `256` | Patch width and height in pixels. |
| `stride` | `int` | `128` | Step between patch origins. `stride < patch_size` → overlap. |
| `mask_extension` | `str \| None` | `None` | Mask extension. `None` uses the same as `extension`. |
| `augmentation_list` | `list \| A.Compose \| None` | `None` | Albumentations transforms. `None` → no augmentation. |
| `data_loader` | `DataLoaderConfig \| None` | `None` | Stored as `ds.data_loader`; consumed by the Lightning `Model`. |
| `selected_bands` | `List[int] \| None` | `None` | 1-based band indices to load. `None` → all bands. |
| `image_dtype` | `str` | `"uint8"` | Cast dtype after reading. See table below. |

### `image_dtype` values

| Value | Behaviour without transform |
|---|---|
| `"uint8"` | Cast to uint8, normalised ÷ 255 → `[0, 1]` |
| `"uint16"` | Cast to uint16, normalised ÷ 65535 → `[0, 1]` |
| `"float32"` | Cast to float32, no normalisation |
| `"native"` | No cast, no normalisation — original file dtype preserved |

When `augmentation_list` is provided, normalisation is the responsibility of the Albumentations pipeline (e.g. `A.Normalize` or `A.ToFloat`). `image_dtype` only affects the array cast before augmentation.

## Output format

`__getitem__` returns a `dict`:

| Key | dtype | shape | Notes |
|---|---|---|---|
| `"image"` | `torch.float32` | `(C, patch_size, patch_size)` | Normalised when no transform and `image_dtype` is `uint8`/`uint16` |
| `"mask"` | `torch.int64` | `(patch_size, patch_size)` | Raw pixel values from the mask file |

This format is compatible with all callbacks, metrics, and loss functions in the framework.

## Multispectral images

Use `selected_bands` to load a subset of bands (1-based, rasterio convention):

```yaml
train_dataset:
  _target_: pytorch_segmentation_models_trainer.dataset_loader.raster_patch_dataset.RasterPatchDataset
  image_dir: /data/sentinel2/images
  mask_dir: /data/sentinel2/masks
  extension: .tif
  patch_size: 256
  stride: 128
  selected_bands: [2, 3, 4]   # RGB bands from a 13-band Sentinel-2 image
  image_dtype: uint16
```

Update `in_channels` in the model config accordingly:

```yaml
model:
  _target_: segmentation_models_pytorch.Unet
  in_channels: 3   # must match len(selected_bands)
```

## Images smaller than `patch_size`

Images whose width or height is smaller than `patch_size` are **silently skipped** with a `UserWarning`. Check the warnings in your training log if the patch count seems lower than expected.

## Differences from `RandomCropSegmentationDataset`

| Property | `RasterPatchDataset` | `RandomCropSegmentationDataset` |
|---|---|---|
| Coverage | Deterministic, exhaustive | Random, stochastic |
| `__len__` | Total patches (fixed) | `samples_per_epoch` (configurable) |
| Requires CSV | No | Yes |
| Patch selection | Grid-based with `stride` | Weighted random by image area |
| Valid-pixel filtering | No | Yes (`min_valid_ratio`) |
| Typical use | Train + val systematic splits | Training with class-balanced sampling |
