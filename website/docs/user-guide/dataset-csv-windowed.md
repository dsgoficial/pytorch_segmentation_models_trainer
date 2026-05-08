---
sidebar_position: 6
---

# CSV Windowed Dataset

`CSVWindowedSegmentationDataset` provides a way to read specific patches from large GeoTIFFs based on coordinates (offsets) defined in a CSV file. It uses `rasterio` windowed read to load only the required pixels, making it extremely memory-efficient for large images.

## When to use

| Scenario | Recommended dataset |
|---|---|
| **Specific patches pre-selected and listed in a CSV** | **`CSVWindowedSegmentationDataset`** |
| Systematic coverage (sliding window) | `RasterPatchDataset` |
| Large GeoTIFFs, random crops with class-based filtering | `RandomCropSegmentationDataset` |
| Pre-tiled images (already cut on disk) listed in a CSV | `SegmentationDataset` |

Use `CSVWindowedSegmentationDataset` when you have a custom sampling strategy (e.g., stratified sampling, focused on rare objects) and you have stored the patch coordinates in a CSV instead of cutting the tiles to disk.

## How it works

The dataset reads the input CSV and, for each row, identifies the image path, mask path, and the window coordinates (`row_off`, `col_off`, `patch_size`).

During `__getitem__(idx)`, it uses `rasterio.windows.Window` to perform a windowed read:

1.  Open the image/mask with `rasterio`.
2.  Define the window: `Window(col_off, row_off, patch_size, patch_size)`.
3.  Read only that window.
4.  Apply augmentations and return tensors.

The full image is never loaded into RAM.

## CSV Structure

The CSV must contain at least the following columns (names are configurable):

| Column | Description |
|---|---|
| `image` | Path to the original full-size image. |
| `mask` | Path to the corresponding mask. |
| `row_off` | Vertical offset (line) where the patch starts. |
| `col_off` | Horizontal offset (column) where the patch starts. |
| `patch_size` | Width and height of the patch (pixels). |

Example CSV:
```csv
image,mask,row_off,col_off,patch_size
/data/img1.tif,/data/mask1.tif,0,0,256
/data/img1.tif,/data/mask1.tif,100,500,256
/data/img2.tif,/data/mask2.tif,2048,1024,256
```

## Quick-start Python

```python
from pytorch_segmentation_models_trainer.dataset_loader.dataset import (
    CSVWindowedSegmentationDataset,
)

ds = CSVWindowedSegmentationDataset(
    input_csv_path="patches.csv",
    image_key="image",
    mask_key="mask",
    row_off_key="row_off",
    col_off_key="col_off",
    patch_size_key="patch_size"
)

print(f"Total patches: {len(ds)}")

item = ds[0]
image = item["image"] # (C, 256, 256)
mask = item["mask"]   # (256, 256)
```

## YAML configuration

```yaml
train_dataset:
  _target_: pytorch_segmentation_models_trainer.dataset_loader.dataset.CSVWindowedSegmentationDataset
  input_csv_path: /data/train_patches.csv
  image_key: image
  mask_key: mask
  row_off_key: row_off
  col_off_key: col_off
  patch_size_key: patch_size
  image_dtype: uint8
  augmentation_list:
    - _target_: albumentations.HorizontalFlip
      p: 0.5
    - _target_: albumentations.Normalize
      mean: [0.485, 0.456, 0.406]
      std: [0.229, 0.224, 0.225]
    - _target_: albumentations.pytorch.ToTensorV2
  data_loader:
    batch_size: 16
    num_workers: 8
    shuffle: true
```

A ready-to-run full example is available at `conf/examples/csv_windowed_segmentation.yaml`.

## Constructor parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `input_csv_path` | `Path \| str` | `None` | Path to the CSV file. |
| `df` | `pd.DataFrame` | `None` | Pre-built DataFrame (alternative to `input_csv_path`). |
| `image_key` | `str` | `"image"` | CSV column for image paths. |
| `mask_key` | `str` | `"mask"` | CSV column for mask paths. |
| `row_off_key` | `str` | `"row_off"` | CSV column for vertical offset. |
| `col_off_key` | `str` | `"col_off"` | CSV column for horizontal offset. |
| `patch_size_key` | `str` | `"patch_size"` | CSV column for patch size. |
| `n_classes` | `int` | `2` | Number of classes. If 2, mask is binarized (>0 -> 1). |
| `selected_bands` | `List[int] \| None` | `None` | 1-based band indices to load. |
| `use_rasterio` | `bool` | `True` | Must be `True` for windowed read. |
| `image_dtype` | `str` | `"uint8"` | Cast dtype after reading. |

## Comparison with other datasets

| Property | `CSVWindowed` | `RasterPatch` | `Segmentation` |
|---|---|---|---|
| **Source** | CSV coordinates | Sliding window | Pre-cut tiles |
| **I/O** | Windowed read | Windowed read | Full file read |
| **Flexibility** | High (any patch) | Fixed grid | Fixed tiles |
| **Disk Space** | Minimal | Minimal | High (tiles) |
