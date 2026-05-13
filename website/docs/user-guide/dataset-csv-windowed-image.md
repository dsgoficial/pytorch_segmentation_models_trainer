---
sidebar_position: 7
---

# CSV Windowed Image Dataset

`CSVWindowedImageDataset` is the image-only counterpart to `CSVWindowedSegmentationDataset`. It reads specific patches from large images based on coordinates (offsets) defined in a CSV file, without requiring masks.

## When to use

| Scenario | Recommended dataset |
|---|---|
| **Specific patches pre-selected for unsupervised learning/inference** | **`CSVWindowedImageDataset`** |
| Patches with masks pre-selected in a CSV | `CSVWindowedSegmentationDataset` |
| Standard image-only dataset from pre-cut tiles | `ImageDataset` |

This dataset is ideal for tasks like:
- Self-supervised pre-training (e.g., Contrastive Learning) where you want to sample patches from specific regions of interest.
- High-throughput inference over specific regions defined by an external tool.

## How it works

The dataset works exactly like `CSVWindowedSegmentationDataset` but skips mask loading. It uses `rasterio` windowed read to efficiently load only the pixels within the specified window.

## CSV Structure

The CSV must contain the following columns:

| Column | Description |
|---|---|
| `image` | Path to the original full-size image. |
| `row_off` | Vertical offset (line) where the patch starts. |
| `col_off` | Horizontal offset (column) where the patch starts. |
| `patch_size` | Width and height of the patch (pixels). |

Example CSV:
```csv
image,row_off,col_off,patch_size
/data/scene_a.tif,0,0,256
/data/scene_a.tif,128,128,256
/data/scene_b.tif,1024,2048,256
```

## YAML configuration

```yaml
train_dataset:
  _target_: pytorch_segmentation_models_trainer.dataset_loader.image_dataset.CSVWindowedImageDataset
  input_csv_path: /data/unlabeled_patches.csv
  image_key: image
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
    batch_size: 32
    num_workers: 4
    shuffle: true
```

## Comparison with `ImageDataset`

| Property | `CSVWindowedImageDataset` | `ImageDataset` |
|---|---|---|
| **I/O Strategy** | Windowed read (rasterio) | Full file read (PIL/rasterio) |
| **Image Source** | Large rasters | Pre-cut tiles |
| **Memory usage** | Low (independent of raster size) | High (proportional to file size) |
| **Flexibility** | High (on-the-fly crops) | Fixed tiles |
