---
sidebar_position: 11
title: Windowed Image Datasets
---

# Windowed Image Datasets

These datasets are designed to extract patches from full-size rasters using a deterministic sliding-window (grid) approach. Unlike random-crop datasets, they allow you to process the entire area of your images in a fixed grid, which is particularly useful for validation, testing, and consistent performance monitoring.

## WindowedImageDataset

`WindowedImageDataset` is a basic dataset that yields only the `image` patch and its `path`. It is useful for tasks like feature extraction, unsupervised learning, or simply extracting patches without requiring any target labels.

### Configuration

```yaml
val_dataset:
  _target_: pytorch_segmentation_models_trainer.dataset_loader.image_dataset.WindowedImageDataset
  image_dir: /path/to/images
  crop_size: [256, 256]
  stride: 256  # Non-overlapping grid
  image_dtype: "uint8"
```

## WindowedImageAutoencoderDataset

`WindowedImageAutoencoderDataset` is specialized for Autoencoder tasks. It returns both `image` (input) and `target` (reconstruction label), where both are initially identical crops from the raster. This class supports optional **corruption augmentations** that apply only to the `image` key, allowing you to train or validate Denoising Autoencoders on a fixed grid.

### Features

- **Deterministic Grid**: Guarantees that the same patches are seen in every epoch.
- **Efficient Reading**: Uses `rasterio` windowed reads to load only the required patch from disk.
- **Corruption Support**: Apply noise, blur, or other corruptions only to the input image.
- **Synchronized Augmentations**: Standard augmentations (like normalization or flips) are applied identically to both input and target.

### Configuration

```yaml
test_dataset:
  _target_: pytorch_segmentation_models_trainer.dataset_loader.image_dataset.WindowedImageAutoencoderDataset
  image_dir: /path/to/test_images
  crop_size: [256, 256]
  stride: 256
  corruption_augmentation_list:
    - _target_: albumentations.GaussNoise
      p: 1.0
  augmentation_list:
    - _target_: albumentations.Normalize
    - _target_: albumentations.pytorch.ToTensorV2
```

## Key Parameters

- **`image_dir`**: Root folder scanned recursively for rasters.
- **`crop_size`**: The size of the extracted patches as `[height, width]`.
- **`stride`**: The distance between consecutive patches. If equal to `crop_size`, it produces a non-overlapping grid. If smaller, patches will overlap.
- **`image_extensions`**: (Optional) List of extensions to include (e.g., `[.tif, .png]`).
- **`image_dtype`**: Output data type (default is `uint8`). Use `native` to keep the raster's original type.
- **`selected_bands`**: (Optional) 1-based list of raster bands to read.
