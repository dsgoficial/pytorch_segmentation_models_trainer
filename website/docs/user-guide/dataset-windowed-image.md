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
- **Optional Window Verification**: Can validate every candidate patch during initialisation and index only readable windows.
- **Window Index Cache**: Stores the verified window index in JSON so later runs skip the validation pass when inputs and config are unchanged.
- **Serialized Raster Reads**: Can serialize window reads per source GeoTIFF when multiple DataLoader workers read the same compressed raster on shared storage.
- **Corruption Support**: Apply noise, blur, or other corruptions only to the input image.
- **Synchronized Augmentations**: Standard augmentations (like normalization or flips) are applied identically to both input and target.

### Configuration

```yaml
test_dataset:
  _target_: pytorch_segmentation_models_trainer.dataset_loader.image_dataset.WindowedImageAutoencoderDataset
  image_dir: /path/to/test_images
  crop_size: [256, 256]
  stride: 256
  verify_windows: true
  window_index_cache: /path/to/cache/test_window_index.json
  serialize_rasterio_reads: true
  rasterio_lock_dir: /tmp/psmt_rasterio_locks
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
- **`verify_windows`**: (Optional, default `false`) When enabled, each candidate window is read once during dataset initialisation. Windows that fail to read or return an unexpected shape are excluded from `len(dataset)` and global indexing.
- **`window_index_cache`**: (Optional) JSON path used with `verify_windows`. The dataset saves the verified window list plus crop, stride, band selection, dtype, image paths, file sizes, and modification timestamps. If any metadata changes, the cache is rebuilt automatically.
- **`serialize_rasterio_reads`**: (Optional, default `false`) When enabled, each `rasterio` window read is protected by a per-file interprocess lock. Use this when `num_workers > 0` causes decoder errors while several workers read the same large GeoTIFF.
- **`rasterio_lock_dir`**: (Optional) Directory for lock files. Defaults to `/tmp/psmt_rasterio_locks`.

### Verifying and Caching Valid Windows

Use `verify_windows` when some rasters contain corrupt blocks, missing overviews, or other local read problems that should not appear during training or validation. The first run is slower because every candidate patch is read once. Set `window_index_cache` to persist the verified index:

```yaml
val_dataset:
  _target_: pytorch_segmentation_models_trainer.dataset_loader.image_dataset.WindowedImageAutoencoderDataset
  image_dir: /data/validation/images
  crop_size: [224, 224]
  stride: 224
  verify_windows: true
  window_index_cache: /data/validation/cache/window_index.json
  serialize_rasterio_reads: true
  rasterio_lock_dir: /tmp/psmt_rasterio_locks
```

The cache stores only window coordinates, not image pixels. Training still reads pixels on demand with the normal rasterio LRU file-handle cache.
