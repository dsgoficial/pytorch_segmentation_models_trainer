---
sidebar_position: 7
title: MBTiles Crops Dataset
---

# MBTiles Crops Dataset

`MBTilesCropsGeoTifMaskDataset` trains segmentation models from **pre-selected crop windows** paired with a spatially-aligned mask from any rasterio-readable source.

Use this dataset when your training windows are already defined — either as pixel-space offsets in a CSV/Parquet file, or as geographic features in a vector file — and your masks are uncropped GeoTIFFs, a VRT mosaicking multiple files, or another MBTile.

## Key properties

| Property | Behaviour |
|---|---|
| Window source | CSV/Parquet (pixel offsets) **or** vector file (bbox-snap) |
| Image source | Any rasterio-readable raster (MBTile, GeoTIFF, VRT, …) |
| Mask source | Any rasterio-readable raster (VRT, GeoTIFF, MBTile, …) |
| Patch size | Always fixed (`patch_size × patch_size`) |
| CRS mismatch | Resolved automatically via `WarpedVRT` |
| Resolution mismatch | Resolved automatically via `WarpedVRT` |
| Mask resampling | Always **nearest-neighbour** (class integrity) |
| Image resampling | Configurable (default `bilinear`) |

## Window sources

### CSV / Parquet

The file must contain at least two columns: `row_off` and `col_off` (pixel-space offsets relative to the image raster). Column names are configurable via `row_off_key` and `col_off_key`.

```csv
row_off,col_off
0,0
0,256
256,0
256,256
```

### Vector file (GeoPackage, Shapefile, GeoJSON, …)

Each feature defines one training window. The feature bounding-box centre is projected to image pixel space and a `patch_size × patch_size` window is centred on that point (bbox-snap), then clamped to the raster bounds. Features with empty geometries are skipped.

## Mask types

### Multi-band (RGB/RGBA color-coded)

Provide a `color_map` mapping `[R, G, B, class_idx]` entries. Pixels whose colour is not in the map are assigned class `0` (background). **Required** when the mask has more than one band — instantiation raises `ValueError` otherwise.

### Single-band (integer class indices)

Omit `color_map`. Pixel values are used directly as class indices. Set `n_classes: 2` to binarise all non-zero values to foreground class `1`.

## Using a VRT to mosaic multiple GeoTIFFs

When your masks span multiple GeoTIFF files, build a VRT with GDAL and pass it as `mask_path`:

```bash
gdalbuildvrt /data/masks/mask.vrt /data/masks/*.tif
```

The `WarpedVRT` inside the dataset handles CRS and resolution differences automatically.

## Configuration reference

| Parameter | Type | Default | Description |
|---|---|---|---|
| `image_mbtiles_path` | str | required | Path to the image raster (reference grid) |
| `mask_path` | str | required | Path to the mask raster (VRT / GeoTIFF / MBTile) |
| `crops_path` | str | required | Path to CSV/Parquet or vector file |
| `patch_size` | int | required | Fixed patch height and width in pixels |
| `color_map` | list | `null` | `[[R,G,B,class], …]` — required for multi-band masks |
| `n_classes` | int | `2` | Classes for single-band masks; `2` binarises non-zero values |
| `selected_bands` | list | `null` | 1-based image band indices (`null` = all bands) |
| `image_dtype` | str | `"uint8"` | `"uint8"` \| `"uint16"` \| `"float32"` \| `"native"` |
| `image_resampling` | str | `"bilinear"` | Resampling for image warping |
| `crops_layer` | str | `null` | Layer name for multi-layer vector files (GPKG) |
| `col_off_key` | str | `"col_off"` | Column name for horizontal pixel offset |
| `row_off_key` | str | `"row_off"` | Column name for vertical pixel offset |
| `augmentation_list` | list | `[]` | Albumentations transforms |
| `data_loader` | dict | — | DataLoader config (shuffle, num_workers, …) |
| `return_metadata` | bool | `false` | Add `row_off`/`col_off` to each sample |
| `window_index_cache` | str | `null` | `.csv` or `.parquet` path to persist window index |

## Example: CSV windows + VRT mask

```yaml
train_dataset:
  _target_: pytorch_segmentation_models_trainer.dataset_loader.mbtiles_crops_dataset.MBTilesCropsGeoTifMaskDataset
  image_mbtiles_path: /data/imagery.mbtiles
  mask_path: /data/masks/mask.vrt
  crops_path: /data/crops/windows.csv
  patch_size: 256
  color_map:
    - [255, 0, 0, 1]
    - [0, 255, 0, 2]
    - [0, 0, 255, 3]
  augmentation_list:
    - _target_: albumentations.HorizontalFlip
      p: 0.5
    - _target_: albumentations.Normalize
      mean: [0.485, 0.456, 0.406]
      std: [0.229, 0.224, 0.225]
    - _target_: albumentations.pytorch.ToTensorV2
  data_loader:
    shuffle: true
    num_workers: 8
```

## Example: vector windows + single-band GeoTIFF mask

```yaml
train_dataset:
  _target_: pytorch_segmentation_models_trainer.dataset_loader.mbtiles_crops_dataset.MBTilesCropsGeoTifMaskDataset
  image_mbtiles_path: /data/imagery.mbtiles
  mask_path: /data/masks/mask.tif
  crops_path: /data/crops/regions.gpkg
  crops_layer: training_windows
  patch_size: 256
  n_classes: 2
  augmentation_list:
    - _target_: albumentations.HorizontalFlip
      p: 0.5
    - _target_: albumentations.pytorch.ToTensorV2
  data_loader:
    shuffle: true
    num_workers: 8
```

## Output format

Each `__getitem__` returns a dict:

| Key | Shape | dtype | Condition |
|---|---|---|---|
| `"image"` | `(C, H, W)` | `float32` | always |
| `"mask"` | `(H, W)` | `int64` | always |
| `"metadata"` | dict | — | only when `return_metadata: true` |

`metadata` contains `row_off` and `col_off` (int) in image pixel coordinates.
