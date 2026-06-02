---
sidebar_position: 7
title: MBTiles Mask Dataset
---

# MBTiles Mask Dataset

`MBTilesMaskWindowedDataset` trains segmentation models from MBTiles imagery and
GeoTIFF masks. The mask is the reference grid: each sampled mask window defines
the output CRS, transform, resolution, and shape. Source imagery is warped into
that grid before returning a training sample.

Use this dataset after validating alignment with
[`export-mbtiles-mask-aligned`](./export-mbtiles-mask-aligned-images.md).
If you still need to generate the masks themselves from vector data, use
[`MBTiles Multiclass Mask Builder`](./mbtiles-multiclass-mask-builder.md) first.

## When to Use

| Data layout | Recommended approach |
|---|---|
| Images already aligned GeoTIFFs | `RasterPatchDataset` |
| MBTiles imagery + GeoTIFF masks | `MBTilesMaskWindowedDataset` |
| Custom patch CSV over aligned rasters | `CSVWindowedSegmentationDataset` |

## Sample Contract

Each item returns:

```python
{
    "image": FloatTensor[C, H, W],
    "mask": LongTensor[H, W],
    "mask_path": "...",
    "row_off": 1024,
    "col_off": 2048,
}
```

Without augmentations, `uint8` and `uint16` imagery are normalized to `[0, 1]`.
When `n_classes == 2`, mask values greater than zero become foreground class
`1`. For multiclass masks, set `n_classes` to the real number of classes so
class IDs are preserved.

## Hydra Example

```yaml title="conf/examples/mbtiles_mask_windowed_segmentation.yaml"
train_dataset:
  _target_: pytorch_segmentation_models_trainer.dataset_loader.mbtiles_mask_dataset.MBTilesMaskWindowedDataset
  mbtiles_path: /data/source_imagery.mbtiles
  mask_dir: /data/train_masks
  patch_size: 512
  stride: 512
  selected_bands: [1, 2, 3]
  image_dtype: uint8
  image_resampling: bilinear
  window_index_cache: /data/cache/train_mbtiles_mask_windows.parquet
  n_classes: 2
```

## Window Index Cache

Set `window_index_cache` to avoid scanning masks and recomputing the sliding
window index on every dataset construction:

```yaml
train_dataset:
  _target_: pytorch_segmentation_models_trainer.dataset_loader.mbtiles_mask_dataset.MBTilesMaskWindowedDataset
  mbtiles_path: /data/source_imagery.mbtiles
  mask_dir: /data/train_masks
  patch_size: 512
  stride: 512
  window_index_cache: /data/cache/train_windows.parquet
```

Supported formats are `.csv` and `.parquet`. When the cache exists, the dataset
loads these columns and skips mask discovery:

```csv
mask_path,row_off,col_off,width,height
```

When the cache does not exist, the dataset scans the masks, builds the index,
and writes the cache for future runs.

## Implementation Notes

For each mask window:

1. Read the mask window directly.
2. Compute `mask_src.window_transform(window)`.
3. Open the MBTiles/source raster through rasterio.
4. Use rasterio `WarpedVRT` to read source pixels into the mask window grid.
5. Return tensors in the same shape expected by the training `Model`.

This avoids assuming that MBTiles tile row/column indexes match mask pixel
coordinates.

## Resampling

- Use `bilinear` or `cubic` for RGB imagery.
- Use `nearest` for categorical source rasters.
- If the MBTiles native zoom is much coarser than the mask resolution, output
  still aligns but image detail is lost.
