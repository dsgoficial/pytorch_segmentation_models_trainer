---
sidebar_position: 10
title: MBTiles Polygon Dataset
---

# MBTiles Polygon Dataset

`MBTilesPolygonDataset` reads paired image and mask tiles directly from
[MBTiles](https://github.com/mapbox/mbtiles-spec) files via rasterio (GDAL
MBTILES driver) and filters training patches by vector polygon regions.

## How it works

```
image.mbtiles  ─┐
                ├─ rasterio windowed read → patch pairs
mask.mbtiles   ─┘

regions.gpkg ──→ containment filter (keeps only patches FULLY inside polygons)
```

1. Both MBTiles are opened as standard rasters by rasterio at their **native
   (maximum) resolution** — no zoom parameter needed.
2. A sliding-window grid of `patch_size × patch_size` pixels is generated over
   the image raster.
3. Each candidate window's geographic bounding box is compared against the union
   of the input polygons.  Only windows **fully contained** within the polygons
   are indexed.
4. For each valid window, the mask is warped onto the exact image-window grid via
   a WarpedVRT, keeping image and mask spatially aligned regardless of their
   original projections.
5. The mask bands (RGB or RGBA) are decoded to class indices via `color_map`.

## Containment filter

A patch is included only when its geographic extent is **completely inside** the
polygon union.  Patches that merely touch or overlap a polygon boundary are
excluded, preventing partial-label noise at annotation edges.

```
polygon boundary
       │
  ✗ partial patch    ✓ contained patch
  ┌────┼──┐           ┌────────┐
  │████│  │    →      │████████│
  └────┼──┘           └────────┘
       │
```

## Mask format

Mask rasters must be **RGB or RGBA PNGs** where each pixel color encodes a class
label.  Provide a `color_map` list with `[R, G, B, class_idx]` entries.  Pixels
whose color is not listed default to class `0` (background).

```yaml
color_map:
  - [255, 0, 0, 1]   # red   → class 1
  - [0, 255, 0, 2]   # green → class 2
  - [0, 0, 255, 3]   # blue  → class 3
```

If `color_map` is omitted, the first mask band is used directly as integer class
indices (suitable for single-band masks already encoded as class numbers).

## Window index cache

On the first run, the polygon-containment pass reads all window bounds and checks
them against the region polygons, which can be slow for large rasters.  Set
`window_index_cache` to persist the result:

```yaml
window_index_cache: /data/cache/train_window_index.csv
```

On subsequent runs, the cache is loaded and the containment pass is skipped.
Delete or move the cache file whenever you change `patch_size`, `stride`, or the
region polygons.

## YAML example

```yaml
train_dataset:
  _target_: pytorch_segmentation_models_trainer.dataset_loader.mbtiles_dataset.MBTilesPolygonDataset
  image_mbtiles_path: /data/imagery.mbtiles
  mask_mbtiles_path: /data/masks.mbtiles
  regions_path: /data/regions.gpkg
  patch_size: 512
  stride: 512
  color_map:
    - [255, 0, 0, 1]
    - [0, 255, 0, 2]
    - [0, 0, 255, 3]
  mask_resampling: nearest
  window_index_cache: /data/cache/train_window_index.csv
  augmentation_list:
    - _target_: albumentations.HorizontalFlip
      p: 0.5
    - _target_: albumentations.Normalize
      mean: [0.485, 0.456, 0.406]
      std: [0.229, 0.224, 0.225]
    - _target_: albumentations.pytorch.ToTensorV2
  data_loader:
    batch_size: 16
    num_workers: 4
    shuffle: true
```

A complete example with train and val splits is at
[`conf/examples/mbtiles_polygon_dataset.yaml`](https://github.com/phborba/pytorch_segmentation_models_trainer/blob/main/conf/examples/mbtiles_polygon_dataset.yaml).

## Parameters reference

| Parameter | Type | Default | Description |
|---|---|---|---|
| `image_mbtiles_path` | str / Path | required | Image raster (MBTiles or any rasterio source) |
| `mask_mbtiles_path` | str / Path | required | Mask raster (RGB/RGBA color-coded) |
| `regions_path` | str / Path | required | Vector file with training region polygons |
| `patch_size` | int | required | Window height and width in pixels |
| `stride` | int | `patch_size` | Sliding-window stride (set `< patch_size` for overlap) |
| `color_map` | list of `[R,G,B,cls]` | `None` | Color-to-class mapping; `None` uses first band directly |
| `selected_bands` | list[int] | `None` (all) | 1-based image band indices |
| `image_dtype` | str | `"uint8"` | Output dtype (`uint8`, `uint16`, `float32`, `native`) |
| `mask_resampling` | str | `"nearest"` | Resampling for mask WarpedVRT alignment |
| `augmentation_list` | list | `None` | Albumentations transform configs |
| `data_loader` | dict | `None` | DataLoader config for the Lightning model |
| `return_metadata` | bool | `False` | Include `row_off`/`col_off` in each sample |
| `window_index_cache` | str / Path | `None` | CSV or Parquet path to cache the window index |
| `regions_layer` | str | `None` | Layer name for multi-layer vector files |

## Output sample

```python
{
    "image": torch.Tensor,   # (C, H, W) float32
    "mask":  torch.Tensor,   # (H, W)    int64  — class indices
    # "metadata": {"row_off": int, "col_off": int}  # only when return_metadata=True
}
```
