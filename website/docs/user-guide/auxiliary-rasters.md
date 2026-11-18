---
sidebar_position: 20
title: Auxiliary Rasters (GEM-15, Forest Typology)
---

# Auxiliary Rasters: GEM-15 and Forest Typology

`tools/aux_rasters` builds per-tile `.npz` caches of two public raster
products, read directly from their Cloud-Optimized GeoTIFFs on Google Cloud
Storage (no Earth Engine account needed). Each tile's window is reprojected
onto the tile's own WGS84 grid, the same convention as the
[AlphaEarth](./aef-embeddings.md) and OlmoEarth caches, so the result lines
up pixel for pixel with them (`64x64` = 10 m for a 256 px tile at 2.5 m).

| Product | Content | Resolution | Year | License |
|---|---|---|---|---|
| [GEM-15](https://developers.google.com/earth-engine/datasets/catalog/projects_climate-and-sustainability_assets_gem-15_v1) (Google Research) | DSM (mean, p5, p50, p95, std) + DTM | 15 m | stereo mostly 2022-2024 | CC-BY 4.0 |
| [Forest Typology 2020 v1.0](https://developers.google.com/earth-engine/datasets/catalog/projects_nature-trace_assets_forest_typology_forest_typology_2020_v1_0_collection) (Google / Nature Trace) | P(primary, naturally regenerating, planted, plantation, tree crops) | 10 m | 2020 | CC-BY 4.0 |

## Downloading

Both commands take a manifest CSV with one row per tile: `id`,
`bbox_west`, `bbox_south`, `bbox_east`, `bbox_north` (WGS84 degrees). They
are threaded, show a progress bar, skip tiles already cached (use
`--overwrite` to redo them) and print how many tiles were written, skipped,
had no data, or failed.

```bash
pytorch-smt-tools download-gem15 --manifest manifest.csv --output-dir cache_gem15 --size 64
pytorch-smt-tools download-forest-typology --manifest manifest.csv --output-dir cache_forty --size 64
```

ForTy's bucket has no spatial index, so the first run lists the files of
each MGRS grid zone the tiles touch and reads every file header once,
caching the footprints in `<output-dir>/forty_index.json` (about 10 s per
zone).

## Cache format

**GEM-15** `<id>.npz`: `dsm` `(5, H, W)` float32 (mean, p5, p50, p95, std),
`dtm` `(H, W)` float32, `pixel_size_m` `(2,)`. NaN is nodata; GEM-15 masks
water, snow and other low-texture surfaces this way, so the validity mask
itself carries information.

**ForTy** `<id>.npz`: `forty` `(5, H, W)` uint8, probability x 250,
255 = nodata.

## Features and loaders

`gem15.terrain_features(dsm, dtm, pixel_size, tpi_radius=5)` returns 7
relative channels: height above ground (`dsm - dtm`), `p95 - dtm`,
`dsm_std`, `p95 - p5`, slope (degrees, from the DTM), topographic position
index and a validity mask. Absolute heights are not exposed: the dataset
reports regional mean offsets of up to about 5 m against airborne lidar,
and its DTM is less accurate under dense canopy.

`forest_typology.forty_probabilities(raw)` returns the 6 class
probabilities, adding `OtherLand = 1 - sum`.

Both loaders have the `load_batch(mask_paths, device, size)` interface of
`AefCacheLoader` and return zeros for a tile without a cache file:

```yaml
gem15_loader:
  _target_: pytorch_segmentation_models_trainer.tools.aux_rasters.gem15.Gem15CacheLoader
  cache_dir: /data/cache_gem15
  tpi_radius: 5

forty_loader:
  _target_: pytorch_segmentation_models_trainer.tools.aux_rasters.forest_typology.ForTyCacheLoader
  cache_dir: /data/cache_forty
```

## Caveats

- GEM-15 is experimental and under review; its stereo dates vary per area,
  so the height of seasonal crops depends on the acquisition date.
- ForTy is a 2020 snapshot with uncalibrated probabilities (planted forest
  F1 58.2% per the dataset page); fast-rotation plantations may have been
  harvested since.
