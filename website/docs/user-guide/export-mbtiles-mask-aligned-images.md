---
sidebar_position: 6
title: Export MBTiles Mask-Aligned Images
---

# Export MBTiles Mask-Aligned Images

Use this tool before training when imagery is stored in MBTiles and labels are
GeoTIFF masks. The mask is treated as the reference grid. Each exported image is
warped into the mask window CRS, transform, resolution, and shape.

This is useful for visual QA: if previews look shifted, the training dataset
would learn from misaligned image/label pairs.

## Output

The exporter writes:

```text
output_dir/
  images/      # source imagery aligned to mask grid
  masks/       # copied mask windows
  previews/    # RGB PNG with mask overlay
  manifest.csv
```

`manifest.csv` contains:

```csv
image_path,mask_path,preview_path,source_mask_path,row_off,col_off,width,height,crs,transform
```

The manifest can be inspected directly or reused as a patch inventory for
follow-up processing.

## CLI

Export one preview per full mask:

```bash
pytorch-smt-tools export-mbtiles-mask-aligned \
  --mbtiles-path /data/source.mbtiles \
  --mask-dir /data/masks \
  --output-dir /data/qa_mbtiles \
  --full-mask
```

Export fixed-size patch previews:

```bash
pytorch-smt-tools export-mbtiles-mask-aligned \
  --mbtiles-path /data/source.mbtiles \
  --mask-dir /data/masks \
  --output-dir /data/qa_mbtiles \
  --patch-size 512 \
  --stride 512 \
  --selected-bands 1,2,3
```

## Hydra Example

```yaml title="conf/examples/mbtiles_mask_aligned_export.yaml"
mbtiles_export:
  _target_: pytorch_segmentation_models_trainer.tools.mbtiles.export_mask_aligned_images.export_mask_aligned_images
  mbtiles_path: /data/source_imagery.mbtiles
  mask_dir: /data/masks
  output_dir: /data/qa_mbtiles_tiles
  patch_size: 512
  stride: 512
  selected_bands: [1, 2, 3]
  image_dtype: uint8
  image_resampling: bilinear
  write_sidecar_png: true
```

## Alignment Model

For each mask window:

1. Read the mask directly from the GeoTIFF.
2. Compute `dst_transform = mask_src.window_transform(window)`.
3. Set `dst_crs = mask_src.crs`.
4. Read the MBTiles/source raster through rasterio and warp it into the mask
   window grid.
5. Write aligned image, mask, and optional PNG overlay.

The MBTiles zoom/resolution controls native source detail. The exported raster
grid is always the mask grid.

## Recommendations

- Use `bilinear` or `cubic` resampling for RGB imagery.
- Use `nearest` for categorical source rasters.
- Keep `write_sidecar_png: true` during QA.
- Enable `skip_empty_masks` when empty background patches are not useful for
  visual review.
- If `rasterio.open()` cannot read the MBTiles file in your GDAL build, convert
  the MBTiles to a GeoTIFF/COG first or add the planned SQLite tile fallback.
