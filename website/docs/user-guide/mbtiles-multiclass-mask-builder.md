---
sidebar_position: 8
title: MBTiles Multiclass Mask Builder
---

# MBTiles Multiclass Mask Builder

Use this tool when you have an MBTiles file that defines the reference grid,
one GeoJSON with frame geometries, and one GeoPackage with class polygons. The
builder writes a single multi-class GeoTIFF per frame and a `dataset.csv`
manifest.

This is the right tool when the mask resolution must match the highest zoom
available in the MBTiles source. The MBTiles file defines the target CRS,
transform, width, height, and pixel spacing.

## Inputs

| Input | Role |
|---|---|
| `reference_mbtiles_path` | Reference raster grid used to derive resolution and alignment |
| `frames_path` | Frame polygons that define each output mask extent |
| `vector_path` | Polygon labels with integer class ids |

## Output

```text
output_dir/
  masks/
    10.tif
    11.tif
  dataset.csv
```

The CSV includes the generated mask path, dimensions, CRS, transform, and the
paths used to create the manifest.

## CLI

```bash
pytorch-smt-tools build-mbtiles-multiclass-masks /data/config.yaml
```

## Example YAML

```yaml title="conf/examples/mbtiles_multiclass_mask_builder.yaml"
reference_mbtiles_path: /data/tiles.mbtiles
frames_path: /data/locais_mascaras.geojson
vector_path: /data/dsg_masks.gpkg
output_dir: /data/output_masks
frame_layer: locais_mascaras
vector_layer: dsg_masks
frame_id_attribute: rect_id
class_attribute: tipo
background_value: 255
```

## Notes

- `background_value: 255` keeps background separate from class ids starting at
  zero.
- `n_workers` controls how many frames are processed in parallel. Use `1` to
  force sequential execution when debugging.
- `progress: true` enables a `tqdm` progress bar for long runs.
- Frames and vector polygons must be in a CRS compatible with the MBTiles
  reference grid.
- Each output mask is written as a single-band `uint8` GeoTIFF.
