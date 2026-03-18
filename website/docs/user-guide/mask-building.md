---
sidebar_position: 4
---

# Building Training Masks from Vector Data

The `build-mask` CLI mode automates the generation of all raster mask files needed for segmentation and frame field training. Given a set of georeferenced raster images and a vector polygon source, it produces every mask type and writes a ready-to-use CSV dataset index.

## Overview

Instead of manually rasterising polygons, the project provides `MaskBuilder` (and its COCO variant) which:

1. Reads polygon geometries from a configurable vector source.
2. Iterates over all raster images found under `root_dir / image_root_dir`.
3. For each image, rasterises the intersecting polygons into the requested mask types.
4. Writes output mask files to separate sub-directories under `root_dir`.
5. Produces a CSV file that maps each image to its corresponding mask files.

## Supported Vector Sources

The vector layer is configured via the `geo_df` sub-config. Four sources are supported:

### `FileGeoDFConfig` — Single GeoJSON / vector file

```yaml
geo_df:
  _target_: pytorch_segmentation_models_trainer.tools.data_handlers.vector_reader.FileGeoDF
  file_name: /path/to/polygons.geojson
```

Use this when all polygon annotations live in a single GeoJSON, Shapefile, or other OGR-readable file.

### `BatchFileGeoDFConfig` — Directory of GeoJSON files

```yaml
geo_df:
  _target_: pytorch_segmentation_models_trainer.tools.data_handlers.vector_reader.BatchFileGeoDF
  root_dir: /path/to/vectors
  file_extension: geojson
```

Use this when annotations are split across multiple files in a directory. All files with the given extension are read and merged.

### `PostgisConfig` — PostGIS database

```yaml
geo_df:
  _target_: pytorch_segmentation_models_trainer.tools.data_handlers.vector_reader.PostgisGeoDF
  database: my_database
  sql: "select id, geom from buildings"
  user: postgres
  password: postgres
  host: localhost
  port: 5432
```

Use this when polygon data is stored in a PostgreSQL/PostGIS database. The `sql` field must select the geometry column.

### `COCOGeoDFConfig` — COCO annotation file

```yaml
geo_df:
  _target_: pytorch_segmentation_models_trainer.tools.data_handlers.vector_reader.COCOGeoDF
  file_name: /path/to/annotations.json
```

Use this when annotations are in COCO JSON format. When using this source, the `COCOMaskBuilder` class (instead of `MaskBuilder`) must be used, as it matches annotation keys to image filenames.

## Mask Types

Each mask type is enabled or disabled by a boolean flag. The output path for each type is controlled by a corresponding `*_folder_name` field.

| Flag                      | Output key          | Description                                          |
|---------------------------|---------------------|------------------------------------------------------|
| `build_polygon_mask`      | `polygon_mask`      | Binary or multi-class polygon fill mask              |
| `build_boundary_mask`     | `boundary_mask`     | Rasterised polygon edge / boundary lines             |
| `build_vertex_mask`       | `vertex_mask`       | Rasterised polygon corner points                     |
| `build_crossfield_mask`   | `crossfield_mask`   | Two-channel dominant edge orientation map            |
| `build_distance_mask`     | `distance_mask`     | Distance transform from nearest polygon edge         |
| `build_size_mask`         | `size_mask`         | Polygon-area normalisation map                       |
| `build_bounding_box_list` | `bounding_boxes`    | Per-image JSON file with bounding box annotations    |
| `build_polygon_list`      | `polygon_lists`     | Per-image JSON file with polygon vertex lists        |

:::note
`build_bounding_box_list` and `build_polygon_list` produce JSON sidecar files rather than raster images. These are consumed by `ObjectDetectionDataset` and `InstanceSegmentationDataset` respectively.
:::

## `MaskBuilder` vs `COCOMaskBuilder`

| Class           | Vector source          | How it matches polygons to images |
|-----------------|------------------------|-----------------------------------|
| `MaskBuilder`   | Any (GeoJSON, PostGIS, Batch) | Spatial intersection using the raster CRS |
| `COCOMaskBuilder` | `COCOGeoDFConfig`    | Filename-based: the image stem is used as the key into the COCO annotation dict |

Use `MaskBuilder` for georeferenced rasters and standard vector files. Use `COCOMaskBuilder` when your annotations are a COCO JSON that uses image filenames as keys.

## Full Configuration Example

```yaml
# configs/mask_config.yaml

mask_builder:
  _target_: pytorch_segmentation_models_trainer.tools.mask_building.mask_builder.MaskBuilder

  # Vector data source
  geo_df:
    _target_: pytorch_segmentation_models_trainer.tools.data_handlers.vector_reader.FileGeoDF
    file_name: /path/to/polygons.geojson

  # Directory structure
  root_dir: /path/to/dataset
  image_root_dir: images          # relative to root_dir
  image_extension: tif
  image_dir_is_relative_to_root_dir: true
  replicate_image_folder_structure: true

  # Output CSV
  output_csv_path: /path/to/output
  dataset_name: my_dataset

  # Mask types to build
  build_polygon_mask: true
  polygon_mask_folder_name: polygon_masks

  build_boundary_mask: true
  boundary_mask_folder_name: boundary_masks

  build_vertex_mask: true
  vertex_mask_folder_name: vertex_masks

  build_crossfield_mask: false
  crossfield_mask_folder_name: crossfield_masks

  build_distance_mask: false
  distance_mask_folder_name: distance_masks

  build_size_mask: false
  size_mask_folder_name: size_masks

  build_bounding_box_list: false
  bounding_box_list_folder_name: bounding_boxes

  build_polygon_list: false
  polygon_list_folder_name: polygon_lists

  # Filtering
  min_polygon_area: 10.0

  # Output raster format
  mask_output_extension: png

  # Path handling in the output CSV
  dataset_has_relative_path: true
  relative_path_on_csv: true
```

### Frame Field Config Example

To generate all masks needed by a frame field model:

```yaml
mask_builder:
  _target_: pytorch_segmentation_models_trainer.tools.mask_building.mask_builder.MaskBuilder
  geo_df:
    _target_: pytorch_segmentation_models_trainer.tools.data_handlers.vector_reader.FileGeoDF
    file_name: /data/buildings/polygons.geojson
  root_dir: /data/buildings
  image_root_dir: images
  image_extension: tif
  output_csv_path: /data/buildings
  dataset_name: buildings_train
  build_polygon_mask: true
  build_boundary_mask: true
  build_vertex_mask: true
  build_crossfield_mask: true
  build_distance_mask: true
  build_size_mask: true
  build_bounding_box_list: false
  build_polygon_list: false
  min_polygon_area: 10.0
  mask_output_extension: png
```

## CLI Command

```bash
pytorch-smt --config-dir ./configs --config-name mask_config +mode=build-mask
```

Where:
- `--config-dir` points to the directory containing your YAML config files
- `--config-name` is the config file name (without `.yaml`)
- `+mode=build-mask` selects the mask-building execution mode

Hydra's config composition is also supported. For example, to use the PostGIS source via a config group override:

```bash
pytorch-smt --config-dir ./configs --config-name mask_config +mode=build-mask geo_df=postgis
```

## Parallel Processing

By default, the executor uses `os.cpu_count()` parallel workers. You can control this with the `simultaneous_tasks` parameter in the top-level config:

```yaml
simultaneous_tasks: 8

mask_builder:
  ...
```

Set this lower if memory is constrained, or higher if I/O is the bottleneck and you have many CPU cores.

:::tip
For large datasets (thousands of GeoTIFF tiles), setting `simultaneous_tasks` to 2–4 times the number of physical cores often gives the best throughput, since each task is a mix of CPU work (rasterisation) and I/O.
:::

## Output CSV Schema

The generated CSV contains one row per image and includes the image path plus a column for each mask type that was built:

| Column            | Description                                                  |
|-------------------|--------------------------------------------------------------|
| `image`           | Path to the source image                                     |
| `width`           | Image width in pixels                                        |
| `height`          | Image height in pixels                                       |
| `bands_means`     | Per-band mean pixel values                                   |
| `bands_stds`      | Per-band standard deviation values                           |
| `class_freq`      | Per-class pixel frequency in the polygon mask                |
| `polygon_mask`    | Path to the polygon mask (if `build_polygon_mask: true`)     |
| `boundary_mask`   | Path to the boundary mask (if `build_boundary_mask: true`)   |
| `vertex_mask`     | Path to the vertex mask (if `build_vertex_mask: true`)       |
| `crossfield_mask` | Path to the crossfield map (if `build_crossfield_mask: true`)|
| `distance_mask`   | Path to the distance mask (if `build_distance_mask: true`)   |
| `size_mask`       | Path to the size mask (if `build_size_mask: true`)           |
| `bounding_boxes`  | Path to the bbox JSON (if `build_bounding_box_list: true`)   |
| `polygon_lists`   | Path to the polygon JSON (if `build_polygon_list: true`)     |

Paths in the CSV are relative (when `dataset_has_relative_path: true`) or absolute. This CSV can be used directly as `input_csv_path` for any dataset class.

### Merging with an Existing CSV

Set `merge_existing: true` to merge the new output into an existing CSV of the same name instead of overwriting it:

```yaml
merge_existing: true
```

This is useful when running mask building in stages (e.g., first generate polygon and boundary masks, then add crossfield masks later).
