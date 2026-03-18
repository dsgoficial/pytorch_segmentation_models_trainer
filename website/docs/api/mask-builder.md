---
sidebar_position: 8
title: Mask Builder API
---

# Mask Builder API

The Mask Builder system generates multi-channel training mask files from geospatial vector data (polygons) paired with raster images. For each input image it can produce up to six mask types, a CSV dataset manifest, and optionally bounding-box or polygon list files.

All classes and helpers live in:

```
pytorch_segmentation_models_trainer.tools.mask_building.mask_builder
```

---

## Overview

The typical workflow is:

1. Configure a **vector source** (`FileGeoDFConfig`, `PostgisConfig`, etc.) that provides the polygon geometries.
2. Configure a **mask builder** (`MaskBuilder` or `COCOMaskBuilder`) pointing at the image root directory and the desired output paths.
3. Call `build_generator()` to iterate over image paths, then `process()` on each to produce masks and a `DatasetEntry`.
4. Collect results and call `build_csv_file_from_concurrent_futures_output()` to write the dataset CSV.

---

## Vector Source Config Classes

These are Hydra-compatible dataclasses that specify where polygon data comes from.

### `FileGeoDFConfig`

**Import path**

```python
from pytorch_segmentation_models_trainer.tools.mask_building.mask_builder import FileGeoDFConfig
```

Loads all geometries from a single vector file (GeoJSON, Shapefile, GPKG, etc.) using GeoPandas.

| Field | Type | Default | Description |
|---|---|---|---|
| `_target_` | `str` | `"...vector_reader.FileGeoDF"` | Hydra instantiation target. |
| `file_name` | `str` | `"../tests/testing_data/data/vectors/test_polygons.geojson"` | Path to the vector file. |

---

### `PostgisConfig`

**Import path**

```python
from pytorch_segmentation_models_trainer.tools.mask_building.mask_builder import PostgisConfig
```

Connects to a PostGIS database and executes a SQL query to retrieve geometries.

| Field | Type | Default | Description |
|---|---|---|---|
| `_target_` | `str` | `"...vector_reader.PostgisGeoDF"` | Hydra instantiation target. |
| `database` | `str` | `"dataset_mestrado"` | Database name. |
| `sql` | `str` | `"select id, geom from buildings"` | SQL query to retrieve geometries. |
| `user` | `str` | `"postgres"` | Database user. |
| `password` | `str` | `"postgres"` | Database password. |
| `host` | `str` | `"localhost"` | Database host. |
| `port` | `int` | `5432` | Database port. |

---

### `BatchFileGeoDFConfig`

**Import path**

```python
from pytorch_segmentation_models_trainer.tools.mask_building.mask_builder import BatchFileGeoDFConfig
```

Loads vector files from a directory in batch, matching by file extension.

| Field | Type | Default | Description |
|---|---|---|---|
| `_target_` | `str` | `"...vector_reader.BatchFileGeoDF"` | Hydra instantiation target. |
| `root_dir` | `str` | `"../tests/testing_data/data/vectors"` | Directory containing vector files. |
| `file_extension` | `str` | `"geojson"` | Extension filter (without dot). |

---

### `COCOGeoDFConfig`

**Import path**

```python
from pytorch_segmentation_models_trainer.tools.mask_building.mask_builder import COCOGeoDFConfig
```

Reads polygon annotations from a COCO-format JSON annotation file.

| Field | Type | Default | Description |
|---|---|---|---|
| `_target_` | `str` | `"...vector_reader.COCOGeoDF"` | Hydra instantiation target. |
| `file_name` | `str` | `MISSING` | Path to the COCO annotation JSON file. Must be provided. |

---

## `TemplateMaskBuilder`

**Import path**

```python
from pytorch_segmentation_models_trainer.tools.mask_building.mask_builder import TemplateMaskBuilder
```

Abstract dataclass base class for mask builders. Defines all configuration fields and shared logic for directory creation, mask type enumeration, and generator construction. Cannot be instantiated directly.

### Fields

| Field | Type | Default | Description |
|---|---|---|---|
| `defaults` | `list` | `[{"geo_df": "batch_file"}]` | Hydra defaults list for composable config. |
| `geo_df` | `VectorReaderConfig` | `MISSING` | Vector data source config. Must be set. |
| `root_dir` | `str` | `"/data"` | Root directory for the dataset. All relative paths are resolved from here. |
| `output_csv_path` | `str` | `"/data"` | Directory where the output CSV manifest is written. |
| `dataset_name` | `str` | `"dsg_dataset"` | Name used for the output CSV file (`<dataset_name>.csv`). |
| `merge_existing` | `bool` | `False` | When `True`, merge the new entries into an existing CSV instead of overwriting. |
| `dataset_has_relative_path` | `bool` | `True` | Store relative paths in the CSV manifest. |
| `image_root_dir` | `str` | `"images"` | Directory containing input raster images. Relative to `root_dir` when `image_dir_is_relative_to_root_dir=True`. |
| `image_extension` | `str` | `"tif"` | Glob extension used to find input images. |
| `image_dir_is_relative_to_root_dir` | `bool` | `True` | Interpret `image_root_dir` as relative to `root_dir`. |
| `replicate_image_folder_structure` | `bool` | `True` | Mirror the input image directory tree in each mask output directory. |
| `relative_path_on_csv` | `bool` | `True` | Write relative paths in the CSV (legacy field, prefer `dataset_has_relative_path`). |
| `build_polygon_mask` | `bool` | `True` | Generate a filled polygon raster mask. |
| `polygon_mask_folder_name` | `str` | `"polygon_masks"` | Output subdirectory name for polygon masks. |
| `build_boundary_mask` | `bool` | `True` | Generate a polygon boundary (edge) raster mask. |
| `boundary_mask_folder_name` | `str` | `"boundary_masks"` | Output subdirectory name for boundary masks. |
| `build_vertex_mask` | `bool` | `True` | Generate a polygon vertex raster mask. |
| `vertex_mask_folder_name` | `str` | `"vertex_masks"` | Output subdirectory name for vertex masks. |
| `build_crossfield_mask` | `bool` | `True` | Generate a 4-channel cross-field orientation mask. |
| `crossfield_mask_folder_name` | `str` | `"crossfield_masks"` | Output subdirectory name for cross-field masks. |
| `build_distance_mask` | `bool` | `True` | Generate a distance-transform mask (distance to nearest polygon edge). |
| `distance_mask_folder_name` | `str` | `"distance_masks"` | Output subdirectory name for distance masks. |
| `build_size_mask` | `bool` | `True` | Generate a size mask (normalised polygon area per pixel). |
| `size_mask_folder_name` | `str` | `"size_masks"` | Output subdirectory name for size masks. |
| `build_bounding_box_list` | `bool` | `False` | Generate a bounding-box annotation file per image. |
| `bounding_box_list_folder_name` | `str` | `"bounding_boxes"` | Output subdirectory name for bounding-box files. |
| `build_polygon_list` | `bool` | `False` | Generate a polygon list annotation file per image. |
| `polygon_list_folder_name` | `str` | `"polygon_lists"` | Output subdirectory name for polygon list files. |
| `min_polygon_area` | `float` | `50.0` | Minimum polygon area (in pixels) below which polygons are discarded. |
| `mask_output_extension` | `str` | `"png"` | File extension for output mask rasters (`"png"`, `"tif"`, etc.). |

### Key Methods

| Method | Signature | Description |
|---|---|---|
| `process()` | `(input_raster_path, output_dir, filter_area) -> DatasetEntry` | **Abstract.** Must be implemented by subclasses. Processes one raster image and returns a `DatasetEntry` for the CSV manifest. |
| `build_generator()` | `() -> generator` | Returns a generator of `(input_raster_path, root_dir, output_dir_list)` tuples for all images found under `image_root_dir`. Used to iterate tasks for a process pool. |
| `get_number_of_tasks()` | `() -> int` | Returns the total count of images to process. Useful for progress reporting. |

---

## `MaskBuilder`

**Import path**

```python
from pytorch_segmentation_models_trainer.tools.mask_building.mask_builder import MaskBuilder
```

Concrete implementation of `TemplateMaskBuilder` for standard vector data sources (file, PostGIS, or batch). Uses the `geo_df` vector source directly — it is applied to every image in the dataset.

### `process()` Method

```python
def process(self, input_raster_path: str, output_dir: str, filter_area: float = None) -> DatasetEntry
```

For the given raster path:
1. Opens the raster to get its bounding box and CRS.
2. Clips `self.geo_df` to the raster extent.
3. Rasterises the clipped polygons into each requested mask type.
4. Writes mask files to the appropriate subdirectories.
5. Returns a `DatasetEntry` dataclass with paths to all generated files and image statistics.

---

## `COCOMaskBuilder`

**Import path**

```python
from pytorch_segmentation_models_trainer.tools.mask_building.mask_builder import COCOMaskBuilder
```

Concrete implementation for COCO-format datasets. Assumes the `geo_df` is a `COCOGeoDF` object that is keyed by image filename stem. For each raster, it calls `self.geo_df.get_geodf_item(key)` to retrieve the specific polygon annotations for that image before processing.

### `process()` Method

```python
def process(self, input_raster_path: str, output_dir: str, filter_area: float = None) -> DatasetEntry
```

Derives the lookup key from `os.path.basename(input_raster_path).split(".")[0]`, retrieves the per-image GeoDataFrame, and delegates to `build_mask_and_ds_entry()`.

---

## Helper Functions

### `build_csv_file_from_concurrent_futures_output()`

```python
def build_csv_file_from_concurrent_futures_output(cfg, result_list) -> str
```

Writes a list of `DatasetEntry` dataclass instances to a CSV file.

- When `cfg.merge_existing` is `False`, writes directly to `<output_csv_path>/<dataset_name>.csv`.
- When `cfg.merge_existing` is `True`, writes to a temporary `temp.csv` first, merges it into the existing dataset CSV using `merge_csv_datasets()`, then removes the temporary file.

Returns the path to the final output CSV.

| Parameter | Description |
|---|---|
| `cfg` | A `TemplateMaskBuilder` config with `output_csv_path`, `dataset_name`, and `merge_existing` fields. |
| `result_list` | List of `DatasetEntry` instances returned by `process()` calls. |

---

### `merge_csv_datasets()`

```python
def merge_csv_datasets(file1, file2, key_column, output_file_name=None) -> None
```

Merges two CSV files on a key column. Rows in `file2` update matching rows in `file1`. Both files are sorted by `key_column` before merging. Columns that are entirely `NaN` in `file2` are dropped. The result is written to `output_file_name` (defaults to `file1`).

| Parameter | Description |
|---|---|
| `file1` | Path to the base (existing) CSV file. |
| `file2` | Path to the new CSV to merge in. |
| `key_column` | Column name to match rows on (e.g., `"image"`). |
| `output_file_name` | Output path. Defaults to `file1`. |

---

## Full Configuration Example

The following YAML config uses `MaskBuilder` with a GeoJSON vector source to build all six mask types for a dataset of GeoTIFF images.

```yaml
# @package _global_

defaults:
  - _self_

mask_builder:
  _target_: pytorch_segmentation_models_trainer.tools.mask_building.mask_builder.MaskBuilder
  root_dir: /data/my_dataset
  output_csv_path: /data/my_dataset
  dataset_name: my_dataset
  merge_existing: false
  image_root_dir: images
  image_extension: tif
  image_dir_is_relative_to_root_dir: true
  replicate_image_folder_structure: true
  dataset_has_relative_path: true

  # Mask types to generate
  build_polygon_mask: true
  polygon_mask_folder_name: polygon_masks
  build_boundary_mask: true
  boundary_mask_folder_name: boundary_masks
  build_vertex_mask: true
  vertex_mask_folder_name: vertex_masks
  build_crossfield_mask: true
  crossfield_mask_folder_name: crossfield_masks
  build_distance_mask: true
  distance_mask_folder_name: distance_masks
  build_size_mask: true
  size_mask_folder_name: size_masks

  # Optional annotation outputs
  build_bounding_box_list: false
  build_polygon_list: false

  # Filtering
  min_polygon_area: 50.0
  mask_output_extension: png

  # Vector data source
  geo_df:
    _target_: pytorch_segmentation_models_trainer.tools.data_handlers.vector_reader.FileGeoDF
    file_name: /data/my_dataset/vectors/buildings.geojson
```

### PostGIS Vector Source Example

```yaml
  geo_df:
    _target_: pytorch_segmentation_models_trainer.tools.data_handlers.vector_reader.PostgisGeoDF
    database: my_gis_db
    sql: "SELECT id, geom FROM buildings WHERE city = 'Rio de Janeiro'"
    user: postgres
    password: secret
    host: localhost
    port: 5432
```

### COCO Format Example

```yaml
mask_builder:
  _target_: pytorch_segmentation_models_trainer.tools.mask_building.mask_builder.COCOMaskBuilder
  root_dir: /data/coco_dataset
  output_csv_path: /data/coco_dataset
  dataset_name: coco_buildings
  image_root_dir: images
  image_extension: jpg
  build_crossfield_mask: false
  build_distance_mask: false
  build_size_mask: false
  mask_output_extension: png
  geo_df:
    _target_: pytorch_segmentation_models_trainer.tools.data_handlers.vector_reader.COCOGeoDF
    file_name: /data/coco_dataset/annotations/instances_train2017.json
```
