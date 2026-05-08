---
sidebar_position: 3
title: Dataset Classes
---

# Dataset Classes

This page is the API reference for all dataset classes in `pytorch_segmentation_models_trainer`.

Most classes live in:

```python
from pytorch_segmentation_models_trainer.dataset_loader.dataset import <ClassName>
```

`RasterPatchDataset` lives in its own module:

```python
from pytorch_segmentation_models_trainer.dataset_loader.raster_patch_dataset import RasterPatchDataset
```

---

## Class Hierarchy

```
torch.utils.data.Dataset
├── RasterPatchDataset          ← sliding-window, folder-based (no CSV)
└── AbstractDataset             ← CSV / DataFrame-based
    ├── ImageDataset
    │   ├── CSVWindowedImageDataset
    │   └── TiledInferenceImageDataset
    ├── SegmentationDataset
    │   ├── SegmentationDatasetFromFolder
    │   ├── CSVWindowedSegmentationDataset
    │   └── FrameFieldSegmentationDataset
    ├── RandomCropSegmentationDataset
    ├── ObjectDetectionDataset
    │   └── InstanceSegmentationDataset
    └── PolygonRNNDataset
```

---

## `RasterPatchDataset`

```python
from pytorch_segmentation_models_trainer.dataset_loader.raster_patch_dataset import RasterPatchDataset
```

Systematic sliding-window dataset for semantic segmentation directly over full-size raster files. Discovers image/mask pairs by **recursive folder scan**, computes all valid `patch_size × patch_size` windows for each image, and maps a global 1-D index to the correct (image, row, column) in O(log N) via binary search. Only the requested window pixels are read from disk at `__getitem__` time — full images are never loaded into memory.

See the [Sliding-Window Patch Dataset](../user-guide/dataset-raster-patch.md) user guide for usage examples and the full YAML reference.

### Constructor Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `image_dir` | `str \| Path` | **required** | Root directory of input images. Scanned recursively. |
| `mask_dir` | `str \| Path` | **required** | Root directory of segmentation masks. Matching is by relative path. |
| `extension` | `str` | `".tif"` | Image file extension. Leading dot is optional and normalized internally. |
| `patch_size` | `int` | `256` | Side length of each square patch in pixels. |
| `stride` | `int` | `128` | Step between patch origins. `stride < patch_size` produces overlapping patches. |
| `mask_extension` | `str \| None` | `None` | Mask file extension. When `None`, uses the same value as `extension`. |
| `augmentation_list` | `list \| A.Compose \| None` | `None` | Albumentations augmentation pipeline. Image is passed as `(H, W, C)`, mask as `(H, W)`. |
| `data_loader` | `Any \| None` | `None` | DataLoader sub-configuration. Stored as `ds.data_loader`; consumed by the Lightning `Model`. |
| `selected_bands` | `List[int] \| None` | `None` | 1-based rasterio band indices to load. `None` loads all bands. |
| `image_dtype` | `str` | `"uint8"` | Array dtype after reading. Accepted: `"uint8"`, `"uint16"`, `"float32"`, `"native"`. |

### Raises

| Exception | When |
|---|---|
| `ValueError` | `image_dtype` is not one of the accepted values. |
| `ValueError` | `selected_bands` contains a non-positive integer. |
| `ValueError` | No valid image/mask pairs are found after scanning both directories. |
| `UserWarning` | An image is smaller than `patch_size` (skipped silently). |
| `UserWarning` | A mask file is missing for a discovered image (skipped silently). |

### Key Attributes

| Attribute | Type | Description |
|---|---|---|
| `image_info` | `List[Dict]` | Per-image metadata: `img_path`, `mask_path`, `height`, `width`, `patches_per_row`, `patches_per_col`. |
| `patch_size` | `int` | Patch side length. |
| `stride` | `int` | Step between patch origins. |
| `image_dtype` | `str` | Configured dtype. |
| `selected_bands` | `List[int] \| None` | Band selection (1-based). |
| `data_loader` | `Any` | Stored DataLoader config. |

### `__len__`

Returns the **total number of patches** across all images, not the number of images.

```
len(ds) = Σ_i  patches_per_row_i × patches_per_col_i
```

### `__getitem__` Returns

`Dict[str, torch.Tensor]` with keys:

| Key | dtype | shape | Notes |
|---|---|---|---|
| `"image"` | `torch.float32` | `(C, patch_size, patch_size)` | Normalised ÷255 (`uint8`) or ÷65535 (`uint16`) when no transform; unchanged for `float32`/`native`. |
| `"mask"` | `torch.int64` | `(patch_size, patch_size)` | Raw pixel values from the mask file. |

When `augmentation_list` is set the return type matches whatever the Albumentations pipeline produces (typically the same keys with transformed tensors after `ToTensorV2`).

---

## `AbstractDataset`

```python
from pytorch_segmentation_models_trainer.dataset_loader.dataset import AbstractDataset
```

Base class for all datasets. Handles CSV loading, root directory resolution, and augmentation setup. Subclasses must implement `__getitem__`.

### Constructor Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `input_csv_path` | `Path` | `None` | Path to the CSV file. Required when `df` is not provided. |
| `df` | `pd.DataFrame` | `None` | Pre-loaded DataFrame. When provided, `input_csv_path` is ignored for loading but still stored. |
| `root_dir` | `Any` | `None` | Root directory prepended to all relative file paths read from the CSV. |
| `augmentation_list` | `Any` | `None` | Albumentations augmentation list or `A.Compose` object. `None` disables augmentation. |
| `data_loader` | `Any` | `None` | DataLoader sub-configuration (see [DataLoader Config Keys](#dataloader-config-keys) below). |
| `image_key` | `str` | `"image"` | CSV column name for image paths. |
| `mask_key` | `str` | `"mask"` | CSV column name for mask paths. |
| `n_first_rows_to_read` | `int` | `None` | If set, only the first N rows are read from the CSV via `pd.read_csv(..., nrows=N)`. |

### Key Methods

| Method | Signature | Description |
|--------|-----------|-------------|
| `__len__` | `() -> int` | Returns the number of rows in the dataset DataFrame. |
| `get_path` | `(idx, key=None, add_root_dir=True) -> str` | Returns the file path for item `idx` under the given CSV column key. |
| `load_image` | `(idx, key=None, is_mask=False, force_rgb=False, is_binary_mask=True) -> np.ndarray` | Loads and returns a numpy array for the given item. |
| `update_df` | `(new_df) -> None` | Replaces the internal DataFrame and updates `self.len`. |

---

## `CSVWindowedImageDataset`

```python
from pytorch_segmentation_models_trainer.dataset_loader.dataset import CSVWindowedImageDataset
```

Extends `ImageDataset` to read specific patches from large images using `rasterio` windowed read. Coordinates (offsets) for each patch are read from the input CSV. Does not include masks.

### Constructor Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `input_csv_path` | `Path \| str` | `None` | Path to the CSV file. |
| `df` | `pd.DataFrame` | `None` | Pre-built DataFrame. |
| `root_dir` | `Any` | `None` | Root directory for path resolution. |
| `augmentation_list` | `Any` | `None` | Albumentations augmentation pipeline. |
| `data_loader` | `Any` | `None` | DataLoader sub-configuration. |
| `image_key` | `str` | `"image"` | CSV column for image paths. |
| `row_off_key` | `str` | `"row_off"` | CSV column for vertical offset. |
| `col_off_key` | `str` | `"col_off"` | CSV column for horizontal offset. |
| `patch_size_key` | `str` | `"patch_size"` | CSV column for patch size. |
| `n_first_rows_to_read` | `int` | `None` | Limit on rows to read. |
| `selected_bands` | `Optional[List[int]]` | `None` | 1-based band indices to load. |
| `use_rasterio` | `bool` | `True` | Forces rasterio for windowed read. |
| `image_dtype` | `str` | `"uint8"` | Data type for rasterio-loaded images. |

### `__getitem__` Returns

`Dict[str, Any]` with keys:

| Key | Type | Description |
|-----|------|-------------|
| `"image"` | `np.ndarray` or `torch.Tensor` | The loaded patch, optionally transformed. |
| `"path"` | `str` | Absolute path to the source image file. |

---

## `ImageDataset`

```python
from pytorch_segmentation_models_trainer.dataset_loader.dataset import ImageDataset
```

Image-only dataset. Returns a dict with the loaded image and its file path. Suitable for inference pipelines that do not require ground-truth masks.

### Constructor Parameters

Inherits all parameters from `AbstractDataset`. The `mask_key` parameter is accepted by the parent but not used.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `input_csv_path` | `Path` | `None` | Path to the CSV file. |
| `df` | `pd.DataFrame` | `None` | Pre-loaded DataFrame. |
| `root_dir` | `Any` | `None` | Root directory for path resolution. |
| `augmentation_list` | `Any` | `None` | Albumentations augmentation pipeline. |
| `data_loader` | `Any` | `None` | DataLoader sub-configuration. |
| `image_key` | `str` | `"image"` | CSV column name for image paths. |
| `n_first_rows_to_read` | `int` | `None` | Limit on rows to read. |

### `__getitem__` Returns

`Dict[str, Any]` with keys:

| Key | Type | Description |
|-----|------|-------------|
| `"image"` | `np.ndarray` or `torch.Tensor` | The loaded image, optionally transformed. |
| `"path"` | `str` | Absolute path to the source image file. |

---

## `SegmentationDataset`

```python
from pytorch_segmentation_models_trainer.dataset_loader.dataset import SegmentationDataset
```

Semantic segmentation dataset. Loads image-mask pairs and supports both PIL (RGB) and rasterio (multi-band/multispectral) image loading.

### Constructor Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `input_csv_path` | `Path` | `None` | Path to the CSV file. Mutually exclusive with `df`; at least one must be provided. |
| `df` | `pd.DataFrame` | `None` | Pre-built DataFrame with `image` and `mask` columns. Allows creating the dataset without a CSV file on disk (e.g. via `SegmentationDatasetFromFolder`). |
| `root_dir` | `Any` | `None` | Root directory for path resolution. |
| `augmentation_list` | `Any` | `None` | Albumentations augmentation pipeline. |
| `data_loader` | `Any` | `None` | DataLoader sub-configuration. |
| `image_key` | `str` | `"image"` | CSV column name for image paths. |
| `mask_key` | `str` | `"mask"` | CSV column name for mask paths. |
| `n_first_rows_to_read` | `int` | `None` | Limit on rows to read. |
| `n_classes` | `int` | `2` | Number of segmentation classes. When `2`, masks are binarized (`> 0`). |
| `selected_bands` | `Optional[List[int]]` | `None` | 1-based list of band indices to load via rasterio. E.g. `[1, 2, 3]` loads the first three bands. When `None`, all bands are loaded. |
| `use_rasterio` | `bool` | `False` | When `True`, forces rasterio for image loading (recommended for multispectral imagery). |
| `image_dtype` | `str` | `"uint8"` | Data type applied to the image array after rasterio loading. Accepted values: `"uint8"`, `"uint16"`, `"float32"`, `"native"`. Only takes effect when rasterio is used (`use_rasterio=True` or `selected_bands` is set). `"native"` skips the cast entirely. Raises `ValueError` for unrecognised values. |
| `reset_augmentation_function` | `bool` | `False` | When `True`, deep-copies the augmentation pipeline before each call to prevent memory leaks from Albumentations caching. |

### `__getitem__` Returns

`Dict[str, Any]` with keys:

| Key | Type | Description |
|-----|------|-------------|
| `"image"` | `torch.Tensor` (C, H, W) | Float tensor. When no transform is set, automatically normalized: `uint8` → `/255`, `uint16` → `/65535`, `float32`/`native` → no division. |
| `"mask"` | `torch.Tensor` (H, W) | Long tensor of class labels. Binary (0/1) when `n_classes == 2`. |

---

## `SegmentationDatasetFromFolder`

```python
from pytorch_segmentation_models_trainer.dataset_loader.dataset import SegmentationDatasetFromFolder
```

Extends `SegmentationDataset` to discover image/mask pairs **recursively from two root folders**, without requiring a CSV file on disk. Matching is performed by **relative subfolder path** and **file stem** (name without extension): only files present in both folders, inside the same subfolder and with the same filename, are included in the dataset.

### Expected Folder Structure

```
images_root/
    area_a/
        tile_001.tif
        tile_002.tif
    area_b/
        tile_003.tif

masks_root/
    area_a/
        tile_001.tif   ← paired with images_root/area_a/tile_001.tif
        tile_002.tif
    area_b/
        tile_003.tif
```

Only files with a matching `(subfolder, stem)` key in both trees are included. Files present in only one folder are silently ignored.

### Constructor Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `image_folder` | `Union[str, Path]` | **required** | Root folder containing the images. |
| `mask_folder` | `Union[str, Path]` | **required** | Root folder containing the masks. |
| `image_extension` | `str` | `".tif"` | File extension for images. The leading dot is optional and normalized internally (e.g. `"tif"` and `".tif"` are equivalent). |
| `mask_extension` | `Optional[str]` | `None` | File extension for masks. When `None`, uses the same value as `image_extension`. |
| `augmentation_list` | `Any` | `None` | Albumentations augmentation pipeline. |
| `data_loader` | `Any` | `None` | DataLoader sub-configuration. |
| `n_classes` | `int` | `2` | Number of segmentation classes. When `2`, masks are binarized (`> 0`). |
| `selected_bands` | `Optional[List[int]]` | `None` | 1-based band indices to load via rasterio. `None` loads all bands. |
| `use_rasterio` | `bool` | `False` | Forces rasterio for image loading. |
| `image_dtype` | `str` | `"uint8"` | Data type for rasterio-loaded images. See `SegmentationDataset`. |
| `reset_augmentation_function` | `bool` | `False` | Deep-copy the transform to prevent Albumentations memory leaks. |

### Raises

| Exception | When |
|-----------|------|
| `ValueError` | No matching image/mask pairs are found (wrong extension, mismatched subfolder structure, etc.). |

### Instance Attributes

After construction the following extra attributes are available:

| Attribute | Type | Description |
|-----------|------|-------------|
| `image_folder` | `Path` | Resolved root folder for images. |
| `mask_folder` | `Path` | Resolved root folder for masks. |
| `image_extension` | `str` | Normalized image extension (with leading dot). |
| `mask_extension` | `str` | Normalized mask extension (with leading dot). |

### `__getitem__` Returns

Same as `SegmentationDataset`:

| Key | Type | Description |
|-----|------|-------------|
| `"image"` | `torch.Tensor` (C, H, W), float32 | Normalized image tensor. |
| `"mask"` | `torch.Tensor` (H, W), int64 | Class-index mask. |

### Static Helper Methods

| Method | Description |
|--------|-------------|
| `_normalize_extension(ext)` | Ensures the extension starts with a dot. |
| `_build_dataframe(image_folder, mask_folder, image_extension, mask_extension)` | Scans both folders recursively, matches pairs, and returns a `pd.DataFrame` with `image` and `mask` columns. Raises `ValueError` if no pairs are found. |

---

## `CSVWindowedSegmentationDataset`

```python
from pytorch_segmentation_models_trainer.dataset_loader.dataset import CSVWindowedSegmentationDataset
```

Extends `SegmentationDataset` to read specific patches from large GeoTIFFs using `rasterio` windowed read. Coordinates (offsets) for each patch are read from the input CSV.

### Constructor Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `input_csv_path` | `Path \| str` | `None` | Path to the CSV file. |
| `df` | `pd.DataFrame` | `None` | Pre-built DataFrame (alternative to `input_csv_path`). |
| `root_dir` | `Any` | `None` | Root directory for path resolution. |
| `augmentation_list` | `Any` | `None` | Albumentations augmentation pipeline. |
| `data_loader` | `Any` | `None` | DataLoader sub-configuration. |
| `image_key` | `str` | `"image"` | CSV column for image paths. |
| `mask_key` | `str` | `"mask"` | CSV column for mask paths. |
| `row_off_key` | `str` | `"row_off"` | CSV column for vertical offset. |
| `col_off_key` | `str` | `"col_off"` | CSV column for horizontal offset. |
| `patch_size_key` | `str` | `"patch_size"` | CSV column for patch size. |
| `n_classes` | `int` | `2` | Number of classes. If 2, mask is binarized (`> 0`). |
| `selected_bands` | `Optional[List[int]]` | `None` | 1-based band indices to load via rasterio. |
| `use_rasterio` | `bool` | `True` | Forces rasterio for windowed read. |
| `image_dtype` | `str` | `"uint8"` | Data type for rasterio-loaded images. |

### `__getitem__` Returns

Same as `SegmentationDataset`:

| Key | Type | Description |
|-----|------|-------------|
| `"image"` | `torch.Tensor` (C, H, W), float32 | Normalized image tensor. |
| `"mask"` | `torch.Tensor` (H, W), int64 | Class-index mask. |

---

## `FrameFieldSegmentationDataset`

```python
from pytorch_segmentation_models_trainer.dataset_loader.dataset import FrameFieldSegmentationDataset
```

Extends `SegmentationDataset` for frame-field polygon learning. Loads multiple auxiliary masks (boundary, vertex, crossfield angle, distance transform, size map) in addition to the primary segmentation mask.

### Constructor Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `input_csv_path` | `Path` | **required** | Path to the CSV file. |
| `root_dir` | `Any` | `None` | Root directory for path resolution. |
| `augmentation_list` | `Any` | `None` | Albumentations augmentation pipeline. |
| `data_loader` | `Any` | `None` | DataLoader sub-configuration. |
| `image_key` | `str` | `"image"` | CSV column name for image paths. |
| `mask_key` | `str` | `"polygon_mask"` | CSV column name for the primary polygon mask. |
| `multi_band_mask` | `bool` | `False` | When `True`, all three masks (polygon, boundary, vertex) are packed into a single multi-band image file. |
| `boundary_mask_key` | `str` | `"boundary_mask"` | CSV column for the boundary mask. |
| `return_boundary_mask` | `bool` | `True` | Whether to load and return the boundary mask. |
| `vertex_mask_key` | `str` | `"vertex_mask"` | CSV column for the vertex mask. |
| `return_vertex_mask` | `bool` | `True` | Whether to load and return the vertex mask. |
| `n_first_rows_to_read` | `int` | `None` | Limit on rows to read. |
| `return_crossfield_mask` | `bool` | `True` | Whether to load the crossfield angle mask. |
| `crossfield_mask_key` | `str` | `"crossfield_mask"` | CSV column for the crossfield angle image. |
| `return_distance_mask` | `bool` | `True` | Whether to load the distance transform mask. |
| `distance_mask_key` | `str` | `"distance_mask"` | CSV column for the distance map. |
| `return_size_mask` | `bool` | `True` | Whether to load the size map. |
| `size_mask_key` | `str` | `"size_mask"` | CSV column for the size map. |
| `image_width` | `int` | `224` | Target image width used in the fallback transform when an augmentation crop is invalid. |
| `image_height` | `int` | `224` | Target image height used in the fallback transform. |
| `gpu_augmentation_list` | `Any` | `None` | Reserved for GPU-side augmentation (not currently applied inside `__getitem__`). |

### `__getitem__` Returns

`Dict[str, Any]` with keys:

| Key | Present when | Type | Description |
|-----|-------------|------|-------------|
| `"idx"` | always | `int` | Dataset index of this item. |
| `"path"` | always | `str` | File path of the source image. |
| `"image"` | always | `torch.Tensor` | Input image tensor. |
| `"gt_polygons_image"` | always | `torch.Tensor` (C, H, W) | Stacked polygon/boundary/vertex masks. |
| `"class_freq"` | always | `torch.Tensor` | Per-channel class frequency used for loss weighting. |
| `"gt_crossfield_angle"` | `return_crossfield_mask=True` | `torch.Tensor` (1, H, W) | Crossfield angle map in radians. |
| `"distances"` | `return_distance_mask=True` | `torch.Tensor` (1, H, W) | Normalized distance transform. |
| `"sizes"` | `return_size_mask=True` | `torch.Tensor` (1, H, W) | Object size map. |

---

## `ObjectDetectionDataset`

```python
from pytorch_segmentation_models_trainer.dataset_loader.dataset import ObjectDetectionDataset
```

Object detection dataset. Loads images with bounding boxes and class labels from JSON annotation files.

### Constructor Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `input_csv_path` | `Path` | **required** | Path to the CSV file. |
| `root_dir` | `Any` | `None` | Root directory for path resolution. |
| `augmentation_list` | `Any` | `None` | Albumentations augmentation pipeline with bbox support. |
| `data_loader` | `Any` | `None` | DataLoader sub-configuration. |
| `image_key` | `str` | `"image"` | CSV column name for image paths. |
| `mask_key` | `str` | `"mask"` | CSV column name (not used for detection but inherited). |
| `bounding_box_key` | `str` | `"bounding_boxes"` | CSV column pointing to JSON files with bounding box annotations. |
| `n_first_rows_to_read` | `int` | `None` | Limit on rows to read. |
| `bbox_format` | `str` | `"xywh"` | Input bounding box format (`"xywh"` or `"xyxy"`). |
| `bbox_output_format` | `str` | `"xyxy"` | Output bounding box format after conversion (`"xywh"` or `"xyxy"`). |
| `bbox_params` | `Any` | `None` | Albumentations `BboxParams` (or equivalent dict/config) passed to `A.Compose`. |

### `__getitem__` Returns

`Tuple[torch.Tensor, Dict[str, torch.Tensor], int]`:

| Position | Type | Description |
|----------|------|-------------|
| `[0]` | `torch.Tensor` | Image tensor (RGB, float32). |
| `[1]` | `dict` | Dict with keys `"boxes"` (float32 tensor of bounding boxes) and `"labels"` (int64 tensor of class indices). |
| `[2]` | `int` | Dataset index of the item. |

---

## `InstanceSegmentationDataset`

```python
from pytorch_segmentation_models_trainer.dataset_loader.dataset import InstanceSegmentationDataset
```

Extends `ObjectDetectionDataset` with optional per-instance mask and keypoint loading.

### Constructor Parameters

Inherits all parameters from `ObjectDetectionDataset`, plus:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `keypoint_key` | `str` | `"keypoints"` | CSV column pointing to JSON files with keypoint or polygon annotations. |
| `return_mask` | `bool` | `True` | When `True`, loads and returns per-instance binary masks. |
| `return_keypoints` | `bool` | `False` | When `True`, loads and attaches keypoint data from the JSON file at `keypoint_key`. |

The `mask_key` default is overridden to `"polygon_mask"`.

### `__getitem__` Returns

Same tuple structure as `ObjectDetectionDataset`. When `return_mask=True` the target dict additionally contains:

| Key | Type | Description |
|-----|------|-------------|
| `"masks"` | `torch.Tensor` (uint8) | Per-instance binary mask tensor. |

---

## `PolygonRNNDataset`

```python
from pytorch_segmentation_models_trainer.dataset_loader.dataset import PolygonRNNDataset
```

Dataset for Polygon-RNN training and validation. Reads pre-cropped images and normalized polygon JSON files produced by the [Dataset Conversion](../user-guide/dataset-conversion.md) pipeline.

### Constructor Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `input_csv_path` | `Path` | **required** | Path to the CSV file (typically the output of `PolygonRNNDatasetConversionStrategy`). |
| `sequence_length` | `int` | `60` | Maximum number of polygon vertices in the output sequence. Shorter polygons are padded. |
| `root_dir` | `Any` | `None` | Root directory for path resolution. |
| `augmentation_list` | `Any` | `None` | Albumentations augmentation pipeline. |
| `data_loader` | `Any` | `None` | DataLoader sub-configuration. |
| `image_key` | `str` | `"image"` | CSV column for cropped image paths. |
| `mask_key` | `str` | `"mask"` | CSV column for polygon JSON paths. |
| `image_width_key` | `str` | `None` | CSV column for image width (not used internally but stored for downstream use). |
| `image_height_key` | `str` | `None` | CSV column for image height. |
| `scale_h_key` | `str` | `"scale_h"` | CSV column for vertical scale factor. |
| `scale_w_key` | `str` | `"scale_w"` | CSV column for horizontal scale factor. |
| `min_col_key` | `str` | `"min_col"` | CSV column for the crop left boundary (column). |
| `min_row_key` | `str` | `"min_row"` | CSV column for the crop top boundary (row). |
| `original_image_path_key` | `str` | `"original_image_path"` | CSV column for the source full-image path. |
| `original_polygon_key` | `str` | `"original_polygon_wkt"` | CSV column for the WKT representation of the original polygon. |
| `n_first_rows_to_read` | `str` | `None` | Limit on rows to read. |
| `dataset_type` | `str` | `"train"` | Either `"train"` or `"val"`. Validation mode returns additional metadata fields per item. |
| `grid_size` | `int` | `28` | Grid resolution used for polygon label encoding. |

### `__getitem__` Returns

`Dict[str, Any]` with keys:

| Key | Present when | Type | Description |
|-----|-------------|------|-------------|
| `"x1"` | always | `torch.Tensor` (float32) | Polygon label array at step 2 (first input to RNN). |
| `"x2"` | always | `torch.Tensor` (float32) | Polygon label array from step 0 to `T-2` (main RNN input sequence). |
| `"x3"` | always | `torch.Tensor` (float32) | Polygon label array from step 1 to `T-1` (shifted sequence). |
| `"ta"` | always | `torch.Tensor` (int64) | Target index array from step 2 to `T` (supervised labels). |
| `"image"` | always | `torch.Tensor` (float32) | Cropped image tensor. |
| `"polygon_wkt"` | `dataset_type="val"` | `str` | WKT of the original polygon for metric computation. |
| `"scale_h"` | `dataset_type="val"` | `float` | Vertical scale factor for back-projection. |
| `"scale_w"` | `dataset_type="val"` | `float` | Horizontal scale factor for back-projection. |
| `"min_col"` | `dataset_type="val"` | `float` | Crop left boundary for back-projection. |
| `"min_row"` | `dataset_type="val"` | `float` | Crop top boundary for back-projection. |
| `"original_image_path"` | `dataset_type="val"` | `str` | Source full-image path for visualization. |

---

## `TiledInferenceImageDataset`

```python
from pytorch_segmentation_models_trainer.dataset_loader.dataset import TiledInferenceImageDataset
```

Extends `ImageDataset` for large-image inference. Slices each image into overlapping tiles so that a fixed-input-size model can process arbitrarily large images. Uses `pytorch_toolbelt.inference.tiles.ImageSlicer` internally.

### Constructor Parameters

Inherits all parameters from `ImageDataset`, plus:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `normalize_output` | `bool` | `True` | When `True`, applies `A.Normalize()` (ImageNet mean/std) to each image before tiling. |
| `pad_if_needed` | `bool` | `False` | When `True`, pads the image to the nearest multiple of `model_input_shape` before tiling. Requires `model_input_shape` to be set. |
| `model_input_shape` | `Any` | `None` | Tuple `(H, W)` of the model's expected input tile size. Required when `pad_if_needed=True`. |
| `step_shape` | `Any` | `(224, 224)` | Tile step (stride) for the `ImageSlicer`. Smaller values increase overlap. |

The `augmentation_list` parameter is accepted but ignored; normalization is controlled by `normalize_output` instead.

### `__getitem__` Returns

`Dict[str, Any]` with keys:

| Key | Type | Description |
|-----|------|-------------|
| `"path"` | `str` | File path of the source image. |
| `"tiles"` | `torch.Tensor` (N_tiles, C, H, W) | Stack of all tiles from this image. |
| `"tile_image_idx"` | `torch.Tensor` (N_tiles,) | Image index repeated for each tile (for reassembly). |
| `"tiler_object"` | `ImageSlicer` | Slicer object needed to merge tile predictions back. |
| `"original_shape"` | `tuple` | `(width, height)` of the original image. |

A `collate_fn` static method is provided for use with `DataLoader` to correctly batch variable-tile-count images.

---

## DataLoader Config Keys

All dataset classes accept a `data_loader` sub-configuration. When used with Hydra this is typically a nested config node. The following keys are recognised:

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `shuffle` | `bool` | `True` | Whether to shuffle the dataset each epoch. Typically `True` for training, `False` for validation. |
| `num_workers` | `int` | `0` | Number of worker processes for data loading. |
| `pin_memory` | `bool` | `False` | Whether to pin memory for faster GPU transfer. |
| `batch_size` | `int` | `8` | Number of samples per batch. |
| `drop_last` | `bool` | `False` | Whether to drop the last incomplete batch. |

Example:

```yaml
data_loader:
  shuffle: true
  num_workers: 4
  pin_memory: true
  batch_size: 16
  drop_last: false
```

---

## CSV Format Conventions

Each dataset type expects specific columns in the input CSV:

### `SegmentationDataset` / `FrameFieldSegmentationDataset`

| Column | Required | Description |
|--------|----------|-------------|
| `image` | yes | Path to the input image (RGB or multi-band). |
| `mask` / `polygon_mask` | yes | Path to the ground-truth mask image. |
| `boundary_mask` | `FrameField` only | Path to the boundary mask image. |
| `vertex_mask` | `FrameField` only | Path to the vertex mask image. |
| `crossfield_mask` | `FrameField` only | Path to the crossfield angle image. |
| `distance_mask` | `FrameField` only | Path to the distance transform image. |
| `size_mask` | `FrameField` only | Path to the size map image. |
| `class_freq` | optional | Pre-computed class frequency string to skip per-sample computation. |

### `ObjectDetectionDataset` / `InstanceSegmentationDataset`

| Column | Required | Description |
|--------|----------|-------------|
| `image` | yes | Path to the input image. |
| `bounding_boxes` | yes | Path to a JSON file with a list of `{"bbox": [...], "class": N}` entries. |
| `polygon_mask` | `InstanceSeg` only | Path to the binary mask image. |
| `keypoints` | `InstanceSeg` only | Path to a JSON file with a `"keypoints"` key. |

### `PolygonRNNDataset`

| Column | Required | Description |
|--------|----------|-------------|
| `image` | yes | Path to the cropped object image. |
| `mask` | yes | Path to the normalized polygon JSON file. |
| `scale_h` | yes | Vertical scale factor for coordinate back-projection. |
| `scale_w` | yes | Horizontal scale factor. |
| `min_col` | yes | Left boundary of the original crop. |
| `min_row` | yes | Top boundary of the original crop. |
| `original_image_path` | yes (val) | Path to the source full-size image. |
| `original_polygon_wkt` | yes (val) | WKT of the original polygon. |
