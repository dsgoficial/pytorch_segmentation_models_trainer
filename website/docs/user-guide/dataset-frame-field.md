---
sidebar_position: 2
---

# Building a Frame Field Dataset

Frame field models (such as the Frame Field Learning approach for polygon extraction) require additional auxiliary masks beyond a standard polygon segmentation mask. This guide explains the extended CSV schema, the `FrameFieldSegmentationDataset` class, and how to configure it.

## What Extra Masks Are Needed

Standard segmentation only requires a polygon mask. Frame field models also learn:

| Mask type        | What it encodes                                                        |
|------------------|------------------------------------------------------------------------|
| `boundary_mask`  | Pixels belonging to polygon edges / boundaries                        |
| `vertex_mask`    | Pixels at polygon corners / vertices                                   |
| `crossfield_mask`| A two-channel angle map encoding the dominant polygon edge orientation |
| `distance_mask`  | Per-pixel distance transform from the nearest polygon edge             |
| `size_mask`      | Per-pixel polygon-size normalisation map                               |

All of these can be generated automatically from vector polygon data using the `build-mask` CLI mode. See the [Building Training Masks from Vector Data](./mask-building.md) guide.

## Extended CSV Schema

The CSV index file extends the base schema with a column for each auxiliary mask. Any column whose corresponding `return_*` flag is `false` may be omitted.

| Column           | Description                                            |
|------------------|--------------------------------------------------------|
| `image`          | Path to the input image                                |
| `polygon_mask`   | Path to the polygon segmentation mask (default mask key) |
| `boundary_mask`  | Path to the boundary mask                              |
| `vertex_mask`    | Path to the vertex mask                                |
| `crossfield_mask`| Path to the crossfield angle map                       |
| `distance_mask`  | Path to the distance transform mask                    |
| `size_mask`      | Path to the size normalisation mask                    |

### Example CSV

```csv
image,polygon_mask,boundary_mask,vertex_mask,crossfield_mask,distance_mask,size_mask
images/tile_001.tif,polygon_masks/tile_001.png,boundary_masks/tile_001.png,vertex_masks/tile_001.png,crossfield_masks/tile_001.png,distance_masks/tile_001.png,size_masks/tile_001.png
images/tile_002.tif,polygon_masks/tile_002.png,boundary_masks/tile_002.png,vertex_masks/tile_002.png,crossfield_masks/tile_002.png,distance_masks/tile_002.png,size_masks/tile_002.png
```

:::tip
The `build-mask` CLI generates all of these mask files and writes the CSV in one step. You rarely need to populate this CSV manually.
:::

## The `FrameFieldSegmentationDataset` Class

`FrameFieldSegmentationDataset` extends `SegmentationDataset` with logic for loading multiple mask channels and assembling the structured output dictionary expected by frame field model trainers.

### Constructor Parameters

| Parameter              | Type        | Default              | Description                                                              |
|------------------------|-------------|----------------------|--------------------------------------------------------------------------|
| `input_csv_path`       | `Path`      | required             | Path to the CSV index file                                               |
| `root_dir`             | `str`       | `None`               | Root directory for relative paths                                        |
| `augmentation_list`    | `list`      | `None`               | albumentations transform list                                            |
| `data_loader`          | config      | `None`               | DataLoader keyword arguments                                             |
| `image_key`            | `str`       | `"image"`            | CSV column name for images                                               |
| `mask_key`             | `str`       | `"polygon_mask"`     | CSV column name for polygon masks                                        |
| `multi_band_mask`      | `bool`      | `False`              | If `True`, load polygon/boundary/vertex from a single 3-band mask file  |
| `boundary_mask_key`    | `str`       | `"boundary_mask"`    | CSV column name for boundary masks                                       |
| `return_boundary_mask` | `bool`      | `True`               | Whether to load and return the boundary mask                             |
| `vertex_mask_key`      | `str`       | `"vertex_mask"`      | CSV column name for vertex masks                                         |
| `return_vertex_mask`   | `bool`      | `True`               | Whether to load and return the vertex mask                               |
| `crossfield_mask_key`  | `str`       | `"crossfield_mask"`  | CSV column name for the crossfield angle map                             |
| `return_crossfield_mask`| `bool`     | `True`               | Whether to load and return the crossfield mask                           |
| `distance_mask_key`    | `str`       | `"distance_mask"`    | CSV column name for the distance transform                               |
| `return_distance_mask` | `bool`      | `True`               | Whether to load and return the distance mask                             |
| `size_mask_key`        | `str`       | `"size_mask"`        | CSV column name for the size mask                                        |
| `return_size_mask`     | `bool`      | `True`               | Whether to load and return the size mask                                 |
| `image_width`          | `int`       | `224`                | Width used by the fallback resize transform when a crop yields no objects|
| `image_height`         | `int`       | `224`                | Height used by the fallback resize transform                             |
| `n_first_rows_to_read` | `int`       | `None`               | Limit CSV rows read                                                      |

### The `multi_band_mask` Option

When `multi_band_mask: true`, the dataset expects the `mask_key` column to point to a single multi-band image whose:

- Band 0 = polygon mask
- Band 1 = boundary mask
- Band 2 = vertex mask

This can reduce the number of files on disk at the cost of more complex mask preparation.

### Fallback Resize Transform

If an augmented crop contains no object pixels (all-background), the dataset automatically falls back to a resize-based transform (`albumentations.Resize` to `image_height × image_width` followed by `Normalize` and `ToTensorV2`) to guarantee that the returned sample has valid content.

## Dataset Output Dictionary

`__getitem__` returns a dictionary with the following keys:

| Key                  | Shape              | dtype           | Description                                              |
|----------------------|--------------------|-----------------|----------------------------------------------------------|
| `idx`                | scalar             | —               | Sample index                                             |
| `path`               | `str`              | —               | Absolute path to the source image                        |
| `image`              | `(C, H, W)`        | `torch.Tensor`  | Image tensor                                             |
| `gt_polygons_image`  | `(3, H, W)`        | `torch.float32` | Stacked polygon / boundary / vertex masks                |
| `class_freq`         | `(3,)`             | `torch.float32` | Per-class mean pixel frequency                           |
| `gt_crossfield_angle`| `(1, H, W)`        | `torch.float32` | Crossfield angle map (only if `return_crossfield_mask`)  |
| `distances`          | `(1, H, W)`        | `torch.float32` | Distance transform (only if `return_distance_mask`)      |
| `sizes`              | `(1, H, W)`        | `torch.float32` | Size map (only if `return_size_mask`)                    |

:::note
The crossfield, distance, and size masks are loaded as raw floating-point arrays (not binarised), unlike the polygon/boundary/vertex masks which are loaded as binary uint8 arrays.
:::

## Full YAML Configuration Example

```yaml
# configs/dataset/frame_field_train.yaml

train_dataset:
  _target_: pytorch_segmentation_models_trainer.dataset_loader.dataset.FrameFieldSegmentationDataset
  input_csv_path: /data/buildings/train.csv
  root_dir: /data/buildings
  # Mask column names (must match CSV header)
  mask_key: polygon_mask
  boundary_mask_key: boundary_mask
  return_boundary_mask: true
  vertex_mask_key: vertex_mask
  return_vertex_mask: true
  crossfield_mask_key: crossfield_mask
  return_crossfield_mask: true
  distance_mask_key: distance_mask
  return_distance_mask: true
  size_mask_key: size_mask
  return_size_mask: true
  # Fallback resize dimensions when a crop has no objects
  image_width: 512
  image_height: 512
  multi_band_mask: false
  augmentation_list:
    - _target_: albumentations.RandomCrop
      height: 512
      width: 512
    - _target_: albumentations.HorizontalFlip
      p: 0.5
    - _target_: albumentations.VerticalFlip
      p: 0.5
    - _target_: albumentations.Normalize
      mean: [0.485, 0.456, 0.406]
      std: [0.229, 0.224, 0.225]
    - _target_: albumentations.pytorch.ToTensorV2
  data_loader:
    shuffle: true
    num_workers: 4
    pin_memory: true
    batch_size: 4
    drop_last: true

val_dataset:
  _target_: pytorch_segmentation_models_trainer.dataset_loader.dataset.FrameFieldSegmentationDataset
  input_csv_path: /data/buildings/val.csv
  root_dir: /data/buildings
  mask_key: polygon_mask
  boundary_mask_key: boundary_mask
  return_boundary_mask: true
  vertex_mask_key: vertex_mask
  return_vertex_mask: true
  crossfield_mask_key: crossfield_mask
  return_crossfield_mask: true
  distance_mask_key: distance_mask
  return_distance_mask: true
  size_mask_key: size_mask
  return_size_mask: true
  image_width: 512
  image_height: 512
  multi_band_mask: false
  augmentation_list:
    - _target_: albumentations.Resize
      height: 512
      width: 512
    - _target_: albumentations.Normalize
      mean: [0.485, 0.456, 0.406]
      std: [0.229, 0.224, 0.225]
    - _target_: albumentations.pytorch.ToTensorV2
  data_loader:
    shuffle: false
    num_workers: 4
    pin_memory: true
    batch_size: 4
    drop_last: false
```

### Disabling Specific Mask Types

If your model does not use distance or size masks, you can disable them to save I/O:

```yaml
return_distance_mask: false
return_size_mask: false
```

The corresponding keys (`distances`, `sizes`) will then be absent from the output dictionary.

## Generating the Masks

All auxiliary masks are generated from polygon vector data using the `build-mask` mode. Refer to the [Building Training Masks from Vector Data](./mask-building.md) guide for the complete workflow.
