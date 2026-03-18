---
sidebar_position: 3
---

# Building Detection & Instance Segmentation Datasets

This guide covers the `ObjectDetectionDataset` and `InstanceSegmentationDataset` classes, which extend the base CSV-driven dataset system with bounding-box and instance mask support.

## CSV Schema for Object Detection

The detection CSV requires at minimum an `image` column and a `bounding_box` column pointing to a JSON file. A `mask` column is optional for detection-only tasks but required for instance segmentation.

| Column          | Required | Description                                                  |
|-----------------|----------|--------------------------------------------------------------|
| `image`         | Yes      | Path to the input image                                      |
| `bounding_boxes`| Yes      | Path to a JSON file containing bounding box annotations      |
| `mask` / `polygon_mask` | Instance seg only | Path to the instance segmentation mask      |

### Example CSV

```csv
image,bounding_boxes
images/scene_001.tif,bounding_boxes/scene_001.json
images/scene_002.tif,bounding_boxes/scene_002.json
```

For instance segmentation:

```csv
image,polygon_mask,bounding_boxes
images/scene_001.tif,polygon_masks/scene_001.png,bounding_boxes/scene_001.json
images/scene_002.tif,polygon_masks/scene_002.png,bounding_boxes/scene_002.json
```

## Bounding Box JSON Format

Each bounding box JSON file is a JSON **array** of objects. Each object must have:

| Field           | Type     | Description                                          |
|-----------------|----------|------------------------------------------------------|
| `bbox`          | array    | Bounding box coordinates (see format below)          |
| `class`         | integer  | Numeric class / category ID                          |

The `bbox` field holds coordinates in the format specified by the `bbox_format` parameter:

- **`xywh`** (default / COCO format): `[x_min, y_min, width, height]`
- **`xyxy`**: `[x_min, y_min, x_max, y_max]`

### Example JSON File

```json
[
  {
    "bbox": [120, 45, 80, 60],
    "class": 1
  },
  {
    "bbox": [300, 200, 55, 70],
    "class": 1
  },
  {
    "bbox": [500, 10, 100, 120],
    "class": 2
  }
]
```

:::note
The JSON field for the class label is `"class"`, not `"category_id"`. The dataset reads `box_item["class"]` for labels and `box_item["bbox"]` for coordinates.
:::

## The `ObjectDetectionDataset` Class

### Constructor Parameters

| Parameter           | Type     | Default          | Description                                                          |
|---------------------|----------|------------------|----------------------------------------------------------------------|
| `input_csv_path`    | `Path`   | required         | Path to the CSV index file                                           |
| `root_dir`          | `str`    | `None`           | Root directory prepended to relative CSV paths                       |
| `augmentation_list` | `list`   | `None`           | albumentations transforms (must include `bbox_params`)              |
| `data_loader`       | config   | `None`           | DataLoader keyword arguments                                         |
| `image_key`         | `str`    | `"image"`        | CSV column for image paths                                           |
| `mask_key`          | `str`    | `"mask"`         | CSV column for mask paths                                            |
| `bounding_box_key`  | `str`    | `"bounding_boxes"` | CSV column pointing to the bounding box JSON file                 |
| `n_first_rows_to_read` | `int` | `None`          | Limit CSV rows read                                                  |
| `bbox_format`       | `str`    | `"xywh"`         | Format of bboxes stored in the JSON file (`"xywh"` or `"xyxy"`)    |
| `bbox_output_format`| `str`    | `"xyxy"`         | Format of bboxes returned in the output dict                         |
| `bbox_params`       | config   | `None`           | albumentations `BboxParams` config for bbox-aware augmentation      |

### Dataset Output

`__getitem__` returns a **3-tuple**: `(image, ds_item_dict, index)`

| Element        | Type                         | Description                                      |
|----------------|------------------------------|--------------------------------------------------|
| `image`        | `torch.Tensor` `(C, H, W)`   | The input image (loaded as RGB)                  |
| `ds_item_dict` | `dict`                       | Dictionary with `boxes` and `labels` tensors     |
| `index`        | `int`                        | Index of this sample in the dataset              |

Keys in `ds_item_dict`:

| Key      | Shape          | dtype            | Description                                      |
|----------|----------------|------------------|--------------------------------------------------|
| `boxes`  | `(N, 4)`       | `torch.float32`  | Bounding boxes in `bbox_output_format`           |
| `labels` | `(N,)`         | `torch.int64`    | Class index for each box                         |

### Required: Custom `collate_fn`

Because different images have varying numbers of bounding boxes, you **must** use the dataset's built-in `collate_fn` with your `DataLoader`. The default PyTorch collate will fail when tensor shapes differ across samples.

```python
from torch.utils.data import DataLoader
from pytorch_segmentation_models_trainer.dataset_loader.dataset import ObjectDetectionDataset

dataset = ObjectDetectionDataset(input_csv_path="train.csv")
loader = DataLoader(
    dataset,
    batch_size=4,
    collate_fn=ObjectDetectionDataset.collate_fn,
)
```

The `collate_fn` stacks images into a single tensor and returns a list of per-image target dictionaries:

```
(images_tensor [B, C, H, W], List[dict], indexes_tensor [B])
```

### YAML Configuration Example

```yaml
# configs/dataset/detection_train.yaml

train_dataset:
  _target_: pytorch_segmentation_models_trainer.dataset_loader.dataset.ObjectDetectionDataset
  input_csv_path: /data/detection/train.csv
  root_dir: /data/detection
  bounding_box_key: bounding_boxes
  bbox_format: xywh          # format stored in the JSON files
  bbox_output_format: xyxy   # format returned by __getitem__
  augmentation_list:
    - _target_: albumentations.RandomCrop
      height: 512
      width: 512
    - _target_: albumentations.HorizontalFlip
      p: 0.5
    - _target_: albumentations.Normalize
      mean: [0.485, 0.456, 0.406]
      std: [0.229, 0.224, 0.225]
    - _target_: albumentations.pytorch.ToTensorV2
  bbox_params:
    format: coco              # albumentations bbox format string
    label_fields: [labels]
    min_visibility: 0.1
  data_loader:
    shuffle: true
    num_workers: 4
    pin_memory: true
    batch_size: 4
    drop_last: true
```

## The `InstanceSegmentationDataset` Class

`InstanceSegmentationDataset` extends `ObjectDetectionDataset` by additionally loading a per-instance segmentation mask and optionally keypoints.

### Additional Constructor Parameters

| Parameter          | Type   | Default           | Description                                              |
|--------------------|--------|-------------------|----------------------------------------------------------|
| `mask_key`         | `str`  | `"polygon_mask"`  | CSV column for instance mask paths                       |
| `return_mask`      | `bool` | `True`            | Whether to load and return the mask                      |
| `keypoint_key`     | `str`  | `"keypoints"`     | CSV column pointing to a keypoints JSON file             |
| `return_keypoints` | `bool` | `False`           | Whether to load and return keypoints                     |

### Dataset Output

The output is the same 3-tuple as `ObjectDetectionDataset`, with `ds_item_dict` gaining an additional key:

| Key        | Shape              | dtype            | Description                               |
|------------|--------------------|------------------|-------------------------------------------|
| `boxes`    | `(N, 4)`           | `torch.float32`  | Bounding boxes                            |
| `labels`   | `(N,)`             | `torch.int64`    | Class labels                              |
| `masks`    | `(1, H, W)`        | `torch.uint8`    | Binary instance mask (if `return_mask`)   |

### YAML Configuration Example

```yaml
# configs/dataset/instance_seg_train.yaml

train_dataset:
  _target_: pytorch_segmentation_models_trainer.dataset_loader.dataset.InstanceSegmentationDataset
  input_csv_path: /data/instance_seg/train.csv
  root_dir: /data/instance_seg
  mask_key: polygon_mask
  bounding_box_key: bounding_boxes
  bbox_format: xywh
  bbox_output_format: xyxy
  return_mask: true
  return_keypoints: false
  augmentation_list:
    - _target_: albumentations.RandomCrop
      height: 512
      width: 512
    - _target_: albumentations.HorizontalFlip
      p: 0.5
    - _target_: albumentations.Normalize
      mean: [0.485, 0.456, 0.406]
      std: [0.229, 0.224, 0.225]
    - _target_: albumentations.pytorch.ToTensorV2
  bbox_params:
    format: coco
    label_fields: [labels]
    min_visibility: 0.1
  data_loader:
    shuffle: true
    num_workers: 4
    pin_memory: true
    batch_size: 4
    drop_last: true
```

## Augmentation Notes

Because bounding boxes must be transformed consistently with the image, the `bbox_params` config is passed to `albumentations.Compose` when building the augmentation pipeline. Without this, augmentations like `RandomCrop` or `HorizontalFlip` would not update box coordinates.

Key `BboxParams` fields:

| Field          | Description                                                              |
|----------------|--------------------------------------------------------------------------|
| `format`       | Albumentations bbox string: `"coco"` (xywh), `"pascal_voc"` (xyxy), etc.|
| `label_fields` | List of field names in the transform dict that hold class labels         |
| `min_visibility`| Minimum fraction of box area that must remain after a crop              |

:::tip
Use `format: coco` in `bbox_params` when your JSON files store `xywh` coordinates — this matches albumentations' COCO format string, which is equivalent to `xywh`.
:::
