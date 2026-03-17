---
sidebar_position: 5
title: Object Detection with Faster R-CNN
---

# Object Detection with Faster R-CNN

A complete example showing how to train a Faster R-CNN model to detect buildings or vehicles in aerial imagery, using bounding box annotations stored in JSON files.

## Use Case

Object detection predicts axis-aligned bounding boxes and class labels for every object instance in an image, rather than producing a pixel-level mask. This is well-suited for:

- Detecting individual buildings in satellite imagery
- Vehicle detection and counting in aerial surveillance
- Infrastructure inspection (solar panels, swimming pools, etc.)

This example uses `fasterrcnn_resnet50_fpn` from `torchvision.models.detection`, which supports variable numbers of objects per image.

:::note Detection vs Segmentation
Unlike `SegmentationDataset`, the `ObjectDetectionDataset` returns a tuple `(image, targets_dict, index)` where `targets_dict` contains `boxes` and `labels` tensors. The `ObjectDetectionPLModel` passes these directly to the Faster R-CNN model, which computes its own internal losses.
:::

## Project Structure

```
detection_project/
├── data/
│   ├── train/
│   │   ├── images/        # RGB images (JPEG or PNG)
│   │   └── annotations/   # JSON files: one per image, list of {bbox, class}
│   └── val/
│       ├── images/
│       └── annotations/
├── configs/
│   └── train.yaml
├── train.csv
├── val.csv
└── outputs/
```

## Step 1: Prepare Bounding Box Annotations

### Annotation JSON Format

Each image has a corresponding JSON file containing a list of bounding boxes. The `bbox` field uses **XYWH format** (`x_min, y_min, width, height`) in pixel coordinates. The `class` field is a 1-based integer (class `0` is reserved for the background by Faster R-CNN).

`data/train/annotations/tile_001.json`:
```json
[
  {"bbox": [120, 85, 45, 60], "class": 1},
  {"bbox": [230, 140, 38, 52], "class": 1},
  {"bbox": [310, 200, 72, 80], "class": 2}
]
```

:::warning Class Indices Must Be 1-Based
Faster R-CNN internally reserves class `0` for the background. Your annotation `class` values must start from `1`. If your annotations use 0-based indexing, add 1 to all class values before saving.
:::

### Convert from COCO JSON

```python
import json
from pathlib import Path

def coco_to_per_image_json(coco_json_path, output_dir):
    """Convert a single COCO-format JSON to one JSON file per image."""
    with open(coco_json_path) as f:
        coco = json.load(f)

    # Build image ID -> filename map
    id_to_filename = {img["id"]: img["file_name"] for img in coco["images"]}

    # Group annotations by image
    from collections import defaultdict
    image_annotations = defaultdict(list)
    for ann in coco["annotations"]:
        image_annotations[ann["image_id"]].append({
            "bbox": ann["bbox"],          # Already XYWH in COCO format
            "class": ann["category_id"],  # Must be >= 1
        })

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    for image_id, annotations in image_annotations.items():
        stem = Path(id_to_filename[image_id]).stem
        output_path = Path(output_dir) / f"{stem}.json"
        with open(output_path, "w") as f:
            json.dump(annotations, f)

    print(f"Wrote {len(image_annotations)} annotation files to {output_dir}")

coco_to_per_image_json("data/annotations/coco_train.json", "data/train/annotations")
coco_to_per_image_json("data/annotations/coco_val.json",   "data/val/annotations")
```

### Create CSV Files

The CSV requires an `image` column and a `bounding_boxes` column pointing to the annotation JSON file for each image.

Create `train.csv`:
```csv
image,bounding_boxes
data/train/images/tile_001.png,data/train/annotations/tile_001.json
data/train/images/tile_002.png,data/train/annotations/tile_002.json
data/train/images/tile_003.png,data/train/annotations/tile_003.json
```

Create `val.csv`:
```csv
image,bounding_boxes
data/val/images/tile_101.png,data/val/annotations/tile_101.json
data/val/images/tile_102.png,data/val/annotations/tile_102.json
```

:::tip Generating CSVs Automatically

```python
import pandas as pd
from pathlib import Path

def create_detection_csv(images_dir, annotations_dir, output_csv, img_ext="png"):
    rows = []
    for img_file in sorted(Path(images_dir).glob(f"*.{img_ext}")):
        ann_file = Path(annotations_dir) / f"{img_file.stem}.json"
        if ann_file.exists():
            rows.append({
                "image": str(img_file),
                "bounding_boxes": str(ann_file),
            })
    df = pd.DataFrame(rows)
    df.to_csv(output_csv, index=False)
    print(f"Created {output_csv} with {len(df)} samples")

create_detection_csv("data/train/images", "data/train/annotations", "train.csv")
create_detection_csv("data/val/images",   "data/val/annotations",   "val.csv")
```
:::

## Step 2: Training Configuration

Create `configs/train.yaml`:

```yaml
# --- Model Architecture ---
# fasterrcnn_resnet50_fpn from torchvision with a custom number of classes.
# num_classes includes the background: for 2 object classes set num_classes: 3.
model:
  _target_: torchvision.models.detection.fasterrcnn_resnet50_fpn
  weights: DEFAULT          # Loads COCO pretrained weights
  # Override the box predictor head for our number of classes:
  # This is handled automatically by torchvision when passing num_classes
  # to fasterrcnn_resnet50_fpn via box_predictor replacement in a wrapper.
  # See the note below for the recommended approach.

# --- Training Dataset ---
train_dataset:
  _target_: pytorch_segmentation_models_trainer.dataset_loader.dataset.ObjectDetectionDataset
  input_csv_path: train.csv
  bbox_format: xywh          # Input annotation format (x_min, y_min, width, height)
  bbox_output_format: xyxy   # Faster R-CNN expects (x_min, y_min, x_max, y_max)
  bounding_box_key: bounding_boxes   # CSV column name for annotation JSON paths
  data_loader:
    shuffle: true
    num_workers: 4
    pin_memory: true
    drop_last: true
    prefetch_factor: 2
  augmentation_list:
    - _target_: albumentations.HorizontalFlip
      p: 0.5
    - _target_: albumentations.VerticalFlip
      p: 0.3
    - _target_: albumentations.RandomBrightnessContrast
      brightness_limit: 0.2
      contrast_limit: 0.2
      p: 0.4
    - _target_: albumentations.Normalize
      mean: [0.485, 0.456, 0.406]
      std: [0.229, 0.224, 0.225]
      p: 1.0
    - _target_: albumentations.pytorch.transforms.ToTensorV2
      always_apply: true
  bbox_params:
    format: pascal_voc        # Albumentations uses pascal_voc (xyxy) internally
    min_visibility: 0.3       # Drop boxes with less than 30% visibility after crop
    label_fields: [labels]

# --- Validation Dataset ---
val_dataset:
  _target_: pytorch_segmentation_models_trainer.dataset_loader.dataset.ObjectDetectionDataset
  input_csv_path: val.csv
  bbox_format: xywh
  bbox_output_format: xyxy
  bounding_box_key: bounding_boxes
  data_loader:
    shuffle: false
    num_workers: 4
    pin_memory: true
    drop_last: false
    prefetch_factor: 2
  augmentation_list:
    - _target_: albumentations.Normalize
      mean: [0.485, 0.456, 0.406]
      std: [0.229, 0.224, 0.225]
      p: 1.0
    - _target_: albumentations.pytorch.transforms.ToTensorV2
      always_apply: true
  bbox_params:
    format: pascal_voc
    min_visibility: 0.3
    label_fields: [labels]

# --- Optimizer ---
# ObjectDetectionPLModel computes loss internally via the torchvision model.
# No loss config is needed here.
optimizer:
  _target_: torch.optim.SGD
  lr: 0.005
  momentum: 0.9
  weight_decay: 5.0e-4

# --- Learning Rate Scheduler ---
scheduler_list:
  - scheduler:
      _target_: torch.optim.lr_scheduler.StepLR
      step_size: 5
      gamma: 0.1
    interval: epoch
    frequency: 1
    name: step_lr

# --- Hyperparameters ---
hyperparameters:
  batch_size: 4     # Detection models are memory-intensive; keep batch size small
  epochs: 30

# --- PyTorch Lightning Trainer ---
pl_trainer:
  max_epochs: ${hyperparameters.epochs}
  accelerator: gpu
  devices: 1
  # Note: mixed precision (precision: 16-mixed) may cause instability with
  # some torchvision detection models. Use 32-true if you encounter NaN losses.
  precision: 32-true
  gradient_clip_val: 5.0
  gradient_clip_algorithm: norm
  check_val_every_n_epoch: 1
  log_every_n_steps: 10

# --- Callbacks ---
callbacks:
  - _target_: pytorch_lightning.callbacks.ModelCheckpoint
    monitor: loss/val
    mode: min
    save_top_k: 3
    save_last: true
    filename: "best-{epoch:02d}-{loss/val:.4f}"
    auto_insert_metric_name: false
  - _target_: pytorch_lightning.callbacks.EarlyStopping
    monitor: loss/val
    mode: min
    patience: 10
    min_delta: 0.001
  - _target_: pytorch_lightning.callbacks.LearningRateMonitor
    logging_interval: epoch

# --- Logger ---
logger:
  _target_: pytorch_lightning.loggers.TensorBoardLogger
  save_dir: ./logs
  name: object_detection

mode: train
device: cuda
```

:::note collate_fn is Handled Automatically
`ObjectDetectionPLModel` overrides `train_dataloader()` and `val_dataloader()` to automatically pass `ObjectDetectionDataset.collate_fn` to the `DataLoader`. This custom collate function stacks image tensors while keeping per-image target dicts as a Python list — exactly the format required by torchvision detection models. You do not need to configure this manually.
:::

:::tip Customizing the Number of Classes
Torchvision's `fasterrcnn_resnet50_fpn` is pretrained on COCO (91 classes). To fine-tune it for your own class set, replace the box predictor head:

```python
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

NUM_CLASSES = 3  # background + 2 object classes

model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights="DEFAULT")
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, NUM_CLASSES)
```

Wrap this in a Hydra-compatible factory function or use a `_target_` pointing to your custom factory.
:::

## Step 3: Run Training

```bash
cd detection_project
pytorch-smt --config-dir ./configs --config-name train
```

## Step 4: Training Metrics

`ObjectDetectionPLModel` logs the following metrics during training and validation:

| Metric key                          | Description                                      |
|-------------------------------------|--------------------------------------------------|
| `loss/train`                        | Total training loss (sum of all RPN + head losses) |
| `loss/val`                          | Total validation loss                            |
| `losses/train_loss_rpn_box_reg`     | RPN bounding box regression loss                |
| `losses/train_loss_objectness`      | RPN objectness (foreground/background) loss      |
| `losses/train_loss_classifier`      | ROI head classification loss                     |
| `losses/train_loss_box_reg`         | ROI head bounding box regression loss            |
| `metrics/val_iou`                   | Mean IoU between predicted and ground truth boxes |

All `losses/train_*` metrics are also available as `losses/val_*` for the validation set.

## Step 5: Inference

`ObjectDetectionInferenceProcessor` tiles large images, runs detection on each tile, and merges overlapping bounding box predictions using NMS (non-maximum suppression).

Create `configs/predict.yaml`:

```yaml
model:
  _target_: torchvision.models.detection.fasterrcnn_resnet50_fpn
  weights: null   # Weights loaded from checkpoint

mode: predict
device: cuda
checkpoint_path: ./logs/object_detection/version_0/checkpoints/best-epoch=XX-loss_val=X.XXXX.ckpt

inference_image_reader:
  _target_: pytorch_segmentation_models_trainer.tools.data_handlers.raster_reader.FolderImageReaderProcessor
  folder_name: ./data/test/images
  recursive: true
  image_extension: png

inference_processor:
  _target_: pytorch_segmentation_models_trainer.tools.inference.inference_processors.ObjectDetectionInferenceProcessor
  model_input_shape: [512, 512]
  step_shape: [256, 256]
  post_process_method: union   # Method for merging overlapping boxes across tiles
  min_visibility: 0.3          # Minimum fraction of box visible within a tile

export_strategy:
  _target_: pytorch_segmentation_models_trainer.tools.inference.export_inference.VectorExportInferenceStrategy
  output_file_path: ./predictions/{input_name}_detections.geojson

inference_threshold: 0.5   # Detection score threshold
save_inference: true
```

```bash
pytorch-smt --config-dir ./configs --config-name predict
```

### Load Results Programmatically

After running `pytorch-smt` with `mode: predict`, you can also load a checkpoint and run inference directly in Python:

```python
import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from pytorch_segmentation_models_trainer.model_loader.detection_model import ObjectDetectionPLModel
from omegaconf import OmegaConf
from PIL import Image
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2

# Load model from checkpoint
cfg = OmegaConf.load("configs/train.yaml")
model = ObjectDetectionPLModel.load_from_checkpoint(
    "logs/object_detection/version_0/checkpoints/best-epoch=05-loss_val=0.4321.ckpt",
    cfg=cfg,
)
model.eval()
model.to("cuda")

# Prepare a single image
transform = A.Compose([
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2(),
])
image = np.array(Image.open("data/test/images/tile_200.png").convert("RGB"))
tensor = transform(image=image)["image"].unsqueeze(0).float().to("cuda")

# Run inference
with torch.no_grad():
    predictions = model.model(tensor)

# predictions is a list of dicts, one per image in the batch
pred = predictions[0]
print(f"Detected {len(pred['boxes'])} objects")
for box, label, score in zip(pred["boxes"], pred["labels"], pred["scores"]):
    if score >= 0.5:
        x1, y1, x2, y2 = box.tolist()
        print(f"  Class {label.item()} | Score {score:.3f} | Box [{x1:.0f}, {y1:.0f}, {x2:.0f}, {y2:.0f}]")
```

## Next Steps

- Try `fasterrcnn_mobilenet_v3_large_fpn` for faster inference at slightly lower accuracy
- Use `InstanceSegmentationPLModel` with `maskrcnn_resnet50_fpn` to obtain both boxes and instance masks
- Tune `min_visibility` and `post_process_method` to reduce missed detections near tile boundaries
- Increase image resolution in `model_input_shape` if small objects are being missed
