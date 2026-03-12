---
sidebar_position: 7
title: Training Object Detection & Instance Segmentation Models
---

# Training Object Detection & Instance Segmentation Models

This guide covers training object detection and instance segmentation models using `ObjectDetectionPLModel` and `InstanceSegmentationPLModel`. Both classes wrap `torchvision.models.detection` architectures inside PyTorch Lightning using the same Hydra config system.

---

## Model Classes

### `ObjectDetectionPLModel`

Trains bounding-box detection models (Faster R-CNN, RetinaNet).

```yaml
pl_model:
  _target_: pytorch_segmentation_models_trainer.model_loader.detection_model.ObjectDetectionPLModel
```

### `InstanceSegmentationPLModel`

Trains models that predict both bounding boxes and instance masks (Mask R-CNN). Inherits all behaviour from `ObjectDetectionPLModel`.

```yaml
pl_model:
  _target_: pytorch_segmentation_models_trainer.model_loader.detection_model.InstanceSegmentationPLModel
```

:::note
Both classes override `get_loss_function` to return `None`. Loss computation is handled internally by the `torchvision` detection models themselves, which return a dict of loss components from their `forward` method during training.
:::

---

## Supported Architectures

Any detection model from `torchvision.models.detection` can be used:

| Architecture | `_target_` | Task |
|---|---|---|
| Faster R-CNN | `torchvision.models.detection.fasterrcnn_resnet50_fpn` | Object detection |
| RetinaNet | `torchvision.models.detection.retinanet_resnet50_fpn` | Object detection |
| Mask R-CNN | `torchvision.models.detection.maskrcnn_resnet50_fpn` | Instance segmentation |

```yaml
# Faster R-CNN example
model:
  _target_: torchvision.models.detection.fasterrcnn_resnet50_fpn
  weights: DEFAULT
  num_classes: 2   # background + 1 object class

# Mask R-CNN example
model:
  _target_: torchvision.models.detection.maskrcnn_resnet50_fpn
  weights: DEFAULT
  num_classes: 2
```

---

## Dataset Format

### Dataset Classes

Use `ObjectDetectionDataset` for bounding-box detection and `InstanceSegmentationDataset` for Mask R-CNN training:

```yaml
train_dataset:
  _target_: pytorch_segmentation_models_trainer.dataset_loader.dataset.ObjectDetectionDataset
  input_csv_path: /data/detection/train.csv
  root_dir: /data/detection
  data_loader:
    shuffle: true
    num_workers: 4
    pin_memory: true
    drop_last: true
    prefetch_factor: 2
```

### Custom `collate_fn`

Detection datasets use a custom `collate_fn` that handles variable-length target lists (each image has a different number of objects). The `ObjectDetectionPLModel` DataLoaders pick this up automatically:

```python
# Applied automatically — no manual configuration needed
collate_fn=self.train_ds.collate_fn
```

### Batch Format

Each batch is a 3-tuple:

```python
images, targets, indices = batch
```

- `images` — list of `torch.Tensor` of shape `(C, H, W)`, one per image
- `targets` — list of dicts, one per image, each containing:
  - `boxes` — `torch.Tensor` of shape `(N, 4)` in `(x1, y1, x2, y2)` (xyxy) format
  - `labels` — `torch.Tensor` of shape `(N,)` with integer class IDs
  - `masks` — *(instance segmentation only)* `torch.Tensor` of shape `(N, H, W)` with binary masks per instance
- `indices` — list of sample indices from the dataset (used for identification)

---

## Training Behaviour

### Loss Computation

The `torchvision` detection models return a dict of losses during training mode and a list of predictions during eval mode. `ObjectDetectionPLModel` exploits this split to log both losses and IoU:

```python
# training_step
images, targets, _ = batch
loss_dict = self.model(images, targets)   # returns dict of loss components
total_loss = sum(loss for loss in loss_dict.values())
```

```python
# validation_step: switch modes within the same step
self.model.train()
loss_dict = self.model(images, targets)   # compute losses

self.model.eval()
outs = self.model(images)                 # compute predictions for IoU
iou = mean(evaluate_box_iou(t, o) for t, o in zip(targets, outs))
```

### Loss Components

The individual loss components logged depend on the architecture:

**Faster R-CNN:**

| Metric key | Description |
|---|---|
| `losses/train_loss_rpn_box_reg` | RPN bounding box regression |
| `losses/train_loss_objectness` | RPN objectness score |
| `losses/train_loss_box_reg` | ROI head box regression |
| `losses/train_loss_classifier` | ROI head classification |

**Mask R-CNN** adds:

| Metric key | Description |
|---|---|
| `losses/train_loss_mask` | Mask head binary cross-entropy |

**Validation metrics:**

| Metric key | Description |
|---|---|
| `loss/val` | Total validation loss |
| `metrics/val_iou` | Mean IoU across images and predictions |

---

## Full Config: Object Detection (Faster R-CNN)

```yaml title="configs/train_faster_rcnn.yaml"
# ── Model selection ──────────────────────────────────────────────────────────
pl_model:
  _target_: pytorch_segmentation_models_trainer.model_loader.detection_model.ObjectDetectionPLModel

# ── Architecture ─────────────────────────────────────────────────────────────
model:
  _target_: torchvision.models.detection.fasterrcnn_resnet50_fpn
  weights: DEFAULT
  num_classes: 2   # background (0) + building (1)

# ── Optimizer ────────────────────────────────────────────────────────────────
optimizer:
  _target_: torch.optim.SGD
  lr: 0.005
  momentum: 0.9
  weight_decay: 5e-4

# ── LR Scheduler ─────────────────────────────────────────────────────────────
scheduler_list:
  - scheduler:
      _target_: torch.optim.lr_scheduler.StepLR
      step_size: 3
      gamma: 0.1
    interval: epoch
    name: step_lr

# ── Hyperparameters ──────────────────────────────────────────────────────────
hyperparameters:
  batch_size: 4

# ── PyTorch Lightning Trainer ────────────────────────────────────────────────
pl_trainer:
  max_epochs: 20
  accelerator: gpu
  devices: 1
  log_every_n_steps: 25

# ── Callbacks ────────────────────────────────────────────────────────────────
callbacks:
  - _target_: pytorch_lightning.callbacks.ModelCheckpoint
    monitor: loss/val
    mode: min
    save_top_k: 3
    save_last: true
    filename: "faster-rcnn-{epoch:02d}-{loss/val:.4f}"
    dirpath: ./checkpoints/detection

  - _target_: pytorch_lightning.callbacks.EarlyStopping
    monitor: metrics/val_iou
    mode: max
    patience: 5

  - _target_: pytorch_lightning.callbacks.LearningRateMonitor
    logging_interval: epoch

# ── Logger ───────────────────────────────────────────────────────────────────
logger:
  _target_: pytorch_lightning.loggers.TensorBoardLogger
  save_dir: ./experiments
  name: faster_rcnn_buildings

# ── Datasets ─────────────────────────────────────────────────────────────────
train_dataset:
  _target_: pytorch_segmentation_models_trainer.dataset_loader.dataset.ObjectDetectionDataset
  input_csv_path: /data/buildings/train.csv
  root_dir: /data/buildings
  data_loader:
    shuffle: true
    num_workers: 4
    pin_memory: true
    drop_last: true
    prefetch_factor: 2

val_dataset:
  _target_: pytorch_segmentation_models_trainer.dataset_loader.dataset.ObjectDetectionDataset
  input_csv_path: /data/buildings/val.csv
  root_dir: /data/buildings
  data_loader:
    shuffle: false
    num_workers: 4
    pin_memory: true
    drop_last: false
    prefetch_factor: 2

mode: train
device: cuda
```

---

## Full Config: Instance Segmentation (Mask R-CNN)

The instance segmentation config is identical to the detection config except for the `pl_model` and `model` keys, and the dataset class:

```yaml title="configs/train_mask_rcnn.yaml"
# ── Model selection ──────────────────────────────────────────────────────────
pl_model:
  _target_: pytorch_segmentation_models_trainer.model_loader.detection_model.InstanceSegmentationPLModel

# ── Architecture ─────────────────────────────────────────────────────────────
model:
  _target_: torchvision.models.detection.maskrcnn_resnet50_fpn
  weights: DEFAULT
  num_classes: 2

# ── Optimizer ────────────────────────────────────────────────────────────────
optimizer:
  _target_: torch.optim.AdamW
  lr: 0.001
  weight_decay: 1e-4

# ── LR Scheduler ─────────────────────────────────────────────────────────────
scheduler_list:
  - scheduler:
      _target_: torch.optim.lr_scheduler.CosineAnnealingLR
      T_max: 30
      eta_min: 1e-6
    interval: epoch
    name: cosine_lr

# ── Hyperparameters ──────────────────────────────────────────────────────────
hyperparameters:
  batch_size: 2   # Mask R-CNN uses more memory per sample

# ── PyTorch Lightning Trainer ────────────────────────────────────────────────
pl_trainer:
  max_epochs: 30
  accelerator: gpu
  devices: 1
  precision: 16
  log_every_n_steps: 25

# ── Callbacks ────────────────────────────────────────────────────────────────
callbacks:
  - _target_: pytorch_lightning.callbacks.ModelCheckpoint
    monitor: metrics/val_iou
    mode: max
    save_top_k: 3
    save_last: true
    filename: "mask-rcnn-{epoch:02d}-iou={metrics/val_iou:.4f}"
    dirpath: ./checkpoints/instance_seg

  - _target_: pytorch_lightning.callbacks.LearningRateMonitor
    logging_interval: epoch

# ── Logger ───────────────────────────────────────────────────────────────────
logger:
  _target_: pytorch_lightning.loggers.TensorBoardLogger
  save_dir: ./experiments
  name: mask_rcnn_buildings

# ── Datasets ─────────────────────────────────────────────────────────────────
train_dataset:
  _target_: pytorch_segmentation_models_trainer.dataset_loader.dataset.InstanceSegmentationDataset
  input_csv_path: /data/buildings/train.csv
  root_dir: /data/buildings
  data_loader:
    shuffle: true
    num_workers: 4
    pin_memory: true
    drop_last: true
    prefetch_factor: 2

val_dataset:
  _target_: pytorch_segmentation_models_trainer.dataset_loader.dataset.InstanceSegmentationDataset
  input_csv_path: /data/buildings/val.csv
  root_dir: /data/buildings
  data_loader:
    shuffle: false
    num_workers: 4
    pin_memory: true
    drop_last: false
    prefetch_factor: 2

mode: train
device: cuda
```

---

## Running Detection Training

```bash
# Object detection
pytorch-smt --config-dir ./configs --config-name train_faster_rcnn

# Instance segmentation
pytorch-smt --config-dir ./configs --config-name train_mask_rcnn
```

:::warning Batch Size for Detection Models
Detection models process images of variable sizes and hold both RPN and ROI head activations in memory simultaneously. Start with `batch_size: 2` for Mask R-CNN and `batch_size: 4` for Faster R-CNN and adjust based on available VRAM.
:::
