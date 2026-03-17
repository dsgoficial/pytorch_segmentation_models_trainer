---
sidebar_position: 6
title: Training a Frame Field Segmentation Model
---

# Training a Frame Field Segmentation Model

Frame field segmentation extends standard semantic segmentation with an additional *crossfield* output that encodes the local orientation of boundaries. This makes the predicted contours geometrically regular and well-suited for building footprint extraction and subsequent polygon reconstruction.

:::note Inspiration
The frame field approach is inspired by [Polygonization by Frame Field Learning](https://github.com/Lydorn/Polygonization-by-Frame-Field-Learning). The `FrameFieldSegmentationPLModel` adapts that work into the PyTorch Lightning / Hydra training framework used by this library.
:::

---

## The `FrameFieldSegmentationPLModel` Class

`FrameFieldSegmentationPLModel` extends the base `Model` class with frame field-specific logic: it builds a `FrameFieldModel` backbone (a segmentation network combined with a crossfield head), constructs a multi-component compound loss, and optionally inserts a `ComputeSegGrads` pre-processor for coupling losses.

```yaml
pl_model:
  _target_: pytorch_segmentation_models_trainer.model_loader.frame_field_model.FrameFieldSegmentationPLModel
```

---

## Required Dataset

Frame field training requires a dataset that produces multi-channel ground truth masks (`gt_polygons_image`) alongside the crossfield angle map (`gt_crossfield_angle`). Use `FrameFieldSegmentationDataset`:

```yaml
train_dataset:
  _target_: pytorch_segmentation_models_trainer.dataset_loader.dataset.FrameFieldSegmentationDataset
  input_csv_path: /data/train.csv
  root_dir: /data
  data_loader:
    shuffle: true
    num_workers: 4
    pin_memory: true
    drop_last: true
    prefetch_factor: 2
```

The dataset returns batches as dictionaries with keys:

- `image` — input image tensor `(N, C, H, W)`
- `gt_polygons_image` — multi-channel mask `(N, C_gt, H, W)`:
  - channel 0: interior mask
  - channel 1: edge mask
  - channel 2: vertex mask (required for `CrossfieldAlign90Loss`)
- `gt_crossfield_angle` — crossfield angle map `(N, 1, H, W)`
- `class_freq` — per-image class frequency for loss weighting
- `sizes` — per-instance sizes for loss weighting

---

## Backbone Configuration

The model is built from a segmentation backbone combined with a crossfield prediction head. Configure the backbone separately:

```yaml
backbone:
  _target_: segmentation_models_pytorch.Unet
  encoder_name: resnet34
  encoder_weights: imagenet
  in_channels: 3
  classes: 2   # interior + edge channels

compute_seg: true
compute_crossfield: true

seg_params:
  compute_interior: true
  compute_edge: true
  compute_vertex: true
```

---

## Compound Loss Configuration

Frame field training uses a `MultiLoss` that combines several loss components. Configure it under `loss_params`:

```yaml
loss_params:
  compound_loss:
    normalize_losses: true   # normalize each component by its running average

  seg_loss_params:
    bce_coef: 0.5
    dice_coef: 0.5
    tversky_focal_coef: 0.0
    use_dist: true           # use distance-weighted loss (U-Net paper)
    use_size: true           # use instance-size weighting
    w0: 10.0
    sigma: 5.0

  multiloss:
    coefs:
      epoch_thresholds: [0, 10, 100]
      seg: 1.0
      crossfield_align: 0.1
      crossfield_align90: 0.05
      crossfield_smooth: 0.025
      seg_interior_crossfield: 0.5
      seg_edge_crossfield: 0.5
      seg_edge_interior: 0.2
```

### Loss Components

| Component | Description |
|---|---|
| `seg` | Combined BCE + Dice loss on interior/edge segmentation masks |
| `crossfield_align` | Aligns frame field tangents to ground truth polygon edges |
| `crossfield_align90` | Aligns 90-degree rotated frame field to edges (excluding vertices) |
| `crossfield_smooth` | Laplacian smoothness penalty on the crossfield away from edges |
| `seg_interior_crossfield` | Coupling: interior seg gradient aligns with crossfield |
| `seg_edge_crossfield` | Coupling: edge seg gradient aligns with crossfield |
| `seg_edge_interior` | Coupling: edge prediction matches interior gradient norm |

:::tip Weight Scheduling
`epoch_thresholds` enables weight scheduling over training. Provide lists of values instead of scalars to interpolate weights between epoch milestones:

```yaml
multiloss:
  coefs:
    epoch_thresholds: [0, 20, 100]
    seg: 1.0
    crossfield_align: [0.0, 0.1, 0.1]  # ramp up crossfield loss from epoch 0→20
    crossfield_smooth: [0.0, 0.025, 0.025]
```
:::

---

## The `ComputeSegGrads` Preprocessor

When coupling losses are enabled (e.g., `seg_interior_crossfield`, `seg_edge_crossfield`, `seg_edge_interior`), the model automatically inserts a `ComputeSegGrads` pre-processor into the `MultiLoss` pipeline before those losses run.

`ComputeSegGrads` computes spatial gradients of the predicted segmentation using a Scharr kernel and adds three tensors to the prediction batch:

- `pred_batch["seg_grads"]` — spatial gradient `(N, C, 2, H, W)`
- `pred_batch["seg_grad_norm"]` — gradient magnitude `(N, C, H, W)`
- `pred_batch["seg_grads_normed"]` — unit-normalized gradient direction `(N, C, 2, H, W)`

The coupling losses use these tensors to enforce consistency between the segmentation boundary shape and the predicted frame field orientation. **No manual configuration is required** — the model inserts `ComputeSegGrads` automatically when `compute_seg` and `compute_crossfield` are both `true` and the relevant `seg_params` flags are set.

---

## Loss Normalization

At the start of training (`on_train_start`), the model computes running normalization values for each loss component over a subset of the training data. This prevents any single component from dominating due to scale differences. The normalization callback is added automatically:

- `FrameFieldComputeWeightNormLossesCallback` is appended to the callback list if not already present

Control the number of samples used:

```yaml
loss_params:
  normalization_params:
    max_samples: 1000
```

---

## GPU Augmentations

GPU augmentations run on the GPU after data loading for maximum throughput. Specify them inside the dataset config:

```yaml
train_dataset:
  _target_: pytorch_segmentation_models_trainer.dataset_loader.dataset.FrameFieldSegmentationDataset
  input_csv_path: /data/train.csv
  root_dir: /data
  gpu_augmentation_list:
    - _target_: kornia.augmentation.RandomHorizontalFlip
      p: 0.5
    - _target_: kornia.augmentation.RandomVerticalFlip
      p: 0.5
    - _target_: kornia.augmentation.RandomRotation
      degrees: 30.0
      p: 0.3
  data_loader:
    shuffle: true
    num_workers: 4
```

The model builds a `torch.nn.Sequential` from this list and applies it to each batch on the GPU inside `training_step`.

---

## Full Training Config Example

```yaml title="configs/train_frame_field.yaml"
# ── Model selection ──────────────────────────────────────────────────────────
pl_model:
  _target_: pytorch_segmentation_models_trainer.model_loader.frame_field_model.FrameFieldSegmentationPLModel

# ── Frame field flags ────────────────────────────────────────────────────────
compute_seg: true
compute_crossfield: true
device: cuda

seg_params:
  compute_interior: true
  compute_edge: true
  compute_vertex: true

# ── Backbone ─────────────────────────────────────────────────────────────────
backbone:
  _target_: segmentation_models_pytorch.Unet
  encoder_name: resnet34
  encoder_weights: imagenet
  in_channels: 3
  classes: 2

# ── Loss ─────────────────────────────────────────────────────────────────────
loss_params:
  compound_loss:
    normalize_losses: true

  seg_loss_params:
    bce_coef: 0.5
    dice_coef: 0.5
    tversky_focal_coef: 0.0
    use_dist: true
    use_size: true
    w0: 10.0
    sigma: 5.0

  multiloss:
    coefs:
      epoch_thresholds: [0, 10, 100]
      seg: 1.0
      crossfield_align: 0.1
      crossfield_align90: 0.05
      crossfield_smooth: 0.025
      seg_interior_crossfield: 0.5
      seg_edge_crossfield: 0.5
      seg_edge_interior: 0.2

  normalization_params:
    max_samples: 1000

# ── Optimizer ────────────────────────────────────────────────────────────────
optimizer:
  _target_: torch.optim.AdamW
  lr: 0.001
  weight_decay: 1e-4

# ── LR Scheduler ─────────────────────────────────────────────────────────────
scheduler_list:
  - scheduler:
      _target_: torch.optim.lr_scheduler.OneCycleLR
      max_lr: 0.001
      epochs: 100
      # steps_per_epoch is auto-computed from the CSV + batch size
    interval: step
    name: one_cycle_lr

# ── Hyperparameters ──────────────────────────────────────────────────────────
hyperparameters:
  batch_size: 4

# ── PyTorch Lightning Trainer ────────────────────────────────────────────────
pl_trainer:
  max_epochs: 100
  accelerator: gpu
  devices: 1
  precision: 16
  log_every_n_steps: 50

# ── Callbacks ────────────────────────────────────────────────────────────────
callbacks:
  - _target_: pytorch_lightning.callbacks.ModelCheckpoint
    monitor: loss/val
    mode: min
    save_top_k: 3
    save_last: true
    filename: "ff-{epoch:02d}-{loss/val:.4f}"
    dirpath: ./checkpoints/frame_field

  - _target_: pytorch_lightning.callbacks.EarlyStopping
    monitor: loss/val
    patience: 20
    mode: min

  - _target_: pytorch_lightning.callbacks.LearningRateMonitor
    logging_interval: step

# ── Logger ───────────────────────────────────────────────────────────────────
logger:
  _target_: pytorch_lightning.loggers.TensorBoardLogger
  save_dir: ./experiments
  name: frame_field_resnet34

# ── Datasets ─────────────────────────────────────────────────────────────────
train_dataset:
  _target_: pytorch_segmentation_models_trainer.dataset_loader.dataset.FrameFieldSegmentationDataset
  input_csv_path: /data/buildings/train.csv
  root_dir: /data/buildings
  gpu_augmentation_list:
    - _target_: kornia.augmentation.RandomHorizontalFlip
      p: 0.5
    - _target_: kornia.augmentation.RandomVerticalFlip
      p: 0.5
  data_loader:
    shuffle: true
    num_workers: 4
    pin_memory: true
    drop_last: true
    prefetch_factor: 2

val_dataset:
  _target_: pytorch_segmentation_models_trainer.dataset_loader.dataset.FrameFieldSegmentationDataset
  input_csv_path: /data/buildings/val.csv
  root_dir: /data/buildings
  data_loader:
    shuffle: false
    num_workers: 4
    pin_memory: true
    drop_last: false
    prefetch_factor: 2

# ── Mode ─────────────────────────────────────────────────────────────────────
mode: train
```

---

## Running Frame Field Training

```bash
pytorch-smt --config-dir ./configs --config-name train_frame_field
```

---

## Logged Metrics

| Metric key | Description |
|---|---|
| `loss/train` | Total weighted loss |
| `loss/val` | Total weighted validation loss |
| `losses/train_seg` | Segmentation loss component |
| `losses/train_crossfield_align` | Crossfield alignment loss |
| `losses/train_crossfield_align90` | 90-degree alignment loss |
| `losses/train_crossfield_smooth` | Crossfield smoothness loss |
| `losses/train_seg_interior_crossfield` | Interior-crossfield coupling loss |
| `losses/train_seg_edge_crossfield` | Edge-crossfield coupling loss |
| `losses/train_seg_edge_interior` | Edge-interior coupling loss |
| `extra/train_crossfield_align_gt_field` | Ground truth field visualization |

:::warning GPU Memory
Frame field models have a larger memory footprint than plain segmentation models due to the crossfield head and the coupling losses that require spatial gradient computation. Start with `batch_size: 2` or `batch_size: 4` and increase after confirming memory usage is within budget.
:::
