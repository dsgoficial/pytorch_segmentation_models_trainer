---
sidebar_position: 5
title: Training a Semantic Segmentation Model
---

# Training a Semantic Segmentation Model

This guide walks through setting up and running a full semantic segmentation training job using the `Model` base class, which wraps a `segmentation_models_pytorch` architecture inside PyTorch Lightning.

## The `Model` Class

The base `Model` class (`pytorch_segmentation_models_trainer.model_loader.model.Model`) is a `pl.LightningModule` that wires together architecture, loss, optimizer, scheduler, datasets, metrics, and GPU augmentations from a single Hydra config object.

To use it, set the `pl_model` key in your config:

```yaml
pl_model:
  _target_: pytorch_segmentation_models_trainer.model_loader.model.Model
```

If `pl_model` is omitted, the training script defaults to `Model` automatically.

## Config Sections Overview

A complete training config contains the following top-level keys:

| Key | Purpose |
|---|---|
| `model` | Neural network architecture |
| `loss` | Loss function (simple) or `loss_params` (compound) |
| `optimizer` | Optimizer class and hyperparameters |
| `scheduler_list` | List of LR scheduler configs |
| `hyperparameters` | Batch size, device count |
| `pl_trainer` | PyTorch Lightning `Trainer` kwargs |
| `callbacks` | Checkpoint, early stopping, LR monitor, etc. |
| `metrics` | `torchmetrics` metrics to compute |
| `logger` | TensorBoard, WandB, CSV logger |
| `train_dataset` | Training dataset and dataloader settings |
| `val_dataset` | Validation dataset — monitored at the end of every epoch during `fit` |
| `test_dataset` | *(optional)* Test dataset — evaluated once after `fit` via `trainer.test()` |
| `mode` | `train` or `predict` |

---

## Supported Architectures

The `model` key accepts any class from [`segmentation_models_pytorch`](https://github.com/qubvel/segmentation_models.pytorch):

| Architecture | `_target_` |
|---|---|
| U-Net | `segmentation_models_pytorch.Unet` |
| U-Net++ | `segmentation_models_pytorch.UnetPlusPlus` |
| DeepLabV3+ | `segmentation_models_pytorch.DeepLabV3Plus` |
| FPN | `segmentation_models_pytorch.FPN` |
| PSPNet | `segmentation_models_pytorch.PSPNet` |
| PAN | `segmentation_models_pytorch.PAN` |
| MAnet | `segmentation_models_pytorch.MAnet` |
| Linknet | `segmentation_models_pytorch.Linknet` |

### Popular Encoders

Any `timm`-compatible encoder name works. Common choices:

- `resnet34` — lightweight baseline, fast to train
- `resnet50` — stronger features, moderate cost
- `efficientnet-b3` — high accuracy per parameter
- `resnext50_32x4d` — strong general-purpose encoder

---

## Scheduler List

The `scheduler_list` is a list of scheduler configs, each with these keys:

```yaml
scheduler_list:
  - scheduler:
      _target_: torch.optim.lr_scheduler.CosineAnnealingLR
      T_max: 100
      eta_min: 1e-6
    monitor: loss/val        # metric to monitor (for ReduceLROnPlateau)
    interval: epoch          # "epoch" or "step"
    name: cosine_annealing   # label shown in logger
```

:::tip OneCycleLR Auto-Configuration
When using `OneCycleLR`, set `steps_per_epoch` to `null` or omit it entirely. The framework automatically computes it from the CSV dataset size and batch size at the start of training:

```
steps_per_epoch = dataset_size // (batch_size * devices * accumulate_grad_batches)
```

The computed value is printed to the console during `setup`.
:::

```yaml
scheduler_list:
  - scheduler:
      _target_: torch.optim.lr_scheduler.OneCycleLR
      max_lr: 0.001
      epochs: 100
      # steps_per_epoch is computed automatically from train_dataset CSV
    interval: step
    name: one_cycle_lr
```

---

## Complete Config Example

```yaml title="configs/train_unet.yaml"
# ── Model ────────────────────────────────────────────────────────────────────
model:
  _target_: segmentation_models_pytorch.Unet
  encoder_name: resnet34
  encoder_weights: imagenet
  in_channels: 3
  classes: 1
  activation: sigmoid

# ── Loss ─────────────────────────────────────────────────────────────────────
loss:
  _target_: segmentation_models_pytorch.losses.DiceLoss
  mode: binary

# ── Optimizer ────────────────────────────────────────────────────────────────
optimizer:
  _target_: torch.optim.AdamW
  lr: 0.001
  weight_decay: 1e-4

# ── LR Schedulers ────────────────────────────────────────────────────────────
scheduler_list:
  - scheduler:
      _target_: torch.optim.lr_scheduler.CosineAnnealingLR
      T_max: 100
      eta_min: 1e-6
    monitor: loss/val
    interval: epoch
    name: cosine_lr

# ── Hyperparameters ──────────────────────────────────────────────────────────
hyperparameters:
  batch_size: 8

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
    filename: "best-{epoch:02d}-{loss/val:.4f}"
    dirpath: ./checkpoints

  - _target_: pytorch_lightning.callbacks.EarlyStopping
    monitor: loss/val
    patience: 15
    mode: min

  - _target_: pytorch_lightning.callbacks.LearningRateMonitor
    logging_interval: epoch

# ── Metrics ──────────────────────────────────────────────────────────────────
metrics:
  - _target_: torchmetrics.JaccardIndex
    task: binary
  - _target_: torchmetrics.F1Score
    task: binary

# ── Logger ───────────────────────────────────────────────────────────────────
logger:
  _target_: pytorch_lightning.loggers.TensorBoardLogger
  save_dir: ./experiments
  name: unet_resnet34

# ── Datasets ─────────────────────────────────────────────────────────────────
train_dataset:
  _target_: pytorch_segmentation_models_trainer.dataset_loader.dataset.SegmentationDataset
  input_csv_path: /data/train.csv
  root_dir: /data
  augmentation_list:
    - _target_: albumentations.HorizontalFlip
      p: 0.5
    - _target_: albumentations.RandomBrightnessContrast
      p: 0.3
    - _target_: albumentations.Normalize
      mean: [0.485, 0.456, 0.406]
      std: [0.229, 0.224, 0.225]
  data_loader:
    shuffle: true
    num_workers: 4
    pin_memory: true
    drop_last: true
    prefetch_factor: 2

val_dataset:
  _target_: pytorch_segmentation_models_trainer.dataset_loader.dataset.SegmentationDataset
  input_csv_path: /data/val.csv
  root_dir: /data
  augmentation_list:
    - _target_: albumentations.Normalize
      mean: [0.485, 0.456, 0.406]
      std: [0.229, 0.224, 0.225]
  data_loader:
    shuffle: false
    num_workers: 4
    pin_memory: true
    drop_last: false
    prefetch_factor: 2

# ── Test Dataset (optional) ───────────────────────────────────────────────────
# When present, trainer.test() is called automatically after trainer.fit().
# All metrics are logged with the "test/" prefix.
test_dataset:
  _target_: pytorch_segmentation_models_trainer.dataset_loader.dataset.SegmentationDataset
  input_csv_path: /data/test.csv
  root_dir: /data
  augmentation_list:
    - _target_: albumentations.Normalize
      mean: [0.485, 0.456, 0.406]
      std: [0.229, 0.224, 0.225]
  data_loader:
    shuffle: false
    num_workers: 4
    pin_memory: true
    drop_last: false
    prefetch_factor: 2

# ── Mode ─────────────────────────────────────────────────────────────────────
mode: train
device: cuda
```

---

## Multi-GPU Training

Enable distributed training by setting `accelerator`, `devices`, and `strategy` in `pl_trainer`:

```yaml
pl_trainer:
  accelerator: gpu
  devices: 2         # number of GPUs
  strategy: ddp      # DistributedDataParallel
  max_epochs: 100

hyperparameters:
  batch_size: 8      # per-GPU batch size
```

:::note Effective Batch Size
With DDP, the effective batch size is `batch_size * devices`. OneCycleLR auto-configuration accounts for this automatically when computing `steps_per_epoch`.
:::

Use `devices: -1` to use all available GPUs on the machine.

---

## Mixed Precision Training

```yaml
pl_trainer:
  precision: 16       # FP16 mixed precision
  # or:
  # precision: "bf16" # BFloat16 (better numerical stability on Ampere+)
```

:::tip Learning Rate with Mixed Precision
When using `precision: 16`, consider increasing the learning rate slightly (e.g., from `0.001` to `0.002`) as mixed precision training can tolerate higher learning rates.
:::

---

## Checkpointing

The `ModelCheckpoint` callback saves models according to a monitored metric:

```yaml
callbacks:
  - _target_: pytorch_lightning.callbacks.ModelCheckpoint
    monitor: loss/val         # metric to watch
    mode: min                 # save when metric decreases
    save_top_k: 3             # keep top 3 checkpoints
    save_last: true           # also save the last epoch
    filename: "epoch={epoch}-val_loss={loss/val:.4f}"
    dirpath: ./checkpoints/my_experiment
```

## Resuming from Checkpoint

To resume a training run from a saved checkpoint, add `resume_from_checkpoint` to `hyperparameters`:

```yaml
hyperparameters:
  batch_size: 8
  resume_from_checkpoint: ./checkpoints/my_experiment/last.ckpt
```

The training script detects this key and calls `load_from_checkpoint` before constructing the trainer, restoring model weights, optimizer state, and epoch counter.

---

## Running Training

```bash
pytorch-smt --config-dir ./configs --config-name train_unet
```

Override individual parameters at the command line without editing the YAML:

```bash
# Change encoder
pytorch-smt --config-dir ./configs --config-name train_unet \
  model.encoder_name=resnet50

# Change batch size and learning rate
pytorch-smt --config-dir ./configs --config-name train_unet \
  hyperparameters.batch_size=16 optimizer.lr=5e-4

# Enable multi-GPU
pytorch-smt --config-dir ./configs --config-name train_unet \
  pl_trainer.devices=4 pl_trainer.strategy=ddp
```

:::tip Preview Your Config
Before training, print the fully resolved config to check interpolations and overrides:

```bash
pytorch-smt --config-dir ./configs --config-name train_unet --cfg job
```
:::

---

## Dataset Splits

The framework follows the standard three-way split used in machine learning:

| Config key | PyTorch Lightning step | When it runs | Metric prefix |
|---|---|---|---|
| `train_dataset` | `training_step` | Every batch during `trainer.fit()` | `train/` |
| `val_dataset` | `validation_step` | End of every epoch during `trainer.fit()` | `val/` |
| `test_dataset` | `test_step` | Once after `trainer.fit()`, via `trainer.test()` | `test/` |

`val_dataset` and `test_dataset` are both **optional**. When `val_dataset` is absent, Lightning skips the validation loop entirely. When `test_dataset` is absent, `trainer.test()` is not called. You can use all three, only train + val, or only train.

:::tip When to use each split
- **`val_dataset`**: use this for epoch-level monitoring during training — it drives early stopping, `ModelCheckpoint`, and LR schedulers that `monitor` a `val/` metric.
- **`test_dataset`**: use this for the final held-out evaluation reported in papers or production metrics. It is intentionally kept separate from `val_dataset` so that hyperparameter tuning decisions are not influenced by test-set performance.
:::

---

## Logged Metrics

During training, the following metrics are written to the logger automatically:

| Metric key | When logged |
|---|---|
| `loss/train` | Each step and epoch |
| `loss/val` | Each validation epoch (requires `val_dataset`) |
| `loss/test` | Once after training (requires `test_dataset`) |
| `train/<metric_name>` | Per step and epoch (from `metrics` list) |
| `val/<metric_name>` | Per validation epoch (from `metrics` list) |
| `test/<metric_name>` | Once after training (from `metrics` list) |
| `losses/train_<name>` | Per component when using compound loss |
| `losses/val_<name>` | Per component when using compound loss |
| `losses/test_<name>` | Per component when using compound loss (test run) |
