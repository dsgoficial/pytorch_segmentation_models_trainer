---
sidebar_position: 8
title: Advanced Training Features
---

# Advanced Training Features

This guide covers advanced configuration topics that apply across all model types: compound losses, GPU augmentations, mixed precision, gradient clipping, OneCycleLR, multispectral weight adaptation, and checkpointing.

---

## Compound Losses with `MultiLoss`

The `MultiLoss` class combines multiple loss components, each with its own weight, into a single differentiable objective. It is the foundation of the frame field loss system and is available for any model type.

### How It Works

`MultiLoss` holds a list of `loss_funcs` and corresponding `weights`. During the forward pass it:

1. Runs any registered `pre_processes` functions that augment the prediction and ground truth dicts.
2. Calls each loss component individually.
3. Scales each result by its weight and sums to produce `total_loss`.
4. Returns `(total_loss, individual_losses_dict, extra_dict)`.

### Configuring via `loss_params`

The `Model.get_loss_function` method supports three configuration paths, checked in order:

**1. Compound loss (recommended):**

```yaml
loss_params:
  compound_loss:
    normalize_losses: true
    # ... additional builder params
```

**2. Legacy multi-loss:**

```yaml
loss_params:
  multi_loss:
    # ... legacy format
```

**3. Simple loss (direct):**

```yaml
loss:
  _target_: segmentation_models_pytorch.losses.DiceLoss
  mode: binary
```

### Weight Scheduling Over Epochs

Instead of a scalar, any loss weight can be a list that gets linearly interpolated between `epoch_thresholds`:

```yaml
loss_params:
  multiloss:
    coefs:
      epoch_thresholds: [0, 25, 100]
      seg: 1.0
      # Start crossfield losses at zero and ramp them up from epoch 0 to 25
      crossfield_align: [0.0, 0.1, 0.1]
      crossfield_smooth: [0.0, 0.025, 0.025]
      seg_interior_crossfield: [0.0, 0.5, 0.5]
```

At epoch 12 (halfway between 0 and 25), `crossfield_align` will be weighted at `0.05`. Beyond epoch 25 it stays at `0.1`.

:::tip When to Use Weight Scheduling
Start with the segmentation loss only (`crossfield_*` weights at 0) for the first N epochs so the model learns a good segmentation baseline before the more complex frame field coupling losses are introduced.
:::

### Individual Loss Logging

When using `MultiLoss`, each component is logged separately:

```
losses/train_seg
losses/train_crossfield_align
losses/train_seg_interior_crossfield
...
losses/val_seg
losses/val_crossfield_align
```

---

## GPU Augmentations

GPU augmentations execute transforms directly on GPU tensors after the data has been moved from the CPU DataLoader. This avoids re-encoding overhead and allows data augmentation to run in parallel with the forward pass of the previous batch.

### Configuration

Add `gpu_augmentation_list` inside `train_dataset` (and optionally `val_dataset`):

```yaml
train_dataset:
  _target_: pytorch_segmentation_models_trainer.dataset_loader.dataset.SegmentationDataset
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
    - _target_: kornia.augmentation.ColorJitter
      brightness: 0.2
      contrast: 0.2
      p: 0.4
  data_loader:
    shuffle: true
    num_workers: 4
```

### How It Works

The `Model.__init__` builds a `torch.nn.Sequential` module from `gpu_augmentation_list` using Hydra `instantiate`. During `training_step`, the images are passed through this sequential before the forward pass:

```python
if self.gpu_train_transform is not None:
    batch["image"] = self.gpu_train_transform(batch["image"])
```

If `gpu_augmentation_list` is absent from the dataset config, `gpu_train_transform` is `None` and the step is skipped with no overhead.

:::note CPU vs GPU Augmentations
Use standard `albumentations` transforms in `augmentation_list` for CPU augmentations that need to operate on PIL images or NumPy arrays (e.g., elastic deformations, grid distortions). Use `gpu_augmentation_list` with `kornia` for transforms that benefit from batched GPU tensor operations (flips, rotations, color jitter).
:::

---

## Mixed Precision Training

Mixed precision reduces memory usage and speeds up training on modern GPUs by storing activations in 16-bit floats while keeping critical accumulators in 32-bit.

```yaml
pl_trainer:
  precision: 16        # FP16 AMP
  # or:
  # precision: "bf16"  # BFloat16 (Ampere+ GPUs: A100, RTX 3090, etc.)
```

### FP16 vs BF16

| Format | Range | When to use |
|---|---|---|
| `16` (FP16) | ±65504 | Older GPUs (Volta, Turing); requires loss scaling |
| `"bf16"` (BF16) | Same range as FP32 | Ampere+ GPUs; more numerically stable, no loss scaling needed |

### Learning Rate Considerations

FP16 training can tolerate slightly higher learning rates. A common heuristic is to scale the LR up by `sqrt(2)` when switching from FP32 to FP16:

```yaml
# FP32 baseline
optimizer:
  lr: 0.001

# FP16 equivalent
optimizer:
  lr: 0.00141   # 0.001 * sqrt(2)
```

### Mixed Precision and Frame Field Losses

The `SegLoss` component automatically switches to `binary_cross_entropy_with_logits` (which is numerically safer under FP16) when `precision: 16` is set in `pl_trainer`:

```python
use_mixed_precision=True if cfg.pl_trainer.precision == 16 else False
```

This happens transparently — no additional configuration is needed.

---

## Gradient Clipping

Gradient clipping prevents exploding gradients, particularly useful when training with large learning rates, long sequences, or complex multi-loss objectives.

```yaml
pl_trainer:
  gradient_clip_val: 1.0              # clip gradients to L2 norm of 1.0
  gradient_clip_algorithm: norm       # "norm" (L2) or "value" (element-wise)
```

:::tip When to Enable
Enable gradient clipping (`gradient_clip_val: 1.0`) when:
- Training frame field models with multiple coupling losses
- Using OneCycleLR with a high `max_lr`
- Observing loss spikes or NaN losses during training
:::

---

## OneCycleLR Auto-Configuration

OneCycleLR is a policy that anneals the learning rate from an initial value up to `max_lr` and back down over the course of training. It requires `steps_per_epoch` (the number of optimizer steps per epoch).

### Automatic Computation

When `steps_per_epoch` is omitted or set to `null`, the framework reads the CSV file from `train_dataset.input_csv_path`, counts the rows, and computes:

```
steps_per_epoch = dataset_size // (batch_size × devices × accumulate_grad_batches)
```

The result is printed to the console at the start of training:

```
============================================================
AUTO-COMPUTED STEPS_PER_EPOCH FROM CONFIG
============================================================
CSV path:               /data/train.csv
Dataset size:               98,201 samples
Batch size:                      8
Devices:                         2
Gradient accumulation:           1
Effective batch size:           16
Steps per epoch:             6,137
============================================================
```

### Config

```yaml
scheduler_list:
  - scheduler:
      _target_: torch.optim.lr_scheduler.OneCycleLR
      max_lr: 0.001
      epochs: 100
      # steps_per_epoch: null  <- omit or set to null for auto-computation
    interval: step           # must be "step" for OneCycleLR
    name: one_cycle_lr

hyperparameters:
  batch_size: 8

pl_trainer:
  accelerator: gpu
  devices: 2
  accumulate_grad_batches: 1
```

### Manual Override

To disable auto-computation and set the value explicitly:

```yaml
scheduler_list:
  - scheduler:
      _target_: torch.optim.lr_scheduler.OneCycleLR
      max_lr: 0.001
      epochs: 100
      steps_per_epoch: 6137   # explicit value; auto-computation is skipped
    interval: step
    name: one_cycle_lr
```

:::warning OneCycleLR with Multi-GPU
When using DDP, `device_count` is factored into the auto-computation automatically. Manual `steps_per_epoch` values must account for this: `steps_per_epoch = dataset_size // (batch_size × num_gpus)`.
:::

---

## Multispectral Weight Adaptation

When working with remote sensing data, images often have more than 3 spectral bands (e.g., RGB + NIR = 4 channels, or full multispectral stacks with 8+ bands). ImageNet-pretrained encoders expect exactly 3 input channels.

The framework automatically adapts the first convolutional layer of the encoder to accept `N` input channels by redistributing the pretrained 3-channel weights.

### Configuration

Set `in_channels` on the model and configure the dataset to read the required bands:

```yaml
model:
  _target_: segmentation_models_pytorch.Unet
  encoder_name: resnet34
  encoder_weights: imagenet
  in_channels: 4          # RGB + NIR
  classes: 1

train_dataset:
  _target_: pytorch_segmentation_models_trainer.dataset_loader.dataset.SegmentationDataset
  input_csv_path: /data/train.csv
  root_dir: /data
  use_rasterio: true        # use rasterio for multi-band reading
  selected_bands: [1, 2, 3, 4]   # 1-indexed band selection
  data_loader:
    shuffle: true
    num_workers: 4
```

:::note How Adaptation Works
`segmentation_models_pytorch` handles first-layer weight adaptation internally when `encoder_weights` is set and `in_channels != 3`. The pretrained 3-channel weights are either repeated or averaged to initialise the N-channel first layer. Using `use_rasterio: true` and `selected_bands` in the dataset config tells the data loader to read the correct spectral channels from multi-band GeoTIFF files.
:::

---

## Checkpointing and Resumption

### Saving Checkpoints

Configure `ModelCheckpoint` in the `callbacks` list:

```yaml
callbacks:
  - _target_: pytorch_lightning.callbacks.ModelCheckpoint
    monitor: loss/val          # metric to track
    mode: min                  # save when metric decreases
    save_top_k: 5              # keep the 5 best checkpoints
    save_last: true            # also save the most recent epoch as "last.ckpt"
    filename: "{epoch:03d}-{loss/val:.4f}"
    dirpath: ./checkpoints/experiment_name
    auto_insert_metric_name: false
```

Checkpoint files are saved as `.ckpt` (PyTorch Lightning format) and contain model weights, optimizer state, scheduler state, and hyperparameters.

### Resuming Training

Add `resume_from_checkpoint` under `hyperparameters` to continue a previous run:

```yaml
hyperparameters:
  batch_size: 8
  resume_from_checkpoint: ./checkpoints/experiment_name/last.ckpt
```

The training script detects this key and calls `load_from_checkpoint` before constructing the trainer, restoring all state including the epoch counter.

### Loading a Specific Checkpoint for Inference

For prediction, specify the checkpoint path in the top-level config:

```yaml
mode: predict
checkpoint_path: ./checkpoints/experiment_name/epoch=042-loss_val=0.1234.ckpt

inference_image_reader:
  _target_: pytorch_segmentation_models_trainer.tools.data_handlers.raster_reader.FolderImageReaderProcessor
  folder_name: /path/to/images

inference_processor:
  _target_: pytorch_segmentation_models_trainer.tools.inference.inference_processors.SingleImageInfereceProcessor

inference_threshold: 0.5
```

:::tip Choosing the Best Checkpoint
Use the checkpoint monitored on `loss/val` (or `metrics/val_iou` for detection) rather than the last epoch. Set `save_last: true` alongside `save_top_k` so you always have both a "best" and a "latest" checkpoint available.
:::

### Freezing the Encoder

To fine-tune only the decoder head while keeping ImageNet features frozen, call `set_encoder_trainable(False)` on the model:

```python
model = Model(cfg)
model.set_encoder_trainable(trainable=False)
```

This sets `requires_grad=False` on all parameters of `model.encoder`. To unfreeze later (e.g., after an initial warm-up phase), call `set_encoder_trainable(True)`.

---

## LR Warmup

A linear learning rate warmup can be prepended to any scheduler (except `OneCycleLR`, which has its own warmup logic) by setting `warmup_epochs` inside `hyperparameters`. During warmup the LR rises linearly from 1% of the base rate to 100% over the configured number of epochs, then the main scheduler takes over.

### LR Warmup Config

```yaml
hyperparameters:
  batch_size: 8
  warmup_epochs: 5     # linear warmup for 5 epochs, then hand off to main scheduler

scheduler_list:
  - scheduler:
      _target_: torch.optim.lr_scheduler.CosineAnnealingLR
      T_max: 95          # automatically reduced by warmup_epochs
      eta_min: 1.0e-6
    interval: epoch
    name: cosine_lr
```

The framework automatically subtracts `warmup_epochs` from `T_max` (or equivalent period parameters) so the total scheduled duration stays at `epochs` as configured.

:::note OneCycleLR
`warmup_epochs` is ignored when `OneCycleLR` is used — that scheduler has its own built-in warm-up phase via its `pct_start` parameter.
:::

:::tip When to use warmup
Warmup is especially helpful when:

- Using large batch sizes (stable gradient estimates develop gradually).
- Fine-tuning a pretrained encoder that would otherwise be disrupted by a high initial LR.
- Combined with `EMACallback`, where early EMA contamination from large weight swings is undesirable.
:::

---

## Accumulating Gradients

Gradient accumulation simulates a larger effective batch size without requiring more GPU memory:

```yaml
pl_trainer:
  accumulate_grad_batches: 4   # effective batch = batch_size × 4

hyperparameters:
  batch_size: 4                # physical batch per step
  # effective batch size: 4 × 4 = 16
```

OneCycleLR auto-configuration accounts for `accumulate_grad_batches` when computing `steps_per_epoch`, so the scheduler cadence remains correct.
