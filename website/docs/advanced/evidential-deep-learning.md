---
sidebar_position: 7
title: Evidential Deep Learning
description: Uncertainty quantification for segmentation using Dirichlet-based evidential models.
---

# Evidential Deep Learning (EDL)

## What is EDL?

Standard segmentation networks output class probabilities via Softmax — a single point estimate with no information about whether the network "knows" what it is seeing. Evidential Deep Learning (EDL) replaces this with a **Dirichlet distribution** over the probability simplex, parameterised directly by the network's output.

This makes it possible to distinguish two types of uncertainty:

| Type | Definition | When it is high |
| --- | --- | --- |
| **Epistemic** (model) | u = K / S | Input pattern unseen during training (OOD) |
| **Aleatoric** (data) | Variance of Dirichlet | Ambiguous pixels even with infinite data |

The uncertainty map produced by EDL is particularly useful in remote sensing workflows where regions outside the training distribution (cloud shadows, new land-cover types, sensor artefacts) need to be flagged automatically rather than silently assigned an incorrect class.

**Reference:** Sensoy, M., Kaplan, L., & Kandemir, M. (2018). Evidential Deep Learning to Quantify Classification Uncertainty. *NeurIPS 2018*. https://arxiv.org/abs/1806.01768

---

## Theory in One Page

The network learns to output **evidence** per class per pixel:

```
evidence_k = Softplus(logit_k)    # e_k ≥ 0
alpha_k    = evidence_k + 1       # α_k ≥ 1 (Dirichlet parameter)
S          = Σ_k alpha_k          # total evidence strength
p̂_k       = alpha_k / S          # expected class probability
u          = K / S                # epistemic uncertainty ∈ (0, 1]
```

When the network has **no evidence** (unseen pattern), all `evidence_k ≈ 0`, so `alpha_k ≈ 1`, `S ≈ K`, and `u ≈ 1`. When it has **high evidence** for one class, `S >> K` and `u → 0`.

### Loss Function

```
L_total = L_MSE + λ_t · L_KL
```

**L_MSE** (integrated over Dirichlet, Sensoy eq. 4):

```
L_MSE = Σ_k [ (y_k - α_k/S)² + α_k(S-α_k)/(S²(S+1)) ]
       = bias²              + variance term
```

**L_KL** (regulariser, Sensoy eq. 8):
Before computing KL, evidence for the correct class is removed so the network is not penalised for *correct* high-confidence predictions:

```
α̃_k = y_k + (1 - y_k) · α_k
L_KL = KL[ Dir(α̃) || Dir(1,...,1) ]
```

**KL annealing** (λ_t): the KL coefficient starts at 0 and is linearly increased to 1.0 over training, ensuring the network first learns to discriminate classes before being penalised for residual wrong-class evidence. This is controlled by the standard `CompoundLoss` weight schedule in the YAML — no special code needed.

---

## Quick Start: 3 Steps

### Step 1 — Choose a YAML

| Scenario | Config file |
| --- | --- |
| Training from scratch | `conf/examples/edl_from_scratch.yaml` |
| Fine-tuning from a pre-trained checkpoint | `conf/examples/edl_finetune.yaml` |

Edit `hyperparameters.classes` to match your number of classes, and set `train_dataset.input_csv_path` / `val_dataset.input_csv_path` to your data.

### Step 2 — Train

```bash
python -m pytorch_segmentation_models_trainer.train \
    --config-name edl_from_scratch
```

During training you will see:

- `loss/train` and `loss/val` — total EDL loss
- `losses/train_edl_mse` and `losses/train_edl_kl` — individual components
- `edl/train_uncertainty` — mean epistemic uncertainty per batch

The uncertainty visualisation callback logs a 4-column diagnostic grid to TensorBoard / WandB / file system every N epochs:

```
[Input image | Predicted class | Uncertainty map (plasma colormap) | Ground truth]
```

### Step 3 — Export Uncertainty GeoTIFF

```bash
python -m pytorch_segmentation_models_trainer.predict \
    --config-name predict \
    model_path=/path/to/checkpoint.ckpt \
    image_path=/path/to/image.tif \
    output_path=/results/probs.tif \
    output_uncertainty_path=/results/uncertainty.tif
```

This produces:

- `probs.tif` — multi-band float32, one band per class (class probabilities)
- `uncertainty.tif` — single-band float32, values ∈ (0, 1], CRS and transform preserved

---

## Building Configuration Files

EDL configs follow the same Hydra YAML structure as the rest of the framework, with three EDL-specific blocks: `model`, `loss_params`, and `callbacks`. The examples below show complete, runnable configs. Copy one that matches your scenario, adjust the highlighted fields, and save it to `conf/` (or any directory Hydra can find).

### Where to place the file

```
your_project/
└── conf/
    ├── examples/
    │   ├── edl_from_scratch.yaml   # bundled example
    │   └── edl_finetune.yaml       # bundled example
    └── my_edl_experiment.yaml      # ← your custom config
```

Run with:

```bash
python -m pytorch_segmentation_models_trainer.train \
    --config-path conf \
    --config-name my_edl_experiment
```

---

### Config: Training from scratch

Use this when you have **no pre-trained weights**. The encoder trains from epoch 0. KL annealing prevents the regulariser from collapsing the evidence to zero before the network has learned to discriminate classes.

```yaml
# ──────────────────────────────────────────────────────────────────────────────
# edl_from_scratch.yaml
# ──────────────────────────────────────────────────────────────────────────────

defaults:
  - _self_

# ── Backbone geometry ─────────────────────────────────────────────────────────
backbone:
  name: resnet50          # any timm / SMP encoder name
  input_width: 512        # must match your dataset tile size
  input_height: 512

# ── Experiment hyperparameters ────────────────────────────────────────────────
hyperparameters:
  model_name: edl_unet_resnet50_scratch   # used for checkpoint naming
  batch_size: 8
  epochs: 100
  max_lr: 1.0e-3
  classes: 5              # ← SET THIS: number of segmentation classes

# ── Model ─────────────────────────────────────────────────────────────────────
# EvidentialWrapper wraps any SMP / HuggingFace / custom model.
# It replaces Softmax with Softplus and adds Dirichlet parameterisation.
model:
  _target_: pytorch_segmentation_models_trainer.custom_models.edl_wrapper.EvidentialWrapper
  freeze_encoder: false   # train encoder from epoch 0
  model:
    _target_: segmentation_models_pytorch.Unet
    encoder_name: ${backbone.name}
    encoder_weights: null   # no pre-trained weights
    in_channels: 3          # ← SET THIS: number of input bands
    classes: ${hyperparameters.classes}

# ── Loss ──────────────────────────────────────────────────────────────────────
# CompoundLoss combines EvidentialMSELoss (constant) and EvidentialKLLoss
# (annealed). epoch_thresholds defines the schedule boundaries.
#
# With epoch_thresholds: [0, 10, 40, 100] and weight: [0.0, 0.0, 1.0, 1.0]:
#   epochs  0– 9  → KL weight = 0.0  (MSE only)
#   epochs 10–39  → KL weight ramps from 0.0 to 1.0
#   epochs 40+    → KL weight = 1.0  (full EDL loss)
loss_params:
  compound_loss:
    epoch_thresholds: [0, 10, 40, 100]
    losses:
      - loss:
          _target_: pytorch_segmentation_models_trainer.custom_losses.edl_loss.EvidentialMSELoss
          name: edl_mse
          num_classes: ${hyperparameters.classes}
        weight: 1.0           # constant throughout training
      - loss:
          _target_: pytorch_segmentation_models_trainer.custom_losses.edl_loss.EvidentialKLLoss
          name: edl_kl
          num_classes: ${hyperparameters.classes}
        weight: [0.0, 0.0, 1.0, 1.0]   # one value per epoch_thresholds interval

# ── Optimizer & Scheduler ─────────────────────────────────────────────────────
optimizer:
  _target_: torch.optim.AdamW
  lr: ${hyperparameters.max_lr}
  weight_decay: 1.0e-4

scheduler_list:
  - scheduler:
      _target_: torch.optim.lr_scheduler.CosineAnnealingLR
      T_max: ${hyperparameters.epochs}
      eta_min: 1.0e-7
    interval: epoch
    frequency: 1

# ── Callbacks ─────────────────────────────────────────────────────────────────
callbacks:
  # EvidentialWarmupCallback: with freeze_encoder=false, the encoder is never
  # frozen. The callback still runs and logs the current training phase.
  - _target_: pytorch_segmentation_models_trainer.custom_callbacks.edl_callbacks.EvidentialWarmupCallback
    warmup_epochs: 0          # no freeze when training from scratch
    freeze_encoder: false     # must match model.freeze_encoder above
    partial_unfreeze_epoch: 0

  # EvidentialUncertaintyVisualizationCallback: logs a 4-column grid
  # (input | prediction | uncertainty | ground truth) every N epochs.
  # norm_params must match the Normalize transform used in the dataset.
  - _target_: pytorch_segmentation_models_trainer.custom_callbacks.edl_callbacks.EvidentialUncertaintyVisualizationCallback
    num_images: 4
    log_every_n_epochs: 10
    norm_params:
      mean: [0.485, 0.456, 0.406]   # ← match your Normalize transform
      std:  [0.229, 0.224, 0.225]

# ── PyTorch Lightning trainer ─────────────────────────────────────────────────
pl_trainer:
  max_epochs: ${hyperparameters.epochs}
  accelerator: auto   # "gpu" | "cpu" | "auto"
  devices: 1
  precision: "16-mixed"   # use "32" if mixed precision causes instability

# ── Datasets ──────────────────────────────────────────────────────────────────
# The CSV must have columns: image_path, mask_path
# Paths can be absolute or relative to root_dir.
train_dataset:
  _target_: pytorch_segmentation_models_trainer.dataset_loader.dataset.SegmentationDataset
  input_csv_path: /data/train.csv   # ← SET THIS
  root_dir: /data                   # ← SET THIS
  augmentation_list:
    - _target_: albumentations.Resize
      height: ${backbone.input_height}
      width:  ${backbone.input_width}
    - _target_: albumentations.HorizontalFlip
      p: 0.5
    - _target_: albumentations.VerticalFlip
      p: 0.5
    - _target_: albumentations.RandomRotate90
      p: 0.5
    - _target_: albumentations.Normalize
      mean: [0.485, 0.456, 0.406]
      std:  [0.229, 0.224, 0.225]
    - _target_: albumentations.pytorch.ToTensorV2
  data_loader:
    batch_size: ${hyperparameters.batch_size}
    num_workers: 8
    shuffle: true
    pin_memory: true
    drop_last: true

val_dataset:
  _target_: pytorch_segmentation_models_trainer.dataset_loader.dataset.SegmentationDataset
  input_csv_path: /data/val.csv   # ← SET THIS
  root_dir: /data                 # ← SET THIS
  augmentation_list:
    - _target_: albumentations.Resize
      height: ${backbone.input_height}
      width:  ${backbone.input_width}
    - _target_: albumentations.Normalize
      mean: [0.485, 0.456, 0.406]
      std:  [0.229, 0.224, 0.225]
    - _target_: albumentations.pytorch.ToTensorV2
  data_loader:
    batch_size: ${hyperparameters.batch_size}
    num_workers: 8
    shuffle: false
    pin_memory: true
    drop_last: false

# ── Metrics ───────────────────────────────────────────────────────────────────
metrics:
  - _target_: torchmetrics.JaccardIndex
    task: multiclass
    num_classes: ${hyperparameters.classes}
  - _target_: torchmetrics.F1Score
    task: multiclass
    num_classes: ${hyperparameters.classes}
    average: macro
```

---

### Config: Fine-tuning from pre-trained weights

Use this when loading an existing checkpoint or ImageNet weights. The encoder is frozen initially so only the Dirichlet head is calibrated, then progressively unfrozen over three phases.

```yaml
# ──────────────────────────────────────────────────────────────────────────────
# edl_finetune.yaml
# ──────────────────────────────────────────────────────────────────────────────

defaults:
  - _self_

backbone:
  name: resnet50
  input_width: 512
  input_height: 512

hyperparameters:
  model_name: edl_unet_resnet50_finetune
  batch_size: 8
  epochs: 50
  max_lr: 5.0e-4    # lower LR for fine-tuning
  classes: 5        # ← must match the checkpoint's number of classes

# ── Model ─────────────────────────────────────────────────────────────────────
model:
  _target_: pytorch_segmentation_models_trainer.custom_models.edl_wrapper.EvidentialWrapper
  freeze_encoder: true    # encoder frozen during Phase 1 (warm-up)
  model:
    _target_: segmentation_models_pytorch.Unet
    encoder_name: ${backbone.name}
    encoder_weights: imagenet   # or null to load from a .ckpt checkpoint
    in_channels: 3
    classes: ${hyperparameters.classes}

# ── Loss ──────────────────────────────────────────────────────────────────────
# Faster KL annealing because features are already good.
# With epoch_thresholds: [0, 5, 15, 50] and weight: [0.0, 0.05, 0.5, 1.0]:
#   epochs  0– 4  → KL = 0.0   (encoder frozen, MSE only)
#   epochs  5–14  → KL ramps from 0.05 to 0.5  (partial unfreeze)
#   epochs 15+    → KL ramps from 0.5  to 1.0  (full unfreeze)
loss_params:
  compound_loss:
    epoch_thresholds: [0, 5, 15, 50]
    losses:
      - loss:
          _target_: pytorch_segmentation_models_trainer.custom_losses.edl_loss.EvidentialMSELoss
          name: edl_mse
          num_classes: ${hyperparameters.classes}
        weight: 1.0
      - loss:
          _target_: pytorch_segmentation_models_trainer.custom_losses.edl_loss.EvidentialKLLoss
          name: edl_kl
          num_classes: ${hyperparameters.classes}
        weight: [0.0, 0.05, 0.5, 1.0]

# ── Optimizer & Scheduler ─────────────────────────────────────────────────────
optimizer:
  _target_: torch.optim.AdamW
  lr: ${hyperparameters.max_lr}
  weight_decay: 1.0e-4

scheduler_list:
  - scheduler:
      _target_: torch.optim.lr_scheduler.CosineAnnealingLR
      T_max: ${hyperparameters.epochs}
      eta_min: 1.0e-7
    interval: epoch
    frequency: 1

# ── Callbacks ─────────────────────────────────────────────────────────────────
# EvidentialWarmupCallback manages the 3-phase encoder freeze schedule:
#   Phase 1 — epochs 0-4:  encoder frozen (warmup_epochs=5)
#   Phase 2 — epochs 5-14: last 2 encoder stages free (partial_unfreeze_epoch=15)
#   Phase 3 — epoch 15+:   all layers free
callbacks:
  - _target_: pytorch_segmentation_models_trainer.custom_callbacks.edl_callbacks.EvidentialWarmupCallback
    warmup_epochs: 5              # ← must be consistent with epoch_thresholds[1]
    freeze_encoder: true          # must match model.freeze_encoder above
    partial_unfreeze_epoch: 15    # ← must be consistent with epoch_thresholds[2]

  - _target_: pytorch_segmentation_models_trainer.custom_callbacks.edl_callbacks.EvidentialUncertaintyVisualizationCallback
    num_images: 4
    log_every_n_epochs: 5
    norm_params:
      mean: [0.485, 0.456, 0.406]
      std:  [0.229, 0.224, 0.225]

# ── PyTorch Lightning trainer ─────────────────────────────────────────────────
pl_trainer:
  max_epochs: ${hyperparameters.epochs}
  accelerator: auto
  devices: 1
  precision: "16-mixed"

# ── Datasets ──────────────────────────────────────────────────────────────────
train_dataset:
  _target_: pytorch_segmentation_models_trainer.dataset_loader.dataset.SegmentationDataset
  input_csv_path: /data/train.csv   # ← SET THIS
  root_dir: /data                   # ← SET THIS
  augmentation_list:
    - _target_: albumentations.Resize
      height: ${backbone.input_height}
      width:  ${backbone.input_width}
    - _target_: albumentations.HorizontalFlip
      p: 0.5
    - _target_: albumentations.Normalize
      mean: [0.485, 0.456, 0.406]
      std:  [0.229, 0.224, 0.225]
    - _target_: albumentations.pytorch.ToTensorV2
  data_loader:
    batch_size: ${hyperparameters.batch_size}
    num_workers: 8
    shuffle: true
    pin_memory: true
    drop_last: true

val_dataset:
  _target_: pytorch_segmentation_models_trainer.dataset_loader.dataset.SegmentationDataset
  input_csv_path: /data/val.csv   # ← SET THIS
  root_dir: /data                 # ← SET THIS
  augmentation_list:
    - _target_: albumentations.Resize
      height: ${backbone.input_height}
      width:  ${backbone.input_width}
    - _target_: albumentations.Normalize
      mean: [0.485, 0.456, 0.406]
      std:  [0.229, 0.224, 0.225]
    - _target_: albumentations.pytorch.ToTensorV2
  data_loader:
    batch_size: ${hyperparameters.batch_size}
    num_workers: 8
    shuffle: false
    pin_memory: true
    drop_last: false

# ── Metrics ───────────────────────────────────────────────────────────────────
metrics:
  - _target_: torchmetrics.JaccardIndex
    task: multiclass
    num_classes: ${hyperparameters.classes}
  - _target_: torchmetrics.F1Score
    task: multiclass
    num_classes: ${hyperparameters.classes}
    average: macro
```

---

### Common adaptations

#### Changing the backbone

Replace `backbone.name` and, if needed, `model.model.encoder_weights`:

```yaml
backbone:
  name: efficientnet-b4   # any timm encoder supported by SMP

model:
  model:
    _target_: segmentation_models_pytorch.Unet
    encoder_name: ${backbone.name}
    encoder_weights: imagenet
```

#### Multispectral / multi-band input

Set `in_channels` to match your raster band count:

```yaml
model:
  model:
    in_channels: 8   # e.g. 8-band Sentinel-2 stack
```

#### Binary segmentation (2 classes)

```yaml
hyperparameters:
  classes: 2

metrics:
  - _target_: torchmetrics.JaccardIndex
    task: binary
  - _target_: torchmetrics.F1Score
    task: binary
```

#### Adjusting KL annealing schedule

The `epoch_thresholds` list and `weight` list must have the same length. Each weight is the value at the *start* of the corresponding interval; values between thresholds are linearly interpolated:

```yaml
loss_params:
  compound_loss:
    epoch_thresholds: [0, 20, 60, 100]
    losses:
      - loss:   # MSE: constant
          ...
        weight: 1.0
      - loss:   # KL: delayed, slow ramp
          ...
        weight: [0.0, 0.0, 0.5, 1.0]
```

#### Consistency constraint: fine-tune schedule

When using `edl_finetune.yaml`, `warmup_epochs` and `partial_unfreeze_epoch` must align with `epoch_thresholds`:

| `epoch_thresholds` | `warmup_epochs` | `partial_unfreeze_epoch` |
| --- | --- | --- |
| `[0, 5, 15, 50]` | `5` | `15` |
| `[0, 10, 30, 80]` | `10` | `30` |

---

## Training From Scratch vs Fine-tuning

### Training from scratch (`freeze_encoder: false`)

The encoder is never frozen. The only warm-up mechanism is **KL annealing**:

| Epochs | KL weight | Effect |
| --- | --- | --- |
| 0–9 | 0.0 | Network learns classes via MSE only |
| 10–39 | 0 → 1.0 | KL regularisation ramps in gradually |
| 40+ | 1.0 | Full EDL training |

Without annealing, the KL term would push all evidences toward zero in the first epoch ("I am uncertain about everything"), preventing the network from ever learning class discrimination. The annealing gives the network time to form good decision boundaries before the uncertainty calibration begins.

### Fine-tuning from pre-trained weights (`freeze_encoder: true`)

Pre-trained weights (e.g. ImageNet) already encode good features. The goal is to **re-interpret** the encoder's output as Dirichlet evidence without corrupting the learned representations:

| Phase | Epochs | Encoder | KL weight |
| --- | --- | --- | --- |
| 1 — Calibration | 0–4 | Frozen | 0.0 |
| 2 — Partial unfreeze | 5–14 | Last 2 stages free | 0 → 0.5 |
| 3 — Full | 15+ | All layers free | 0.5 → 1.0 |

The `EvidentialWarmupCallback` manages the encoder freeze schedule. The `EvidentialKLLoss` weight schedule in the YAML manages the KL annealing.

**Note:** models trained with Softmax do not need weight re-initialisation. The `EvidentialWrapper` applies `Softplus` to the same logits the encoder produces — the pre-trained features remain valid without any modification.

---

## Architecture

```text
EvidentialWrapper
└── model (any SMP / HuggingFace / timm / custom model)
      ↓ forward(x)
    logits [B, K, H, W]
      ↓ Softplus
    evidence [B, K, H, W]  ≥ 0
      ↓ + 1
    alpha [B, K, H, W]     ≥ 1
      ↓ / S
    probs [B, K, H, W]     (expected class probabilities)
    uncertainty [B, 1, H, W]  = K / S  ∈ (0, 1]
```

The wrapper detects and handles:

- Plain tensor output (standard SMP, custom models)
- Tuple output (e.g. `(logits, aux_features)`)
- Dict output with `"out"` or `"seg"` key (torchvision DeepLab style)

---

## Interpreting the Uncertainty Map

| Uncertainty value | Interpretation |
| --- | --- |
| u ≈ 0.0 | Very high confidence — strong evidence for one class |
| u ≈ 0.5 | Moderate uncertainty — evidence spread across classes |
| u ≈ 1.0 | Maximum uncertainty — no evidence for any class (OOD region) |

In practice, values above 0.7–0.8 often correspond to:

- Cloud shadows or atmospheric artefacts
- Transition zones between classes
- Sensor artifacts (striping, saturation)
- Land-cover types not represented in training data

Use the uncertainty map as a **quality mask**: pixels where `u > threshold` can be flagged for manual review or excluded from downstream analysis.

---

## Configuration Reference

### `EvidentialWrapper`

| Field | Type | Default | Description |
| --- | --- | --- | --- |
| `model` | dict | MISSING | Nested model config (any segmentation model) |
| `freeze_encoder` | bool | `false` | Freeze encoder at init (set `true` for fine-tuning) |

### `EvidentialMSELoss`

| Field | Type | Default | Description |
| --- | --- | --- | --- |
| `name` | str | `"edl_mse"` | Loss name for logging |
| `num_classes` | int | MISSING | Number of segmentation classes K |
| `ignore_index` | int | `255` | Label value to exclude from loss |

### `EvidentialKLLoss`

| Field | Type | Default | Description |
| --- | --- | --- | --- |
| `name` | str | `"edl_kl"` | Loss name for logging |
| `num_classes` | int | MISSING | Number of segmentation classes K |
| `ignore_index` | int | `255` | Label value to exclude from loss |

### `EvidentialWarmupCallback`

| Field | Type | Default | Description |
| --- | --- | --- | --- |
| `warmup_epochs` | int | `5` | Epochs with encoder fully frozen (fine-tuning only) |
| `freeze_encoder` | bool | `false` | Must match `model.freeze_encoder` |
| `partial_unfreeze_epoch` | int | `10` | Epoch to fully unfreeze encoder |

### `EvidentialUncertaintyVisualizationCallback`

| Field | Type | Default | Description |
| --- | --- | --- | --- |
| `num_images` | int | `4` | Samples per logged grid |
| `log_every_n_epochs` | int | `5` | Log frequency |
| `norm_params` | dict | `null` | `{mean: [...], std: [...]}` for de-normalisation |

### `EvidentialInferenceProcessor`

| Field | Type | Default | Description |
| --- | --- | --- | --- |
| `output_uncertainty_path` | str | `null` | Path for uncertainty GeoTIFF output |
| `num_classes` | int | `2` | Number of classes K |
| `export_alpha` | bool | `false` | Also export K-band alpha parameters |
