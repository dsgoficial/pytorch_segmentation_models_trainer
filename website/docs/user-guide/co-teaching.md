---
sidebar_position: 22
title: Co-Teaching Training
---

# Co-Teaching Training

Co-teaching is a noise-robust training strategy for weakly supervised
segmentation. It runs **two networks with independent initialisation** and
has each network teach the other using only its own most-confident samples.

This implementation follows Xiao et al. (2026, JSTARS):
*"Distilling 10-m Land Cover Maps from Multi-Source Consensus via AlphaEarth
Embeddings and Noise-Aware Weak Supervision."*

:::info Prerequisites
Co-teaching requires P_soft and W_conf rasters produced by the
[Soft-Label Training](soft-label-training.md) pipeline.

To ablate the border-distance contribution, generate two W_conf sets — one
with `--no-border` (original paper) and one without — and train
`CoTeachingModel` on each.  See [W_conf formula](soft-label-training.md#w_conf-formula).
:::

---

## Components

| Component | Class | Purpose |
|-----------|-------|---------|
| Scheduler | `CurriculumScheduler` | Ramps sample retention rate P_e from P_start → P_end over E_warm epochs |
| Loss | `CoTeachingLoss` | Curriculum masking + class-balanced cross-update + optional L_reg |
| Model | `CoTeachingModel` | Dual-branch LightningModule with manual optimization |

---

## How it works

### 1 — Confidence-guided curriculum masking (eq. 6-7)

At epoch *e*:

```
P_e = min(P_start + (e / E_warm) × (P_end - P_start), P_end)
τ   = Percentile(W_conf, 1 - P_e)
M_curr,i = I(W_conf,i > τ)
```

Early training retains only the 50 % most-confident pixels; by epoch 20 the
threshold relaxes to 95 % (configurable via `p_start`, `p_end`, `e_warm`).

### 2 — Class-balanced low-loss selection (eq. 9)

Within the retained pixels, each network ranks samples by current loss per
class and keeps only the `(1 - R_e)` fraction with the **lowest** loss:

```
J_A = ⋃_c argtop-k_c { ℓ_A(x_i) | x_i ∈ B_c }
```

Forget rate R_e ramps linearly from 0 to 0.3 over E_warm epochs.

### 3 — Symmetric cross-update (eq. 11)

fθA is optimized on J_B (B's low-loss selection) and vice versa:

```
L_total = Σ_{i∈J_B} W_conf · L_CE^A(x_i)
        + Σ_{j∈J_A} W_conf · L_CE^B(x_j)
        + λ · L_reg
```

### 4 — Neighbourhood regularization (eq. 10, optional)

```
L_reg = Σ_i Σ_{j∈N(i)} max(0, S_ij) · KL(p(·|x_i) || p(·|x_j))
```

Where S_ij = cosine similarity between input feature vectors at pixels i and j
in the 3×3 spatial neighbourhood. Penalizes prediction inconsistency between
feature-similar pixels. Controlled by `lambda_reg` (set to 0 to disable).

---

## Configuration

```yaml title="conf/examples/co_teaching_unet.yaml"
_target_: pytorch_segmentation_models_trainer.model_loader.co_teaching_model.CoTeachingModel

curriculum:
  e_warm: 20
  p_start: 0.5
  p_end: 0.95

loss:
  _target_: pytorch_segmentation_models_trainer.custom_losses.co_teaching_loss.CoTeachingLoss
  name: co_teaching
  num_classes: 4
  e_warm: 20
  p_start: 0.5
  p_end: 0.95
  lambda_reg: 0.1   # set to 0 to disable neighbourhood regularization

model:
  _target_: segmentation_models_pytorch.Unet
  encoder_name: resnet34
  encoder_weights: imagenet
  in_channels: 3
  classes: 4

optimizer:
  _target_: torch.optim.AdamW
  lr: 1.0e-4
  weight_decay: 1.0e-4
```

:::note Two optimizers
`CoTeachingModel` uses PyTorch Lightning **manual optimization** with one
`optimizer` config shared between both branches. Each branch (fθA = `model`,
fθB = `model_b`) receives an independent optimizer instance over its own
parameters.
:::

---

## API Reference

### `CurriculumScheduler`

```python
from pytorch_segmentation_models_trainer.custom_losses.co_teaching_loss import (
    CurriculumScheduler,
)

sched = CurriculumScheduler(e_warm=20, p_start=0.5, p_end=0.95)

# Retention rate at epoch e
p_e = sched.retention_rate(epoch=10)   # 0.725

# Dynamic threshold and binary mask from W_conf
mask = sched.curriculum_mask(w_conf, epoch=10)  # (B, 1, H, W) float
```

### `CoTeachingLoss`

```python
from pytorch_segmentation_models_trainer.custom_losses.co_teaching_loss import (
    CoTeachingLoss,
)

loss_fn = CoTeachingLoss(name="cot", num_classes=4, lambda_reg=0.1)

# Validation path (single logits tensor)
loss = loss_fn.compute(logits, {"mask": p_soft, "w_conf": w_conf})

# Training path (co-teaching dict)
loss_fn.current_epoch = 5
loss = loss_fn.compute(
    {"logits_a": logits_a, "logits_b": logits_b, "features": images},
    {"mask": p_soft, "w_conf": w_conf},
)

# Neighbourhood regularization only
l_reg = loss_fn.compute_neighborhood_reg(logits, features)
```

### `CoTeachingModel`

Drop-in replacement for `SoftLabelModel` with dual-branch training.
Validation and inference always use fθA (`self.model`).

```yaml
_target_: pytorch_segmentation_models_trainer.model_loader.co_teaching_model.CoTeachingModel
```

---

## Hyperparameter guidance

| Hyperparameter | Paper default | Effect |
|---------------|---------------|--------|
| `e_warm` | 20 | Longer warm-up → gentler noise filtering early on |
| `p_start` | 0.5 | Lower → more aggressive filtering at epoch 0 |
| `p_end` | 0.95 | Lower (e.g. 0.8) → always filter bottom 20%; slightly reduces OA |
| `lambda_reg` | 0.1 | Higher → stronger spatial smoothness constraint |

Setting `lambda_reg: 0` disables the neighbourhood regularization and reduces
co-teaching to the standard dual-network cross-update (Experiment variant CoT
in the paper).

---

## Experiment variants

| Variant | Description |
|---------|-------------|
| E0 | Hard labels, standard CE — baseline |
| E1 | `SoftLabelModel` + P_soft, no W_conf |
| E2 | `SoftLabelModel` + P_soft + W_conf |
| E4 | `SoftLabelModel` + AEF per-pixel embeddings |
| **CoT-Curr** | `CoTeachingModel` + curriculum masking only (`lambda_reg: 0`) |
| **CoT-Full** | `CoTeachingModel` + curriculum + class-balanced + L_reg (this doc) |
