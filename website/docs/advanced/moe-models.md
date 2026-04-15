---
sidebar_position: 5
title: Mixture of Experts Models
---

# Mixture of Experts (MoE) Models

Two UPerNet variants replace selected decoder convolutions with **Mixture of Experts (MoE)** blocks, adding conditional computation capacity without a proportional increase in inference-time FLOPs.

---

## Overview

| Model | Class | Key feature |
| --- | --- | --- |
| `UPerNetMoE` | `custom_models/upernet_moe.py` | Sparse MoE at fusion and/or FPN blocks |
| `UPerNetMEDoE` | `custom_models/upernet_medoe.py` | Multiple-Expert Dropout of Experts — MoE with structured expert dropout for better regularisation |

Both models follow the same encoder–decoder–head pattern as `smp.SegmentationModel` and are fully compatible with the standard training pipeline.

---

## UPerNetMoE

Replaces the UPerNet decoder's fusion block and/or FPN conv blocks with `MoEConv2dReLU` — a drop-in replacement for `Conv2dReLU` that routes each spatial position to a subset of expert convolutions.

### Model Config

```yaml
model:
  _target_: pytorch_segmentation_models_trainer.custom_models.upernet_moe.UPerNetMoE
  encoder_name: resnet50
  encoder_weights: imagenet
  in_channels: 3
  classes: 5
  decoder_channels: 256
  moe_num_experts: 8       # total number of experts per MoE block
  moe_top_k: 2             # active experts per position (token_choice)
  moe_routing: token_choice  # "token_choice" or "expert_choice"
  moe_aux_loss_weight: 0.01  # weight for load-balancing auxiliary loss
  moe_at_fusion: true      # apply MoE at the fusion block
  moe_at_fpn: false        # apply MoE at FPN conv blocks (higher cost)
  moe_use_shared_expert: false  # add a shared dense expert per block
```

### MoE Parameters

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `moe_num_experts` | `int` | `8` | Total number of expert convolutions per MoE block. |
| `moe_top_k` | `int` | `2` | Number of experts activated per spatial position (`token_choice`). |
| `moe_noise_std` | `float` | `1.0` | Noise injected to gating logits during training (encourages exploration). |
| `moe_capacity_factor` | `float` | `1.25` | Expert capacity multiplier for `expert_choice` routing. |
| `moe_use_shared_expert` | `bool` | `False` | Add a shared dense expert alongside the sparse experts. |
| `moe_routing` | `str` | `"token_choice"` | Routing algorithm: `"token_choice"` (each token picks top-k experts) or `"expert_choice"` (each expert picks top-k tokens). |
| `moe_aux_loss_weight` | `float` | `0.01` | Weight for the load-balancing auxiliary loss. Set to `0` when using `expert_choice`. |
| `moe_at_fusion` | `bool` | `True` | Replace the fusion block with a MoE block. |
| `moe_at_fpn` | `bool` | `False` | Replace FPN conv blocks with MoE blocks (increases capacity but raises cost). |

### Auxiliary Loss

The load-balancing auxiliary loss (`moe_aux_loss_weight`) discourages expert collapse (all tokens routing to the same expert). It is automatically detected and added to the main task loss by `_shared_step` — no manual loss configuration is required.

```yaml
# No special loss config needed; aux loss is handled automatically.
loss:
  _target_: segmentation_models_pytorch.losses.DiceLoss
  mode: multiclass
```

:::tip token_choice vs expert_choice
- **token_choice**: each token selects its top-k experts. Load imbalance can occur; address with `moe_noise_std > 0` and `moe_aux_loss_weight > 0`.
- **expert_choice**: each expert selects its top-k tokens. Guaranteed perfect load balance; set `moe_aux_loss_weight: 0`.
:::

---

## UPerNetMEDoE

MEDoE (Multiple-Expert Dropout of Experts) extends `UPerNetMoE` with structured expert dropout during training: a random subset of experts is dropped per forward pass, forcing the remaining experts to cover the full input distribution. This improves regularisation and reduces reliance on any single expert.

### Model Config

```yaml
model:
  _target_: pytorch_segmentation_models_trainer.custom_models.upernet_medoe.UPerNetMEDoE
  encoder_name: resnet50
  encoder_weights: imagenet
  in_channels: 3
  classes: 5
  decoder_channels: 256
  moe_num_experts: 8
  moe_top_k: 2
  moe_routing: token_choice
  moe_aux_loss_weight: 0.01
  moe_at_fusion: true
  moe_at_fpn: false
  medoe_drop_rate: 0.1    # fraction of experts dropped per forward pass
```

### Additional MEDoE Parameter

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `medoe_drop_rate` | `float` | `0.1` | Fraction of experts randomly dropped during training. Has no effect at inference. |

### MEDoE Diagnostics Logging

During training, `_shared_step` automatically logs per-expert activation statistics when a MEDoE model is detected:

```
extra/train_medoe_expert_utilization
extra/train_medoe_expert_entropy
```

These help diagnose expert collapse or imbalanced routing during training.

---

## Full Training Example

```yaml
model:
  _target_: pytorch_segmentation_models_trainer.custom_models.upernet_moe.UPerNetMoE
  encoder_name: efficientnet-b4
  encoder_weights: imagenet
  in_channels: 3
  classes: 6
  decoder_channels: 256
  moe_num_experts: 8
  moe_top_k: 2
  moe_routing: token_choice
  moe_aux_loss_weight: 0.01
  moe_at_fusion: true
  moe_at_fpn: false

loss:
  _target_: segmentation_models_pytorch.losses.DiceLoss
  mode: multiclass

train_dataset:
  _target_: pytorch_segmentation_models_trainer.dataset_loader.dataset.RandomCropSegmentationDataset
  input_csv_path: /data/train.csv
  crop_size: 512
  samples_per_epoch: 8000
  n_classes: 6
  augmentation_list:
    - _target_: albumentations.HorizontalFlip
      p: 0.5
    - _target_: albumentations.Normalize
      mean: [0.485, 0.456, 0.406]
      std: [0.229, 0.224, 0.225]
    - _target_: albumentations.pytorch.ToTensorV2
  data_loader:
    shuffle: true
    num_workers: 8
    batch_size: 16
    drop_last: true
```
