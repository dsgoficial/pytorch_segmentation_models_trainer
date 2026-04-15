---
sidebar_position: 6
title: Dual-Head Training
---

# Dual-Head Training

`UPerNetDualHead` extends UPerNet with **two independent decoders** sharing a single encoder. The two heads are supervised with different label types simultaneously:

- **Head A** — supervised with hard labels (integer class indices, e.g. aggregated annotations).
- **Head B** — supervised with soft labels (float probabilities, e.g. Dawid–Skene posteriors or annotator-uncertainty posteriors).

A consistency loss couples the two heads during training, encouraging them to agree while allowing each to specialise.

---

## Architecture

```text
encoder (shared)
    ├── decoder_A (UPerNet) → seg_head_A → logits_A   (hard labels)
    └── decoder_B (UPerNet) → seg_head_B → logits_B   (soft labels)
```

`forward()` returns the primary head output selected by `inference_head`. Intermediate logits from both heads are stored as `last_logits_A` / `last_logits_B` so the loss function can access them without changing the standard `forward() → loss(pred, target)` interface.

---

## When to Use

Dual-head training is beneficial when:

- You have **two label sources** for the same data: hard expert-aggregated masks and soft probabilistic labels from a Bayesian annotation model (e.g. Dawid–Skene).
- You want a model that is calibrated to annotator uncertainty while also being decisive at inference time.
- You want to regularise a hard-label model with a soft-label auxiliary objective.

---

## Model Config

```yaml
model:
  _target_: pytorch_segmentation_models_trainer.custom_models.upernet_dual_head.UPerNetDualHead
  encoder_name: tu-convnextv2_tiny.fcmae_ft_in22k_in1k_384
  encoder_weights: imagenet
  in_channels: 3
  classes: 6
  decoder_channels: 256
  inference_head: average   # "A", "B", or "average"
```

### Parameters

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `encoder_name` | `str` | `"tu-convnextv2_tiny..."` | timm/SMP encoder name. |
| `encoder_depth` | `int` | `5` | Number of encoder stages. |
| `encoder_weights` | `str` | `"imagenet"` | Pretrained weights identifier. |
| `decoder_channels` | `int` | `256` | Intermediate channels per decoder. |
| `in_channels` | `int` | `3` | Input image bands. |
| `classes` | `int` | `6` | Number of output segmentation classes. |
| `inference_head` | `str` | `"average"` | Head used at inference: `"A"` (hard-trained), `"B"` (soft-trained), or `"average"` (mean of both logits). |

---

## Dataset Configuration

The training dataset must supply both hard and soft masks. `RandomCropSegmentationDataset` with `soft_labels: true` provides the required output format.

```yaml
train_dataset:
  _target_: pytorch_segmentation_models_trainer.dataset_loader.dataset.RandomCropSegmentationDataset
  input_csv_path: /data/dual_label/train.csv
  crop_size: 512
  samples_per_epoch: 8000
  n_classes: 6
  soft_labels: true       # enables float soft-label output alongside hard masks
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

The CSV should include a `soft_mask` column in addition to the standard `image` and `mask` columns:

```csv
image,mask,soft_mask
/data/scenes/area_a.tif,/data/masks/hard/area_a.tif,/data/masks/soft/area_a.tif
```

---

## Loss Configuration

The `_shared_step` training loop detects dual-head models automatically and routes Head A to the hard-label loss and Head B to the soft-label loss. A consistency regularisation term between the two heads is also computed internally.

```yaml
loss:
  _target_: segmentation_models_pytorch.losses.DiceLoss
  mode: multiclass
```

No special dual-head loss configuration is needed; the consistency coupling is handled inside `_shared_step`.

---

## Inference

At inference time only one head is active, controlled by `inference_head`:

| `inference_head` | Behaviour |
| --- | --- |
| `"A"` | Returns logits from the hard-label head. Most decisive. |
| `"B"` | Returns logits from the soft-label head. Better calibrated. |
| `"average"` | Returns the mean of both heads' logits (default). Balances decisiveness and calibration. |

Change the inference head without retraining by editing the YAML and reloading the checkpoint:

```yaml
model:
  inference_head: B   # switch to soft-label head at inference
```

---

## Full Training Example

```yaml
model:
  _target_: pytorch_segmentation_models_trainer.custom_models.upernet_dual_head.UPerNetDualHead
  encoder_name: tu-convnextv2_tiny.fcmae_ft_in22k_in1k_384
  encoder_weights: imagenet
  in_channels: 3
  classes: 6
  decoder_channels: 256
  inference_head: average

loss:
  _target_: segmentation_models_pytorch.losses.DiceLoss
  mode: multiclass

train_dataset:
  _target_: pytorch_segmentation_models_trainer.dataset_loader.dataset.RandomCropSegmentationDataset
  input_csv_path: /data/dual_label/train.csv
  crop_size: 512
  samples_per_epoch: 8000
  n_classes: 6
  soft_labels: true
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

val_dataset:
  _target_: pytorch_segmentation_models_trainer.dataset_loader.dataset.RandomCropSegmentationDataset
  input_csv_path: /data/dual_label/val.csv
  crop_size: 512
  grid_mode: true
  n_classes: 6
  augmentation_list:
    - _target_: albumentations.Normalize
      mean: [0.485, 0.456, 0.406]
      std: [0.229, 0.224, 0.225]
    - _target_: albumentations.pytorch.ToTensorV2
  data_loader:
    shuffle: false
    num_workers: 4
    batch_size: 16
    drop_last: false

callbacks:
  - _target_: pytorch_segmentation_models_trainer.custom_callbacks.training_callbacks.EMACallback
    decay: 0.999
  - _target_: pytorch_lightning.callbacks.ModelCheckpoint
    monitor: loss/val
    mode: min
    save_top_k: 3
```
