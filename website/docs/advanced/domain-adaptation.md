---
sidebar_position: 3
title: Domain Adaptation
---

# Domain Adaptation

Domain adaptation (DA) lets you take a segmentation model trained on a **labeled source domain** and transfer its knowledge to an **unlabeled (or weakly labeled) target domain**, without requiring full annotations in the target.

This is common in remote sensing: a model trained on images from one sensor, region, or time period often degrades significantly when applied to images from a different sensor, region, or season.

---

## Architecture Overview

The DA module follows the same patterns used in the rest of the framework:

- **`DomainAdaptationModel`** (`model_loader/domain_adaptation_model.py`) is the `pl.LightningModule` that orchestrates the full training loop.
- **`BaseDomainAdaptationMethod`** (`domain_adaptation/base_method.py`) is a plain `nn.Module` owned by `DomainAdaptationModel` — it encapsulates the DA-specific loss computation.

This mirrors the `FrameFieldSegmentationPLModel` / `FrameFieldModel` pattern: the LightningModule handles the training infrastructure, the `nn.Module` handles the domain-specific logic.

```
DomainAdaptationModel (LightningModule)
├── self.model                    ← segmentation network (SMP, torchvision, …)
├── self.method                   ← BaseDomainAdaptationMethod (nn.Module)
│     ├── compute_da_loss()       ← your implementation lives here
│     └── get_extra_parameter_groups()
├── CombinedLoader(source, target)
└── configure_optimizers()
```

---

## Training Loop

Each training step:

1. Source batch and target batch arrive from a `CombinedLoader`.
2. Both batches are forwarded through `self.model`.
3. **Segmentation loss** is computed on the source batch only (labeled).
4. **DA loss** is computed by `self.method.compute_da_loss(...)`.
5. Losses are combined: `total = seg_loss + λ * da_loss`.
6. All scalars are logged to TensorBoard / W&B.

---

## Getting Started

### 1. Write your config

Create a YAML config with a `domain_adaptation` section. The entry point is the same `train.py` used for all other models:

```yaml
pl_model:
  _target_: pytorch_segmentation_models_trainer.model_loader.domain_adaptation_model.DomainAdaptationModel

model:
  _target_: segmentation_models_pytorch.FPN
  encoder_name: resnet50
  encoder_weights: imagenet
  in_channels: 3
  classes: 2

loss:
  _target_: torch.nn.CrossEntropyLoss

optimizer:
  _target_: torch.optim.Adam
  lr: 1e-4

domain_adaptation:
  method:
    _target_: my_package.methods.MyMethod
    lambda_da: 1.0

  source_dataset:
    _target_: pytorch_segmentation_models_trainer.dataset_loader.dataset.SegmentationDataset
    input_csv_path: /data/source/train.csv
    data_loader:
      batch_size: 8
      num_workers: 4
      shuffle: true
      pin_memory: true
      drop_last: true
      prefetch_factor: 2
      persistent_workers: false

  target_dataset:
    _target_: pytorch_segmentation_models_trainer.dataset_loader.dataset.SegmentationDataset
    input_csv_path: /data/target/train.csv   # sem rótulos
    data_loader:
      batch_size: 8
      num_workers: 4
      shuffle: true
      pin_memory: true
      drop_last: true
      prefetch_factor: 2
      persistent_workers: false

  # Validation sets — optional but strongly recommended
  source_val_dataset:
    _target_: pytorch_segmentation_models_trainer.dataset_loader.dataset.SegmentationDataset
    input_csv_path: /data/source/val.csv
    data_loader:
      batch_size: 8
      num_workers: 4
      shuffle: false
      drop_last: false

  target_val_dataset:
    _target_: pytorch_segmentation_models_trainer.dataset_loader.dataset.SegmentationDataset
    input_csv_path: /data/target/val.csv
    data_loader:
      batch_size: 8
      num_workers: 4
      shuffle: false
      drop_last: false

  feature_layers: []   # e.g. ["encoder.layer3"] if method.requires_features = True
  pretrained_checkpoint: null

hyperparameters:
  batch_size: 8
  epochs: 50

pl_trainer:
  max_epochs: ${hyperparameters.epochs}
  accelerator: gpu
  devices: 1
  default_root_dir: /experiments/my_da_run

callbacks:
  - _target_: pytorch_lightning.callbacks.LearningRateMonitor
  - _target_: pytorch_lightning.callbacks.ModelCheckpoint
    monitor: iou/target_val
    mode: max
    save_top_k: 3
  - _target_: pytorch_segmentation_models_trainer.domain_adaptation.callbacks.DomainAdaptationMonitorCallback
    num_classes: 2
    class_names: ["background", "building"]
    log_every_n_epochs: 1
    forgetting_threshold: 0.05
```

### 2. Run training

```bash
python -m pytorch_segmentation_models_trainer.train \
  --config-path /path/to/conf \
  --config-name my_da_experiment
```

---

## Warm-Starting from a Source Checkpoint

If you have a model already trained on the source domain, you can use it as a starting point. This loads only the model weights — optimizer state, epoch counter, and scheduler are reset, which is what you want for a new adaptation phase.

Add this section to your config:

```yaml
domain_adaptation:
  pretrained_checkpoint:
    path: /checkpoints/source_model.ckpt
    source_format: pytorch_lightning   # or "pytorch" for .pt/.pth files
    strict_loading: true
```

`source_format` controls how the state dict is read:

| Value | Expected format |
|---|---|
| `pytorch_lightning` | `.ckpt` file saved by PL. Weights are under `state_dict` with `model.` prefix. |
| `pytorch` | Plain `.pt` / `.pth` file. Keys are model parameter names with no prefix. |

Set `strict_loading: false` if the checkpoint has a slightly different architecture (e.g. different number of output classes).

This is **different from `resume_from_checkpoint`** in the `pl_trainer` section, which resumes the full training state including optimizer and epoch counter.

---

## Monitoring Training

Add the `DomainAdaptationMonitorCallback` to your `callbacks` list to track two key risks:

**Adaptation progress** — is the model learning on the target domain?

**Catastrophic forgetting** — is the model degrading on the source domain?

The callback logs these scalars every epoch:

| Metric | Meaning |
|---|---|
| `iou/target_val` | IoU on target validation set — should rise |
| `iou/source_val` | IoU on source validation set — should stay stable |
| `iou/gap_source_minus_target` | Domain gap — should shrink |
| `iou/source_drop_from_baseline` | Forgetting indicator — should stay near 0 |

A console warning is printed when `source_drop_from_baseline > forgetting_threshold`.

```yaml
callbacks:
  - _target_: pytorch_segmentation_models_trainer.domain_adaptation.callbacks.DomainAdaptationMonitorCallback
    num_classes: 2
    class_names: ["background", "building"]
    log_every_n_epochs: 1
    forgetting_threshold: 0.05   # warn if source IoU drops more than 5 pp
```

---

## Lambda Schedulers

The DA loss weight `λ` can evolve over training. Configure it inside your method config:

```yaml
domain_adaptation:
  method:
    _target_: my_package.methods.MyMethod
    lambda_da: 1.0         # used as fallback if no lambda_schedule is set
    lambda_schedule:
      _target_: pytorch_segmentation_models_trainer.domain_adaptation.schedulers.DANNScheduler
      gamma: 10.0
```

Available schedulers:

| Class | Behavior |
|---|---|
| `ConstantScheduler` | Fixed value throughout training |
| `LinearScheduler` | Linear ramp from `start_value` to `end_value` |
| `DANNScheduler` | Sigmoid-shaped ramp from Ganin et al. (2016): `2/(1+exp(-γ·p)) - 1` |

---

## Next Steps

- [Implementing a DA Method](domain-adaptation-implementing-methods.md) — how to write your own method class
- [Configuration Reference](domain-adaptation-config-reference.md) — all available config fields
