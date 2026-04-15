---
sidebar_position: 5
title: DA Config Reference
---

# Domain Adaptation — Configuration Reference

Full reference for all fields in the `domain_adaptation` config section used by `DomainAdaptationModel`.

---

## Top-level config keys

| Key | Type | Required | Description |
|---|---|---|---|
| `pl_model._target_` | string | yes | Must be `pytorch_segmentation_models_trainer.model_loader.domain_adaptation_model.DomainAdaptationModel` |
| `model` | dict | yes | Segmentation network config (same as standard training) |
| `loss` | dict | yes | Segmentation loss config (applied to source domain only) |
| `optimizer` | dict | yes | Optimizer config |
| `hyperparameters` | dict | yes | Must contain at least `batch_size` and `epochs` |
| `pl_trainer` | dict | yes | PyTorch Lightning `Trainer` kwargs |
| `domain_adaptation` | dict | yes | DA-specific config (detailed below) |
| `scheduler_list` | list | no | LR schedulers — same format as standard training |
| `callbacks` | list | no | PL callbacks — same format as standard training |
| `logger` | dict | no | TensorBoard, W&B, CSV logger |
| `metrics` | list | no | `torchmetrics` metrics |

---

## `domain_adaptation` section

### `method`

The DA method to use. Hydra instantiates this as a `BaseDomainAdaptationMethod` subclass.

```yaml
domain_adaptation:
  method:
    _target_: my_package.methods.MyMethod   # required
    lambda_da: 1.0                          # default: 1.0
    # ... any extra kwargs accepted by your method's __init__
```

| Field | Type | Default | Description |
|---|---|---|---|
| `_target_` | string | — | Fully-qualified class name of the method |
| `lambda_da` | float | `1.0` | Global DA loss weight: `total = seg_loss + lambda_da * da_loss`. Used as fallback when no `lambda_schedule` is set on the method |

---

### `source_dataset`

Labeled source-domain training dataset.

```yaml
domain_adaptation:
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
```

| Field | Type | Default | Description |
|---|---|---|---|
| `_target_` | string | — | Dataset class to instantiate |
| `input_csv_path` | string | — | Path to the CSV listing image/mask pairs |
| `data_loader.batch_size` | int | `8` | Samples per batch |
| `data_loader.num_workers` | int | `4` | DataLoader worker processes |
| `data_loader.shuffle` | bool | `true` | Shuffle at each epoch |
| `data_loader.pin_memory` | bool | `true` | Pin tensors to CPU memory |
| `data_loader.drop_last` | bool | `true` | Drop incomplete final batch |
| `data_loader.prefetch_factor` | int | `2` | Batches prefetched per worker |
| `data_loader.persistent_workers` | bool | `false` | Keep workers alive between epochs |

---

### `target_dataset`

Unlabeled (or weakly labeled) target-domain training dataset. Same structure as `source_dataset`.

:::note
For UDA (unsupervised DA), the target dataset CSV may omit the mask column or point to empty masks. The `mask` key is only read from the target batch when `method.requires_target_labels = True`.
:::

---

### `source_val_dataset` *(optional)*

Labeled source-domain validation dataset. Used to detect catastrophic forgetting.

When omitted (`null`), source-domain validation is skipped and `DomainAdaptationMonitorCallback` will log a warning that forgetting monitoring is disabled.

```yaml
domain_adaptation:
  source_val_dataset:
    _target_: pytorch_segmentation_models_trainer.dataset_loader.dataset.SegmentationDataset
    input_csv_path: /data/source/val.csv
    data_loader:
      batch_size: 8
      num_workers: 4
      shuffle: false
      drop_last: false
```

---

### `target_val_dataset` *(optional)*

Labeled target-domain validation dataset. Used to measure adaptation progress.

:::tip
Even in UDA settings, it is common to keep a small labeled target validation set (not used during training) to measure true target performance.
:::

---

### `feature_layers` *(optional)*

List of dot-separated layer name strings whose outputs will be captured and passed to `compute_da_loss` as `source_features` / `target_features`.

Only active when `method.requires_features = True`. When empty and `requires_features = True`, a warning is logged and both feature dicts will be empty.

```yaml
domain_adaptation:
  feature_layers:
    - encoder.layer3
    - encoder.layer4
```

Use `dict(model.named_modules()).keys()` to inspect available layer names for your architecture.

---

### `pretrained_checkpoint` *(optional)*

Warm-start the segmentation model from an existing checkpoint before DA training begins. Loads weights only — optimizer state, epoch counter, and scheduler are reset.

```yaml
domain_adaptation:
  pretrained_checkpoint:
    path: /checkpoints/source_model.ckpt   # required
    source_format: pytorch_lightning        # "pytorch_lightning" or "pytorch"
    strict_loading: true
```

| Field | Type | Default | Description |
|---|---|---|---|
| `path` | string | — | Absolute path to the checkpoint file |
| `source_format` | string | `pytorch_lightning` | `"pytorch_lightning"`: reads `ckpt["state_dict"]` and strips `"model."` prefix. `"pytorch"`: reads file directly as state dict |
| `strict_loading` | bool | `true` | Passed to `load_state_dict(strict=...)`. Set `false` for partial loads |

:::caution Difference from `resume_from_checkpoint`
`pretrained_checkpoint` loads weights only. `pl_trainer.resume_from_checkpoint` (or `hyperparameters.resume_from_checkpoint`) restores the complete training state including optimizer and epoch counter. Use the former when starting a new adaptation phase; use the latter when resuming an interrupted training run.
:::

---

## Lambda Schedulers

Configure inside your method config under `lambda_schedule`. `DomainAdaptationModel` calls `method.lambda_schedule.get_lambda(epoch, total_epochs)` each step if the attribute exists.

### `ConstantScheduler`

Returns a fixed value throughout training.

```yaml
lambda_schedule:
  _target_: pytorch_segmentation_models_trainer.domain_adaptation.schedulers.ConstantScheduler
  value: 1.0
```

| Field | Default | Description |
|---|---|---|
| `value` | `1.0` | Constant lambda |

---

### `LinearScheduler`

Linearly interpolates from `start_value` to `end_value`.

```yaml
lambda_schedule:
  _target_: pytorch_segmentation_models_trainer.domain_adaptation.schedulers.LinearScheduler
  start_value: 0.0
  end_value: 1.0
```

| Field | Default | Description |
|---|---|---|
| `start_value` | `0.0` | Lambda at epoch 0 |
| `end_value` | `1.0` | Lambda at the final epoch |

---

### `DANNScheduler`

Sigmoid-shaped ramp from Ganin et al. (2016): `λ = 2 / (1 + exp(-γ·p)) - 1` where `p` is training progress in `[0, 1]`.

```yaml
lambda_schedule:
  _target_: pytorch_segmentation_models_trainer.domain_adaptation.schedulers.DANNScheduler
  gamma: 10.0
```

| Field | Default | Description |
|---|---|---|
| `gamma` | `10.0` | Steepness of the growth curve. Higher = faster transition |

---

## `DomainAdaptationMonitorCallback`

```yaml
callbacks:
  - _target_: pytorch_segmentation_models_trainer.domain_adaptation.callbacks.DomainAdaptationMonitorCallback
    num_classes: 2
    class_names: ["background", "building"]
    log_every_n_epochs: 1
    forgetting_threshold: 0.05
    eval_batch_size: 8
    eval_num_workers: 0
```

| Field | Default | Description |
|---|---|---|
| `num_classes` | — | Number of segmentation classes (required) |
| `class_names` | `["Class 0", …]` | Class names for logging |
| `log_every_n_epochs` | `1` | How often to evaluate both domains |
| `forgetting_threshold` | `0.05` | IoU drop (fraction) that triggers a warning |
| `eval_batch_size` | `8` | Batch size for evaluation forward passes |
| `eval_num_workers` | `0` | DataLoader workers for evaluation (0 = main process) |

Metrics logged by this callback:

| Metric | Description |
|---|---|
| `iou/source_val` | Mean IoU on source validation set |
| `iou/target_val` | Mean IoU on target validation set (shown in progress bar) |
| `iou/gap_source_minus_target` | Difference between source and target IoU |
| `iou/source_drop_from_baseline` | IoU drop since epoch 0 on source domain |
| `iou/source_baseline` | Baseline IoU recorded at `on_fit_start` |
