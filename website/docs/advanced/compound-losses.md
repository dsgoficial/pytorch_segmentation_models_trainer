---
sidebar_position: 1
---

# Compound Loss Functions

This guide covers the full loss system in `pytorch_segmentation_models_trainer`, from the base `Loss` class to `MultiLoss` composition, weight scheduling, and distributed normalization.

## How `MultiLoss` Combines Losses

`MultiLoss` accepts a list of individual loss functions and a list of corresponding weights, then computes a weighted sum at each forward pass:

```
total_loss = sum(weight_i * loss_i(pred, gt) for each loss_i, weight_i)
```

Each individual loss is optionally normalized by a running average of its own magnitude (see [Distributed Normalization](#distributed-normalization) below) before being weighted. The `forward` method returns three values:

- `total_loss` — the scalar weighted sum
- `individual_losses_dict` — a `{name: value}` dict of each component (detached)
- `extra_dict` — a `{name: extra_info}` dict of any extra tensors the losses set in `self.extra_info`

```python
total_loss, individual_losses, extra_info = multi_loss(pred_batch, gt_batch, normalize=True, epoch=current_epoch)
```

## The Three Dispatch Paths in `Model.get_loss_function()`

`Model.get_loss_function()` checks config keys in priority order:

| Priority | Config key | Description |
|----------|-----------|-------------|
| 1 (highest) | `loss_params.compound_loss` | New flexible YAML-based compound loss |
| 2 | `loss_params.multi_loss` | Legacy multi-loss configuration (backward compatible) |
| 3 (lowest) | `loss` | Simple single loss, instantiated directly |

```python
# Path 1 — compound loss (recommended)
if hasattr(self.cfg, 'loss_params') and hasattr(self.cfg.loss_params, 'compound_loss'):
    return build_compound_loss_from_config(self.cfg.loss_params.compound_loss)

# Path 2 — legacy multi_loss
if hasattr(self.cfg, 'loss_params') and hasattr(self.cfg.loss_params, 'multi_loss'):
    return build_loss_from_config(self.cfg)

# Path 3 — simple loss
if "loss" in self.cfg:
    return instantiate(self.cfg.loss, _recursive_=False)
```

If none of the keys are present, a `ValueError` is raised.

## The `LossWrapper` Class

`LossWrapper` wraps any standard `torch.nn.Module` loss so it is compatible with the `MultiLoss` interface (i.e., it accepts `pred_batch`/`gt_batch` dicts and returns `(loss, extra_info)`).

```python
from pytorch_segmentation_models_trainer.custom_losses.base_loss import LossWrapper
import torch.nn as nn

wrapped = LossWrapper(name="ce", loss_func=nn.CrossEntropyLoss())
loss_val, extra = wrapped(pred_batch, gt_batch)
```

This allows you to mix custom `Loss` subclasses and standard PyTorch losses within the same `MultiLoss`.

## `SegLoss`: BCE + Dice Compound Loss

`SegLoss` is the primary segmentation loss. It combines Binary Cross-Entropy (or Cross-Entropy for multi-class) and Dice loss using configurable coefficients.

```python
from pytorch_segmentation_models_trainer.custom_losses.base_loss import SegLoss

seg_loss = SegLoss(
    name="seg",
    gt_channel_selector=slice(0, 2),  # which channels of gt_polygons_image to use
    n_classes=2,
    bce_coef=0.5,
    dice_coef=0.5,
    tversky_focal_coef=0.0,
    use_mixed_precision=False,
)
```

### Constructor Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `name` | `str` | required | Name used for logging and norm tracking |
| `gt_channel_selector` | `int` or `slice` | required | Index/slice selecting channels from `gt_polygons_image` to compare against `seg` predictions |
| `n_classes` | `int` | `2` | Number of output classes; binary uses `F.binary_cross_entropy`, multi-class uses `nn.CrossEntropyLoss` |
| `bce_coef` | `float` | `0.5` | Weight of the cross-entropy term |
| `dice_coef` | `float` | `0.5` | Weight of the Dice loss term |
| `tversky_focal_coef` | `float` | `0` | Weight of the focal Tversky loss term (disabled when 0) |
| `use_mixed_precision` | `bool` | `False` | When `True`, uses `F.binary_cross_entropy_with_logits` instead of `F.binary_cross_entropy` for AMP stability |
| `use_mixup` | `bool` | `False` | Enable MixUp augmentation loss |
| `mixup_alpha` | `float` | `0.5` | Alpha parameter for MixUp Beta distribution |
| `use_label_smooth` | `bool` | `False` | Enable label smoothing |
| `smooth_factor` | `float` | `0.0` | Label smoothing factor |

The computed loss is:

```
loss = bce_coef * mean_bce + dice_coef * mean_dice + tversky_focal_coef * mean_focal_tversky
```

Expected batch keys:
- `pred_batch["seg"]` — shape `(N, C_pred, H, W)`
- `gt_batch["gt_polygons_image"]` — shape `(N, C_gt, H, W)`; `gt_channel_selector` picks which of the `C_gt` channels to use

## Weight Scheduling

Each weight in `MultiLoss` can be either a fixed scalar or a **list of values** corresponding to `epoch_thresholds`. When a list is provided, scipy's `interp1d` interpolates between threshold values at each epoch.

```python
MultiLoss(
    loss_funcs=[loss_a, loss_b],
    weights=[
        1.0,                     # fixed scalar for loss_a
        [0.0, 0.0, 1.0, 1.0],   # per-epoch schedule for loss_b
    ],
    epoch_thresholds=[0, 10, 20, 50],
)
```

In the example above, `loss_b` starts with weight 0 and linearly ramps up to 1.0 between epoch 10 and epoch 20, then stays at 1.0. At epochs outside the specified range, the boundary values are used (`fill_value=(weight[0], weight[-1])`).

During `forward`, the current `epoch` must be passed to resolve scheduled weights:

```python
total_loss, losses, extra = multi_loss(pred, gt, epoch=self.current_epoch)
```

### YAML Configuration Example

```yaml
loss_params:
  compound_loss:
    normalize_losses: true
    normalization_params:
      max_samples: 1000
    loss_list:
      - _target_: pytorch_segmentation_models_trainer.custom_losses.base_loss.SegLoss
        name: seg
        gt_channel_selector: 0
        bce_coef: 0.5
        dice_coef: 0.5
        weight: 1.0

      - _target_: pytorch_segmentation_models_trainer.custom_losses.base_loss.CrossfieldAlignLoss
        name: crossfield_align
        weight:
          - 0.0
          - 0.0
          - 1.0
          - 1.0

    epoch_thresholds: [0, 5, 10, 50]
```

## Distributed Normalization

Each `Loss` instance carries a `norm` parameter that tracks a running average of the loss magnitude. Before contributing to the weighted sum, the raw loss value is divided by `norm`. This keeps all loss components at roughly the same scale regardless of their natural magnitudes.

### Methods

| Method | Description |
|--------|-------------|
| `Loss.update_norm(pred_batch, gt_batch, nums)` | Compute the raw loss for a batch and update the running average in `norm_meter`; updates `norm` in place |
| `Loss.reset_norm()` | Reset `norm_meter` and set `norm` to 1.0; called before re-computing normalization |
| `Loss.sync(world_size)` | All-reduce `norm` across all GPUs via `torch.distributed.all_reduce`, then divide by `world_size`; safe no-op if distribution is unavailable |

`MultiLoss` delegates these methods to all its member losses:

```python
# At the start of training (ComputeWeightNormLossesCallback does this automatically):
multi_loss.reset_norm()
for batch in dataloader:
    pred = model(batch["image"])
    multi_loss.update_norm(pred, batch, batch["image"].shape[0])

# After computing norms on GPU 0, sync across all ranks:
multi_loss.sync(world_size=trainer.world_size)
```

The `ComputeWeightNormLossesCallback` and `FrameFieldComputeWeightNormLossesCallback` automate this workflow at the start of training. See the [Callbacks API reference](../api/callbacks.md) for details.
