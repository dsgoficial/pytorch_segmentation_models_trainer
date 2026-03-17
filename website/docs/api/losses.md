---
sidebar_position: 4
title: Loss Functions
---

# Loss Functions

This page is the API reference for all loss classes in `pytorch_segmentation_models_trainer`.

Base classes and segmentation/frame-field losses live in:

```python
from pytorch_segmentation_models_trainer.custom_losses.base_loss import <ClassName>
```

The `LossWrapper` and builder utilities live in:

```python
from pytorch_segmentation_models_trainer.custom_losses.loss_builder import LossWrapper
```

---

## `Loss`

```python
from pytorch_segmentation_models_trainer.custom_losses.base_loss import Loss
```

Abstract base class for all custom loss functions. Extends `torch.nn.Module`. Implements automatic loss normalization via a running average meter, and supports distributed training synchronization.

### Constructor

```python
Loss(name: str)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `name` | `str` | **required** | Human-readable name used for logging and `__repr__`. |

### `forward` Signature

```python
def forward(self, pred_batch, gt_batch, normalize: bool = True) -> Tuple[torch.Tensor, Dict]
```

Calls `self.compute(pred_batch, gt_batch)`, optionally divides the result by `self.norm[0]` (the running average of past loss values), then returns a `(loss_value, extra_info)` tuple.

- When `normalize=True` (default), the raw loss is divided by the running norm. The norm must be `> 1e-9`; an `AssertionError` is raised otherwise.
- `extra_info` is a dict populated by `self.compute()` with intermediate tensors useful for visualization.

### Normalization Methods

| Method | Description |
|--------|-------------|
| `reset_norm()` | Resets the running average meter to an initial value of `1`. |
| `update_norm(pred_batch, gt_batch, nums)` | Runs a forward pass (without gradient) and updates the running average with the result. Call this before training begins to calibrate the norm. |
| `sync(world_size)` | Performs an `all_reduce` to average `self.norm` across all GPUs when using distributed training. |

### Abstract Method

```python
def compute(self, pred_batch, gt_batch) -> torch.Tensor
```

Subclasses must implement this method. It should return a scalar tensor representing the raw (unnormalized) loss value.

### YAML Config Snippet

```yaml
loss:
  _target_: pytorch_segmentation_models_trainer.custom_losses.base_loss.SegLoss
  name: my_loss
  gt_channel_selector: 0
```

---

## `LossWrapper`

```python
from pytorch_segmentation_models_trainer.custom_losses.loss_builder import LossWrapper
```

Wraps any `torch.nn.Module` loss function to make it compatible with `MultiLoss`. This enables standard PyTorch losses (e.g. `nn.BCELoss`, `nn.CrossEntropyLoss`) and third-party losses to participate in the compound loss system without modification.

### Constructor

```python
LossWrapper(loss_func: nn.Module, name: str = None)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `loss_func` | `nn.Module` | **required** | The loss function to wrap. Can be a custom `Loss` subclass or any standard PyTorch/third-party loss module. |
| `name` | `str` | `None` | Name for the wrapper. If `None`, uses `loss_func.name` (if present) or `loss_func.__class__.__name__`. |

### `forward` Signature

```python
def forward(self, pred_batch, gt_batch, normalize: bool = True) -> Tuple[torch.Tensor, Dict]
```

- If the wrapped loss is a `Loss` subclass, delegates to `loss_func(pred_batch, gt_batch, normalize=normalize)`.
- If the wrapped loss is a standard PyTorch loss, calls `loss_func(pred_batch, gt_batch)` and returns `(loss_value, {})`.

### YAML Config Snippet

```yaml
compound_loss:
  losses:
    - loss:
        _target_: segmentation_models_pytorch.losses.DiceLoss
        mode: binary
      weight: 1.0
```

Non-custom losses are wrapped automatically by `build_compound_loss_from_config`.

---

## `MultiLoss`

```python
from pytorch_segmentation_models_trainer.custom_losses.base_loss import MultiLoss
```

Combines multiple loss functions with scalar or epoch-scheduled weights. Runs optional pre-processing steps before computing individual losses.

### Constructor

```python
MultiLoss(
    loss_funcs: List[nn.Module],
    weights: List[Union[float, List[float]]],
    epoch_thresholds: Optional[List[float]] = None,
    pre_processes: Optional[List[Callable]] = None,
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `loss_funcs` | `List[nn.Module]` | **required** | List of individual loss functions. Must be the same length as `weights`. |
| `weights` | `List[Union[float, List[float]]]` | **required** | Per-loss weights. Each entry can be a scalar `float` for a constant weight, or a `list` of floats for epoch-scheduled weights (interpolated via `scipy.interpolate.interp1d` against `epoch_thresholds`). |
| `epoch_thresholds` | `List[float]` | `None` | Epoch values corresponding to each position in a list-type weight. Required when any weight is a list. |
| `pre_processes` | `List[Callable]` | `None` | List of callables with signature `(pred_batch, gt_batch) -> (pred_batch, gt_batch)`. Called in order before any loss is computed. |

### `forward` Signature

```python
def forward(
    self,
    pred_batch,
    gt_batch,
    normalize: bool = True,
    epoch: Optional[float] = None,
) -> Tuple[torch.Tensor, Dict, Dict]
```

Returns `(total_loss, individual_losses_dict, extra_dict)`:

- `total_loss`: Weighted sum of all individual losses.
- `individual_losses_dict`: `{loss_name: scalar_tensor}` for each component.
- `extra_dict`: `{loss_name: extra_info_dict}` forwarded from each component's `forward`.

When `epoch` is provided and a weight is a scheduled list, the weight at that epoch is interpolated and applied.

### Weight Scheduling Example

```yaml
compound_loss:
  epoch_thresholds: [0, 5, 10]
  losses:
    - loss:
        _target_: pytorch_segmentation_models_trainer.custom_losses.base_loss.SegLoss
        name: seg
        gt_channel_selector: 0
        bce_coef: 0.5
        dice_coef: 0.5
      weight: [0.0, 0.5, 1.0]   # ramps from 0 to 1 over 10 epochs
    - loss:
        _target_: pytorch_segmentation_models_trainer.custom_losses.base_loss.CrossfieldAlignLoss
        name: crossfield_align
      weight: 1.0
```

---

## `SegLoss`

```python
from pytorch_segmentation_models_trainer.custom_losses.base_loss import SegLoss
```

Combined segmentation loss mixing Binary Cross Entropy (BCE) and Dice loss. For multi-class tasks, uses `torch.nn.CrossEntropyLoss`. Optionally supports mixed precision, MixUp augmentation, and label smoothing.

### Constructor

```python
SegLoss(
    name: str,
    gt_channel_selector: int,
    n_classes: int = 2,
    bce_coef: float = 0.5,
    dice_coef: float = 0.5,
    tversky_focal_coef: float = 0,
    use_mixed_precision: bool = False,
    use_mixup: bool = False,
    mixup_alpha: float = 0.5,
    use_label_smooth: bool = False,
    smooth_factor: float = 0.0,
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `name` | `str` | **required** | Loss name for logging. |
| `gt_channel_selector` | `int` or `slice` | **required** | Selects which channel(s) of `gt_batch["gt_polygons_image"]` to compare against `pred_batch["seg"]`. |
| `n_classes` | `int` | `2` | Number of classes. When `2`, uses `F.binary_cross_entropy`; otherwise uses `CrossEntropyLoss`. |
| `bce_coef` | `float` | `0.5` | Weight applied to the BCE/cross-entropy term. |
| `dice_coef` | `float` | `0.5` | Weight applied to the Dice loss term. |
| `tversky_focal_coef` | `float` | `0` | Weight applied to the focal Tversky loss term. `0` disables this term. |
| `use_mixed_precision` | `bool` | `False` | When `True`, uses `F.binary_cross_entropy_with_logits` instead of `F.binary_cross_entropy` (skip the sigmoid in the model head). |
| `use_mixup` | `bool` | `False` | When `True`, expects MixUp-augmented fields in `gt_batch` (`mixup_pred`, `mixup_y_a`, `mixup_y_b`, `mixup_lam`). |
| `mixup_alpha` | `float` | `0.5` | Alpha parameter for the Beta distribution used in MixUp. |
| `use_label_smooth` | `bool` | `False` | When `True`, applies label smoothing to the cross-entropy term. |
| `smooth_factor` | `float` | `0.0` | Smoothing factor for label smoothing. |

The final loss value is: `bce_coef * BCE + dice_coef * Dice + tversky_focal_coef * FocalTversky`.

### YAML Config Snippet

```yaml
compound_loss:
  losses:
    - loss:
        _target_: pytorch_segmentation_models_trainer.custom_losses.base_loss.SegLoss
        name: seg
        gt_channel_selector: 0
        n_classes: 2
        bce_coef: 0.5
        dice_coef: 0.5
      weight: 10.0
```

---

## `CrossfieldAlignLoss`

```python
from pytorch_segmentation_models_trainer.custom_losses.base_loss import CrossfieldAlignLoss
```

Frame-field alignment loss. Penalizes the deviation of the predicted frame field (`c0`, `c2`) from the ground-truth tangent field (`gt_field`) along polygon edges. The loss is masked to zero on non-edge pixels.

### Constructor

```python
CrossfieldAlignLoss(name: str)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `name` | `str` | **required** | Loss name for logging. |

### `compute` Signature

```python
def compute(self, pred_batch, gt_batch) -> torch.Tensor
```

Expected batch keys:

| Batch | Key | Description |
|-------|-----|-------------|
| `pred_batch` | `"crossfield"` | Tensor (N, 4, H, W): first 2 channels are `c0`, last 2 are `c2`. |
| `gt_batch` | `"gt_field"` | Complex-valued ground-truth tangent field (2-channel tensor). |
| `gt_batch` | `"gt_polygons_image"` | At least 2-channel mask; channel 1 is the edge mask used for spatial weighting. |

Stores `gt_batch["gt_field"]` in `self.extra_info` for visualization.

### YAML Config Snippet

```yaml
compound_loss:
  losses:
    - loss:
        _target_: pytorch_segmentation_models_trainer.custom_losses.base_loss.CrossfieldAlignLoss
        name: crossfield_align
      weight: 1.0
```

---

## `CrossfieldAlign90Loss`

```python
from pytorch_segmentation_models_trainer.custom_losses.base_loss import CrossfieldAlign90Loss
```

90-degree symmetry loss for frame fields. Enforces that the predicted frame field is also well-aligned when the ground-truth tangent is rotated 90°. This encourages right-angle corners. The loss is applied on edge pixels minus vertex pixels.

### Constructor

```python
CrossfieldAlign90Loss(name: str)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `name` | `str` | **required** | Loss name for logging. |

### `compute` Signature

```python
def compute(self, pred_batch, gt_batch) -> torch.Tensor
```

Expected batch keys:

| Batch | Key | Description |
|-------|-----|-------------|
| `pred_batch` | `"crossfield"` | Tensor (N, 4, H, W). |
| `gt_batch` | `"gt_field"` | 2-channel ground-truth tangent field. |
| `gt_batch` | `"gt_polygons_image"` | Exactly 3-channel mask (interior, edge, vertex). |

### YAML Config Snippet

```yaml
compound_loss:
  losses:
    - loss:
        _target_: pytorch_segmentation_models_trainer.custom_losses.base_loss.CrossfieldAlign90Loss
        name: crossfield_align90
      weight: 1.0
```

---

## `CrossfieldSmoothLoss`

```python
from pytorch_segmentation_models_trainer.custom_losses.base_loss import CrossfieldSmoothLoss
```

Spatial smoothness loss for frame fields. Penalizes high-frequency spatial variation in the frame field using a Laplacian penalty, but only in non-edge regions. This prevents the frame field from oscillating in homogeneous areas.

### Constructor

```python
CrossfieldSmoothLoss(name: str)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `name` | `str` | **required** | Loss name for logging. |

Internally creates a `frame_field_utils.LaplacianPenalty(channels=4)` module.

### `compute` Signature

```python
def compute(self, pred_batch, gt_batch) -> torch.Tensor
```

Expected batch keys:

| Batch | Key | Description |
|-------|-----|-------------|
| `pred_batch` | `"crossfield"` | Tensor (N, 4, H, W). |
| `gt_batch` | `"gt_polygons_image"` | At least 2-channel mask; channel 1 is the edge mask (inverted to define the smooth region). |

### YAML Config Snippet

```yaml
compound_loss:
  losses:
    - loss:
        _target_: pytorch_segmentation_models_trainer.custom_losses.base_loss.CrossfieldSmoothLoss
        name: crossfield_smooth
      weight: 0.1
```

---

## `ComputeSegGrads`

```python
from pytorch_segmentation_models_trainer.custom_losses.base_loss import ComputeSegGrads
```

Not a loss function. A **pre-processing callable** used inside `MultiLoss.pre_processes`. Computes spatial gradients of the predicted segmentation mask and populates `pred_batch` with derived tensors required by `SegCrossfieldLoss` and `SegEdgeInteriorLoss`.

### Constructor

```python
ComputeSegGrads(device)
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `device` | `str` or `torch.device` | Device on which to instantiate the `SpatialGradient` (Scharr filter) operator. |

### `__call__` Signature

```python
def __call__(self, pred_batch, gt_batch) -> Tuple[dict, dict]
```

Adds the following keys to `pred_batch`:

| Key | Shape | Description |
|-----|-------|-------------|
| `"seg_grads"` | (N, C, 2, H, W) | Spatial gradients of `pred_batch["seg"]` computed with the Scharr operator (scaled by 2). |
| `"seg_grad_norm"` | (N, C, H, W) | L2 norm of the spatial gradient at each pixel. |
| `"seg_grads_normed"` | (N, C, 2, H, W) | Unit-normalized gradient vectors (`seg_grads / (seg_grad_norm + 1e-6)`). |

### Usage in Config

`ComputeSegGrads` is instantiated programmatically by `build_combined_loss` when coupling losses require gradient information. When using `compound_loss`, add it via `pre_processes`:

```yaml
compound_loss:
  pre_processes:
    - _target_: pytorch_segmentation_models_trainer.custom_losses.base_loss.ComputeSegGrads
      device: cuda
  losses:
    - loss:
        _target_: pytorch_segmentation_models_trainer.custom_losses.base_loss.SegLoss
        name: seg
        gt_channel_selector: 0
      weight: 10.0
```

---

## Loss Configuration Patterns

The framework supports three configuration patterns for specifying losses in your Hydra config.

### Pattern 1: Single Loss

Use `loss` directly under the model config for a simple, single loss function:

```yaml
loss:
  _target_: pytorch_segmentation_models_trainer.custom_losses.base_loss.SegLoss
  name: seg
  gt_channel_selector: 0
  bce_coef: 0.5
  dice_coef: 0.5
```

### Pattern 2: Compound Loss (recommended)

Use `compound_loss` under `loss_params` to combine multiple losses with individual weights. Supports both custom `Loss` subclasses and standard PyTorch/third-party losses via automatic `LossWrapper` wrapping:

```yaml
loss_params:
  compound_loss:
    epoch_thresholds: [0, 5, 10]    # required if any weight is a list
    losses:
      - loss:
          _target_: pytorch_segmentation_models_trainer.custom_losses.base_loss.SegLoss
          name: seg
          gt_channel_selector: 0
          bce_coef: 0.5
          dice_coef: 0.5
        weight: 10.0

      - loss:
          _target_: pytorch_segmentation_models_trainer.custom_losses.base_loss.CrossfieldAlignLoss
          name: crossfield_align
        weight: [0.0, 0.5, 1.0]     # ramps up over 10 epochs

      - loss:
          _target_: segmentation_models_pytorch.losses.DiceLoss
          mode: binary
        weight: 1.0                  # third-party loss, auto-wrapped
```

### Pattern 3: Loss List (legacy `multiloss`)

Use `multi_loss` / `multiloss` under `loss_params` with the legacy `build_combined_loss` builder. This is the older configuration style and is maintained for backward compatibility:

```yaml
loss_params:
  multiloss:
    coefs:
      epoch_thresholds: [0, 10]
      seg: 10.0
      crossfield_align: [0.0, 1.0]
      crossfield_align90: [0.0, 1.0]
      crossfield_smooth: 0.1
      seg_interior_crossfield: 1.0
      seg_edge_crossfield: 1.0
      seg_edge_interior: 1.0
  seg_loss_params:
    bce_coef: 0.5
    dice_coef: 0.5
    use_dist: false
    use_size: false
    w0: 10.0
    sigma: 5.0
```

The `build_loss_from_config` utility automatically selects between Pattern 2 and Pattern 3 based on which keys are present in `cfg.loss_params`.
