---
sidebar_position: 7
title: Custom Optimizers
---

# Custom Optimizers

The project provides drop-in replacements for the standard PyTorch optimizers that add optional **Gradient Centralization (GC)** support. They are located in:

```
pytorch_segmentation_models_trainer.optimizers.gradient_centralization
```

All classes are subclasses of `torch.optim.Optimizer` and accept the same hyperparameters as their PyTorch equivalents, with three extra keyword arguments that control GC behaviour.

---

## About Gradient Centralization

Gradient Centralization (GC) is a technique introduced in [Yong et al. (2020)](https://arxiv.org/abs/2004.01461) that centres the gradient of each weight tensor around zero before the parameter update step. For a weight tensor with shape `[out, in, ...]`, the mean across all dimensions except the first is subtracted from every gradient element.

**Benefits**

- Smooths the loss landscape, often leading to faster convergence and better generalisation.
- Acts as a form of implicit regularisation with negligible computational overhead.
- Particularly effective for convolutional layers with large spatial kernels.

**When to use it**

- GC is most beneficial when training from scratch or fine-tuning on a new domain.
- For very small models or fully connected layers only, the gains are marginal. Use `gc_conv_only=True` to restrict GC to convolutional layers (`len(shape) > 3`).
- The `gc_loc` flag controls *where* GC is applied in the update: when `True`, GC is applied to the raw gradient before the moment updates; when `False`, GC is applied to the final parameter update direction (after moment correction). The default differs per optimizer — check the table below.

### GC Parameters (common to all optimizers)

| Parameter | Type | Default | Description |
|---|---|---|---|
| `use_gc` | `bool` | `False` | Enable gradient centralization. |
| `gc_conv_only` | `bool` | `False` | Restrict GC to parameters with more than 3 dimensions (convolutional weights). |
| `gc_loc` | `bool` | varies | Apply GC before moment accumulation (`True`) or after (`False`). |

---

## `Adam`

**Import path**

```python
from pytorch_segmentation_models_trainer.optimizers.gradient_centralization import Adam
```

Adam with optional gradient centralization. Matches `torch.optim.Adam` behaviour when `use_gc=False`.

### Constructor Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `params` | iterable | required | Model parameters or parameter groups. |
| `lr` | `float` | `1e-3` | Learning rate. |
| `betas` | `tuple[float, float]` | `(0.9, 0.999)` | Coefficients for running averages of gradient and squared gradient. |
| `eps` | `float` | `1e-8` | Numerical stability term. |
| `weight_decay` | `float` | `0` | L2 penalty. |
| `amsgrad` | `bool` | `False` | Use the AMSGrad variant. |
| `use_gc` | `bool` | `False` | Enable gradient centralization. |
| `gc_conv_only` | `bool` | `False` | Restrict GC to convolutional weights. |
| `gc_loc` | `bool` | `False` | Apply GC before (`True`) or after (`False`) moment accumulation. |

### Example Config

```yaml
optimizer:
  _target_: pytorch_segmentation_models_trainer.optimizers.gradient_centralization.Adam
  lr: 1.0e-4
  betas: [0.9, 0.999]
  weight_decay: 1.0e-5
  use_gc: true
  gc_conv_only: false
  gc_loc: false
```

---

## `AdamW`

**Import path**

```python
from pytorch_segmentation_models_trainer.optimizers.gradient_centralization import AdamW
```

AdamW (decoupled weight decay) with optional gradient centralization. Matches `torch.optim.AdamW` behaviour when `use_gc=False`. Note that `gc_loc` defaults to `True` here (GC applied to the raw gradient), which is the recommended placement for AdamW.

### Constructor Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `params` | iterable | required | Model parameters or parameter groups. |
| `lr` | `float` | `1e-3` | Learning rate. |
| `betas` | `tuple[float, float]` | `(0.9, 0.999)` | Moment coefficients. |
| `eps` | `float` | `1e-8` | Numerical stability term. |
| `weight_decay` | `float` | `1e-2` | Decoupled weight decay coefficient. |
| `amsgrad` | `bool` | `False` | Use the AMSGrad variant. |
| `use_gc` | `bool` | `False` | Enable gradient centralization. |
| `gc_conv_only` | `bool` | `False` | Restrict GC to convolutional weights. |
| `gc_loc` | `bool` | `True` | Apply GC before moment accumulation (recommended for AdamW). |

### Example Config

```yaml
optimizer:
  _target_: pytorch_segmentation_models_trainer.optimizers.gradient_centralization.AdamW
  lr: 3.0e-4
  weight_decay: 1.0e-2
  use_gc: true
  gc_loc: true
```

---

## `RAdam`

**Import path**

```python
from pytorch_segmentation_models_trainer.optimizers.gradient_centralization import RAdam
```

Rectified Adam with optional gradient centralization. RAdam uses a variance-rectified adaptive learning rate that avoids the unstable early training dynamics of plain Adam by buffering step-size corrections and only activating the adaptive term once the variance estimate is reliable. When the variance is not yet reliable, the optimizer degenerates to SGD (controlled by `degenerated_to_sgd`).

### Constructor Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `params` | iterable | required | Model parameters or parameter groups. |
| `lr` | `float` | `1e-3` | Learning rate. |
| `betas` | `tuple[float, float]` | `(0.9, 0.999)` | Moment coefficients. |
| `eps` | `float` | `1e-8` | Numerical stability term. |
| `weight_decay` | `float` | `0` | L2 penalty. |
| `degenerated_to_sgd` | `bool` | `True` | Fall back to SGD when variance estimate is unreliable. |
| `use_gc` | `bool` | `False` | Enable gradient centralization. |
| `gc_conv_only` | `bool` | `False` | Restrict GC to convolutional weights. |
| `gc_loc` | `bool` | `False` | Apply GC before (`True`) or after (`False`) moment accumulation. |

### Example Config

```yaml
optimizer:
  _target_: pytorch_segmentation_models_trainer.optimizers.gradient_centralization.RAdam
  lr: 1.0e-3
  weight_decay: 1.0e-4
  degenerated_to_sgd: true
  use_gc: true
  gc_conv_only: false
```

---

## `PlainRAdam`

**Import path**

```python
from pytorch_segmentation_models_trainer.optimizers.gradient_centralization import PlainRAdam
```

A simplified version of RAdam without the internal step buffer. Uses the same variance-rectification formula but re-computes the step size on every iteration rather than caching it. Slightly higher computational cost but lower memory usage.

### Constructor Parameters

Same as `RAdam` (without the internal buffer — the `degenerated_to_sgd` parameter is still present).

### Example Config

```yaml
optimizer:
  _target_: pytorch_segmentation_models_trainer.optimizers.gradient_centralization.PlainRAdam
  lr: 1.0e-3
  weight_decay: 1.0e-4
  use_gc: true
```

---

## `SGD`

**Import path**

```python
from pytorch_segmentation_models_trainer.optimizers.gradient_centralization import SGD
```

SGD (with optional momentum and Nesterov) with optional gradient centralization. GC is always applied to the raw gradient (before momentum accumulation), regardless of `gc_loc` (which is not a parameter for SGD). This is the optimizer used internally by `TensorPolyOptimizer` during the active polygon refinement stage.

### Constructor Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `params` | iterable | required | Model parameters or parameter groups. |
| `lr` | `float` | required | Learning rate. |
| `momentum` | `float` | `0` | Momentum factor. |
| `dampening` | `float` | `0` | Dampening for momentum. |
| `weight_decay` | `float` | `0` | L2 penalty. |
| `nesterov` | `bool` | `False` | Enable Nesterov momentum. Requires `momentum > 0` and `dampening == 0`. |
| `use_gc` | `bool` | `False` | Enable gradient centralization. |
| `gc_conv_only` | `bool` | `False` | Restrict GC to convolutional weights. |

### Example Config

```yaml
optimizer:
  _target_: pytorch_segmentation_models_trainer.optimizers.gradient_centralization.SGD
  lr: 1.0e-2
  momentum: 0.9
  weight_decay: 1.0e-4
  nesterov: true
  use_gc: true
  gc_conv_only: false
```

---

## Internal Usage: Polygon Optimizers

The `pytorch_segmentation_models_trainer.optimizers.poly_optimizers` module uses the GC-enabled optimizers internally for the differentiable polygon and skeleton refinement passes:

- `TensorPolyOptimizer` uses `gradient_centralization.SGD` with `use_gc=True` and `momentum=0.9` to refine polygon vertex positions against the frame-field alignment loss.
- `TensorSkeletonOptimizer` uses `gradient_centralization.Adam` with `use_gc=True` to refine skeleton node positions.

These classes are not intended to be configured directly via YAML; they are instantiated programmatically during the polygonisation pipeline.
