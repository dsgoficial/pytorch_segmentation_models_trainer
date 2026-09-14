---
sidebar_position: 28
title: Boundary Label Relaxation
---

# Boundary Label Relaxation

Boundary label relaxation is a noise-robust loss for pixels near class
boundaries, where hard annotations are the least reliable (contours are
rarely pixel-perfect, and a border pixel's receptive field often straddles
two real classes).

This implementation follows Zhu, Y., Sapra, K., Reda, F. A., Shih, K. J.,
Newsam, S., Tao, A., Catanzaro, B., *"Improving Semantic Segmentation via
Video Propagation and Label Relaxation,"* CVPR 2019 — only the
**label-relaxation** component. The paper's other contribution (synthesizing
extra image-label pairs via video-prediction label propagation) does not
apply here: this project's remote-sensing imagery is not video.

---

## How it works

### 1 — Neighbourhood class set

For each pixel *i*, let 𝒩(i) be the set of classes present anywhere in the
3×3 window of its **hard label** centred on *i*. A pixel is a "border"
pixel if `|𝒩(i)| > 1` (at least one neighbour disagrees with its own
class); otherwise it's "interior."

### 2 — Relaxed loss

Instead of maximizing the probability of the single annotated class, the
loss maximizes the probability mass assigned to the union of 𝒩(i):

```
L(i) = -log( Σ_{C ∈ 𝒩(i)} P(C | i) )
```

The model is free to put all its probability mass on *any* class in 𝒩(i)
without penalty — it isn't forced to pick a winner among ambiguous
neighbouring classes.

### 3 — Exact reduction to cross-entropy

When `|𝒩(i)| = 1` (interior pixel), the formula collapses exactly to
standard per-pixel cross-entropy — this is a strict generalization of CE,
not a second loss term blended in. There is **no extra hyperparameter**
beyond the (fixed, 3×3) neighbourhood window itself.

### Implementation note

Neighbourhood membership is computed with a single 3×3 max-pool over the
one-hot label map — `F.max_pool2d(one_hot, kernel_size=3, stride=1,
padding=1)` — rather than a per-pixel loop or `F.unfold`. `ignore_index`
pixels (default `255`) are excluded from the loss mean, and their one-hot
row is zeroed *before* the max-pool so an ignored region cannot leak a
spurious class into a valid neighbour's target set.

---

## Configuration

```yaml title="conf/examples/zhu2019_boundary_relaxation.yaml"
loss:
  _target_: pytorch_segmentation_models_trainer.custom_losses.boundary_relaxation_loss.BoundaryLabelRelaxationLoss
  name: boundary_relaxation
  num_classes: 6
  ignore_index: 255
```

No model or dataset changes are required — this loss operates on the same
hard-label batches (`(B, H, W)` int/long) as standard cross-entropy
training, and plugs into any `Model` via the ordinary `cfg.loss` path (no
`Model` subclass, unlike [Co-Teaching](co-teaching.md), which needs
`CoTeachingModel` for its dual-branch training loop).

---

## API Reference

```python
from pytorch_segmentation_models_trainer.custom_losses.boundary_relaxation_loss import (
    BoundaryLabelRelaxationLoss,
)

loss_fn = BoundaryLabelRelaxationLoss(name="boundary", num_classes=6)

# gt can be a plain hard-label tensor (B, H, W)...
loss = loss_fn.compute(logits, hard_mask)

# ...or a dict with a custom mask_key
loss_fn = BoundaryLabelRelaxationLoss(name="boundary", num_classes=6, mask_key="hard")
loss = loss_fn.compute(logits, {"hard": hard_mask})

# Loss.forward() wrapper (used internally by Model) also works directly:
loss_val, extra_info = loss_fn(logits, hard_mask)
```

:::note ignore_index default
Defaults to `255`, matching `WeightedDiceCrossEntropyLoss` and
`Model._soft_to_hard_masks` elsewhere in this project.
:::

---

## When to use this vs. alternatives

| Loss | Ambiguity source it targets |
|------|------------------------------|
| `BoundaryLabelRelaxationLoss` | Geometric — border-pixel class ambiguity in an otherwise-trusted hard label |
| [`SoftLabelWeightedCELoss`](soft-label-training.md) | Multi-source label disagreement, encoded as a soft distribution + confidence weight |
| [`CoTeachingLoss`](co-teaching.md) | Sample-level label noise, filtered via cross-network agreement |

Boundary label relaxation is the cheapest of the three to add to an existing
run — it's a drop-in loss swap with no extra data, no second model, and no
new hyperparameter to tune.
