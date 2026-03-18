---
sidebar_position: 2
---

# Frame Field Models In Depth

This guide explains what frame fields are, why they improve polygonization, and how the `FrameFieldSegmentationPLModel` and its associated losses work together.

## What Are Frame Fields?

A **frame field** is a per-pixel 2D cross-field: at each pixel location, it encodes 4 unit vectors that are 90 degrees apart from each other (hence "cross-field"). In practice this is represented as two complex numbers `c0` and `c2` (packed as a 4-channel tensor), which together describe the orientation of the cross at that location.

The cross-field captures **boundary direction**: along polygon edges, the field aligns with the edge; at corners, it aligns with both adjacent edges simultaneously. This orientation information is complementary to the binary segmentation mask, which only provides occupancy.

## Why Frame Fields Improve Polygonization

Standard segmentation masks produce soft, rasterized boundaries. Converting them to polygons using contour-tracing or threshold + vectorize methods loses angular sharpness at corners and produces jagged edges.

Frame fields give active contour and active skeleton methods an **orientation cue**:

- Along straight edges, the contour can follow the aligned field direction efficiently
- At corners, the 90-degree symmetry of the cross-field signals a direction change
- Smooth regions away from boundaries have smoothly varying fields, providing stable interpolation for the optimizer

The result is sharper corners, fewer superfluous vertices, and more geometrically accurate polygons, especially for building footprint extraction.

## The Three Frame Field Loss Components

All three losses operate on `pred_batch["crossfield"]`, a `(N, 4, H, W)` tensor containing `c0` (channels 0–1) and `c2` (channels 2–3), and on ground-truth data in `gt_batch`.

### `CrossfieldAlignLoss`

**Purpose:** Aligns the predicted frame field to boundary gradients computed from the ground-truth annotation.

The loss uses `gt_batch["gt_field"]`, a complex-valued tensor encoding the angle of the ground truth boundary tangent (computed by `compute_gt_field` from `gt_batch["gt_crossfield_angle"]`). The error is masked by `gt_polygons_image[:, 1, ...]` (the boundary/edge channel), so it only applies where edges are present.

```
align_loss = framefield_align_error(c0, c2, z)  # z = gt_field
avg_align_loss = mean(align_loss * gt_edges)
```

### `CrossfieldAlign90Loss`

**Purpose:** Enforces the 90-degree rotational symmetry of the cross-field at edge pixels that are not vertices.

It rotates the ground truth field by 90 degrees (`z_90 = [-imag(z), real(z)]`) and computes alignment error at edge pixels, excluding vertices:

```
mask = (gt_edges - gt_vertices).clamp(0, 1)  # edges minus vertices
align90_loss = mean(framefield_align_error(c0, c2, z_90) * mask)
```

This prevents the field from collapsing to a degenerate single direction — at each edge pixel, the cross-field must be equally well aligned to the edge tangent and to the normal at 90 degrees.

### `CrossfieldSmoothLoss`

**Purpose:** Spatial smoothness regularization. Penalizes rapid changes in the cross-field away from edges.

It applies a Laplacian penalty (`LaplacianPenalty`) to the full 4-channel cross-field tensor, weighted by the inverse of the edge mask (so the penalty is only active in smooth regions, not on edges where field changes are expected):

```
gt_edges_inv = 1 - gt_polygons_image[:, 1, ...]
penalty = LaplacianPenalty(c0c2)
avg_penalty = mean(penalty * gt_edges_inv[:, None, ...])
```

## `ComputeSegGrads`: The Pre-processing Step

`ComputeSegGrads` is a pre-processing callable (not a `Loss` subclass) that adds spatial gradient information to `pred_batch` before the coupling losses are computed. It is added to `MultiLoss.pre_processes`.

```python
class ComputeSegGrads:
    def __init__(self, device):
        self.spatial_gradient = SpatialGradient(mode="scharr", coord="ij", normalized=True, device=device)

    def __call__(self, pred_batch, gt_batch):
        pred_batch["seg_grads"] = 2 * self.spatial_gradient(pred_batch["seg"])
        pred_batch["seg_grad_norm"] = pred_batch["seg_grads"].norm(dim=2)          # (N, C, H, W)
        pred_batch["seg_grads_normed"] = pred_batch["seg_grads"] / (
            pred_batch["seg_grad_norm"][:, :, None, ...] + 1e-6
        )                                                                           # (N, C, 2, H, W)
        return pred_batch, gt_batch
```

`ComputeSegGrads` is required when the coupling losses `SegCrossfieldLoss` or `SegEdgeInteriorLoss` are active.

## `FrameFieldComputeWeightNormLossesCallback`

This callback normalizes the magnitudes of individual loss components **before training begins**. Without normalization, losses with large natural magnitudes dominate the weighted sum regardless of the specified coefficients.

The callback:
1. Runs at `on_fit_start` (after DDP worker processes are spawned, avoiding CUDA init errors)
2. Iterates over a configurable number of training batches (`min_samples` and `max_samples` from `loss_params.multiloss.normalization_params`)
3. Calls `multi_loss.reset_norm()`, then `multi_loss.update_norm(pred, batch, batch_size)` for each batch
4. Calls `torch.distributed.barrier()` to synchronize across ranks in DDP setups

For the general (non-frame-field) case, use `ComputeWeightNormLossesCallback` instead.

## `FrameFieldSegmentationPLModel` Output Format

The frame field model produces a dict output with two keys:

| Key | Shape | Description |
|-----|-------|-------------|
| `"seg"` | `(N, C_seg, H, W)` | Segmentation probabilities. `C_seg` depends on `seg_params` (interior, edge, vertex channels) |
| `"crossfield"` | `(N, 4, H, W)` | Cross-field angles. Channels 0–1 are `c0` (primary direction), channels 2–3 are `c2` (secondary direction) |

This dict is passed directly as `pred_batch` to all loss functions in `MultiLoss`.

## Full Training Configuration Example

```yaml
backbone:
  _target_: some_segmentation_backbone.FrameFieldNet

compute_seg: true
compute_crossfield: true

seg_params:
  compute_interior: true
  compute_edge: true
  compute_vertex: true

loss_params:
  multiloss:
    coefs:
      epoch_thresholds: [0, 5, 20, 200]
      seg: 1.0
      crossfield_align:  [0.0, 0.0, 1.0, 1.0]
      crossfield_align90: [0.0, 0.0, 1.0, 1.0]
      crossfield_smooth:  [0.0, 0.0, 0.1, 0.1]
      seg_interior_crossfield: [0.0, 0.0, 0.2, 0.2]
      seg_edge_crossfield:     [0.0, 0.0, 0.2, 0.2]
      seg_edge_interior:       [0.0, 0.0, 0.2, 0.2]
    normalization_params:
      min_samples: 4000
      max_samples: 8000
  seg_loss_params:
    bce_coef: 0.5
    dice_coef: 0.5
    use_dist: true
    use_size: true
    w0: 10
    sigma: 5

callbacks:
  - _target_: pytorch_segmentation_models_trainer.custom_callbacks.training_callbacks.FrameFieldComputeWeightNormLossesCallback
```
