---
sidebar_position: 3
title: Test Time Augmentation (TTA)
---

# Test Time Augmentation (TTA)

Test Time Augmentation improves prediction quality without retraining: the model runs multiple times on geometrically transformed versions of the same input, each prediction is reversed back to the original orientation, and the results are averaged.

---

## How it works

For each configured augmentation the pipeline executes:

```
input → augmentation → model → inverse augmentation → de-augmented prediction
                                                               ↓
                                                  average all predictions
                                                               ↓
                                                     final consolidated output
```

In **sliding-window inference** TTA is applied per tile — the averaged, de-augmented prediction is handed to the `TileMerger`, which then performs the normal spatial blending. The sliding-window code itself is unchanged.

In the **model test step** (`trainer.test()`), TTA is applied to each full image batch.

---

## Available augmentations

The eight supported augmentations form the **D4 dihedral group** — all symmetries of a square.  Each has an exact inverse, so de-augmentation introduces no spatial artifacts.

| Name in config   | Transformation                                       | Inverse              |
|------------------|------------------------------------------------------|----------------------|
| `rot0`           | Identity (original image, no transformation)         | `rot0`               |
| `rot90`          | 90° counter-clockwise rotation                       | `rot270`             |
| `rot180`         | 180° rotation                                        | `rot180`             |
| `rot270`         | 270° counter-clockwise rotation                      | `rot90`              |
| `flip_h`         | Horizontal flip (left-right mirror)                  | `flip_h`             |
| `flip_v`         | Vertical flip (up-down mirror)                       | `flip_v`             |
| `flip_h_rot90`   | Horizontal flip + 90° counter-clockwise rotation     | `rot270` + `flip_h`  |
| `flip_v_rot90`   | Vertical flip + 90° counter-clockwise rotation       | `rot270` + `flip_v`  |

:::tip Recommended presets
- **4 rotations (default):** `rot0`, `rot90`, `rot180`, `rot270` — 4× forward passes, excellent cost-benefit ratio.
- **Full D4 group:** all 8 augmentations — 8× forward passes, maximum symmetry coverage.
- **Minimum:** `rot0`, `rot180` — 2× forward passes, useful for overhead imagery with no dominant orientation.
:::

---

## TTA in inference (`predict`)

Add `use_tta: true` and `tta_augmentations` to the `inference_processor` block of your predict YAML:

```yaml title="configs/predict_with_tta.yaml"
checkpoint_path: /checkpoints/unet_best.ckpt
device: cuda:0

pl_model:
  _target_: pytorch_segmentation_models_trainer.model_loader.model.Model

hyperparameters:
  batch_size: 8

inference_image_reader:
  _target_: pytorch_segmentation_models_trainer.tools.data_handlers.raster_reader.FolderImageReaderProcessor
  folder_name: /data/test_images/
  image_extension: tif

inference_processor:
  _target_: pytorch_segmentation_models_trainer.tools.inference.inference_processors.SingleImageInfereceProcessor
  model_input_shape: [448, 448]
  step_shape: [224, 224]
  # ── TTA ──────────────────────────────────────────────────────────────────
  use_tta: true
  tta_augmentations:
    - rot0          # original image (identity)
    - rot90         # 90° counter-clockwise rotation
    - rot180        # 180° rotation
    - rot270        # 270° counter-clockwise rotation
    - flip_h        # horizontal flip (left-right mirror)
    - flip_v        # vertical flip (up-down mirror)
    - flip_h_rot90  # horizontal flip + 90° counter-clockwise rotation
    - flip_v_rot90  # vertical flip + 90° counter-clockwise rotation
  # ─────────────────────────────────────────────────────────────────────────

export_strategy:
  _target_: pytorch_segmentation_models_trainer.tools.inference.export_inference.RasterExportInferenceStrategy
  output_file_path: /output/prediction_tta.tif

inference_threshold: 0.5
```

The same parameters work for `MultiClassInferenceProcessor`:

```yaml
inference_processor:
  _target_: pytorch_segmentation_models_trainer.tools.inference.inference_processors.MultiClassInferenceProcessor
  model_input_shape: [512, 512]
  step_shape: [256, 256]
  num_classes: 5
  use_tta: true
  tta_augmentations:
    - rot0     # original image (identity)
    - rot90    # 90° counter-clockwise rotation
    - rot180   # 180° rotation
    - rot270   # 270° counter-clockwise rotation
```

---

## TTA in the test step (`trainer.test()`)

To enable TTA during model evaluation, add `use_tta` directly to the training / evaluation config (at the same level as the model):

```yaml title="configs/train_with_tta_eval.yaml"
# ... other training settings ...

# ── TTA for test_step ────────────────────────────────────────────────────────
use_tta: true
tta_augmentations:
  - rot0     # original image (identity)
  - rot90    # 90° counter-clockwise rotation
  - rot180   # 180° rotation
  - rot270   # 270° counter-clockwise rotation
```

When `use_tta: true`, `test_step` automatically applies the augmentations to each batch, averages the de-augmented predictions, and uses the result for loss and metric computation.

:::note
TTA in the test step **does not affect** training (`training_step`) or validation (`validation_step`) — it is activated exclusively during `trainer.test()`.
:::

---

## TTA with frame-field models (`SingleImageFromFrameFieldProcessor`)

Frame-field processors produce two output tensors: `seg` (segmentation mask) and `crossfield` (tangent-field tensor).

The `crossfield` encodes tangent angles. Correctly de-augmenting it requires transforming the angle values themselves — a non-trivial operation that is out of scope for the current implementation.

**Behaviour with TTA enabled:**

| Output       | TTA treatment                                                                |
|--------------|------------------------------------------------------------------------------|
| `seg`        | De-augmented and averaged across all augmentations                           |
| `crossfield` | Taken from the identity (`rot0`) pass; if `rot0` is absent, from the first pass |

```yaml
inference_processor:
  _target_: pytorch_segmentation_models_trainer.tools.inference.inference_processors.SingleImageFromFrameFieldProcessor
  model_input_shape: [448, 448]
  step_shape: [224, 224]
  mask_bands: 1
  use_tta: true
  tta_augmentations:
    - rot0     # required — crossfield is taken from this pass
    - rot90    # 90° counter-clockwise rotation
    - rot180   # 180° rotation
    - rot270   # 270° counter-clockwise rotation
```

:::caution
Always include `rot0` in `tta_augmentations` when using `SingleImageFromFrameFieldProcessor` with TTA, so that `crossfield` is taken from the unmodified pass.
:::

---

## Computational cost

Each augmentation in `tta_augmentations` adds one full model forward pass per tile. With equivalent `batch_size` and `step_shape`:

| Augmentations         | Forward passes | Relative cost |
|-----------------------|----------------|---------------|
| None (TTA disabled)   | 1×             | 1×            |
| 4 rotations (default) | 4×             | ~4×           |
| Full D4 group (8 aug) | 8×             | ~8×           |

Memory cost per inference does not increase significantly — augmentations are applied and accumulated sequentially per batch.

---

## Python API

```python
from pytorch_segmentation_models_trainer.tools.tta.tta import (
    apply_tta,
    ROTATION_AUGMENTATIONS,   # ["rot0", "rot90", "rot180", "rot270"]
    D4_AUGMENTATIONS,         # all 8 symmetries
    ROT0, ROT90, ROT180, ROT270,
    FLIP_H, FLIP_V, FLIP_H_ROT90, FLIP_V_ROT90,
)

# Direct usage with any callable
pred = apply_tta(
    model_fn=model,
    batch=tiles_batch,            # torch.Tensor [B, C, H, W]
    augmentations=["rot0", "rot90", "rot180", "rot270"],
)

# With skip_keys — for dict-output models where certain tensors
# should not be spatially de-augmented (e.g. crossfield in frame-field models):
pred = apply_tta(
    model_fn=model,
    batch=tiles_batch,
    augmentations=ROTATION_AUGMENTATIONS,
    skip_keys=frozenset({"crossfield"}),
)
```
