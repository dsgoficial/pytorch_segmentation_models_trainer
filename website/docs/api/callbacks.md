---
sidebar_position: 6
title: Custom Callbacks
---

# Custom Callbacks

Custom callbacks extend PyTorch Lightning's `Callback` and `BasePredictionWriter` interfaces to add visualisation, metrics reporting, loss normalisation, and polygonisation steps at specific points in the training or prediction loop.

Callbacks are split across three modules:

| Module | Path |
|---|---|
| Image visualisation | `pytorch_segmentation_models_trainer.custom_callbacks.image_callbacks` |
| Metrics | `pytorch_segmentation_models_trainer.custom_callbacks.metrics_callbacks` |
| Training utilities | `pytorch_segmentation_models_trainer.custom_callbacks.training_callbacks` |

---

## Adding Callbacks to Your Config

Callbacks are listed under the `callbacks:` key in your training YAML. Hydra instantiates each entry using its `_target_` field.

```yaml
trainer:
  _target_: pytorch_lightning.Trainer
  max_epochs: 100
  callbacks:
    - _target_: pytorch_segmentation_models_trainer.custom_callbacks.training_callbacks.WarmupCallback
      warmup_epochs: 5
    - _target_: pytorch_segmentation_models_trainer.custom_callbacks.image_callbacks.ImageSegmentationResultCallback
      n_samples: 4
      log_every_k_epochs: 5
    - _target_: pytorch_segmentation_models_trainer.custom_callbacks.metrics_callbacks.ConfusionMatrixCallback
      num_classes: 6
      class_names: [background, building, road, water, vegetation, bare_soil]
      log_every_n_epochs: 10
```

---

## Image Visualisation Callbacks

### `ImageSegmentationResultCallback`

**Import path**

```python
from pytorch_segmentation_models_trainer.custom_callbacks.image_callbacks import ImageSegmentationResultCallback
```

After each validation epoch, runs the model on a small sample of validation images, generates side-by-side ground-truth / prediction plots, saves them to disk, and logs them to TensorBoard.

#### Fires on

- `on_sanity_check_end` — resolves and creates the output directory.
- `on_validation_epoch_end` — generates and logs visualisations (rank 0 only).

#### Constructor Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `n_samples` | `int` | `None` | Number of images to visualise. Defaults to the validation batch size when `None`. |
| `output_path` | `str` | `None` | Directory to write PNG files to. Defaults to `<log_dir>/image_logs`. |
| `normalized_input` | `bool` | `True` | Whether the input images are normalised; controls denormalisation before plotting. |
| `norm_params` | `dict` | `{}` | kwargs passed to `denormalize_np_array()`. |
| `log_every_k_epochs` | `int` | `1` | Only generate visualisations every k epochs. |

#### Example Config

```yaml
- _target_: pytorch_segmentation_models_trainer.custom_callbacks.image_callbacks.ImageSegmentationResultCallback
  n_samples: 8
  log_every_k_epochs: 5
  normalized_input: true
```

---

### `FrameFieldResultCallback`

**Import path**

```python
from pytorch_segmentation_models_trainer.custom_callbacks.image_callbacks import FrameFieldResultCallback
```

Subclass of `ImageSegmentationResultCallback` tailored for frame-field models. Reads `"gt_polygons_image"` from batches (a 2-channel ground-truth with polygon and boundary channels) and plots both channels alongside their predictions.

#### Fires on

Same as `ImageSegmentationResultCallback`.

#### Example Config

```yaml
- _target_: pytorch_segmentation_models_trainer.custom_callbacks.image_callbacks.FrameFieldResultCallback
  n_samples: 4
  log_every_k_epochs: 10
```

---

### `FrameFieldOverlayedResultCallback`

**Import path**

```python
from pytorch_segmentation_models_trainer.custom_callbacks.image_callbacks import FrameFieldOverlayedResultCallback
```

Subclass of `ImageSegmentationResultCallback` that generates an overlayed visualisation combining the segmentation mask and the cross-field orientation on top of the original image. Logs directly to TensorBoard without saving files to disk.

#### Fires on

- `on_validation_epoch_end`

#### Example Config

```yaml
- _target_: pytorch_segmentation_models_trainer.custom_callbacks.image_callbacks.FrameFieldOverlayedResultCallback
  n_samples: 4
```

---

### `ObjectDetectionResultCallback`

**Import path**

```python
from pytorch_segmentation_models_trainer.custom_callbacks.image_callbacks import ObjectDetectionResultCallback
```

Subclass of `ImageSegmentationResultCallback` for object detection models. Draws predicted bounding boxes (filtered by a score threshold) on the input image and logs the result to TensorBoard.

#### Additional Constructor Parameter

| Parameter | Type | Default | Description |
|---|---|---|---|
| `threshold` | `float` | `0.5` | Minimum confidence score for a box to be drawn. |

#### Fires on

- `on_validation_epoch_end`

#### Example Config

```yaml
- _target_: pytorch_segmentation_models_trainer.custom_callbacks.image_callbacks.ObjectDetectionResultCallback
  n_samples: 8
  threshold: 0.4
```

---

### `PolygonRNNResultCallback`

**Import path**

```python
from pytorch_segmentation_models_trainer.custom_callbacks.image_callbacks import PolygonRNNResultCallback
```

Subclass of `ImageSegmentationResultCallback` for PolygonRNN models. Visualises ground-truth polygon vertices alongside predicted vertices overlaid on the source image.

#### Fires on

- `on_validation_epoch_end`

#### Example Config

```yaml
- _target_: pytorch_segmentation_models_trainer.custom_callbacks.image_callbacks.PolygonRNNResultCallback
  n_samples: 8
```

---

### `ModPolyMapperResultCallback`

**Import path**

```python
from pytorch_segmentation_models_trainer.custom_callbacks.image_callbacks import ModPolyMapperResultCallback
```

Subclass of `PolygonRNNResultCallback` for the ModPolyMapper architecture (detection + PolygonRNN). Handles the combined detection + polygon output format.

#### Additional Constructor Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `threshold` | `float` | `0.5` | Detection score threshold. |
| `show_label_scores` | `bool` | `False` | Whether to overlay detection scores on the visualisation. |
| `n_samples` | `int` | `16` | Defaults to 16 (overrides parent default). |

#### Fires on

- `on_validation_epoch_end`

#### Example Config

```yaml
- _target_: pytorch_segmentation_models_trainer.custom_callbacks.image_callbacks.ModPolyMapperResultCallback
  n_samples: 16
  threshold: 0.5
  show_label_scores: true
```

---

## Metrics Callbacks

### `ConfusionMatrixCallback`

**Import path**

```python
from pytorch_segmentation_models_trainer.custom_callbacks.metrics_callbacks import ConfusionMatrixCallback
```

Builds a multiclass confusion matrix across the full validation set using `torchmetrics.ConfusionMatrix` (GPU-efficient). At the end of every `log_every_n_epochs` epochs, the matrix is plotted as a seaborn heatmap, saved to disk, and logged to TensorBoard. Per-class accuracy and precision scalars are also logged.

#### Fires on

- `on_sanity_check_end` — resolves the output directory.
- `on_validation_epoch_start` — resets the confusion matrix accumulator.
- `on_validation_batch_end` — updates the accumulator with batch predictions.
- `on_validation_epoch_end` — computes, plots, and logs the matrix (rank 0 only).

#### Constructor Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `num_classes` | `int` | required | Total number of classes. |
| `class_names` | `list[str]` | `None` | Human-readable class labels. Defaults to `["Class 0", "Class 1", ...]`. |
| `normalize` | `str` | `"true"` | Normalisation mode: `"true"` (row-normalised), `"pred"` (col-normalised), `"all"` (global), or `None` (raw counts). |
| `log_every_n_epochs` | `int` | `10` | Plot frequency. |
| `figsize` | `tuple` | `(12, 10)` | Matplotlib figure size. |
| `output_path` | `str` | `None` | Save directory. Defaults to `<log_dir>/confusion_matrices`. |

#### Example Config

```yaml
- _target_: pytorch_segmentation_models_trainer.custom_callbacks.metrics_callbacks.ConfusionMatrixCallback
  num_classes: 6
  class_names: [background, building, road, water, vegetation, bare_soil]
  normalize: true
  log_every_n_epochs: 10
  figsize: [14, 12]
```

---

### `ClassificationReportCallback`

**Import path**

```python
from pytorch_segmentation_models_trainer.custom_callbacks.metrics_callbacks import ClassificationReportCallback
```

Generates a `sklearn.metrics.classification_report` (precision, recall, F1, support per class) at the end of every `log_every_n_epochs` epochs and writes it to a timestamped `.txt` file.

#### Fires on

- `on_sanity_check_end` — resolves the output directory.
- `on_validation_batch_end` — accumulates flattened predictions and targets.
- `on_validation_epoch_end` — computes and saves the report (rank 0 only).

#### Constructor Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `num_classes` | `int` | required | Number of classes. |
| `class_names` | `list[str]` | `None` | Class labels. Defaults to index strings. |
| `log_every_n_epochs` | `int` | `10` | Report frequency. |
| `output_path` | `str` | `None` | Save directory. Defaults to `<log_dir>/classification_reports`. |

#### Example Config

```yaml
- _target_: pytorch_segmentation_models_trainer.custom_callbacks.metrics_callbacks.ClassificationReportCallback
  num_classes: 6
  class_names: [background, building, road, water, vegetation, bare_soil]
  log_every_n_epochs: 10
```

---

## Training Utility Callbacks

### `EMACallback`

**Import path**

```python
from pytorch_segmentation_models_trainer.custom_callbacks.training_callbacks import EMACallback
```

Maintains an Exponential Moving Average (EMA) of model weights alongside the online (SGD-updated) weights. The EMA model tends to be smoother and generalises better than the checkpoint from any single epoch.

**How it works:**

At each optimizer step the shadow weights are updated as:

```text
shadow = decay_eff × shadow + (1 − decay_eff) × param
```

where `decay_eff = min(decay, (step+1) / (step+10))` applies a warmup that prevents early random-initialisation weights from contaminating the EMA. During validation the shadow weights are swapped in so that metrics reflect the EMA model; after validation the online weights are restored so training continues normally. The EMA state is saved inside every checkpoint automatically.

:::tip Loading EMA checkpoints
Because `EMACallback` injects the EMA weights into `state_dict` at save time, loading a checkpoint produces the EMA model automatically. `load_from_checkpoint` must be called with `strict=False` (the framework already does this in `predict.py`).
:::

#### Fires on

- `on_fit_start` — initialises shadow weights from the current model parameters.
- `on_train_batch_end` — updates shadow weights (only when the optimizer has actually stepped).
- `on_validation_epoch_start` — swaps in shadow weights.
- `on_validation_epoch_end` — restores online weights.
- `on_save_checkpoint` — injects EMA weights into the checkpoint state dict.

#### Constructor Parameters

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `decay` | `float` | `0.999` | Target EMA decay. Higher values → slower-moving average. Typical range: `0.99`–`0.9999`. |

#### Example Config

```yaml
callbacks:
  - _target_: pytorch_segmentation_models_trainer.custom_callbacks.training_callbacks.EMACallback
    decay: 0.999

  - _target_: pytorch_lightning.callbacks.ModelCheckpoint
    monitor: loss/val
    mode: min
    save_top_k: 3
```

---

### `MixStyleCallback`

**Import path**

```python
from pytorch_segmentation_models_trainer.custom_callbacks.training_callbacks import MixStyleCallback
```

Applies MixStyle (Zhou et al., ICLR 2021) as a forward hook on the encoder. MixStyle mixes feature-level statistics (mean and standard deviation) between random pairs of instances in a batch, creating implicit domain augmentation at zero parameter cost. This improves generalisation to unseen acquisition conditions, sensors, or temporal shifts.

**How it works:**

At each forward pass during training, with probability `p`, the feature map output of each hooked encoder stage is normalised to zero mean / unit variance and then re-scaled with a Beta-distributed convex combination of the original instance's and a random other instance's statistics:

```text
mu_mix  = λ × mu_i  + (1−λ) × mu_j
sig_mix = λ × sig_i + (1−λ) × sig_j
```

The hook is only active during training (`on_train_batch_start` / `on_train_batch_end` guard). Hooks are registered on `encoder.model.stages` (SMP TimmUniversalEncoder) or `encoder.stages`; a warning is emitted if neither attribute is found and MixStyle is silently disabled.

#### Fires on

- `on_fit_start` — registers forward hooks on the configured encoder stages.
- `on_train_batch_start` — enables the hook.
- `on_train_batch_end` — disables the hook.
- `on_validation_epoch_start` — ensures the hook is disabled during validation.
- `on_fit_end` — removes all hooks.

#### Constructor Parameters

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `p` | `float` | `0.5` | Probability of applying MixStyle on a given forward pass. |
| `alpha` | `float` | `0.1` | Beta distribution parameter. Smaller values produce mixing closer to the original style. |
| `stages` | `list[int]` | `[0, 1]` | Encoder stage indices to hook. Early stages carry more domain/style information. |

:::tip Choosing stages
Hook only the earliest stages (`[0, 1]` or just `[0]`). Later stages encode semantic content that should not be mixed. For ResNet-style encoders, stages 0 and 1 correspond to the first two residual groups.
:::

#### Example Config

```yaml
callbacks:
  - _target_: pytorch_segmentation_models_trainer.custom_callbacks.training_callbacks.MixStyleCallback
    p: 0.5
    alpha: 0.1
    stages: [0, 1]
```

---

### `WarmupCallback`

**Import path**

```python
from pytorch_segmentation_models_trainer.custom_callbacks.training_callbacks import WarmupCallback
```

Freezes the model encoder for the first `warmup_epochs` training epochs to let the decoder head stabilise before end-to-end fine-tuning begins. Calls `pl_module.set_encoder_trainable(trainable)` on the Lightning module, so the module must implement that method.

#### Fires on

- `on_fit_start` — checks whether warmup has already elapsed (for resumed training).
- `on_train_epoch_start` — freezes encoder weights if still in warmup.
- `on_train_epoch_end` — unfreezes encoder weights once warmup is complete.

#### Constructor Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `warmup_epochs` | `int` | `2` | Number of epochs during which the encoder is frozen. |

#### Example Config

```yaml
- _target_: pytorch_segmentation_models_trainer.custom_callbacks.training_callbacks.WarmupCallback
  warmup_epochs: 5
```

---

### `FrameFieldOnlyCrossfieldWarmupCallback`

**Import path**

```python
from pytorch_segmentation_models_trainer.custom_callbacks.training_callbacks import FrameFieldOnlyCrossfieldWarmupCallback
```

Variant of `WarmupCallback` for frame-field models. During warmup, all weights except the cross-field head are frozen by calling `pl_module.set_all_but_crossfield_trainable(trainable)`. This allows the cross-field head to initialise before full model training.

#### Fires on

Same as `WarmupCallback`.

#### Constructor Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `warmup_epochs` | `int` | `2` | Warmup duration in epochs. |

#### Example Config

```yaml
- _target_: pytorch_segmentation_models_trainer.custom_callbacks.training_callbacks.FrameFieldOnlyCrossfieldWarmupCallback
  warmup_epochs: 3
```

---

### `ComputeWeightNormLossesCallback`

**Import path**

```python
from pytorch_segmentation_models_trainer.custom_callbacks.training_callbacks import ComputeWeightNormLossesCallback
```

General-purpose callback for computing loss normalisation weights before training starts. Designed for models that use a compound (multi-term) loss where each term needs to be normalised to a comparable scale. Runs a partial forward pass over the training set on rank 0, then syncs normalisation values across DDP ranks.

The callback is a no-op if:
- Normalisation has already been computed.
- The model does not have `loss_params` in its config.
- The compound loss does not have `normalization_params`.
- The loss norm has already been updated.

#### Fires on

- `on_train_start` — computes normalisation (rank 0 only).

#### Constructor Parameters

This callback takes no constructor arguments.

#### Example Config

```yaml
- _target_: pytorch_segmentation_models_trainer.custom_callbacks.training_callbacks.ComputeWeightNormLossesCallback
```

---

### `FrameFieldComputeWeightNormLossesCallback`

**Import path**

```python
from pytorch_segmentation_models_trainer.custom_callbacks.training_callbacks import FrameFieldComputeWeightNormLossesCallback
```

Frame-field-specific variant of `ComputeWeightNormLossesCallback`. Uses the `multiloss.normalization_params.min_samples` and `max_samples` config fields to determine the number of batches, then calls `pl_module.compute_loss_norms()` directly.

#### Fires on

- `on_fit_start` — computes and syncs loss normalisation.

#### Constructor Parameters

This callback takes no constructor arguments.

#### Example Config

```yaml
- _target_: pytorch_segmentation_models_trainer.custom_callbacks.training_callbacks.FrameFieldComputeWeightNormLossesCallback
```

---

### `FrameFieldPolygonizerCallback`

**Import path**

```python
from pytorch_segmentation_models_trainer.custom_callbacks.training_callbacks import FrameFieldPolygonizerCallback
```

A `BasePredictionWriter` callback that runs polygonisation on each prediction batch during `trainer.predict()`. Instantiates the polygonizer from the model config and processes `(seg, crossfield)` output pairs in a thread pool to avoid blocking the GPU.

#### Fires on

- `on_predict_batch_end`

#### Constructor Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `write_interval` | `str` | `"batch"` | Lightning write interval (`"batch"` or `"epoch"`). |

#### Example Config

```yaml
- _target_: pytorch_segmentation_models_trainer.custom_callbacks.training_callbacks.FrameFieldPolygonizerCallback
  write_interval: batch
```

---

### `ActiveSkeletonsPolygonizerCallback`

**Import path**

```python
from pytorch_segmentation_models_trainer.custom_callbacks.training_callbacks import ActiveSkeletonsPolygonizerCallback
```

`BasePredictionWriter` callback that runs the Active Skeletons polygonisation algorithm on each prediction batch. Falls back to per-image processing when the batch-level call raises an exception, skipping only images that continue to fail.

#### Fires on

- `on_predict_batch_end`

#### Constructor Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `write_interval` | `str` | `"batch"` | Lightning write interval. |

#### Example Config

```yaml
- _target_: pytorch_segmentation_models_trainer.custom_callbacks.training_callbacks.ActiveSkeletonsPolygonizerCallback
  write_interval: batch
```

---

### `ModPolymapperPolygonizerCallback`

**Import path**

```python
from pytorch_segmentation_models_trainer.custom_callbacks.training_callbacks import ModPolymapperPolygonizerCallback
```

`BasePredictionWriter` callback for the ModPolyMapper prediction pipeline. Processes detection output through a PolygonRNN polygonizer in parallel threads. Optionally reprojects output polygon coordinates to world/CRS coordinates using the source raster profile.

#### Fires on

- `on_predict_batch_end`

#### Constructor Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `convert_output_to_world_coords` | `bool` | `True` | When `True`, reads the raster CRS profile and reprojects polygon coordinates. |
| `write_interval` | `str` | `"batch"` | Lightning write interval. |

#### Example Config

```yaml
- _target_: pytorch_segmentation_models_trainer.custom_callbacks.training_callbacks.ModPolymapperPolygonizerCallback
  convert_output_to_world_coords: true
  write_interval: batch
```
