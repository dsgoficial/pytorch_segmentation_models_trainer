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
