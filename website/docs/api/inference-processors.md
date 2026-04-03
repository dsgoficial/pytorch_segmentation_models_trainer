---
sidebar_position: 5
title: Inference Processors
---

# Inference Processors

Inference processors orchestrate the full prediction pipeline for a trained model: reading a geospatial image, slicing it into tiles, running the model on each tile, merging the results back into a full-resolution output, and saving the inference to disk.

All processors live in:

```
pytorch_segmentation_models_trainer.tools.inference.inference_processors
```

---

## Sliding Window Inference

Large remote-sensing images cannot be processed in a single forward pass. The processors use a sliding-window approach powered by `pytorch_toolbelt.inference.tiles`:

1. The input image is padded to the nearest multiple of `model_input_shape` using reflection padding.
2. An `ImageSlicer` splits the padded image into overlapping tiles of size `model_input_shape` with a step of `step_shape`.
3. Each tile is normalized, converted to a tensor, and batched for GPU inference.
4. A `TileMerger` accumulates tile predictions using **distance-weighted averaging**: pixels near the centre of a tile receive a higher weight than pixels near the edges, smoothing boundary artefacts.
5. The merged prediction is centre-cropped back to the original image dimensions.

**Controlling overlap**

| Parameter | Default | Effect |
|---|---|---|
| `model_input_shape` | `(448, 448)` | Tile size fed to the model |
| `step_shape` | `(224, 224)` | Stride between tiles. Smaller values increase overlap and inference time but reduce edge artefacts. |

Setting `step_shape` equal to `model_input_shape` disables overlap (no blending). A step of half the tile size gives 50% overlap on each axis, which is the recommended default.

---

## `AbstractInferenceProcessor`

**Import path**

```python
from pytorch_segmentation_models_trainer.tools.inference.inference_processors import AbstractInferenceProcessor
```

Abstract base class that defines the shared interface and shared utilities for all inference processors. Cannot be instantiated directly.

### Constructor Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `model` | `torch.nn.Module` | required | Trained PyTorch model, moved to `device` during init. |
| `device` | `str` or `torch.device` | required | Target device (`"cuda"`, `"cpu"`, etc.). |
| `batch_size` | `int` | required | Number of tiles processed per forward pass. |
| `export_strategy` | object | required | Strategy object that serialises inference outputs to disk. Pass `None` to skip saving. |
| `polygonizer` | `TemplatePolygonizerProcessor` | `None` | Optional polygonizer run after inference. |
| `model_input_shape` | `tuple[int, int]` | `(448, 448)` | Tile height and width in pixels. |
| `step_shape` | `tuple[int, int]` | `(224, 224)` | Sliding-window stride in pixels. |
| `mask_bands` | `int` | `1` | Number of output channels (bands) in the prediction mask. |
| `normalize_mean` | `list[float]` | `None` | Per-channel mean for normalisation. Falls back to ImageNet defaults when `None`. |
| `normalize_std` | `list[float]` | `None` | Per-channel standard deviation for normalisation. Falls back to ImageNet defaults when `None`. |
| `normalize_max_value` | `float` | `None` | Maximum pixel value used to scale the image before applying mean/std normalisation (`max_pixel_value` in `A.Normalize`). When `None`, Albumentations uses its default of `255.0`. Set to `65535.0` for 16-bit imagery or `1.0` when the image is already in `[0, 1]`. |
| `config` | any | `None` | Optional Hydra config object passed through for advanced use. |
| `group_output_by_image_basename` | `bool` | `False` | Group polygonizer output under subdirectories named after the input image stem. |

### Key Methods

| Method | Description |
|---|---|
| `process(image_path, threshold, save_inference_output, polygonizer, restore_geo_transform, **kwargs)` | Main entry point. Reads the image, runs `make_inference()`, optionally runs a polygonizer, and calls `save_inference()`. Returns a dict with keys `"inference"` and optionally `"polygons"`. |
| `read_image_and_profile(image_path, restore_geo_transform)` | Opens a raster with rasterio, returns `(image_ndarray, profile_dict)`. The image is returned with its **native dtype** (no cast applied). Set `restore_geo_transform=False` to strip CRS from the profile. |
| `get_normalization_function()` | Returns an Albumentations `Normalize` transform configured from `normalize_mean`, `normalize_std`, and `normalize_max_value`. |
| `save_inference(image_path, threshold, profile, inference, output_dict, apply_threshold)` | Thresholds the `"seg"` band (if `apply_threshold=True`), enriches `profile` with `input_name` and `suffix`, and delegates serialisation to `export_strategy`. |
| `make_inference(image, **kwargs)` | **Abstract.** Must be implemented by subclasses. Receives a raw NumPy image array and returns a dict of output arrays. |

---

## `SingleImageInfereceProcessor`

**Import path**

```python
from pytorch_segmentation_models_trainer.tools.inference.inference_processors import SingleImageInfereceProcessor
```

Concrete processor for standard binary or single-output segmentation models. Inherits all parameters from `AbstractInferenceProcessor` with no additional constructor arguments.

### How It Works

`make_inference()` performs the following steps:

1. Normalises the image with `get_normalization_function()`.
2. Pads the image to a multiple of `model_input_shape` using `cv2.BORDER_REFLECT_101`.
3. Slices the padded image into overlapping tiles using `ImageSlicer`.
4. Runs the model on tile batches (inside `torch.no_grad()` with AMP `autocast`).
5. Accumulates predictions with `TileMerger` (weighted averaging).
6. Crops the merged mask back to the original size.
7. Returns `{"seg": ndarray}`.

### Example YAML Config

8-bit RGB:
```yaml
inference_processor:
  _target_: pytorch_segmentation_models_trainer.tools.inference.inference_processors.SingleImageInfereceProcessor
  model_input_shape: [448, 448]
  step_shape: [224, 224]
  batch_size: 4
  mask_bands: 1
  normalize_mean: [0.485, 0.456, 0.406]
  normalize_std: [0.229, 0.224, 0.225]
  # normalize_max_value omitted → defaults to 255.0 (standard 8-bit)
```

16-bit multispectral (e.g. Sentinel-2):
```yaml
inference_processor:
  _target_: pytorch_segmentation_models_trainer.tools.inference.inference_processors.SingleImageInfereceProcessor
  model_input_shape: [256, 256]
  step_shape: [128, 128]
  batch_size: 4
  mask_bands: 1
  normalize_mean: [0.5, 0.5, 0.5, 0.4]
  normalize_std: [0.2, 0.2, 0.2, 0.15]
  normalize_max_value: 65535.0   # scales uint16 values before mean/std normalisation
```

---

## `MultiClassInferenceProcessor`

**Import path**

```python
from pytorch_segmentation_models_trainer.tools.inference.inference_processors import MultiClassInferenceProcessor
```

Extends `SingleImageInfereceProcessor` for models that output per-class probability maps (logits) with shape `[B, num_classes, H, W]`. After the weighted tile merge, an **argmax** is applied across the class dimension, producing a single-band output where each pixel value is the predicted class index.

### Additional Constructor Parameter

| Parameter | Type | Default | Description |
|---|---|---|---|
| `num_classes` | `int` | `2` | Number of output classes. This value is passed as `mask_bands` to the tile merger so all class channels are merged before argmax. |

Note: `mask_bands` is not exposed directly; set `num_classes` instead.

### Behaviour Differences from `SingleImageInfereceProcessor`

- `merge_masks()` applies `np.argmax(..., axis=0)` after merging, yielding a `[H, W, 1]` array of class indices.
- `save_inference()` forces `profile["count"] = 1` and `profile["dtype"] = "uint8"` before writing; no threshold is applied.

### Example YAML Config

```yaml
inference_processor:
  _target_: pytorch_segmentation_models_trainer.tools.inference.inference_processors.MultiClassInferenceProcessor
  model_input_shape: [512, 512]
  step_shape: [256, 256]
  batch_size: 4
  num_classes: 5
  normalize_mean: [0.485, 0.456, 0.406]
  normalize_std: [0.229, 0.224, 0.225]
```

---

## `SingleImageFromFrameFieldProcessor`

**Import path**

```python
from pytorch_segmentation_models_trainer.tools.inference.inference_processors import SingleImageFromFrameFieldProcessor
```

Extends `SingleImageInfereceProcessor` for frame-field segmentation models that produce two outputs simultaneously: a segmentation mask (`"seg"`) and a cross-field tensor (`"crossfield"`, 4 channels encoding the two principal directions of the frame field). Both outputs are processed with independent `TileMerger` instances so overlap blending is applied to each separately.

### Constructor Parameters

Same as `SingleImageInfereceProcessor`. The cross-field merger is created internally with 4 bands.

### Output

`make_inference()` returns:

```python
{
    "seg": ndarray,        # shape [H, W, mask_bands]
    "crossfield": ndarray  # shape [H, W, 4]
}
```

Both arrays are passed to the polygonizer or export strategy as-is (no threshold in `make_inference()`).

### Example YAML Config

```yaml
inference_processor:
  _target_: pytorch_segmentation_models_trainer.tools.inference.inference_processors.SingleImageFromFrameFieldProcessor
  model_input_shape: [448, 448]
  step_shape: [224, 224]
  batch_size: 4
  mask_bands: 2
  normalize_mean: [0.485, 0.456, 0.406]
  normalize_std: [0.229, 0.224, 0.225]
```

---

## `ObjectDetectionInferenceProcessor`

**Import path**

```python
from pytorch_segmentation_models_trainer.tools.inference.inference_processors import ObjectDetectionInferenceProcessor
```

Processor for object detection models that return bounding boxes rather than dense masks. Uses a `BboxTileMerger` (from `pytorch_segmentation_models_trainer.tools.detection.bbox_handler`) to merge box predictions across overlapping tiles using either union or NMS.

### Additional Constructor Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `post_process_method` | `str` | `"union"` | Method used to merge overlapping detections across tiles. |
| `min_visibility` | `float` | `0.3` | Minimum fraction of a box that must be visible within a tile for it to be retained. |

### Behaviour

- `process()` always sets `output_inferences=True` in kwargs so the raw detection list is returned alongside any saved outputs.
- `make_inference()` returns a list of per-image dicts with keys such as `"boxes"`, `"scores"`, and `"labels"` (standard Torchvision detection format).
- `save_inference()` serialises the detection list as JSON-compatible dicts.

### Example YAML Config

```yaml
inference_processor:
  _target_: pytorch_segmentation_models_trainer.tools.inference.inference_processors.ObjectDetectionInferenceProcessor
  model_input_shape: [640, 640]
  step_shape: [320, 320]
  batch_size: 2
  post_process_method: union
  min_visibility: 0.3
```

---

## `PolygonRNNInferenceProcessor`

**Import path**

```python
from pytorch_segmentation_models_trainer.tools.inference.inference_processors import PolygonRNNInferenceProcessor
```

Specialised processor for PolygonRNN-family models. These models do not produce dense segmentation maps; instead they auto-regressively generate polygon vertex sequences for each object crop. This processor:

- Fixes `model_input_shape` to `(224, 224)` (the PolygonRNN input resolution).
- Always sets `save_inference_output=False` since output is a polygon list, not a raster.
- Crops and resizes each bounding box to 224x224 before feeding the model.

### Additional Constructor Parameter

| Parameter | Type | Default | Description |
|---|---|---|---|
| `sequence_length` | `int` | `60` | Maximum number of vertices to predict per polygon. |

### `make_inference()` Input

Takes an additional `bboxes` argument — a list of `[min_row, min_col, max_row, max_col]` arrays defining objects to polygonise.

### Example YAML Config

```yaml
inference_processor:
  _target_: pytorch_segmentation_models_trainer.tools.inference.inference_processors.PolygonRNNInferenceProcessor
  batch_size: 16
  sequence_length: 60
  normalize_mean: [0.485, 0.456, 0.406]
  normalize_std: [0.229, 0.224, 0.225]
```
