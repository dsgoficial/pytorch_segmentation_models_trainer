---
sidebar_position: 9
title: Running Inference
---

# Running Inference

After training a model, you can run inference on new images using either of two CLI modes: `predict` for single-image sliding-window processing, or `predict-from-batch` for batch processing via PyTorch Lightning's `Trainer.predict`.

---

## Choosing the Right Mode

| Mode | Entry Point | Best For |
|---|---|---|
| `predict` | `predict.py` | Large images, sliding-window tiling, one-by-one processing |
| `predict-from-batch` | `predict_from_batch.py` | Batch inference with PyTorch Lightning, CSV-driven datasets |

### Tutorial notebook

See [Notebook Tutorials](../tutorials/notebooks.md) for a step-by-step inference walkthrough that follows the Potsdam training notebook.

---

## `predict` Mode

The `predict` script reads a list of images from an **image reader**, processes each image through an **inference processor** (with optional sliding-window tiling), applies a threshold, and writes results using an **export strategy**.

### Running the Command

```bash
pytorch-smt --config-dir ./configs --config-name predict_config
```

Override parameters on the command line without editing the YAML:

```bash
pytorch-smt --config-dir ./configs --config-name predict_config \
  checkpoint_path=/new/model.ckpt \
  device=cuda:1 \
  inference_threshold=0.4
```

### Top-Level Config Keys

| Key | Type | Description |
|---|---|---|
| `checkpoint_path` | `str` | Path to the `.ckpt` file saved during training |
| `device` | `str` | Compute device: `cuda:0`, `cuda:1`, `cpu`, etc. |
| `inference_image_reader` | object | Defines which images to process |
| `inference_processor` | object | Defines how each image is processed |
| `export_strategy` | object | Defines where and how output is saved |
| `inference_threshold` | `float` | Binarisation threshold (default `0.5`) |
| `save_inference` | `bool` | Whether to write output files (default `true`) |
| `pl_model` | object | PyTorch Lightning model class (same as training config) |
| `hyperparameters` | object | Must include `batch_size` |
| `seg_params` | object | Optional; controls number of output mask bands |

---

## Image Readers

Image readers are set under `inference_image_reader` and tell the predict script which files to process.

### `SingleImageReaderProcessor`

Processes a single image file.

```yaml
inference_image_reader:
  _target_: pytorch_segmentation_models_trainer.tools.data_handlers.raster_reader.SingleImageReaderProcessor
  file_name: /data/images/tile_001.tif
```

### `FolderImageReaderProcessor`

Scans a folder for all images matching a given extension.

```yaml
inference_image_reader:
  _target_: pytorch_segmentation_models_trainer.tools.data_handlers.raster_reader.FolderImageReaderProcessor
  folder_name: /data/images/
  image_extension: tif     # file extension to match (without dot)
  recursive: true          # search subfolders recursively
```

| Parameter | Default | Description |
|---|---|---|
| `folder_name` | required | Root folder to scan |
| `image_extension` | `tif` | Extension to match (e.g. `tif`, `png`, `jpg`) |
| `recursive` | `true` | Recurse into subdirectories |

### `CSVImageReaderProcessor`

Reads image paths from a CSV file column.

```yaml
inference_image_reader:
  _target_: pytorch_segmentation_models_trainer.tools.data_handlers.raster_reader.CSVImageReaderProcessor
  input_csv_path: /data/test.csv
  key: image           # column name containing image paths
  root_dir: /data      # prepended to relative paths
```

---

## Inference Processors

The inference processor does the actual model forward pass and handles tiling for large images.

### Sliding-Window Tiling

All processors based on `SingleImageInfereceProcessor` use **sliding-window tiling** from `pytorch-toolbelt`:

1. The input image is padded to a multiple of `model_input_shape`.
2. An `ImageSlicer` divides the padded image into overlapping tiles of size `model_input_shape`, with a stride of `step_shape`.
3. Each batch of tiles is passed through the model with `torch.no_grad()` and AMP (`autocast`).
4. A `TileMerger` accumulates predictions using **weighted averaging** in the overlap regions, reducing seam artifacts.
5. The merged mask is cropped back to the original image dimensions.

Overlap is controlled by the gap between `model_input_shape` and `step_shape`. A smaller `step_shape` means more overlap and smoother blending at the cost of more forward passes.

:::tip Choosing step_shape
A common setting is `step_shape = model_input_shape / 2`. For example, with `model_input_shape: [448, 448]`, set `step_shape: [224, 224]` for 50% overlap. Reduce the step further for very high-detail predictions.
:::

### Test Time Augmentation (TTA)

TTA improves prediction quality by running the model on multiple transformed versions of each tile, reversing the transformation on each output, and averaging the results before handing the tile to the `TileMerger`. The sliding-window pipeline itself is not affected — it always receives one already-consolidated prediction per tile.

Add `use_tta: true` and `tta_augmentations` to any `inference_processor` block:

```yaml
inference_processor:
  _target_: pytorch_segmentation_models_trainer.tools.inference.inference_processors.SingleImageInfereceProcessor
  model_input_shape: [448, 448]
  step_shape: [224, 224]
  use_tta: true
  tta_augmentations:
    - rot0          # original image (identity — always include this)
    - rot90         # 90° counter-clockwise rotation
    - rot180        # 180° rotation
    - rot270        # 270° counter-clockwise rotation
    - flip_h        # horizontal flip (left-right mirror)
    - flip_v        # vertical flip (up-down mirror)
    - flip_h_rot90  # horizontal flip + 90° counter-clockwise rotation
    - flip_v_rot90  # vertical flip + 90° counter-clockwise rotation
```

The eight augmentations form the **D4 dihedral group** — all symmetries of a square. You can use any subset; the default when `tta_augmentations` is omitted is the four 90° rotations (`rot0`, `rot90`, `rot180`, `rot270`).

| Augmentation name | Transformation |
|---|---|
| `rot0` | Identity — original image |
| `rot90` | 90° counter-clockwise rotation |
| `rot180` | 180° rotation |
| `rot270` | 270° counter-clockwise rotation |
| `flip_h` | Horizontal flip (left-right mirror) |
| `flip_v` | Vertical flip (up-down mirror) |
| `flip_h_rot90` | Horizontal flip + 90° counter-clockwise rotation |
| `flip_v_rot90` | Vertical flip + 90° counter-clockwise rotation |

Each augmentation has an exact mathematical inverse — predictions are de-augmented before averaging, so no spatial artifact is introduced.

:::caution TTA and computational cost
Each augmentation in `tta_augmentations` adds one full model forward pass per tile. Four rotations = 4× the inference time; the full D4 group = 8×. Balance quality vs. speed accordingly.
:::

:::note Frame-field models (`SingleImageFromFrameFieldProcessor`)
The `crossfield` output encodes tangent angles that require non-trivial transformation under spatial rotations. TTA is applied normally to `seg`; `crossfield` is always taken from the identity (`rot0`) pass. Include `rot0` in `tta_augmentations` when using this processor with TTA enabled.
:::

See the **[TTA advanced guide](../advanced/tta.md)** for full details, API reference, and config examples.

### `SingleImageInfereceProcessor`

General-purpose sliding-window processor for binary or single-output segmentation.

```yaml
inference_processor:
  _target_: pytorch_segmentation_models_trainer.tools.inference.inference_processors.SingleImageInfereceProcessor
  model_input_shape: [448, 448]   # tile size fed to the model
  step_shape: [224, 224]          # sliding step (controls overlap)
```

| Parameter | Default | Description |
|---|---|---|
| `model_input_shape` | `[448, 448]` | Height and width of each tile |
| `step_shape` | `[224, 224]` | Stride of the sliding window |
| `normalize_mean` | `null` | Per-channel mean for normalization (ImageNet default if null) |
| `normalize_std` | `null` | Per-channel std for normalization (ImageNet default if null) |

### `MultiClassInferenceProcessor`

For multi-class segmentation models that return `[B, num_classes, H, W]` logits. After tile merging, `argmax` is applied across the class dimension to produce a single-band class-index map with values `0` to `num_classes - 1`.

```yaml
inference_processor:
  _target_: pytorch_segmentation_models_trainer.tools.inference.inference_processors.MultiClassInferenceProcessor
  model_input_shape: [512, 512]
  step_shape: [256, 256]
  num_classes: 5
```

| Parameter | Default | Description |
|---|---|---|
| `num_classes` | `2` | Number of output classes |
| `model_input_shape` | `[448, 448]` | Tile size |
| `step_shape` | `[224, 224]` | Sliding stride |

The output raster has 1 band, `dtype=uint8`, with pixel values equal to the predicted class index. The threshold parameter is ignored for this processor.

### `SingleImageFromFrameFieldProcessor`

For frame field segmentation models (e.g. `FrameFieldSegmentationPLModel`) that return both a segmentation mask and a cross-field tensor. Both outputs are tiled and merged independently.

```yaml
inference_processor:
  _target_: pytorch_segmentation_models_trainer.tools.inference.inference_processors.SingleImageFromFrameFieldProcessor
  model_input_shape: [448, 448]
  step_shape: [224, 224]
  mask_bands: 1
```

This processor is required when using any frame-field-based polygonizer (e.g. `ASMPolygonizerProcessor`, `ACMPolygonizerProcessor`), since they consume the `crossfield` key from the inference output.

### `BinaryMaskInferenceProcessor` / Base Class Threshold

All segmentation processors apply `threshold` after merging:

```
output_mask = (merged_probability > threshold).astype(uint8)
```

Set the threshold in the top-level config:

```yaml
inference_threshold: 0.5
```

---

## Normalization

By default, all processors normalize images using ImageNet statistics via `albumentations.Normalize()` (mean `[0.485, 0.456, 0.406]`, std `[0.229, 0.224, 0.225]`).

To override normalization, set `normalize_mean` and `normalize_std` in the `inference_processor` block:

```yaml
inference_processor:
  _target_: pytorch_segmentation_models_trainer.tools.inference.inference_processors.SingleImageInfereceProcessor
  model_input_shape: [448, 448]
  step_shape: [224, 224]
  normalize_mean: [0.5, 0.5, 0.5]
  normalize_std: [0.25, 0.25, 0.25]
```

:::note Custom Normalization
The mean and std values must match whatever normalization was applied during training. Mismatched normalization is a common source of poor inference results.
:::

---

## Export Strategies

Export strategies control how inference outputs are written to disk or a database. Set them under the `export_strategy` key.

### `RasterExportInferenceStrategy`

Saves a single output raster (the `seg` band) to a fixed output path.

```yaml
export_strategy:
  _target_: pytorch_segmentation_models_trainer.tools.inference.export_inference.RasterExportInferenceStrategy
  output_file_path: /output/prediction.tif
```

### `MultipleRasterExportInferenceStrategy`

Saves multiple output bands (e.g. `seg` and `crossfield`) as separate GeoTIFF files in a folder, one file per input image.

```yaml
export_strategy:
  _target_: pytorch_segmentation_models_trainer.tools.inference.export_inference.MultipleRasterExportInferenceStrategy
  output_folder: /output/predictions/
  output_basename: pred.tif
```

Output filenames follow the pattern `{band_key}_{input_name}_{output_basename}`.

### `VectorFileExportInferenceStrategy`

Writes inference output as a vector file (GeoJSON or Shapefile). This is typically used together with a polygonizer.

```yaml
export_strategy:
  _target_: pytorch_segmentation_models_trainer.tools.inference.export_inference.VectorFileExportInferenceStrategy
  output_file_path: /output/predictions.geojson
  driver: GeoJSON    # or "ESRI Shapefile"
```

### `VectorDatabaseExportInferenceStrategy`

Writes polygons directly to a PostGIS-enabled PostgreSQL database.

```yaml
export_strategy:
  _target_: pytorch_segmentation_models_trainer.tools.inference.export_inference.VectorDatabaseExportInferenceStrategy
  user: postgres
  password: secret
  database: geodata
  host: localhost
  port: 5432
  sql: "SELECT id, geom FROM buildings"
  table_name: buildings
  geometry_column: geom
```

---

## `predict-from-batch` Mode

The `predict-from-batch` script loads images from a CSV (or builds one from a folder), groups them by spatial dimensions to enable efficient batching, and runs PyTorch Lightning's `Trainer.predict`.

### Running the Command

```bash
pytorch-smt --config-dir ./configs --config-name predict_batch_config +mode=predict-from-batch
```

### Dataset Configuration

Three input modes are supported, tried in this priority order:

**Mode 1 — CSV file:**
```yaml
inference_dataset:
  input_csv_path: /data/test.csv
  root_dir: /data
  data_loader:
    num_workers: 4
    prefetch_factor: 2
```

**Mode 2 — Build CSV from folder:**
```yaml
inference_dataset:
  build_csv_from_folder:
    enabled: true
    images_folder: /data/images/
    root_dir: /data
```

**Mode 3 — Legacy (val_dataset):**
```yaml
val_dataset:
  input_csv_path: /data/val.csv
  root_dir: /data
```

:::note Windowed Inference
Set `use_inference_processor: true` in the batch config to use `TiledInferenceImageDataset`, which applies sliding-window tiling within the dataloader rather than inside the processor. This requires `inference_processor.model_input_shape` and `inference_processor.step_shape` to be defined.
:::

---

## Full Config Examples

### Binary Segmentation (Single Image)

```yaml title="configs/predict_binary.yaml"
# ── Checkpoint ────────────────────────────────────────────────────────────────
checkpoint_path: /checkpoints/unet_best.ckpt
device: cuda:0

# ── Model (must match training config) ───────────────────────────────────────
pl_model:
  _target_: pytorch_segmentation_models_trainer.model_loader.model.Model

hyperparameters:
  batch_size: 8

# ── Image Reader ──────────────────────────────────────────────────────────────
inference_image_reader:
  _target_: pytorch_segmentation_models_trainer.tools.data_handlers.raster_reader.FolderImageReaderProcessor
  folder_name: /data/test_images/
  image_extension: tif
  recursive: true

# ── Inference Processor ───────────────────────────────────────────────────────
inference_processor:
  _target_: pytorch_segmentation_models_trainer.tools.inference.inference_processors.SingleImageInfereceProcessor
  model_input_shape: [448, 448]
  step_shape: [224, 224]

# ── Export Strategy ───────────────────────────────────────────────────────────
export_strategy:
  _target_: pytorch_segmentation_models_trainer.tools.inference.export_inference.MultipleRasterExportInferenceStrategy
  output_folder: /output/predictions/
  output_basename: pred.tif

# ── Threshold ─────────────────────────────────────────────────────────────────
inference_threshold: 0.5
```

### Multi-Class Segmentation

```yaml title="configs/predict_multiclass.yaml"
checkpoint_path: /checkpoints/multiclass_best.ckpt
device: cuda:0

pl_model:
  _target_: pytorch_segmentation_models_trainer.model_loader.model.Model

hyperparameters:
  batch_size: 4

inference_image_reader:
  _target_: pytorch_segmentation_models_trainer.tools.data_handlers.raster_reader.FolderImageReaderProcessor
  folder_name: /data/test_images/
  image_extension: tif
  recursive: false

inference_processor:
  _target_: pytorch_segmentation_models_trainer.tools.inference.inference_processors.MultiClassInferenceProcessor
  model_input_shape: [512, 512]
  step_shape: [256, 256]
  num_classes: 5

export_strategy:
  _target_: pytorch_segmentation_models_trainer.tools.inference.export_inference.RasterExportInferenceStrategy
  output_file_path: /output/multiclass_pred.tif
```

### Sliding-Window on Large Aerial Images

```yaml title="configs/predict_large_image.yaml"
checkpoint_path: /checkpoints/building_detector.ckpt
device: cuda:0

pl_model:
  _target_: pytorch_segmentation_models_trainer.model_loader.model.Model

hyperparameters:
  batch_size: 16   # larger batch for tile processing

inference_image_reader:
  _target_: pytorch_segmentation_models_trainer.tools.data_handlers.raster_reader.SingleImageReaderProcessor
  file_name: /data/large_ortho.tif

inference_processor:
  _target_: pytorch_segmentation_models_trainer.tools.inference.inference_processors.SingleImageInfereceProcessor
  model_input_shape: [448, 448]
  step_shape: [112, 112]   # 75% overlap for high-detail output
  normalize_mean: [0.485, 0.456, 0.406]
  normalize_std: [0.229, 0.224, 0.225]

export_strategy:
  _target_: pytorch_segmentation_models_trainer.tools.inference.export_inference.RasterExportInferenceStrategy
  output_file_path: /output/large_ortho_pred.tif

inference_threshold: 0.45
save_inference: true
```

---

## PyTorch Lightning Trainer Settings for Inference

Both `predict` and `predict-from-batch` use the `pl_trainer` config block. For inference you typically only need accelerator and device settings:

```yaml
pl_trainer:
  accelerator: gpu
  devices: [0]       # specific GPU index; use -1 for all GPUs
```
