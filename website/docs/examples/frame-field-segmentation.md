---
sidebar_position: 4
title: Building Extraction with Frame Field Models
---

# Building Extraction with Frame Field Models

A complete example showing how to train a Frame Field segmentation model for precise building footprint extraction from aerial imagery, with post-processing to produce clean polygon vectors.

## Use Case

Standard segmentation models produce raster masks that require post-processing (e.g., contour extraction) to become polygon vectors. Frame Field models simultaneously predict:

1. **Interior mask** — which pixels belong to buildings
2. **Boundary mask** — the outlines of buildings
3. **Crossfield** — a tangent frame field that encodes the local orientation of building edges

The crossfield guides polygonization algorithms to produce axis-aligned, geometrically clean building footprints — significantly better than simple contour tracing.

:::note Inspiration
The Frame Field approach is based on [Polygonization by Frame Field Learning](https://github.com/Lydorn/Polygonization-by-Frame-Field-Learning). The `FrameFieldSegmentationPLModel` and `FrameFieldSegmentationDataset` in this library implement a compatible training pipeline.
:::

## Project Structure

```
frame_field_project/
├── data/
│   ├── train/
│   │   ├── images/          # RGB aerial imagery
│   │   ├── polygon_mask/    # Interior (building footprint) binary masks
│   │   ├── boundary_mask/   # Boundary (edge) binary masks
│   │   ├── vertex_mask/     # Vertex (corner) binary masks
│   │   ├── crossfield_mask/ # 2-channel crossfield angle images
│   │   ├── distance_mask/   # Distance transform masks
│   │   └── size_mask/       # Building size masks
│   └── val/
│       └── ...              # Same structure as train
├── configs/
│   └── train.yaml
├── train.csv
├── val.csv
└── outputs/
```

## Step 1: Generate Multi-Channel Masks

Frame Field training requires several mask types generated from polygon annotations (typically GeoJSON or shapefile). Use the project's built-in mask building tools with `--mode build_mask`.

```bash
pytorch-smt-mask-builder \
    --annotations data/annotations/buildings_train.geojson \
    --images-dir data/train/images \
    --output-dir data/train \
    --mode build_mask
```

This produces the following files per image tile:
- `polygon_mask/` — binary mask: 1 = building interior, 0 = background
- `boundary_mask/` — binary mask: 1 = building boundary pixels
- `vertex_mask/` — binary mask: 1 = corner/vertex pixels
- `crossfield_mask/` — 2-band float32 image encoding tangent frame angles
- `distance_mask/` — distance transform from each interior pixel to the nearest edge
- `size_mask/` — per-building normalized size map

:::tip See the Mask Building Guide
The [Mask Building user guide](/docs/user-guide/mask-building) explains the mask generation process in detail, including how to work with GeoJSON, shapefiles, and tiled imagery.
:::

## Step 2: Create CSV Files

The CSV must include columns for every mask type loaded by `FrameFieldSegmentationDataset`.

Create `train.csv`:
```csv
image,polygon_mask,boundary_mask,vertex_mask,crossfield_mask,distance_mask,size_mask
data/train/images/tile_001.png,data/train/polygon_mask/tile_001.png,data/train/boundary_mask/tile_001.png,data/train/vertex_mask/tile_001.png,data/train/crossfield_mask/tile_001.tif,data/train/distance_mask/tile_001.tif,data/train/size_mask/tile_001.tif
data/train/images/tile_002.png,data/train/polygon_mask/tile_002.png,data/train/boundary_mask/tile_002.png,data/train/vertex_mask/tile_002.png,data/train/crossfield_mask/tile_002.tif,data/train/distance_mask/tile_002.tif,data/train/size_mask/tile_002.tif
```

:::tip Generating the CSV Programmatically

```python
import pandas as pd
from pathlib import Path

def create_frame_field_csv(base_dir, output_csv):
    images = sorted(Path(base_dir, "images").glob("*.png"))
    rows = []
    for img in images:
        stem = img.stem
        row = {
            "image":           str(Path(base_dir, "images",          f"{stem}.png")),
            "polygon_mask":    str(Path(base_dir, "polygon_mask",    f"{stem}.png")),
            "boundary_mask":   str(Path(base_dir, "boundary_mask",   f"{stem}.png")),
            "vertex_mask":     str(Path(base_dir, "vertex_mask",     f"{stem}.png")),
            "crossfield_mask": str(Path(base_dir, "crossfield_mask", f"{stem}.tif")),
            "distance_mask":   str(Path(base_dir, "distance_mask",   f"{stem}.tif")),
            "size_mask":       str(Path(base_dir, "size_mask",       f"{stem}.tif")),
        }
        # Only include row if all mask files exist
        if all(Path(v).exists() for v in row.values()):
            rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(output_csv, index=False)
    print(f"Created {output_csv} with {len(df)} samples")

create_frame_field_csv("data/train", "train.csv")
create_frame_field_csv("data/val",   "val.csv")
```
:::

## Step 3: Training Configuration

Create `configs/train.yaml`:

```yaml
# --- Backbone segmentation network ---
# The backbone produces the feature maps used by the frame field head.
backbone:
  _target_: segmentation_models_pytorch.Unet
  encoder_name: resnet34
  encoder_weights: imagenet
  in_channels: 3
  classes: 3      # Interior + Boundary + Vertex channels
  activation: sigmoid

# --- Frame Field model flags ---
compute_seg: true          # Predict interior/boundary/vertex segmentation
compute_crossfield: true   # Predict the tangent frame field

# --- Segmentation output flags ---
seg_params:
  compute_interior: true
  compute_edge: true
  compute_vertex: true

# --- Training Dataset ---
train_dataset:
  _target_: pytorch_segmentation_models_trainer.dataset_loader.dataset.FrameFieldSegmentationDataset
  input_csv_path: train.csv
  image_width: 224
  image_height: 224
  # Mask column keys (must match CSV column names)
  mask_key: polygon_mask
  boundary_mask_key: boundary_mask
  vertex_mask_key: vertex_mask
  crossfield_mask_key: crossfield_mask
  distance_mask_key: distance_mask
  size_mask_key: size_mask
  # Which mask channels to load
  return_boundary_mask: true
  return_vertex_mask: true
  return_crossfield_mask: true
  return_distance_mask: true
  return_size_mask: true
  data_loader:
    shuffle: true
    num_workers: 4
    pin_memory: true
    drop_last: true
    prefetch_factor: 2
  augmentation_list:
    - _target_: albumentations.RandomRotate90
      p: 0.5
    - _target_: albumentations.HorizontalFlip
      p: 0.5
    - _target_: albumentations.VerticalFlip
      p: 0.5
    - _target_: albumentations.RandomBrightnessContrast
      brightness_limit: 0.2
      contrast_limit: 0.2
      p: 0.4
    - _target_: albumentations.RandomCrop
      height: 224
      width: 224
      always_apply: true
    - _target_: albumentations.Normalize
      mean: [0.485, 0.456, 0.406]
      std: [0.229, 0.224, 0.225]
      p: 1.0
    - _target_: albumentations.pytorch.transforms.ToTensorV2
      always_apply: true

# --- Validation Dataset ---
val_dataset:
  _target_: pytorch_segmentation_models_trainer.dataset_loader.dataset.FrameFieldSegmentationDataset
  input_csv_path: val.csv
  image_width: 224
  image_height: 224
  mask_key: polygon_mask
  boundary_mask_key: boundary_mask
  vertex_mask_key: vertex_mask
  crossfield_mask_key: crossfield_mask
  distance_mask_key: distance_mask
  size_mask_key: size_mask
  return_boundary_mask: true
  return_vertex_mask: true
  return_crossfield_mask: true
  return_distance_mask: true
  return_size_mask: true
  data_loader:
    shuffle: false
    num_workers: 4
    pin_memory: true
    drop_last: false
    prefetch_factor: 2
  augmentation_list:
    - _target_: albumentations.Resize
      height: 224
      width: 224
      always_apply: true
    - _target_: albumentations.Normalize
      mean: [0.485, 0.456, 0.406]
      std: [0.229, 0.224, 0.225]
      p: 1.0
    - _target_: albumentations.pytorch.transforms.ToTensorV2
      always_apply: true

# --- Compound Loss Configuration ---
# Frame Field training uses multiple loss terms that are normalized and summed.
# The compound_loss system handles all component losses automatically.
loss_params:
  compound_loss:
    normalize_losses: true
    losses:
      # Segmentation losses (applied to interior, boundary, and vertex channels)
      - name: seg_interior_bce
        _target_: torch.nn.BCEWithLogitsLoss
        weight: 1.0
      - name: seg_boundary_bce
        _target_: torch.nn.BCEWithLogitsLoss
        weight: 1.0
      - name: seg_crossfield_align
        # Coupling loss: aligns the crossfield tangents to building edges
        _target_: pytorch_segmentation_models_trainer.custom_losses.frame_field_losses.CrossfieldAlignLoss
        weight: 1.0
      - name: seg_smoothness
        # Crossfield smoothness regularization
        _target_: pytorch_segmentation_models_trainer.custom_losses.frame_field_losses.CrossfieldSmoothLoss
        weight: 0.1

# --- Optimizer ---
optimizer:
  _target_: torch.optim.AdamW
  lr: 0.001
  weight_decay: 1.0e-4
  eps: 1.0e-8

# --- Learning Rate Scheduler ---
scheduler_list:
  - scheduler:
      _target_: torch.optim.lr_scheduler.CosineAnnealingLR
      T_max: 100
      eta_min: 1.0e-7
    interval: epoch
    frequency: 1
    name: cosine_lr

# --- Hyperparameters ---
hyperparameters:
  batch_size: 8
  epochs: 100

# --- PyTorch Lightning Trainer ---
pl_trainer:
  max_epochs: ${hyperparameters.epochs}
  accelerator: gpu
  devices: 1
  precision: 16-mixed
  gradient_clip_val: 1.0
  gradient_clip_algorithm: norm
  check_val_every_n_epoch: 1
  log_every_n_steps: 20

# --- Callbacks ---
callbacks:
  - _target_: pytorch_lightning.callbacks.ModelCheckpoint
    monitor: loss/val
    mode: min
    save_top_k: 3
    save_last: true
    filename: "best-{epoch:02d}-{loss/val:.4f}"
    auto_insert_metric_name: false
  - _target_: pytorch_lightning.callbacks.EarlyStopping
    monitor: loss/val
    mode: min
    patience: 20
    min_delta: 0.0005
  - _target_: pytorch_lightning.callbacks.LearningRateMonitor
    logging_interval: epoch

# --- Logger ---
logger:
  _target_: pytorch_lightning.loggers.TensorBoardLogger
  save_dir: ./logs
  name: frame_field_seg

mode: train
device: cuda

# Device passed to ComputeSegGrads preprocessor inside FrameFieldSegmentationPLModel
device: cuda
```

:::note Loss Normalization
`normalize_losses: true` enables automatic normalization of each loss component so that no single term dominates training due to scale differences. Loss normalisation values are computed at the start of training from a subset of training batches.
:::

## Step 4: Run Training

```bash
cd frame_field_project
pytorch-smt --config-dir ./configs --config-name train
```

The trainer logs individual component losses (e.g., `losses/train_seg_interior_bce`, `losses/train_seg_crossfield_align`) in addition to the total loss, making it easy to diagnose training issues per loss term.

## Step 5: Inference with Polygonization

`SingleImageFromFrameFieldProcessor` runs tiled inference and merges both the segmentation (`seg`) and crossfield (`crossfield`) outputs across overlapping tiles.

Create `configs/predict.yaml`:

```yaml
backbone:
  _target_: segmentation_models_pytorch.Unet
  encoder_name: resnet34
  encoder_weights: null
  in_channels: 3
  classes: 3
  activation: sigmoid

compute_seg: true
compute_crossfield: true
seg_params:
  compute_interior: true
  compute_edge: true
  compute_vertex: true

mode: predict
device: cuda
checkpoint_path: ./logs/frame_field_seg/version_0/checkpoints/best-epoch=XX-loss_val=X.XXXX.ckpt

inference_image_reader:
  _target_: pytorch_segmentation_models_trainer.tools.data_handlers.raster_reader.FolderImageReaderProcessor
  folder_name: ./data/test/images
  recursive: true
  image_extension: png

inference_processor:
  _target_: pytorch_segmentation_models_trainer.tools.inference.inference_processors.SingleImageFromFrameFieldProcessor
  model_input_shape: [224, 224]
  step_shape: [112, 112]
  mask_bands: 3   # interior + boundary + vertex channels

# Polygonizer converts the segmentation + crossfield into polygon vectors
polygonizer:
  _target_: pytorch_segmentation_models_trainer.tools.polygonization.polygonizer.TemplatePolygonizerProcessor
  seg_threshold: 0.5
  min_area: 10        # Minimum polygon area in pixels to keep

export_strategy:
  _target_: pytorch_segmentation_models_trainer.tools.inference.export_inference.VectorExportInferenceStrategy
  output_file_path: ./predictions/{input_name}_buildings.geojson

inference_threshold: 0.5
save_inference: true
```

```bash
pytorch-smt --config-dir ./configs --config-name predict
```

## Understanding the Output

Inference produces two outputs per input image:

1. **Raster mask** — a multi-band GeoTIFF containing the interior, boundary, and vertex probability maps.
2. **Vector polygons** — a GeoJSON file with building footprint polygons derived via frame-field-guided polygonization.

The polygonization step uses the crossfield to snap polygon edges to dominant orientations, producing axis-aligned buildings typical of cartographic mapping.

### Inspect Polygon Output

```python
import geopandas as gpd
import matplotlib.pyplot as plt

gdf = gpd.read_file("predictions/tile_001_buildings.geojson")
print(f"Detected {len(gdf)} building polygons")
print(gdf.geometry.area.describe())

fig, ax = plt.subplots(figsize=(10, 10))
gdf.plot(ax=ax, facecolor="orange", edgecolor="red", linewidth=0.5, alpha=0.6)
ax.set_title("Extracted Building Footprints")
ax.axis("off")
plt.tight_layout()
plt.show()
```

## Next Steps

- Experiment with `encoder_name: resnet50` or `efficientnet-b4` for higher-capacity feature extraction
- Increase `image_width` / `image_height` to capture larger buildings at higher resolution
- Tune `min_area` in the polygonizer to filter small false-positive detections
- See the [Frame Field Dataset guide](/docs/user-guide/dataset-frame-field) for more on the multi-channel mask format
