---
sidebar_position: 3
title: Multi-Class Semantic Segmentation
---

# Multi-Class Semantic Segmentation

A complete example showing how to train a model for land cover classification with multiple mutually-exclusive classes such as buildings, vegetation, roads, and water.

## Use Case

Multi-class segmentation assigns each pixel to exactly one of N classes. This differs from binary segmentation, where a single foreground/background decision is made per pixel.

For this example the classes are:

| Class Index | Label       |
|-------------|-------------|
| 0           | Background  |
| 1           | Buildings   |
| 2           | Vegetation  |
| 3           | Roads       |
| 4           | Water       |

:::note Key Differences from Binary Segmentation
- The model outputs `N` logit channels (one per class) instead of a single channel.
- Loss uses `CrossEntropyLoss` which expects integer class-index masks (not float binary masks).
- Masks must contain raw class indices (0, 1, 2, ...) as pixel values.
- Inference uses `argmax` over the class dimension, not a sigmoid threshold.
:::

## Project Structure

```
multiclass_project/
├── data/
│   ├── train/
│   │   ├── images/      # RGB images
│   │   └── masks/       # Single-channel PNG: pixel values = class indices
│   └── val/
│       ├── images/
│       └── masks/
├── configs/
│   └── train.yaml
├── train.csv
├── val.csv
└── outputs/
```

## Step 1: Prepare Multi-Class Masks

Each mask is a single-channel PNG image where every pixel holds an integer class index from `0` to `N-1`. Do **not** scale values to 255 — values must be the raw class indices.

Example: for a 4-class problem, a pixel belonging to "Vegetation" has value `2`.

### Generate Masks from GeoJSON Labels

```python
import numpy as np
import rasterio
from rasterio.features import rasterize
from shapely.geometry import shape
import json
from pathlib import Path

CLASS_MAP = {
    "background": 0,
    "buildings": 1,
    "vegetation": 2,
    "roads": 3,
    "water": 4,
}

def rasterize_geojson(geojson_path, reference_tif, output_mask_path):
    """Burn GeoJSON polygons into a class-index mask matching the reference raster."""
    with rasterio.open(reference_tif) as src:
        transform = src.transform
        out_shape = (src.height, src.width)
        crs = src.crs

    with open(geojson_path) as f:
        features = json.load(f)["features"]

    # Build (geometry, class_value) pairs — later classes overwrite earlier ones
    shapes = [
        (shape(feat["geometry"]), CLASS_MAP[feat["properties"]["class"]])
        for feat in features
        if feat["properties"]["class"] in CLASS_MAP
    ]

    mask = rasterize(
        shapes,
        out_shape=out_shape,
        transform=transform,
        fill=0,             # Background
        dtype=np.uint8,
    )

    # Save as single-band PNG
    profile = {
        "driver": "PNG",
        "dtype": "uint8",
        "height": out_shape[0],
        "width": out_shape[1],
        "count": 1,
    }
    with rasterio.open(output_mask_path, "w", **profile) as dst:
        dst.write(mask[np.newaxis, ...])
    print(f"Saved mask to {output_mask_path}")
```

### Verify Mask Values

```python
import numpy as np
from PIL import Image

mask = np.array(Image.open("data/train/masks/tile_001.png"))
print(f"Unique class indices present: {np.unique(mask)}")
# Expected output: [0 1 2 3 4]  (a subset is fine if not all classes appear in every tile)
print(f"Mask dtype: {mask.dtype}")   # Must be uint8
print(f"Mask shape: {mask.shape}")   # (H, W)  — single channel, no colour dimension
```

### Create CSV Files

Create `train.csv`:
```csv
image,mask
data/train/images/tile_001.png,data/train/masks/tile_001.png
data/train/images/tile_002.png,data/train/masks/tile_002.png
data/train/images/tile_003.png,data/train/masks/tile_003.png
```

## Step 2: Training Configuration

Create `configs/train.yaml`:

```yaml
# --- Model Architecture ---
model:
  _target_: segmentation_models_pytorch.Unet
  encoder_name: resnet34
  encoder_weights: imagenet
  in_channels: 3
  classes: 5          # One output channel per class (including background)
  activation: null    # No activation — CrossEntropyLoss expects raw logits

# --- Training Dataset ---
train_dataset:
  _target_: pytorch_segmentation_models_trainer.dataset_loader.dataset.SegmentationDataset
  input_csv_path: train.csv
  n_classes: 5        # Tells the dataset NOT to binarize the mask
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
    - _target_: albumentations.ShiftScaleRotate
      shift_limit: 0.05
      scale_limit: 0.1
      rotate_limit: 15
      p: 0.4
    - _target_: albumentations.RandomCrop
      height: 256
      width: 256
      always_apply: true
    - _target_: albumentations.Normalize
      mean: [0.485, 0.456, 0.406]
      std: [0.229, 0.224, 0.225]
      p: 1.0
    - _target_: albumentations.pytorch.transforms.ToTensorV2
      always_apply: true

# --- Validation Dataset ---
val_dataset:
  _target_: pytorch_segmentation_models_trainer.dataset_loader.dataset.SegmentationDataset
  input_csv_path: val.csv
  n_classes: 5
  data_loader:
    shuffle: false
    num_workers: 4
    pin_memory: true
    drop_last: false
    prefetch_factor: 2
  augmentation_list:
    - _target_: albumentations.Resize
      height: 256
      width: 256
      always_apply: true
    - _target_: albumentations.Normalize
      mean: [0.485, 0.456, 0.406]
      std: [0.229, 0.224, 0.225]
      p: 1.0
    - _target_: albumentations.pytorch.transforms.ToTensorV2
      always_apply: true

# --- Loss Function ---
# CrossEntropyLoss is the standard choice for multi-class segmentation.
# It expects: predictions shape [B, C, H, W] (logits), masks shape [B, H, W] (long).
# Optional: pass class weights to handle imbalanced datasets.
loss:
  _target_: torch.nn.CrossEntropyLoss
  # Uncomment to weight rare classes higher:
  # weight: [0.1, 2.0, 1.5, 2.5, 3.0]   # [background, buildings, vegetation, roads, water]
  ignore_index: 255     # Pixels labelled 255 are ignored (useful for uncertain regions)

# --- Optimizer ---
optimizer:
  _target_: torch.optim.AdamW
  lr: 0.001
  weight_decay: 1.0e-4
  eps: 1.0e-8

# --- Learning Rate Scheduler ---
scheduler_list:
  - scheduler:
      _target_: torch.optim.lr_scheduler.ReduceLROnPlateau
      mode: min
      factor: 0.5
      patience: 7
      min_lr: 1.0e-7
    monitor: loss/val
    interval: epoch
    frequency: 1
    name: plateau_lr

# --- Hyperparameters ---
hyperparameters:
  batch_size: 8
  epochs: 60

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
    patience: 15
    min_delta: 0.001
  - _target_: pytorch_lightning.callbacks.LearningRateMonitor
    logging_interval: epoch

# --- Metrics ---
metrics:
  - _target_: torchmetrics.JaccardIndex
    task: multiclass
    num_classes: 5
    average: macro
  - _target_: torchmetrics.Accuracy
    task: multiclass
    num_classes: 5
    average: macro

# --- Logger ---
logger:
  _target_: pytorch_lightning.loggers.TensorBoardLogger
  save_dir: ./logs
  name: multiclass_seg

mode: train
device: cuda
```

:::tip n_classes Controls Mask Loading Behaviour
Setting `n_classes: 5` in the dataset config tells `SegmentationDataset` to load masks as raw integer indices (`is_binary_mask=False`). If `n_classes` is left at the default value of `2`, the mask is binarized (`(mask > 0).astype(uint8)`), which destroys multi-class label information.
:::

## Step 3: Run Training

```bash
cd multiclass_project
pytorch-smt --config-dir ./configs --config-name train
```

## Step 4: Inference

The `MultiClassInferenceProcessor` merges overlapping tiles by averaging softmax probabilities across tiles and then applies `argmax` to produce a single-band class-index raster.

Create `configs/predict.yaml`:

```yaml
model:
  _target_: segmentation_models_pytorch.Unet
  encoder_name: resnet34
  encoder_weights: null
  in_channels: 3
  classes: 5
  activation: null

mode: predict
device: cuda
checkpoint_path: ./logs/multiclass_seg/version_0/checkpoints/best-epoch=XX-loss_val=X.XXXX.ckpt

inference_image_reader:
  _target_: pytorch_segmentation_models_trainer.tools.data_handlers.raster_reader.FolderImageReaderProcessor
  folder_name: ./data/test/images
  recursive: true
  image_extension: png

inference_processor:
  _target_: pytorch_segmentation_models_trainer.tools.inference.inference_processors.MultiClassInferenceProcessor
  model_input_shape: [256, 256]
  step_shape: [128, 128]
  num_classes: 5   # Must match model classes

export_strategy:
  _target_: pytorch_segmentation_models_trainer.tools.inference.export_inference.RasterExportInferenceStrategy
  output_file_path: ./predictions/{input_name}_classes.tif

inference_threshold: 0.5   # Not applied for multi-class; argmax is used instead
save_inference: true
```

```bash
pytorch-smt --config-dir ./configs --config-name predict
```

## Interpreting the Output

The output raster is a single-band `uint8` GeoTIFF where every pixel contains a class index:

| Pixel Value | Meaning     |
|-------------|-------------|
| 0           | Background  |
| 1           | Buildings   |
| 2           | Vegetation  |
| 3           | Roads       |
| 4           | Water       |

### Visualize with a Colour Map

```python
import numpy as np
import rasterio
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

CLASS_COLOURS = [
    "#000000",   # 0 Background  — black
    "#FF4500",   # 1 Buildings   — red-orange
    "#228B22",   # 2 Vegetation  — forest green
    "#808080",   # 3 Roads       — grey
    "#1E90FF",   # 4 Water       — dodger blue
]
CLASS_LABELS = ["Background", "Buildings", "Vegetation", "Roads", "Water"]

with rasterio.open("predictions/tile_001_classes.tif") as src:
    class_map = src.read(1)

cmap = mcolors.ListedColormap(CLASS_COLOURS)
bounds = [-0.5, 0.5, 1.5, 2.5, 3.5, 4.5]
norm = mcolors.BoundaryNorm(bounds, cmap.N)

fig, ax = plt.subplots(figsize=(8, 8))
im = ax.imshow(class_map, cmap=cmap, norm=norm, interpolation="nearest")
cbar = fig.colorbar(im, ax=ax, ticks=[0, 1, 2, 3, 4])
cbar.ax.set_yticklabels(CLASS_LABELS)
ax.set_title("Predicted Land Cover Classes")
ax.axis("off")
plt.tight_layout()
plt.savefig("predictions/tile_001_coloured.png", dpi=150)
plt.show()
```

## Next Steps

- Add per-class IoU logging by configuring `torchmetrics.JaccardIndex` with `average: none`
- Experiment with `FocalLoss` to down-weight easy background pixels
- Try DeepLabV3+ or PAN architectures for larger receptive fields on high-resolution imagery
