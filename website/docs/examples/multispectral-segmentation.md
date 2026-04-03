---
sidebar_position: 2
title: Multispectral Satellite Imagery Segmentation
---

# Multispectral Satellite Imagery Segmentation

A complete example showing how to train a U-Net model for building and vegetation segmentation on 4-band RGBI (Red, Green, Blue, Near-Infrared) satellite imagery stored as GeoTIFF files.

## Use Case

Multispectral satellite sensors capture information beyond the visible spectrum. A common 4-band product includes:

- **Band 1**: Red
- **Band 2**: Green
- **Band 3**: Blue
- **Band 4**: Near-Infrared (NIR)

The NIR band is especially valuable for distinguishing vegetation from built-up surfaces, making it a powerful input for building footprint and vegetation mapping tasks.

:::note Rasterio Required
Multispectral GeoTIFF files must be loaded with `use_rasterio: true`. Standard PIL-based loading only handles 3-channel RGB images.
:::

## Project Structure

```
multispectral_project/
├── data/
│   ├── train/
│   │   ├── images/        # 4-band GeoTIFF files
│   │   └── masks/         # Binary PNG masks
│   └── val/
│       ├── images/
│       └── masks/
├── configs/
│   └── train.yaml
├── train.csv
├── val.csv
└── outputs/
```

## Step 1: Prepare Your Dataset

### File Format

Images must be multi-band GeoTIFF files readable by rasterio. Masks are single-channel binary PNG files where pixel value `1` (or `255`) indicates the target class (building or vegetation) and `0` is background.

### Create CSV Files

The CSV files must contain at minimum an `image` column and a `mask` column with absolute or relative paths.

Create `train.csv`:
```csv
image,mask
data/train/images/tile_001.tif,data/train/masks/tile_001.png
data/train/images/tile_002.tif,data/train/masks/tile_002.png
data/train/images/tile_003.tif,data/train/masks/tile_003.png
```

Create `val.csv`:
```csv
image,mask
data/val/images/tile_101.tif,data/val/masks/tile_101.png
data/val/images/tile_102.tif,data/val/masks/tile_102.png
```

:::tip Generating CSVs from a Directory
Use this script to auto-generate CSVs from a folder of GeoTIFFs and their corresponding masks:

```python
import pandas as pd
from pathlib import Path

def create_multispectral_csv(images_dir, masks_dir, output_csv):
    data = []
    for img_file in sorted(Path(images_dir).glob("*.tif")):
        mask_file = Path(masks_dir) / f"{img_file.stem}.png"
        if mask_file.exists():
            data.append({
                "image": str(img_file),
                "mask": str(mask_file),
            })
    df = pd.DataFrame(data)
    df.to_csv(output_csv, index=False)
    print(f"Created {output_csv} with {len(df)} samples")

create_multispectral_csv("data/train/images", "data/train/masks", "train.csv")
create_multispectral_csv("data/val/images", "data/val/masks", "val.csv")
```
:::

### Verify Band Statistics

Before training, inspect your imagery to understand the per-band value range:

```python
import rasterio
import numpy as np

with rasterio.open("data/train/images/tile_001.tif") as src:
    print(f"Number of bands: {src.count}")          # Should be 4
    print(f"Dtype: {src.dtypes[0]}")                # Typically uint8 or uint16
    print(f"Shape: {src.height} x {src.width}")
    for i in range(1, src.count + 1):
        band = src.read(i)
        print(f"Band {i} — min: {band.min()}, max: {band.max()}, mean: {band.mean():.1f}")
```

:::tip 16-bit Imagery
If your GeoTIFFs are 16-bit (`uint16`), set `image_dtype: uint16` in the dataset config. This preserves the full 16-bit precision and applies the correct normalization factor (`/65535`) automatically in the no-transform path. When using an augmentation pipeline, add `A.ToFloat(max_value=65535)` as the first transform. See the [16-bit example](#step-2b-16-bit-imagery-sentinel-2-landsat) below.
:::

## Step 2: Training Configuration

### Step 2a: 8-bit Imagery (standard GeoTIFF)

Create `configs/train.yaml`:

```yaml
# --- Model Architecture ---
model:
  _target_: segmentation_models_pytorch.Unet
  encoder_name: resnet34
  encoder_weights: imagenet
  in_channels: 4          # 4-band RGBI input
  classes: 1              # Binary segmentation output
  activation: null        # Raw logits; sigmoid applied by loss

# --- Training Dataset ---
train_dataset:
  _target_: pytorch_segmentation_models_trainer.dataset_loader.dataset.SegmentationDataset
  input_csv_path: train.csv
  use_rasterio: true          # Required for GeoTIFF multi-band loading
  selected_bands: [1, 2, 3, 4]  # 1-based band indices; loads all 4 bands
  image_dtype: uint8          # default; explicit for clarity
  n_classes: 2
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
      height: 256
      width: 256
      always_apply: true
    # Normalize each band independently using per-channel statistics.
    # Adjust mean/std to match your dataset's statistics.
    - _target_: albumentations.Normalize
      mean: [0.485, 0.456, 0.406, 0.350]   # R, G, B, NIR
      std: [0.229, 0.224, 0.225, 0.180]    # R, G, B, NIR
      max_pixel_value: 255.0
      p: 1.0
    - _target_: albumentations.pytorch.transforms.ToTensorV2
      always_apply: true

# --- Validation Dataset ---
val_dataset:
  _target_: pytorch_segmentation_models_trainer.dataset_loader.dataset.SegmentationDataset
  input_csv_path: val.csv
  use_rasterio: true
  selected_bands: [1, 2, 3, 4]
  image_dtype: uint8
  n_classes: 2
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
      mean: [0.485, 0.456, 0.406, 0.350]
      std: [0.229, 0.224, 0.225, 0.180]
      max_pixel_value: 255.0
      p: 1.0
    - _target_: albumentations.pytorch.transforms.ToTensorV2
      always_apply: true
```

### Step 2b: 16-bit Imagery (Sentinel-2, Landsat)

For sensors that store values as `uint16` (0–65535), set `image_dtype: uint16` and add `A.ToFloat` as the first augmentation transform. Albumentations does not handle `uint16` arrays natively, so the conversion to `float32` must happen before any other transform.

```yaml
train_dataset:
  _target_: pytorch_segmentation_models_trainer.dataset_loader.dataset.SegmentationDataset
  input_csv_path: train.csv
  use_rasterio: true
  selected_bands: [1, 2, 3, 4]
  image_dtype: uint16           # preserves full 16-bit precision
  n_classes: 2
  data_loader:
    shuffle: true
    num_workers: 4
    pin_memory: true
    drop_last: true
    prefetch_factor: 2
  augmentation_list:
    - _target_: albumentations.ToFloat
      max_value: 65535.0        # uint16 → float32 in [0, 1]
    - _target_: albumentations.RandomRotate90
      p: 0.5
    - _target_: albumentations.HorizontalFlip
      p: 0.5
    - _target_: albumentations.RandomCrop
      height: 256
      width: 256
      always_apply: true
    - _target_: albumentations.Normalize
      mean: [0.5, 0.5, 0.5, 0.4]
      std: [0.2, 0.2, 0.2, 0.15]
      max_pixel_value: 1.0      # values are already in [0, 1] after ToFloat
      p: 1.0
    - _target_: albumentations.pytorch.transforms.ToTensorV2
      always_apply: true

val_dataset:
  _target_: pytorch_segmentation_models_trainer.dataset_loader.dataset.SegmentationDataset
  input_csv_path: val.csv
  use_rasterio: true
  selected_bands: [1, 2, 3, 4]
  image_dtype: uint16
  n_classes: 2
  data_loader:
    shuffle: false
    num_workers: 4
    pin_memory: true
    drop_last: false
    prefetch_factor: 2
  augmentation_list:
    - _target_: albumentations.ToFloat
      max_value: 65535.0
    - _target_: albumentations.Resize
      height: 256
      width: 256
      always_apply: true
    - _target_: albumentations.Normalize
      mean: [0.5, 0.5, 0.5, 0.4]
      std: [0.2, 0.2, 0.2, 0.15]
      max_pixel_value: 1.0
      p: 1.0
    - _target_: albumentations.pytorch.transforms.ToTensorV2
      always_apply: true
```

:::note
The sections below — `loss_params`, `optimizer`, `scheduler_list`, `hyperparameters`, `pl_trainer`, `callbacks`, `metrics`, and `logger` — are the same regardless of `image_dtype`. Add them to your `configs/train.yaml` after the dataset section.
:::

```yaml
# --- Loss Function ---
# DiceLoss addresses class imbalance; BCEWithLogitsLoss stabilizes early training.
loss_params:
  compound_loss:
    normalize_losses: true
    losses:
      - name: dice
        _target_: segmentation_models_pytorch.losses.DiceLoss
        mode: binary
        smooth: 1.0
        weight: 0.5
      - name: bce
        _target_: torch.nn.BCEWithLogitsLoss
        weight: 0.5

# --- Optimizer ---
optimizer:
  _target_: torch.optim.AdamW
  lr: 0.0005
  weight_decay: 1.0e-4
  eps: 1.0e-8

# --- Learning Rate Scheduler ---
scheduler_list:
  - scheduler:
      _target_: torch.optim.lr_scheduler.CosineAnnealingLR
      T_max: 50
      eta_min: 1.0e-7
    interval: epoch
    frequency: 1
    name: cosine_lr

# --- Hyperparameters ---
hyperparameters:
  batch_size: 8
  epochs: 50

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
    patience: 12
    min_delta: 0.0005
  - _target_: pytorch_lightning.callbacks.LearningRateMonitor
    logging_interval: epoch

# --- Metrics ---
metrics:
  - _target_: torchmetrics.Dice
    num_classes: 1
  - _target_: torchmetrics.JaccardIndex
    task: binary

# --- Logger ---
logger:
  _target_: pytorch_lightning.loggers.TensorBoardLogger
  save_dir: ./logs
  name: multispectral_seg

mode: train
device: cuda
```

## Step 3: Run Training

```bash
cd multispectral_project
pytorch-smt --config-dir ./configs --config-name train
```

Monitor training in TensorBoard:

```bash
tensorboard --logdir ./logs
```

## Step 4: Inference on New GeoTIFFs

Create `configs/predict.yaml`:

```yaml
# Re-use the same model definition to ensure in_channels: 4 is respected
model:
  _target_: segmentation_models_pytorch.Unet
  encoder_name: resnet34
  encoder_weights: null     # Weights come from the checkpoint
  in_channels: 4
  classes: 1
  activation: null

mode: predict
device: cuda
checkpoint_path: ./logs/multispectral_seg/version_0/checkpoints/best-epoch=XX-loss_val=X.XXXX.ckpt

inference_image_reader:
  _target_: pytorch_segmentation_models_trainer.tools.data_handlers.raster_reader.FolderImageReaderProcessor
  folder_name: ./data/test/images
  recursive: true
  image_extension: tif

inference_processor:
  _target_: pytorch_segmentation_models_trainer.tools.inference.inference_processors.SingleImageInfereceProcessor
  model_input_shape: [256, 256]
  step_shape: [128, 128]
  # Match the same normalization used during training.
  # For 16-bit imagery add: normalize_max_value: 65535.0
  normalize_mean: [0.485, 0.456, 0.406, 0.350]
  normalize_std: [0.229, 0.224, 0.225, 0.180]
  # normalize_max_value: 65535.0   # uncomment for uint16 images

export_strategy:
  _target_: pytorch_segmentation_models_trainer.tools.inference.export_inference.RasterExportInferenceStrategy
  output_file_path: ./predictions/{input_name}_pred.tif

inference_threshold: 0.5
save_inference: true
```

```bash
pytorch-smt --config-dir ./configs --config-name predict
```

:::tip Preserving Geospatial Metadata
The `SingleImageInfereceProcessor` reads the full rasterio profile (CRS, transform, projection) from the input GeoTIFF and writes it to the output prediction raster. This means your output predictions are georeferenced and can be loaded directly in QGIS or ArcGIS.
:::

## Using Only a Subset of Bands

The `selected_bands` parameter is 1-based and maps directly to rasterio's `src.read(bands)` call. You can use any combination:

```yaml
# RGB only (skip NIR)
selected_bands: [1, 2, 3]
# in_channels must match: 3

# NIR only (single channel)
selected_bands: [4]
# in_channels must match: 1

# NIR + Red (for NDVI-inspired features)
selected_bands: [4, 1]
# in_channels must match: 2
```

Always ensure the `in_channels` value in the model config matches the length of `selected_bands`.

## Next Steps

- Try `encoder_name: efficientnet-b3` or `resnet50` for potentially higher accuracy
- Experiment with 5-band or 8-band imagery (e.g., WorldView-3) by extending `selected_bands`
- Use `InstanceSegmentationPLModel` to detect individual building instances rather than a binary mask
