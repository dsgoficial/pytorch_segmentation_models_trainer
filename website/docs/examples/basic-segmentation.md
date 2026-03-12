---
sidebar_position: 1
title: Basic Semantic Segmentation
---

# Basic Semantic Segmentation

A complete example showing how to train a U-Net model for binary segmentation using a custom dataset.

## Project Structure

```
segmentation_project/
├── data/
│   ├── train/
│   │   ├── images/
│   │   └── masks/
│   └── val/
│       ├── images/
│       └── masks/
├── configs/
│   ├── model/
│   │   └── unet.yaml
│   ├── dataset/
│   │   └── custom.yaml
│   └── train.yaml
├── train.csv
├── val.csv
└── outputs/
```

## Prepare Your Dataset

### Step 1: Organize Files

```
data/
├── train/
│   ├── images/
│   │   ├── img_001.jpg
│   │   ├── img_002.jpg
│   │   └── ...
│   └── masks/
│       ├── mask_001.png
│       ├── mask_002.png
│       └── ...
└── val/
    ├── images/
    │   ├── val_001.jpg
    │   └── ...
    └── masks/
        ├── val_001.png
        └── ...
```

### Step 2: Create CSV Files

Create `train.csv`:
```csv
image,mask
data/train/images/img_001.jpg,data/train/masks/mask_001.png
data/train/images/img_002.jpg,data/train/masks/mask_002.png
data/train/images/img_003.jpg,data/train/masks/mask_003.png
```

:::tip Automatic CSV Generation
Use this Python script to auto-generate CSV files:

```python
import os
import pandas as pd
from pathlib import Path

def create_csv(images_dir, masks_dir, output_csv):
    data = []
    for img_file in Path(images_dir).glob('*.jpg'):
        mask_file = Path(masks_dir) / f"{img_file.stem}.png"
        if mask_file.exists():
            data.append({
                'image': str(img_file),
                'mask': str(mask_file)
            })

    df = pd.DataFrame(data)
    df.to_csv(output_csv, index=False)
    print(f"Created {output_csv} with {len(df)} samples")

# Generate CSV files
create_csv('data/train/images', 'data/train/masks', 'train.csv')
create_csv('data/val/images', 'data/val/masks', 'val.csv')
```
:::

## Configuration Files

### Base Model Configuration

Create `configs/model/unet.yaml`:
```yaml
_target_: segmentation_models_pytorch.Unet
encoder_name: resnet34
encoder_weights: imagenet
in_channels: 3
classes: 1
activation: null  # We'll use sigmoid in loss
```

### Dataset Configuration

Create `configs/dataset/custom.yaml`:
```yaml
# Training Dataset
train_dataset:
  _target_: pytorch_segmentation_models_trainer.dataset_loader.dataset.SegmentationDataset
  input_csv_path: train.csv
  data_loader:
    shuffle: true
    num_workers: 4
    pin_memory: true
    drop_last: true
  augmentation_list:
    - _target_: albumentations.RandomRotate90
      p: 0.5
    - _target_: albumentations.HorizontalFlip
      p: 0.5
    - _target_: albumentations.VerticalFlip
      p: 0.2
    - _target_: albumentations.RandomBrightnessContrast
      brightness_limit: 0.2
      contrast_limit: 0.2
      p: 0.5
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

# Validation Dataset
val_dataset:
  _target_: pytorch_segmentation_models_trainer.dataset_loader.dataset.SegmentationDataset
  input_csv_path: val.csv
  data_loader:
    shuffle: false
    num_workers: 4
    pin_memory: true
    drop_last: false
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
```

### Main Training Configuration

Create `configs/train.yaml`:
```yaml
defaults:
  - model: unet
  - dataset: custom

loss:
  _target_: pytorch_segmentation_models_trainer.custom_losses.base_loss.CombinedLoss
  losses:
    dice:
      _target_: segmentation_models_pytorch.utils.losses.DiceLoss
      mode: binary
      smooth: 1.0
    bce:
      _target_: torch.nn.BCEWithLogitsLoss
  weights: [0.5, 0.5]

optimizer:
  _target_: torch.optim.AdamW
  lr: 0.001
  weight_decay: 1e-4
  eps: 1e-8

scheduler_list:
  - scheduler:
      _target_: torch.optim.lr_scheduler.ReduceLROnPlateau
      mode: min
      factor: 0.5
      patience: 5
      min_lr: 1e-7
    monitor: val_loss
    interval: epoch
    name: lr_scheduler

hyperparameters:
  batch_size: 8
  epochs: 50

pl_trainer:
  max_epochs: ${hyperparameters.epochs}
  accelerator: gpu
  devices: 1
  precision: 16
  gradient_clip_val: 1.0
  gradient_clip_algorithm: norm
  check_val_every_n_epoch: 1
  log_every_n_steps: 20

callbacks:
  - _target_: pytorch_lightning.callbacks.ModelCheckpoint
    monitor: val_loss
    mode: min
    save_top_k: 3
    save_last: true
    filename: 'best-{epoch:02d}-{val_loss:.4f}'
    auto_insert_metric_name: false
  - _target_: pytorch_lightning.callbacks.EarlyStopping
    monitor: val_loss
    mode: min
    patience: 10
    min_delta: 0.001
  - _target_: pytorch_lightning.callbacks.LearningRateMonitor
    logging_interval: epoch

metrics:
  - _target_: torchmetrics.Dice
    num_classes: 1
  - _target_: torchmetrics.JaccardIndex
    task: binary
  - _target_: torchmetrics.Accuracy
    task: binary

logger:
  _target_: pytorch_lightning.loggers.TensorBoardLogger
  save_dir: ./logs
  name: basic_segmentation
  version: ${now:%Y%m%d_%H%M%S}

mode: train
device: cuda
```

## Training

### Start Training

```bash
cd segmentation_project
pytorch-smt --config-dir ./configs --config-name train
```

### Monitor Progress

```bash
# In another terminal
tensorboard --logdir ./logs
```

Open http://localhost:6006 to view training/validation loss, metrics, and sample predictions.

## Making Predictions

Create `configs/predict.yaml`:
```yaml
defaults:
  - model: unet

mode: predict
device: cuda
checkpoint_path: ./lightning_logs/version_0/checkpoints/best-epoch=XX-val_loss=X.XXXX.ckpt

inference_image_reader:
  _target_: pytorch_segmentation_models_trainer.tools.data_handlers.raster_reader.FolderImageReaderProcessor
  folder_name: ./data/test/images
  recursive: true
  image_extension: jpg

inference_processor:
  _target_: pytorch_segmentation_models_trainer.tools.inference.inference_processors.SingleImageInfereceProcessor
  model_input_shape: [256, 256]
  step_shape: [128, 128]

export_strategy:
  _target_: pytorch_segmentation_models_trainer.tools.inference.export_inference.RasterExportInferenceStrategy
  output_file_path: ./predictions/mask_{input_name}.png

inference_threshold: 0.5
save_inference: true
```

```bash
pytorch-smt --config-dir ./configs --config-name predict
```

## Results Analysis

### Visualize Predictions

```python
import matplotlib.pyplot as plt
from PIL import Image

def visualize_results(image_path, mask_path, pred_path):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(Image.open(image_path))
    axes[0].set_title('Original Image')
    axes[0].axis('off')

    axes[1].imshow(Image.open(mask_path), cmap='gray')
    axes[1].set_title('Ground Truth')
    axes[1].axis('off')

    axes[2].imshow(Image.open(pred_path), cmap='gray')
    axes[2].set_title('Prediction')
    axes[2].axis('off')

    plt.tight_layout()
    plt.show()
```

## Next Steps

1. Try different encoders: `efficientnet-b3`, `resnet50`, `resnext50_32x4d`
2. Experiment with loss functions: FocalLoss, TverskyLoss
3. Add more augmentations: ElasticTransform, GridDistortion
