---
sidebar_position: 2
title: Quick Start
---

# Quick Start

Get up and running with your first segmentation model in 5 minutes!

## Prerequisites

- [Installation completed](/docs/getting-started/installation)
- Basic understanding of semantic segmentation
- Your dataset prepared (or use our sample data)

## Your First Model

### Step 1: Prepare Your Data

Create a CSV file listing your images and masks:

```csv
image,mask
/path/to/images/img1.jpg,/path/to/masks/mask1.png
/path/to/images/img2.jpg,/path/to/masks/mask2.png
/path/to/images/img3.jpg,/path/to/masks/mask3.png
```

:::tip Data Format
- Images: JPG, PNG, TIFF
- Masks: PNG with pixel values 0 (background) and 255 (foreground)
- For multi-class: pixel values 0, 1, 2, ... for each class
:::

### Step 2: Create Configuration

Create `configs/basic_unet.yaml`:

```yaml title="configs/basic_unet.yaml"
# Model Configuration
model:
  _target_: segmentation_models_pytorch.Unet
  encoder_name: resnet34
  encoder_weights: imagenet
  in_channels: 3
  classes: 1

# Loss Function
loss:
  _target_: segmentation_models_pytorch.utils.losses.DiceLoss

# Optimizer
optimizer:
  _target_: torch.optim.AdamW
  lr: 0.001
  weight_decay: 1e-4

# Training Parameters
hyperparameters:
  batch_size: 4
  epochs: 10

# PyTorch Lightning Trainer
pl_trainer:
  max_epochs: ${hyperparameters.epochs}
  gpus: 1  # Set to 0 for CPU

# Training Dataset
train_dataset:
  _target_: pytorch_segmentation_models_trainer.dataset_loader.dataset.SegmentationDataset
  input_csv_path: /path/to/train.csv
  data_loader:
    shuffle: true
    num_workers: 4
    pin_memory: true
  augmentation_list:
    - _target_: albumentations.RandomCrop
      height: 256
      width: 256
      always_apply: true
    - _target_: albumentations.HorizontalFlip
      p: 0.5
    - _target_: albumentations.Normalize
      p: 1.0
    - _target_: albumentations.pytorch.transforms.ToTensorV2
      always_apply: true

# Validation Dataset
val_dataset:
  _target_: pytorch_segmentation_models_trainer.dataset_loader.dataset.SegmentationDataset
  input_csv_path: /path/to/val.csv
  data_loader:
    shuffle: false
    num_workers: 4
    pin_memory: true
  augmentation_list:
    - _target_: albumentations.Resize
      height: 256
      width: 256
      always_apply: true
    - _target_: albumentations.Normalize
      p: 1.0
    - _target_: albumentations.pytorch.transforms.ToTensorV2
      always_apply: true

# Mode
mode: train
```

:::tip Config Builder
Use the [Config Builder](/config-builder/) to generate this YAML visually — no manual editing needed!
:::

### Step 3: Train Your Model

```bash
pytorch-smt --config-dir ./configs --config-name basic_unet
```

That's it! Your model will start training and save checkpoints automatically.

## Monitor Training

### TensorBoard (Optional)

Add to your config:

```yaml
logger:
  _target_: pytorch_lightning.loggers.TensorBoardLogger
  save_dir: ./logs
  name: basic_unet
```

Then view logs:
```bash
tensorboard --logdir ./logs
```

## Make Predictions

### Step 1: Create Prediction Config

Create `configs/predict.yaml`:

```yaml
model:
  _target_: segmentation_models_pytorch.Unet
  encoder_name: resnet34
  encoder_weights: imagenet
  in_channels: 3
  classes: 1

mode: predict
checkpoint_path: /path/to/your/checkpoint.ckpt
device: cuda  # or cpu

inference_image_reader:
  _target_: pytorch_segmentation_models_trainer.tools.data_handlers.raster_reader.FolderImageReaderProcessor
  folder_name: /path/to/test/images
  image_extension: jpg

inference_processor:
  _target_: pytorch_segmentation_models_trainer.tools.inference.inference_processors.SingleImageInfereceProcessor
  model_input_shape: [256, 256]

export_strategy:
  _target_: pytorch_segmentation_models_trainer.tools.inference.export_inference.RasterExportInferenceStrategy
  output_file_path: /path/to/output/prediction.tif

inference_threshold: 0.5
```

### Step 2: Run Prediction

```bash
pytorch-smt --config-dir ./configs --config-name predict
```

## Common Adjustments

### Reduce Memory Usage
```yaml
hyperparameters:
  batch_size: 1  # Smaller batches

pl_trainer:
  precision: 16  # Half precision
```

### Speed Up Training
```yaml
train_dataset:
  data_loader:
    num_workers: 8
    pin_memory: true

pl_trainer:
  accelerator: gpu
  devices: 2  # Multi-GPU
```

### Multi-Class Segmentation
```yaml
model:
  classes: 3  # Number of classes

loss:
  _target_: torch.nn.CrossEntropyLoss
```

## Next Steps

- [Training Guide](/docs/user-guide/training-segmentation) - Full training configuration reference
- [Dataset Building](/docs/user-guide/dataset-segmentation) - Prepare your own dataset
- [Examples](/docs/examples/basic-segmentation) - More worked examples
