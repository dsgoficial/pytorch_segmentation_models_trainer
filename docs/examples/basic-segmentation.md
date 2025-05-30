# Basic Semantic Segmentation

A complete example showing how to train a U-Net model for binary segmentation using a custom dataset.

## 📁 Project Structure

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

## 🗂️ Prepare Your Dataset

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

Create `val.csv`:
```csv
image,mask
data/val/images/val_001.jpg,data/val/masks/val_001.png
data/val/images/val_002.jpg,data/val/masks/val_002.png
```

!!! tip "Automatic CSV Generation"
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

## ⚙️ Configuration Files

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
    # Geometric augmentations
    - _target_: albumentations.RandomRotate90
      p: 0.5
    - _target_: albumentations.HorizontalFlip
      p: 0.5
    - _target_: albumentations.VerticalFlip
      p: 0.2
    
    # Intensity augmentations  
    - _target_: albumentations.RandomBrightnessContrast
      brightness_limit: 0.2
      contrast_limit: 0.2
      p: 0.5
    - _target_: albumentations.HueSaturationValue
      hue_shift_limit: 20
      sat_shift_limit: 30
      val_shift_limit: 20
      p: 0.3
    
    # Spatial augmentations
    - _target_: albumentations.RandomCrop
      height: 256
      width: 256
      always_apply: true
    
    # Normalization (always last)
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
    # Only resize and normalize for validation
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

# Loss Function - Combined for better performance
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

# Optimizer
optimizer:
  _target_: torch.optim.AdamW
  lr: 0.001
  weight_decay: 1e-4
  eps: 1e-8

# Learning Rate Scheduler  
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

# Training Hyperparameters
hyperparameters:
  batch_size: 8
  epochs: 50

# PyTorch Lightning Trainer Configuration
pl_trainer:
  max_epochs: ${hyperparameters.epochs}
  accelerator: gpu
  devices: 1
  precision: 16  # Mixed precision for faster training
  
  # Gradient clipping
  gradient_clip_val: 1.0
  gradient_clip_algorithm: norm
  
  # Validation
  check_val_every_n_epoch: 1
  log_every_n_steps: 20

# Callbacks
callbacks:
  # Model checkpointing
  - _target_: pytorch_lightning.callbacks.ModelCheckpoint
    monitor: val_loss
    mode: min
    save_top_k: 3
    save_last: true
    filename: 'best-{epoch:02d}-{val_loss:.4f}'
    auto_insert_metric_name: false
  
  # Early stopping
  - _target_: pytorch_lightning.callbacks.EarlyStopping
    monitor: val_loss
    mode: min
    patience: 10
    min_delta: 0.001
    
  # Learning rate monitoring
  - _target_: pytorch_lightning.callbacks.LearningRateMonitor
    logging_interval: epoch

# Metrics
metrics:
  - _target_: torchmetrics.Dice
    num_classes: 1
  - _target_: torchmetrics.JaccardIndex
    task: binary
  - _target_: torchmetrics.Accuracy
    task: binary

# Logger
logger:
  _target_: pytorch_lightning.loggers.TensorBoardLogger
  save_dir: ./logs
  name: basic_segmentation
  version: ${now:%Y%m%d_%H%M%S}

# Mode
mode: train
device: cuda
```

## 🏃‍♂️ Training

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

Open http://localhost:6006 to view:
- Training/validation loss curves
- Metrics (Dice, IoU, Accuracy)
- Learning rate schedule
- Sample predictions

## 🔮 Making Predictions

### Create Prediction Config

Create `configs/predict.yaml`:
```yaml
defaults:
  - model: unet

# Prediction mode
mode: predict
device: cuda
checkpoint_path: ./lightning_logs/version_0/checkpoints/best-epoch=XX-val_loss=X.XXXX.ckpt

# Input images
inference_image_reader:
  _target_: pytorch_segmentation_models_trainer.tools.data_handlers.raster_reader.FolderImageReaderProcessor
  folder_name: ./data/test/images
  recursive: true
  image_extension: jpg

# Inference processor
inference_processor:
  _target_: pytorch_segmentation_models_trainer.tools.inference.inference_processors.SingleImageInfereceProcessor
  model_input_shape: [256, 256]
  step_shape: [128, 128]  # Overlap for large images

# Export predictions
export_strategy:
  _target_: pytorch_segmentation_models_trainer.tools.inference.export_inference.RasterExportInferenceStrategy
  output_file_path: ./predictions/mask_{input_name}.png

# Threshold for binary predictions
inference_threshold: 0.5
save_inference: true
```

### Run Predictions

```bash
pytorch-smt --config-dir ./configs --config-name predict
```

## 📊 Results Analysis

### Visualize Predictions

```python
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

def visualize_results(image_path, mask_path, pred_path):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Original image
    img = Image.open(image_path)
    axes[0].imshow(img)
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    # Ground truth mask
    mask = Image.open(mask_path)
    axes[1].imshow(mask, cmap='gray')
    axes[1].set_title('Ground Truth')
    axes[1].axis('off')
    
    # Prediction
    pred = Image.open(pred_path)
    axes[2].imshow(pred, cmap='gray')
    axes[2].set_title('Prediction')
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.show()

# Visualize results
visualize_results(
    'data/val/images/val_001.jpg',
    'data/val/masks/val_001.png', 
    'predictions/mask_val_001.png'
)
```

### Calculate Metrics

```python
import torch
import torchmetrics
from PIL import Image
import numpy as np

def calculate_metrics(mask_dir, pred_dir):
    dice = torchmetrics.Dice(num_classes=1)
    iou = torchmetrics.JaccardIndex(task='binary')
    
    dice_scores = []
    iou_scores = []
    
    for mask_file in Path(mask_dir).glob('*.png'):
        pred_file = Path(pred_dir) / f"mask_{mask_file.name}"
        
        if pred_file.exists():
            # Load masks
            gt_mask = np.array(Image.open(mask_file)) / 255.0
            pred_mask = np.array(Image.open(pred_file)) / 255.0
            
            # Convert to tensors
            gt_tensor = torch.from_numpy(gt_mask).float()
            pred_tensor = torch.from_numpy(pred_mask).float()
            
            # Calculate metrics
            dice_score = dice(pred_tensor, gt_tensor)
            iou_score = iou(pred_tensor, gt_tensor)
            
            dice_scores.append(dice_score.item())
            iou_scores.append(iou_score.item())
    
    print(f"Average Dice Score: {np.mean(dice_scores):.4f} ± {np.std(dice_scores):.4f}")
    print(f"Average IoU Score: {np.mean(iou_scores):.4f} ± {np.std(iou_scores):.4f}")

# Calculate metrics
calculate_metrics('data/val/masks', 'predictions')
```

## 🚀 Next Steps

### Improve Performance

1. **Try different encoders**:
   ```yaml
   model:
     encoder_name: efficientnet-b3  # or resnet50, resnext50_32x4d
   ```

2. **Experiment with loss functions**:
   ```yaml
   loss:
     _target_: segmentation_models_pytorch.utils.losses.FocalLoss
     mode: binary
   ```

3. **Advanced augmentations**:
   ```yaml
   augmentation_list:
     - _target_: albumentations.ElasticTransform
       alpha: 1
       sigma: 50
       p: 0.3
   ```

### Scale Up

- [Multi-GPU Training](../user-guide/training.md#multi-gpu-training)
- [Mixed Precision](../user-guide/training.md#mixed-precision) 
- [Large Image Processing](../advanced/frame-field.md#large-images)

### Advanced Features

- [Frame Field Models](../advanced/frame-field.md) for precise boundaries
- [Polygonization](../advanced/polygonization.md) for vector outputs
- [Object Detection](../advanced/object-detection.md) for instance-level results