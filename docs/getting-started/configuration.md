# Configuration System

This library uses [Hydra](https://hydra.cc/) for configuration management, enabling flexible, composable, and reproducible experiments through YAML files.

## 🏗️ Configuration Structure

Every config file has these main sections:

```yaml
# Core Components
model: {...}           # Neural network architecture
loss: {...}            # Loss function  
optimizer: {...}       # Optimization algorithm
hyperparameters: {...} # Training parameters

# Data
train_dataset: {...}   # Training data configuration
val_dataset: {...}     # Validation data configuration

# Training
pl_trainer: {...}      # PyTorch Lightning trainer settings

# Mode and other settings
mode: train           # Operation mode
device: cuda          # Computing device
```

## 🎯 The `_target_` Pattern

Hydra uses `_target_` to specify which class to instantiate:

```yaml
# This creates: torch.optim.AdamW(lr=0.001, weight_decay=1e-4)
optimizer:
  _target_: torch.optim.AdamW
  lr: 0.001
  weight_decay: 1e-4

# This creates: segmentation_models_pytorch.Unet(...)  
model:
  _target_: segmentation_models_pytorch.Unet
  encoder_name: resnet34
  classes: 1
```

## 📁 File Organization

Organize configs by purpose:

```
configs/
├── model/
│   ├── unet.yaml
│   ├── deeplabv3.yaml
│   └── frame_field.yaml
├── dataset/
│   ├── cityscapes.yaml
│   ├── custom.yaml
│   └── coco.yaml
├── optimizer/
│   ├── adam.yaml
│   └── sgd.yaml
└── experiment/
    ├── quick_test.yaml
    └── production.yaml
```

## 🔧 Configuration Composition

### Basic Composition

Create a base config:

```yaml title="configs/base.yaml"
defaults:
  - model: unet
  - optimizer: adam
  - dataset: custom

hyperparameters:
  batch_size: 4
  epochs: 10

mode: train
```

Then create specific components:

```yaml title="configs/model/unet.yaml"
_target_: segmentation_models_pytorch.Unet
encoder_name: resnet34
encoder_weights: imagenet
in_channels: 3
classes: 1
```

```yaml title="configs/optimizer/adam.yaml" 
_target_: torch.optim.AdamW
lr: 0.001
weight_decay: 1e-4
```

### Using Composition

```bash
# Use base config with defaults
pytorch-smt --config-dir ./configs --config-name base

# Override specific components
pytorch-smt --config-dir ./configs --config-name base \
  model=deeplabv3 optimizer=sgd

# Override individual parameters
pytorch-smt --config-dir ./configs --config-name base \
  hyperparameters.batch_size=8 optimizer.lr=0.01
```

## 🎛️ Advanced Features

### Variable Interpolation

Reference other config values:

```yaml
hyperparameters:
  batch_size: 4
  epochs: 100
  
pl_trainer:
  max_epochs: ${hyperparameters.epochs}  # References epochs above
  
callbacks:
  - _target_: pytorch_lightning.callbacks.ModelCheckpoint
    every_n_epochs: ${div:${hyperparameters.epochs},10}  # epochs/10
```

### Conditional Configuration

```yaml
# Based on model type
defaults:
  - model: unet
  - optimizer: adam
  - loss: dice  # Default loss
  - override /loss: focal_loss  # Use if model has many classes

model:
  classes: 10  # This would trigger focal_loss override
```

### Environment-Specific Configs

```yaml title="configs/local.yaml"
# For local development
hyperparameters:
  batch_size: 2
  epochs: 2
  
pl_trainer:
  max_epochs: 2
  fast_dev_run: true
```

```yaml title="configs/production.yaml"
# For full training runs
hyperparameters:
  batch_size: 16
  epochs: 100

pl_trainer:
  max_epochs: 100
  precision: 16
```

## 📋 Common Configuration Patterns

### Multi-GPU Training

```yaml
pl_trainer:
  accelerator: gpu
  devices: 2
  strategy: ddp
  
hyperparameters:
  batch_size: 8  # Per GPU
```

### Mixed Precision Training

```yaml
pl_trainer:
  precision: 16
  
# Might need to adjust learning rate
optimizer:
  lr: 0.002  # Higher LR for mixed precision
```

### Experiment Tracking

```yaml
logger:
  _target_: pytorch_lightning.loggers.TensorBoardLogger
  save_dir: ./experiments
  name: ${model.encoder_name}_${optimizer._target_}
  version: ${now:%Y-%m-%d_%H-%M-%S}
```

### Custom Callbacks

```yaml
callbacks:
  - _target_: pytorch_lightning.callbacks.ModelCheckpoint
    monitor: val_loss
    mode: min
    save_top_k: 3
    filename: 'best-{epoch:02d}-{val_loss:.2f}'
    
  - _target_: pytorch_lightning.callbacks.EarlyStopping
    monitor: val_loss
    patience: 10
    
  - _target_: pytorch_lightning.callbacks.LearningRateMonitor
    logging_interval: epoch
```

## 🔄 Mode-Specific Configurations

### Training Mode

```yaml
mode: train

# Training-specific settings
pl_trainer:
  max_epochs: 100
  log_every_n_steps: 50
  
# Checkpointing
callbacks:
  - _target_: pytorch_lightning.callbacks.ModelCheckpoint
    save_top_k: 3
```

### Prediction Mode

```yaml
mode: predict
checkpoint_path: /path/to/model.ckpt

inference_image_reader:
  _target_: pytorch_segmentation_models_trainer.tools.data_handlers.raster_reader.FolderImageReaderProcessor
  folder_name: /path/to/images

inference_processor:
  _target_: pytorch_segmentation_models_trainer.tools.inference.inference_processors.SingleImageInfereceProcessor
  
inference_threshold: 0.5
```

### Validation Mode

```yaml
mode: validate-config

# This mode just validates your config without training
# Useful for debugging configuration issues
```

## 🐛 Debugging Configurations

### Print Resolved Config

```bash
# See the final resolved configuration
pytorch-smt --config-dir ./configs --config-name my_config --cfg job
```

### Validate Config

```bash
# Just validate without running
pytorch-smt --config-dir ./configs --config-name my_config +mode=validate-config
```

### Override Validation

```bash
# Test parameter overrides
pytorch-smt --config-dir ./configs --config-name my_config \
  hyperparameters.batch_size=1 +mode=validate-config
```

## 💡 Best Practices

1. **Use composition**: Break configs into reusable components
2. **Name meaningfully**: Use descriptive names for config files
3. **Version control**: Track config changes alongside code
4. **Document experiments**: Use descriptive names and comments
5. **Validate early**: Use config validation to catch errors

## 🔍 Examples

### Experiment Configuration

```yaml title="configs/experiments/building_extraction.yaml"
# @package _global_
defaults:
  - base
  - model: unet
  - optimizer: adamw
  - loss: combined_loss
  - override /callbacks: [checkpoint, early_stopping, lr_monitor]

# Experiment-specific overrides  
model:
  encoder_name: efficientnet-b3
  classes: 1

hyperparameters:
  batch_size: 6
  epochs: 150
  
optimizer:
  lr: 0.0003
  weight_decay: 1e-5

# Custom augmentation for buildings
train_dataset:
  augmentation_list:
    - _target_: albumentations.RandomRotate90
    - _target_: albumentations.HorizontalFlip
    - _target_: albumentations.RandomBrightnessContrast
      brightness_limit: 0.1
      contrast_limit: 0.1
    - _target_: albumentations.RandomCrop
      height: 512
      width: 512
    - _target_: albumentations.Normalize
    - _target_: albumentations.pytorch.transforms.ToTensorV2

# Experiment metadata
experiment:
  name: building_extraction_efficientnet_b3
  description: "Building extraction with EfficientNet-B3 encoder"
  tags: [buildings, efficientnet, urban]
```

## 📚 Learn More

- [Hydra Documentation](https://hydra.cc/docs/intro) - Official Hydra docs
- [User Guide](../user-guide/training.md) - Detailed usage examples  
- [Examples](../examples/basic-segmentation.md) - Real configuration examples
- [API Reference](../api/main.md) - Available configuration options