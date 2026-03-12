---
sidebar_position: 1
title: Main Module
---

# Main Module

The main entry point for the pytorch-segmentation-models-trainer CLI and core functionality.

## Overview

The main module provides the primary interface for training, prediction, and other operations. It uses Hydra for configuration management and supports multiple execution modes.

## CLI Usage

```bash
# Training
pytorch-smt --config-dir ./configs --config-name my_config +mode=train

# Prediction
pytorch-smt --config-dir ./configs --config-name my_config +mode=predict

# Mask building
pytorch-smt --config-dir ./configs --config-name my_config +mode=build-mask

# Configuration validation
pytorch-smt --config-dir ./configs --config-name my_config +mode=validate-config
```

## Supported Modes

| Mode | Description | Use Case |
|------|-------------|----------|
| `train` | Train a model | Model training with PyTorch Lightning |
| `predict` | Run inference | Batch prediction on images |
| `predict-from-batch` | Batch prediction | Efficient batch processing |
| `validate-config` | Config validation | Debug configuration files |
| `build-mask` | Build masks | Generate training masks from vectors |
| `evaluate-experiments` | Run evaluation pipeline | Compare model outputs against ground truth |
| `convert-dataset` | Convert dataset format | Prepare data for Polygon-RNN models |

## Configuration Structure

The main function expects a Hydra configuration with the following structure:

```yaml
# Operation mode
mode: train  # or predict, build-mask, etc.

# Model configuration
model:
  _target_: segmentation_models_pytorch.Unet
  # ... model parameters

# Training configuration (for train mode)
pl_trainer:
  max_epochs: 100
  gpus: 1

# Prediction configuration (for predict mode)
checkpoint_path: /path/to/model.ckpt
inference_threshold: 0.5
```

## Examples

### Programmatic Usage

```python
from pytorch_segmentation_models_trainer.main import main
from omegaconf import DictConfig

# Create configuration
config = DictConfig({
    "mode": "train",
    "model": {
        "_target_": "segmentation_models_pytorch.Unet",
        "encoder_name": "resnet34",
        "classes": 1
    },
    "pl_trainer": {
        "max_epochs": 10,
        "gpus": 0
    }
    # ... other config
})

# Run training
main(config)
```

### Custom Entry Point

```python
import hydra
from omegaconf import DictConfig
from pytorch_segmentation_models_trainer.main import main

@hydra.main(config_path="configs", config_name="my_config")
def my_main(cfg: DictConfig):
    # Custom preprocessing
    cfg.custom_param = "my_value"

    # Run main function
    return main(cfg)

if __name__ == "__main__":
    my_main()
```

## Logging

Configure logging in your config:

```yaml
hydra:
  job:
    chdir: true
  run:
    dir: ./outputs/${now:%Y-%m-%d}/${now:%H-%M-%S}

logging:
  level: INFO
  format: '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
```
