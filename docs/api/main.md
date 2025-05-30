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
| `predict-mod-polymapper-from-batch` | Specialized prediction | ModPolymapper model inference |
| `validate-config` | Config validation | Debug configuration files |
| `build-mask` | Build masks | Generate training masks from vectors |
| `convert-dataset` | Dataset conversion | Convert between dataset formats |

## API Reference

::: pytorch_segmentation_models_trainer.main
    options:
      show_root_heading: true
      show_source: true
      heading_level: 3

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

# Other mode-specific configurations...
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

## Error Handling

The main function includes error handling for common issues:

- **Configuration errors**: Missing or invalid configuration parameters
- **Model loading errors**: Issues with model instantiation or checkpoint loading  
- **Data loading errors**: Problems with dataset access or format
- **CUDA/GPU errors**: Device availability and memory issues

## Logging

The main module sets up logging configuration:

- **Hydra logs**: Configuration resolution and parameter overrides
- **PyTorch Lightning logs**: Training progress and metrics
- **Application logs**: Custom logging from the library components

Configure logging levels in your config:

```yaml
hydra:
  job:
    chdir: true
  run:
    dir: ./outputs/${now:%Y-%m-%d}/${now:%H-%M-%S}

# Custom logging configuration  
logging:
  level: INFO
  format: '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
```

## Related Modules

- [`train`](train.md) - Training functionality
- [`predict`](predict.md) - Prediction functionality  
- [`config_utils`](../api/config_utils.md) - Configuration utilities
- [`build_mask`](../api/build_mask.md) - Mask building functionality