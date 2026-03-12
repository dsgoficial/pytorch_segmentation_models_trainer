---
sidebar_position: 1
title: Introduction
---

# PyTorch Segmentation Models Trainer

[![Torch](https://img.shields.io/badge/-PyTorch-red?logo=pytorch&labelColor=gray)](https://pytorch.org/get-started/locally/)
[![Pytorch Lightning](https://img.shields.io/badge/code-Lightning-blueviolet?logo=pytorchlightning&labelColor=gray)](https://pytorchlightning.ai/)
[![Hydra](https://img.shields.io/badge/conf-hydra-blue)](https://hydra.cc/)
[![PyPI package](https://img.shields.io/pypi/v/pytorch-segmentation-models-trainer?logo=pypi&color=green)](https://pypi.org/project/pytorch-segmentation-models-trainer/)

A comprehensive framework for training semantic segmentation models using PyTorch, PyTorch Lightning, and Hydra configuration management.

## Key Features

- **Configuration-Driven**: Use YAML files to define your entire training pipeline
- **Multiple Model Types**: Support for semantic segmentation, object detection, instance segmentation, and specialized models
- **Advanced Polygonization**: Frame field models, active contours, and Polygon-RNN for precise boundary extraction
- **Easy Training & Inference**: Simple CLI commands for training and prediction
- **Flexible Data Loading**: Support for various dataset formats and augmentation pipelines
- **Built-in Visualization**: Tools for visualizing results and debugging
- **Docker Ready**: Pre-built containers with all dependencies

## Quick Example

Train a U-Net model with just a configuration file:

```yaml
model:
  _target_: segmentation_models_pytorch.Unet
  encoder_name: resnet34
  encoder_weights: imagenet
  in_channels: 3
  classes: 1

loss:
  _target_: segmentation_models_pytorch.utils.losses.DiceLoss

optimizer:
  _target_: torch.optim.AdamW
  lr: 0.001

train_dataset:
  _target_: pytorch_segmentation_models_trainer.dataset_loader.dataset.SegmentationDataset
  input_csv_path: /path/to/train.csv
  augmentation_list:
    - _target_: albumentations.RandomCrop
      height: 256
      width: 256
    - _target_: albumentations.Normalize
    - _target_: albumentations.pytorch.transforms.ToTensorV2
```

Then train with:
```bash
pytorch-smt --config-dir ./configs --config-name my_config +mode=train
```

## Supported Model Types

| Model Type | Description | Use Cases |
|------------|-------------|-----------|
| **Semantic Segmentation** | Standard U-Net, DeepLab, PSPNet, etc. | General image segmentation |
| **Frame Field Models** | Boundary-aware segmentation with crossfield | Building extraction, precise boundaries |
| **Object Detection** | FRCNN, RetinaNet, etc. | Object localization |
| **Instance Segmentation** | Mask R-CNN variants | Individual object instances |
| **Polygon RNN** | Sequential polygon vertex prediction | Precise polygon extraction |

## Polygonization Methods

Transform segmentation masks into precise vector polygons:

- **Active Skeletons**: Skeleton-based optimization
- **Active Contours**: Energy minimization approach
- **Simple Polygonization**: Fast contour extraction
- **Polygon RNN**: Neural polygon vertex prediction

## Getting Started

1. **[Installation](getting-started/installation)** - Set up the environment
2. **[Quick Start](getting-started/quickstart)** - Your first training job
3. **[Configuration](getting-started/configuration)** - Understanding config files
4. **[Examples](examples/basic-segmentation)** - Working examples

## Citation

If you use this library in your research, please cite:

```bibtex
@software{philipe_borba_2021_5115127,
  author       = {Philipe Borba},
  title        = {{phborba/pytorch\_segmentation\_models\_trainer:
                   Version 0.8.0}},
  month        = jul,
  year         = 2021,
  publisher    = {Zenodo},
  version      = {v0.8.0},
  doi          = {10.5281/zenodo.5115127},
  url          = {https://doi.org/10.5281/zenodo.5115127}
}
```
