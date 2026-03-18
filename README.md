# pytorch_segmentation_models_trainer


[![Torch](https://img.shields.io/badge/-PyTorch-red?logo=pytorch&labelColor=gray)](https://pytorch.org/get-started/locally/)
[![Pytorch Lightning](https://img.shields.io/badge/code-Lightning-blueviolet?logo=pytorchlightning&labelColor=gray)](https://pytorchlightning.ai/)
[![Hydra](https://img.shields.io/badge/conf-hydra-blue)](https://hydra.cc/)
[![Segmentation Models](https://img.shields.io/badge/models-segmentation_models_pytorch-yellow)](https://github.com/qubvel/segmentation_models.pytorch)
[![Python application](https://github.com/phborba/pytorch_segmentation_models_trainer/actions/workflows/python-app.yml/badge.svg)](https://github.com/phborba/pytorch_segmentation_models_trainer/actions/workflows/python-app.yml)
[![Upload Python Package](https://github.com/phborba/pytorch_segmentation_models_trainer/actions/workflows/python-publish.yml/badge.svg)](https://github.com/phborba/pytorch_segmentation_models_trainer/actions/workflows/python-publish.yml)
[![Publish Docker image](https://github.com/phborba/pytorch_segmentation_models_trainer/actions/workflows/docker-publish.yml/badge.svg)](https://github.com/phborba/pytorch_segmentation_models_trainer/actions/workflows/docker-publish.yml)
[![pre-commit.ci status](https://results.pre-commit.ci/badge/github/phborba/pytorch_segmentation_models_trainer/main.svg)](https://results.pre-commit.ci/latest/github/phborba/pytorch_segmentation_models_trainer/main)
[![PyPI package](https://img.shields.io/pypi/v/pytorch-segmentation-models-trainer?logo=pypi&color=green)](https://pypi.org/project/pytorch-segmentation-models-trainer/)
[![codecov](https://codecov.io/gh/phborba/pytorch_segmentation_models_trainer/branch/main/graph/badge.svg?token=PRJL5GVOL2)](https://codecov.io/gh/phborba/pytorch_segmentation_models_trainer)
[![CodeQL](https://github.com/phborba/pytorch_segmentation_models_trainer/actions/workflows/codeql-analysis.yml/badge.svg)](https://github.com/phborba/pytorch_segmentation_models_trainer/actions/workflows/codeql-analysis.yml)
[![maintainer](https://img.shields.io/badge/maintainer-phborba-blue.svg)](https://github.com/phborba)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.4573996.svg)](https://doi.org/10.5281/zenodo.4573996)


A comprehensive PyTorch + PyTorch Lightning framework for training semantic segmentation models on satellite and aerial imagery, with Hydra configuration management and extensive support for multispectral data.

## Config Builder (Web Interface)

**[Open Config Builder](https://dsgoficial.github.io/pytorch_segmentation_models_trainer/)**

A visual web interface hosted on GitHub Pages for building YAML configuration files without editing text by hand. Supports the **Training** and **Predict** workflows.

### What it does

- **Training tab**: configure model architecture and encoder, normalization parameters, class definitions, hyperparameters, loss function, optimizer, PyTorch Lightning trainer, metrics, callbacks, and train/val datasets (including data augmentation pipeline).
- **Predict tab**: configure checkpoint path, device, hyperparameters, PL trainer, model, inference processor (sliding window shape, optional normalization), image reader (folder, extension, recursive), and export strategy.
- **Live YAML preview**: the generated YAML is shown side-by-side and updates in real time as you fill the form.
- **Import from YAML**: paste an existing config file to populate the form fields automatically.
- **Searchable dropdowns**: all selectors (architecture, encoder, loss, optimizer, metrics, augmentations, etc.) are filterable comboboxes.

### How the schema stays up to date

A Python script (`scripts/generate_schema.py`) introspects the installed versions of `segmentation_models_pytorch`, `albumentations`, `torchmetrics`, and `torch` at build time, writing `web/src/assets/schema.json`. The GitHub Actions workflow ([`.github/workflows/deploy-config-builder.yml`](.github/workflows/deploy-config-builder.yml)) runs on every push to `main` (when `web/**` or the schema script changes), on manual dispatch, and on a weekly schedule to pick up library updates automatically.

---

## Features

- **Multiple Architectures**: UNet, DeepLabV3Plus, FPN, PSPNet with various encoders (ResNet34/101/152, EfficientNet, etc.)
- **Multispectral Support**: Native handling of 3, 4, 6, and 12-band satellite imagery
- **Transfer Learning**: Automatic weight adaptation from ImageNet pretrained models for multispectral data
- **Flexible Loss Functions**: Compound loss system with dynamic weight scheduling, supporting BCE, Dice, Focal, and custom losses
- **Advanced Inference**: Sliding window inference with configurable overlap for large imagery processing
- **Comprehensive Evaluation**: Multi-experiment evaluation pipeline with spatial alignment and parallel processing
- **Hydra Configuration**: Full configuration composition and management with YAML
- **Geospatial Tools**: Built-in support for GeoTIFF, coordinate systems, and PostGIS integration

## Installation

### From Source

```bash
# Clone the repository
git clone https://github.com/dsgoficial/pytorch_segmentation_models_trainer.git
cd pytorch_segmentation_models_trainer

# Install in editable mode
pip install -e .
```

### Using Docker

```bash
docker pull phborba/pytorch_segmentation_models_trainer:latest
```

### Using pip

```bash
pip install pytorch-segmentation-models-trainer
```

### Dependencies

Core dependencies include:
- PyTorch >= 2.0
- PyTorch Lightning >= 2.0
- Hydra >= 1.3
- segmentation_models_pytorch
- rasterio (for geospatial data)
- albumentations (for augmentations)
- torchmetrics

## Quick Start

The framework provides a CLI tool (`pytorch-smt`) and supports multiple modes:

```bash
# Training
pytorch-smt --config-dir /path/to/configs --config-name train +mode=train

# Inference
pytorch-smt --config-dir /path/to/configs --config-name predict +mode=predict

# Evaluation
python -m pytorch_segmentation_models_trainer.evaluate_experiments \
    --config-dir configs/evaluation --config-name pipeline_config
```

## Configuration Examples

### 1. Basic Training Configuration

```yaml
# configs/train_unet_resnet34.yaml

# Model Architecture
pl_model:
  _target_: pytorch_segmentation_models_trainer.model_loader.model.Model

backbone:
  name: resnet34
  input_width: 512
  input_height: 512

model:
  _target_: segmentation_models_pytorch.Unet
  encoder_name: resnet34
  encoder_weights: imagenet
  in_channels: 3
  classes: 6

# Hyperparameters
hyperparameters:
  model_name: unet_resnet34
  batch_size: 16
  epochs: 100
  max_lr: 0.001
  classes: 6

# Optimizer
optimizer:
  - _target_: torch.optim.AdamW
    lr: ${hyperparameters.max_lr}
    weight_decay: 0.0001

# Learning Rate Scheduler
scheduler_list:
  - scheduler:
      _target_: torch.optim.lr_scheduler.OneCycleLR
      max_lr: ${hyperparameters.max_lr}
      epochs: ${hyperparameters.epochs}
      steps_per_epoch: 1000  # Auto-computed from dataset
    interval: step
    frequency: 1

# Loss Function
loss_params:
  compound_loss:
    losses:
      - _target_: pytorch_segmentation_models_trainer.custom_losses.seg_loss.SegLoss
        bce_coef: 0.8
        dice_coef: 0.2
        weight: 1.0

# Dataset
train_dataset:
  _target_: pytorch_segmentation_models_trainer.dataset_loader.dataset.SegmentationDataset
  input_csv_path: /data/train.csv
  root_dir: /data
  augmentation_list:
    - _target_: albumentations.HorizontalFlip
      p: 0.5
    - _target_: albumentations.VerticalFlip
      p: 0.5
    - _target_: albumentations.RandomRotate90
      p: 0.5
  data_loader:
    shuffle: true
    num_workers: 8
    batch_size: ${hyperparameters.batch_size}
    pin_memory: true

val_dataset:
  _target_: pytorch_segmentation_models_trainer.dataset_loader.dataset.SegmentationDataset
  input_csv_path: /data/val.csv
  root_dir: /data
  data_loader:
    shuffle: false
    num_workers: 8
    batch_size: ${hyperparameters.batch_size}

# Trainer Configuration
pl_trainer:
  max_epochs: ${hyperparameters.epochs}
  accelerator: gpu
  devices: -1  # Use all available GPUs
  precision: "16-mixed"  # Mixed precision training
  default_root_dir: /experiments/${backbone.name}_${hyperparameters.model_name}
  
# Metrics
metrics:
  - _target_: torchmetrics.JaccardIndex
    task: multiclass
    num_classes: ${hyperparameters.classes}
  - _target_: torchmetrics.F1Score
    task: multiclass
    num_classes: ${hyperparameters.classes}
    average: macro

# Callbacks
callbacks:
  - _target_: pytorch_lightning.callbacks.ModelCheckpoint
    monitor: val/JaccardIndex
    mode: max
    save_top_k: 3
    filename: "{epoch:02d}-{val/JaccardIndex:.4f}"
  - _target_: pytorch_lightning.callbacks.EarlyStopping
    monitor: val/JaccardIndex
    patience: 20
    mode: max
  - _target_: pytorch_lightning.callbacks.LearningRateMonitor
    logging_interval: step
```

### 2. Multispectral Training (12-band Imagery)

```yaml
# configs/train_multispectral_12band.yaml

backbone:
  name: resnet101
  input_width: 512
  input_height: 512

model:
  _target_: segmentation_models_pytorch.DeepLabV3Plus
  encoder_name: resnet101
  encoder_weights: imagenet
  in_channels: 12  # 12-band multispectral
  classes: 7

# Weight adaptation strategy for multispectral
# The framework automatically adapts ImageNet weights
# Options: "mean", "random", "copy_first"
weight_adaptation_strategy: mean  # Recommended for multispectral

hyperparameters:
  model_name: deeplabv3plus_resnet101_12band
  batch_size: 8  # Smaller batch for 12 bands
  epochs: 150
  max_lr: 0.0005
  classes: 7

# Multispectral augmentations
train_dataset:
  input_csv_path: /data/multispectral_train.csv
  root_dir: /data
  augmentation_list:
    - _target_: albumentations.HorizontalFlip
      p: 0.5
    - _target_: albumentations.VerticalFlip
      p: 0.5
    - _target_: albumentations.RandomRotate90
      p: 0.5
    - _target_: albumentations.RandomBrightnessContrast
      brightness_limit: 0.2
      contrast_limit: 0.2
      p: 0.5
```

### 3. Compound Loss Configuration

```yaml
# configs/loss/compound_loss_example.yaml

loss_params:
  compound_loss:
    losses:
      # Segmentation Loss
      - _target_: pytorch_segmentation_models_trainer.custom_losses.seg_loss.SegLoss
        bce_coef: 0.7
        dice_coef: 0.3
        weight: 10.0
        name: seg_loss
      
      # Boundary Loss (optional)
      - _target_: pytorch_segmentation_models_trainer.custom_losses.boundary_loss.BoundaryLoss
        weight: 1.0
        name: boundary_loss
    
    # Dynamic weight scheduling
    weight_schedules:
      seg_loss:
        type: constant
        value: 10.0
      boundary_loss:
        type: epoch_threshold
        epoch_thresholds: [0, 20, 50]
        values: [0.0, 1.0, 2.0]
    
    # Normalization
    normalize_losses: true
    normalization_params:
      min_samples: 10
      max_samples: 1000
```

### 4. Inference Configuration

```yaml
# configs/predict_sliding_window.yaml

# Checkpoint
checkpoint_path: /experiments/best_model.ckpt
device: cuda:0

# Model config (inherited from training)
pl_model:
  _target_: pytorch_segmentation_models_trainer.model_loader.model.Model

hyperparameters:
  batch_size: 16
  classes: 6

# Image reader
inference_image_reader:
  _target_: pytorch_segmentation_models_trainer.tools.inference.inference_image_reader.InferenceImageReader
  input_folder: /data/test_images
  image_pattern: "*.tif"
  output_folder: /data/predictions

# Inference processor
inference_processor:
  _target_: pytorch_segmentation_models_trainer.tools.inference.inference_processors.MultiClassInferenceProcessor
  num_classes: 6
  
  # Sliding window parameters
  model_input_shape: [512, 512]
  step_shape: [384, 384]  # 25% overlap (512 - 384 = 128)
  
  # Export strategy
  export_strategy:
    _target_: pytorch_segmentation_models_trainer.tools.inference.export_strategies.ExportToGeoTiff
    compress: lzw
    tiled: true
    
  # Normalization (must match training)
  normalize_mean: [0.485, 0.456, 0.406]
  normalize_std: [0.229, 0.224, 0.225]

# Inference parameters
inference_threshold: 0.5
save_inference: true
```

### 5. Evaluation Pipeline Configuration

```yaml
# configs/evaluation/pipeline_config.yaml

# Experiments to evaluate
experiments:
  - name: unet_resnet34_3band
    predict_config: configs/predict_unet_r34.yaml
    checkpoint_path: /experiments/unet_r34/best.ckpt
    output_folder: /evaluations/unet_r34_predictions
  
  - name: deeplabv3_resnet101_12band
    predict_config: configs/predict_deeplabv3_r101.yaml
    checkpoint_path: /experiments/deeplabv3_r101/best.ckpt
    output_folder: /evaluations/deeplabv3_predictions

# Evaluation dataset
evaluation_dataset:
  # Option 1: Use existing CSV
  input_csv_path: /data/test.csv
  
  # Option 2: Build CSV from folders
  build_csv_from_folders:
    enabled: true
    images_folder: /data/test/images
    masks_folder: /data/test/masks
    image_pattern: "*.tif"
    mask_pattern: "*.tif"
    output_csv_path: /data/test_dataset.csv

# Metrics to compute
metrics:
  num_classes: 6
  segmentation_metrics:
    - _target_: torchmetrics.JaccardIndex
      task: multiclass
      num_classes: 6
      average: macro
    - _target_: torchmetrics.F1Score
      task: multiclass
      num_classes: 6
      average: macro
    - _target_: torchmetrics.Accuracy
      task: multiclass
      num_classes: 6
      average: macro

# Output configuration
output:
  base_dir: /evaluations/results
  structure:
    experiments_folder: experiments
    comparisons_folder: comparisons
  files:
    per_image_metrics_pattern: "{experiment_name}_per_image_metrics.csv"
    confusion_matrix_data_pattern: "{experiment_name}_confusion_matrix.npy"

# Visualization
visualization:
  enabled: true
  plot_confusion_matrices: true
  plot_comparison_charts: true
  max_samples_to_visualize: 10

# Pipeline options
pipeline_options:
  skip_existing_predictions: false
  skip_existing_metrics: false
  
  # Parallel inference
  parallel_inference:
    enabled: true
    max_workers: 4
    sequential_experiments: true  # Process experiments sequentially, parallelize within
```

### 6. CSV Dataset Format

The framework expects CSV files with the following format:

```csv
image,mask
/data/images/tile_001.tif,/data/masks/tile_001.tif
/data/images/tile_002.tif,/data/masks/tile_002.tif
```

You can also build CSVs automatically:

```python
from pytorch_segmentation_models_trainer.tools.inference.inference_csv_builder import build_csv_from_folders

csv_path = build_csv_from_folders(
    images_folder="/data/images",
    masks_folder="/data/masks",
    image_pattern="*.tif",
    mask_pattern="*.tif",
    output_csv_path="/data/dataset.csv"
)
```

## Supported Architectures

### Encoders
- ResNet (34, 50, 101, 152)
- ResNeXt
- EfficientNet (B0-B7)
- DenseNet (121, 161, 169, 201)
- MobileNet
- VGG (11, 13, 16, 19)
- And more via `segmentation_models_pytorch`

### Decoders
- **UNet**: Classic U-Net architecture
- **UNet++**: Nested U-Net with dense skip connections
- **DeepLabV3+**: Atrous Spatial Pyramid Pooling
- **FPN**: Feature Pyramid Network
- **PSPNet**: Pyramid Scene Parsing Network
- **PAN**: Path Aggregation Network
- **LinkNet**: Efficient architecture for real-time segmentation
- **MANet**: Multi-scale Attention Network

## Dataset Preparation

### Creating Masks from Vector Data

```bash
# Using the mask builder tool
python -m pytorch_segmentation_models_trainer.tools.mask_building.mask_builder \
    --config-dir configs/mask_building \
    --config-name build_masks
```

Example mask building configuration:

```yaml
# configs/mask_building/build_masks.yaml
geo_df:
  _target_: pytorch_segmentation_models_trainer.tools.data_handlers.vector_reader.FileGeoDF
  file_name: /data/vectors/buildings.geojson

root_dir: /data
image_root_dir: images
image_extension: tif

# Mask types to build
build_polygon_mask: true
polygon_mask_folder_name: polygon_masks

build_boundary_mask: true
boundary_mask_folder_name: boundary_masks

build_distance_mask: false
build_size_mask: false

# Options
replicate_image_folder_structure: true
min_polygon_area: 50.0
mask_output_extension: tif
```

## Training

### Single GPU Training

```bash
pytorch-smt --config-dir configs --config-name train_unet +mode=train
```

### Multi-GPU Training (Distributed Data Parallel)

```bash
# Automatic - uses all available GPUs
pytorch-smt --config-dir configs --config-name train_unet +mode=train \
    pl_trainer.devices=-1

# Specific GPUs
pytorch-smt --config-dir configs --config-name train_unet +mode=train \
    pl_trainer.devices=[0,1,2,3]
```

### Mixed Precision Training

```bash
pytorch-smt --config-dir configs --config-name train_unet +mode=train \
    pl_trainer.precision="16-mixed"
```

### Resume from Checkpoint

```bash
pytorch-smt --config-dir configs --config-name train_unet +mode=train \
    hyperparameters.resume_from_checkpoint=/path/to/checkpoint.ckpt
```

### Override Configuration Parameters

```bash
# Override multiple parameters
pytorch-smt --config-dir configs --config-name train_unet +mode=train \
    hyperparameters.batch_size=32 \
    hyperparameters.max_lr=0.001 \
    hyperparameters.epochs=200
```

## Inference

### Single Image Inference

```bash
pytorch-smt --config-dir configs --config-name predict +mode=predict
```

### Batch Inference with Sliding Window

For large images that don't fit in memory, use sliding window inference:

```yaml
inference_processor:
  model_input_shape: [512, 512]  # Model's expected input size
  step_shape: [384, 384]  # Overlap: 512 - 384 = 128 pixels (25%)
```

Performance considerations:
- **0% overlap** (`step_shape = model_input_shape`): Fastest, may have artifacts at tile boundaries
- **25% overlap** (`step_shape = [384, 384]` for 512×512): Good balance
- **50% overlap** (`step_shape = [256, 256]` for 512×512): Higher quality, ~4× slower

### Inference with Normalization

Ensure normalization matches your training configuration:

```yaml
inference_processor:
  normalize_mean: [0.485, 0.456, 0.406]  # ImageNet stats
  normalize_std: [0.229, 0.224, 0.225]
```

For custom normalization, compute from your training data:

```python
import numpy as np
from tqdm import tqdm
import rasterio

def compute_normalization_stats(image_paths, bands=[0, 1, 2]):
    """Compute mean and std for dataset normalization."""
    means = []
    stds = []
    
    for img_path in tqdm(image_paths):
        with rasterio.open(img_path) as src:
            img = src.read(bands)
            means.append(img.mean(axis=(1, 2)))
            stds.append(img.std(axis=(1, 2)))
    
    mean = np.array(means).mean(axis=0)
    std = np.array(stds).mean(axis=0)
    
    return mean.tolist(), std.tolist()
```

## Evaluation

### Comprehensive Evaluation Pipeline

The evaluation pipeline supports:
- Multiple experiments comparison
- Automatic CSV generation from image folders
- Spatial alignment of predictions and ground truth
- Parallel processing with configurable workers
- Per-image and aggregated metrics
- Confusion matrix computation
- Visualization generation

```bash
python -m pytorch_segmentation_models_trainer.evaluate_experiments \
    --config-dir configs/evaluation \
    --config-name pipeline_config
```

### Metrics

Supported metrics via `torchmetrics`:
- Intersection over Union (IoU / Jaccard Index)
- F1 Score
- Accuracy
- Precision & Recall
- Confusion Matrix
- Per-class metrics

### Direct Folder Evaluation

For quick evaluation when you already have predictions:

```python
from pytorch_segmentation_models_trainer.tools.evaluation.direct_folder_evaluator import DirectFolderEvaluator

evaluator = DirectFolderEvaluator(
    pred_folder="/path/to/predictions",
    gt_folder="/path/to/ground_truth",
    num_classes=6
)

# Create evaluation CSV
df = evaluator.create_evaluation_csv("/output/eval.csv")

# Compute metrics
results = evaluator.evaluate(df)
```

## Advanced Features

### Custom Loss Functions

Create custom loss functions by extending `BaseLoss`:

```python
from pytorch_segmentation_models_trainer.custom_losses.base_loss import BaseLoss
import torch
import torch.nn as nn

class CustomLoss(BaseLoss):
    def __init__(self, weight=1.0, **kwargs):
        super().__init__(weight=weight, **kwargs)
        self.criterion = nn.CrossEntropyLoss()
    
    def forward(self, pred, batch):
        return self.criterion(pred['seg'], batch['mask'])
```

### GPU Augmentations

Apply augmentations on GPU for faster training:

```yaml
train_dataset:
  gpu_augmentation_list:
    - _target_: kornia.augmentation.RandomHorizontalFlip
      p: 0.5
    - _target_: kornia.augmentation.RandomVerticalFlip
      p: 0.5
    - _target_: kornia.augmentation.ColorJitter
      brightness: 0.2
      contrast: 0.2
      p: 0.5
```

### Custom Callbacks

```python
from pytorch_lightning.callbacks import Callback

class CustomCallback(Callback):
    def on_epoch_end(self, trainer, pl_module):
        # Your custom logic here
        pass
```

Add to config:

```yaml
callbacks:
  - _target_: your_module.CustomCallback
    param1: value1
```

### Visualization Callbacks

Built-in visualization during training:

```yaml
callbacks:
  - _target_: pytorch_segmentation_models_trainer.custom_callbacks.image_callbacks.SegmentationVisualizationCallback
    n_samples: 4
    output_path: /experiments/visualizations
    normalized_input: true
    norm_params:
      mean: [0.485, 0.456, 0.406]
      std: [0.229, 0.224, 0.225]
    log_every_k_epochs: 5
    colormap: tab10
    num_classes: 6
    class_names: ["Background", "Building", "Road", "Tree", "Water", "Car"]
```

## Project Structure

```
pytorch_segmentation_models_trainer/
├── pytorch_segmentation_models_trainer/
│   ├── model_loader/          # Model and Lightning module wrappers
│   ├── dataset_loader/        # Dataset classes
│   ├── custom_losses/         # Loss functions
│   ├── custom_callbacks/      # Training callbacks
│   ├── tools/
│   │   ├── inference/         # Inference processors
│   │   ├── evaluation/        # Evaluation pipeline
│   │   ├── mask_building/     # Mask generation from vectors
│   │   └── data_handlers/     # Raster and vector I/O
│   ├── utils/                 # Utility functions
│   ├── train.py              # Training script
│   ├── predict.py            # Inference script
│   ├── main.py               # CLI entry point
│   └── evaluate_experiments.py  # Evaluation pipeline
├── configs/                   # Configuration files
│   ├── train/
│   ├── predict/
│   └── evaluation/
├── config_definitions/        # Typed config dataclasses
├── tests/                     # Unit tests
└── setup.py
```


## Troubleshooting

### CUDA Out of Memory
- Reduce `batch_size`
- Enable `gradient_checkpointing` in model config
- Use mixed precision: `pl_trainer.precision="16-mixed"`
- Reduce `num_workers` in dataloader

### Slow Training
- Increase `num_workers` in dataloader
- Enable mixed precision
- Use GPU augmentations instead of CPU
- Check I/O bottlenecks with profiling

### Poor Convergence
- Adjust learning rate
- Increase model capacity
- Add more augmentations
- Check data quality and class balance

### Inference Memory Issues
- Reduce `batch_size` in inference config
- Use smaller sliding window `model_input_shape`
- Process images one at a time

## Citation

If you use this framework in your research, please cite:

```bibtex
@software{philipe_borba_2025_17581320,
  author       = {Philipe Borba},
  title        = {dsgoficial/pytorch\_segmentation\_models\_trainer:
                   Version 1.0.0
                  },
  month        = nov,
  year         = 2025,
  publisher    = {Zenodo},
  version      = {v.1.0.0},
  doi          = {10.5281/zenodo.17581320},
  url          = {https://doi.org/10.5281/zenodo.17581320},
  swhid        = {swh:1:dir:6279d2f90c1b1bde6f7704758ecdfce0a5d3eb14
                   ;origin=https://doi.org/10.5281/zenodo.4573996;vis
                   it=swh:1:snp:68534bb09abd3eadef762f11e7f24038025b4
                   df5;anchor=swh:1:rel:7a642f966fff89a28215316b2f5e2
                   716e4ec5bd4;path=dsgoficial-
                   pytorch\_segmentation\_models\_trainer-e94787b
                  },
}
```

## Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Submit a pull request

## License

This project is licensed under the GNU General Public License v2.0 or later.
