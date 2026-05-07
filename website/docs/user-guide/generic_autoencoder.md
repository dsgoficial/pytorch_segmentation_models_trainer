---
sidebar_position: 10
title: Generic Autoencoder
---

# Generic Autoencoder

The `GenericAutoencoder` is a flexible architecture designed for image reconstruction and self-supervised learning tasks. It allows combining encoders from **Segmentation Models PyTorch (SMP)** or **HuggingFace Transformers** with a reconstruction decoder.

## Key Features

- **Unified Encoder API**: Use any SMP-supported backbone (including `timm` models) or native HuggingFace models.
- **Automatic Reshaping**: Handles the conversion of 1D visual tokens (from ViTs) into 2D spatial feature maps.
- **Reconstruction Trainer**: Dedicated `AutoencoderModel` (LightningModule) for managing MSE/L1 reconstruction loss.
- **Simplified Dataset**: `AutoencoderDataset` designed for training without masks.

## Configuration

To use the Generic Autoencoder, you need to configure three main components in your Hydra YAML:

### 1. The Dataset
Use `AutoencoderDataset`, which requires only a CSV with image paths.

```yaml
train_dataset:
  _target_: pytorch_segmentation_models_trainer.dataset_loader.dataset.AutoencoderDataset
  input_csv_path: path/to/images.csv
  root_dir: /data
  augmentation_list:
    - _target_: albumentations.Resize
      height: 256
      width: 256
    - _target_: albumentations.Normalize
    - _target_: albumentations.pytorch.ToTensorV2
```

### 2. The Model
Configure `GenericAutoencoder` with your desired backbone.

```yaml
model:
  _target_: pytorch_segmentation_models_trainer.custom_models.generic_autoencoder.GenericAutoencoder
  encoder_name: mit_b2  # Or any SMP/HF name
  use_huggingface: false # Set true for pure HF models
  in_channels: 3
  latent_dim: 128        # Optional bottleneck dimension
```

### 3. The Trainer
Use `AutoencoderModel` to enable the reconstruction training loop.

```yaml
pl_trainer_model:
  _target_: pytorch_segmentation_models_trainer.model_loader.autoencoder_model.AutoencoderModel

loss:
  _target_: torch.nn.MSELoss
```

## Usage with HuggingFace

If you want to use a model directly from HuggingFace Hub that is not yet mapped in SMP:

```yaml
model:
  _target_: pytorch_segmentation_models_trainer.custom_models.generic_autoencoder.GenericAutoencoder
  encoder_name: facebook/vit-mae-base
  use_huggingface: true
  in_channels: 3
```

The model will automatically attempt to reshape the `last_hidden_state` into a 2D spatial map based on the number of tokens.
