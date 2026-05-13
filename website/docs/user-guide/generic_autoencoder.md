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
- **Variational Training**: `GenericVariationalAutoencoder`, `VariationalAutoencoderModel`, and `VariationalAutoencoderLoss` implement reconstruction plus KL regularization.
- **Simplified Dataset**: `AutoencoderDataset` designed for training without masks.
- **Folder Random Crops**: `AutoencoderRandomCropDataset` scans image folders and samples crops on-the-fly from large rasters.

## Configuration

To use the Generic Autoencoder, you need to configure three main components in your Hydra YAML:

### 1. The Dataset
Use `AutoencoderDataset`, which requires only a CSV with image paths.

```yaml
train_dataset:
  _target_: pytorch_segmentation_models_trainer.dataset_loader.image_dataset.AutoencoderDataset
  input_csv_path: path/to/images.csv
  root_dir: /data
  augmentation_list:
    - _target_: albumentations.Resize
      height: 256
      width: 256
    - _target_: albumentations.Normalize
    - _target_: albumentations.pytorch.ToTensorV2
```

For unlabeled image folders or large rasters, use `AutoencoderRandomCropDataset`.
It recursively discovers images, splits train/validation deterministically, and
reads only the sampled window for each item.

```yaml
train_dataset:
  _target_: pytorch_segmentation_models_trainer.dataset_loader.image_dataset.AutoencoderRandomCropDataset
  image_dir: /data/unlabeled_images
  split: train
  val_fraction: 0.2
  split_seed: 42
  image_extensions: [".tif", ".tiff", ".png", ".jpg", ".jpeg"]
  crop_size: [256, 256]
  samples_per_epoch: 20000
  augmentation_list:
    - _target_: albumentations.Normalize
    - _target_: albumentations.pytorch.ToTensorV2

val_dataset:
  _target_: pytorch_segmentation_models_trainer.dataset_loader.image_dataset.AutoencoderRandomCropDataset
  image_dir: /data/unlabeled_images
  split: val
  val_fraction: 0.2
  split_seed: 42
  crop_size: [256, 256]
  samples_per_epoch: 2000
  augmentation_list:
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

## Variational Autoencoder

Use the VAE path when the latent representation should be regularized against a
standard normal prior. The model returns reconstruction, `mu`, `logvar`, and the
sampled latent tensor `z`; the loss combines reconstruction with the analytic KL
term `KL(q(z|x) || N(0, I))`.

```yaml
pl_model:
  _target_: pytorch_segmentation_models_trainer.model_loader.variational_autoencoder_model.VariationalAutoencoderModel

model:
  _target_: pytorch_segmentation_models_trainer.custom_models.variational_autoencoder.GenericVariationalAutoencoder
  encoder_name: resnet18
  use_huggingface: false
  in_channels: 3
  latent_dim: 128
  pretrained: false

loss:
  _target_: pytorch_segmentation_models_trainer.custom_losses.autoencoder_losses.VariationalAutoencoderLoss
  reconstruction_loss: mse
  reconstruction_weight: 1.0
  beta: 1.0
```

`VariationalAutoencoderLoss` supports `mse`, `l1`, and `bce_with_logits` as the
reconstruction term. The `beta` parameter controls the strength of the KL
regularization. See
`conf/examples/generic_variational_autoencoder_random_crop_folder.yaml` for a
complete folder-based random-crop training config with train, validation, and
test datasets.

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

## Monitoring and Visualization

To monitor the reconstruction quality during training, you can use the `AutoencoderResultCallback`. This callback logs input and reconstructed images side-by-side to TensorBoard and saves them to the log directory.

```yaml
pl_trainer:
  callbacks:
    - _target_: pytorch_segmentation_models_trainer.custom_callbacks.image_callbacks.AutoencoderResultCallback
      n_samples: 8           # Number of samples to visualize
      log_every_k_epochs: 1  # How often to log visualizations
```
