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
pl_model:
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

`VariationalAutoencoderLoss` supports `mse`, `l1`, `smooth_l1`, `ms_ssim`,
`smooth_l1_ms_ssim`, and `bce_with_logits` as the reconstruction term. The
`beta` parameter controls the strength of the KL regularization. See
`conf/examples/generic_variational_autoencoder_random_crop_folder.yaml` for a
complete folder-based random-crop training config with train, validation, and
test datasets.

### MS-SSIM and Smooth L1 Reconstruction

Use `ms_ssim` when structural similarity matters more than pixel-wise error.
Use `smooth_l1_ms_ssim` to combine local robust reconstruction with multi-scale
structure. MS-SSIM must receive image-intensity tensors, not standardized
`Normalize(mean, std)` tensors. For datasets converted with
`albumentations.ToFloat(max_value=255.0)`, keep `ms_ssim_data_range: 1.0`. For
raw uint8-scale tensors, use `ms_ssim_data_range: 255.0`. If your dataset uses
`albumentations.Normalize`, set `ms_ssim_input_is_normalized: true` and pass the
same mean/std values used by the augmentation.

```yaml
loss:
  _target_: pytorch_segmentation_models_trainer.custom_losses.autoencoder_losses.VariationalAutoencoderLoss
  reconstruction_loss: smooth_l1_ms_ssim
  reconstruction_weight: 1.0
  beta: 1.0
  smooth_l1_beta: 0.1
  smooth_l1_weight: 0.8
  ms_ssim_weight: 0.2
  ms_ssim_data_range: 1.0
  ms_ssim_alpha: 1.0
  ms_ssim_compensation: 1.0
  ms_ssim_input_is_normalized: true
  ms_ssim_mean: [0.485, 0.456, 0.406]
  ms_ssim_std: [0.229, 0.224, 0.225]
```

The VAE LightningModule logs the combined `reconstruction_loss` plus component
terms: `smooth_l1_loss`, `ms_ssim_loss`, `weighted_smooth_l1_loss`, and
`weighted_ms_ssim_loss`. By default `ms_ssim_alpha: 1.0` disables Kornia's
internal L1 blend, so `smooth_l1_ms_ssim` means Smooth L1 plus pure MS-SSIM.
Denormalization is applied only to the MS-SSIM branch; Smooth L1 remains in the
training tensor space, so it still matches what the decoder is optimizing.
See
`conf/examples/generic_variational_autoencoder_ms_ssim.yaml` for a complete
configuration.

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

## Decoder Upsampling Modes

Both `GenericAutoencoder` and `GenericVariationalAutoencoder` accept an `upsample_mode`
parameter that controls how each decoder stage doubles spatial resolution.

| Mode | Mechanism | Works in | Notes |
|---|---|---|---|
| `"bilinear"` | `F.interpolate` / `nn.Upsample` | `GenericDecoder`, `ProgressiveDecoder` | Default; no extra parameters |
| `"transposed_conv"` | Learnable `ConvTranspose2d` | `GenericDecoder`, `ProgressiveDecoder` | Avoids fixed interpolation kernel |
| `"pixel_shuffle"` | Sub-pixel conv (ESPCN) | `ProgressiveDecoder` only | Best quality-per-param for upsampling |

> **Note:** `"pixel_shuffle"` requires `use_progressive_decoder: true`. Using it with
> `GenericDecoder` raises a `ValueError` because single-shot channel expansion
> (`out_channels × scale_factor²`) is impractical for large `scale_factor` values.

### Example — ProgressiveDecoder with pixel shuffle

```yaml
model:
  _target_: pytorch_segmentation_models_trainer.custom_models.variational_autoencoder.GenericVariationalAutoencoder
  encoder_name: resnet18
  in_channels: 3
  latent_dim: 8
  encoder_depth: 3
  output_activation: sigmoid
  use_progressive_decoder: true
  upsample_mode: pixel_shuffle   # or "bilinear" / "transposed_conv"
```

### Example — GenericDecoder with transposed conv

```yaml
model:
  _target_: pytorch_segmentation_models_trainer.custom_models.generic_autoencoder.GenericAutoencoder
  encoder_name: resnet18
  in_channels: 3
  use_progressive_decoder: false
  upsample_mode: transposed_conv
```

See `conf/examples/autoencoder_decoder_upsample_modes.yaml` for a complete
training config comparing all three modes.

## Latent Clustering Metrics

`AutoencoderModel` and `VariationalAutoencoderModel` can log epoch-level
clustering diagnostics for the encoder latent space. The implementation keeps
the accumulated embeddings on the active torch device, clusters them with the
framework's PyTorch `MiniBatchKMeans`, and computes TorchMetrics clustering
scores without moving tensors through scikit-learn.

```yaml
latent_metrics:
  _target_: pytorch_segmentation_models_trainer.custom_metrics.autoencoder_latent_clustering.AutoencoderLatentClusteringMetrics
  n_clusters: 8
  max_samples: 2048
  kmeans_max_iter: 50
  kmeans_batch_size: 1024
  random_state: 42
  normalize: true
  latent_reduction: adaptive_avg_pool
  compute_silhouette: true
  compute_dunn: false
  label_key: null
```

The default metrics are logged at validation/test epoch end:

- `latent_calinski_harabasz`
- `latent_davies_bouldin`
- `latent_silhouette` when `compute_silhouette: true`
- `latent_dunn` when `compute_dunn: true`
- `latent_adjusted_rand` and `latent_normalized_mutual_info` when `label_key`
  points to a batch label tensor

For VAEs, `vae_latent: mu` is the default because the posterior mean is
deterministic. Set `vae_latent: z` to cluster sampled latents instead.

```yaml
latent_metrics:
  _target_: pytorch_segmentation_models_trainer.custom_metrics.autoencoder_latent_clustering.AutoencoderLatentClusteringMetrics
  n_clusters: 8
  vae_latent: mu
  max_samples: 2048
```

See `conf/examples/autoencoder_latent_clustering.yaml` for a complete
configuration.

## Monitoring and Visualization

To monitor the reconstruction quality during training, you can use the `AutoencoderResultCallback`. This callback logs input and reconstructed images side-by-side to TensorBoard and saves them to the log directory.

```yaml
pl_trainer:
  callbacks:
    - _target_: pytorch_segmentation_models_trainer.custom_callbacks.image_callbacks.AutoencoderResultCallback
      n_samples: 8           # Number of samples to visualize
      log_every_k_epochs: 1  # How often to log visualizations
```
