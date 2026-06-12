---
sidebar_position: 2
---

# Model Classes

This page is the API reference for all PyTorch Lightning model classes in `pytorch_segmentation_models_trainer`.

---

## `Model`

Base model class. All other model classes extend `Model`.

**Import path:** `pytorch_segmentation_models_trainer.model_loader.model.Model`

### Constructor

```python
Model(cfg, inference_mode=False)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `cfg` | `DictConfig` | required | Hydra configuration object |
| `inference_mode` | `bool` | `False` | When `True`, skips dataset and loss instantiation (for inference-only use) |

### Key Methods

| Method | Signature | Description |
|--------|-----------|-------------|
| `get_model` | `() -> nn.Module` | Instantiates the backbone from `cfg.model` via Hydra's `instantiate`. Optionally replaces activations if `cfg.replace_model_activation` is set |
| `get_loss_function` | `() -> Union[nn.Module, MultiLoss]` | Resolves the loss function using three dispatch paths (see below) |
| `training_step` | `(batch, batch_idx) -> Tensor` | Unpacks `batch` as `(images, masks)`, runs forward pass, computes loss (simple or compound), logs `loss/train` and individual component losses |
| `validation_step` | `(batch, batch_idx) -> Tensor` | Same as `training_step` but logs `loss/val`; also logs `val/` metrics if `cfg.metrics` is configured |
| `test_step` | `(batch, batch_idx) -> Tensor` | Runs after training via `trainer.test()`; logs `loss/test` and `test/` metrics. Applies `gpu_test_transform` if configured |
| `configure_optimizers` | `() -> Tuple[List, List]` | Builds optimizer from `cfg.optimizer`; builds scheduler list from `cfg.scheduler_list`. Automatically computes `steps_per_epoch` for `OneCycleLR` by reading the training CSV |
| `train_dataloader` | `() -> DataLoader` | Creates `DataLoader` for `train_ds` using `cfg.hyperparameters.batch_size` and `cfg.train_dataset.data_loader` settings |
| `val_dataloader` | `() -> Optional[DataLoader]` | Creates `DataLoader` for `val_ds`, or returns `None` if `val_dataset` is absent from the config |
| `test_dataloader` | `() -> Optional[DataLoader]` | Creates `DataLoader` for `test_ds`, or returns `None` if `test_dataset` is absent from the config |
| `forward` | `(x) -> Tensor` | Delegates to `self.model(x)` |
| `predict_step` | `(batch, batch_idx, dataloader_idx=0) -> Tensor` | Runs `self(batch)` for inference |
| `set_encoder_trainable` | `(trainable=False) -> None` | Freezes or unfreezes the `model.encoder` parameters |
| `_zero_init_extra_input_channels` | `(model, n_base=3) -> None` | Static. Zeroes `weight[:, n_base:, :, :]` in the first `Conv2d` whose `in_channels > n_base`. Call via `get_model()` by setting `zero_init_extra_input_channels: true` in the config |

### Configuration Keys

| Key | Required | Description |
|-----|----------|-------------|
| `model` | Yes | Hydra target for the backbone architecture |
| `loss` | Conditional | Simple loss (lowest priority dispatch path) |
| `loss_params.compound_loss` | Conditional | New YAML-based compound loss config (highest priority) |
| `loss_params.multi_loss` | Conditional | Legacy multi-loss config (medium priority) |
| `optimizer` | Yes | Hydra target for the optimizer |
| `scheduler_list` | No | List of scheduler configs; each entry has a `scheduler` key and Lightning interval/frequency keys |
| `hyperparameters.batch_size` | Yes | Batch size for train, val, and test dataloaders |
| `hyperparameters.epochs` | No | Used when auto-computing `steps_per_epoch` |
| `pl_trainer` | No | Passed to the Lightning `Trainer`; also checked for `devices` and `accumulate_grad_batches` |
| `callbacks` | No | List of Hydra-instantiated callbacks |
| `metrics` | No | List of `torchmetrics` metrics to compute; auto-prefixed `train/`, `val/`, or `test/` |
| `logger` | No | Logger configuration |
| `train_dataset` | Yes (unless `inference_mode`) | Dataset config including `input_csv_path` and `data_loader` sub-config |
| `val_dataset` | No | Dataset config for per-epoch validation during `fit`. When absent, `val_dataloader()` returns `None` and Lightning skips the validation loop |
| `test_dataset` | No | Dataset config for final held-out evaluation. When present, `trainer.test()` is called automatically after `fit`; metrics are logged with `test/` prefix |
| `replace_model_activation` | No | Dict with `old_activation` and `new_activation` for activation replacement |
| `zero_init_extra_input_channels` | No | When `true`, zeroes the extra input channels (channels `3:`) of the first `Conv2d` after instantiation. Required when extending a 3-channel ImageNet encoder with additional binary input channels (e.g. one-hot LULC bands) so that at t=0 the model behaves identically to the 3-channel baseline. See example below |

### Zero-initializing extra input channels

When using a pre-trained encoder with more than 3 input channels, `segmentation_models_pytorch` fills
the extra channels by cycling the 3 RGB filter weights (scaled by `3 / in_channels`). This means the
extra channels start with non-zero weights that were trained on RGB images, creating a mismatch when the
extra channels carry a different signal domain (e.g. binary one-hot LULC maps).

Setting `zero_init_extra_input_channels: true` zeroes those weights after instantiation, so the model
behaves exactly like the 3-channel RGB baseline at epoch 0 and learns the extra channels from scratch.

```yaml
# Extend a ResNet-50 encoder with 18 one-hot LULC channels (3 RGB + 18 LULC = 21).
# Channels 3-20 are zero-initialized so training starts from the RGB baseline.
zero_init_extra_input_channels: true  # must be a top-level key, NOT inside model:

model:
  _target_: segmentation_models_pytorch.Unet
  encoder_name: resnet50
  encoder_weights: imagenet
  in_channels: 21
  classes: 6
```

A full working example with datasets, metrics, and scheduler is available at
`conf/examples/multichannel_zero_init_segmentation.yaml`.

---

## `VariationalAutoencoderModel`

LightningModule for VAE reconstruction training. It expects the configured
domain model to return a `VariationalAutoencoderOutput` and the configured loss
to return either a scalar tensor or a dictionary containing `loss` plus optional
component losses.

**Import path:** `pytorch_segmentation_models_trainer.model_loader.variational_autoencoder_model.VariationalAutoencoderModel`

### Constructor

```python
VariationalAutoencoderModel(cfg, inference_mode=False)
```

### Logged Values

| Metric | Description |
|--------|-------------|
| `train/loss`, `val/loss`, `test/loss` | Total VAE objective |
| `train/reconstruction_loss`, `val/reconstruction_loss`, `test/reconstruction_loss` | Reconstruction term when returned by the loss |
| `train/kl_loss`, `val/kl_loss`, `test/kl_loss` | Analytic KL term when returned by the loss |

### YAML Config Snippet

```yaml
pl_model:
  _target_: pytorch_segmentation_models_trainer.model_loader.variational_autoencoder_model.VariationalAutoencoderModel

model:
  _target_: pytorch_segmentation_models_trainer.custom_models.variational_autoencoder.GenericVariationalAutoencoder
  encoder_name: resnet18
  in_channels: 3
  latent_dim: 128

loss:
  _target_: pytorch_segmentation_models_trainer.custom_losses.autoencoder_losses.VariationalAutoencoderLoss
  reconstruction_loss: mse
  beta: 1.0
```

---

## `FrameFieldSegmentationPLModel`

PyTorch Lightning model for frame field segmentation. Extends `Model` with frame-field-specific loss building, normalization, and output handling.

**Import path:** `pytorch_segmentation_models_trainer.model_loader.frame_field_model.FrameFieldSegmentationPLModel`

### Constructor

```python
FrameFieldSegmentationPLModel(cfg: DictConfig)
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `cfg` | `DictConfig` | Hydra configuration. Must include `backbone`, `compute_seg`, `compute_crossfield`, and `seg_params` in addition to the base `Model` keys |

Additional configuration keys beyond `Model`:

| Key | Description |
|-----|-------------|
| `backbone` | Hydra target for the frame field backbone (e.g., a `FrameFieldModel` wrapper) |
| `compute_seg` | Whether to compute segmentation output |
| `compute_crossfield` | Whether to compute cross-field output |
| `seg_params.compute_interior` | Whether to predict interior mask channel |
| `seg_params.compute_edge` | Whether to predict edge mask channel |
| `seg_params.compute_vertex` | Whether to predict vertex mask channel |

### Key Methods

| Method | Signature | Description |
|--------|-----------|-------------|
| `training_step` | `(batch, batch_idx) -> Tensor` | Runs forward pass, computes `MultiLoss`, logs `loss/train` and all component losses under `losses/train_{name}`; also logs `iou/train_IoU_{threshold}` |
| `validation_step` | `(batch, batch_idx) -> Tensor` | Same as training but logs `loss/val`, `losses/val_{name}`, and `iou/val_IoU_{threshold}` |
| `test_step` | `(batch, batch_idx) -> Tensor` | Runs after training via `trainer.test()`; logs `loss/test`, `losses/test_{name}`, and `iou/test_IoU_{threshold}` |
| `on_train_start` | `() -> None` | Triggers loss normalization computation via `_compute_loss_normalization()` |

Model output dict keys:

| Key | Shape | Description |
|-----|-------|-------------|
| `"seg"` | `(N, C_seg, H, W)` | Segmentation probabilities |
| `"crossfield"` | `(N, 4, H, W)` | Cross-field angles (c0 + c2) |

---

## `ObjectDetectionPLModel`

PyTorch Lightning model for object detection using Torchvision-style detection models (e.g., Faster R-CNN).

**Import path:** `pytorch_segmentation_models_trainer.model_loader.detection_model.ObjectDetectionPLModel`

### Constructor

```python
ObjectDetectionPLModel(cfg)
```

The loss function is managed internally by the detection model itself (Torchvision detection models return a `loss_dict` in training mode), so `get_loss_function()` returns `None`.

### Behavior

- `training_step`: calls `self.model(images, targets)`, sums the returned `loss_dict`, logs total and individual losses
- `validation_step`: computes loss in train mode, then switches to eval mode to compute box IoU; logs `loss/val` and `metrics/val_iou`
- `train_dataloader` / `val_dataloader`: uses the dataset's `collate_fn` (required for detection batches with variable-length box lists)

---

## `InstanceSegmentationPLModel`

Extends `ObjectDetectionPLModel` for instance segmentation (e.g., Mask R-CNN). No additional logic — the mask handling is done by the underlying Torchvision model.

**Import path:** `pytorch_segmentation_models_trainer.model_loader.detection_model.InstanceSegmentationPLModel`

### Constructor

```python
InstanceSegmentationPLModel(cfg)
```

Inherits all behavior from `ObjectDetectionPLModel`.

---

## `PolygonRNNPLModel`

PyTorch Lightning model for vertex-by-vertex polygon prediction using a recurrent neural network.

**Import path:** `pytorch_segmentation_models_trainer.model_loader.polygon_rnn_model.PolygonRNNPLModel`

### Constructor

```python
PolygonRNNPLModel(cfg, grid_size=28)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `cfg` | `DictConfig` | required | Hydra configuration |
| `grid_size` | `int` | `28` | Grid size for the polygon RNN output; output logits have `grid_size * grid_size + 3` classes (grid cells + EOS + first vertex + padding) |

### Configuration Keys

| Key | Description |
|-----|-------------|
| `model.load_vgg` | Whether to load pretrained VGG weights for the CNN encoder (default: `False`) |
| `val_dataset.sequence_length` | Maximum sequence length for test-time polygon generation |

### Key Methods

| Method | Signature | Description |
|--------|-----------|-------------|
| `training_step` | `(batch, batch_idx) -> Tensor` | Calls `compute(batch)` then `compute_loss_acc(batch, result)`; logs `loss/train` and `metrics/train_acc` |
| `validation_step` | `(batch, batch_idx) -> dict` | Same as training plus runs `model.test(...)` for sequence generation; evaluates PoLiS metric and IoU |
| `compute` | `(batch) -> Tensor` | Runs `self.model(image, x1, x2, x3)` and reshapes output to `(-1, grid_size*grid_size+3)` |
| `compute_loss_acc` | `(batch, result) -> Tuple[Tensor, Tensor]` | Computes cross-entropy loss and per-token accuracy |
| `evaluate_batch` | `(batch, result) -> Tuple` | Converts predicted/GT token sequences back to polygon vertex lists; computes batch PoLiS and IoU |

The loss function is `nn.CrossEntropyLoss()` (fixed).
