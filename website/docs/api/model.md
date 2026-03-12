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
| `configure_optimizers` | `() -> Tuple[List, List]` | Builds optimizer from `cfg.optimizer`; builds scheduler list from `cfg.scheduler_list`. Automatically computes `steps_per_epoch` for `OneCycleLR` by reading the training CSV |
| `train_dataloader` | `() -> DataLoader` | Creates `DataLoader` for `train_ds` using `cfg.hyperparameters.batch_size` and `cfg.train_dataset.data_loader` settings |
| `val_dataloader` | `() -> DataLoader` | Creates `DataLoader` for `val_ds` using `cfg.hyperparameters.batch_size` and `cfg.val_dataset.data_loader` settings |
| `forward` | `(x) -> Tensor` | Delegates to `self.model(x)` |
| `predict_step` | `(batch, batch_idx, dataloader_idx=0) -> Tensor` | Runs `self(batch)` for inference |
| `set_encoder_trainable` | `(trainable=False) -> None` | Freezes or unfreezes the `model.encoder` parameters |

### Configuration Keys

| Key | Required | Description |
|-----|----------|-------------|
| `model` | Yes | Hydra target for the backbone architecture |
| `loss` | Conditional | Simple loss (lowest priority dispatch path) |
| `loss_params.compound_loss` | Conditional | New YAML-based compound loss config (highest priority) |
| `loss_params.multi_loss` | Conditional | Legacy multi-loss config (medium priority) |
| `optimizer` | Yes | Hydra target for the optimizer |
| `scheduler_list` | No | List of scheduler configs; each entry has a `scheduler` key and Lightning interval/frequency keys |
| `hyperparameters.batch_size` | Yes | Batch size for both train and val dataloaders |
| `hyperparameters.epochs` | No | Used when auto-computing `steps_per_epoch` |
| `pl_trainer` | No | Passed to the Lightning `Trainer`; also checked for `devices` and `accumulate_grad_batches` |
| `callbacks` | No | List of Hydra-instantiated callbacks |
| `metrics` | No | List of `torchmetrics` metrics to compute; auto-prefixed `train/` or `val/` |
| `logger` | No | Logger configuration |
| `train_dataset` | Yes (unless `inference_mode`) | Dataset config including `input_csv_path` and `data_loader` sub-config |
| `val_dataset` | Yes (unless `inference_mode`) | Dataset config for validation |
| `replace_model_activation` | No | Dict with `old_activation` and `new_activation` for activation replacement |

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
| `training_step` | `(batch, batch_idx) -> Tensor` | Runs forward pass, computes `MultiLoss`, logs `loss/train` and all component losses under `losses/train_{name}` |
| `validation_step` | `(batch, batch_idx) -> Tensor` | Same as training but logs `loss/val` and `losses/val_{name}` |
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
