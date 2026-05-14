---
sidebar_position: 4
title: RandomCrop Dataset
---

# RandomCropSegmentationDataset

`RandomCropSegmentationDataset` reads large images on-the-fly using rasterio windowed reads instead of pre-generating tiles on disk. This eliminates the disk space overhead of a tile library and allows crop size, augmentation, and sampling strategy to be changed without reprocessing data.

---

## When to Use It

| Scenario | Recommended class |
| --- | --- |
| Pre-tiled dataset on disk | `SegmentationDataset` |
| Structured folder hierarchy, no CSV needed | `SegmentationDatasetFromFolder` |
| Large full-scene images, on-the-fly cropping | **`RandomCropSegmentationDataset`** |
| Systematic sliding-window evaluation | `RasterPatchDataset` |

Use `RandomCropSegmentationDataset` when your source imagery is stored as large GeoTIFFs (whole scenes, country-scale mosaics) and pre-cutting tiles is impractical.

---

## CSV Format

The input CSV must point to the **full-size** source images and masks, not to pre-cut tiles.

```csv
image,mask
/data/scenes/area_a.tif,/data/masks/area_a.tif
/data/scenes/area_b.tif,/data/masks/area_b.tif
```

Required columns: `image`, `mask`.

---

## Basic Configuration

```yaml
train_dataset:
  _target_: pytorch_segmentation_models_trainer.dataset_loader.dataset.RandomCropSegmentationDataset
  input_csv_path: /data/full_scenes/train.csv
  root_dir: /data/full_scenes
  crop_size: 512          # square crop in pixels
  samples_per_epoch: 4000 # virtual epoch size (number of __getitem__ calls)
  n_classes: 2
  augmentation_list:
    - _target_: albumentations.HorizontalFlip
      p: 0.5
    - _target_: albumentations.VerticalFlip
      p: 0.5
    - _target_: albumentations.Normalize
      mean: [0.485, 0.456, 0.406]
      std: [0.229, 0.224, 0.225]
    - _target_: albumentations.pytorch.ToTensorV2
  data_loader:
    shuffle: true
    num_workers: 8
    pin_memory: true
    batch_size: 16
    drop_last: true
```

---

## Constructor Parameters

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `input_csv_path` | `str` | required | CSV listing full-size image and mask paths. |
| `root_dir` | `str` | `None` | Root directory prepended to relative paths in the CSV. |
| `crop_size` | `int` | `512` | Side length of the square crop in pixels. |
| `samples_per_epoch` | `int` | `1000` | Number of `__getitem__` calls per epoch (virtual epoch size). |
| `n_classes` | `int` | `2` | Number of segmentation classes. |
| `augmentation_list` | list | `None` | Albumentations transforms applied after cropping. |
| `data_loader` | config | `None` | DataLoader keyword arguments. |
| `lru_cache_size` | `int` | `64` | Maximum number of simultaneously open rasterio file handles. |
| `class_balanced_sampling` | `bool` | `False` | Weight image selection by class frequency to oversample rare classes. |
| `cutmix_prob` | `float` | `0.0` | Probability of applying class-aware CutMix per sample. |
| `cutmix_alpha` | `float` | `1.0` | Beta distribution parameter for CutMix bounding box size. |
| `classmix_prob` | `float` | `0.0` | Probability of applying ClassMix (copy-paste of a class region). |
| `soft_labels` | `bool` | `False` | Return float soft-label masks instead of integer hard masks. |
| `grid_mode` | `bool` | `False` | Use deterministic sliding-window grid positions instead of random crops. |
| `grid_step` | `int` | `None` | Step between grid crops. Defaults to `crop_size` (no overlap). |
| `serialize_rasterio_reads` | `bool` | `False` | Serialize reads from the same source raster across DataLoader workers. |
| `rasterio_lock_dir` | `str` | `/tmp/psmt_rasterio_locks` | Directory used for per-raster lock files. |
| `reopen_rasterio_on_read` | `bool` | `False` | Open and close the raster inside each locked read instead of reusing per-worker cached handles. |
| `n_first_rows_to_read` | `int` | `None` | Limit the number of CSV rows read. |

---

## File Handle Caching (`_RasterioLRUCache`)

Opening a rasterio dataset has non-trivial OS overhead. The class maintains a per-worker LRU cache of open `DatasetReader` handles. When the cache is full the least-recently-used handle is closed explicitly, releasing the file descriptor immediately rather than waiting for garbage collection.

`lru_cache_size` (default `64`) controls how many handles are kept open simultaneously. A larger cache helps when training on many images per epoch; a smaller cache reduces file descriptor pressure on systems with tight limits.

```yaml
train_dataset:
  _target_: ...RandomCropSegmentationDataset
  lru_cache_size: 128   # keep up to 128 scenes open per worker
```

### Concurrent Reads From One Raster

Some GDAL/libtiff and filesystem combinations fail when multiple DataLoader workers read compressed windows from the same large GeoTIFF concurrently. If errors such as `TIFFReadEncodedTile() failed`, `LZWDecode`, or partial tile reads appear only when `num_workers > 0`, enable serialized reads:

```yaml
train_dataset:
  _target_: ...RandomCropSegmentationDataset
  serialize_rasterio_reads: true
  rasterio_lock_dir: /tmp/psmt_rasterio_locks
  reopen_rasterio_on_read: true
```

Workers can still read different rasters in parallel. Reads from the same raster are processed one at a time.

---

## Class-Balanced Sampling

By default, images are selected proportionally to their pixel area (larger images are sampled more often). Setting `class_balanced_sampling: true` instead weights image selection by the inverse frequency of rare classes: images that contain under-represented classes are sampled more often.

```yaml
train_dataset:
  _target_: ...RandomCropSegmentationDataset
  class_balanced_sampling: true
```

This is computed once at dataset initialisation by reading the mask histograms from the CSV.

:::tip
Class-balanced sampling is complementary to loss weighting (e.g. `CrossEntropyLoss` with `class_weights`). Use both together for severe class imbalance.
:::

---

## CutMix and ClassMix

### CutMix (`cutmix_prob`)

Class-aware CutMix pastes a rectangular region from a second crop onto the primary crop. The second image is selected to maximise class diversity: the sampler prefers an image that contains a different dominant class from the first.

```yaml
train_dataset:
  _target_: ...RandomCropSegmentationDataset
  cutmix_prob: 0.5      # 50% of samples apply CutMix
  cutmix_alpha: 1.0     # Beta(1, 1) = uniform box size
```

The mixed label is the pixel-wise union of both masks in the pasted region.

### ClassMix (`classmix_prob`)

ClassMix copies a randomly selected class region from a second image and pastes it onto the primary crop. This is particularly effective for rare classes.

```yaml
train_dataset:
  _target_: ...RandomCropSegmentationDataset
  classmix_prob: 0.3
```

---

## Soft Labels

When `soft_labels: true`, the dataset returns float masks in `[0, 1]` instead of integer class indices. This enables label smoothing at the pixel level and is used in training setups with label noise or probabilistic annotations.

```yaml
train_dataset:
  _target_: ...RandomCropSegmentationDataset
  soft_labels: true
```

The `_shared_step` training loop detects soft labels automatically and uses the appropriate loss path.

---

## Grid Mode

`grid_mode: true` switches from random crop positions to a deterministic sliding-window grid. This is useful for **validation** with `RandomCropSegmentationDataset` (reproducible coverage) or for pseudo-labelling where every pixel must be covered.

```yaml
val_dataset:
  _target_: ...RandomCropSegmentationDataset
  input_csv_path: /data/full_scenes/val.csv
  crop_size: 512
  grid_mode: true
  grid_step: 256        # 50% overlap between crops
```

The `samples_per_epoch` parameter is ignored in grid mode — the dataset length is determined by the number of grid positions computed from all images.

`configure_optimizers` accounts for grid mode automatically when computing `steps_per_epoch` for `OneCycleLR`.

---

## Full YAML Example

```yaml
train_dataset:
  _target_: pytorch_segmentation_models_trainer.dataset_loader.dataset.RandomCropSegmentationDataset
  input_csv_path: /data/full_scenes/train.csv
  root_dir: /data/full_scenes
  crop_size: 512
  samples_per_epoch: 8000
  n_classes: 5
  lru_cache_size: 64
  class_balanced_sampling: true
  cutmix_prob: 0.3
  cutmix_alpha: 1.0
  classmix_prob: 0.2
  augmentation_list:
    - _target_: albumentations.HorizontalFlip
      p: 0.5
    - _target_: albumentations.VerticalFlip
      p: 0.5
    - _target_: albumentations.RandomBrightnessContrast
      p: 0.3
    - _target_: albumentations.Normalize
      mean: [0.485, 0.456, 0.406]
      std: [0.229, 0.224, 0.225]
    - _target_: albumentations.pytorch.ToTensorV2
  data_loader:
    shuffle: true
    num_workers: 8
    pin_memory: true
    batch_size: 16
    drop_last: true

val_dataset:
  _target_: pytorch_segmentation_models_trainer.dataset_loader.dataset.RandomCropSegmentationDataset
  input_csv_path: /data/full_scenes/val.csv
  root_dir: /data/full_scenes
  crop_size: 512
  grid_mode: true
  grid_step: 256
  n_classes: 5
  augmentation_list:
    - _target_: albumentations.Normalize
      mean: [0.485, 0.456, 0.406]
      std: [0.229, 0.224, 0.225]
    - _target_: albumentations.pytorch.ToTensorV2
  data_loader:
    shuffle: false
    num_workers: 4
    pin_memory: true
    batch_size: 16
    drop_last: false
```
