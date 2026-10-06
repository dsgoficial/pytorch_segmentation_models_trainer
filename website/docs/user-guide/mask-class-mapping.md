---
sidebar_position: 7
---

# Mask Class Mapping

`mask_class_mapping` remaps mask class values **on the fly**, when each mask
window is read, so the same mask rasters can be trained with a different class
scheme — merging classes, renaming them or sending some to the ignore index —
without building a new dataset.

`RasterPatchDataset`, `MBTilesMaskWindowedDataset` and `LulcInputWindowedDataset`
do not inherit from `SegmentationDataset`, so they accept the same parameter
through their own constructors, backed by the same helper.

Supported by:

| Dataset | Module | Where the mapping is applied |
|---|---|---|
| `SegmentationDataset` (and `SegmentationDatasetFromFolder`, `FullImageSegmentationDataset`) | `dataset_loader.dataset` | `SegmentationDataset.load_target_mask`, after reading the full mask |
| `CSVWindowedSegmentationDataset` | `dataset_loader.dataset` | right after reading the mask window (overrides `load_target_mask` so it is applied only once) |
| `RasterPatchDataset` | `dataset_loader.raster_patch_dataset` | right after reading the mask window |
| `MBTilesMaskWindowedDataset` (and `MBTilesLulcInputMaskWindowedDataset`) | `dataset_loader.mbtiles_mask_dataset` / `lulc_input_dataset` | right after reading the mask window (`read_mask_window(class_lut=...)`) |
| `LulcInputWindowedDataset` | `dataset_loader.lulc_input_dataset` | right after reading the mask window |

## Semantics

- Dict `{source_class: target_class}`; keys and values are integers in `[0, 255]`.
- Classes not listed are kept unchanged.
- Single pass, not chained: `{1: 2, 2: 3}` sends 1 → 2 and 2 → 3.
- Applied right after reading the mask window, **before** the `n_classes == 2`
  binarization and before Albumentations transforms.
- Invalid mappings raise `ValueError` when the dataset is built.
- The mapping does not change `n_classes`: set it to the number of classes
  **after** remapping, and keep the loss/metric `ignore_index` consistent when
  mapping a class to `255`.

## YAML example

Merge class 5 into class 3 of a 6-class mask, training with 5 classes:

```yaml title="conf/examples/mask_class_mapping.yaml"
train_dataset:
  _target_: pytorch_segmentation_models_trainer.dataset_loader.dataset.CSVWindowedSegmentationDataset
  input_csv_path: /data/train_windows.csv
  n_classes: 5
  mask_class_mapping:
    5: 3     # merge class 5 into class 3
```

Send a class to the ignore index instead:

```yaml
  mask_class_mapping:
    4: 255
```

## Python

```python
from pytorch_segmentation_models_trainer.dataset_loader.dataset import (
    CSVWindowedSegmentationDataset,
)

ds = CSVWindowedSegmentationDataset(
    input_csv_path="/data/train_windows.csv",
    n_classes=5,
    mask_class_mapping={5: 3},
)
```

The helpers behind it are
`dataset_loader.mask_class_mapping.build_mask_class_lut` (validates the mapping
and returns a 256-entry `uint8` lookup table) and `apply_mask_class_lut`.

## On the fly vs. `remap_mask_classes`

The `remap_mask_classes` tool rewrites the mask TIFFs on disk
(`conf/examples/remap_mask_classes.yaml`). Use it when the new class scheme is
permanent; use `mask_class_mapping` to train several class schemes from the
same masks (e.g. a coarse model and a fine model) without duplicating data.
