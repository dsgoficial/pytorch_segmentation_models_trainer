# Unreleased

## Domain Adaptation

- Added initial domain adaptation module (`domain_adaptation/`) with an extensible base class (`BaseDomainAdaptationMethod`), feature hook extraction (`feature_hooks.py`), adaptation schedulers (`schedulers.py`), and a monitoring callback (`callbacks/monitor_callback.py`).
- Added `DomainAdaptationModel` (inherits `Model`) that orchestrates source/target dataloaders, adaptation loss weighting, and per-epoch scheduler stepping.
- Added `DomainAdaptationConfig` dataclass in `config_definitions/domain_adaptation_config.py`.
- Added comprehensive test suite for domain adaptation (`test_base_method.py`, `test_domain_adaptation_config.py`, `test_domain_adaptation_model.py`, `test_feature_hooks.py`, `test_schedulers.py`).
- Added documentation: user guide (`advanced/domain-adaptation.md`), config reference (`advanced/domain-adaptation-config-reference.md`), and implementing custom methods guide (`advanced/domain-adaptation-implementing-methods.md`).

## Test Time Augmentation (TTA)

- Added `tools/tta/tta.py` implementing TTA with all 8 symmetries of the D4 dihedral group (4 rotations × 2 flips). Each augmentation has an exact inverse so predictions are de-augmented and averaged with no spatial artifacts.
- Added `apply_tta()` helper for applying TTA to any segmentation model callable.
- Exposed `use_tta` and `tta_augmentations` fields on all inference processor classes and `InferenceProcessorConfig`/`PredictSingleImageConfig`. `SingleImageFromFrameFieldProcessor` automatically skips the `crossfield` output during TTA de-augmentation.
- `test_step()` in `Model` now applies TTA when `cfg.use_tta=True`.
- Added TTA user guide (`website/docs/advanced/tta.md`) and inference documentation section.

## Transformer & Foundation Model Support

- Added `HuggingFaceSegmentationWrapper` (`custom_models/huggingface_models.py`): loads any `AutoModelForSemanticSegmentation` from the Hub or local path, bypasses the HF processor, and upsamples logits back to input resolution.
- Added `TimmEncoderWithSMPDecoder` (`custom_models/timm_models.py`): combines a `timm` `features_only` backbone with SMP UNet/FPN/PAN decoders.
- Added `TerraTorchSegmentationWrapper` (`custom_models/terratorch_models.py`): bridges TerraTorch foundation model encoders (Prithvi, Clay, SatMAE) with a linear or FPN segmentation head; supports single- and multi-temporal inputs.
- Added `ModelOutputAdapter` (`custom_models/transformer_adapters.py`): normalises any model output (HF dataclass, dict, tuple, or plain tensor) to a `(B, C, H, W)` tensor with optional bilinear upsampling.
- Added LoRA / PEFT fine-tuning support (`fine_tuning/lora_utils.py`): `apply_fine_tuning_strategy()` supports `full`, `freeze_backbone`, `linear_probe`, and `lora` strategies; `LoraAdapterConfig` and `FineTuningConfig` dataclasses added; `merge_lora_weights()` for deployment.
- `predict.py` now auto-merges LoRA adapter weights before inference; a `keep_lora_adapters` flag skips the merge for fine-tuning resumption.
- Hardened training loop in `Model`: `_unpack_batch()` replaces fragile `batch.values()` unpacking with configurable `image_key`/`mask_key`; `set_encoder_trainable()` no longer assumes a `.encoder` attribute; `_prepare_preds_for_metrics()` guards metric calls against malformed outputs.
- Added 6 example YAML configs: `smp_mit_b2.yaml`, `smp_tu_convnext.yaml`, `segformer_hf.yaml`, `segformer_lora.yaml`, `vit_linear_probe.yaml`, `prithvi_terratorch.yaml`.
- Added `[transformers]` pip extras group and a dedicated CI job for the transformer test suite.

## RasterPatchDataset

- Added `RasterPatchDataset` (`dataset_loader/raster_patch_dataset.py`): scans image/mask directory pairs recursively and exposes every `patch_size × patch_size` window (with configurable stride) as an independent dataset item. Global index to `(image, row, col)` mapping runs in O(log N) via `bisect` over cumulative patch counts; rasterio windowed reads ensure full images never enter RAM.
- Supports augmentations, `selected_bands`, `image_dtype`, `mask_extension`, `n_classes` (binary binarisation when `n_classes=2`), and `reset_augmentation_function`.
- Emits `UserWarning` for orphaned mask files (mask without a corresponding image) to surface dataset misconfiguration.
- Added `RasterPatchDatasetConfig` dataclass and example YAML (`conf/examples/raster_patch_segmentation.yaml`).

## Dataset improvements

- Added `test_dataset` support with `test_step()`, `test_dataloader()`, and `test_metrics` (prefixed `test/`) in `Model` and `FrameFieldSegmentationPLModel`. `trainer.test()` is now called automatically after `trainer.fit()` when `test_dataset` is present in the config. This completes the three-way dataset split: `train_dataset` → training loop; `val_dataset` → per-epoch monitoring during fit; `test_dataset` → final held-out evaluation after fit.
- Added `test_dataset` field to `TrainConfig` dataclass and `test_dataset` block to all example YAML configs.
- Added `SegmentationDatasetFromFolder`: a new dataset class that discovers image/mask pairs recursively from two root folders, without requiring a CSV file. Matching is done by relative subfolder path and file stem. Supports all parameters of `SegmentationDataset`. Raises `ValueError` when no valid pairs are found.
- `SegmentationDataset.__init__` now accepts an optional `df` parameter (pre-built `pd.DataFrame`) in addition to `input_csv_path`, enabling programmatic dataset creation without a CSV file on disk. Fully backwards-compatible.
- Added configurable `image_dtype` field to `SegmentationDataset`, `RandomCropSegmentationDataset`, and their configs, accepting `uint8` (default), `uint16`, `float32`, or `native`. Auto-normalization scales correctly per dtype (`/255`, `/65535`, or no division). Fully backwards-compatible.

## Inference improvements

- Added `normalize_max_value` parameter to all inference processor classes (`AbstractInferenceProcessor` and all subclasses), exposing Albumentations' `max_pixel_value` for the normalization step. Default `None` preserves the previous behaviour (`255.0`). Use `normalize_max_value: 65535.0` for uint16 imagery or `1.0` for pre-normalized float32.

## Bug fixes

- Fixed bug in `Model.__init__`: `gpu_val_transform` and `gpu_train_transform` were accessed via `cfg.val_dataset`/`cfg.train_dataset` without checking if those keys exist, causing `AttributeError` when the corresponding dataset config was omitted.
- Fixed `val_dataloader()` to return `None` gracefully when `val_ds` is `None` (i.e. `val_dataset` absent from config), allowing training-only runs without a validation loop.
- Removed the orphan `set_test_dataset()` method from `FrameFieldSegmentationPLModel` (superseded by the new `test_ds` attribute set in `Model.__init__`).

# Version 1.0.1

- Bug fix on prediction in a multi gpu environment;

# Version 1.0.0

- Updated to newest pytorch lightning;
- Semantic Segmentation model updated for multi class;
- New image callback for multi class semantic segmentation;
- Added improvements for OneCycleLR scheduler (automatic calculation of steps_per_epoch);
- Added band selection on SemanticSegmentation dataset;
- Added compound_loss for Semantic Segmentation models;
- Added experiment evaluation pipeline;
- New multi class semantic segmentation inference pipeline (the old one worked only on binary semantic segmentation);
- Bug fixes on inferences;

# Version 0.17.0

- New evaluation metrics;
- New evaluation on test set;
- New inference service with image upload;
- Bug fixes;

# Version 0.16.4


- sahi version bump;
- Bug fix with parameter grid on image callbacks and mod polymapper.

# Version 0.16.3

- Bug fixes on ModPolymapper training when some parts are frozen.

# Version 0.16.2

- Bug fix on ModPolyMapper when choosing not to evaluate while training.
- Added the option of freezing some parts of ModPolyMapper.

# Version 0.16.1

- Dependencies fix.

# Version 0.16.0

- New Mod PolyMapper model;
- Matching methods added;
- Evaluation methods added;

# Veresion 0.15.0

- New Naive Mod PolyMapper model (Object Detection + PolygonRNN);
- New Naive Mod Polymapper dataset;
- New callback: Frame Field Only Crossfield Warmup Callback;
- New inference processors for Object Detection and PolygonRNN;
- Bug fix on object detection model;
- Bug fix on bounding box mask building;
- Bug fix on polygon iou with invalid geometries;
- Minor code refactor;

# Veresion 0.14.2

- Bug fix on PolygonRNN polygon tokenizer.

# Veresion 0.14.1

- Bug fix on convert dataset;
- Bug fix on PolygonRNNDataset;
- Bug fix on PolygonRNNResultCallback when using gpu;
- Bug fix on PolygonRNNPLModel;

# Version 0.14.0

- Vector IOU;
- Polis metric added;
- IoU added to PolygonRNN training loop;
- Object detection visualization callback added;
- PolygonRNN visualization callback added;
- Bug fix on polygon building on build mask geometry handling;

# Version 0.13.1

- Bug fixes on SegLoss parameters;

# Version 0.13.0

- Dataset conversion added. It is possible to convert between some formats of dataset;
- Tversky Loss and Focal Tversky Loss added;
- LabelSmoothingLoss added;
- MixUpAugmentationLoss added;
- KnowledgeDistillationLoss added;
- Mixup augmentation added to Frame Field Model;

# Version 0.12.1

- Bug fixes on mask building;
- Bug fixes on detection model training.
- New mode on build masks;

# Version 0.12.0

- Minor improvements on polygonization methods;
- Inference server added;


# Version 0.11.0

- Gradient Centralization added;


# Version 0.10.0

- Object Detection added;
- Instance Segmentation added;

# Version 0.9.0

- PolygonRNN model added;
- Added the option of choosing the number of images on ImageCallback;
- Added the option of adding created masks to existing csv;
- Added the option of generating bounding boxes in create masks;
- Added the option of converting csv dataset to coco dataset;

# Version 0.8.2

- Fixes on requirements;


# Version 0.8.1

- Minor improvements and bug fixes on polygon building inference;
- Bug fixes on mask builder;
- Performance improvement on mask builder using coco format;

# Version 0.8.0

- Added inference features;
- Improved polygon inference;

# Version 0.7.2

- Changed the versions of pytorch and torchvision.

# Version 0.7.1

- Added MANIFEST.in to include missing yml on pypi packaging.

# Version 0.7.0

- Bug fix on loss sync;
- Custom models from Frame Field implementation (to compare training results);
- New HRNet-OCR-W48 backbone;
- Fixed bugs on new versions of pytorch-lightning;
- Build mask from COCO dataset format;

# Version 0.6.0

- Polygon inference
- Unittests to Polygon inference;
- Bug fixes warmup callback (invalid signature on method);
- FrameFieldResultCallback renamed to FrameFieldOverlayedResultCallback;
- New implementation of FrameFieldResultCallback;
- Invalid mask handling (frame field training mask with only polygon mask and empty vertex and boundary masks);
- Added multiple schedulers option;
- Added IoU 10, 25, 50, 75 and 90;
- Added GPU augmentation using kornia;

# Version 0.5.1

- Bug fixes when inputs are RGBA images;
- Bug fixes on frame field model with models other than U-Net;
- Bug fixes on FrameFieldResultCallback (all black image fixed).

# Version 0.5.0

- Added frame field training image visualization callback.

# Version 0.4.1

Bug fixes on missing entrypoints and mask process execution.

# Version 0.4

## Polygoniztion by Frame Field Learning features

- FrameField dataset
- FrameField Learning
- Polygonization

# Version 0.3.2

Bug fixes on image callback when Pytorch Lightning DDP is used.

# Version 0.3.1

Bug fixes when Pytorch Lightning DDP is used.

# Version 0.3.0

- Custom metric option in the model config;
- pytorch_toolbelt added as required package. This enables usage of the models, losses and metrics in the training;
- Added the option of setting a limit of rows to be read in the csv dataset;
- Added the option of setting a root_dir to the dataset. This root_dir will be concatenated to the entry in the csv dataset before loading the image;
- Bug fixes on image_callback;

# Version 0.2.1

Fixes relative path bug on dataset

# Version 0.2.0

## New custom callbacks:

- ImageSegmentationResultCallback: Callback that logs the results of the training on TensorBoard and on saved files; and
- WarmupCallback: Applies freeze weight on encoder during callback epochs and then unfreezes the weights after the warmup epochs.

## Metrics added to Segmentation Model:

- Accuracy;
- Precision;
- Recall; and
- Jaccard Index (IoU).

# Version 0.1.4

First version of metrics added.

Bug fixes on dataset reading with prefix path.

# Version 0.1.3

Bug fix on entry points and --config-dir syntax.

# Version 0.1.2

Bug fix on Python's version.

# Minor bug fix

Bug fix.

# First Release

Framework based on Pytorch, Pytorch Lightning, segmentation_models.pytorch and hydra to train semantic segmentation models using yaml config files.
