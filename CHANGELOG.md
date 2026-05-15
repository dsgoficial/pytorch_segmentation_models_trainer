# Unreleased

## Training Callbacks

- Added `FinalMetricsCallback`: saves all epoch-averaged metrics and losses from the last training epoch to a JSON file via `trainer.callback_metrics`. Relative `output_path` values resolve against `trainer.log_dir` so the file lands alongside TensorBoard/CSV logs. A second hook (`on_test_end`) merges test-set metrics into the same file without overwriting train/val entries. Safe for multi-GPU training via `@rank_zero_only`. Exported from `custom_callbacks` and usable as a Hydra callback with `_target_: pytorch_segmentation_models_trainer.custom_callbacks.FinalMetricsCallback`.

## Image Callbacks

- Added `shuffle_indices_seed` parameter (default `None`) to `ImageSegmentationResultCallback` (and all subclasses via inheritance, including `AutoencoderResultCallback`). When set, `AutoencoderResultCallback` samples a reproducible random subset of validation indices instead of always visualizing the first N rows. The seed is applied once per epoch via `numpy.random.RandomState`, so the shown subset is stable across epochs. When `None`, behaviour is unchanged (first N samples).

## ProgressiveDecoder

- Added `ProgressiveDecoder` to `custom_models/generic_autoencoder.py`: multi-stage convolutional decoder that doubles spatial resolution at each step with two conv+ReLU layers, replacing the single bilinear interpolation of `GenericDecoder`. Supports the same `output_activation` options (`None`, `"sigmoid"`, `"tanh"`) and validates that `scale_factor` is a power of 2.
- Wired `use_progressive_decoder` parameter (default `False`) into `GenericAutoencoder` and `GenericVariationalAutoencoder`. Set to `True` to swap in `ProgressiveDecoder`; also exposed `base_channels` (default 128) and `min_channels` (default 32) for channel schedule control.
- Updated `GenericVariationalAutoencoderConfig` dataclass with `use_progressive_decoder`, `base_channels`, and `min_channels` fields for Hydra/YAML configuration.
- Added 22 tests in `tests/test_generic_autoencoder.py` covering standalone shape contracts (scale factors 2–32), activation bounds, non-power-of-2 and invalid-activation guards, gradient flow, and end-to-end integration with both `GenericAutoencoder` and `GenericVariationalAutoencoder`.

## KL Annealing

- Added `start_after` parameter (default `0`) to `KLAnnealingCallback`. When set to a positive integer, beta is held at `min_beta` for the first `start_after` steps (or epochs when `use_epochs=True`) before the annealing ramp begins. The ramp then spans exactly `annealing_steps` units starting from `start_after`. `KLAnnealingCallbackConfig` exposes the same field. Zero preserves previous behaviour unchanged.

## VAE Loss

- Added `smooth_l1` reconstruction mode to `VariationalAutoencoderLoss` using `F.smooth_l1_loss` (Huber loss). New `smooth_l1_beta` parameter (default `0.1`) controls the L2-to-L1 transition threshold and is independent of the KL `beta` weight. The term is governed by `reconstruction_weight` like all other reconstruction modes.

## Image Callbacks

- Fixed `ImageSegmentationResultCallback.on_validation_epoch_end` ignoring `log_every_k_epochs` — it now skips visualization on non-matching epochs, consistent with `AutoencoderResultCallback` and `EnhancedImageSegmentationResultCallback`.
- Fixed `ImageSegmentationResultCallback.on_validation_epoch_end` mutating `self.n_samples` as side-effect — replaced with local variable.
- Fixed `FrameFieldResultCallback`, `FrameFieldOverlayedResultCallback`, and `PolygonRNNResultCallback` ignoring `log_every_k_epochs` — all now respect the frequency setting.
- Fixed `AutoencoderResultCallback` calling `val_dataloader()` twice — cached in a local variable.
- All callbacks now warn (`logger.warning`) and cap gracefully when `n_samples > len(val_ds)`, instead of silently showing fewer images without explanation.
- Fixed duplicate filenames when multiple crops of the same image are logged: `save_plot_to_disk` now accepts `sample_idx` and embeds it as `_idx{i}` in the filename; TensorBoard tags also include the index.
- Added 5 new tests: `test_save_plot_to_disk_includes_sample_idx`, `test_autoencoder_callback_n_samples_capped_with_warning`, `test_autoencoder_callback_filename_includes_idx`, `test_log_every_k_epochs_skips_non_matching_epochs`, `test_autoencoder_log_every_k_epochs_respected`.

## Reproducibility / Seed

- `set_training_seed` now delegates to `pytorch_lightning.seed_everything` instead of calling `random.seed`/`np.random.seed`/`torch.manual_seed` directly. This ensures `PL_GLOBAL_SEED` is set so DDP-spawned subprocesses inherit the seed automatically.
- `deterministic_cudnn=True` now also calls `torch.use_deterministic_algorithms(True)`, matching the behaviour of `Trainer(deterministic=True)` and covering all ops, not just CuDNN.
- `train()` passes `deterministic=True` to the `Trainer` when `cfg.deterministic_cudnn` is `True`, completing the Lightning integration.
- Added tests: `test_sets_pl_global_seed`, `test_pl_global_seed_uses_seed32`, `test_trainer_gets_deterministic_true_when_deterministic_cudnn`, `test_trainer_no_deterministic_when_cudnn_false`; updated `test_deterministic_cudnn_true` and `test_deterministic_cudnn_false_by_default` to also assert on `torch.are_deterministic_algorithms_enabled()`.

## KL Annealing (VAE)

- Added `KLAnnealingCallback` (`custom_callbacks/kl_annealing_callback.py`): PyTorch Lightning callback that gradually increases the KL weight (`beta`) in `VariationalAutoencoderLoss` during training. Supports three schedules — `linear`, `cosine`, and `cyclical`. Can operate step-based (default) or epoch-based via `use_epochs`. Logs the current beta to TensorBoard as `scheduler/kl_beta` on every update.
- Added `KLAnnealingCallbackConfig` dataclass to `config_definitions/autoencoder_config.py` with Hydra ConfigStore registration under `group="callbacks"`, `name="kl_annealing"`.
- Extended `VariationalAutoencoderLoss.forward()` to return two additional keys — `weighted_reconstruction_loss` and `weighted_kl_loss` — representing the actual contribution of each term to the total ELBO loss. These are automatically logged to TensorBoard alongside the existing `reconstruction_loss` and `kl_loss` keys.
- Added example config `conf/examples/vae_with_kl_annealing.yaml` demonstrating cosine KL annealing over 5000 steps with a free-reconstruction warmup (`min_beta=0`, `max_beta=1`).
- Added 20 unit tests in `tests/test_kl_annealing.py` covering schedule shapes, hook routing, beta clamping, TensorBoard logging, and config validation.
- Added `free_bits: float = 0.0` parameter to `VariationalAutoencoderLoss`. When positive, clamps each latent spatial position's KL to at least `free_bits` nats before averaging — blocking the gradient for collapsed positions so they are not over-regularised while the decoder still gets reconstruction gradients through them. Prevents posterior collapse without requiring careful beta tuning. Recommended: 0.1–0.5 nats for spatial VAEs.
- Added `kl_balance: bool = False` parameter to `VariationalAutoencoderLoss`. When `True`, scales the KL term by `(C × H × W) / (Cz × Hz × Wz)` so reconstruction and KL are proportional to the same total-information budget (matching the theoretical ELBO). Particularly useful when `encoder_depth` is high (e.g. depth=5 yields a ~24× ratio for 224×224 inputs). The raw `kl_loss` logging key is unaffected; only `weighted_kl_loss` and `loss` reflect the scaling.
- Updated `conf/examples/vae_with_kl_annealing.yaml` to demonstrate `free_bits: 0.25` and `kl_balance: true` alongside cosine KL annealing.

## Bug fixes

- Fixed `VariationalAutoencoderModel` and `AutoencoderModel` not computing or logging YAML-configured `metrics` (e.g. PSNR, SSIM) during training/validation. Both `_shared_step` overrides now extract the reconstruction tensor and call the `MetricCollection` when present.
- Fixed `AutoencoderResultCallback` (and base `ImageSegmentationResultCallback`) producing sepia-tinted images when the dataset uses `albumentations.ToFloat` instead of `albumentations.Normalize`. Root cause: `normalized_input=True` was the default, so ImageNet denormalization (`x * std + mean`) was applied to `[0, 1]` images that were never mean/std-normalized, compressing the range and adding an unequal warm channel shift. Fix: (1) added `normalized_input: false` to the `AutoencoderResultCallback` in `generic_variational_autoencoder_random_crop_folder.yaml`; (2) added `np.clip(0, 1)` after denormalization in `prepare_image_to_plot` to prevent out-of-range values from causing rendering artifacts.

## Autoencoder / VAE stability

- Added `output_activation` parameter to `GenericDecoder`, `GenericAutoencoder`, and `GenericVariationalAutoencoder`. Supported values: `None` (default, unchanged behaviour), `"sigmoid"` (output bounded to `[0, 1]`, correct for uint8/255 targets), `"tanh"` (output bounded to `[-1, 1]`). Without this, the unbounded decoder logits caused PSNR to go negative whenever MSE > 1.
- Fixed KL divergence scaling in `VariationalAutoencoderLoss`: replaced `torch.sum(...) / batch_size` with `torch.mean(...)` so the KL term is a per-element average, matching the scale of `F.mse_loss` and preventing the KL from dominating the total loss with large spatial latents.
- Added `logvar_clamp` parameter to `GenericVariationalAutoencoder`. Clamping log-variance before exponentiation prevents fp16 overflow in `exp(logvar)` and eliminates numerically negative KL values seen with `precision="16"`. Recommended value: `(-4.0, 4.0)` for fp16 training.
- Updated `GenericVariationalAutoencoderConfig` to expose `output_activation` and `logvar_clamp`.

## Tooling CLI

- Added `pytorch-smt-tools` entry point as a generic CLI tooling hub for command-line utilities.
- Added `compute-stats` subcommand (`pytorch-smt-tools compute-stats <yaml>`) that instantiates the training dataset defined in a YAML config, streams all samples through a DataLoader to compute per-channel mean and standard deviation, and writes the results back to the same YAML file.
- The command inserts an `albumentations.Normalize` entry (with the computed mean/std and the correct `max_pixel_value` for the dataset's `image_dtype`) into the `augmentation_list` of every dataset key (`train_dataset`, `val_dataset`, `test_dataset`) found in the YAML.
- If the YAML contains image-visualization callbacks (e.g. `AutoencoderResultCallback`, `ImageSegmentationResultCallback`), the command also updates their `norm_params` and sets `normalized_input: true`, eliminating the need to copy-paste statistics manually.
- Added `--dry-run`, `--skip-callbacks`, `--dataset-key`, `--batch-size`, and `--num-workers` flags.
- Added `click>=8.0.0` and `ruamel.yaml>=0.18.0` as project dependencies (ruamel.yaml preserves YAML comments and formatting on round-trip).
- Added unit tests in `tests/test_compute_dataset_stats.py` with 100% coverage.
- Changed `compute-stats` output format: mean/std values are now written to a top-level `normalization_parameters: {mean: [...], std: [...]}` key, and the `albumentations.Normalize` entry and callback `norm_params` reference them via Hydra interpolation (`${normalization_parameters.mean}`, `${normalization_parameters.std}`) instead of inlining the values directly. This makes it easy to override both at once from the command line or a child config without touching every dataset key.

## Dataset
- Added `WindowedImageDataset` for deterministic sliding-window (grid) patch extraction from rasters without requiring pre-generated masks.
- Added `WindowedImageAutoencoderDataset` specifically for Autoencoder validation/testing, yielding `(image, target)` pairs where `image` can be optionally corrupted while `target` remains clean.
- Added `IterableWindowedImageDataset` and `IterableWindowedImageAutoencoderDataset`, which shard whole source rasters across DataLoader workers to avoid concurrent reads from the same GeoTIFF.
- Both datasets support global indexing across multiple images of varying sizes using efficient binary search (bisect).
- Added `verify_windows` and `window_index_cache` to windowed image datasets so unreadable raster windows can be excluded from indexing during initialisation and the verified window coordinates can be reused on later runs.
- Added `serialize_rasterio_reads`, `rasterio_lock_dir`, and `reopen_rasterio_on_read` to windowed image and random-crop raster datasets so DataLoader workers can serialize reads from the same compressed GeoTIFF on shared storage and optionally avoid persistent GDAL handles.
- Fixed `RasterPatchDataset` linting by defining its module logger before the window-read error path uses it.
- Added example configuration `conf/examples/windowed_image_autoencoder.yaml`.
- Added unit tests in `tests/test_windowed_datasets.py` with 100% coverage.

# Version 1.2.0 - 2026-05-11

## Tests
- Alcançados 100% de test coverage para os módulos `pytorch_segmentation_models_trainer/tools/inference/inference_csv_builder.py` e `pytorch_segmentation_models_trainer/tools/evaluation/csv_builder.py`.
- Adicionados novos arquivos de testes unitários: `tests/test_inference_csv_builder.py`, `tests/test_csv_builder.py` e `tests/test_image_processing_worker.py`.
- Aumentada a cobertura global dos módulos `inference` e `evaluation` através de testes de casos de borda e fluxos de erro.
- Achieved 100% test coverage for all files in the `pytorch_segmentation_models_trainer/config_definitions/` directory.
- Created individual test files for all configuration dataclasses: `test_coco_dataset_config.py`, `test_dataset_config.py`, `test_dataset_distillation_config.py`, `test_edl_config.py`, `test_evaluation_config.py`, `test_experiments_runner_config.py`, `test_fine_tuning_config.py`, `test_inference_config.py`, `test_loss_config_definition.py`, `test_mc_dropout_config.py`, `test_predict_config.py`, `test_tools_config_def.py`, and `test_train_config.py`.

## Bug fixes
- Fixed `TrainConfig` dataclass in `config_definitions/train_config.py` which had an invalid `default_factory` for `callbacks` (was a list instead of a callable) and a problematic `default_factory` for `pl_model` and `model` (was calling `Model()` without required arguments).
- Fixed `LossParamsConfig` in `config_definitions/loss_config_definition.py` where `seg_loss_params` had an incorrect type hint (`SegParamsConfig` instead of `SegLossParamsConfig`).
- Fixed `LossWeightConfig` in `config_definitions/loss_config_definition.py` to use `Any` for the `weight` field, resolving an `OmegaConf` limitation with `Union[float, List[float]]`.

## Dataset
- Added Apache Parquet support for all datasets inheriting from `AbstractDataset`.
- Implemented an automatic caching mechanism that converts `.csv` metadata files to `.cache.parquet` on the first run. Subsequent runs read the Parquet file if the CSV has not been modified, significantly improving metadata loading speed and memory efficiency.
- Metadata files (e.g., `input_csv_path`) can now be provided directly as `.parquet` files.
- Added a CLI tool `csv-to-parquet` for manual conversion of CSV datasets to Parquet (supports single files and recursive directory conversion).
- Integrated `pyarrow` as a new dependency.
- Added unit tests for Parquet reading, caching logic, and CLI tool in `tests/test_dataframe_utils.py`.

## Experiments Runner

- Added `ExperimentsRunner` class (`tools/experiments_runner/experiments_runner.py`) that runs successive training experiments in series, each with an isolated seed and output directory.
- Seeds can be specified explicitly (`seeds: [42, 101, 28]`) or generated automatically at runtime by supplying only `n_runs: 5`. Providing both values is accepted when they are consistent; a conflict raises a `ValueError` with a clear message.
- Each run receives its seed via the existing `set_training_seed()` mechanism (Python `random`, NumPy, PyTorch CPU/CUDA, DataLoader workers), and writes checkpoints and logs to `<output_base_dir>/run_<idx:02d>_seed<seed>/` — the seed is visible directly in the filesystem path.
- Wall-clock training time and all Lightning `callback_metrics` (`train/*`, `val/*`, `test/*`) are captured per run, including test metrics when a `test_dataset` block is present.
- When `save_summary: true` (default), a `summary.csv` is updated incrementally after each run with per-run rows plus aggregated `mean` and `std` rows for all metrics and the training duration.
- After every completed run a `runner_state.json` is written to `output_base_dir`. Set `resume: true` in the config to skip already-completed runs on restart; auto-generated seeds are loaded from the state file so they remain stable across restarts.
- If a `logger:` block is present in the training config, the runner stamps the run identity into it: `version` is set to `"run_<idx:02d>_seed<seed>"` for TensorBoard/CSV loggers; `name` is appended with the same tag for WandB-style loggers.
- Added `ExperimentsRunnerConfig` dataclass in `config_definitions/experiments_runner_config.py` with new `resume` field, registered in Hydra's ConfigStore.
- Added new dispatch mode `run-experiments` in `main.py`.
- Added example configuration `conf/examples/experiments_runner.yaml` (UNet / ResNet-34, 3 fixed seeds).
- Added user documentation `website/docs/user-guide/experiments_runner.md`.
- Added 51 unit tests in `tests/test_experiments_runner.py` covering validation, seed resolution, config mutation, logger injection, metric collection, state file persistence, resume logic, and the full `run()` integration with a mocked trainer.

## Dataset Distillation
- Added `dataset_distillation.py` utilities for dataset distillation using Coreset of Medoids (Optimal Quantization).
- Implemented `extract_all_latents` for high-throughput embedding extraction from trained Autoencoders.
- Implemented `find_coreset_medoids` using K-Means and L2 distance (`torch.cdist`) to find representative real samples closest to cluster centroids.
- Added `KMeansClusteringTool` for orchestrating DDOQ pipelines, including OOM-safe Medoid search and Voronoi weight calculation.
- Implemented **DDOQ (Dataset Distillation by Optimal Quantization)** with variance reduction heuristics (Square Root) and a toggle for Vanilla density weights.
- Added `DDOQDistilledDataset` which returns the triplet `(image, mask, weight)`, supporting both hard-labels and teacher-generated soft-labels.
- Added `StudentSegmentationModel` (`pl.LightningModule`) implementing the DDOQ weighted loss: `min_theta sum(w * Loss(x, y, theta))`.
- Added support for **Adaptive K search** via the Elbow Method (Perpendicular Distance) in `kmeans_calculator.py`.
- Added utilities to save and load DDOQ results (indices and weights).
- Added `DatasetDistillationConfig` Hydra dataclass and registered it in the ConfigStore.
- Added example configuration `conf/examples/dataset_distillation.yaml`.
- Added user documentation in `website/docs/user-guide/dataset_distillation.md`.

## GPU K-Means
- Added `MiniBatchKMeans` PyTorch implementation for high-performance clustering on GPU.
- Added `KMeansClusteringTool` for orchestrating clustering pipelines with GeoPandas and PostGIS support.
- Supports K-Means++ initialization and efficient mini-batch updates for large datasets.
- Added GeoParquet and PostGIS export capabilities for clustered spatial data.

## Autoencoder
- Added `GenericVariationalAutoencoder` with SMP/HuggingFace encoder support, posterior `mu`/`logvar` projections, and the reparameterization trick for differentiable latent sampling.
- Added `VariationalAutoencoderLoss`, a composite reconstruction-plus-analytic-KL objective supporting MSE, L1, and BCE-with-logits reconstruction terms.
- Added `VariationalAutoencoderModel` to log total, reconstruction, and KL losses across train, validation, and test steps.
- Added Hydra dataclasses, API/user documentation, and a complete `AutoencoderRandomCropDataset` VAE example config with random horizontal, vertical, and transpose mirror flips.
- Moved image-only datasets to `dataset_loader/image_dataset.py` (`ImageDataset`, `CSVWindowedImageDataset`, `TiledInferenceImageDataset`, `AutoencoderDataset`, and `AutoencoderRandomCropDataset`), while keeping lazy compatibility exports from `dataset_loader.dataset` for existing configs.
- Added `AutoencoderRandomCropDataset` for self-supervised reconstruction from unlabeled image folders or CSV-backed full-size rasters. It discovers images recursively, supports deterministic train/validation splits, rasterio windowed random crops, selected bands, dtype handling, and input-only corruption augmentations.
- Added Hydra dataclasses and a folder-based example config for random-crop autoencoder training.
- Added `GenericAutoencoder` model supporting SMP and Transformers encoders.
- Added `AutoencoderModel` LightningModule for image reconstruction tasks.
- Added `AutoencoderDataset` for self-supervised learning and reconstruction.
- Added `AutoencoderResultCallback` for side-by-side visualization of input and reconstructed images during validation.
- Added example configuration `conf/examples/generic_autoencoder.yaml`.
- Updated documentation in `website/docs/user-guide/generic_autoencoder.md`.

# Version 1.1.0 - 2026-04-28

## Bug fixes

- Fixed `tests/test_build_mask.py::Test_BuildMask::test_build_output_dirs_raises_exception`: changed the output path to be a sub-directory of the input path in the test, ensuring the path validation logic in `build_destination_dirs` is correctly triggered and the expected exception is raised.
- Fixed `tests/test_configs/predict.yaml` Hydra composition: added the `@_global_` package override to the `train_config_used_in_predict_test` default inclusion. This ensures the configuration fields are merged into the root scope, allowing `train_dataset.input_csv_path` to be successfully overridden during prediction tests.
- Fixed `NameError: name 'DictConfig' is not defined` in `dataset_loader/dataset.py` (`load_augmentation_object`): `DictConfig` was used but not imported from `omegaconf`, causing all augmentation loading to silently fall back to raw OmegaConf objects and crash with `albumentations >= 2.x` when `A.Compose` tried to access `.available_keys` on them.
- Fixed `Model._unpack_batch` (`model_loader/model.py`) to respect the `image_key` and `mask_key` config fields (was hardcoded to `"image"` / `"mask"`). Also propagated the fix to `_shared_step` so training/validation steps honour custom keys end-to-end.
- Fixed `ValueError: prefetch_factor option could only be specified in multiprocessing` in `Model.train_dataloader`, `val_dataloader`, and `test_dataloader`: when `num_workers=0`, `prefetch_factor` is now set to `None` as required by PyTorch's DataLoader API.
- Fixed `MCDropoutInferenceProcessor` (`tools/inference/mc_dropout_inference_processor.py`) to save uncertainty rasters with suffix `_mc_uncertainty` instead of the generic `_uncertainty` from the parent class, matching the test expectation and making it distinguishable from TTA uncertainty maps.
- Fixed `merge_lora_weights` (`fine_tuning/lora_utils.py`) to use `is_peft_model()` for the PEFT check rather than a local `isinstance` guard; this makes the function testable without a real PEFT installation and consistent with `is_peft_model`.
- Fixed `TimmEncoderWithSMPDecoder` (`custom_models/timm_models.py`) for SMP 0.5.0 compatibility: replaced removed `use_batchnorm` kwarg with `use_norm` in `UnetDecoder.__init__`, and changed the decoder `forward` call to pass features as a single list instead of splatted positional arguments (both `UnetDecoder` and `FPNDecoder` now use `forward(features: list)` in SMP 0.5.x).
- Updated `test_caches.py::TestClassPresenceCacheAutoSave::test_auto_save_json_structure` to account for the `_config` metadata key now present in the class-presence cache JSON, filtering it out before counting data entries.
- Fixed `Model._shared_step` (`model_loader/model.py`) to apply `_prepare_preds_for_metrics` before computing train/val metrics: binary models output `[B, 1, H, W]` but torchmetrics expects `[B, H, W]`, causing a shape mismatch `RuntimeError` during training. The same fix was applied to the per-class IoU path.
- Fixed `test_frame_field_model.py::_make_ff_batch`: `class_freq` was built as `torch.ones(B)` (1-D) but `compute_seg_loss_weights` in `base_loss.py` sums over `dim=1`, requiring `[B, C]`. Also added the missing `gt_crossfield_angle: torch.zeros(B, 1, H, W)` key, required by `compute_gt_field` when `compute_crossfield: true`.
- Fixed `mod_polymapper.py` validation metric logging: added `numel() > 0` guard to skip empty tensors (produced when no polygons are detected in `fast_dev_run`), and added `.float()` before `.mean()` to handle Long-dtype metric values, preventing `ValueError: tensor must have a single element` and `RuntimeError: mean() dtype`.
- Fixed `test_polygonizer.py::test_polygonizer_acm_processor`: replaced `geopandas.testing.geom_almost_equals` (fixed 5×10⁻⁷ m tolerance, too strict for GeoJSON coordinate rounding) with a 1 cm tolerance wrapper; regenerated the `acm_polygonizer.geojson` baseline because the ACM algorithm produces different vertex coordinates with newer shapely/geopandas.
- Fixed `custom_callbacks/image_callbacks.py`: `CombinedLoader.loaders` was renamed to `CombinedLoader.iterables` in PyTorch Lightning ≥ 2.x; added a `getattr` fallback so the callback works on both old and new PL versions. Also fixed the follow-up `DataLoader.loader.dataset` chain: PL ≥ 2.x exposes plain `DataLoader` objects in `iterables`, so `.dataset` is now accessed directly with a fallback to the old `.loader.dataset` path.

## Installation & Environment

- Migrated to `uv` as the primary project and dependency manager.
- Added `uv sync` as the recommended installation method in `README.md` and documentation.
- Updated project requirements to Python 3.12+ and PyTorch 2.0+ (Lightning 2.4+).
- Updated website documentation (`website/docs/getting-started/installation.md`) with comprehensive `uv` installation guide and updated troubleshooting for CUDA 11.8.

## Reproducibility (Training Seed)

- Added `seed: Optional[int]` and `deterministic_cudnn: bool` fields to the `TrainConfig` dataclass (`config_definitions/train_config.py`). Both default to `None` / `False` so all existing configs are fully backward-compatible.
- Added `set_training_seed(seed, deterministic_cudnn=False)` utility function (`utils/seed_utils.py`). A single call seeds all randomness sources before model or dataset creation: Python `random`, NumPy `np.random`, `torch.manual_seed`, `torch.cuda.manual_seed_all`, and `PYTHONHASHSEED`. Optionally sets `torch.backends.cudnn.deterministic = True` and `torch.backends.cudnn.benchmark = False`. Returns a `torch.Generator` seeded with the same value for use in DataLoaders.
- Modified `train()` (`train.py`) to call `set_training_seed` as the very first operation when `cfg.seed` is present, ensuring model weight initialisation (SMP, timm, HuggingFace, custom architectures) is also reproducible.
- Modified `_worker_init_fn` (`dataset_loader/dataset.py`) to additionally seed Python's `random` module alongside NumPy. Since PyTorch sets `torch.initial_seed() = global_seed + worker_id` per worker automatically, both seeds are deterministic and unique per worker once a global seed is set.
- Modified `Model.train_dataloader`, `Model.val_dataloader`, and `Model.test_dataloader` (`model_loader/model.py`) to pass a `torch.Generator().manual_seed(cfg.seed)` to each `DataLoader` when `cfg.seed` is set, making the shuffle sampler sequence reproducible. Added `Model._make_dataloader_generator()` helper method.
- Added reference YAML config `conf/examples/reproducible_training.yaml` with inline comments explaining every field and listing all controlled randomness sources.
- Added reproducibility block (commented) to `conf/examples/smp_mit_b2.yaml` so users can enable it in one line.
- Added user documentation `website/docs/user-guide/reproducibility.md` covering the `seed` field, `deterministic_cudnn` trade-offs, Python API usage, and known limitations.

## MC Dropout (test-time uncertainty)

- Added `mc_dropout_utils.py` (`utils/mc_dropout_utils.py`): three pure utility functions with no inference-framework dependency. `enable_mc_dropout(model)` sets all `Dropout` / `Dropout2d` / `Dropout3d` layers to train mode while leaving the rest of the model in eval mode (BatchNorm keeps running statistics, only dropout randomness is re-enabled). `warn_if_no_dropout(model)` emits a `UserWarning` if the model has no dropout layers — in that case all T samples are identical and uncertainty is zero. `compute_uncertainty(samples, mode)` accepts a `[T, B, C, H, W]` tensor of softmax probabilities and returns `[B, 1, H, W]` uncertainty: `"entropy"` computes predictive entropy of the mean distribution (total uncertainty); `"mutual_information"` computes BALD — the difference between the entropy of the mean and the mean of the individual entropies (epistemic uncertainty only).
- Added `MCDropoutInferenceProcessor` (`tools/inference/mc_dropout_inference_processor.py`): extends `MultiClassInferenceProcessor` with MC Dropout inference. Overrides `predict_and_merge` to run `n_samples` stochastic forward passes per tile (after calling `enable_mc_dropout`), average the softmax probabilities for the class prediction, and — only when `export_uncertainty_map=True` — compute and merge the per-pixel uncertainty map. Uncertainty tensors are only allocated when requested, keeping the no-uncertainty path free of extra memory cost. Overrides `process` to skip the striped inference path (which uses a single TileMerger and cannot merge uncertainty); a warning is logged for large images. Exports `{stem}_mc_uncertainty.tif` (float32 single-band GeoTIFF, range `[0, log C]`) alongside the segmentation output when `export_uncertainty_map=True`.
- Modified `MultiClassInferenceProcessor` (`tools/inference/inference_processors.py`): added three new parameters `export_uncertainty_map` (default `False`), `uncertainty_mode` (default `"entropy"`), and `output_uncertainty_dir` (default `None`). When `tta_mode` is set and `export_uncertainty_map=True`, per-sample softmax probabilities are kept during the TTA loop and used to compute uncertainty via `compute_uncertainty()`; the result is merged via a separate `TileMerger` and exported as `{stem}_uncertainty.tif`. The no-uncertainty path (default) is fully unchanged. Added `_save_uncertainty_raster` helper for writing the float32 GeoTIFF with source CRS and transform preserved.
- Added `MCDropoutInferenceProcessorConfig` (`config_definitions/mc_dropout_config.py`): Hydra dataclass registered in the ConfigStore under `group="inference_processor"`, `name="mc_dropout"`. Fields: `n_samples`, `uncertainty_mode`, `export_uncertainty_map`, `num_classes`, `model_input_shape`, `step_shape`, `output_uncertainty_dir`.
- Added reference YAML config `mc_dropout_inference.yaml` under `conf/examples/` with inline comments explaining the `decoder_dropout` requirement and the `export_uncertainty_map` flag.

## Evidential Deep Learning (EDL)

- Added `EvidentialWrapper` (`custom_models/edl_wrapper.py`): architecture-agnostic wrapper that replaces Softmax with a Dirichlet parameterisation (`evidence = Softplus(logits)`, `alpha = evidence + 1`). Works with any model that returns `[B, K, H, W]` tensors including SMP, HuggingFace, timm, and custom models. Forward pass returns a dict with `logits`, `evidence`, `alpha`, `probs`, and `uncertainty` keys. Handles tuple, dict, and plain-tensor model outputs automatically.
- Added `EvidentialMSELoss` and `EvidentialKLLoss` (`custom_losses/edl_loss.py`): two-component EDL loss designed for use with the existing `CompoundLoss` / `MultiLoss` epoch-weight scheduling mechanism. `EvidentialMSELoss` computes the MSE integrated analytically over the Dirichlet distribution (bias² + variance terms). `EvidentialKLLoss` computes `KL[Dir(α̃) || Dir(1,...,1)]` after removing evidence for the correct class, penalising residual wrong-class evidence. KL annealing is expressed as a dynamic weight list in the YAML — no custom scheduler needed.
- Added `edl_utils.py` (`custom_losses/edl_utils.py`): analytic helpers for Dirichlet statistics: `one_hot_encode` (hard + soft label support, ignore_index handling), `dirichlet_strength`, `epistemic_uncertainty` (u = K/S), `dirichlet_kl_divergence`, `kl_divergence_to_uniform`, `edl_kl_regulariser`.
- Modified `Model._shared_step` to detect `EvidentialWrapper` output (dict with `"alpha"` key): `probs` is extracted for metrics and logging; mean uncertainty `edl/{prefix}_uncertainty` is logged automatically at each step. The change is backward-compatible — non-EDL models are unaffected.
- Added `EvidentialWarmupCallback` (`custom_callbacks/edl_callbacks.py`): three-phase encoder freeze schedule for fine-tuning. Phase 1 (`epoch < warmup_epochs`): encoder frozen. Phase 2 (`warmup_epochs ≤ epoch < partial_unfreeze_epoch`): last two encoder stages unfrozen. Phase 3 (`epoch ≥ partial_unfreeze_epoch`): full encoder unfrozen. Setting `freeze_encoder=False` (training from scratch) disables all freezing while preserving the logging hooks.
- Added `EvidentialUncertaintyVisualizationCallback` (`custom_callbacks/edl_callbacks.py`): logs a 4-column diagnostic grid `[input | predicted class | uncertainty map | ground truth]` every N validation epochs. Compatible with TensorBoard, WandB, and plain file-system fallback.
- Added `EvidentialInferenceProcessor` (`tools/inference/edl_inference_processor.py`): extends `SingleImageInfereceProcessor` with separate `TileMerger` instances for probabilities and uncertainty. Exports a single-band float32 GeoTIFF of the uncertainty map (u = K/S) via `save_uncertainty_raster`, preserving source CRS and transform exactly. Optional alpha band export via `export_alpha=True`.
- Added `edl_config.py` (`config_definitions/edl_config.py`): Hydra dataclass configs for `EvidentialWrapper`, `EvidentialMSELoss`, `EvidentialKLLoss`, `EvidentialWarmupCallback`, `EvidentialUncertaintyVisualizationCallback`, and `EvidentialInferenceProcessor`, all registered in the ConfigStore.
- Added reference YAML configs `edl_from_scratch.yaml` and `edl_finetune.yaml` under `conf/examples/` with inline comments explaining every field.

## RandomCropSegmentationDataset

- Added `RandomCropSegmentationDataset`: reads large GeoTIFF images on-the-fly using rasterio windowed reads instead of pre-generating tiles on disk. Eliminates the disk-space overhead of a tile library and allows crop size, augmentation, and sampling strategy to be changed without reprocessing data.
- Per-worker LRU cache (`_RasterioLRUCache`) keeps a configurable number of open `DatasetReader` handles (`lru_cache_size`, default `64`) to avoid repeated rasterio open/close overhead.
- `class_balanced_sampling`: weights image selection by inverse class frequency so images containing rare classes are sampled more often. Computed once at dataset initialisation from mask histograms in the CSV.
- Class-aware `CutMix` (`cutmix_prob`, `cutmix_alpha`): pastes a rectangular region from a second crop chosen to maximise class diversity.
- `ClassMix` (`classmix_prob`): copies a randomly selected class region from a second image and pastes it onto the primary crop; particularly effective for rare classes.
- `soft_labels` mode: returns float masks in `[0, 1]` for label-noise and probabilistic annotation workflows. The `_shared_step` training loop detects soft labels automatically and uses the appropriate loss path.
- `grid_mode` / `grid_step`: switches from random crop positions to a deterministic sliding-window grid for reproducible validation coverage and pseudo-labelling. `configure_optimizers` accounts for grid mode when computing `steps_per_epoch` for `OneCycleLR`.
- Added `RandomCropSegmentationDatasetConfig` dataclass.

## Mixture of Experts Models

- Added `UPerNetMoE` (`custom_models/upernet_moe.py`): UPerNet variant that replaces fusion and/or FPN convolutions with `MoEConv2dReLU` blocks. Supports `token_choice` (each token picks top-k experts) and `expert_choice` (each expert picks top-k tokens) routing, configurable noise injection, capacity factor, and an optional shared dense expert. Load-balancing auxiliary loss is automatically detected and added to the training loss in `_shared_step`.
- Added `UPerNetMEDoE` (`custom_models/upernet_medoe.py`): extends `UPerNetMoE` with structured expert dropout during training (randomly drops a fraction of experts per forward pass) to improve regularisation and reduce reliance on any single expert. `_shared_step` automatically logs `extra/train_medoe_expert_utilization` and `extra/train_medoe_expert_entropy` when a MEDoE model is detected.

## Dual-Head Training

- Added `UPerNetDualHead` (`custom_models/upernet_dual_head.py`): UPerNet with two independent decoders sharing a single encoder. Head A is supervised with hard labels (integer class indices); Head B is supervised with soft labels (float probabilities). A consistency loss couples the two heads during training. `inference_head` controls which head is active at inference time: `"A"`, `"B"`, or `"average"` (default).

## TTA improvements

- Added `tta_mode` compact interface: passing `tta_mode: "d4"` or `tta_mode: "flip"` to any inference processor or to `test_step` automatically selects the corresponding augmentation preset (`d4` = all 8 D4 symmetries; `flip` = 4 flip/rotation combinations), without listing augmentations explicitly.
- `SingleImageInfereceProcessor` now accepts `tta_mode` as an alias for `use_tta=True` + the corresponding augmentations list. Backward-compatible with the existing `use_tta` + `tta_augmentations` interface.
- `MultiClassInferenceProcessor` now accepts `tta_mode` (replaces the previous `use_tta` parameter on that class).
- `_get_tta_augmentations()` in `Model` checks `cfg.tta_mode` first; falls back to `cfg.use_tta` + `cfg.tta_augmentations` for backward compatibility.

## MultiClassInferenceProcessor improvements

- Added striped inference (`make_inference_striped`): very large images are automatically split into horizontal stripes when pixel count exceeds `striped_threshold_pixels` (default 50 MP), processed in parallel via `ThreadPoolExecutor`, and reassembled in-memory. Stripe height is configurable via `stripe_height` (default 4096 px).
- Added confidence map output: `confidence_mode` (`"max_prob"` or `"entropy"`) computes per-pixel confidence scores alongside the class prediction. Saved to `output_probs_dir` when provided.
- Added `tile_weight` parameter (passed through to `AbstractInferenceProcessor`) to control how overlapping tile predictions are merged.
- `process()` override: automatically routes each image to the striped or standard inference path based on image size.

## Callbacks

- Added `EMACallback`: maintains an exponential moving average of model weights (configurable `decay`). During validation the shadow EMA weights are swapped in so validation metrics reflect the averaged model; the original weights are restored immediately after. Checkpoints saved during a validation epoch contain the EMA weights.
- Added `MixStyleCallback`: applies MixStyle feature-level domain augmentation via forward hooks on selected encoder stages. `stages` controls which encoder stage indices receive the hook; `p` and `alpha` control application probability and Beta distribution parameter respectively.

## LR Warmup

- Added `warmup_epochs` support in `hyperparameters`: a linear LR warmup is prepended to any scheduler (except `OneCycleLR`, which has its own warmup via `pct_start`). The framework automatically subtracts `warmup_epochs` from `T_max` (or equivalent period parameters) so the total scheduled duration remains correct.

## Inference pipeline

- `predict_from_batch.py` updated: when `inference_processor` is present in the config, batch prediction is routed through `instantiate_inference_processor`, enabling sliding-window, striped, and TTA inference modes. The legacy `trainer.predict()` path is preserved for backward compatibility.
- `InferenceProcessorConfig` and `PredictSingleImageConfig` updated with new fields: `tta_mode`, `tile_weight`, `confidence_mode`, `striped_threshold_pixels`, `stripe_height`, `output_probs_dir`.

## Domain Adaptation — DANN with Gradient Reversal

- Added `GradientReversalFunction` and `GradientReversalLayer` (`domain_adaptation/methods/gradient_reversal.py`): a `torch.autograd.Function` implementing the identity forward pass with gradient negation in backward, wrapped in a parameter-free `nn.Module` with a `set_lambda()` method.
- Added `DomainClassifier` (`domain_adaptation/methods/dann.py`): resolution-agnostic MLP domain classifier using `AdaptiveAvgPool2d(1)` so it works with any encoder spatial size.
- Added `DANNMethod` (`domain_adaptation/methods/dann.py`): full DANN implementation via `BaseDomainAdaptationMethod`. Features: `requires_features=True`, configurable `feature_layer`, dedicated `discriminator_lr` parameter group, and GRL lambda initialized to 0 to avoid adversarial pressure at epoch 0.
- Added `step_mode` parameter to `DANNMethod` (`"epoch"` or `"batch"`): controls the granularity at which the lambda schedule is applied. `"batch"` mode updates λ every training step via the new `on_train_batch_start` hook, producing smoother growth closer to the original Ganin et al. implementation; `"epoch"` (default) updates once per epoch.
- Added `_current_lambda` attribute to `DANNMethod`: caches the most recently computed lambda so `DomainAdaptationModel._get_lambda_da()` always returns the correct value regardless of update granularity.
- Added `on_train_batch_start` lifecycle hook to `BaseDomainAdaptationMethod` (no-op default) and forwarded it in `DomainAdaptationModel.on_train_batch_start`.
- Updated `DomainAdaptationModel._get_lambda_da()` to prefer `method._current_lambda` when present, falling back to the epoch-level schedule query.
- Refactored `BaseLambdaScheduler.get_lambda` signature from `(epoch, total_epochs)` to `(step, total_steps)` — the same formula, now granularity-agnostic. All scheduler subclasses (`ConstantScheduler`, `LinearScheduler`, `DANNScheduler`) updated accordingly.
- Added full working example config (`conf/examples/dann_domain_adaptation.yaml`) for U-Net ResNet-34 with DANN.
- Added test suite for GRL (`tests/test_gradient_reversal.py`, 17 tests) and DANN (`tests/test_dann_method.py`, 49 tests including 12 new `TestDANNMethodStepMode` tests).
- Added dedicated DANN user guide (`website/docs/advanced/dann-method.md`) covering mechanism, `in_channels` lookup table, full config reference, `step_mode` guidance, monitoring, and limitations.
- Updated `website/docs/advanced/domain-adaptation.md` with a "Built-in Methods" section linking to the DANN guide.
- Updated `website/docs/advanced/domain-adaptation-implementing-methods.md` Example 2 to use the built-in `GradientReversalLayer` and add a callout pointing to the dedicated DANN guide.

## Domain Adaptation — Initial structure

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
