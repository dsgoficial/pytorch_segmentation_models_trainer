---
sidebar_position: 11
title: Evaluation Pipeline
---

# Evaluation Pipeline

The evaluation pipeline lets you compare the segmentation quality of one or more trained models side-by-side on a shared test dataset. It runs predictions for each experiment, computes pixel-level metrics from confusion matrices, aggregates results across images, and optionally generates comparison visualizations.

---

## When to Use It

Use the evaluation pipeline when you want to:
- Compare multiple model architectures or training runs on the same test set.
- Compute IoU, F1, precision, and recall across all test images.
- Generate per-image metric CSVs and aggregated JSON reports.
- Produce confusion matrix plots for a qualitative view of class-level errors.

---

## Running the Pipeline

```bash
python pytorch_segmentation_models_trainer/evaluate_experiments.py \
  --config-dir ./configs/evaluation \
  --config-name pipeline_config
```

Override parameters without editing YAML:

```bash
python evaluate_experiments.py \
  experiments[0].checkpoint_path=/new/model.ckpt \
  pipeline_options.parallel_inference.enabled=false
```

Build the evaluation CSV automatically from image and mask folders:

```bash
python evaluate_experiments.py \
  evaluation_dataset.build_csv_from_folders.enabled=true \
  evaluation_dataset.build_csv_from_folders.images_folder=/data/images \
  evaluation_dataset.build_csv_from_folders.masks_folder=/data/masks
```

---

## Pipeline Execution Steps

The `EvaluationPipeline` runs the following steps in order:

1. **Prepare dataset** — build or load the ground-truth CSV.
2. **Run predictions** — call `predict.py` as a subprocess for each experiment (sequential or parallel across GPUs).
3. **Calculate metrics** — compute confusion matrices and derive per-image and aggregated metrics.
4. **Aggregate results** — combine per-experiment results into a unified structure.
5. **Generate visualizations** — confusion matrix plots and comparison charts (if enabled).
6. **Save summary report** — write a JSON summary to `output.base_dir`.

---

## Config Structure

### Dataset Configuration

Three modes are supported for specifying the evaluation dataset:

**Mode 1 — Existing CSV:**
```yaml
evaluation_dataset:
  input_csv_path: /data/test_dataset.csv   # CSV with 'image' and 'mask' columns
  build_csv_from_folders:
    enabled: false
```

**Mode 2 — Build CSV from image/mask folders:**
```yaml
evaluation_dataset:
  build_csv_from_folders:
    enabled: true
    images_folder: /data/test/images/
    masks_folder: /data/test/masks/
```

**Mode 3 — Direct folder evaluation:**
```yaml
evaluation_dataset:
  direct_folder_evaluation:
    enabled: true
    ground_truth_folder: /data/gt_masks/
    predictions_folder: /data/predictions/   # optional pre-computed predictions
    gt_pattern: "*.tif"
    pred_pattern: "*.tif"
    force_rebuild: false
```

---

### Experiments List

Each entry in `experiments` defines one model run. The pipeline calls `predict.py` for each, saving outputs to `output_folder`.

```yaml
experiments:
  - name: unet_resnet34
    checkpoint_path: /checkpoints/unet_resnet34_best.ckpt
    predict_config: /configs/predict_unet.yaml
    output_folder: /output/eval/unet_resnet34/predictions/
    overrides:             # optional Hydra overrides for this experiment
      inference_threshold: 0.45

  - name: deeplab_resnet50
    checkpoint_path: /checkpoints/deeplab_resnet50_best.ckpt
    predict_config: /configs/predict_deeplab.yaml
    output_folder: /output/eval/deeplab_resnet50/predictions/
```

:::note predict_config
The `predict_config` path must point to a valid predict config YAML (see the [Running Inference](./inference) guide). The pipeline invokes it as a subprocess, passing `checkpoint_path` and `device` overrides automatically.
:::

---

### Metrics Configuration

```yaml
metrics:
  num_classes: 2
  class_names:
    - background
    - building
  segmentation_metrics: []   # reserved for future torchmetrics overrides
```

All metrics are computed from accumulated confusion matrices. The following are reported for each experiment at both the per-image and aggregated levels:

| Metric | Key | Description |
|---|---|---|
| Accuracy | `Accuracy` | Macro-averaged per-class accuracy |
| IoU / Jaccard | `JaccardIndex` | Macro-averaged intersection over union |
| F1 Score | `F1Score` | Macro-averaged Dice / F1 coefficient |
| Precision | `Precision` | Macro-averaged precision |
| Recall | `Recall` | Macro-averaged recall |

Per-class metrics are also reported as `IoU_{class_name}`, `F1_{class_name}`, `Precision_{class_name}`, `Recall_{class_name}`, and `Accuracy_{class_name}`.

---

### Pipeline Options

```yaml
pipeline_options:
  skip_existing_predictions: true   # skip experiments whose output_folder already has .tif files
  skip_existing_evaluations: false  # skip metric calculation if results already exist

  parallel_inference:
    enabled: false                  # set true to run experiments in parallel across GPUs
    strategy: one_experiment_per_gpu   # or "sequential"
    gpus: null                      # null = auto-detect; or [0, 1, 2] for specific GPUs

  parallel_image_processing:
    enabled: true                   # process images within each experiment in parallel
    num_workers: null               # null = os.cpu_count()
```

---

### Output Configuration

```yaml
output:
  base_dir: /output/evaluation/
  timestamp_folders: false
  structure:
    experiments_folder: experiments
    comparison_folder: comparisons
  files:
    summary_report: summary_report.json
    per_image_metrics_pattern: "{experiment_name}_per_image_metrics.csv"
    confusion_matrix_data_pattern: "{experiment_name}_confusion_matrix.npy"
```

---

### Visualization Configuration

```yaml
visualization:
  comparison_plots:
    enabled: true
  confusion_matrix:
    save_individual: true    # one plot per experiment
    save_comparison: true    # side-by-side grid of all experiments
```

---

## GPUDistributor: Parallel Inference

When `pipeline_options.parallel_inference.enabled: true`, the `GPUDistributor` class allocates experiments to available GPUs.

### Strategies

| Strategy | Behavior |
|---|---|
| `one_experiment_per_gpu` | Round-robin assignment: experiments are distributed cyclically across available GPUs |
| `sequential` | All experiments run on the first available GPU or CPU |

The `GPUDistributor` auto-detects GPUs using PyTorch. Any GPU with less than 1 GB of free memory is excluded. You can pin specific GPUs with `pipeline_options.parallel_inference.gpus: [0, 2]`.

```yaml
pipeline_options:
  parallel_inference:
    enabled: true
    strategy: one_experiment_per_gpu
    gpus: [0, 1]   # use only GPU 0 and GPU 1
```

:::warning Parallel Inference Memory
Each parallel experiment loads a separate copy of the model into its assigned GPU. Ensure each GPU has sufficient memory for the model plus the inference batch before enabling parallel mode.
:::

---

## Pre-computed Predictions

If you already have prediction rasters and only want to run metric calculation, use `load_predictions_from_folder`:

```yaml
pipeline_options:
  load_predictions_from_folder:
    enabled: true
    base_folder: /output/predictions/   # each experiment gets a subfolder named by experiment.name
```

Or point each experiment to its own folder:

```yaml
experiments:
  - name: unet_resnet34
    checkpoint_path: /checkpoints/unet_resnet34_best.ckpt
    predict_config: /configs/predict_unet.yaml
    output_folder: /output/eval/unet_resnet34/predictions/
    precomputed_predictions_folder: /precomputed/unet_resnet34/
```

---

## Prediction-to-Ground-Truth Matching

The `MetricsCalculator` matches prediction files to ground-truth masks by filename. Matching uses these strategies in priority order:

1. **Exact stem match** — `image001.tif` matches `image001.tif`.
2. **Strip common prefixes** — strips `mask_`, `gt_` from the ground-truth stem before matching.
3. **Substring match** — if the cleaned GT stem is a substring of any prediction stem (or vice versa).

Ground-truth rasters and prediction rasters are spatially aligned using rasterio bounding-box intersection before metric calculation. This handles cases where prediction and GT tiles have slightly different extents.

---

## Output Files

After a successful run, the output directory contains:

```
/output/evaluation/
├── summary_report.json                          # full pipeline summary
├── experiments/
│   ├── unet_resnet34/
│   │   └── metrics/
│   │       ├── unet_resnet34_per_image_metrics.csv
│   │       ├── aggregated_metrics.json
│   │       └── unet_resnet34_confusion_matrix.npy
│   └── deeplab_resnet50/
│       └── metrics/
│           ├── deeplab_resnet50_per_image_metrics.csv
│           ├── aggregated_metrics.json
│           └── deeplab_resnet50_confusion_matrix.npy
└── comparisons/
    ├── confusion_matrices/
    │   ├── unet_resnet34_confusion_matrix.png
    │   └── deeplab_resnet50_confusion_matrix.png
    └── confusion_matrix_comparison.png
```

---

## Full Example Config

```yaml title="configs/evaluation/pipeline_config.yaml"
# ── Dataset ───────────────────────────────────────────────────────────────────
evaluation_dataset:
  input_csv_path: /data/test_dataset.csv
  build_csv_from_folders:
    enabled: false

# ── Experiments ───────────────────────────────────────────────────────────────
experiments:
  - name: unet_resnet34
    checkpoint_path: /checkpoints/unet_resnet34_best.ckpt
    predict_config: /configs/predict_unet.yaml
    output_folder: /output/eval/unet_resnet34/predictions/
    overrides:
      inference_threshold: 0.5

  - name: unet_resnet50
    checkpoint_path: /checkpoints/unet_resnet50_best.ckpt
    predict_config: /configs/predict_unet.yaml
    output_folder: /output/eval/unet_resnet50/predictions/
    overrides:
      inference_threshold: 0.5

# ── Metrics ───────────────────────────────────────────────────────────────────
metrics:
  num_classes: 2
  class_names:
    - background
    - building
  segmentation_metrics: []

# ── Pipeline Options ──────────────────────────────────────────────────────────
pipeline_options:
  skip_existing_predictions: true
  skip_existing_evaluations: false
  parallel_inference:
    enabled: false
    strategy: one_experiment_per_gpu
    gpus: null
  parallel_image_processing:
    enabled: true
    num_workers: 8

# ── Output ────────────────────────────────────────────────────────────────────
output:
  base_dir: /output/evaluation/
  timestamp_folders: false
  structure:
    experiments_folder: experiments
    comparison_folder: comparisons
  files:
    summary_report: summary_report.json
    per_image_metrics_pattern: "{experiment_name}_per_image_metrics.csv"
    confusion_matrix_data_pattern: "{experiment_name}_confusion_matrix.npy"

# ── Visualization ─────────────────────────────────────────────────────────────
visualization:
  comparison_plots:
    enabled: true
  confusion_matrix:
    save_individual: true
    save_comparison: true

# ── Logging ───────────────────────────────────────────────────────────────────
logging:
  level: INFO
  save_to_file: true
  log_file: evaluation.log
```
