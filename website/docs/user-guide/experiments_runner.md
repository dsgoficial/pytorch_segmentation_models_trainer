---
sidebar_position: 13
title: Experiments Runner
---

# Experiments Runner

The Experiments Runner lets you repeat a training configuration multiple times
with different random seeds — in series — and automatically aggregates the
results into a single CSV file.  This is the recommended workflow for
**reproducibility studies**, **variance estimation**, and **ablation
comparisons**.

---

## Quick start

```yaml
# my_experiment.yaml
mode: run-experiments

experiments_runner:
  seeds: [42, 101, 28]          # one run per seed
  output_base_dir: outputs/my_study
  save_summary: true
  summary_metrics:
    - val/loss
    - val/F1Score

# ... rest of the config is identical to a regular training config ...
backbone:
  name: resnet34
  input_width: 256
  input_height: 256

hyperparameters:
  model_name: unet_resnet34
  backbone: resnet34
  batch_size: 8
  epochs: 50
  max_lr: 1e-3
  classes: 1
# ...
```

```bash
pytorch-smt --config-path . --config-name my_experiment
```

---

## `experiments_runner` block reference

| Field | Type | Required | Default | Description |
|---|---|---|---|---|
| `seeds` | `list[int]` | one of seeds/n_runs | — | Explicit seed list.  Determines the number of runs. |
| `n_runs` | `int` | one of seeds/n_runs | — | Number of runs with auto-generated seeds.  Required when `seeds` is absent. |
| `output_base_dir` | `str` | no | `outputs/experiments_runner` | Root directory for per-run outputs. |
| `save_summary` | `bool` | no | `true` | Write `summary.csv` after all runs. |
| `summary_metrics` | `list[str]` | no | `[val/loss]` | Metric keys logged to the run summary table. |

### Seeds vs n\_runs

| Configuration | Result |
|---|---|
| `seeds: [42, 101, 28]` | 3 runs with seeds 42, 101, 28 |
| `n_runs: 5` | 5 runs with cryptographically random seeds |
| `seeds: [42, 101, 28]`, `n_runs: 3` | 3 runs (consistent — accepted) |
| `seeds: [42, 101, 28]`, `n_runs: 1` | **Validation error** (conflict) |

---

## Output layout

```
outputs/my_study/
├── run_00/          ← Lightning checkpoints & logs (seed 42)
│   └── ...
├── run_01/          ← Lightning checkpoints & logs (seed 101)
│   └── ...
├── run_02/          ← Lightning checkpoints & logs (seed 28)
│   └── ...
└── summary.csv
```

### summary.csv format

```
run,seed,duration_s,train/loss,val/loss,val/F1Score
0,42,142.30,0.210000,0.340000,0.820000
1,101,139.80,0.190000,0.330000,0.825000
2,28,141.10,0.200000,0.350000,0.818000
mean,-,141.07,0.200000,0.340000,0.821000
std,-,1.26,0.010000,0.010000,0.003606
```

Every metric logged by PyTorch Lightning (`train/*`, `val/*`, `test/*`) is
included automatically.  The `summary_metrics` field only controls which
metrics appear in the run-level log output — the CSV always contains all
available metrics.

---

## Using random seeds

When `seeds` is omitted and only `n_runs` is specified, the runner generates
cryptographically random 31-bit seeds at runtime.  They are stored in
`summary.csv` so you can reproduce any individual run later:

```yaml
experiments_runner:
  n_runs: 5
  output_base_dir: outputs/random_study
  save_summary: true
```

To replay run 2 (seed from `summary.csv`, e.g. 1084739421):

```yaml
mode: train
seed: 1084739421
pl_trainer:
  default_root_dir: outputs/random_study/run_02_replay
# ... rest of config unchanged ...
```

---

## Test dataset

If your config contains a `test_dataset` block, PyTorch Lightning's
`trainer.test()` is called at the end of each run and the `test/*` metrics are
captured in `summary.csv` automatically — no extra configuration required.

---

## Relation to Reproducible Training

The Experiments Runner builds on the [Reproducible Training](reproducibility.md)
feature.  Each run receives its seed through the same `set_training_seed()`
mechanism (seeding Python `random`, NumPy, PyTorch CPU/CUDA, and DataLoader
workers).  You can also set `deterministic_cudnn: true` in the config for fully
deterministic GPU ops at the cost of throughput.

---

## Full example config

See [`conf/examples/experiments_runner.yaml`](https://github.com/dsgoficial/pytorch_segmentation_models_trainer/blob/main/pytorch_segmentation_models_trainer/conf/examples/experiments_runner.yaml)
for a complete working example with a UNet / ResNet-34 backbone.
