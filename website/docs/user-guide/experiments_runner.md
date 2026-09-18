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
| `save_summary` | `bool` | no | `true` | Update `summary.csv` after every completed run. |
| `summary_metrics` | `list[str]` | no | `[val/loss]` | Metric keys logged to the run summary table. |
| `resume` | `bool` | no | `true` | Skip already-completed runs on restart using `runner_state.json`. Runs are idempotent by default — re-launching the same config never repeats finished work. Set to `false` to always start every run fresh. |
| `overwrite` | `bool \| list[int]` | no | `false` | Forces re-execution of runs `resume` would otherwise skip. `true` forces every run; a list forces only those `run_idx` values. Deletes each forced run's previous output directory first, and replaces (not duplicates) its `runner_state.json` / `summary.csv` entry. |

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
├── run_00_seed42/        ← Lightning checkpoints & logs
│   └── lightning_logs/
├── run_01_seed101/
│   └── lightning_logs/
├── run_02_seed28/
│   └── lightning_logs/
├── runner_state.json     ← written after each run; drives resume
└── summary.csv           ← updated after each run
```

The seed is embedded in every directory name so you can identify an experiment directly from the filesystem without consulting `summary.csv`.

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
cryptographically random 31-bit seeds at runtime.  They are saved in
`runner_state.json` immediately so `resume: true` always uses the same seeds:

```yaml
experiments_runner:
  n_runs: 5
  output_base_dir: outputs/random_study
  save_summary: true
  resume: false
```

To replay an individual run, read its seed from the directory name
(`run_02_seed1084739421/`) or from `summary.csv`:

```yaml
mode: train
seed: 1084739421
pl_trainer:
  default_root_dir: outputs/random_study/run_02_seed1084739421_replay
# ... rest of config unchanged ...
```

---

## Resuming an interrupted run sequence

Runs are idempotent by default (`resume: true`) — if training is
interrupted between runs, just re-launch the same command:

```yaml
experiments_runner:
  seeds: [42, 101, 28]
  output_base_dir: outputs/my_study
  # resume: true is the default — no need to set it explicitly
```

The runner reads `runner_state.json`, identifies which runs already have
results, and starts from the first pending run.  For within-run resumption
(interrupted mid-epoch), configure PyTorch Lightning's `ModelCheckpoint`
callback and set `resume_from_checkpoint` in `hyperparameters` as usual.

A run whose `output_dir` exists but is **not** recorded as complete (e.g. it
crashed mid-training) has that leftover directory wiped before it is
retried, so stale checkpoints from the failed attempt never mix with the
new one.

Set `resume: false` to disable this entirely and always start every run
fresh, ignoring any existing state.

---

## Forcing specific runs to redo (`overwrite`)

`resume` only skips runs that are missing or incomplete. To force a
already-completed run to redo — for example, run 1 (seed 101) finished
but its checkpoint turned out corrupted — use `overwrite`:

```yaml
experiments_runner:
  seeds: [42, 101, 28]
  output_base_dir: outputs/my_study
  resume: true
  overwrite: [1]          # redo only run_idx 1; runs 0 and 2 stay skipped
```

`overwrite: true` forces every completed run to redo (equivalent to
deleting `runner_state.json` and starting over, but it also cleans each
run's old output directory first — a plain `resume: false` restart does
not). Each forced run's stale entry in `runner_state.json` and
`summary.csv` is replaced by the new result, not duplicated.

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

## Running a sequence of different experiments

The Experiments Runner repeats **one** config with different seeds. To run
a **sequence of different steps** (data-prep tools, different
models/losses/datasets, different baselines) unattended on a remote
server, see the [Pipeline](pipeline.md) — each of its steps can itself be
an Experiments Runner config.

---

## Full example config

See [`conf/examples/experiments_runner.yaml`](https://github.com/dsgoficial/pytorch_segmentation_models_trainer/blob/main/pytorch_segmentation_models_trainer/conf/examples/experiments_runner.yaml)
for a complete working example with a UNet / ResNet-34 backbone.
