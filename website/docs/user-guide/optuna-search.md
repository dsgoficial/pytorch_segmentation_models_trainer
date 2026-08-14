---
sidebar_position: 15
title: Optuna Hyperparameter Search
---

# Optuna Hyperparameter Search

The framework integrates [Optuna](https://optuna.org/) for automated hyperparameter optimisation (HPO). When `experiments_runner.optuna_search` is configured, `ExperimentsRunner` delegates to `OptunaRunner`, which runs `n_trials` training experiments with Optuna-suggested hyperparameter values and saves study artefacts.

## Quick start

Add `optuna_search` to your `experiments_runner` block:

```yaml
experiments_runner:
  output_base_dir: outputs/hpo
  optuna_search:
    n_trials: 50
    metric: val/JaccardIndex
    direction: maximize
    sampler: TPE
    storage: sqlite:///outputs/hpo/study.db
    study_name: unet_hpo
    search_space:
      - key: optimizer.lr
        type: float
        low: 1.0e-5
        high: 1.0e-2
        log: true
      - key: train_dataset.batch_size
        type: int
        low: 8
        high: 64
```

## Search parameter types

| `type` | Required fields | Example |
|---|---|---|
| `float` | `low`, `high` | Learning rate, weight decay |
| `int` | `low`, `high` | Batch size, max epochs |
| `categorical` | `choices` | Encoder name, bool flag |
| `config_block` | `choices` (list of `name`/`values` dicts) | Loss function, scheduler |

### `float`

```yaml
- key: optimizer.lr
  type: float
  low: 1.0e-5
  high: 1.0e-2
  log: true   # sample on log scale (recommended for LR)
```

### `int`

```yaml
- key: train_dataset.batch_size
  type: int
  low: 8
  high: 64
  step: 8   # optional step (default: 1)
```

### `categorical`

```yaml
- key: model.encoder_name
  type: categorical
  choices: [resnet34, resnet50, efficientnet-b0]
```

### `config_block`

Swaps an entire config subtree. Each choice has a `name` (used as the Optuna categorical value) and a `values` dict that is deep-merged into the config at `key`:

```yaml
- key: loss
  type: config_block
  choices:
    - name: cross_entropy
      values:
        _target_: torch.nn.CrossEntropyLoss
        weight: null
    - name: focal
      values:
        _target_: kornia.losses.FocalLoss
        alpha: 0.5
        gamma: 2.0
```

Use `config_block` any time the different options have heterogeneous parameters (different loss classes, different scheduler classes, etc.).

## Samplers

| `sampler` | Description |
|---|---|
| `TPE` (default) | Tree-structured Parzen Estimator — Bayesian, learns from previous trials |
| `GP` | Gaussian Process — best for small budgets and continuous spaces |
| `CmaES` | Evolution strategy — good for correlated continuous params |
| `Random` | Pure random — fast baseline, no learning |
| `Grid` | Exhaustive grid over categorical/int choices |

### Choosing a sampler

**Rule of thumb by trial budget:**

| Budget | Recommended sampler |
|---|---|
| < 20 trials | `GP` |
| 20–100 trials | `TPE` ← correct default for most cases |
| > 100 trials, all continuous params | `CmaES` |
| Small fixed space (< 50 combinations) | `Grid` |
| Debug / pipeline validation | `Random` |

**`TPE`** is the right default for segmentation experiments. It is Bayesian — each trial informs the next, focusing the search on promising regions. It handles a mix of `float`, `int`, `categorical`, and `config_block` params naturally.

**`GP`** is more sample-efficient than `TPE` when the budget is very small (< 20 trials). Its compute cost grows as O(n³) with the number of trials, so it becomes slow beyond ~50 trials.

**`CmaES`** (Covariance Matrix Adaptation Evolution Strategy) works best when all search params are continuous and correlated — for example, jointly tuning `lr`, `weight_decay`, `momentum`, and `dropout`. It needs ~50+ trials to converge and does not handle categorical params well. If your search space mixes continuous and categorical params, use `TPE` instead.

**`Grid`** does not learn from previous trials — it exhaustively evaluates all combinations. Use it only for small discrete spaces where you want guaranteed coverage, such as 3 loss functions × 4 encoders = 12 trials. With continuous params, `Grid` requires manual discretisation and the number of combinations explodes quickly.

**`Random`** provides a useful baseline: if `TPE` does not outperform `Random`, the search space is likely too large or the metric is too noisy to optimise reliably.

**For typical segmentation HPO** (LR + encoder + loss + batch size): use `TPE` with 30–50 trials. This is the best cost/quality trade-off without tuning the sampler itself.

```yaml
optuna_search:
  sampler: TPE      # safe default
  n_trials: 50      # 30–50 is usually enough to find a good region
```

## Mode B — seed loop after HPO

When `seeds` or `n_runs` is also configured, the runner executes a standard seed loop with the best trial's config after the HPO study finishes:

```yaml
experiments_runner:
  output_base_dir: outputs/hpo
  seeds: [42, 101, 28]          # seed loop runs AFTER optuna finishes
  representative_metric: val/JaccardIndex
  optuna_search:
    n_trials: 50
    metric: val/JaccardIndex
    ...
```

## Resuming an interrupted study

Set `storage` to a SQLite path. On restart, the study is automatically resumed from where it stopped:

```yaml
optuna_search:
  storage: sqlite:///outputs/hpo/study.db
  study_name: unet_hpo   # must match the original study name
```

## Output files

After the study, `output_base_dir` contains:

| File | Contents |
|---|---|
| `best_trial_config.yaml` | Full training config with best trial's hyperparameters applied |
| `trial_summary.csv` | One row per trial: trial number, state, objective value, duration, all HP values |
| `param_importances.json` | fANOVA importance scores (requires ≥ 2 completed trials) |
| `plots/optimization_history.html` | Objective value vs. trial number |
| `plots/param_importances.html` | Bar chart of HP importances |
| `plots/parallel_coordinates.html` | High-dimensional HP relationship view |
| `plots/contour.html` | Contour plot of the two most important HPs |

Visualisation files require [plotly](https://plotly.com/python/) (`pip install plotly`).

## Full example

See `conf/examples/optuna_search.yaml` for a complete config including all four search param types.
