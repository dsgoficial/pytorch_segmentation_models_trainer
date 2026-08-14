---
sidebar_position: 26
title: Sampler Weights
---

# Sampler Weights

`compute-sampler-weights` computes per-patch sampler weights from class proportion columns
in a coreset CSV, enabling **Weighted Random Sampling** (WRS) during training to counteract
class imbalance.

---

## Formulas

Two formulas are supported:

### `sqrt_inverse_freq` (recommended)

```
ω_i = Σ_c sqrt(p_{i,c} / f_c)
```

where `p_{i,c}` is the proportion of class `c` in patch `i`, and `f_c = mean_i(p_{i,c})` is the
global class frequency. The square root provides **smoother upweighting** of rare classes compared
to linear inverse frequency — rare classes receive higher weights but extreme outliers are
dampened. This formula was validated in the land-cover segmentation article (best mIoU = 0.554).

### `inverse_freq`

```
ω_i = Σ_c p_{i,c} / f_c
```

Stronger upweighting — rare classes can dominate sampling if not clipped downstream.

Zero-frequency classes (all-zero column) contribute nothing to the weight (treated as infinite
frequency).

---

## CLI usage

```bash
pytorch-smt-tools compute-sampler-weights path/to/config.yaml
```

---

## Configuration

```yaml
input_csv_path: /data/coreset.csv
output_csv_path: /data/coreset_weighted.csv

# Formula: "sqrt_inverse_freq" (recommended) or "inverse_freq"
formula: sqrt_inverse_freq

# Source of class proportions:
# "topographic_vector_source" — reads c0..c{num_classes-1} columns from the CSV
source: topographic_vector_source

num_classes: 6
weight_column: sampler_weight

# Optional: column name for an exclusion flag (boolean; excluded patches get weight 0)
# excluded_column: excluded
```

The `source: topographic_vector_source` mode expects columns `c0`, `c1`, ..., `c{num_classes-1}`
in the input CSV, each holding the per-patch proportion of that class (values in [0, 1], summing
approximately to 1 per row).

---

## Output

The output CSV contains all input columns plus `sampler_weight`. Excluded patches (if
`excluded_column` is set) receive weight 0.0. All other patches receive a positive weight
proportional to their rare-class content.

---

## Python API

```python
import numpy as np
from pytorch_segmentation_models_trainer.tools.sampling.weight_calculator import (
    compute_class_weights_from_proportions,
)

# props: (N, C) array of class proportions
props = df[["c0", "c1", "c2", "c3", "c4", "c5"]].values
weights = compute_class_weights_from_proportions(props, formula="sqrt_inverse_freq")

df["sampler_weight"] = weights
```

---

## Using weights in a DataLoader

```python
import torch
from torch.utils.data import WeightedRandomSampler

weights = df["sampler_weight"].values
sampler = WeightedRandomSampler(
    weights=torch.tensor(weights, dtype=torch.double),
    num_samples=len(weights),
    replacement=True,
)
loader = DataLoader(dataset, sampler=sampler, batch_size=16)
```

---

## Relationship to `compute_sampler_weights` in CoreSetSelector

The [`CoreSetSelector.compute_sampler_weights`](./coreset-selection.md) method provides a
complementary approach that reads class distributions from a JSON column (`class_dist_json`) and
normalises weights relative to their mean — useful after `select-coreset` when the JSON column
is available. The `compute-sampler-weights` CLI is designed for the simpler case where class
proportions are already stored as numeric columns (`c0`–`c5`), which is the typical output of
the balanced-dataset pipeline.
