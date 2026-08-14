---
sidebar_position: 7
title: CoreSet Selection
---

# CoreSet Selection

After generating a balanced dataset CSV with `build-balanced-dataset`, the pool may still be
too large to train within a compute budget. CoreSet selection identifies the most informative
subset — filtering out redundant, homogeneous, and noisy patches — so a model trained on 25–50%
of the pool can match or exceed full-dataset performance.

> **Reference:** Nogueira et al. (2026), *Core-Set Selection for Data-Efficient Land Cover
> Segmentation*, IEEE Access, DOI [10.1109/ACCESS.2026.3659734](https://doi.org/10.1109/ACCESS.2026.3659734).

---

## Overview

The paper introduces six **model-agnostic** methods evaluated on three high-resolution datasets
(DFC2022, Vaihingen, Potsdam) using U-Net (ResNet-18 backbone) and SegFormer. All six methods
consistently outperform random sampling and the greedy CoreSet baseline across all datasets and
architectures.

**Key finding:** On DFC2022 (noisy labels), a 25% subset selected by the best methods yields
*higher* SegFormer mIoU than training on 100% of the data. On the cleaner Vaihingen and Potsdam
datasets, 75% suffices to surpass the full-dataset baseline. The computational overhead of
coreset selection itself is negligible — approximately one forward pass over the full pool.

---

## Six Selection Methods

The methods fall into three categories based on what they analyse:

| Category | Method | Key | Requires |
|---|---|---|---|
| **Label-based** | Label Complexity | `lc` | `class_entropy` column |
| **Label-based** | Class Balance | `cb` | `class_dist_json` column |
| **Image-based** | Feature Activation | `fa` | embeddings |
| **Image-based** | Feature Space Diversity | `fd` | embeddings |
| **Hybrid** | LC/FD hybrid | `lc_fd` | both |
| **Hybrid** | FA/CB hybrid | `fa_cb` | both |

Label-based methods (`lc`, `cb`) run in seconds on 100k+ patches — no embedding I/O needed.
Image-based methods (`fa`, `fd`, `lc_fd`, `fa_cb`) require pre-computed embeddings (see
[Embedding Sources](#embedding-sources)).

---

## Method Reference

### A. Label Complexity (LC) — Label-based

Scores each patch by the **Shannon entropy of its class distribution**. Patches with many
mixed classes score high; patches dominated by a single class score low.

```
H(M_i) = -Σ_c  p_{i,c} · log(p_{i,c})
s_i^LC = H(M_i) / max(H)
```

`p_{i,c}` is the proportion of pixels belonging to class `c` in mask `M_i`. Unknown/ignored
classes are excluded. Scores are normalised to [0, 1] by the maximum entropy in the pool.

**Use when:** embeddings are unavailable; want a fast, label-only ranking.

### B. Class Balance (CB) — Label-based

Greedy algorithm that at each step selects the patch that **maximally increases the global class
entropy** of the already-selected set. This ensures the selected subset approaches a uniform
class distribution, directly addressing class imbalance.

Given `r_i` = selection rank of patch `i` (first selected = highest rank):

```
s_i^CB = (N - r_i) / (N - 1)
```

The greedy loop is O(budget × N), not O(N²), because it early-stops after `budget_count`
selections. With `cb_use_spatial: true`, the tile centroid coordinates (min-max normalised)
are appended to the class-frequency vector, enforcing geographic spread alongside class balance.

**Use when:** class imbalance is the primary concern and embeddings are unavailable.

### C. Feature Space Diversity (FD) — Image-based

Selects a **visually diverse** set by clustering image embeddings with K-Means and assigning
scores by round-robin selection order across clusters. This guarantees that the selected set
covers distinct semantic visual groups.

K-selection: starting from K=2, K is incremented until the change in average per-cluster
**Vendi score** (a diversity metric) falls below δ=0.005 for three consecutive iterations.
Alternatively, the elbow method (default in the framework) can be used.

Scores: first patch selected by round-robin receives `s=1`; last receives `s=0`.

The paper uses ResNet-18 (ImageNet) feature vectors of dimension 512. The framework accepts any
pre-computed embedding (e.g. DINOv2), stored in the `embedding_column` of the CSV or loaded
from a Parquet file.

**Use when:** visual diversity is the primary concern; dataset has many redundant scenes.

### D. Feature Activation (FA) — Image-based

Scores patches by **embedding mean and standard deviation**, assuming that patches with high
activation magnitude *and* high embedding variance carry richer information.

The framework follows the paper's gamma-based FA score. It computes the raw embedding mean (µ)
and standard deviation (σ), then normalises the gamma values inversely so lower gamma means
higher FA score:

```
γ_i = -(1 - µ_i) · log(σ_i)
s_i^FA = 1 - (γ_i - min_j γ_j) / (max_j γ_j - min_j γ_j)
```

For numerical stability, the implementation evaluates `log(σ_i + eps)`, matching the reference
implementation's guard against `log(0)`.

**Use when:** want to prioritise visually complex, information-rich patches.

### E. LC/FD Hybrid — Hybrid

Two-stage selection combining diversity (for small subsets) and label complexity (for larger
ones). The paper motivates this by observing that FD outperforms LC at small budgets, while LC
adds value at medium to large budgets.

Algorithm:
1. Rank all patches by FD.
2. Rank all patches by LC.
3. Take the top-`m` patches from the FD ranking.
4. Fill remaining positions with LC-ranked patches not already included.

`lc_fd_cutoff_m` controls `m` (default: 10% of N). In the paper's DFC2022 experiments, m=770
(≈10% of the ~7700-patch pool) was found optimal.

**Use when:** budget spans a wide range and dataset may be noisy.

### F. FA/CB Hybrid — Hybrid

Linear combination of the FA and CB scores:

```
s_i^{FA/CB} = λ × s_i^FA + (1-λ) × s_i^CB
```

`fa_cb_lambda` controls λ (default 0.5 = equal weight, as used in the paper). Jointly exploits
label representativeness (CB) and visual richness (FA).

**Use when:** both label balance and visual diversity matter.

---

## Quick Start

### Step 1 — Generate balanced_dataset.csv

```bash
pytorch-smt-tools build-balanced-dataset conf/examples/balanced_dataset_local.yaml
```

### Step 2 — Run coreset selection

```bash
pytorch-smt-tools select-coreset conf/examples/coreset_local.yaml
```

The output CSV extends the input with two columns:

| Column | Type | Description |
|---|---|---|
| `coreset_score` | float [0, 1] | Higher = more informative |
| `coreset_selected` | bool | `True` for patches within budget |

---

## YAML Configuration

```yaml
# conf/examples/coreset_local.yaml
coreset:
  # Method: lc | cb | fa | fd | lc_fd | fa_cb
  method: cb

  # Budget: fraction [0, 1] of the pool to select
  budget: 0.4
  budget_mode: fraction   # "fraction" | "count"

  input_csv_path: /path/to/balanced_dataset.csv
  output_csv_path: /path/to/coreset.csv

  # Required for fa, fd, lc_fd, fa_cb — leave null for lc/cb
  embedding_column: null  # e.g. "embedding"

  # Parquet file from embedding_extractor.py.
  # When set and embedding_column absent from CSV, merged automatically.
  parquet_path: null

  # GPU acceleration: null = auto (cuda if available), "cpu" = force CPU
  device: null

  # Spatial diversity: append (lat_norm, lon_norm) to class-freq vector (CB)
  cb_use_spatial: false

  # Spatial diversity: append (lat_norm, lon_norm) to embeddings (FA/FD/LC-FD/FA-CB)
  fd_use_spatial: false

  # Pool filtering: include homogeneous patches (has_low_entropy=True)?
  exclude_low_entropy: true

  # FD / LC-FD K-selection
  fd_k_min: 2
  fd_k_max: 20
  fd_use_vendi: false      # true = Vendi score; false = elbow method

  # LC/FD hybrid: top-m FD patches before LC fill (null = 10% of N)
  lc_fd_cutoff_m: null

  # FA/CB hybrid mixing coefficient (paper default: 0.5)
  fa_cb_lambda: 0.5

  score_column: coreset_score
```

---

## Pool Filtering

Before scoring, `CoreSetSelector` excludes degraded patches:

| Column | Excluded when | Configurable |
|---|---|---|
| `has_mask_border_nodata` | `True` | No — always excluded |
| `has_image_black_border` | `True` | No — always excluded |
| `has_high_nodata` | `True` | No — always excluded |
| `has_low_entropy` | `True` | `exclude_low_entropy` flag |

Set `exclude_low_entropy: false` to keep homogeneous patches (e.g. solid built-up areas). The
paper's qualitative analysis confirms that consistently rejected patches are homogeneous scenes
like large water bodies and parking lots — setting this flag `true` (default) replicates the
paper's experimental protocol.

---

## Embedding Sources

The paper uses ResNet-18 (ImageNet) feature vectors (R^512, after spatial pooling). The framework
accepts any pre-computed embedding, including DINOv2 patch tokens, stored in the input CSV or in
a separate Parquet file.

**Embeddings already in the CSV** — ensure the column named by `embedding_column` is present in
`balanced_dataset.csv`.

**Embeddings in a separate Parquet file** — set `parquet_path` to the Parquet file produced by
`embedding_extractor.py`. When `embedding_column` is absent from the CSV, `CoreSetSelector`
merges automatically on `(image_path, row_off, col_off)`:

```yaml
coreset:
  method: fd
  embedding_column: embedding
  parquet_path: /path/to/embeddings.parquet
  input_csv_path: /path/to/balanced_dataset.csv
  output_csv_path: /path/to/coreset.csv
```

---

## Spatial Diversity Augmentation

The framework extends the paper's methods with optional geographic spread enforcement,
not present in the original paper:

**`cb_use_spatial: true`** — appends tile centroid (lat, lon), min-max normalised to [0,1], to
the class-frequency vector before CB greedy selection. Prevents the greedy loop from clustering
all selected patches in the same geographic area even when they are class-balanced.

**`fd_use_spatial: true`** — appends centroid coordinates to the embedding matrix before FD
K-Means clustering and FA scoring. Also applies to `lc_fd` and `fa_cb` because they delegate
to `score_fd` / `score_fa` internally. Requires `tile_minx`, `tile_maxx`, `tile_miny`,
`tile_maxy` columns in the input CSV (produced by `build-balanced-dataset`).

---

## GPU Acceleration

`score_cb` and `score_fd` dispatch to GPU-accelerated paths when `device` resolves to `"cuda"`:

- **CB GPU path** — greedy entropy loop runs on float32 torch tensors.
- **FD GPU path** — elbow search and final K-Means use the framework's `MiniBatchKMeans` with
  CUDA tensors; inertia computation is batched to avoid OOM on large pools.

CPU and GPU paths are numerically equivalent (minor float32 vs float64 differences at identical
entropy values are expected).

```yaml
coreset:
  device: null    # auto (cuda when available)
  # device: "cpu"   # force CPU
  # device: "cuda"  # force GPU (RuntimeError if unavailable)
```

Progress is reported with `tqdm` bars: per-patch for CB, per-k for FD elbow search.

---

## Benchmark Results (from paper)

The table below shows representative results from the paper (% mIoU, U-Net, DFC2022 dataset).
All proposed methods outperform both baselines at their best-performing budget.

| Method | 25% | 50% | 75% | 100% (baseline) |
|---|---|---|---|---|
| Random (baseline) | lower | lower | lower | 100% reference |
| CoreSet (greedy) | lower | lower | lower | — |
| **LC** | ≥ baseline | ≥ baseline | ≥ baseline | — |
| **CB** | ≥ baseline | ≥ baseline | ≥ baseline | — |
| **FA/CB** | ≥ baseline | **best** | ≥ baseline | — |

Key conclusions from the paper:

- **Label-based methods (LC, CB)** outperform baselines across all datasets and architectures
  and achieve the best overall performance in 3 dataset–model combinations.
- **Image-based methods (FA, FD)** show more moderate gains but still outperform baselines
  and achieve best performance for SegFormer on Vaihingen.
- **Hybrid methods (LC/FD, FA/CB)** outperform baselines across all datasets and both
  architectures, achieving best results in 2 dataset–model combinations.
- **DFC2022 (noisy labels):** near-peak mIoU at 25–50%. Some methods exceed 100%-data baseline.
- **Vaihingen / Potsdam (clean labels):** performance improves with budget; 75% suffices
  to surpass the full-dataset baseline.
- **Training time:** the 50%-subset best model on DFC2022 completes training >24 h faster than
  the 100% baseline while achieving *higher* test mIoU.

---

## Sampler Weights

After selection, generate `WeightedRandomSampler` weights from the selected subset's class
distribution (framework extension, not in original paper):

```python
import pandas as pd
from pytorch_segmentation_models_trainer.tools.coreset import CoreSetConfig, CoreSetSelector

df = pd.read_csv("balanced_dataset.csv")
cfg = CoreSetConfig(method="cb", budget=0.4, output_csv_path="coreset.csv")
selector = CoreSetSelector(cfg)

result = selector.select(df)
result = selector.compute_sampler_weights(result, cap=0.25)

# Non-selected patches have weight 0; selected patches have weight > 0
weights = result["sampler_weight"].values
```

`compute_sampler_weights` implements the *sampler_weight_v3* recipe:

1. Measure global class frequency over selected patches only.
2. Per-patch weight = dot product of class proportions × normalised inverse-frequency weights.
3. Cap at `cap` (default 0.25) to prevent rare-class dominance.
4. Non-selected patches receive `weight = 0.0`.

Pass the coreset CSV to training with `weighted_sampler: true` to activate
`WeightedRandomSampler`.

---

## Python API

```python
import pandas as pd
from pytorch_segmentation_models_trainer.tools.coreset import CoreSetConfig, CoreSetSelector

df = pd.read_csv("balanced_dataset.csv")

# LC — no embeddings needed (seconds on 100k+ patches)
cfg = CoreSetConfig(
    method="lc",
    budget=0.5,
    budget_mode="fraction",
    output_csv_path="coreset_lc.csv",
)
result = CoreSetSelector(cfg).select(df)

# CB with spatial diversity enforcement
cfg_cb = CoreSetConfig(
    method="cb",
    budget=0.4,
    cb_use_spatial=True,
    output_csv_path="coreset_cb_spatial.csv",
)
result_cb = CoreSetSelector(cfg_cb).select(df)

# FD from Parquet embeddings with spatial augmentation
cfg_fd = CoreSetConfig(
    method="fd",
    budget=0.3,
    embedding_column="embedding",
    parquet_path="/path/to/embeddings.parquet",
    fd_use_spatial=True,
    output_csv_path="coreset_fd.csv",
)
result_fd = CoreSetSelector(cfg_fd).select(df)
```

---

## Parameter Reference

### CoreSetConfig

| Parameter | Type | Default | Description |
|---|---|---|---|
| `method` | str | `"lc"` | `lc` \| `cb` \| `fa` \| `fd` \| `lc_fd` \| `fa_cb` |
| `budget` | float | `0.5` | Fraction [0,1] or integer count |
| `budget_mode` | str | `"fraction"` | `"fraction"` or `"count"` |
| `embedding_column` | str or None | `None` | Column with embeddings (required for FA/FD) |
| `parquet_path` | str or None | `None` | Parquet with embeddings to auto-merge |
| `device` | str or None | `None` | `None`=auto, `"cpu"`, or `"cuda"` |
| `cb_use_spatial` | bool | `False` | Append tile centroid to CB feature vector |
| `fd_use_spatial` | bool | `False` | Append tile centroid to embeddings (FA/FD/LC-FD/FA-CB) |
| `exclude_low_entropy` | bool | `True` | Exclude `has_low_entropy=True` patches from pool |
| `fd_k_min` | int | `2` | Minimum K for FD K-selection |
| `fd_k_max` | int | `20` | Maximum K for FD K-selection |
| `fd_use_vendi` | bool | `False` | Vendi score K-selection (paper default); `false` = elbow |
| `fd_vendi_delta` | float | `0.005` | Convergence threshold δ for Vendi K-search (paper value) |
| `lc_fd_cutoff_m` | int or None | `None` | Top-m FD before LC fill (null = 10% of N) |
| `fa_cb_lambda` | float | `0.5` | FA/CB mixing λ (paper value: 0.5) |
| `score_column` | str | `"coreset_score"` | Output score column name |
| `input_csv_path` | str | `""` | Input CSV path (used by CLI) |
| `output_csv_path` | str | `"coreset.csv"` | Output CSV path |

---

## Related

- [Balanced Dataset Sampling](balanced-dataset-sampling.md) — generates `balanced_dataset.csv`
- [Training a Semantic Segmentation Model](training-segmentation.md) — `weighted_sampler` option
- [Hybrid Coreset Selection](hybrid-coreset-selection.md) — spatial vector intersection + FD/LC-FD for rare-class guarantee
- [Sampler Weights](sampler-weights.md) — compute per-patch WRS weights from `c0..c5` columns
- [SAM Label Correction](sam-label-correction.md) — correct noisy masks before coreset selection
