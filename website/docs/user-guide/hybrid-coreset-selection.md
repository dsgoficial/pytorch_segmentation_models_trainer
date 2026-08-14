---
sidebar_position: 25
title: Hybrid Coreset Selection
---

# Hybrid Coreset Selection

`select-hybrid-coreset` extends the standard [CoreSet Selection](./coreset-selection.md) workflow
with a **spatial vector intersection** phase for rare classes and **FD / LC-FD embedding diversity**
selection for common classes, followed by an entropy sweep to fill any remaining budget.

This approach was developed to address classes that are too rare for the embedding-based methods
to sample reliably: even with FD round-robin, a class that appears in only 0.2% of patches may
be entirely absent from a 30% coreset. Spatial intersection with reference polygon layers
guarantees coverage.

---

## Selection pipeline

Selection proceeds in three phases:

### Phase 1 — Vector intersection (rare classes)

For each `vector_steps` entry, patches are spatial-joined with a GeoPackage layer (e.g.
official field-boundary polygons). Any patch whose intersection area with the reference polygons
exceeds `min_intersect_area_m2` is selected and tagged with `vector:<layer_name>`.

### Phase 2 — Embedding diversity (common classes)

For each `embedding_steps` entry, the pool is first filtered by `class_filter` (e.g. patches
where `c4 > 0.5` for forest-dominant tiles), then either:

- **`fd`** — K-Means Facility-Location diversity: clusters the filtered pool into `k` groups and
  round-robins through cluster centroids, picking the nearest-to-centroid unselected patch from
  each cluster in turn.
- **`lc_fd`** — LC/FD hybrid: allocates `lc_fraction` of the budget to the highest-entropy patches
  (Least Confidence, forces rare-event coverage), then fills the remainder with FD round-robin.

### Phase 3 — Entropy sweep

If `entropy_sweep: true` and the budget is not yet exhausted, the remaining slots are filled by the
highest-entropy unselected patches (descending `class_entropy`).

At every phase, already-selected patches are excluded from subsequent steps (no double-counting).

---

## CLI usage

```bash
pytorch-smt-tools select-hybrid-coreset path/to/config.yaml
```

---

## Configuration

```yaml
input_csv_path: /data/pool.csv
embeddings_parquet: /data/embeddings.parquet   # joined to pool by index
vectors_gpkg: /data/reference.gpkg
output_csv_path: /data/coreset_hybrid.csv

pool_fraction: 0.30    # fraction of the pool to select as total budget

# Phase 1: spatial vector steps
vector_steps:
  - gpkg_layer: grassland_polygons       # layer name inside vectors_gpkg
    class_indices: [c3]                  # informational — not used in spatial filter
    min_intersect_area_m2: 1000.0        # minimum intersection area threshold

  - gpkg_layer: cropland_polygons
    class_indices: [c5]
    min_intersect_area_m2: 1000.0

# Phase 2: embedding diversity steps
embedding_steps:
  - class_filter:
      c4: [">", 0.50]    # forest-dominant patches
    budget: 500
    k: 10
    method: lc_fd        # "fd" or "lc_fd"
    lc_fraction: 0.40

  - class_filter:
      c1: [">", 0.30]    # water patches
    budget: 200
    k: 5
    method: fd

# Phase 3
entropy_sweep: true

# Spatial CRS and bbox column names
crs: EPSG:3857
bbox_columns: [tile_minx, tile_miny, tile_maxx, tile_maxy]

# Column names
embedding_col: embedding
entropy_col: class_entropy
random_state: 42
```

### `class_filter` operators

The `class_filter` dict maps column names to `[operator, threshold]` pairs.
Supported operators: `>`, `>=`, `<`, `<=`.

```yaml
class_filter:
  c4: [">", 0.50]    # forest proportion > 50%
  c1: [">=", 0.10]   # water proportion >= 10%
```

---

## Output columns

The output CSV contains all columns from `input_csv_path` plus:

| Column | Type | Description |
|--------|------|-------------|
| `coreset_selected` | int (0/1) | 1 if patch is in the coreset |
| `selection_step` | str | Phase that selected this patch |

`selection_step` values: `vector:<layer_name>`, `embedding:fd:<step_num>`,
`embedding:lc_fd:<step_num>`, `entropy_sweep`.

---

## Python API

```python
import pandas as pd
from pytorch_segmentation_models_trainer.tools.coreset import (
    HybridVectorCoresetConfig,
    HybridVectorCoresetSelector,
)

config = HybridVectorCoresetConfig(
    input_csv_path="/data/pool.csv",
    embeddings_parquet="/data/emb.parquet",
    vectors_gpkg="/data/reference.gpkg",
    output_csv_path="/data/coreset.csv",
    pool_fraction=0.30,
    vector_steps=[
        {"gpkg_layer": "grassland", "class_indices": ["c3"], "min_intersect_area_m2": 1000.0}
    ],
    embedding_steps=[
        {"class_filter": {"c4": (">", 0.50)}, "budget": 500, "k": 10, "method": "lc_fd"}
    ],
    entropy_sweep=True,
)

pool_df = pd.read_csv(config.input_csv_path)
result = HybridVectorCoresetSelector(config).select(pool_df)
print(result["selection_step"].value_counts())
```

### Primitive functions

The selection primitives in `tools/coreset/vector_selector.py` are also available independently:

```python
from pytorch_segmentation_models_trainer.tools.coreset import (
    compute_intersection_areas,
    select_by_vector_intersection,
    fd_embedding_select,
    lc_fd_select,
    entropy_sweep_select,
)
```

---

## Relationship to standard CoreSet Selection

| | `select-coreset` | `select-hybrid-coreset` |
|---|---|---|
| Rare-class guarantee | No | Yes (vector steps) |
| Embedding diversity | Yes (FD / LC-FD / FA / CB) | Yes (FD / LC-FD) |
| Requires GeoPackage | No | Yes |
| Requires embeddings | Yes | Yes |
| `selection_step` tagging | No | Yes |

Use `select-hybrid-coreset` when your dataset has rare classes with polygon reference data
(e.g. field boundaries, wetland inventories). Use `select-coreset` for the standard diversity-only
workflow.
