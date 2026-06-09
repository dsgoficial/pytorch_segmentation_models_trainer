---
sidebar_position: 6
title: Balanced Dataset Sampling
---

# Balanced Dataset Sampling

Class imbalance is a common problem in geospatial segmentation: a dataset of 155 000 patches
may have 90% background and only 2% of a rare urban class. Training on a uniformly sampled
dataset biases the model toward the majority class.

The balanced dataset sampling pipeline produces a `sampler_weight` column for every patch.
When loaded into training with `weighted_sampler: true`, PyTorch's
`WeightedRandomSampler` over-samples rare/unique patches and under-samples common ones
each epoch, without discarding any data.

---

## Two Signals, One Weight

Each patch receives a weight from two independent signals that are combined into one:

| Signal | How it's computed | What it captures |
|---|---|---|
| **Semantic composition** | Centered log-ratio (CLR) transform → KMeans clustering | Which classes appear in the patch and in what proportion |
| **Visual uniqueness** | DINOv2 CLS-token embeddings → cosine kNN distance | How visually distinct the patch is from its neighbors |

Six combination strategies are available (`sampling_method` key):

| Method | Formula | Use when |
|---|---|---|
| `rank_max` *(default)* | `max(rank_composition, rank_uniqueness)` | Best general-purpose choice |
| `rank_multiply` | `rank_comp × rank_uniq` | Penalise patches that score low on both |
| `rank_add` | `rank_comp + rank_uniq` | Smooth blend |
| `multiplicative` | `comp_score × uniq_score` | Raw signal product |
| `composition_only` | `rank_composition` | Skip embeddings (faster) |
| `uniqueness_only` | `rank_uniqueness` | Skip clustering |

---

## Step 1 — Generate the Balanced CSV

### Install optional dependencies

```bash
# Required for all modes
pip install rasterio scikit-learn pyarrow

# For visual uniqueness (mode: embeddings)
pip install transformers torch

# For fast kNN on large datasets (recommended for N > 50k)
pip install hnswlib          # approximate HNSW, ~10-30 s for 155k patches
# or
pip install faiss-cpu        # exact BLAS search, ~1-3 min for 155k patches

# For postgres backend
pip install psycopg2-binary

# For GeoJSON output (QGIS inspection)
pip install shapely
```

### Local filesystem (masks_only mode)

The fastest option — no GPU, no embeddings. Uses only mask class distributions.

```yaml
# conf/balanced_train.yaml
balanced_dataset:
  mode: masks_only
  sampling_method: rank_max

  input:
    patch_records:
      - image_path: /data/images/tile_001.tif
        mask_path:  /data/masks/tile_001.tif
        row_off: 0
        col_off: 0
        patch_size: 256
      # ... one entry per patch

  clustering:
    k_min: 4
    k_max: 16
    k_selection: elbow   # automatic elbow heuristic
    random_state: 42

  exclusion:
    exclude_border_nodata: true
    exclude_black_border: true
    nodata_class: 0

  output:
    csv_path: /data/balanced_train.csv
    # Optional: patch footprints as GeoJSON for QGIS inspection
    # geojson_path: /data/balanced_train.geojson
```

```bash
pytorch-smt-tools build-balanced-dataset conf/balanced_train.yaml
```

### Local filesystem (embeddings mode)

Adds visual uniqueness via DINOv2. Embeddings are cached to a parquet file and reloaded
on subsequent runs.

```yaml
balanced_dataset:
  mode: embeddings
  sampling_method: rank_max

  input:
    patch_records: ...      # same as above

  clustering:
    k_min: 4
    k_max: 16
    k_selection: elbow

  uniqueness:
    k_neighbors: 5
    # backend options (fastest to slowest for N=155k):
    #   hnsw    — approximate, O(N log N), requires hnswlib    (~10-30 sec)
    #   faiss   — exact BLAS, requires faiss-cpu               (~1-3 min)
    #   sklearn — exact, uses all CPU cores (n_jobs=-1)        (~2-4 min)
    backend: hnsw

  embeddings:
    dino_model: facebook/dinov2-base
    batch_size: 32
    num_workers: 4
    use_gpu: false
    parquet_path: /data/embeddings.parquet   # cache; reloaded if file exists

  exclusion:
    exclude_border_nodata: true
    exclude_black_border: true

  output:
    csv_path: /data/balanced_train.csv
    geojson_path: /data/balanced_train.geojson
```

### PostgreSQL backend (dataset_explorer)

Reads patch metadata directly from a `dataset_explorer` tile table.

```yaml
balanced_dataset:
  mode: embeddings
  sampling_method: rank_max

  input:
    host: localhost
    port: 5432
    database: dataset_explorer
    user: postgres
    password: ""
    table: tiles
    where_clause: "nodata_ratio < 0.05"

  clustering:
    k_min: 4
    k_max: 20
    k_selection: elbow

  uniqueness:
    k_neighbors: 5
    backend: hnsw

  embeddings:
    dino_model: facebook/dinov2-base
    batch_size: 32
    num_workers: 4
    use_gpu: false
    parquet_path: /data/embeddings.parquet

  exclusion:
    exclude_border_nodata: true
    exclude_black_border: true

  output:
    csv_path: /data/balanced_train.csv
```

---

## Step 2 — Use the Balanced CSV in Training

Pass the balanced CSV as `input_csv_path` and enable `weighted_sampler: true` in the
`data_loader` block of `train_dataset`.

A complete working example is available at
`conf/examples/weighted_sampler_segmentation.yaml`.

```yaml
train_dataset:
  _target_: pytorch_segmentation_models_trainer.dataset_loader.dataset.CSVWindowedSegmentationDataset
  input_csv_path: /data/balanced_train.csv   # ← balanced CSV with sampler_weight column
  n_classes: 6
  use_rasterio: true
  image_dtype: uint8
  augmentation_list:
    - _target_: albumentations.HorizontalFlip
      p: 0.5
    - _target_: albumentations.Normalize
      mean: [0.485, 0.456, 0.406]
      std: [0.229, 0.224, 0.225]
    - _target_: albumentations.pytorch.ToTensorV2
  data_loader:
    shuffle: true           # ignored when weighted_sampler: true
    num_workers: 4
    pin_memory: true
    drop_last: true
    weighted_sampler: true  # ← activates WeightedRandomSampler

val_dataset:
  _target_: pytorch_segmentation_models_trainer.dataset_loader.dataset.CSVWindowedSegmentationDataset
  input_csv_path: /data/val.csv   # val does NOT need sampler_weight
  n_classes: 6
  use_rasterio: true
  data_loader:
    shuffle: false
    num_workers: 4
```

`val_dataset` and `test_dataset` are **not affected** — weighted sampling is only
applied to the training loader.

### Optional `data_loader` keys

| Key | Type | Default | Description |
|---|---|---|---|
| `weighted_sampler` | bool | `false` | Activate WeightedRandomSampler |
| `weighted_sampler_num_samples` | int | `len(dataset)` | Samples drawn per epoch |
| `weighted_sampler_replacement` | bool | `true` | Sample with replacement |

:::tip `shuffle` vs `weighted_sampler`
When `weighted_sampler: true` is set, `shuffle: true` is automatically disabled.
PyTorch's `DataLoader` does not allow both a custom sampler and `shuffle=True`
simultaneously. The `WeightedRandomSampler` provides the randomness instead.
:::

:::note What datasets are compatible?
Any dataset class that inherits from `AbstractDataset` (which includes all built-in
dataset classes) is compatible. The only requirement is that the DataFrame loaded from
`input_csv_path` contains a `sampler_weight` column — produced by
`build-balanced-dataset`.
:::

---

## Inspecting Results in QGIS

When `geojson_path` is set, `build-balanced-dataset` writes a GeoJSON where each
feature is a patch polygon with `sampler_weight`, `freq_cluster_id`,
`uniqueness_score`, and `excluded` as properties.

Load it in QGIS with **Layer → Add Layer → Add Vector Layer**, then style by
`sampler_weight` (graduated color) to visually inspect which patches the sampler
will favour.

Requires `shapely`:

```bash
pip install shapely
```

---

## Performance Notes for Large Datasets

For datasets with N > 50 000 patches, the kNN step (`compute_uniqueness`) dominates
runtime in `mode: embeddings`.

| Backend | Complexity | N=155k, D=768 | Install |
|---|---|---|---|
| `sklearn` *(default)* | O(N²D), multi-core | ~2-4 min | — |
| `faiss` | O(N²D), BLAS+SIMD | ~1-3 min | `pip install faiss-cpu` |
| `hnsw` *(recommended)* | O(N log N) | ~10-30 sec | `pip install hnswlib` |

Switch backend in the YAML:

```yaml
uniqueness:
  k_neighbors: 5
  backend: hnsw
```

DINOv2 embedding extraction time depends on GPU availability. With `use_gpu: false`
and `batch_size: 32`, expect ~10-30 minutes for 155k patches on a modern CPU.
Embeddings are cached to `parquet_path` — subsequent runs skip extraction entirely.
