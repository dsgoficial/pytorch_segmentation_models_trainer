---
sidebar_position: 21
title: Soft-Label Training
---

# Soft-Label Training

Soft-label training replaces hard, one-hot segmentation masks with
**probabilistic labels** (P_soft) derived from multiple LULC source products.
An optional **per-pixel confidence weight** (W_conf) down-weights uncertain
pixels during loss computation.

This approach follows the methodology of Xiao et al. (2026, JSTARS):
*"Distilling 10-m Land Cover Maps from Multi-Source Consensus via AlphaEarth
Embeddings and Noise-Aware Weak Supervision."*

The framework extends the original paper with an additional **border-distance
component** in W_conf that penalises class-boundary pixels where label
ambiguity is highest.  See [W_conf formula](#w_conf-formula) for details.

---

## Components

| Component | Class | Purpose |
|-----------|-------|---------|
| Dataset | `SoftLabelDataset` | Reads P_soft and W_conf GeoTIFFs, returns `batch["mask"]` as a dict |
| Windowed Dataset | `SoftLabelWindowedDataset` | Reads patches on-the-fly from full-scene rasters via row/col offsets |
| Cached Dataset | `SoftLabelCachedDataset` | Computes P_soft lazily from `sources_csv`, caches to disk; supports windowed read |
| MBTiles Dataset | `MBTilesSoftLabelMaskWindowedDataset` | Reads RGB from MBTiles and computes P_soft/W_conf on mask-referenced windows |
| Loss | `SoftLabelWeightedCELoss` | Pixel-wise soft cross-entropy weighted by W_conf |
| Model | `SoftLabelModel` | Subclass of `Model` that passes W_conf through the loss pipeline |
| Preprocessing | `build_soft_labels.py` | Computes P_soft and W_conf from multiple LULC sources; optionally blends AEF embeddings |
| AEF download | `download_aef_embeddings.py` | Downloads AEF embeddings from GCS or HuggingFace |

---

## Step 1 — Build P_soft and W_conf Rasters

Use the CLI tool to compute P_soft (per-pixel class probability distributions)
and W_conf (confidence weights) from multiple LULC source rasters.

```bash
pytorch-smt-tools build-soft-labels sources.csv \
    --output-dir /data/soft_labels \
    --num-classes 4 \
    --alpha 0.6 \
    --mask-key mask_path \
    --lulc-key mapbiomas_path \
    --lulc-key esri_path \
    --lulc-key dw_path \
    --max-workers 8
```

> **Legacy scripts:** `python scripts/build_soft_labels.py` accepts the same
> flags and is kept for backward compatibility.

**sources.csv format:**

```csv
tile_id,image_path,mask_path,mapbiomas_path,esri_path,dw_path
tile_0,/data/images/tile_0.tif,/data/carta/tile_0.tif,/data/mapbiomas/tile_0.tif,/data/esri/tile_0.tif,/data/dw/tile_0.tif
```

All LULC rasters are automatically reprojected to the image's CRS, resolution,
and extent using nearest-neighbour resampling — the `image_path` column is the
spatial reference.

The script writes a manifest CSV with columns `tile_id`, `image_path`,
`p_soft_path`, `w_conf_path`.

---

## W_conf Formula

### With border-distance component (framework contribution)

This is the **default** behaviour.  It extends the original paper by adding a
border-distance penalty that reduces confidence near class boundaries — the
region with the highest label ambiguity across multi-source products.

```
W_conf = alpha · w_entropy + (1 - alpha - beta) · w_border + beta · w_embed
```

| Term | Description |
|------|-------------|
| `w_entropy` | Entropy-based confidence (see [Entropy normalisation](#entropy-normalisation)) |
| `w_border` | Distance transform from class boundaries, normalised to [0, 1] — high far from borders |
| `w_embed` | AEF cosine similarity to class centroid — high when the pixel matches its class in embedding space |

`alpha + beta ≤ 1.0` is required; the remainder `(1 - alpha - beta)` goes to
the border-distance term.

### Without border-distance component (original paper formula)

Use `--no-border` to reproduce the formula from the original paper exactly.
The border computation is skipped and the weights are renormalised over the
remaining terms:

```
W_conf = alpha · w_entropy                                        (no AEF)
W_conf = (alpha · w_entropy + beta · w_embed) / (alpha + beta)   (with AEF)
```

When `--no-border` is set, the `alpha + beta ≤ 1.0` constraint is relaxed.

### Entropy normalisation

The `--entropy-norm` option controls how `w_entropy` is computed from the
Shannon entropy `H(P_soft)`.  Two modes are available:

| Mode | Formula | Use case |
|------|---------|----------|
| `max_entropy` (default) | `w_entropy = 1 - H / log(C)` | Standard normalisation; scale is absolute |
| `minmax` | `w_entropy = 1 - (H - min H) / (max H - min H)` | Per-tile relative normalisation; matches LaTeX Eq. 9–10 (Experiments E4/E5) |

With `minmax`, the pixel with the lowest entropy in a tile always gets
`w_entropy = 1` and the pixel with the highest entropy gets `w_entropy = 0`,
regardless of the absolute entropy range.  When all pixels share the same
entropy (degenerate tile), `w_entropy = 1` for all pixels.

```bash
# Experiment E4 / E5 — per-tile min-max entropy normalisation
pytorch-smt-tools build-soft-labels sources.csv \
    --output-dir /data/soft_labels_e4 \
    --num-classes 6 --alpha 1.0 --no-border \
    --entropy-norm minmax
```

### BAGS border distance for Experiment E5

Experiment E5 uses the cartographic BAGS mask, not `argmax(P_soft)`, to compute
the border-distance component:

```text
w_border_carta(i) = min(1, d_carta(i) / R)
```

`d_carta(i)` is the Euclidean distance from pixel `i` to the nearest 3x3
morphological boundary pixel in the BAGS mask. The default radius is `R=10`
pixels.

The source CSV must contain the BAGS cartographic mask column selected by
`--mask-key`, plus any external LULC columns listed with `--lulc-key`:

```csv
tile_id,image_path,mask_path,mapbiomas_path,esri_path,dw_path
tile_0,/data/images/tile_0.tif,/data/lulc/bags_0.tif,/data/lulc/mapbiomas_0.tif,/data/lulc/esri_0.tif,/data/lulc/dw_0.tif
```

```bash
pytorch-smt-tools build-soft-labels sources.csv \
    --output-dir /data/soft_labels_e5 \
    --num-classes 6 \
    --alpha 0.6 \
    --entropy-norm minmax \
    --mask-key mask_path \
    --lulc-key mapbiomas_path \
    --lulc-key esri_path \
    --lulc-key dw_path \
    --border-radius 10
```

:::tip Ablation study

Run the same training with and without `--no-border` to quantify the impact of
the border-distance contribution:

```bash
# Original paper formula
pytorch-smt-tools build-soft-labels sources.csv \
    --output-dir /data/soft_labels_no_border \
    --num-classes 4 --alpha 0.6 --no-border

# With border-distance contribution
pytorch-smt-tools build-soft-labels sources.csv \
    --output-dir /data/soft_labels_border \
    --num-classes 4 --alpha 0.6
```

Then train with `conf/examples/soft_label_no_border.yaml` vs
`conf/examples/soft_label_unet.yaml` and compare OA / mIoU.
:::

---

## Step 1b — Download AEF Embeddings (Optional)

### GCS mode — per-pixel dense embeddings

Requires `gsutil` and GCS access to the AlphaEarth Foundation bucket.

```bash
pytorch-smt-tools download-aef-embeddings \
    --source gcs \
    --gcs-paths-csv gcs_paths.csv \
    --output-dir /data/aef_embeddings
```

**gcs_paths.csv format:**

```csv
tile_id,gcs_uri
tile_0,gs://alphaearth_foundations/embeddings/tile_0.tif
tile_1,gs://alphaearth_foundations/embeddings/tile_1.tif
```

Each downloaded file is a multi-band GeoTIFF `{tile_id}.tif` with shape
`(D, H, W)`.  The per-pixel cosine similarity to each pixel's within-tile class
centroid is computed:

```
w_embed(i) = (cos_sim(emb(i), centroid(argmax_class(i))) + 1) / 2
```

### HuggingFace mode — patch-level embeddings

Downloads patch-level 64-D embeddings from
[`Major-TOM/Core-AlphaEarth-Embeddings`](https://huggingface.co/datasets/Major-TOM/Core-AlphaEarth-Embeddings)
on HuggingFace.  Requires `pip install datasets`.

```bash
pytorch-smt-tools download-aef-embeddings \
    --source hf \
    --tiles-csv tiles.csv \
    --output-dir /data/aef_hf_embeddings
```

**tiles.csv format:**

```csv
tile_id,image_path
tile_0,/data/images/tile_0.tif
tile_1,/data/images/tile_1.tif
```

For each tile, the script finds the nearest Major-TOM grid cell by geographic
proximity and saves its embedding as `{tile_id}.npy`.

When used with `--aef-source hf`, the scalar cosine similarity between the
tile embedding and the dominant-class cross-tile centroid is broadcast
uniformly to all pixels:

```
w_embed(i) = (cos_sim(emb_tile, centroid(dominant_class)) + 1) / 2
```

### Source Cooperative mode — cropped per-pixel COG embeddings

Downloads 64-band per-pixel AEF embedding crops from the public
[Source Cooperative AEF annual COG collection](https://source.coop/tge-labs/aef/v1/annual).
This mode reads the STAC GeoParquet index, selects the annual COG intersecting
each tile footprint, and writes only the crop needed by the tile as
`{tile_id}.tif`.

```bash
pytorch-smt-tools download-aef-embeddings \
    --source sourcecoop \
    --tiles-csv tiles.csv \
    --output-dir /data/aef_sourcecoop_embeddings \
    --year 2025
```

When `--year` is omitted, the downloader uses a `year` column in `tiles.csv` or
the first 4-digit year found in `image_path`.

```csv
tile_id,image_path,year
tile_0,/data/images/tile_20250625_20260106.tif,2025
tile_1,/data/images/tile_20240101.tif,2024
```

The resulting files are per-pixel GeoTIFF embeddings, so use them in the build
step with `--aef-source gcs`:

```bash
pytorch-smt-tools build-soft-labels sources.csv \
    --output-dir /data/soft_labels \
    --num-classes 4 \
    --alpha 0.5 \
    --beta 0.2 \
    --aef-embeddings-dir /data/aef_sourcecoop_embeddings \
    --aef-source gcs
```

**Comparison of AEF modes:**

| | GCS (per-pixel) | HF (patch-level) |
|--|--|--|
| Spatial granularity | Per pixel | Uniform per tile |
| Embedding dimension | Varies (e.g. 256) | 64 |
| Requires `gsutil` | Yes | No |
| Requires `datasets` | No | Yes |
| File per tile | `{tile_id}.tif` | `{tile_id}.npy` |

Then build soft labels with AEF blending (see [conf/examples/soft_label_aef_gcs.yaml](https://github.com)):

```bash
pytorch-smt-tools build-soft-labels sources.csv \
    --output-dir /data/soft_labels \
    --num-classes 4 \
    --alpha 0.5 \
    --beta 0.2 \
    --aef-embeddings-dir /data/aef_embeddings \
    --aef-source gcs \
    --aef-resampling auto \
    --save-aef-aligned \
    --aef-aligned-dtype int8
```

:::info AEF resampling is local and vector-aware

The preprocessing tool does not require `aef-loader` for local GeoTIFF files.
It includes local AEF helpers for NoData handling, dequantization, vector
aggregation, nearest-neighbor upsampling, and L2 normalization.

Raw AEF `int8` rasters use `-128` as NoData.  This value is converted to `NaN`
before any aggregation so invalid pixels do not become negative embedding
components.

`--aef-resampling auto` chooses the spatial operation from the source and target
pixel areas:

- **Downsampling** (training image is coarser): dequantize -> element-wise
  vector sum -> L2 normalize.
- **Upsampling** (training image is finer): nearest-neighbor assignment -> L2
  normalize.

Bilinear, cubic, and average interpolation are not exposed for AEF vectors
because they create synthetic off-manifold embeddings and can corrupt cosine
similarities.

See [AlphaEarth Foundation Embeddings](./aef-embeddings.md) for resampling
modes, aligned embedding export, and examples.
:::

---

## Step 1c — Windowed Patch Manifest (Optional)

For large tiles, use `--patch-size` to generate a patch-level manifest for
`SoftLabelWindowedDataset` — no pre-cut patch files are written:

```bash
pytorch-smt-tools build-soft-labels sources.csv \
    --output-dir /data/soft_labels \
    --num-classes 4 \
    --alpha 0.6 \
    --patch-size 512 \
    --stride 256        # 50% overlap; defaults to patch-size if omitted
```

Output: `soft_label_patches.csv` with columns `tile_id`, `image_path`,
`p_soft_path`, `w_conf_path`, `row_off`, `col_off`, `patch_size`.

See [`conf/examples/soft_label_windowed_unet.yaml`](#e5--windowed-dataset-large-tiles) for the
corresponding training configuration.

---

## Step 2 — Generate Train/Val/Test Splits

```bash
pytorch-smt-tools generate-training-csv \
    /data/soft_labels/soft_label_manifest.csv \
    --output-dir /data/splits \
    --train-ratio 0.70 \
    --val-ratio 0.15 \
    --seed 42
```

This produces `train.csv`, `val.csv`, and `test.csv` with columns:
`tile_id`, `image_path`, `p_soft_path`, `w_conf_path`.

If `image_path` is absent from the manifest (e.g. you removed it), pass
`--image-dir` to inject the column:

```bash
pytorch-smt-tools generate-training-csv manifest.csv \
    --output-dir /data/splits \
    --image-dir /data/images \
    --image-extension .tif
```

---

## Step 3 — Training Configuration

### E1 — P_soft only, no confidence weighting

```yaml title="conf/examples/soft_label_no_wconf.yaml"
_target_: pytorch_segmentation_models_trainer.model_loader.soft_label_model.SoftLabelModel

train_dataset:
  _target_: pytorch_segmentation_models_trainer.dataset_loader.soft_label_dataset.SoftLabelDataset
  input_csv_path: /data/splits/train.csv
  image_key: image_path
  p_soft_key: p_soft_path
  # w_conf_key omitted → unweighted soft cross-entropy

loss:
  _target_: pytorch_segmentation_models_trainer.custom_losses.soft_label_loss.SoftLabelWeightedCELoss
  name: soft_label_ce
  num_classes: 4
```

### E2 — P_soft + W_conf (with border contribution)

```yaml title="conf/examples/soft_label_unet.yaml"
_target_: pytorch_segmentation_models_trainer.model_loader.soft_label_model.SoftLabelModel

train_dataset:
  _target_: pytorch_segmentation_models_trainer.dataset_loader.soft_label_dataset.SoftLabelDataset
  input_csv_path: /data/splits/train.csv
  image_key: image_path
  p_soft_key: p_soft_path
  w_conf_key: w_conf_path      # W_conf includes border-distance component

loss:
  _target_: pytorch_segmentation_models_trainer.custom_losses.soft_label_loss.SoftLabelWeightedCELoss
  name: soft_label_ce
  num_classes: 4
  mask_key: mask
  weight_key: w_conf
```

### E2-NB — P_soft + W_conf (original paper, no border)

```yaml title="conf/examples/soft_label_no_border.yaml"
_target_: pytorch_segmentation_models_trainer.model_loader.soft_label_model.SoftLabelModel

train_dataset:
  _target_: pytorch_segmentation_models_trainer.dataset_loader.soft_label_dataset.SoftLabelDataset
  input_csv_path: /data/splits_no_border/train.csv   # built with --no-border
  image_key: image_path
  p_soft_key: p_soft_path
  w_conf_key: w_conf_path      # W_conf = w_entropy only (original paper formula)

loss:
  _target_: pytorch_segmentation_models_trainer.custom_losses.soft_label_loss.SoftLabelWeightedCELoss
  name: soft_label_ce
  num_classes: 4
  mask_key: mask
  weight_key: w_conf
```

### E4 — P_soft + W_conf + AEF GCS embeddings

```yaml title="conf/examples/soft_label_aef_gcs.yaml"
_target_: pytorch_segmentation_models_trainer.model_loader.soft_label_model.SoftLabelModel

train_dataset:
  _target_: pytorch_segmentation_models_trainer.dataset_loader.soft_label_dataset.SoftLabelDataset
  input_csv_path: /data/splits/train.csv
  image_key: image_path
  p_soft_key: p_soft_path
  w_conf_key: w_conf_path      # W_conf includes entropy + AEF + border

loss:
  _target_: pytorch_segmentation_models_trainer.custom_losses.soft_label_loss.SoftLabelWeightedCELoss
  name: soft_label_ce
  num_classes: 4
  mask_key: mask
  weight_key: w_conf

model:
  _target_: segmentation_models_pytorch.Unet
  encoder_name: resnet50
  encoder_weights: imagenet
  in_channels: 3
  classes: 4
```

### E5 — Windowed dataset (large tiles)

```yaml title="conf/examples/soft_label_windowed_unet.yaml"
_target_: pytorch_segmentation_models_trainer.model_loader.soft_label_model.SoftLabelModel

train_dataset:
  _target_: pytorch_segmentation_models_trainer.dataset_loader.soft_label_windowed_dataset.SoftLabelWindowedDataset
  input_csv_path: /data/splits/train.csv  # patch manifest from --patch-size
  image_key: image_path
  p_soft_key: p_soft_path
  w_conf_key: w_conf_path
  row_off_key: row_off
  col_off_key: col_off
  patch_size_key: patch_size
  augmentation_list:
    - _target_: albumentations.HorizontalFlip
      p: 0.5
    - _target_: albumentations.Normalize
      mean: [0.485, 0.456, 0.406]
      std: [0.229, 0.224, 0.225]
    - _target_: albumentations.pytorch.ToTensorV2
```

---

## API Reference

### `SoftLabelDataset`

```python
from pytorch_segmentation_models_trainer.dataset_loader.soft_label_dataset import (
    SoftLabelDataset,
)

ds = SoftLabelDataset(
    input_csv_path="train.csv",
    image_key="image_path",
    p_soft_key="p_soft_path",
    w_conf_key="w_conf_path",  # omit if CSV has no w_conf_path column
)

sample = ds[0]
# sample["image"]          — (3, H, W) float32 in [0, 1]
# sample["mask"]["mask"]   — (C, H, W) float32, sums to 1 per pixel
# sample["mask"]["w_conf"] — (1, H, W) float32 in [0, 1]
# sample["path"]           — image file path
```

### `SoftLabelWeightedCELoss`

```python
from pytorch_segmentation_models_trainer.custom_losses.soft_label_loss import (
    SoftLabelWeightedCELoss,
)

loss_fn = SoftLabelWeightedCELoss(name="soft_ce", num_classes=4)
loss = loss_fn.compute(
    logits,                          # (B, C, H, W)
    {"mask": p_soft, "w_conf": w},   # or just p_soft tensor
)
```

The loss computes:

```
L(i) = W_conf(i) · [-Σ_c P_soft(i,c) · log(softmax(logits)(i,c))]
```

When `w_conf` is absent (or all ones), this reduces to standard soft cross-entropy.

### `SoftLabelWindowedDataset`

Reads patches on-the-fly from full-scene rasters using `rasterio.windows.Window`.
Requires a patch manifest CSV (produced by `build_soft_labels --patch-size`):

```python
from pytorch_segmentation_models_trainer.dataset_loader.soft_label_windowed_dataset import (
    SoftLabelWindowedDataset,
)

ds = SoftLabelWindowedDataset(
    input_csv_path="patches.csv",
    image_key="image_path",
    p_soft_key="p_soft_path",
    w_conf_key="w_conf_path",   # optional
    row_off_key="row_off",
    col_off_key="col_off",
    patch_size_key="patch_size",
)
sample = ds[0]
# sample["image"]          — (3, patch_size, patch_size) float32
# sample["mask"]["mask"]   — (C, patch_size, patch_size) float32
# sample["mask"]["w_conf"] — (1, patch_size, patch_size) float32 (when present)
```

### `SoftLabelCachedDataset`

Computes P_soft lazily from a `sources_csv` file (same format as `build-soft-labels`),
writes the result as a full-tile GeoTIFF cache on first access, and reads from the cache
on subsequent accesses.  W_conf is recomputed on every access so that `alpha`,
`entropy_norm`, and `use_border` can be changed without invalidating the cache.

When `patch_size` is given the dataset operates in **windowed mode**: it enumerates
all `(patch_size, patch_stride)` patches across every tile and each `__getitem__`
call reads only that window from the image and cache via rasterio windowed reads.

```python
from pytorch_segmentation_models_trainer.dataset_loader.soft_label_cached_dataset import (
    SoftLabelCachedDataset,
)

# Full-tile mode (one item per tile)
ds = SoftLabelCachedDataset(
    sources_csv="sources.csv",
    cache_dir="/data/cache/p_soft",
    num_classes=6,
    alpha=0.6,
    use_border=True,
    entropy_norm="minmax",  # Experiment E5
    mask_key="mask_path",
    lulc_keys=["mapbiomas_path", "esri_path", "dw_path"],
    border_radius=10,
)

# Windowed mode (one item per patch)
ds_win = SoftLabelCachedDataset(
    sources_csv="sources.csv",
    cache_dir="/data/cache/p_soft",   # same cache — no rebuild needed
    num_classes=6,
    alpha=0.6,
    use_border=True,
    mask_key="mask_path",
    lulc_keys=["mapbiomas_path", "esri_path", "dw_path"],
    border_radius=10,
    patch_size=(256, 256),
    patch_stride=(256, 256),          # no overlap; use (128, 128) for 50% overlap
)

sample = ds[0]
# sample["image"]          — (3, H, W) float32 in [0, 1]
# sample["mask"]["mask"]   — (C, H, W) float32, sums to 1 per pixel
# sample["mask"]["w_conf"] — (1, H, W) float32 in [0, 1]
# sample["path"]           — image file path
```

**Caching contract:**

- The full-tile P_soft is cached once in `cache_dir/{tile_id}.tif` regardless of
  whether you use full-tile or windowed mode.
- `alpha`, `entropy_norm`, `use_border`, `mask_key`, `lulc_keys`, and `border_radius`
  do **not** affect the cache key —
  change them between runs without clearing the cache.
- Writes are atomic (`{tile_id}.tif.tmp` → rename) so concurrent DataLoader workers
  (multi-process) cannot corrupt a partial write.
- Stale `.tmp` files left by crashed workers are overwritten on the next access.

### `MBTilesSoftLabelMaskWindowedDataset`

Use this dataset when RGB imagery lives in MBTiles and mask GeoTIFFs define the
training grid. The soft label is computed per sampled window as an equal vote
over the BAGS mask window and each LULC raster/VRT listed in `lulc_paths`.

```yaml
train_dataset:
  _target_: pytorch_segmentation_models_trainer.dataset_loader.mbtiles_soft_label_dataset.MBTilesSoftLabelMaskWindowedDataset
  mbtiles_path: /data/tiles.mbtiles
  mask_dir: /data/masks
  window_index_cache: /data/coreset_windows.csv
  patch_size: 256
  selected_bands: [1, 2, 3]
  lulc_paths:
    - /data/mapbiomas_edgv_2_5m.vrt
    - /data/esri_edgv_2_5m.vrt
    - /data/dynamic_world_edgv_2_5m.vrt
  num_classes: 6
  return_w_conf: true
  alpha: 0.6
  use_border: true
  entropy_norm: minmax
  border_radius: 10
```

Set `return_w_conf: false` for uniform-weight soft-label training (E3). Use
`return_w_conf: true`, `use_border: false`, and `alpha: 1.0` for entropy-only
weighting (E4). Use `use_border: true` and `alpha: 0.6` for entropy plus BAGS
border weighting (E5).

### `SoftLabelModel`

Drop-in replacement for `Model`.  All other `Model` behaviour
(metrics, logging, optimisers, LR schedulers, dual-head, OHEM, EDL, MoE)
is inherited without modification.

```yaml
_target_: pytorch_segmentation_models_trainer.model_loader.soft_label_model.SoftLabelModel
```

---

## Experiment variants

| Experiment | W_conf formula | CLI flag | Config file |
|------------|---------------|----------|-------------|
| E0 | — (hard labels, CE) | — | — |
| E1 | — (no weighting) | `--alpha 0.6` | `soft_label_no_wconf.yaml` |
| E2-NB | `alpha·w_entropy` | `--alpha 0.6 --no-border` | `soft_label_no_border.yaml` |
| E2 | `alpha·w_entropy + (1-alpha)·w_border` | `--alpha 0.6` | `soft_label_unet.yaml` |
| E3 | `alpha·w_e + beta·w_embed + (1-a-b)·w_border` | `--alpha 0.5 --beta 0.2 --aef-source hf` | `soft_label_unet.yaml` |
| E4 | `alpha·w_entropy` with min-max entropy | `--alpha 1.0 --no-border --entropy-norm minmax` | `soft_label_no_border.yaml` |
| E5 | `alpha·w_entropy + (1-alpha)·w_border_carta` | `--alpha 0.6 --entropy-norm minmax --mask-key mask_path --lulc-key mapbiomas_path --lulc-key esri_path --lulc-key dw_path --border-radius 10` | `soft_label_cached.yaml` |

> **E2-NB** ("no border") is the formula from the original paper.
> **E2** adds the border-distance component — a framework contribution not in the paper.
> The difference between E2-NB and E2 quantifies the impact of that contribution.
