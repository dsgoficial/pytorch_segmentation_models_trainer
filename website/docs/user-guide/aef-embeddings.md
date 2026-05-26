---
sidebar_position: 19
title: AlphaEarth Foundation Embeddings
---

# AlphaEarth Foundation Embeddings

The soft-label preprocessing tools can blend AlphaEarth Foundation (AEF)
embeddings into `W_conf` through `w_embed`.  The implementation uses local
GeoTIFF or NumPy files and does not require `aef-loader`.

## Supported formats

GCS mode expects one multi-band GeoTIFF per tile:

```text
/data/aef_embeddings/{tile_id}.tif
```

The GeoTIFF is read as `(D, H, W)` and converted to `(H, W, D)` before computing
cosine similarity.

HuggingFace mode expects one patch-level NumPy vector per tile:

```text
/data/aef_hf_embeddings/{tile_id}.npy
```

HF vectors are not spatially resampled because each file represents one tile.

## Local AEF conversion

Raw AEF rasters store embeddings as `int8`.  The local converter handles the
AEF NoData value before numerical operations:

1. `-128` is treated as NoData and converted to `NaN`.
2. Valid values are dequantized with `sign(v) * (abs(v) / 127.5)^2`.
3. Invalid vectors are excluded from aggregation and centroid calculation.
4. Output vectors are L2-normalized before cosine similarity.

This avoids treating NoData as a valid negative embedding component.

## Resampling modes

Use `--aef-resampling` with `build-soft-labels`:

```bash
pytorch-smt-tools build-soft-labels sources.csv \
    --output-dir /data/soft_labels \
    --num-classes 4 \
    --alpha 0.5 \
    --beta 0.2 \
    --aef-embeddings-dir /data/aef_embeddings \
    --aef-source gcs \
    --aef-resampling auto
```

Available modes:

| Mode | Behavior |
|--|--|
| `auto` | Recommended default. Uses `aggregate` when target pixels are coarser than AEF and `nearest` when target pixels are finer. |
| `aggregate` | Downsamples with dequantize -> vector sum -> L2 normalize. Raises if target pixels are finer than AEF. |
| `nearest` | Reprojects with nearest neighbor, then L2-normalizes vectors. Use for upsampling. |
| `none` | Loads native raster without target-grid resampling. Only useful when no `dst_*` target grid is requested from code. |

## Downsampling

When the training image grid is coarser than the AEF raster, use vector
aggregation:

1. Dequantize raw AEF values.
2. Sum each embedding component into the target pixel footprint.
3. L2-normalize the summed vector.

This preserves embedding directions better than averaging or interpolating each
component independently.

## Upsampling

When the training image grid is finer than the AEF raster, use nearest-neighbor
assignment.  This copies the closest native AEF vector to each finer target
pixel and avoids creating synthetic vectors.

Bilinear, cubic, and average interpolation are not exposed for AEF embeddings
because they create off-manifold vectors and can corrupt cosine similarity.

## Example workflow

Download GCS embeddings:

```bash
pytorch-smt-tools download-aef-embeddings \
    --source gcs \
    --gcs-paths-csv gcs_paths.csv \
    --output-dir /data/aef_embeddings
```

Build soft labels with AEF blending:

```bash
pytorch-smt-tools build-soft-labels sources.csv \
    --output-dir /data/soft_labels \
    --num-classes 4 \
    --alpha 0.5 \
    --beta 0.2 \
    --aef-embeddings-dir /data/aef_embeddings \
    --aef-source gcs \
    --aef-resampling auto
```

Use [soft-label training](./soft-label-training.md) to train with the generated
`p_soft_path` and `w_conf_path` rasters.
