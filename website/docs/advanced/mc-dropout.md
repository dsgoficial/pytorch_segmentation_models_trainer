---
sidebar_position: 8
title: MC Dropout (Uncertainty)
description: Test-time uncertainty estimation via Monte Carlo Dropout for segmentation models.
---

# Monte Carlo Dropout at Test Time

## What is MC Dropout?

MC Dropout (Gal & Ghahramani, 2016) produces uncertainty estimates from any
model that already uses Dropout — with **no changes to training**.

The technique keeps Dropout layers **active during inference** and runs **T
stochastic forward passes**.  Because each pass uses a different random dropout
mask, each pass is equivalent to sampling from an approximate posterior over
model weights.  The mean prediction across the T passes is the final output;
their disagreement is a proxy for epistemic uncertainty.

| Phase | Standard inference | MC Dropout |
|---|---|---|
| Training | Dropout active (regularisation) | Identical — no change |
| Inference | `model.eval()` disables dropout | Dropout kept active |
| Forward passes | 1 (deterministic) | T (stochastic) |
| Output | Single prediction | Mean prediction + uncertainty map |

**Reference:** Gal, Y. & Ghahramani, Z. (2016). *Dropout as a Bayesian
Approximation: Representing Model Uncertainty in Deep Learning.* ICML 2016.
https://arxiv.org/abs/1506.02142

---

## Prerequisite: the model must have Dropout layers

Without Dropout layers, all T forward passes are identical and the uncertainty
map is uniformly zero.  A `UserWarning` is emitted at processor construction
time if no layers are found.

For SMP models, set `decoder_dropout` in the model config:

```yaml
model:
  _target_: segmentation_models_pytorch.Unet
  encoder_name: resnet50
  encoder_weights: imagenet
  in_channels: 3
  classes: 5
  decoder_dropout: 0.2   # <-- activates Dropout2d in the decoder
```

Other architectures that already have `Dropout2d` layers (e.g. HRNetOCR,
custom UNet with `dropout_2d`) work out of the box.

---

## Architecture compatibility

| Model type | MC Dropout compatible? | Notes |
|---|---|---|
| SMP models with `decoder_dropout` set | ✅ Yes | Set `decoder_dropout > 0` in config |
| SMP models without `decoder_dropout` | ⚠️ Degenerate | All samples identical; uncertainty = 0 |
| Custom UNet with `dropout_2d` | ✅ Yes | Already has Dropout2d |
| HRNetOCR | ✅ Yes | Internal Dropout2d layers |
| EDL (`EvidentialWrapper`) | ⚠️ Not recommended | Two separate uncertainty sources; semantics conflict |
| MoE models | ⚠️ Risky | Router noise + dropout may interact unpredictably |
| Detection / PolygonMapper | ❌ Out of scope | Multi-head outputs; aggregation undefined |

---

## Uncertainty modes

### `"entropy"` (default)

Predictive entropy of the mean probability distribution:

```
p̄(c|x)  = (1/T) Σ_t p_t(c|x)         # mean of T samples
H        = -Σ_c p̄_c · log(p̄_c)        # entropy of the mean
```

**What it measures:** Total uncertainty (epistemic + aleatoric combined).
High wherever the model is unsure — class boundaries, shadows, out-of-distribution regions.

**Range:** `[0, log(num_classes)]`

---

### `"mutual_information"` (BALD)

```
MI = H[p̄] − (1/T) Σ_t H[p_t]
```

**What it measures:** Epistemic uncertainty only — the disagreement between
the T samples, regardless of each sample's individual confidence.

High when the samples give very different predictions (the model genuinely
does not know).  Low when all samples agree, even if they are all uncertain.

**Range:** `[0, log(num_classes)]`

---

### When to use which

| Goal | Recommended mode |
|---|---|
| Filter low-quality predictions | `entropy` |
| Active learning: pick images to label | `mutual_information` |
| Out-of-distribution detection | `mutual_information` |
| General quality control | `entropy` |

---

## The uncertainty map

The uncertainty map is a **single-band float32 GeoTIFF** with the same
resolution and CRS as the input image.

- Each pixel contains a scalar value in `[0, log(num_classes)]`
- Low values (near 0) → model is confident
- High values → model is uncertain

### Visualising in QGIS

1. Open the `_mc_uncertainty.tif` file in QGIS
2. Layer Properties → Symbology → Render type: Singleband pseudocolor
3. Choose a colormap (e.g. *viridis* or *hot*)
4. Apply: dark regions = confident, bright regions = uncertain

---

## Usage: `MCDropoutInferenceProcessor`

### Without uncertainty map (faster)

Runs T stochastic forward passes and returns the **mean prediction** — more
robust than a single deterministic pass, with no extra output files.

```yaml
inference_processor:
  _target_: pytorch_segmentation_models_trainer.tools.inference.mc_dropout_inference_processor.MCDropoutInferenceProcessor
  n_samples: 10
  export_uncertainty_map: false   # default — no uncertainty file written
  num_classes: 5
  model_input_shape: [512, 512]
  step_shape: [256, 256]
```

### With uncertainty map

```yaml
inference_processor:
  _target_: pytorch_segmentation_models_trainer.tools.inference.mc_dropout_inference_processor.MCDropoutInferenceProcessor
  n_samples: 10
  uncertainty_mode: entropy          # or mutual_information
  export_uncertainty_map: true
  num_classes: 5
  model_input_shape: [512, 512]
  step_shape: [256, 256]
  output_uncertainty_dir: /results/uncertainty   # optional; defaults to seg dir
```

Output files per image:
- `{stem}.tif` — class-index segmentation (uint8, 1 band)
- `{stem}_mc_uncertainty.tif` — uncertainty map (float32, 1 band)

---

## Usage: TTA uncertainty in `MultiClassInferenceProcessor`

TTA also generates T predictions (one per augmentation).  When
`export_uncertainty_map=True` and a `tta_mode` is set, the processor
computes the uncertainty from those T TTA samples and exports a
`_uncertainty.tif` alongside the segmentation.

```yaml
inference_processor:
  _target_: pytorch_segmentation_models_trainer.tools.inference.inference_processors.MultiClassInferenceProcessor
  tta_mode: d4                       # "d4" (8 transforms) or "flip" (4)
  uncertainty_mode: entropy
  export_uncertainty_map: true
  num_classes: 5
  model_input_shape: [512, 512]
  step_shape: [256, 256]
```

### MC Dropout vs TTA uncertainty

| Source | Stochasticity origin | What it measures |
|---|---|---|
| MC Dropout | Random dropout masks | Model weight uncertainty |
| TTA | Geometric augmentations | Sensitivity to input orientation |

- **MC Dropout** uncertainty is high where the model's internal representations are unstable.
- **TTA** uncertainty is high where the model's output changes when the input is rotated or flipped — indicating lack of geometric invariance.

Both produce a `[0, log C]`-valued float32 GeoTIFF that can be interpreted
identically in QGIS.

---

## Sliding-window behaviour

In both processors, the uncertainty map is computed **per tile** and merged
back to full image resolution using the same Gaussian/pyramid importance
weighting as the segmentation output.  Tile overlap regions receive weighted
contributions from all tiles that cover them — the same mechanism that
eliminates border artefacts in the segmentation.

```
For each tile:
    T forward passes → T probability maps
    p_mean   → argmax → class prediction   (fed to seg TileMerger)
    uncertainty       → per-pixel scalar   (fed to uncertainty TileMerger)

After all tiles:
    Merge seg tiles   → full-image class-index raster
    Merge unc tiles   → full-image uncertainty raster
```

---

## Choosing `n_samples`

| `n_samples` | Inference time | Uncertainty quality |
|---|---|---|
| 5 | ~5× standard | Coarse estimate |
| 10 | ~10× standard | Good for most use cases |
| 30 | ~30× standard | High-quality estimate |
| 50+ | ~50×+ standard | Diminishing returns |

The mean prediction quality also improves with more samples, but converges
quickly (around 10–20 for typical segmentation models).

---

## Limitations

- **Large images:** The `MCDropoutInferenceProcessor` skips the striped
  inference path (used automatically for images > 50 MP) and falls back to
  standard tiled inference.  A warning is logged.  Reduce `model_input_shape`
  or use a machine with more VRAM if OOM occurs.
- **n_samples = 1:** Valid but equivalent to a single stochastic forward pass.
  `mutual_information` uncertainty will be zero; `entropy` gives single-sample
  entropy (not a MC estimate).
- **No training changes:** The model trains exactly as before.  You can enable
  MC Dropout on any existing checkpoint that was trained with Dropout.
