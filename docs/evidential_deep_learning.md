# Evidential Deep Learning (EDL)

## What is EDL?

Standard segmentation networks output class probabilities via Softmax — a single point estimate with no information about whether the network "knows" what it is seeing. Evidential Deep Learning (EDL) replaces this with a **Dirichlet distribution** over the probability simplex, parameterised directly by the network's output.

This makes it possible to distinguish two types of uncertainty:

| Type | Definition | When it is high |
|---|---|---|
| **Epistemic** (model) | u = K / S | Input pattern unseen during training (OOD) |
| **Aleatoric** (data) | Variance of Dirichlet | Ambiguous pixels even with infinite data |

The uncertainty map produced by EDL is particularly useful in remote sensing workflows where regions outside the training distribution (cloud shadows, new land-cover types, sensor artefacts) need to be flagged automatically rather than silently assigned an incorrect class.

**Reference:** Sensoy, M., Kaplan, L., & Kandemir, M. (2018). Evidential Deep Learning to Quantify Classification Uncertainty. *NeurIPS 2018*. https://arxiv.org/abs/1806.01768

---

## Theory in One Page

The network learns to output **evidence** per class per pixel:

```
evidence_k = Softplus(logit_k)    # e_k ≥ 0
alpha_k    = evidence_k + 1       # α_k ≥ 1 (Dirichlet parameter)
S          = Σ_k alpha_k          # total evidence strength
p̂_k       = alpha_k / S          # expected class probability
u          = K / S                # epistemic uncertainty ∈ (0, 1]
```

When the network has **no evidence** (unseen pattern), all `evidence_k ≈ 0`, so `alpha_k ≈ 1`, `S ≈ K`, and `u ≈ 1`. When it has **high evidence** for one class, `S >> K` and `u → 0`.

### Loss Function

```
L_total = L_MSE + λ_t · L_KL
```

**L_MSE** (integrated over Dirichlet, Sensoy eq. 4):
```
L_MSE = Σ_k [ (y_k - α_k/S)² + α_k(S-α_k)/(S²(S+1)) ]
       = bias²              + variance term
```

**L_KL** (regulariser, Sensoy eq. 8):
Before computing KL, evidence for the correct class is removed so the network is not penalised for *correct* high-confidence predictions:
```
α̃_k = y_k + (1 - y_k) · α_k
L_KL = KL[ Dir(α̃) || Dir(1,...,1) ]
```

**KL annealing** (λ_t): the KL coefficient starts at 0 and is linearly increased to 1.0 over training, ensuring the network first learns to discriminate classes before being penalised for residual wrong-class evidence. This is controlled by the standard `CompoundLoss` weight schedule in the YAML — no special code needed.

---

## Quick Start: 3 Steps

### Step 1 — Choose a YAML

| Scenario | Config file |
|---|---|
| Training from scratch | `conf/examples/edl_from_scratch.yaml` |
| Fine-tuning from a pre-trained checkpoint | `conf/examples/edl_finetune.yaml` |

Edit `hyperparameters.classes` to match your number of classes, and set `train_dataset.input_csv_path` / `val_dataset.input_csv_path` to your data.

### Step 2 — Train

```bash
python -m pytorch_segmentation_models_trainer.train \
    --config-name edl_from_scratch
```

During training you will see:
- `loss/train` and `loss/val` — total EDL loss
- `losses/train_edl_mse` and `losses/train_edl_kl` — individual components
- `edl/train_uncertainty` — mean epistemic uncertainty per batch

The uncertainty visualisation callback logs a 4-column diagnostic grid to TensorBoard / WandB / file system every N epochs:

```
[Input image | Predicted class | Uncertainty map (plasma colormap) | Ground truth]
```

### Step 3 — Export Uncertainty GeoTIFF

```bash
python -m pytorch_segmentation_models_trainer.predict \
    --config-name predict \
    model_path=/path/to/checkpoint.ckpt \
    image_path=/path/to/image.tif \
    output_path=/results/probs.tif \
    output_uncertainty_path=/results/uncertainty.tif
```

This produces:
- `probs.tif` — multi-band float32, one band per class (class probabilities)
- `uncertainty.tif` — single-band float32, values ∈ (0, 1], CRS and transform preserved

---

## Training From Scratch vs Fine-tuning

### Training from scratch (`freeze_encoder: false`)

The encoder is never frozen. The only warm-up mechanism is **KL annealing**:

| Epochs | KL weight | Effect |
|---|---|---|
| 0–9 | 0.0 | Network learns classes via MSE only |
| 10–39 | 0 → 1.0 | KL regularisation ramps in gradually |
| 40+ | 1.0 | Full EDL training |

Without annealing, the KL term would push all evidences toward zero in the first epoch ("I am uncertain about everything"), preventing the network from ever learning class discrimination. The annealing gives the network time to form good decision boundaries before the uncertainty calibration begins.

### Fine-tuning from pre-trained weights (`freeze_encoder: true`)

Pre-trained weights (e.g. ImageNet) already encode good features. The goal is to **re-interpret** the encoder's output as Dirichlet evidence without corrupting the learned representations:

| Phase | Epochs | Encoder | KL weight |
|---|---|---|---|
| 1 — Calibration | 0–4 | Frozen | 0.0 |
| 2 — Partial unfreeze | 5–14 | Last 2 stages free | 0 → 0.5 |
| 3 — Full | 15+ | All layers free | 0.5 → 1.0 |

The `EvidentialWarmupCallback` manages the encoder freeze schedule. The `EvidentialKLLoss` weight schedule in the YAML manages the KL annealing.

**Note:** models trained with Softmax do not need weight re-initialisation. The `EvidentialWrapper` applies `Softplus` to the same logits the encoder produces — the pre-trained features remain valid without any modification.

---

## Architecture

```
EvidentialWrapper
└── model (any SMP / HuggingFace / timm / custom model)
      ↓ forward(x)
    logits [B, K, H, W]
      ↓ Softplus
    evidence [B, K, H, W]  ≥ 0
      ↓ + 1
    alpha [B, K, H, W]     ≥ 1
      ↓ / S
    probs [B, K, H, W]     (expected class probabilities)
    uncertainty [B, 1, H, W]  = K / S  ∈ (0, 1]
```

The wrapper detects and handles:
- Plain tensor output (standard SMP, custom models)
- Tuple output (e.g. `(logits, aux_features)`)
- Dict output with `"out"` or `"seg"` key (torchvision DeepLab style)

---

## Interpreting the Uncertainty Map

| Uncertainty value | Interpretation |
|---|---|
| u ≈ 0.0 | Very high confidence — strong evidence for one class |
| u ≈ 0.5 | Moderate uncertainty — evidence spread across classes |
| u ≈ 1.0 | Maximum uncertainty — no evidence for any class (OOD region) |

In practice, values above 0.7–0.8 often correspond to:
- Cloud shadows or atmospheric artefacts
- Transition zones between classes
- Sensor artifacts (striping, saturation)
- Land-cover types not represented in training data

Use the uncertainty map as a **quality mask**: pixels where `u > threshold` can be flagged for manual review or excluded from downstream analysis.

---

## Configuration Reference

### `EvidentialWrapper`

| Field | Type | Default | Description |
|---|---|---|---|
| `model` | dict | MISSING | Nested model config (any segmentation model) |
| `freeze_encoder` | bool | `false` | Freeze encoder at init (set `true` for fine-tuning) |

### `EvidentialMSELoss`

| Field | Type | Default | Description |
|---|---|---|---|
| `name` | str | `"edl_mse"` | Loss name for logging |
| `num_classes` | int | MISSING | Number of segmentation classes K |
| `ignore_index` | int | `255` | Label value to exclude from loss |

### `EvidentialKLLoss`

| Field | Type | Default | Description |
|---|---|---|---|
| `name` | str | `"edl_kl"` | Loss name for logging |
| `num_classes` | int | MISSING | Number of segmentation classes K |
| `ignore_index` | int | `255` | Label value to exclude from loss |

### `EvidentialWarmupCallback`

| Field | Type | Default | Description |
|---|---|---|---|
| `warmup_epochs` | int | `5` | Epochs with encoder fully frozen (fine-tuning only) |
| `freeze_encoder` | bool | `false` | Must match `model.freeze_encoder` |
| `partial_unfreeze_epoch` | int | `10` | Epoch to fully unfreeze encoder |

### `EvidentialUncertaintyVisualizationCallback`

| Field | Type | Default | Description |
|---|---|---|---|
| `num_images` | int | `4` | Samples per logged grid |
| `log_every_n_epochs` | int | `5` | Log frequency |
| `norm_params` | dict | `null` | `{mean: [...], std: [...]}` for de-normalisation |

### `EvidentialInferenceProcessor`

| Field | Type | Default | Description |
|---|---|---|---|
| `output_uncertainty_path` | str | `null` | Path for uncertainty GeoTIFF output |
| `num_classes` | int | `2` | Number of classes K |
| `export_alpha` | bool | `false` | Also export K-band alpha parameters |
