---
sidebar_position: 30
title: Generalized Few-Shot Segmentation
---

# Generalized few-shot segmentation (GFSS)

A segmentation model trained on **base** classes is adapted with a few labelled
**support** windows to also predict **novel** classes, and evaluated on a
**query** set with all classes. The framework covers:

- standard GFSS (DIaM, ClassTrans), where novel classes were *background*
  during base training;
- **category splitting**, where a novel class was part of a base class during
  base training (e.g. *low vegetation* → *grassland* + *cropland*).

Both are described by a `ClassHierarchy`: each novel class is a child of a
base **mother** class.

## Pipeline

1. Train the base model normally (e.g. with the [Experiments Runner](./experiments_runner.md),
   merging the novel class into its mother with
   [`mask_class_mapping`](./mask-class-mapping.md)).
2. Draw fixed support episodes: `mode: build-fewshot-episodes`.
3. Run the GFSS method with `pl_model: GFSSModel` for every base seed × episode:
   `mode: run-experiments` with an `episodes` block.

## 1. Support episodes

```yaml
mode: build-fewshot-episodes
fewshot_episodes:
  window_index_cache: /data/tiles_data/train_pampa.csv
  mask_base_path: /data/masks
  novel_classes: [5]
  shots: [1, 3, 5, 10]
  n_draws: 5
  min_novel_fraction: 0.05
  seed: 2026
  output_csv: outputs/episodes/pampa.csv
```

A window is eligible for a novel class when the class covers at least
`min_novel_fraction` of its valid (non-`ignore_index`) pixels. For each draw
and novel class the eligible windows are shuffled once and the first K are
taken for every K, so supports are **nested** (K=1 ⊂ K=3 ⊂ …) and identical
for every method. The window index may use pixel or bounds columns (same
formats as `MBTilesMaskWindowedDataset`).

### Diversity- and uncertainty-driven supports (optional)

By default supports are drawn at random. A `selection` block computes, with a
frozen base model and **without labels**, one embedding and one uncertainty
per window of the pool, and can pick diverse supports:

```yaml
fewshot_episodes:
  # ... as above
  selection:
    method: kcenter            # random (only measures) | kcenter
    uncertainty: null          # null | base_entropy | base_vacuity | base_dissonance
    gamma: 1.0                 # weight u^gamma in the k-center score
    pooling: superclass        # superclass | tile
    mothers: [3]
    model: {_target_: segmentation_models_pytorch.UPerNet, encoder_name: resnet50, encoder_weights: null, classes: 5}
    base_checkpoint: {from_runner: outputs/baselines/r2_pampa, seed: 42}
    dataset: {...}             # image dataset over the same windows (window_index_cache is injected)
```

- **Embedding:** mean decoder feature over the pixels the base model predicts
  as a mother (`superclass`, tile mean if none) — the diversity of what has to
  be split — or over the whole tile (`tile`); L2-normalised.
- **Uncertainty:** mean over the same pixels of the normalised entropy of the
  base probabilities, and for an evidential base its vacuity and dissonance.
- **`kcenter`:** for each draw, the first window is random among the eligible
  ones and the others follow the greedy farthest-point order, maximising
  `distance to the selected set × u^gamma` (`uncertainty: null` or `gamma: 0`
  = pure diversity). Supports stay nested across K.
- Output columns: `selection`, `diversity` (Vendi score of each support's
  embeddings, cosine kernel: 1 = all alike, K = all different) and
  `uncertainty`. With `method: random` the block only **measures** the
  diversity of the random supports (for correlating it with the results).

## 2. `GFSSModel`

`GFSSModel` is a LightningModule run by the regular `train()` entry point:

| Config | Role |
|---|---|
| `model` | Base model, exactly as trained (e.g. 5 classes). Its `segmentation_head[0]` must be a 1×1 convolution (UPerNet, DeepLabV3+, FPN). smp U-Net has a 3×3 head and is rejected. |
| `gfss.base_checkpoint` | `path`, or `from_runner` (base experiment `output_base_dir`; the run with the same `seed` is used). Loaded with strict `load_state_dict`. |
| `gfss.hierarchy` | `{mother: [children]}`, e.g. `{3: [3, 5]}`; several mothers allowed: `{3: [3, 5], 4: [4, 6]}`. Novel indices must be contiguous after the base classes and every mother must list itself. |
| `gfss.method` | `_target_` of a `BaseGFSSMethod` subclass and its hyperparameters. |
| `train_dataset` | Support set (injected per episode by the runner). |
| `test_dataset` | Query set, with masks of the final classes. |
| `val_dataset` | Optional. |

The base model is frozen (and kept in eval mode). Methods work on the decoder
features (`FrozenLinearHeadSegmenter.features`) and the base classifier
(`weight`, `bias` of the 1×1 head):

- `trainer.fit` = adaptation. The method is initialised from the whole support
  set, then `support_loss` is minimised for `pl_trainer.max_steps` steps. Use
  `max_steps: 0` for methods without trainable parameters.
- Transductive methods are adapted again on each test batch inside
  `test_step`; set `pl_trainer.inference_mode: false`.
- Do not add a `ModelCheckpoint` callback (`enable_checkpointing: false`): the
  adapted state lives in memory.

### Test-time augmentation

`GFSSModel` honours the same TTA options as the base `Model` (`tta_mode: d8`
/ `d4` / `flip`, or `use_tta` + `tta_augmentations`), **at test time only**:
the method's logits, every decoding variant and the unmodified base model
(reference for `locality`/`split_ceiling`) are averaged over the
de-augmented views with the framework's `apply_tta`. Transductive methods
adapt once on the original view and predict on all views; uncertainty maps
come from the original view.

```yaml
tta_mode: d8   # same as the base models, so GFSS and R1/R2 are comparable
```

### Support label regimes

- **Full labels**: support masks with every class.
- **S-novel** (only the novel class is labelled): send base classes to
  `gfss.not_novel_index` (default 254) in the support dataset:
  `mask_class_mapping: {0: 254, 1: 254, 2: 254, 3: 254, 4: 254}`. Methods read
  254 as "labelled, not any novel class".

### Metrics

`GFSSMetrics` logs, as `test/...` (and `val/...`):

| Key | Meaning |
|---|---|
| `iou/<c>`, `precision/<c>`, `recall/<c>` | Per class (`nan` if absent). |
| `miou`, `miou_base`, `miou_novel` | Means over present classes. |
| `oem_score` | `0.4·miou_base + 0.6·miou_novel` (OpenEarthMap Few-Shot Challenge). |
| `locality` | Accuracy of the adapted model ÷ accuracy of the base model on pixels of untouched classes (base classes that are not mothers). 1 = unchanged. |
| `split_ceiling/<c>` | For a mother and its children: fraction of true pixels of `c` that the base model assigns to the mother — upper bound of any method that only splits the mother. |
| `boundary/iou/<c>`, `boundary/miou{,_base,_novel}` | Only with `gfss.boundary_width: w > 0`: IoU restricted to the *trimap* band — pixels within `w` px of a ground-truth label change (ignore pixels and the tile border are not boundaries). Measures boundary quality, e.g. of the novel child against its sibling. |

## Methods

| `_target_` | Description |
|---|---|
| `pytorch_segmentation_models_trainer.few_shot.methods.base_only.BaseOnly` | Frozen base model; never predicts novel classes (floor). |
| `pytorch_segmentation_models_trainer.few_shot.methods.prototype.PrototypeImprinting` | Training-free: novel rows = L2-normalised support prototypes × `scale` (`base_norm`, `mother_norm`, `unit` or a number), bias `zero`, `base_mean` or `mother`. With `mother_norm` + `mother`, a pixel is novel iff it is closer (cosine) to the novel prototype than to the mother's row (uses the hierarchy). |
| `pytorch_segmentation_models_trainer.few_shot.methods.diam.DIaM` | DIaM (CVPR 2023), port of the official classifier. Transductive. |
| `pytorch_segmentation_models_trainer.few_shot.methods.classtrans.ClassTrans` | ClassTrans (CVPRW 2024), port of the official `TransitionClassifier`. Transductive. |
| `pytorch_segmentation_models_trainer.few_shot.methods.bcm.BCM` | BCM (NeurIPS 2024), port of the official classifier: per mapped base class, a logistic regression on the frozen features; the base prediction is overwritten only where it is that class. |
| `pytorch_segmentation_models_trainer.few_shot.methods.finetune.FineTune` | Fine-tuning baselines (B1a isolated head, B1b full ± KD, LoRA) on the support. |
| `pytorch_segmentation_models_trainer.few_shot.methods.hisplit.HiSplit` | Hierarchical split of each mother among its children over the frozen base (category splitting). |

### Fine-tuning baselines (`FineTune` + `gfss.backbone`)

`FineTune` is transfer learning on the K support tiles, used as the
reference GFSS methods must beat:

```yaml
gfss:
  backbone:
    trainable: none          # none | decoder | all | lora
    # lora: {r: 8, alpha: 16, target_modules: [qkv]}
  method:
    _target_: pytorch_segmentation_models_trainer.few_shot.methods.finetune.FineTune
    train_rows: children     # children | novel | all
    init: mother             # mother | prototype
    kd_weight: 0.0           # > 0: hierarchical KD to the frozen base model
pl_trainer:
  max_steps: 200
```

| Baseline | `gfss.backbone.trainable` | `train_rows` | `kd_weight` |
|---|---|---|---|
| B1a — isolated fine-tuning of the split | `none` | `children` | 0 |
| B1b — full fine-tuning | `all` | `all` | 0 |
| B1b + KD | `all` | `all` | > 0 |
| M3 — LoRA (transformer encoder, e.g. Swin-T) | `lora` | `all` | 0 or > 0 |

**How this differs from the framework's `fine_tuning` strategies.** The
`Model`'s `fine_tuning.strategy` (`full`, `freeze_backbone`,
`linear_probe`, `lora`; configured by the `fine_tuning` node, see
`config_definitions/fine_tuning_config.py::FineTuningConfig`)
decides *which parameters train* in an ordinary training run with the loss
in `cfg.loss` and a head with the number of classes of `cfg.model`. The
GFSS fine-tuning baselines need more than that, so they live in
`GFSSModel` + `FineTune`:

| Aspect | Framework `fine_tuning` | GFSS `FineTune` |
|---|---|---|
| 5 → 6 classes | a new `classes: 6` head is randomly initialised (the 5-class checkpoint does not load into it) | base rows copied from the checkpoint; novel rows initialised from the **mother** row (or the scaled prototype) |
| Which head rows train | whole head | only the superclass children (`children`, isolated fine-tuning), only the novel rows, or all |
| Support labelled "only the novel class" (`not_novel_index`) | plain CE cannot use it (pixels would have to be ignored) | CE projected as in DIaM: "not novel" pixels use the sum of base probabilities |
| Forgetting | no distillation to the old model (the lib's `KnowledgeDistillationLoss` is plain KD) | hierarchical KD: KL(novel summed into its mother ‖ frozen base model) |
| BatchNorm | trained in train mode | statistics frozen (K = 1–10 support tiles) |
| LoRA | `get_peft_model` wraps the whole model | `peft.inject_adapter_in_model` into the encoder, in place (keeps `encoder`/`decoder` separable) |
| Evaluation | `Model` metrics | GFSS metrics (`locality` against the **unmodified** base model, `split_ceiling`, OEM score) and the episodes axis of the runner |

What is shared: `decoder`/`all` unfreeze the backbone with the framework's
`fine_tuning.lora_utils.freeze_modules_by_name` (same mechanics as
`freeze_backbone`/`full`), and LoRA uses the same `peft` package (extra
`transformers`). With a trainable backbone, `GFSSModel` keeps a frozen copy
of the base model for the KD snapshot and for the base predictions used by
`locality`, and optimises the method's parameters plus the unfrozen
backbone parameters with `cfg.optimizer`.

### HiSplit

Only the split of each mother `m` is learned:
`p(child) = p_base(m) · q(child | x)` with `Σ q = 1` over the children of `m`;
every other class keeps `p_base`. Variants of `q` (`q:`):

| `q` | Scores | Training |
|---|---|---|
| `proto` | `tau · cos(f, μ_child)` (support prototypes) | none (`max_steps: 0`) |
| `proto_prob` | diagonal Gaussian log-likelihood; variance `shared` by the children of a mother or `per_class`, floored at `var_floor` | none |
| `linear` | `w·f + b`, initialised as `proto` | support CE in `trainer.fit` |
| `trans` | `linear` + per-query-tile SGD on support CE + entropy and KL(q̄‖π) weighted by `p_base(m)` | fit + transductive (`inference_mode: false`) |
| `edl` | evidential pair head: `e = softplus(w·f + b)`, `α = e + 1`, `q = α / S` within each mother | EDL MSE + KL regulariser (annealed over `kl_anneal_steps`) in `trainer.fit` |
| `logreg` | BCM's classifier (cross-validated logistic regression, class balancing) on the HiSplit support targets of each mother | fitted at `init_from_support` (`max_steps: 0`) |

Options for comparisons with BCM and H²EDL (defaults keep the original
HiSplit): `feature_power` (Tukey's power on the features used by q),
`negatives: superclass | all` ("not novel" support pixels are negatives of
the split only where the base model predicts the mother — default — or
everywhere, as BCM's g_β; `all` needs one mother) and, for `q: edl`,
`edl_base_rate: inverse_frequency` (`a_c ∝ (n_c + s)^-τ` from the support
targets, `α = e + K·a`, as H²EDL's base-rate variant).

Support targets: pixels labelled with a child; and, in the S-novel regime,
pixels labelled "not novel" that the base model predicts as the mother become
examples of the child that keeps the mother's index ("free negatives";
noisy where the base model is wrong).

#### Boundary of the superclass (P2)

The split alone cannot recover novel pixels that the base model assigned to
another class. Options of `HiSplit`:

| Option | Effect |
|---|---|
| `leak: true` (`leak_init`) | Hierarchical leak ("HierTrans"): learned `t[n, c] = sigmoid(θ)` moves mass from each base class `c` (not the mother) into novel child `n`: `p'(n) = p_m q_n + Σ_c t p_c`, `p'(c) = p_c (1 − Σ_n t)`. Trained in `trainer.fit` (`max_steps > 0`) with the NLL of the final distribution on the support (novel/base labels `−log p'`, "not novel" `−log(1 − Σ p'(novel))`). Hierarchical decoding first decides the superclass, which absorbs the leaked mass, then the split. A pixel outside the superclass either keeps the base model's class or becomes a child: all classes outside the superclass get the same per-pixel shift `log(1 − T_top)`, so uneven leak fractions never swap neighbours. |
| `widen: prob` or `dissonance`, `widen_threshold` | Superclass widened: where the mother is the base model's second choice and `p''(m) ≥ threshold` (`prob`) or the base dissonance is `≥ threshold` (`dissonance`, evidential base), the pixel becomes the novel child if q prefers a novel child; otherwise the base decision is kept. |
| `novel_prior_weight` | Multiplies the novel shares of q before renormalisation (`> 1` favours novel). |
| `sweep: [τ…]` | Extra decoding variants `widen_<τ>` evaluated in the same pass (`test/var/widen_<τ>/...`): the preservation (`locality`) × correction (novel IoU/recall) curve. Requires `widen`. |

`ClassTrans(hierarchical_mask=true)` is the matching ablation: the transition
into each novel class only comes from its mother's column.

Methods may expose extra decoding variants with `variant_names()` /
`decode_variants(features)`; `GFSSModel` evaluates each with its own
`GFSSMetrics` under `<stage>/var/<name>/`.

#### Evidential base model and uncertainty

With a base model trained with the lib's `EvidentialWrapper` (Dirichlet
head), set `model` to the wrapper exactly as in its training config; the
GFSS model unwraps it and the methods receive `base_output: evidential`.
HiSplit then uses the Dirichlet mean as `p_base` and splits the mother's
evidence **and** base rate by `q` (`e_c = e_m·q_c`, `a_c = a_m·q_c`), so
`α_c = α_m·q_c` (exact Dirichlet aggregation) and beliefs stay non-negative.

Methods that estimate uncertainty (HiSplit) are evaluated automatically,
logged as `test/unc/...`:

| Map | When |
|---|---|
| `split_entropy` | always: normalised entropy of the split of the mother predicted by the base model |
| `split_vacuity` | `q: edl`: `n_children / S` of the pair head |
| `base_vacuity`, `dissonance` | evidential base: vacuity `K/S` of the base and Jøsang dissonance of the split opinion |

For each map: `aurc/<map>` — area under the risk–coverage curve of the split
decisions (pixels whose true class is a child and that the base model assigns
to its mother; abstaining = sending them back to the mother), and
`tile_mean/<map>`, `tile_spearman/<map>` — Spearman correlation between the
tile mean uncertainty and the tile error (1 − pixel accuracy). Also:
`coverage@<t>/<map>` and `risk@<t>/<map>` (abstention operating points:
decisions with `u ≤ t` retained, the rest sent back to the mother) and
`ece/<map>` (calibration of the confidence `1 − u`). Options in
`gfss.uncertainty_eval`:

```yaml
gfss:
  uncertainty_eval:
    abstain_thresholds: [0.25, 0.5, 0.75]
    ece_bins: 10
    n_bins: 1000
```

`HiSplit.abstention(features, measure, threshold)` returns the abstention
map itself: pixels predicted as a mother whose split uncertainty exceeds the
threshold (their label stays the mother class).

`decoding: hierarchical` (default) keeps the base argmax and splits only the
pixels predicted as a mother, so the predictions of all other classes are
those of the base model (`locality` = 1 up to ties within `decode_eps`);
`decoding: flat` takes the argmax of `log p` over all classes, where the split
mass can lose to neighbour classes.

### DIaM

Port of `src/classifier.py` of the [official repository](https://github.com/sinahmr/DIaM).
For every query batch, one linear classifier per query tile (base rows +
novel rows initialised with normalised support prototypes) is optimised for
`adapt_iter` SGD steps (no momentum) on
`w_ce·CE_support + w_kl·KL(marginal‖π) + w_ent·H(p_query) + w_kd·KD`
(`weights: [100, 1, 1, 100]`, `lr: 1.25e-3`, `adapt_iter: 100`, π estimated by
the model and re-estimated at iteration 10). The support CE projects "not
novel" pixels onto the sum of base probabilities (Eq. 5); the KD sums each
novel class into its **mother** (the background in the original).

Differences from the official code: novel classes go to their mother instead
of the background; all query pixels are treated as valid (the official test
reads the query ground truth to drop ignored pixels).

### ClassTrans

Port of the [official code](https://github.com/earth-insights/ClassTrans),
which differs from the paper (the paper's loss is LDAM + λ·L_π without KD).
Faithful to the code:

- novel rows initialised by entropic optimal transport (Sinkhorn) between
  support novel features and base-predicted regions;
- logits = classification branch + `layer_scale ⊙ (S(f) · W_base f)`,
  `S = (W_c f + b_c) ⊗ (W_r f + b_r)`, with `layer_scale` starting at 0;
- LDAM counts: `class_counts` (all classes), or `base_class_counts` (list or
  the JSON of `mode: count-class-pixels` on the training split with the base
  mapping) + novel counts from the support — as in the paper ("estimated via
  D_train and D_support"); default: counted on the support only;
- loss `650·CE + 3·H(p_query) + 16·KL(marginal‖π) + 7·KD`, the CE replaced
  by LDAM (margins ∝ n^-1/4, max 6) from iteration 101 of 130; SGD
  (`lr 9e-5`, momentum 0.9, weight decay 5e-4).

Not ported: the OpenEarthMap post-processing of `test.py` (vision-language
and CascadePSP masks, zeroed classes). Differences: one classifier per query
tile (the official code handles one query image), transition layers drawn
once from the support, LDAM counts from `class_counts` or from the support
labels (the official code hard-codes OpenEarthMap counts), query valid mask
= all pixels.

### BCM

Port of `src/bcm.py` of the [official repository](https://github.com/IBM/BCM)
(Sakai et al., "A Surprisingly Simple Approach to Generalized Few-Shot
Semantic Segmentation", NeurIPS 2024):

- **mapping** — `mined` (official): for each novel class, the top-`top_k` base
  classes predicted by the base model on its support pixels; `hierarchy`: the
  novel class's mother;
- **g_β** — for each mapped base class β, a logistic regression on the frozen
  features of all support pixels ("not novel" = 0 vs the novel classes of β),
  C chosen in `logspace(-5, 5, n_C)` by stratified `n_splits`-fold mean
  average precision, with class balancing (`sampling: us | os | bg`);
- **inference** — the base prediction is kept, except where it is β and g_β
  predicts a novel class; decided at the output resolution after upsampling
  the base and g_β probabilities separately (method hook `decode_to`).

Differences: `sklearnex` replaced by `sklearn` (same API), a seeded
`RandomState` instead of NumPy's global generator. Paper setting (§4.5):
`beta: 0.5` (Tukey's ladder of powers; needs non-negative features) and
`ensemble: true` (one model per support tile, weight 1, plus the all-tiles
model, weight `ensemble_full_weight` = 5). The ensemble's one-shot datasets are
single support tiles (official: the i-th shot of every novel class — the same
with one novel class); the official weight list gives 5 only for K = 5.
Like HiSplit's hierarchical decoding, BCM keeps every base class outside the
mapping exactly as predicted by the base model (its Proposition 4.1).

### Parity with the official code

`tests/test_few_shot_diam.py`, `tests/test_few_shot_classtrans.py` and
`tests/test_few_shot_bcm.py` compare
the ports with tensors produced by the official classifiers on small random
inputs, with the standard GFSS hierarchy `{0: [0, novel...]}` (background
as mother): prototypes / transport initialisation and final logits match
(`rtol 1e-4`).

### Writing a method

```python
from pytorch_segmentation_models_trainer.few_shot.base_method import BaseGFSSMethod

class MyMethod(BaseGFSSMethod):
    def __init__(self, lr_scale: float = 1.0, **kwargs):
        super().__init__(**kwargs)
        ...

    def init_from_support(self, features, masks):   # whole support set
        ...

    def support_loss(self, features, masks):         # optional
        ...

    def forward(self, features):                     # (B, num_classes, h, w)
        ...
```

`self.hierarchy`, `self.base_weight`, `self.base_bias` and
`self.not_novel_index` are available after `setup`. Masks reach the method
downsampled (nearest) to the feature resolution.

## Refinement-aware base training

For category splitting the base model is trained with the future children
merged into their mother. Cross-entropy then has no reason to keep the
structure *inside* the mother, which is what the few-shot split needs.
`RefinementAwareWrapper` (`custom_models/refinement_aware.py`) wraps an smp
model (encoder → decoder → 1x1 head) and adds two **label-free** auxiliary
losses on the decoder features:

| Term | What it does | Options |
|---|---|---|
| `edge` | A 1x1 head predicts the **Canny** edge map of the input image (kornia, computed on the fly from the de-normalized image and max-pooled to the feature resolution); BCE with `pos_weight = #neg/#pos` (capped). | `edge_weight`, `edge_region: all \| superclass`, `edge_low_threshold`, `edge_high_threshold`, `edge_max_pos_weight`, `normalization_mean/std` |
| `subproto` | `M` learnable sub-prototypes of the superclass; superclass pixels get a softmax of cosine similarities / temperature. Loss = mean per-pixel entropy (sharp) − entropy of the mean assignment (balanced) + `separation_weight`·mean `relu(cos(P_j,P_k) − margin)` (separated). | `subprototype_weight`, `num_subprototypes`, `temperature`, `separation_margin`, `separation_weight` |

The forward returns plain logits, so metrics, TTA and checkpoints are those
of the wrapped model. `Model` adds every model exposing
`compute_auxiliary_losses(masks)` to the loss and logs
`losses/<stage>_edge` / `losses/<stage>_subproto`. For GFSS, use the same
`model` node when loading the base checkpoint: `FrozenLinearHeadSegmenter`
calls `gfss_unwrap()` and uses the inner model (the edge head and the
prototypes are not used after training). Example:
`conf/examples/refinement_aware_base.yaml`.

## Separability probe

`mode: separability-probe` (`tools/few_shot/separability_probe.py`) is a
diagnostic of the frozen base features, not a GFSS method: for each base
seed and support episode it collects the decoder features of the
ground-truth pixels of two classes (e.g. field 3 × cultivated 5, which the
base model saw merged) on the **fully labelled** support tiles, fits

* `logreg` — standardized features + L2 logistic regression (balanced), and
* `proto` — cosine to the two class means (training-free),

and scores them on the same two classes over the test windows (random pixel
subsample, `test_keep_fraction`, capped per class). The CSV has one row per
base seed × (K, draw) with `n_support_*`, `n_test_*`, `auroc_*`,
`balacc_*` and `ap_logreg`. Comparing base models (e.g. plain vs
refinement-aware training) on the same episodes shows whether the training
kept the structure inside the superclass. The datasets must return the
unmerged labels (no `mask_class_mapping`). Example:
`conf/examples/separability_probe.yaml`.

## Full example

See `conf/examples/build_fewshot_episodes.yaml`,
`conf/examples/gfss_base_only.yaml` and the per-method configs
`gfss_prototype.yaml`, `gfss_diam.yaml`, `gfss_classtrans.yaml`, `gfss_hisplit.yaml` (Hydra
`defaults` on top of `gfss_base_only`).
