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

## Methods

| `_target_` | Description |
|---|---|
| `pytorch_segmentation_models_trainer.few_shot.methods.base_only.BaseOnly` | Frozen base model; never predicts novel classes (floor). |
| `pytorch_segmentation_models_trainer.few_shot.methods.prototype.PrototypeImprinting` | Training-free: novel rows = L2-normalised support prototypes × `scale` (`base_norm`, `unit` or a number), zero bias. |
| `pytorch_segmentation_models_trainer.few_shot.methods.diam.DIaM` | DIaM (CVPR 2023), port of the official classifier. Transductive. |
| `pytorch_segmentation_models_trainer.few_shot.methods.classtrans.ClassTrans` | ClassTrans (CVPRW 2024), port of the official `TransitionClassifier`. Transductive. |
| `pytorch_segmentation_models_trainer.few_shot.methods.hisplit.HiSplit` | Hierarchical split of each mother among its children over the frozen base (category splitting). |

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

Support targets: pixels labelled with a child; and, in the S-novel regime,
pixels labelled "not novel" that the base model predicts as the mother become
examples of the child that keeps the mother's index ("free negatives";
noisy where the base model is wrong).

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
tile mean uncertainty and the tile error (1 − pixel accuracy).

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
- loss `650·CE + 3·H(p_query) + 16·KL(marginal‖π) + 7·KD`, the CE replaced
  by LDAM (margins ∝ n^-1/4, max 6) from iteration 101 of 130; SGD
  (`lr 9e-5`, momentum 0.9, weight decay 5e-4).

Not ported: the OpenEarthMap post-processing of `test.py` (vision-language
and CascadePSP masks, zeroed classes). Differences: one classifier per query
tile (the official code handles one query image), transition layers drawn
once from the support, LDAM counts from `class_counts` or from the support
labels (the official code hard-codes OpenEarthMap counts), query valid mask
= all pixels.

### Parity with the official code

`tests/test_few_shot_diam.py` and `tests/test_few_shot_classtrans.py` compare
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

## Full example

See `conf/examples/build_fewshot_episodes.yaml`,
`conf/examples/gfss_base_only.yaml` and the per-method configs
`gfss_prototype.yaml`, `gfss_diam.yaml`, `gfss_classtrans.yaml`, `gfss_hisplit.yaml` (Hydra
`defaults` on top of `gfss_base_only`).
