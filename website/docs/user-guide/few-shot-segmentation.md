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

See `conf/examples/build_fewshot_episodes.yaml` and
`conf/examples/gfss_base_only.yaml`.
