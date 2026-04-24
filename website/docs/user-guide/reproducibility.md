---
sidebar_position: 9
title: Reproducible Training
---

# Reproducible Training

Adding a `seed` to your training YAML guarantees that two runs with the same configuration, dataset, and hardware produce byte-identical results. This is essential for ablation studies, debugging, and comparing experiments.

---

## Quick Start

Add two lines to any existing config:

```yaml
seed: 42
deterministic_cudnn: false   # set true only when you need full GPU determinism
```

That is all. No other changes are required.

---

## What `seed` Controls

Setting `seed` calls `set_training_seed()` as the very first operation in `train()`, before any model or dataset is created. The following sources of randomness are seeded:

| Source | Mechanism |
|--------|-----------|
| Python `random` module | `random.seed(seed)` |
| NumPy (`np.random`) | `np.random.seed(seed)` — crop sampling, CutMix, ClassMix, class-balanced image selection |
| PyTorch CPU | `torch.manual_seed(seed)` — weight initialisation, Dropout |
| PyTorch CUDA | `torch.cuda.manual_seed_all(seed)` — GPU kernels |
| Python hash (`PYTHONHASHSEED`) | `os.environ["PYTHONHASHSEED"] = str(seed)` |
| DataLoader shuffle sampler | `torch.Generator().manual_seed(seed)` passed to every `DataLoader` |
| DataLoader workers | `_worker_init_fn` seeds `np.random` and `random` from `torch.initial_seed()`, which equals `seed + worker_id` — unique and deterministic per worker |

Because the seed is applied before `Model(cfg)` is called, **model weight initialisation** (SMP, timm, HuggingFace, custom architectures) is also reproducible.

---

## `deterministic_cudnn`

By default (`deterministic_cudnn: false`), CuDNN selects the fastest convolution algorithm at runtime. This choice can vary between runs, causing tiny floating-point differences on GPU.

Setting `deterministic_cudnn: true` disables this optimisation:

```yaml
seed: 42
deterministic_cudnn: true
```

This forces `torch.backends.cudnn.deterministic = True` and `torch.backends.cudnn.benchmark = False`. Training will be fully deterministic on GPU but **convolutions may be slower** on some architectures.

:::tip When to use `deterministic_cudnn`
Enable it when debugging a numerical issue or verifying that a code change has zero effect on model output. Leave it `false` for routine experiments and production training runs.
:::

---

## Full Example Config

See [`conf/examples/reproducible_training.yaml`](https://github.com/dsgoficial/pytorch_segmentation_models_trainer/blob/main/pytorch_segmentation_models_trainer/conf/examples/reproducible_training.yaml) for a complete annotated example.

The key additions to any existing config are:

```yaml
# ── Reproducibility ──────────────────────────────────────────────────────────
seed: 42
deterministic_cudnn: false

backbone:
  name: mit_b2
  input_width: 512
  input_height: 512

hyperparameters:
  batch_size: 8
  epochs: 50
  # ... rest of config unchanged
```

---

## Seeding in Existing Configs

The `seed` field is **optional**. Configs without it behave exactly as before — no seeds are set and no performance impact occurs. You can add it to any existing YAML:

```yaml
# smp_mit_b2.yaml (or any other config)
seed: 42        # add this line
deterministic_cudnn: false   # add this line (defaults to false if omitted)

backbone:
  # ... existing content unchanged
```

---

## Python API

If you use the library programmatically rather than through Hydra:

```python
from pytorch_segmentation_models_trainer.utils.seed_utils import set_training_seed
from torch.utils.data import DataLoader

# Call BEFORE creating any model or dataset
generator = set_training_seed(42)

# Pass generator to every DataLoader for reproducible shuffling
train_loader = DataLoader(train_dataset, shuffle=True, generator=generator)
val_loader   = DataLoader(val_dataset,   shuffle=False, generator=generator)
```

`set_training_seed` returns a `torch.Generator` seeded with the same value. Pass it to all `DataLoader` instances to make the shuffle order deterministic.

---

## Limitations

- **Resume from checkpoint**: When resuming with `resume_from_checkpoint`, the seed is still applied at the start of `train()`. This seeds weight initialisation (irrelevant for a loaded checkpoint) and resets the DataLoader sampler state. The training sequence from the resumed epoch onwards will be deterministic but different from the sequence that would have continued from the original run.
- **Multi-GPU (DDP)**: Each process calls `set_training_seed` independently from the same `cfg.seed`. PyTorch Lightning synchronises gradients across GPUs, so the training is still reproducible, but each GPU processes a different data shard — as expected.
- **Non-deterministic custom ops**: Third-party CUDA extensions or custom operators that do not honour PyTorch's determinism settings are outside the scope of this feature.
