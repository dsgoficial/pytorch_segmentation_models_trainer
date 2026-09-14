---
sidebar_position: 29
title: Mean-Teacher AIO2
---

# Mean-Teacher AIO2

AIO2 (Adaptively trIggered Online Object-wise correction) is a noisy-label
training baseline: a **mean-teacher** pair (student trained by backprop,
teacher an EMA copy) drives two modules — **ACT**, which detects without
ground truth when to stop trusting the noisy label, and **O2C**, which then
corrects it object-by-object using the teacher's predictions.

This implementation follows Liu, C., Albrecht, C. M., Wang, Y., Li, Q., Zhu,
X. X., *"AIO2: Online Correction of Object Labels for Deep Learning with
Incomplete Annotation in Remote Sensing Image Segmentation,"* IEEE TGRS,
2024 — cross-checked against the
[official reference implementation](https://github.com/zhu-xlab/AIO2), not
the paper text alone (several details below, like the ACT window sizes and
the two-phase checkpoint/resume design, are only visible in the code).

:::caution Noise-type mismatch
AIO2 is designed for **incomplete label noise** (a missing object — false
negative). If your noisy label instead has *every* pixel annotated but some
pixels carry the *wrong* class (e.g. geometric boundary misalignment), every
pixel already has a class, so O2C's "add the missed object" mechanism has no
natural empty region to add into — recovering an object there necessarily
means overwriting a different, already-assigned class. This baseline still
runs on that kind of noise (useful evidence that a false-negative-oriented
SOTA method doesn't solve a different noise type), but don't present it as a
like-for-like comparison without this caveat.
:::

---

## Components

| Component | Class | Purpose |
|-----------|-------|---------|
| Student/teacher pair | `MeanTeacherWrapper` | EMA-updated teacher, no gradient |
| Trigger | `ACTTracker` | Detects the warm-up → correction switch epoch, no ground truth needed |
| Correction | `online_object_correction` | Per-class connected-component "missed object" correction (O2C) |
| Model | `MeanTeacherModel` | Orchestrates both phases, EMA updates, checkpointing |

---

## How it works

### 1 — Mean teacher (§2.1)

The teacher starts as an exact copy of the student and is updated only by
EMA, never by gradient:

```
θ_teacher ← α·θ_teacher + (1-α)·θ_student      (α = 0.999 default)
```

Decay ramps up from a low value over the first ~10 steps (same warmup
formula as `EMACallback`), so early updates track the student closely
instead of being contaminated by shared random-init weights.

### 2 — ACT: when to stop trusting the noisy label (§2.3)

Each warm-up epoch, `ACTTracker` gets the teacher's training IoU (against
the still-noisy label) and:

1. Fits a local linear regression over several sliding window sizes
   (`[10, 20, 30, 40]` epochs by default) to estimate the accuracy curve's
   growth rate.
2. Detects `I_t` — the epoch where that growth rate stops shrinking (end of
   the "transition stage," right before the model starts memorizing noise).
3. Fits an exponential-saturation curve up to `I_t` to estimate `I_e` — the
   end of early learning.
4. Triggers at `I_r = floor((I_e + I_t) / 2)`.

No ground truth involved anywhere in this — it's inferred purely from the
shape of the (noisy) training accuracy curve.

### 3 — O2C: correcting the label, per class (§2.4, generalized)

The official algorithm is **binary** (one foreground class, building
footprints). This project's LULC data has 6 classes, so
`online_object_correction` runs the same connected-component overlap-check
**once per class** in `classes_to_correct`:

1. Connected components of `teacher_probs[c] > 0.5`.
2. Any component overlapping an existing class-`c` pixel in the noisy label
   is already "marked" — discarded.
3. Any component with **zero** overlap is a "missed object of class `c`" —
   proposed as a correction.

Two multiclass-only additions, neither present in the official (single-class)
code — both documented decisions, not silent extrapolations:

- **Conflict resolution.** Two classes can propose the same pixel (the
  official code never has this problem — only one class exists).
  `conflict_resolution: priority_order` lets the first class in
  `classes_to_correct` win; `conflict_resolution: teacher_confidence` lets
  the higher `teacher_probs[c]` win at that pixel instead — mirrors how
  `apply_region_correction` (SAM/SLICO) breaks ties by mask confidence. Both
  are implemented; a pilot run decides which goes into the full experiment
  matrix.
- **Fixed classes.** Every pixel whose *original* class isn't in
  `classes_to_correct` is restored after conflict resolution, discarding
  any proposal for it — replicated from `apply_region_correction`
  (`correction.py:83-85`): eligibility is gated by the pixel's *current*
  class, never the proposed one.

See `online_object_correction`'s docstring for the full writeup, including
the soft-boundary → hard-erosion adaptation (`o2c_filter_size`) needed
because this project's loss takes a hard target, unlike the official code's
binary sigmoid+BCE loss which can accept a continuous soft boundary.

---

## When to cache (`correct_base`)

`online_object_correction` runs per image (connected-component labelling has
no batched-GPU form — same constraint as the official code, which also
loops over the batch). Every training step under O2C does a forward pass
plus per-image CC work, which is real cost at `batch_size=64`.

Caching **across runs** doesn't make sense here: the correction is a
function of the teacher's weights at that exact point in training, and the
teacher's trajectory (hence its weights at "epoch N") differs between seeds
and reruns — a cached correction from one run is simply wrong for another.
This is unlike the SAM/SLICO mask cache elsewhere in this project: SAM is
*fixed* throughout training, so reusing its output is always valid; the
AIO2 teacher *co-evolves* with training, so there's no stable key to cache
against across independent runs.

What **is** available, and is what `correct_base` controls, is a form of
caching *within* a single run — the official code's own two supported
modes:

- `"iter"` (default): recompute every batch from the source network's
  current, live state. True to "online" in O2C's name — nothing is ever
  reused, exactly like the paper's default.
- `"epoch"`: take one frozen snapshot of the source network at the start of
  each epoch (`on_train_epoch_start`) and reuse it for every batch in that
  epoch — amortizes the forward + CC cost from per-batch to
  per-epoch-×-unique-images, at the cost of a correction that's up to one
  epoch stale relative to the teacher's latest EMA update.

Don't default to `"epoch"` speculatively — switch only if a pilot run shows
`"iter"`'s per-batch cost is the actual bottleneck (see
`plano_experimentos_detalhado.md` §5.2 in the article repository, which
already flags AIO2's per-iteration overhead as something to measure before
committing to the full run matrix).

---

## Two-phase orchestration

Unlike a normal single `Trainer.fit()` run, AIO2 needs **two** — mirroring
the official reference implementation's two separate CLI invocations
(`--resume`/`--resume_from_detection`), not an in-run weight rewind:

```python
from pytorch_lightning import Trainer
from pytorch_segmentation_models_trainer.model_loader.mean_teacher_model import (
    MeanTeacherModel,
    find_nearest_checkpoint,
)

model = MeanTeacherModel(cfg)

# Phase 1 — warm-up. Stops itself once ACT triggers.
trainer1 = Trainer(max_epochs=cfg.hyperparameters.epochs, ...)
trainer1.fit(model)
assert model.resume_epoch is not None  # ACT triggered; otherwise raise max_epochs

# Phase 2 — correction, from the nearest warm-up checkpoint.
ckpt = find_nearest_checkpoint(cfg.act_checkpoint_dir, model.resume_epoch)
model.o2c_active = True
trainer2 = Trainer(max_epochs=cfg.hyperparameters.epochs, ...)
trainer2.fit(model, ckpt_path=str(ckpt))
```

`Trainer.fit(model, ckpt_path=...)` restores optimizer state natively —
more correct than a manual weights-only reload would be.

A single `train.py` invocation on `conf/examples/mean_teacher_aio2.yaml`
only runs phase 1; write the driver script above for the full two-phase run.

---

## Configuration

```yaml title="conf/examples/mean_teacher_aio2.yaml"
pl_model:
  _target_: pytorch_segmentation_models_trainer.model_loader.mean_teacher_model.MeanTeacherModel

alpha: 0.999
correct_model: teacher              # or: student
correct_base: iter                  # or: epoch — see "When to cache" below
classes_to_correct: [1, 2]          # ordered — see conflict_resolution
conflict_resolution: priority_order # or: teacher_confidence
o2c_filter_size: -1                 # odd int > 0 to enable boundary erosion
act_window_sizes: [10, 20, 30, 40]
checkpoint_every_n_epochs: 5
act_checkpoint_dir: /data/outputs/mean_teacher_aio2/warmup_checkpoints

loss:
  _target_: pytorch_segmentation_models_trainer.custom_losses.loss.WeightedDiceCrossEntropyLoss
  num_classes: 6
  ignore_index: 255
```

No new loss class — `WeightedDiceCrossEntropyLoss` (CE + Dice) already
covers what AIO2 needs.

---

## API Reference

### `ACTTracker`

```python
from pytorch_segmentation_models_trainer.utils.act_trigger import ACTTracker

tracker = ACTTracker(window_sizes=[10, 20, 30, 40])
for epoch_iou in teacher_train_iou_per_epoch:
    i_r = tracker.update(epoch_iou)
    if i_r is not None:
        break  # trigger fired; i_r is the epoch to resume training from
```

### `online_object_correction`

```python
from pytorch_segmentation_models_trainer.utils.o2c_correction import (
    online_object_correction,
)

corrected = online_object_correction(
    noisy_hard,            # (H, W) long
    teacher_probs,         # (C, H, W) float, softmax probabilities
    classes_to_correct=[1, 2],
    conflict_resolution="priority_order",
    filter_size=-1,
)
```

### `MeanTeacherModel`

Drop-in replacement for `Model` — same `cfg.model`/`cfg.loss`/`cfg.optimizer`
contract, plus the AIO2-specific fields above.

```yaml
pl_model:
  _target_: pytorch_segmentation_models_trainer.model_loader.mean_teacher_model.MeanTeacherModel
```

`forward()` (used for the base class's own metrics, and by
`trainer.predict`/inference) uses the **student**, not the teacher — keeps
evaluation on the same footing as every other condition in the experiment
matrix. Not fixed by the paper; revisit if a pilot suggests otherwise.

---

## Experiment matrix context

| Condition | Label source | What it tests |
|-----------|--------------|----------------|
| Base | TVS hard label | Baseline |
| SAM-all / SLICO-all | Region-corrected | This project's proposal (+ ablation) |
| [Zhu2019](boundary-label-relaxation.md) | TVS hard label + relaxed loss | SOTA noisy-label baseline #1 |
| **AIO2** (this doc) | TVS hard label + online object correction | SOTA noisy-label baseline #2 |

Zhu2019 is a pure loss swap (cheapest to add, no second model). AIO2 is the
most involved: a second network, a two-phase run, and a noise-type mismatch
worth flagging in the write-up (see the caution box above).
