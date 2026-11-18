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
(`--resume`/`--resume_from_detection`), not an in-run weight rewind.

**`train()` does this automatically** whenever `pl_model` is
`MeanTeacherModel` — a plain `mode: train` run, or `mode: run-experiments`
(one two-phase run per seed), needs no driver script:

1. **Warm-up** (`lightning_logs/version_0`): trains on the noisy label with
   the config's callbacks **minus `EarlyStopping`** — when to stop warming
   up is ACT's decision, not validation patience's. Saves
   `epoch_{N}.ckpt` every `checkpoint_every_n_epochs` and stops itself
   once ACT reports `I_r`.
2. **Correction** (`lightning_logs/version_1`): resumes from the warm-up
   checkpoint holding the state after `I_r` epochs (or the nearest one
   *before* it — never after), with `o2c_active = True` and the config's
   callbacks **unchanged**, so it stops like any other condition (early
   stopping, `max_epochs`).
3. `trainer.test()` uses phase 2's best checkpoint.

Under the hood this is `fit_two_phase(model, make_trainer)`, usable from
your own script too:

```python
from pytorch_segmentation_models_trainer.model_loader.mean_teacher_model import (
    MeanTeacherModel,
    fit_two_phase,
)

model = MeanTeacherModel(cfg)
trainer, info = fit_two_phase(model, lambda phase: Trainer(...))
# info: {"aio2/act_triggered": 1.0, "aio2/warmup_epochs": ...,
#        "aio2/resume_epoch": I_r, "aio2/resume_checkpoint_epoch": ...}
```

Details worth knowing:

- **Callback state is not carried across.** The warm-up checkpoint is
  copied without its `callbacks` entry (`prepare_phase2_checkpoint`)
  before phase 2 resumes from it. Otherwise phase 2's `ModelCheckpoint`
  would inherit the warm-up's best score and `trainer.test()` could end up
  evaluating a warm-up (pre-correction) model. Weights (student + EMA
  teacher), optimizer, LR scheduler and epoch/step counters *are* restored
  (`Trainer.fit(ckpt_path=...)`).
- **Epoch conventions.** `I_r` is a 1-indexed count of completed epochs;
  checkpoint files are named by Lightning's 0-indexed `current_epoch`
  (`epoch_9.ckpt` = state after 10 epochs). The driver converts between the
  two; `aio2/resume_checkpoint_epoch` is reported in the same convention as
  `aio2/resume_epoch`.
- **`act_checkpoint_dir` is optional.** When unset, warm-up checkpoints go
  to `<pl_trainer.default_root_dir>/act_warmup_checkpoints` — per seed under
  `ExperimentsRunner`, which is what you want for multi-seed runs (a fixed
  directory shared by all seeds would let one seed resume from another's
  warm-up).
- **If ACT never triggers** within `max_epochs`, phase 2 is skipped, the run
  is a plain mean-teacher model, and `aio2/act_triggered` is `0`. The
  `aio2/*` values land in each seed's `_seed_result.json` and in
  `summary.csv` next to the test metrics, so such runs are visible.
- **Epoch budget.** With `act_window_sizes: [10, 20, 30, 40]` ACT cannot
  fire before ~65 epochs of history (largest window + its `mean(W)` buffer),
  so `max_epochs` must be well above that.
- **Use a constant learning rate with ACT.** ACT waits for the slope of the
  teacher's training IoU to rise again (memorization starting). A decaying
  schedule such as `CosineAnnealingLR` makes that slope fall monotonically,
  so ACT never fires; the official code trains with a fixed learning rate.

### Alternative trigger: `phase2_trigger: val_best`

`phase2_trigger: act` (default) is the paper. Some label noise never
produces the curve shape ACT looks for: with class-swap noise in a
multiclass land-cover set, the teacher's training IoU kept rising smoothly,
memorization was gradual, and ACT did not fire within 116 epochs at a
constant learning rate while validation IoU had peaked at epoch 35.

`phase2_trigger: val_best` replaces ACT with the validation set:

- the warm-up **keeps `EarlyStopping`** and stops like any other run;
- phase 2 restarts from the warm-up `ModelCheckpoint`'s **best model**
  (callback state stripped as above) with O2C on;
- ACT is not consulted and no `act_warmup_checkpoints/` are written
  (`act/teacher_train_iou` is still logged).

This is an adaptation, not AIO2 as published: AIO2 assumes no clean labels,
and this mode uses the validation labels to decide when to start correcting.
It is fair within an experiment where every condition already uses the same
validation set for early stopping and checkpoint selection; report it as such.
Diagnostics: `aio2/phase2_started` (both modes), `aio2/phase2_trigger_val_best`,
`aio2/resume_checkpoint_epoch` (completed epochs at the restart checkpoint).

### Resuming an interrupted run

With `resumable: true` (see
[Experiments Runner](experiments_runner.md#resuming-inside-a-run-resumable-true))
and `phase2_trigger: val_best`, each phase saves
`resume_ckpt/warmup_last.ckpt` / `resume_ckpt/correction_last.ckpt`, and once
the warm-up is done the phase-2 hand-off (init checkpoint + `aio2/*` info) is
recorded in `resume_ckpt/aio2_state.json`. On retry:

| Interrupted during | Resumes |
|---|---|
| warm-up | the warm-up from its last epoch (`version_0`), then phase 2 as usual |
| phase 2, before its first epoch was saved | phase 2 from the recorded init checkpoint; warm-up not redone |
| phase 2 | phase 2 from its last epoch (`version_1`); warm-up not redone |

`phase2_trigger: act` is **not** resumed: `ACTTracker`'s slope history is not
part of the checkpoint, so ACT mode writes no resume state and an
interrupted run restarts from scratch, as before.

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
# act_checkpoint_dir: optional — defaults to <default_root_dir>/act_warmup_checkpoints
# phase2_trigger: act               # or: val_best — see "Alternative trigger" above

loss:
  _target_: pytorch_segmentation_models_trainer.custom_losses.loss.WeightedDiceCrossEntropyLoss
  num_classes: 6
  ignore_index: 255
```

No new loss class — `WeightedDiceCrossEntropyLoss` (CE + Dice) already
covers what AIO2 needs.

---

## GPU acceleration for O2C

O2C's connected-component labelling runs once per image, every batch —
CPU-bound by default. It auto-detects and uses a GPU-native path (cuCIM,
zero-copy from CUDA tensors) whenever available, with zero config changes
here — see the [README's "AIO2 O2C GPU Acceleration"
section](../../../README.md#aio2-o2c-gpu-acceleration-optional) for
installation (`uv sync --extra gpu-ml`) and hardware requirements
(compute capability 7.0+, e.g. Tesla V100).

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
