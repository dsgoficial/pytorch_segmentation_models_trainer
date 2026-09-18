---
sidebar_position: 13.5
title: Pipeline
---

# Pipeline

The Pipeline runs a **DAG of independent experiment steps**, scheduled in
parallel across your available GPUs/CPU, unattended — the recommended
workflow when you need to launch an entire experiment matrix (data-prep
tools, different models, losses, datasets, baselines) on a remote server
and walk away (`nohup`/`tmux`), rather than babysitting each subprocess by
hand.

This is a different axis from the [Experiments Runner](experiments_runner.md):
the Experiments Runner repeats **one** config with different seeds; the
Pipeline runs a set of **different** steps. A step can itself be an
Experiments Runner config — the pipeline transparently runs that step's
whole multi-seed sweep (as its own subprocess) before its dependents start.
A step doesn't even have to be a `pytorch-smt` invocation — it can be any
command, which is what lets a pipeline chain a data-prep tool (e.g.
`pytorch-smt-tools build-slico-corrected-masks`) into the training run(s)
that consume its output.

Two things make this different from just running commands in a `for` loop:

- **Dependencies, not list order** — a step declares `depends_on: [...]`;
  independent steps (no dependency relationship) run concurrently instead
  of waiting in line.
- **Resource-aware scheduling** — steps declare how many GPUs (and how
  much VRAM) they need; the scheduler bin-packs small steps onto shared
  GPUs, reserves whole GPUs exclusively for multi-GPU distributed steps,
  and never over-subscribes.

---

## Quick start

```yaml
# my_pipeline.yaml
mode: run-pipeline

pipeline:
  output_base_dir: outputs/my_matrix
  resume: true # default — safe to re-launch after a crash
  on_error: stop
  resources:
    gpus:
      - { id: 0, vram_gb: 24 }
      - { id: 1, vram_gb: 24 }
    max_parallel_cpu_steps: 4

  steps:
    - name: slico_gc
      command: ["pytorch-smt-tools", "build-slico-corrected-masks",
                "conf/examples/slico_label_correction.yaml"]

    - name: zhu2019_baseline # no depends_on -> runs concurrently with slico_gc
      config_dir: conf/examples
      config_name: zhu2019_boundary_relaxation
      gpus: 1
      vram_gb: 12

    - name: unet_slico_gc
      config_dir: conf/examples
      config_name: experiments_runner # this step's own mode is run-experiments
      overrides:
        - train_dataset.input_csv_path=/data/masks_slico_gc/train.csv
      depends_on: [slico_gc]
      gpus: 1
      vram_gb: 24
```

```bash
pytorch-smt --config-dir . --config-name my_pipeline
```

Each step runs as its own subprocess — either a full, independent
`pytorch-smt --config-dir/--config-name [overrides]` Hydra invocation, or
(with `command`) any other CLI invocation verbatim.

---

## `pipeline` block reference

| Field | Type | Required | Default | Description |
|---|---|---|---|---|
| `steps` | `list[Step]` | yes | — | Steps to schedule. Must be non-empty; every step needs a unique `name`. |
| `output_base_dir` | `str` | no | `outputs/pipeline` | Root for `pipeline_state.json`, `pipeline.log`, and per-step logs (`logs/`). Independent of each step's own output directory (defined inside that step's config). |
| `resources` | `Resources` | no | none | Available GPUs/CPU for the scheduler — see [Parallel scheduling](#parallel-scheduling) below. Omit entirely for a CPU-only pipeline (every step `gpus: 0`). |
| `resume` | `bool` | no | `true` | Skip steps already recorded `done` in `pipeline_state.json` — idempotent by default, same as the Experiments Runner. |
| `overwrite` | `bool \| list[str]` | no | `false` | Forces re-execution of steps `resume` would skip. `true` forces every step; a list forces only those step names. Each step can also set its own `overwrite` to override this default. |
| `on_error` | `"stop" \| "continue"` | no | `stop` | `stop` halts new admissions on the first failed step (already-running steps finish naturally, none are killed), then raises listing every step that failed. `continue` proceeds scheduling everything not blocked by that failure. |
| `preflight` | `bool` | no | `true` | Compose every enabled Hydra-mode step's config (`mode=validate-config`, no training) before anything actually runs — catches config typos across the whole pipeline up front. Does not apply to `command` steps. |
| `pytorch_smt_bin` | `str` | no | `pytorch-smt` | Executable used to launch each Hydra-mode step. Override when it's not on `PATH` (e.g. a specific virtualenv). |

### Step reference

| Field | Type | Required | Default | Description |
|---|---|---|---|---|
| `name` | `str` | yes | — | Stable identity — key in `pipeline_state.json`, per-step log filenames, and `depends_on` references. Identity is by name, not list position. |
| `config_dir` | `str` | one of `config_dir`+`config_name` / `command` | — | Passed as `--config-dir` to the step's `pytorch-smt` invocation. Ignored when `command` is set. |
| `config_name` | `str` | (see above) | — | Passed as `--config-name`. Ignored when `command` is set. |
| `command` | `list[str]` | (see above) | `[]` | Full command to run verbatim instead of a `pytorch-smt --config-dir/--config-name` invocation — use for anything that isn't a `pytorch-smt` Hydra call, most notably `pytorch-smt-tools` data-prep commands. Takes precedence over `config_dir`/`config_name` when non-empty. |
| `overrides` | `list[str]` | no | `[]` | Extra CLI arguments appended to the command — Hydra dotlist overrides for a `config_dir`/`config_name` step, or verbatim CLI args for a `command` step. |
| `depends_on` | `list[str]` | no | `[]` | Names of steps that must be `done` before this one is eligible to start. See [Dependencies](#dependencies-depends_on). |
| `gpus` | `int` | no | `0` | GPUs this step needs. See [Parallel scheduling](#parallel-scheduling). |
| `vram_gb` | `float` | no | none (full GPU) | VRAM this step needs — only consulted when `gpus == 1`. |
| `overwrite` | `bool` | no | inherits pipeline-level | Per-step override of the pipeline-level `overwrite` default. |
| `enabled` | `bool` | no | `true` | `false` skips the step entirely — not run, not marked done, not scheduled, not preflighted — useful to pull one step out temporarily without deleting its definition. |

---

## Dependencies (`depends_on`)

```yaml
steps:
  - name: slico_gc
    command: [...]

  - name: zhu2019_baseline # no depends_on -> eligible immediately, in parallel with slico_gc
    config_dir: conf/examples
    config_name: zhu2019_boundary_relaxation

  - name: unet_slico_gc
    depends_on: [slico_gc] # only eligible once slico_gc = done
    config_dir: conf/examples
    config_name: experiments_runner
    overrides: ["train_dataset.input_csv_path=/data/masks_slico_gc/train.csv"]
```

`depends_on` — not list order — is what makes one step wait for another.
Steps with no (or already-satisfied) dependencies are eligible to run
concurrently, bounded only by `resources`. Every name in `depends_on` must
refer to another step in the same pipeline, and dependency cycles
(including self-references) are rejected when the pipeline is constructed
— both fail fast, before anything runs.

**A step whose dependency failed is never attempted** — it's recorded
`skipped` in `pipeline_state.json` and in the returned results, distinct
from `done`/`failed`, so a broken upstream step doesn't cascade into
confusing downstream failures.

---

## Parallel scheduling

```yaml
pipeline:
  resources:
    gpus:
      - { id: 0, vram_gb: 24 }
      - { id: 1, vram_gb: 24 }
    max_parallel_cpu_steps: 4
```

Each physical GPU is identified by `id` (the value handed to the
subprocess via `CUDA_VISIBLE_DEVICES`, remapped internally to logical
device `0`) and a `vram_gb` capacity used for bin-packing. CPU-bound steps
(`gpus: 0`, e.g. a data-prep `command` step) draw from
`max_parallel_cpu_steps` instead — a separate pool, so they never wait
behind GPU steps or vice versa.

A step's `gpus` value picks one of three scheduling behaviors:

| `gpus` | Behavior |
|---|---|
| `0` (default) | CPU-bound — competes for `max_parallel_cpu_steps`. |
| `1` | Bin-packed onto any GPU with enough free `vram_gb` (`vram_gb: X` on the step) — may **share** that GPU with other `gpus: 1` steps whose combined `vram_gb` still fits. Omitting `vram_gb` reserves the GPU's *entire* capacity (safe default, no sharing). |
| `>1` | **Distributed**, e.g. PyTorch Lightning DDP driven by that many `pl_trainer.devices` in the step's own config. Reserves that many whole GPUs *exclusively* — no bin-packing, no sharing — released together when the step finishes. `vram_gb` is not consulted. |

For a `gpus > 0` step, the scheduler sets `CUDA_VISIBLE_DEVICES` to the
reserved physical id(s) before launching the subprocess — the step's own
config just asks for "N devices" (`pl_trainer.devices: N`) without
needing to know which physical GPU ids it was handed.

An unsatisfiable request — `gpus` greater than the configured GPU count,
`vram_gb` bigger than every configured GPU's capacity, or any `gpus > 0`
step when `pipeline.resources` isn't set at all — is rejected when the
pipeline is constructed, not discovered by hanging at runtime.

### Example: AIO2's distributed warm-up alongside independent steps

```yaml
pipeline:
  resources:
    gpus: [{ id: 0, vram_gb: 24 }, { id: 1, vram_gb: 24 }]
    max_parallel_cpu_steps: 4

  steps:
    - name: slico_gc # gpus: 0 — CPU pool, runs immediately
      command: [pytorch-smt-tools, build-slico-corrected-masks, slico_label_correction.yaml]

    - name: zhu2019_baseline # gpus: 1 — runs concurrently with slico_gc
      config_dir: conf/examples
      config_name: zhu2019_boundary_relaxation
      gpus: 1
      vram_gb: 12

    - name: aio2_phase1_warmup
      config_dir: conf/examples
      config_name: mean_teacher_aio2
      gpus: 2 # takes both GPUs exclusively — zhu2019_baseline above must
      # finish (and free its GPU) before this can start, even though
      # aio2_phase1_warmup has no depends_on on it — that's resource
      # contention, not a dependency.

    - name: aio2_phase2_correction
      config_dir: conf/examples
      config_name: mean_teacher_aio2 # same act_checkpoint_dir as phase 1
      overrides: [model.o2c_active=true]
      depends_on: [aio2_phase1_warmup]
      gpus: 2
```

This version does not capture and inject values dynamically between steps
(e.g. resolving AIO2's warm-up checkpoint path automatically) — `depends_on`
gets the ordering right, and `act_checkpoint_dir` being a fixed, known
location in the step's own config is what lets phase 2 find phase 1's
checkpoint without the pipeline needing to pass anything between them.

---

## Chaining a mask-correction step into training

The motivating use case: SAM/SLICO label correction
(`pytorch-smt-tools build-sam-corrected-masks` /
`build-slico-corrected-masks`) is a data-prep step, not a `pytorch-smt`
Hydra invocation — it takes a positional yaml path, not
`--config-dir`/`--config-name`. Use `command` for it, `depends_on` to
order the training step after it, and point the training step's
`train_dataset.input_csv_path` (or `masks_dir`, depending on the dataset
config) at its output:

```yaml
steps:
  - name: slico_gc
    command: ["pytorch-smt-tools", "build-slico-corrected-masks",
              "conf/examples/slico_label_correction.yaml"]

  - name: unet_slico_gc
    depends_on: [slico_gc]
    config_dir: conf/examples
    config_name: experiments_runner
    overrides:
      - train_dataset.input_csv_path=/data/masks_slico_gc/train.csv
```

---

## Output layout

```
outputs/my_matrix/
├── logs/
│   ├── 000_slico_gc.log       ← raw stdout/stderr of that step's subprocess
│   ├── 001_unet_slico_gc.log
│   └── ...
├── pipeline_state.json        ← per-step status; drives resume
└── pipeline.log                ← one line per START/DONE/FAILED/SKIPPED event
```

`pipeline_state.json`:

```json
{
  "steps": {
    "slico_gc": {
      "name": "slico_gc",
      "status": "done",
      "exit_code": 0,
      "duration_s": 340.1,
      "log_path": "outputs/my_matrix/logs/000_slico_gc.log"
    }
  }
}
```

`status` is one of `done`, `failed`, or `skipped` (dependency failed —
never attempted; `exit_code`/`log_path` are `null`).

---

## Resuming an interrupted pipeline

Idempotent by default — if the run is interrupted (crash, server reboot,
killed `tmux` session), just re-launch the same command:

```bash
pytorch-smt --config-dir . --config-name my_pipeline
```

The pipeline reads `pipeline_state.json`, skips every step already `done`,
and resumes scheduling from the first pending ones. This composes with the
[Experiments Runner's own idempotency](experiments_runner.md#resuming-an-interrupted-run-sequence):
if a step is itself a multi-seed `run-experiments` config that crashed
partway (3 of 5 seeds done), re-running the pipeline re-invokes that
step's subprocess, and the Experiments Runner inside it skips the 3
already-done seeds on its own — no extra pipeline-level logic needed.

---

## Forcing specific steps to redo (`overwrite`)

```yaml
pipeline:
  resume: true
  overwrite: [aio2_phase2_correction] # redo only this step
```

Or force everything:

```yaml
pipeline:
  overwrite: true
```

Or override per step, regardless of the pipeline-level default:

```yaml
pipeline:
  overwrite: false
  steps:
    - name: aio2_phase2_correction
      overwrite: true # forced anyway
```

`overwrite` at the pipeline level only affects whether `pipeline_state.json`
treats the step as needing a re-run — it does **not** delete that step's
own underlying output directory (checkpoints, logs from its own config).
That directory is owned by the step's own config, not the pipeline; for a
`run-experiments` step, its own `overwrite`/`resume` fields (see
[Experiments Runner](experiments_runner.md#forcing-specific-runs-to-redo-overwrite))
control that at the run level.

---

## Handling errors

```yaml
pipeline:
  on_error: stop # default
```

A step that exits non-zero is recorded `failed` in `pipeline_state.json`.
`on_error: stop` halts *new admissions* the moment a failure is seen —
already-running independent steps are left to finish naturally (nothing
is killed mid-training) — then raises, listing every step that failed and
pointing at each one's log file
(`outputs/my_matrix/logs/NNN_name.log`) for diagnosis. Fix the issue and
re-run the same command — `resume` (the default) picks up exactly where
it left off.

`on_error: continue` instead keeps scheduling everything not blocked by
that failure (steps depending on it are recorded `skipped`, independent
branches proceed normally) — useful when you'd rather let the rest of a
200-step matrix finish overnight than stall on one broken config, and
triage failures the next morning from `pipeline_state.json`.

---

## Preflight validation

```yaml
pipeline:
  preflight: true # default
```

Before anything actually runs, every **enabled, Hydra-mode** step's config
is composed via `mode=validate-config` (no training) — this catches bad
keys, broken `${...}` interpolations, and similarly-scoped
composition-time errors across the whole pipeline up front. If any step
fails to compose, the pipeline raises immediately and **nothing runs** — a
typo in step 47 of a 200-step matrix is caught before steps 1-46 spend any
compute. `command` steps (non-Hydra CLIs) are not preflighted.

Preflight only validates that Hydra can *compose* each config — it does
not catch deeper semantic or training-time errors (e.g. a dataset path
that resolves but points at the wrong data). Set `preflight: false` to
skip this pass (e.g. while iterating quickly on a single step during
development).

---

## Full example config

See [`conf/examples/pipeline.yaml`](https://github.com/dsgoficial/pytorch_segmentation_models_trainer/blob/main/pytorch_segmentation_models_trainer/conf/examples/pipeline.yaml).
