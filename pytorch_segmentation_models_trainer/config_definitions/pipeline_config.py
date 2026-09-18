# -*- coding: utf-8 -*-
"""
/***************************************************************************
 pytorch_segmentation_models_trainer
                              -------------------
        begin                : 2026-09-17
        copyright            : (C) 2026 by Philipe Borba
        email                : philipeborba at gmail dot com
 ***************************************************************************/
/***************************************************************************
 *                                                                         *
 *   This program is free software; you can redistribute it and/or modify  *
 *   it under the terms of the GNU General Public License as published by  *
 *   the Free Software Foundation; either version 2 of the License, or     *
 *   (at your option) any later version.                                   *
 *                                                                         *
 ***************************************************************************/
"""

from dataclasses import dataclass, field
from typing import List, Optional

from hydra.core.config_store import ConfigStore


@dataclass
class PipelineGpu:
    """One physical GPU available to the pipeline's scheduler.

    Args:
        id: GPU index as understood by ``CUDA_VISIBLE_DEVICES`` (the
            physical device index on this node).
        vram_gb: Total VRAM capacity used for bin-packing single-GPU
            (``gpus: 1``) steps via each step's ``vram_gb`` request. Not
            consulted for multi-GPU (``gpus > 1``) steps, which always
            reserve whole GPUs exclusively regardless of this value.
    """

    id: int = 0
    vram_gb: float = 0.0


@dataclass
class PipelineResources:
    """The pipeline's available compute, for the parallel scheduler.

    Args:
        gpus: Physical GPUs available on this node. Omit (or leave empty)
            to disable GPU-aware scheduling entirely — steps requesting
            ``gpus > 0`` will then never become schedulable and the
            pipeline raises rather than hang.
        max_parallel_cpu_steps: How many ``gpus: 0`` (CPU-bound) steps may
            run concurrently — independent of the GPU pool, so a data-prep
            step never waits behind GPU steps or vice versa.
    """

    gpus: List[PipelineGpu] = field(default_factory=list)
    max_parallel_cpu_steps: int = 1


@dataclass
class PipelineStep:
    """A single step in a pipeline's sequential run list.

    Each step launches a full, independent subprocess — its own config,
    its own ``mode`` (``train``, ``run-experiments``, etc.), or an
    entirely different CLI. This is what lets one step itself be a
    multi-seed :class:`ExperimentsRunner` sweep, another step a plain
    training run, and another a data-prep tool — all chained in order.

    Two ways to specify what to run — exactly one must be given:

    1. ``config_dir`` + ``config_name`` (the common case): launches
       ``pytorch-smt --config-dir <config_dir> --config-name <config_name>
       <overrides...>`` — a Hydra-mode training/experiments-runner config.
    2. ``command``: launches that command verbatim, with ``overrides``
       appended as extra CLI arguments. Use this for anything that isn't a
       ``pytorch-smt`` Hydra invocation — most notably the ``pytorch-smt-tools``
       click CLI used for data-prep steps like SAM/SLICO label correction
       (``build-sam-corrected-masks``, ``build-slico-corrected-masks``),
       which take a positional yaml path, not ``--config-dir``/``--config-name``.
       This is what lets a pipeline chain a mask-correction step into the
       training run(s) that consume its output.

    Args:
        name: Stable identity for this step, used as the key in
            ``pipeline_state.json`` and in per-step log filenames. Must be
            unique within the pipeline. Identity is by name, not list
            position, so reordering or inserting steps never confuses
            resume.
        config_dir: Directory passed as ``--config-dir`` to ``pytorch-smt``.
            Ignored when ``command`` is set.
        config_name: Config file name (without extension) passed as
            ``--config-name``. Ignored when ``command`` is set.
        command: Full command to run verbatim instead of building a
            ``pytorch-smt --config-dir/--config-name`` invocation, e.g.
            ``["pytorch-smt-tools", "build-slico-corrected-masks",
            "conf/examples/slico_label_correction.yaml"]``. Takes
            precedence over ``config_dir``/``config_name`` when non-empty.
        overrides: Extra CLI arguments appended to the command. For a
            ``config_dir``/``config_name`` step these are Hydra dotlist
            overrides (``"model.o2c_active=true"``); for a ``command``
            step they are appended verbatim (e.g. ``["--start-idx",
            "0", "--end-idx", "999"]`` for a ``pytorch-smt-tools`` command).
        overwrite: Per-step override of the pipeline-level ``overwrite``
            setting. ``None`` (default) inherits the pipeline-level value.
        enabled: When ``False``, the step is skipped entirely (not run,
            not marked done) — useful to temporarily pull one step out of
            a pipeline without deleting its definition.
        depends_on: Names of steps that must be ``done`` before this step
            becomes eligible to start. Replaces list order as the
            dependency mechanism — steps with no (transitively satisfied)
            ``depends_on`` are eligible to run concurrently. Every name
            must refer to another step in the same pipeline; cycles are
            rejected at construction time.
        gpus: GPUs this step needs, reserved from ``pipeline.resources.gpus``
            for the duration of the subprocess (via ``CUDA_VISIBLE_DEVICES``,
            remapped to logical devices ``0..gpus-1`` inside the
            subprocess). ``0`` (default) — no GPU; the step instead
            competes for ``max_parallel_cpu_steps``. ``1`` — bin-packed
            onto any GPU with ``vram_gb`` free capacity, and may share
            that GPU with other ``gpus: 1`` steps. ``>1`` — a distributed
            step (e.g. PyTorch Lightning DDP driven by that many
            ``pl_trainer.devices`` in the step's own config): reserves
            that many whole GPUs *exclusively* (no bin-packing, no
            sharing), released together when the step finishes.
        vram_gb: VRAM this step needs, only consulted when ``gpus == 1``.
            ``None`` (default) reserves a GPU's *entire* capacity — safe
            default that never oversubscribes. Set explicitly to let
            several small steps bin-pack onto the same GPU.
    """

    name: str = ""
    config_dir: str = ""
    config_name: str = ""
    command: List[str] = field(default_factory=list)
    overrides: List[str] = field(default_factory=list)
    overwrite: Optional[bool] = None
    enabled: bool = True
    depends_on: List[str] = field(default_factory=list)
    gpus: int = 0
    vram_gb: Optional[float] = None


@dataclass
class PipelineConfig:
    """Configuration for running a sequence of independent experiment steps.

    Unlike :class:`ExperimentsRunnerConfig` (which repeats *one* config with
    different seeds), this runs a list of *different* steps — different
    models, losses, datasets, even different CLIs entirely (a data-prep
    tool followed by training) — one after another, each as its own
    subprocess.

    Args:
        steps: Ordered list of :class:`PipelineStep`, executed sequentially.
        output_base_dir: Root directory for ``pipeline_state.json``, the
            master ``pipeline.log``, and per-step log files (under
            ``logs/``). Independent of each step's own output directory,
            which is defined inside that step's own config.
        resume: When ``True`` (default) and a ``pipeline_state.json``
            exists, steps already recorded as done are skipped —
            idempotent by default, same as :class:`ExperimentsRunnerConfig`.
            Set to ``False`` to always re-run every step.
        overwrite: Forces re-execution of steps already marked done.
            ``True`` forces every step; a list of step ``name`` values
            forces only those. Individual steps can also set their own
            ``overwrite`` to override this pipeline-level default.
        on_error: ``"stop"`` (default) halts the sequence on the first
            failed step, leaving it marked ``failed`` in
            ``pipeline_state.json`` for inspection/retry. ``"continue"``
            logs the failure and proceeds to the next step.
        preflight: When ``True`` (default), every Hydra-mode step's config
            is composed (via ``mode=validate-config``, no training) before
            step 1 actually runs, so a config error in a late step is
            caught before spending compute on earlier ones. This only
            catches composition-time errors (bad keys, broken
            interpolations, missing referenced files) — not deeper
            semantic/training errors, and does not apply to ``command``
            steps (non-Hydra CLIs).
        pytorch_smt_bin: Executable used to launch each Hydra-mode step.
            Override when it is not on ``PATH`` (e.g. a specific
            virtualenv).
        resources: Available compute for the parallel scheduler — see
            :class:`PipelineResources`. ``None`` (default) means no GPUs
            are known to the scheduler; every ``gpus: 0`` step still runs
            (bounded by ``max_parallel_cpu_steps``, default 1 — effectively
            sequential), but any step requesting ``gpus > 0`` raises
            immediately rather than hang forever waiting for a GPU that
            was never declared.

    Example YAML:

    .. code-block:: yaml

        mode: run-pipeline

        pipeline:
          output_base_dir: outputs/isprs_matrix
          resume: true
          on_error: stop
          resources:
            gpus:
              - {id: 0, vram_gb: 24}
              - {id: 1, vram_gb: 24}
            max_parallel_cpu_steps: 4
          steps:
            - name: slico_gc
              # data-prep step: pytorch-smt-tools, not pytorch-smt — use `command`
              command: ["pytorch-smt-tools", "build-slico-corrected-masks",
                        "conf/examples/slico_label_correction.yaml"]
            - name: zhu2019_baseline   # no depends_on — runs in parallel with slico_gc
              config_dir: conf/examples
              config_name: zhu2019_boundary_relaxation
              gpus: 1
              vram_gb: 12
            - name: unet_slico_gc
              config_dir: conf/examples
              config_name: experiments_runner
              overrides: ["train_dataset.input_csv_path=/data/masks_slico_gc/train.csv"]
              depends_on: [slico_gc]
              gpus: 1
              vram_gb: 24
            - name: aio2_phase1_warmup
              config_dir: conf/examples
              config_name: mean_teacher_aio2
              gpus: 2   # distributed — reserves 2 whole GPUs exclusively
            - name: aio2_phase2_correction
              config_dir: conf/examples
              config_name: mean_teacher_aio2
              overrides: ["model.o2c_active=true"]
              depends_on: [aio2_phase1_warmup]
              gpus: 2
              overwrite: true   # redo just this one, regardless of pipeline default
    """

    steps: List[PipelineStep] = field(default_factory=list)
    output_base_dir: str = "outputs/pipeline"
    resume: bool = True
    overwrite: bool = False
    on_error: str = "stop"
    preflight: bool = True
    pytorch_smt_bin: str = "pytorch-smt"
    resources: Optional[PipelineResources] = None


def _register_configs() -> None:
    cs = ConfigStore.instance()
    cs.store(
        name="pipeline_config",
        node=PipelineConfig,
        group="pipeline",
    )


_register_configs()
