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

import json
import logging
import os
import queue
import subprocess
import threading
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Set, Union

from omegaconf import DictConfig, OmegaConf

logger = logging.getLogger(__name__)

_STATE_FILE = "pipeline_state.json"
_LOG_DIR = "logs"


class _ResourcePool:
    """Tracks GPU/CPU capacity for the scheduler — reserve/release only,
    all from the main scheduler thread (worker threads never touch this).

    A single accounting model covers both cases: reserving "the full
    capacity" of a GPU (the default when a step gives no ``vram_gb``) looks
    identical, bookkeeping-wise, to a multi-GPU step reserving N whole GPUs
    exclusively — both just mean "this GPU's free capacity drops to 0 until
    released". Bin-packing (`gpus: 1` with an explicit `vram_gb`) is the
    only case where a GPU's free capacity can be partially consumed and
    shared with other steps.
    """

    def __init__(self, resources_cfg) -> None:
        gpus_cfg = list(resources_cfg.get("gpus", []) or []) if resources_cfg else []
        self.capacity: Dict[int, float] = {
            int(g.id): float(g.vram_gb) for g in gpus_cfg
        }
        self.free: Dict[int, float] = dict(self.capacity)
        self.cpu_capacity: int = (
            int(resources_cfg.get("max_parallel_cpu_steps", 1)) if resources_cfg else 1
        )
        self.cpu_free: int = self.cpu_capacity

    def can_ever_satisfy(self, step) -> bool:
        """Static feasibility check — could this step *ever* run, given
        the configured resources? Used at construction time so an
        unsatisfiable request fails fast instead of stalling the scheduler."""
        gpus_needed = int(step.get("gpus", 0) or 0)
        if gpus_needed == 0:
            return True
        if gpus_needed > len(self.capacity):
            return False
        if gpus_needed == 1:
            vram = step.get("vram_gb", None)
            if vram is None:
                return len(self.capacity) > 0
            return any(cap >= float(vram) for cap in self.capacity.values())
        return True

    def try_reserve(self, step) -> Optional[Dict[str, Any]]:
        """Attempt to reserve this step's resources right now.

        Returns a reservation token to pass back to :meth:`release`, or
        ``None`` if not currently available (caller should retry later).
        """
        gpus_needed = int(step.get("gpus", 0) or 0)

        if gpus_needed == 0:
            if self.cpu_free > 0:
                self.cpu_free -= 1
                return {"type": "cpu"}
            return None

        if gpus_needed == 1:
            vram = step.get("vram_gb", None)
            if vram is None:
                candidates = [
                    gid for gid, cap in self.capacity.items() if self.free[gid] == cap
                ]
                if not candidates:
                    return None
                gid = candidates[0]
                self.free[gid] = 0.0
                return {"type": "gpu", "ids": [gid], "full": True}
            need = float(vram)
            candidates = sorted(
                (gid for gid, free in self.free.items() if free >= need),
                key=lambda g: self.free[g],
            )
            if not candidates:
                return None
            gid = candidates[0]  # best-fit: tightest free slot that still fits
            self.free[gid] -= need
            return {"type": "gpu", "ids": [gid], "full": False}

        # Multi-GPU (exclusive): needs `gpus_needed` completely idle GPUs.
        candidates = [
            gid for gid, cap in self.capacity.items() if self.free[gid] == cap
        ]
        if len(candidates) < gpus_needed:
            return None
        chosen = candidates[:gpus_needed]
        for gid in chosen:
            self.free[gid] = 0.0
        return {"type": "gpu", "ids": chosen, "full": True}

    def release(self, step, reservation: Dict[str, Any]) -> None:
        if reservation["type"] == "cpu":
            self.cpu_free += 1
            return
        if reservation["full"]:
            for gid in reservation["ids"]:
                self.free[gid] = self.capacity[gid]
        else:
            gid = reservation["ids"][0]
            self.free[gid] += float(step.vram_gb)


class PipelineRunner:
    """Runs a sequence of independent experiment steps, one after another.

    Unlike :class:`~pytorch_segmentation_models_trainer.tools.experiments_runner.experiments_runner.ExperimentsRunner`
    (which repeats *one* config with different seeds), this runs a list of
    *different* steps — different models, losses, datasets, even entirely
    different CLIs — sequentially. Each step is launched as its own
    subprocess, so a step whose own config sets ``mode: run-experiments``
    transparently runs its own multi-seed sweep before the pipeline moves
    on to the next step. A step can also run a verbatim ``command`` instead
    of a ``pytorch-smt`` Hydra invocation — e.g. a ``pytorch-smt-tools
    build-slico-corrected-masks ...`` data-prep step chained before the
    training steps that consume its output.

    Idempotent by default, mirroring :class:`ExperimentsRunner`: steps
    already recorded as ``done`` in ``pipeline_state.json`` are skipped on
    restart (``resume: true``). ``overwrite`` forces specific (or all)
    completed steps to re-run.

    Args:
        cfg: Full Hydra config including the ``pipeline`` sub-tree.

    Raises:
        ValueError: If ``steps`` is empty, has duplicate/blank names, a
            step is missing both ``command`` and ``config_dir``/
            ``config_name``, or ``on_error`` is not one of
            ``"stop"``/``"continue"``.

    Example::

        runner = PipelineRunner(cfg)
        results = runner.run()
        for r in results:
            print(r["name"], r["status"])
    """

    def __init__(self, cfg: DictConfig) -> None:
        self.cfg = cfg
        self.pipeline_cfg = cfg.pipeline
        self._validate()

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def _validate(self) -> None:
        steps = self.pipeline_cfg.get("steps", None) or []
        if len(steps) == 0:
            raise ValueError("pipeline: 'steps' must be a non-empty list.")

        names = []
        for step in steps:
            name = step.get("name", "")
            if not name:
                raise ValueError("pipeline: every step needs a non-empty 'name'.")
            has_command = bool(step.get("command", None))
            has_hydra_cfg = bool(step.get("config_dir", "")) and bool(
                step.get("config_name", "")
            )
            if not has_command and not has_hydra_cfg:
                raise ValueError(
                    f"pipeline: step '{name}' needs either 'command', or "
                    "both 'config_dir' and 'config_name'."
                )
            names.append(name)

        duplicates = {n for n in names if names.count(n) > 1}
        if duplicates:
            raise ValueError(
                f"pipeline: step names must be unique, found duplicates: {sorted(duplicates)}."
            )

        on_error = self.pipeline_cfg.get("on_error", "stop")
        if on_error not in ("stop", "continue"):
            raise ValueError(
                f"pipeline.on_error must be 'stop' or 'continue', got {on_error!r}."
            )

        self._validate_dag(steps, set(names))
        self._validate_resources(steps)

    def _validate_dag(self, steps, names: Set[str]) -> None:
        by_name = {s.name: s for s in steps}
        for step in steps:
            for dep in step.get("depends_on", []) or []:
                if dep not in names:
                    raise ValueError(
                        f"pipeline: step '{step.name}' depends_on unknown step '{dep}'."
                    )

        visited: Dict[str, bool] = {}

        def visit(name: str, stack: List[str]) -> None:
            if name in stack:
                cycle = " -> ".join(stack[stack.index(name) :] + [name])
                raise ValueError(f"pipeline: dependency cycle detected: {cycle}.")
            if visited.get(name):
                return
            for dep in by_name[name].get("depends_on", []) or []:
                visit(dep, stack + [name])
            visited[name] = True

        for name in names:
            visit(name, [])

    def _validate_resources(self, steps) -> None:
        resources_cfg = self.pipeline_cfg.get("resources", None)
        pool = _ResourcePool(resources_cfg)
        for step in steps:
            if not step.get("enabled", True):
                continue
            if not pool.can_ever_satisfy(step):
                raise ValueError(
                    f"pipeline: step '{step.name}' requests gpus={step.get('gpus', 0)}"
                    f"{'' if step.get('vram_gb', None) is None else f', vram_gb={step.vram_gb}'} "
                    "which pipeline.resources can never satisfy — check "
                    "resources.gpus (count/vram_gb) is configured to cover it."
                )

    # ------------------------------------------------------------------
    # Paths / state persistence
    # ------------------------------------------------------------------

    def _output_base_dir(self) -> str:
        return self.pipeline_cfg.output_base_dir

    def _state_path(self) -> str:
        return os.path.join(self._output_base_dir(), _STATE_FILE)

    def _pipeline_log_path(self) -> str:
        return os.path.join(self._output_base_dir(), "pipeline.log")

    def _step_log_path(self, idx: int, name: str) -> str:
        return os.path.join(self._output_base_dir(), _LOG_DIR, f"{idx:03d}_{name}.log")

    def _load_state(self) -> Dict[str, Any]:
        with open(self._state_path()) as f:
            return json.load(f)

    def _save_state(self, state: Dict[str, Any]) -> None:
        os.makedirs(self._output_base_dir(), exist_ok=True)
        with open(self._state_path(), "w") as f:
            json.dump(state, f, indent=2)

    def _append_pipeline_log(self, line: str) -> None:
        os.makedirs(self._output_base_dir(), exist_ok=True)
        ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        with open(self._pipeline_log_path(), "a") as f:
            f.write(f"[{ts}] {line}\n")

    # ------------------------------------------------------------------
    # Overwrite resolution
    # ------------------------------------------------------------------

    def _resolve_overwrite(self) -> Union[bool, Set[str]]:
        """Return the resolved pipeline-level ``overwrite`` setting.

        Returns:
            ``True`` if every completed step must be forced to re-run, or
            a (possibly empty) set of step ``name`` values to force
            individually.
        """
        overwrite = self.pipeline_cfg.get("overwrite", False)
        if overwrite is True:
            return True
        if not overwrite:
            return set()
        return {str(n) for n in overwrite}

    def _step_should_force(
        self, step: DictConfig, pipeline_overwrite: Union[bool, Set[str]]
    ) -> bool:
        """Whether this step must be forced to re-run despite being done.

        A per-step ``overwrite`` (when not ``None``) takes precedence over
        the pipeline-level default.
        """
        step_overwrite = step.get("overwrite", None)
        if step_overwrite is not None:
            return bool(step_overwrite)
        return pipeline_overwrite is True or (
            isinstance(pipeline_overwrite, set) and step.name in pipeline_overwrite
        )

    # ------------------------------------------------------------------
    # Command construction
    # ------------------------------------------------------------------

    def _build_command(self, step: DictConfig) -> List[str]:
        """Build the subprocess command for one step.

        A ``command`` step is used verbatim (for anything that isn't a
        ``pytorch-smt`` Hydra invocation, e.g. the ``pytorch-smt-tools``
        CLI used for SAM/SLICO label-correction data-prep steps).
        Otherwise builds a ``pytorch-smt --config-dir ... --config-name
        ...`` Hydra invocation. Either way, ``overrides`` is appended as
        extra CLI arguments.
        """
        raw_command = step.get("command", None)
        if raw_command:
            cmd = [str(c) for c in raw_command]
        else:
            cmd = [
                self.pipeline_cfg.get("pytorch_smt_bin", "pytorch-smt"),
                "--config-dir",
                str(step.config_dir),
                "--config-name",
                str(step.config_name),
            ]
        cmd += [str(o) for o in (step.get("overrides", None) or [])]
        return cmd

    def _is_hydra_step(self, step: DictConfig) -> bool:
        """Whether this step is a `pytorch-smt` Hydra invocation.

        `command` steps (e.g. `pytorch-smt-tools ...`) are not — they
        don't understand `mode=validate-config` and are excluded from
        preflight.
        """
        return not bool(step.get("command", None))

    # ------------------------------------------------------------------
    # Subprocess execution (isolated for mockability in tests)
    # ------------------------------------------------------------------

    def _run_preflight_check(self, cmd: List[str]) -> "subprocess.CompletedProcess":
        return subprocess.run(cmd, capture_output=True, text=True)

    def _run_step(
        self, cmd: List[str], log_path: str, env: Optional[Dict[str, str]] = None
    ) -> int:
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        run_env = {**os.environ, **env} if env else None
        with open(log_path, "w") as lf:
            proc = subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT, env=run_env)
        return proc.returncode

    def _cuda_env_for(
        self, reservation: Optional[Dict[str, Any]]
    ) -> Optional[Dict[str, str]]:
        """``CUDA_VISIBLE_DEVICES`` for a GPU reservation.

        Remaps the reserved physical GPU ids to logical devices
        ``0..N-1`` inside the subprocess — a multi-GPU step's own config
        (e.g. ``pl_trainer.devices: 2``) just asks for "2 devices" without
        needing to know which physical ids the pipeline handed it.
        """
        if reservation is None or reservation.get("type") != "gpu":
            return None
        return {"CUDA_VISIBLE_DEVICES": ",".join(str(g) for g in reservation["ids"])}

    # ------------------------------------------------------------------
    # Preflight
    # ------------------------------------------------------------------

    def _preflight(self, steps: List[DictConfig]) -> None:
        """Compose every enabled Hydra-mode step's config before step 1 runs.

        Catches composition-time errors — bad keys, broken interpolations,
        missing referenced files — across the *whole* pipeline up front, so
        a typo in step 47 doesn't surface only after steps 1-46 finished.
        Does not catch deeper semantic/training errors. ``command`` steps
        (non-Hydra CLIs, e.g. ``pytorch-smt-tools``) are skipped — they
        don't understand ``mode=validate-config``.

        Raises:
            RuntimeError: If any enabled step fails to compose, listing
                every failure (not just the first).
        """
        errors = []
        for step in steps:
            if not step.get("enabled", True):
                continue
            if not self._is_hydra_step(step):
                continue
            cmd = self._build_command(step) + ["mode=validate-config"]
            proc = self._run_preflight_check(cmd)
            if proc.returncode != 0:
                stderr = getattr(proc, "stderr", "") or ""
                errors.append((step.name, stderr[-4000:]))

        if errors:
            detail = "\n".join(f"- {name}: {err}" for name, err in errors)
            raise RuntimeError(
                f"PipelineRunner preflight failed for {len(errors)} "
                f"step{'' if len(errors) == 1 else 's'}, nothing was run:\n{detail}"
            )

    # ------------------------------------------------------------------
    # Parallel scheduling
    # ------------------------------------------------------------------

    def _step_worker(
        self,
        name: str,
        cmd: List[str],
        log_path: str,
        env: Optional[Dict[str, str]],
        start: float,
        reservation: Dict[str, Any],
        results_q: "queue.Queue",
    ) -> None:
        """Thread body: blocks on `_run_step`, never raises — an exception
        here (e.g. a broken mock in tests) is reported as a failed step
        rather than crashing silently in a background thread."""
        try:
            exit_code = self._run_step(cmd, log_path, env=env)
        except Exception:
            logger.exception("PipelineRunner — step '%s' raised unexpectedly.", name)
            exit_code = 1
        elapsed = time.perf_counter() - start
        results_q.put((name, exit_code, elapsed, reservation))

    def run(self) -> List[Dict[str, Any]]:
        """Run every enabled step, honoring dependencies, resources,
        resume, and overwrite.

        Steps whose ``depends_on`` are all satisfied and whose resource
        request (``gpus``/``vram_gb``) can be reserved run concurrently,
        up to what ``pipeline.resources`` allows. A step whose dependency
        failed is never attempted — recorded ``skipped`` instead.

        Returns:
            List of per-step result dicts (``name``, ``status`` — ``done``,
            ``failed``, or ``skipped`` —, ``exit_code``, ``duration_s``,
            ``log_path``), ordered as in ``steps`` (disabled steps
            excluded).

        Raises:
            RuntimeError: On preflight failure, or when any step fails and
                ``on_error: stop`` (the default) — raised after every
                already-running step finishes naturally (none are killed).
        """
        steps = list(self.pipeline_cfg.steps)
        os.makedirs(self._output_base_dir(), exist_ok=True)
        os.makedirs(os.path.join(self._output_base_dir(), _LOG_DIR), exist_ok=True)

        if self.pipeline_cfg.get("preflight", True):
            self._preflight(steps)

        resume = self.pipeline_cfg.get("resume", True)
        state_exists = os.path.exists(self._state_path())
        state = self._load_state() if (resume and state_exists) else {"steps": {}}
        state.setdefault("steps", {})

        pipeline_overwrite = self._resolve_overwrite()
        on_error = self.pipeline_cfg.get("on_error", "stop")
        pool = _ResourcePool(self.pipeline_cfg.get("resources", None))
        idx_by_name = {s.name: i for i, s in enumerate(steps)}
        by_name = {s.name: s for s in steps if s.get("enabled", True)}

        for s in steps:
            if not s.get("enabled", True):
                logger.info("PipelineRunner — step '%s' disabled, skipping.", s.name)

        results_by_name: Dict[str, Dict[str, Any]] = {}
        done_names: Set[str] = set()
        pending_names: List[str] = []
        for s in steps:
            if not s.get("enabled", True):
                continue
            prior = state["steps"].get(s.name)
            force = self._step_should_force(s, pipeline_overwrite)
            if prior is not None and prior.get("status") == "done" and not force:
                logger.info(
                    "PipelineRunner — step '%s' already done, skipping.", s.name
                )
                results_by_name[s.name] = prior
                done_names.add(s.name)
            else:
                pending_names.append(s.name)

        failed_names: Set[str] = set()
        skipped_dep_failed: Set[str] = set()
        running: Dict[str, Dict[str, Any]] = {}
        results_q: "queue.Queue" = queue.Queue()
        stop_admitting = False

        def ready_names() -> List[str]:
            out = []
            for name in pending_names:
                if name in running or name in results_by_name:
                    continue
                step = by_name[name]
                deps = list(step.get("depends_on", []) or [])
                if any((d in failed_names or d in skipped_dep_failed) for d in deps):
                    skipped_dep_failed.add(name)
                    continue
                if all(d in done_names for d in deps):
                    out.append(name)
            return out

        while True:
            if not stop_admitting:
                for name in ready_names():
                    step = by_name[name]
                    reservation = pool.try_reserve(step)
                    if reservation is None:
                        continue
                    log_path = self._step_log_path(idx_by_name[name], name)
                    cmd = self._build_command(step)
                    env = self._cuda_env_for(reservation)
                    logger.info(
                        "PipelineRunner — starting step '%s': %s", name, " ".join(cmd)
                    )
                    self._append_pipeline_log(f"START {name} :: {' '.join(cmd)}")
                    start = time.perf_counter()
                    thread = threading.Thread(
                        target=self._step_worker,
                        args=(name, cmd, log_path, env, start, reservation, results_q),
                        daemon=True,
                    )
                    running[name] = {"log_path": log_path}
                    thread.start()

            for name in list(skipped_dep_failed):
                if name in results_by_name:
                    continue
                step_result = {
                    "name": name,
                    "status": "skipped",
                    "exit_code": None,
                    "duration_s": 0.0,
                    "log_path": None,
                }
                results_by_name[name] = step_result
                state["steps"][name] = step_result
                self._save_state(state)
                self._append_pipeline_log(f"SKIPPED {name} (dependency failed)")
                logger.warning(
                    "PipelineRunner — step '%s' skipped: a dependency failed.", name
                )

            remaining_pending = [
                n
                for n in pending_names
                if n not in results_by_name and n not in running
            ]

            if not running and not remaining_pending:
                break
            if not running and remaining_pending:
                if stop_admitting:
                    break
                raise RuntimeError(
                    "PipelineRunner: scheduling stalled — steps "
                    f"{remaining_pending} never became ready. This should have "
                    "been caught by static resource validation; check "
                    "pipeline.resources and each step's depends_on/gpus."
                )
            if not running:
                break

            name, exit_code, elapsed, reservation = results_q.get()
            step = by_name[name]
            pool.release(step, reservation)
            log_path = running.pop(name)["log_path"]

            status = "done" if exit_code == 0 else "failed"
            step_result = {
                "name": name,
                "status": status,
                "exit_code": exit_code,
                "duration_s": elapsed,
                "log_path": log_path,
            }
            results_by_name[name] = step_result
            state["steps"][name] = step_result
            self._save_state(state)
            self._append_pipeline_log(
                f"{'DONE' if status == 'done' else 'FAILED'} {name} "
                f"({elapsed:.1f}s, exit={exit_code})"
            )
            logger.info(
                "PipelineRunner — step '%s' %s in %.1fs (exit=%s). Log: %s",
                name,
                status,
                elapsed,
                exit_code,
                log_path,
            )

            if status == "done":
                done_names.add(name)
            else:
                failed_names.add(name)
                if on_error == "stop":
                    stop_admitting = True
                else:
                    logger.warning(
                        "PipelineRunner — step '%s' failed, on_error=continue, "
                        "proceeding.",
                        name,
                    )

        if failed_names and on_error == "stop":
            names_str = ", ".join(sorted(failed_names))
            raise RuntimeError(
                f"PipelineRunner: step(s) failed: {names_str}. See each step's "
                f"log under {os.path.join(self._output_base_dir(), _LOG_DIR)}. "
                "Fix the issue and re-run with resume (the default) to continue."
            )

        return [
            results_by_name[s.name]
            for s in steps
            if s.get("enabled", True) and s.name in results_by_name
        ]
