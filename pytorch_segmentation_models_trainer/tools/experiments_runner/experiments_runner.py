# -*- coding: utf-8 -*-
"""
/***************************************************************************
 pytorch_segmentation_models_trainer
                              -------------------
        begin                : 2026-05-07
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

import csv
import dataclasses
import json
import logging
import multiprocessing as mp
import os
import secrets
import statistics
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from omegaconf import DictConfig, OmegaConf

from pytorch_segmentation_models_trainer.utils.spatial_kfold import SpatialKFoldSplitter

# train is imported inside _seed_subprocess_worker so it runs in the subprocess,
# not in the parent ER process.  The _collect_metrics method is also unused now
# (metrics are collected inside the subprocess) but kept for API compatibility.

logger = logging.getLogger(__name__)

_STATE_FILE = "runner_state.json"
_SEED_RESULT_FILE = "_seed_result.json"


def _seed_subprocess_worker(yaml_cfg: str, output_dir: str, result_path: str) -> None:
    """Subprocess entry point: runs one training seed and writes results to a JSON file.

    Runs in a fresh ``spawn``-ed process so that CUDA contexts, DataLoader
    worker processes (persistent_workers=True / forkserver), and any other
    per-seed state are fully isolated between seeds.  The parent ExperimentsRunner
    process never touches a GPU or a DataLoader — it only reads the JSON result.

    Args:
        yaml_cfg: OmegaConf YAML string of the per-seed training config.
        output_dir: Directory for this seed's outputs (created if absent).
        result_path: Path where this function writes the JSON result dict.
    """
    import json as _json
    import os as _os

    import yaml as _yaml
    from omegaconf import OmegaConf as _OmegaConf
    from pytorch_segmentation_models_trainer.train import train as _train

    _os.makedirs(output_dir, exist_ok=True)

    try:
        cfg = _OmegaConf.create(_yaml.safe_load(yaml_cfg))
        trainer = _train(cfg)

        all_metrics: Dict[str, float] = {}
        try:
            all_metrics = {k: float(v) for k, v in trainer.callback_metrics.items()}
        except Exception:
            pass

        train_m = {
            k: v
            for k, v in all_metrics.items()
            if k.startswith("train/") or k.endswith("/train") or k.endswith("/train_epoch")
        }
        val_m = {
            k: v
            for k, v in all_metrics.items()
            if k.startswith("val/") or k.endswith("/val")
        }
        test_m = {
            k: v
            for k, v in all_metrics.items()
            if k.startswith("test/") or k.endswith("/test")
        }

        ckpt_cb = getattr(trainer, "checkpoint_callback", None)
        best_ckpt = (getattr(ckpt_cb, "best_model_path", "") or "") if ckpt_cb else ""
        epochs = trainer.current_epoch + 1

        result = {
            "ok": True,
            "train_metrics": train_m,
            "val_metrics": val_m,
            "test_metrics": test_m,
            "best_checkpoint_path": best_ckpt,
            "epochs_trained": epochs,
        }
    except Exception as exc:
        result = {"ok": False, "error": repr(exc)}

    with open(result_path, "w") as _f:
        _json.dump(result, _f)


@dataclass
class RunResult:
    """Result produced by a single training run.

    Args:
        run_idx: Zero-based index of this run within the experiment sequence.
        seed: Random seed used for this run.
        training_time_seconds: Wall-clock seconds from ``trainer.fit`` start to
            end (includes ``trainer.test`` when a test dataset is configured).
        train_metrics: Final training metrics from
            ``trainer.callback_metrics``, keyed with the ``train/`` prefix.
        val_metrics: Final validation metrics, keyed with the ``val/`` prefix.
        test_metrics: Final test metrics, keyed with the ``test/`` prefix.
            Empty when no test dataset is configured.
        output_dir: Absolute path to the per-run output directory
            (``<output_base_dir>/run_<run_idx:02d>_seed<seed>``).
        fold_idx: Zero-based fold index when in k-fold mode; ``None`` otherwise.
        epochs_trained: Number of epochs actually completed, including early
            stopping.  ``None`` when not yet populated.
        best_checkpoint_path: Absolute path to the best checkpoint saved by
            ``ModelCheckpoint`` for this run.  Empty string when no checkpoint
            callback is configured.  ``None`` when not yet populated.
    """

    run_idx: int
    seed: int
    training_time_seconds: float
    train_metrics: Dict[str, float]
    val_metrics: Dict[str, float]
    test_metrics: Dict[str, float]
    output_dir: str
    fold_idx: Optional[int] = None
    epochs_trained: Optional[int] = None
    best_checkpoint_path: Optional[str] = None


class ExperimentsRunner:
    """Runs successive training experiments in series with different seeds.

    The runner takes a full Hydra ``DictConfig`` that contains both the usual
    training configuration and an ``experiments_runner`` block.  For each run
    it strips that block, overrides the seed, adjusts the Lightning
    ``default_root_dir``, and injects run identity into the logger before
    delegating to :func:`train`.

    After every completed run the runner writes a ``runner_state.json`` to
    ``output_base_dir`` and, when ``save_summary`` is enabled, updates
    ``summary.csv``.  Set ``resume: true`` to skip already-completed runs
    on restart.

    Args:
        cfg: Full Hydra config including the ``experiments_runner`` sub-tree.

    Raises:
        ValueError: If neither ``seeds`` nor ``n_runs`` is provided, or if
            both are provided with inconsistent values.

    Example::

        runner = ExperimentsRunner(cfg)
        results = runner.run()
        for r in results:
            print(r.seed, r.val_metrics)
    """

    def __init__(self, cfg: DictConfig) -> None:
        self.cfg = cfg
        self.runner_cfg = cfg.experiments_runner
        self._validate()

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def _validate(self) -> None:
        seeds = self.runner_cfg.get("seeds", None)
        n_runs = self.runner_cfg.get("n_runs", None)

        if seeds is None and n_runs is None:
            raise ValueError(
                "experiments_runner: either 'seeds' or 'n_runs' must be provided."
            )
        if seeds is not None and n_runs is not None and n_runs != len(seeds):
            raise ValueError(
                f"experiments_runner.n_runs ({n_runs}) must equal "
                f"len(experiments_runner.seeds) ({len(seeds)}) when both are provided."
            )

    # ------------------------------------------------------------------
    # Public helpers (also used in tests)
    # ------------------------------------------------------------------

    def _n_runs(self) -> int:
        """Return the resolved number of runs."""
        seeds = self.runner_cfg.get("seeds", None)
        if seeds is not None:
            return len(seeds)
        return int(self.runner_cfg.n_runs)

    def _resolve_seeds(self) -> List[int]:
        """Return the list of seeds for all runs.

        Uses explicit seeds when provided; otherwise generates
        ``n_runs`` cryptographically random 31-bit integers.
        """
        seeds = self.runner_cfg.get("seeds", None)
        if seeds is not None:
            return list(seeds)
        return [secrets.randbelow(2**31) for _ in range(self._n_runs())]

    def _build_run_cfg(
        self,
        run_idx: int,
        seed: int,
        fold_idx: Optional[int] = None,
        fold_paths: Optional[Tuple] = None,
    ) -> Tuple[DictConfig, str]:
        """Build a training config for a single run.

        Removes the ``experiments_runner`` block, sets the seed, adjusts
        ``pl_trainer.default_root_dir`` to an isolated per-run directory, and
        injects the run identity into the configured logger (if any).

        When ``fold_idx`` and ``fold_paths`` are provided (k-fold mode), the
        ``train_dataset.input_csv_path`` and ``val_dataset.input_csv_path``
        fields are overridden with the fold-specific CSVs and the output
        directory name includes the fold tag.

        Args:
            run_idx: Zero-based run index (unique across all seed × fold combos).
            seed: Seed to inject into the config.
            fold_idx: Zero-based fold index.  ``None`` in seed-only mode.
            fold_paths: Tuple of ``(train_csv_path, val_csv_path)`` for the
                current fold.  ``None`` in seed-only mode.

        Returns:
            Tuple of (run_cfg, output_dir).
        """
        keys = [k for k in self.cfg if k != "experiments_runner"]
        run_cfg = OmegaConf.masked_copy(self.cfg, keys)

        # Allow writing to the (possibly struct-locked) config copy.
        OmegaConf.set_struct(run_cfg, False)
        OmegaConf.update(run_cfg, "seed", seed, merge=True)

        if fold_idx is None:
            output_dir = os.path.join(
                self.runner_cfg.output_base_dir, f"run_{run_idx:02d}_seed{seed}"
            )
        else:
            output_dir = os.path.join(
                self.runner_cfg.output_base_dir, f"fold_{fold_idx:02d}_seed{seed}"
            )

        if "pl_trainer" in run_cfg:
            OmegaConf.update(
                run_cfg, "pl_trainer.default_root_dir", output_dir, merge=True
            )

        if fold_idx is not None and fold_paths is not None:
            train_csv, val_csv = fold_paths
            if "train_dataset" in run_cfg:
                OmegaConf.update(
                    run_cfg, "train_dataset.input_csv_path", str(train_csv), merge=True
                )
            if "val_dataset" in run_cfg:
                OmegaConf.update(
                    run_cfg, "val_dataset.input_csv_path", str(val_csv), merge=True
                )

        self._inject_run_identity_into_logger(run_cfg, run_idx, seed)

        return run_cfg, output_dir

    def _inject_run_identity_into_logger(
        self, run_cfg: DictConfig, run_idx: int, seed: int
    ) -> None:
        """Stamp run_idx and seed into the logger config.

        * If the logger has a ``version`` field (TensorBoardLogger, CSVLogger):
          set it to ``"run_<idx:02d>_seed<seed>"``.
        * If the logger has a ``name`` field but no ``version`` (WandbLogger):
          append ``_run_<idx:02d>_seed<seed>`` to the name.
        * If no ``logger`` key or logger is a plain bool (Lightning default):
          do nothing.

        Args:
            run_cfg: Per-run config (already stripped of experiments_runner).
            run_idx: Zero-based run index.
            seed: Seed for this run.
        """
        if "logger" not in run_cfg:
            return
        logger_cfg = run_cfg.logger
        # Plain bool → Lightning default logger, nothing to modify.
        if not isinstance(logger_cfg, DictConfig):
            return

        run_tag = f"run_{run_idx:02d}_seed{seed}"
        if "version" in logger_cfg:
            OmegaConf.update(run_cfg, "logger.version", run_tag, merge=True)
        elif "name" in logger_cfg:
            current_name = OmegaConf.select(run_cfg, "logger.name")
            OmegaConf.update(
                run_cfg, "logger.name", f"{current_name}_{run_tag}", merge=True
            )

    def _all_run_metrics(self, r: RunResult) -> Dict[str, float]:
        """Merge train, val and test metrics into a single dict."""
        return {**r.train_metrics, **r.val_metrics, **r.test_metrics}

    def _resolve_representative_metric(self, results: List[RunResult]) -> Optional[str]:
        """Return the metric key used for representative / best-run selection.

        Uses ``experiments_runner.representative_metric`` when set; otherwise
        falls back to the first val metric key found (alphabetical order).
        Returns ``None`` when no val metrics exist across all results.
        """
        metric_from_cfg = self.runner_cfg.get("representative_metric", None)
        if metric_from_cfg:
            return metric_from_cfg
        all_val_keys = sorted({k for r in results for k in r.val_metrics})
        return all_val_keys[0] if all_val_keys else None

    def _select_representative_run(
        self, results: List[RunResult], metric_key: str
    ) -> Optional[RunResult]:
        """Return the run whose metric value is closest to the mean (representative).

        Args:
            results: All completed runs.
            metric_key: Metric to compare (searched across train/val/test).

        Returns:
            The :class:`RunResult` closest to the mean, or ``None`` when the
            metric is absent in every run.
        """
        eligible = [r for r in results if metric_key in self._all_run_metrics(r)]
        if not eligible:
            return None
        mean_val = statistics.mean(
            self._all_run_metrics(r)[metric_key] for r in eligible
        )
        return min(
            eligible, key=lambda r: abs(self._all_run_metrics(r)[metric_key] - mean_val)
        )

    def _select_best_run(
        self, results: List[RunResult], metric_key: str
    ) -> Optional[RunResult]:
        """Return the run with the highest value of ``metric_key``.

        Args:
            results: All completed runs.
            metric_key: Metric to maximise (searched across train/val/test).

        Returns:
            The :class:`RunResult` with the highest metric value, or ``None``
            when the metric is absent in every run.
        """
        eligible = [r for r in results if metric_key in self._all_run_metrics(r)]
        if not eligible:
            return None
        return max(eligible, key=lambda r: self._all_run_metrics(r)[metric_key])

    def _collect_metrics(
        self, trainer
    ) -> Tuple[Dict[str, float], Dict[str, float], Dict[str, float]]:
        """Extract and split ``trainer.callback_metrics`` by prefix.

        Args:
            trainer: A fitted :class:`pytorch_lightning.Trainer` instance.

        Returns:
            Tuple of (train_metrics, val_metrics, test_metrics).
        """
        all_metrics = {k: float(v) for k, v in trainer.callback_metrics.items()}
        train_metrics = {
            k: v
            for k, v in all_metrics.items()
            if k.startswith("train/") or k.endswith("/train")
        }
        val_metrics = {
            k: v
            for k, v in all_metrics.items()
            if k.startswith("val/") or k.endswith("/val")
        }
        test_metrics = {
            k: v
            for k, v in all_metrics.items()
            if k.startswith("test/") or k.endswith("/test")
        }
        return train_metrics, val_metrics, test_metrics

    # ------------------------------------------------------------------
    # State persistence
    # ------------------------------------------------------------------

    def _state_path(self) -> str:
        return os.path.join(self.runner_cfg.output_base_dir, _STATE_FILE)

    def _save_state(self, all_seeds: List[int], results: List[RunResult]) -> None:
        """Persist seed list, completed runs, and selection info to ``runner_state.json``.

        Args:
            all_seeds: Full ordered seed list for the experiment sequence.
            results: Completed :class:`RunResult` objects so far.
        """
        os.makedirs(self.runner_cfg.output_base_dir, exist_ok=True)
        metric_key = self._resolve_representative_metric(results)
        rep_run = (
            self._select_representative_run(results, metric_key) if metric_key else None
        )
        best_run = self._select_best_run(results, metric_key) if metric_key else None

        state = {
            "all_seeds": all_seeds,
            "completed_runs": [dataclasses.asdict(r) for r in results],
            "representative": (
                {
                    "run_idx": rep_run.run_idx,
                    "checkpoint_path": rep_run.best_checkpoint_path or "",
                }
                if rep_run
                else None
            ),
            "best_run": (
                {
                    "run_idx": best_run.run_idx,
                    "checkpoint_path": best_run.best_checkpoint_path or "",
                }
                if best_run
                else None
            ),
        }
        with open(self._state_path(), "w") as f:
            json.dump(state, f, indent=2)

    def _load_state(self) -> dict:
        """Load state from ``runner_state.json``."""
        with open(self._state_path()) as f:
            return json.load(f)

    # ------------------------------------------------------------------
    # Core execution
    # ------------------------------------------------------------------

    def _run_single(
        self,
        run_idx: int,
        seed: int,
        fold_idx: Optional[int] = None,
        fold_paths: Optional[Tuple] = None,
    ) -> RunResult:
        """Execute one training run in an isolated subprocess and return its result.

        Each seed is run inside a freshly ``spawn``-ed process so that CUDA
        contexts and DataLoader worker processes (``persistent_workers=True`` /
        ``forkserver``) are fully torn down between seeds, preventing the
        inter-seed crash that plagued the original in-process approach.

        Args:
            run_idx: Zero-based index of this run.
            seed: Seed for this run.
            fold_idx: Zero-based fold index when in k-fold mode; ``None`` otherwise.
            fold_paths: ``(train_csv, val_csv)`` paths for the current fold.

        Returns:
            :class:`RunResult` with timing and metrics.

        Raises:
            RuntimeError: If the subprocess exits with a non-zero code or if
                the worker reports an internal exception.
        """
        run_cfg, output_dir = self._build_run_cfg(
            run_idx, seed, fold_idx=fold_idx, fold_paths=fold_paths
        )
        os.makedirs(output_dir, exist_ok=True)

        logger.info(
            "ExperimentsRunner — starting run %d | seed=%d%s | output=%s",
            run_idx + 1,
            seed,
            f" | fold={fold_idx}" if fold_idx is not None else "",
            output_dir,
        )

        yaml_cfg = OmegaConf.to_yaml(run_cfg)
        result_path = os.path.join(output_dir, _SEED_RESULT_FILE)

        start = time.perf_counter()
        ctx = mp.get_context("spawn")
        p = ctx.Process(
            target=_seed_subprocess_worker,
            args=(yaml_cfg, output_dir, result_path),
        )
        p.start()
        p.join()
        elapsed = time.perf_counter() - start

        if p.exitcode != 0:
            raise RuntimeError(
                f"ExperimentsRunner: run {run_idx} (seed={seed}) subprocess "
                f"exited with code {p.exitcode}. Check logs in {output_dir}."
            )

        with open(result_path) as f:
            data = json.load(f)

        if not data.get("ok"):
            raise RuntimeError(
                f"ExperimentsRunner: run {run_idx} (seed={seed}) failed — "
                f"{data.get('error', 'unknown error')}"
            )

        train_metrics = data["train_metrics"]
        val_metrics = data["val_metrics"]
        test_metrics = data["test_metrics"]
        epochs_trained = data["epochs_trained"]
        best_checkpoint_path = data["best_checkpoint_path"]

        logger.info(
            "ExperimentsRunner — run %d done in %.1fs | epochs=%d | val: %s",
            run_idx + 1,
            elapsed,
            epochs_trained,
            val_metrics,
        )

        return RunResult(
            run_idx=run_idx,
            seed=seed,
            training_time_seconds=elapsed,
            train_metrics=train_metrics,
            val_metrics=val_metrics,
            test_metrics=test_metrics,
            output_dir=output_dir,
            fold_idx=fold_idx,
            epochs_trained=epochs_trained,
            best_checkpoint_path=best_checkpoint_path,
        )

    def run(self) -> List[RunResult]:
        """Run all experiments in series, with optional resume and k-fold support.

        When ``experiments_runner.kfold`` is present, iterates over all
        ``seed × fold`` combinations (total = ``len(seeds) × n_splits`` runs).
        Otherwise uses the original seed-only loop.

        Returns:
            List of :class:`RunResult`, one per run, ordered by ``run_idx``.
        """
        kfold_cfg = self.runner_cfg.get("kfold", None)
        if kfold_cfg is not None:
            return self._run_kfold_loop(kfold_cfg)
        return self._run_seed_loop()

    def _run_seed_loop(self) -> List[RunResult]:
        """Original seed-only run loop (no k-fold).

        On each completed run the state is persisted to ``runner_state.json``
        and ``summary.csv`` is updated (when ``save_summary`` is enabled).
        When ``resume: true`` is set and a state file exists, already-completed
        runs are skipped; seeds are loaded from the state file so
        auto-generated seeds remain stable across restarts.

        Returns:
            List of :class:`RunResult`, one per run, ordered by ``run_idx``.
        """
        resume = self.runner_cfg.get("resume", False)
        state_exists = os.path.exists(self._state_path())

        if resume and state_exists:
            state = self._load_state()
            all_seeds = state["all_seeds"]
            completed_results = [RunResult(**r) for r in state["completed_runs"]]
            logger.info(
                "ExperimentsRunner — resuming: %d/%d runs already complete.",
                len(completed_results),
                len(all_seeds),
            )
        else:
            all_seeds = self._resolve_seeds()
            completed_results = []

        completed_indices = {r.run_idx for r in completed_results}
        results: List[RunResult] = list(completed_results)

        for run_idx, seed in enumerate(all_seeds):
            if run_idx in completed_indices:
                logger.info(
                    "ExperimentsRunner — run %d (seed=%d) already done, skipping.",
                    run_idx,
                    seed,
                )
                continue

            result = self._run_single(run_idx, seed)
            results.append(result)
            results_sorted = sorted(results, key=lambda r: r.run_idx)

            self._save_state(all_seeds, results_sorted)
            if self.runner_cfg.get("save_summary", True):
                self._save_summary(results_sorted)

        return sorted(results, key=lambda r: r.run_idx)

    def _run_kfold_loop(self, kfold_cfg: DictConfig) -> List[RunResult]:
        """K-fold loop: iterates seed × fold, injecting per-fold CSV paths.

        Fold CSVs are generated once (idempotent; same seed → same split) and
        reused across seeds.  State persistence and resume work identically to
        the seed-only loop, with ``run_idx`` spanning all seed × fold combos.

        Args:
            kfold_cfg: The ``experiments_runner.kfold`` sub-config.

        Returns:
            List of :class:`RunResult` ordered by ``run_idx``.
        """
        output_base_dir = self.runner_cfg.output_base_dir
        folds_dir = os.path.join(output_base_dir, kfold_cfg.save_fold_csvs_dir)

        splitter = SpatialKFoldSplitter(kfold_cfg)
        fold_paths = splitter.generate_and_save_folds(
            kfold_cfg.input_csv_path, folds_dir
        )
        n_folds = len(fold_paths)

        resume = self.runner_cfg.get("resume", False)
        state_exists = os.path.exists(self._state_path())

        if resume and state_exists:
            state = self._load_state()
            all_seeds = state["all_seeds"]
            completed_results = [RunResult(**r) for r in state["completed_runs"]]
            logger.info(
                "ExperimentsRunner — kfold resuming: %d/%d runs already complete.",
                len(completed_results),
                len(all_seeds) * n_folds,
            )
        else:
            all_seeds = self._resolve_seeds()
            completed_results = []

        completed_indices = {r.run_idx for r in completed_results}
        results: List[RunResult] = list(completed_results)
        total_runs = len(all_seeds) * n_folds

        for seed_idx, seed in enumerate(all_seeds):
            for fold_idx, paths in enumerate(fold_paths):
                run_idx = seed_idx * n_folds + fold_idx

                if run_idx in completed_indices:
                    logger.info(
                        "ExperimentsRunner — run %d (seed=%d fold=%d) already done, skipping.",
                        run_idx,
                        seed,
                        fold_idx,
                    )
                    continue

                result = self._run_single(
                    run_idx, seed, fold_idx=fold_idx, fold_paths=paths
                )
                results.append(result)
                results_sorted = sorted(results, key=lambda r: r.run_idx)

                self._save_state(all_seeds, results_sorted)
                if self.runner_cfg.get("save_summary", True):
                    self._save_summary(results_sorted)

        return sorted(results, key=lambda r: r.run_idx)

    # ------------------------------------------------------------------
    # Summary CSV
    # ------------------------------------------------------------------

    def _save_summary(self, results: List[RunResult]) -> None:
        """Write ``summary.csv`` with per-run metrics and mean ± std rows.

        Columns included beyond the metric keys:

        * ``epochs_trained`` — actual epoch count (early-stop aware).
        * ``best_checkpoint_path`` — path to the best checkpoint for that run.
        * ``representative`` — ``"*"`` for the run closest to the mean metric.
        * ``best_run`` — ``"*"`` for the run with the highest metric value.

        When any result has a non-``None`` ``fold_idx``, a ``fold_idx`` column
        is included between ``run`` and ``seed``.

        Args:
            results: List of completed :class:`RunResult` objects.
        """
        output_base_dir = self.runner_cfg.output_base_dir
        os.makedirs(output_base_dir, exist_ok=True)
        summary_path = os.path.join(output_base_dir, "summary.csv")

        has_fold = any(r.fold_idx is not None for r in results)
        all_metric_keys = sorted({k for r in results for k in self._all_run_metrics(r)})

        base_cols = [
            "run",
            "seed",
            "duration_s",
            "epochs_trained",
            "best_checkpoint_path",
        ]
        if has_fold:
            base_cols = [
                "run",
                "fold_idx",
                "seed",
                "duration_s",
                "epochs_trained",
                "best_checkpoint_path",
            ]
        fieldnames = base_cols + all_metric_keys + ["representative", "best_run"]

        def _fmt(v) -> str:
            return f"{v:.6f}"

        metric_key = self._resolve_representative_metric(results)
        rep_run = (
            self._select_representative_run(results, metric_key) if metric_key else None
        )
        best_run_sel = (
            self._select_best_run(results, metric_key) if metric_key else None
        )

        rows = []
        for r in results:
            m = self._all_run_metrics(r)
            row: Dict = {
                "run": r.run_idx,
                "seed": r.seed,
                "duration_s": f"{r.training_time_seconds:.2f}",
                "epochs_trained": (
                    str(r.epochs_trained) if r.epochs_trained is not None else ""
                ),
                "best_checkpoint_path": r.best_checkpoint_path or "",
                "representative": (
                    "*"
                    if (rep_run is not None and r.run_idx == rep_run.run_idx)
                    else ""
                ),
                "best_run": (
                    "*"
                    if (best_run_sel is not None and r.run_idx == best_run_sel.run_idx)
                    else ""
                ),
            }
            if has_fold:
                row["fold_idx"] = r.fold_idx if r.fold_idx is not None else ""
            for k in all_metric_keys:
                row[k] = _fmt(m[k]) if k in m else ""
            rows.append(row)

        durations = [r.training_time_seconds for r in results]
        epochs_list = [
            r.epochs_trained for r in results if r.epochs_trained is not None
        ]

        mean_row: Dict = {
            "run": "mean",
            "seed": "-",
            "duration_s": _fmt(statistics.mean(durations)),
            "epochs_trained": (
                f"{statistics.mean(epochs_list):.2f}" if epochs_list else ""
            ),
            "best_checkpoint_path": "",
            "representative": "",
            "best_run": "",
        }
        std_row: Dict = {
            "run": "std",
            "seed": "-",
            "duration_s": (
                _fmt(statistics.stdev(durations)) if len(durations) >= 2 else "0.000000"
            ),
            "epochs_trained": (
                f"{statistics.stdev(epochs_list):.2f}"
                if len(epochs_list) >= 2
                else ("0.00" if epochs_list else "")
            ),
            "best_checkpoint_path": "",
            "representative": "",
            "best_run": "",
        }
        if has_fold:
            mean_row["fold_idx"] = ""
            std_row["fold_idx"] = ""

        for k in all_metric_keys:
            vals = [
                self._all_run_metrics(r)[k]
                for r in results
                if k in self._all_run_metrics(r)
            ]
            mean_row[k] = _fmt(statistics.mean(vals)) if vals else ""
            std_row[k] = (
                _fmt(statistics.stdev(vals))
                if len(vals) >= 2
                else ("0.000000" if vals else "")
            )

        rows.extend([mean_row, std_row])

        with open(summary_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

        logger.info("ExperimentsRunner — summary saved to %s", summary_path)
