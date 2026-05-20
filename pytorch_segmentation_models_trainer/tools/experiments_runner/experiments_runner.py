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
import os
import secrets
import statistics
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from omegaconf import DictConfig, OmegaConf

from pytorch_segmentation_models_trainer.train import train
from pytorch_segmentation_models_trainer.utils.spatial_kfold import SpatialKFoldSplitter

logger = logging.getLogger(__name__)

_STATE_FILE = "runner_state.json"


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
    """

    run_idx: int
    seed: int
    training_time_seconds: float
    train_metrics: Dict[str, float]
    val_metrics: Dict[str, float]
    test_metrics: Dict[str, float]
    output_dir: str
    fold_idx: Optional[int] = None


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
        train_metrics = {k: v for k, v in all_metrics.items() if k.startswith("train/")}
        val_metrics = {k: v for k, v in all_metrics.items() if k.startswith("val/")}
        test_metrics = {k: v for k, v in all_metrics.items() if k.startswith("test/")}
        return train_metrics, val_metrics, test_metrics

    # ------------------------------------------------------------------
    # State persistence
    # ------------------------------------------------------------------

    def _state_path(self) -> str:
        return os.path.join(self.runner_cfg.output_base_dir, _STATE_FILE)

    def _save_state(self, all_seeds: List[int], results: List[RunResult]) -> None:
        """Persist seed list and completed runs to ``runner_state.json``.

        Args:
            all_seeds: Full ordered seed list for the experiment sequence.
            results: Completed :class:`RunResult` objects so far.
        """
        os.makedirs(self.runner_cfg.output_base_dir, exist_ok=True)
        state = {
            "all_seeds": all_seeds,
            "completed_runs": [dataclasses.asdict(r) for r in results],
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
        """Execute one training run and return its result.

        Args:
            run_idx: Zero-based index of this run.
            seed: Seed for this run.
            fold_idx: Zero-based fold index when in k-fold mode; ``None`` otherwise.
            fold_paths: ``(train_csv, val_csv)`` paths for the current fold.

        Returns:
            :class:`RunResult` with timing and metrics.
        """
        run_cfg, output_dir = self._build_run_cfg(
            run_idx, seed, fold_idx=fold_idx, fold_paths=fold_paths
        )

        logger.info(
            "ExperimentsRunner — starting run %d | seed=%d%s | output=%s",
            run_idx + 1,
            seed,
            f" | fold={fold_idx}" if fold_idx is not None else "",
            output_dir,
        )

        start = time.perf_counter()
        trainer = train(run_cfg)
        elapsed = time.perf_counter() - start

        train_metrics, val_metrics, test_metrics = self._collect_metrics(trainer)

        logger.info(
            "ExperimentsRunner — run %d done in %.1fs | val: %s",
            run_idx + 1,
            elapsed,
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

        When any result has a non-``None`` ``fold_idx``, a ``fold_idx`` column
        is included between ``run`` and ``seed``.

        Args:
            results: List of completed :class:`RunResult` objects.
        """
        output_base_dir = self.runner_cfg.output_base_dir
        os.makedirs(output_base_dir, exist_ok=True)
        summary_path = os.path.join(output_base_dir, "summary.csv")

        has_fold = any(r.fold_idx is not None for r in results)
        all_metric_keys = sorted(
            {
                k
                for r in results
                for k in {**r.train_metrics, **r.val_metrics, **r.test_metrics}
            }
        )
        if has_fold:
            fieldnames = ["run", "fold_idx", "seed", "duration_s"] + all_metric_keys
        else:
            fieldnames = ["run", "seed", "duration_s"] + all_metric_keys

        def _all_metrics(r: RunResult) -> Dict[str, float]:
            return {**r.train_metrics, **r.val_metrics, **r.test_metrics}

        def _fmt(v) -> str:
            return f"{v:.6f}"

        rows = []
        for r in results:
            m = _all_metrics(r)
            row: Dict = {
                "run": r.run_idx,
                "seed": r.seed,
                "duration_s": f"{r.training_time_seconds:.2f}",
            }
            if has_fold:
                row["fold_idx"] = r.fold_idx if r.fold_idx is not None else ""
            for k in all_metric_keys:
                row[k] = _fmt(m[k]) if k in m else ""
            rows.append(row)

        durations = [r.training_time_seconds for r in results]
        mean_row: Dict = {
            "run": "mean",
            "seed": "-",
            "duration_s": _fmt(statistics.mean(durations)),
        }
        std_row: Dict = {
            "run": "std",
            "seed": "-",
            "duration_s": (
                _fmt(statistics.stdev(durations)) if len(durations) >= 2 else "0.000000"
            ),
        }
        if has_fold:
            mean_row["fold_idx"] = ""
            std_row["fold_idx"] = ""

        for k in all_metric_keys:
            vals = [_all_metrics(r)[k] for r in results if k in _all_metrics(r)]
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
