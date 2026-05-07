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
import logging
import os
import secrets
import statistics
import time
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

from omegaconf import DictConfig, OmegaConf

from pytorch_segmentation_models_trainer.train import train

logger = logging.getLogger(__name__)


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
            (``<output_base_dir>/run_<run_idx:02d>``).
    """

    run_idx: int
    seed: int
    training_time_seconds: float
    train_metrics: Dict[str, float]
    val_metrics: Dict[str, float]
    test_metrics: Dict[str, float]
    output_dir: str


class ExperimentsRunner:
    """Runs successive training experiments in series with different seeds.

    The runner takes a full Hydra ``DictConfig`` that contains both the usual
    training configuration and an ``experiments_runner`` block.  For each run
    it strips that block, overrides the seed and the Lightning
    ``default_root_dir``, and delegates to :func:`train`.

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

    def _build_run_cfg(self, run_idx: int, seed: int) -> Tuple[DictConfig, str]:
        """Build a training config for a single run.

        Removes the ``experiments_runner`` block, sets the seed, and adjusts
        ``pl_trainer.default_root_dir`` to an isolated per-run directory.

        Args:
            run_idx: Zero-based run index.
            seed: Seed to inject into the config.

        Returns:
            Tuple of (run_cfg, output_dir).
        """
        keys = [k for k in self.cfg if k != "experiments_runner"]
        run_cfg = OmegaConf.masked_copy(self.cfg, keys)

        # Allow writing to the (possibly struct-locked) config copy.
        OmegaConf.set_struct(run_cfg, False)
        OmegaConf.update(run_cfg, "seed", seed, merge=True)

        output_dir = os.path.join(
            self.runner_cfg.output_base_dir, f"run_{run_idx:02d}"
        )
        if "pl_trainer" in run_cfg:
            OmegaConf.update(
                run_cfg, "pl_trainer.default_root_dir", output_dir, merge=True
            )

        return run_cfg, output_dir

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
    # Core execution
    # ------------------------------------------------------------------

    def _run_single(self, run_idx: int, seed: int) -> RunResult:
        """Execute one training run and return its result.

        Args:
            run_idx: Zero-based index of this run.
            seed: Seed for this run.

        Returns:
            :class:`RunResult` with timing and metrics.
        """
        run_cfg, output_dir = self._build_run_cfg(run_idx, seed)

        logger.info(
            "ExperimentsRunner — starting run %d/%d | seed=%d | output=%s",
            run_idx + 1,
            self._n_runs(),
            seed,
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
        )

    def run(self) -> List[RunResult]:
        """Run all experiments in series.

        Returns:
            List of :class:`RunResult`, one per run, in execution order.
        """
        seeds = self._resolve_seeds()
        results: List[RunResult] = []

        for run_idx, seed in enumerate(seeds):
            result = self._run_single(run_idx, seed)
            results.append(result)

        if self.runner_cfg.get("save_summary", True):
            self._save_summary(results)

        return results

    # ------------------------------------------------------------------
    # Summary CSV
    # ------------------------------------------------------------------

    def _save_summary(self, results: List[RunResult]) -> None:
        """Write ``summary.csv`` with per-run metrics and mean ± std rows.

        Args:
            results: List of completed :class:`RunResult` objects.
        """
        output_base_dir = self.runner_cfg.output_base_dir
        os.makedirs(output_base_dir, exist_ok=True)
        summary_path = os.path.join(output_base_dir, "summary.csv")

        all_metric_keys = sorted(
            {k for r in results for k in {**r.train_metrics, **r.val_metrics, **r.test_metrics}}
        )
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
            "duration_s": _fmt(statistics.stdev(durations)) if len(durations) >= 2 else "0.000000",
        }

        for k in all_metric_keys:
            vals = [_all_metrics(r)[k] for r in results if k in _all_metrics(r)]
            mean_row[k] = _fmt(statistics.mean(vals)) if vals else ""
            std_row[k] = (
                _fmt(statistics.stdev(vals)) if len(vals) >= 2 else ("0.000000" if vals else "")
            )

        rows.extend([mean_row, std_row])

        with open(summary_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

        logger.info("ExperimentsRunner — summary saved to %s", summary_path)
