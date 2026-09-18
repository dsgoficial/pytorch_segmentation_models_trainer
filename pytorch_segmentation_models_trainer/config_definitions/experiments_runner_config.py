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

from dataclasses import dataclass, field
from typing import Any, List, Optional

from hydra.core.config_store import ConfigStore


@dataclass
class ExperimentsRunnerConfig:
    """Configuration for running successive training experiments with different seeds.

    Exactly one of ``seeds`` or ``n_runs`` must be supplied (or both, provided
    ``n_runs == len(seeds)``).  If only ``n_runs`` is given the seeds are
    generated randomly at runtime using :mod:`secrets`.

    Args:
        n_runs: Number of successive runs. Required when ``seeds`` is absent.
            Ignored (or validated for consistency) when ``seeds`` is present.
        seeds: Explicit seed list — one entry per run.  Determines the number
            of runs.  When absent, ``n_runs`` random seeds are generated.
        output_base_dir: Root directory for per-run outputs.  Each run writes
            to ``<output_base_dir>/run_<idx:02d>_seed<seed>/``.
        save_summary: When ``True`` (default) a ``summary.csv`` with per-run
            metrics and mean ± std aggregation is written incrementally to
            ``output_base_dir`` after each run finishes.
        summary_metrics: Metric keys (as logged by Lightning, e.g.
            ``"val/loss"``) to highlight in log output.  All available
            metrics are always written to the CSV regardless of this list.
        resume: When ``True`` (default) and a ``runner_state.json`` exists in
            ``output_base_dir``, already-completed runs are skipped and
            execution continues from the first pending run — i.e. runs are
            idempotent by default: re-launching the same config is safe and
            will not repeat or overwrite finished work. Seeds are loaded
            from the state file so auto-generated seeds are stable across
            restarts. Set explicitly to ``False`` to always start every run
            fresh, ignoring any existing state.
        overwrite: Forces re-execution of runs that are already marked
            complete in ``runner_state.json`` (which ``resume`` would
            otherwise skip). ``True`` forces every run; a list of ints
            forces only those ``run_idx`` values (the ``run`` column in
            ``summary.csv`` / the ``run_XX_seed...`` output directory
            prefix). ``False`` or ``None`` (default) forces nothing. Each
            forced run's previous output directory is deleted before it is
            re-run, and its stale entry in ``runner_state.json`` /
            ``summary.csv`` is replaced (not duplicated) by the new result.
        representative_metric: Metric key used to select the representative
            run (closest to mean) and the best run (highest value).  When
            absent, the first val metric found alphabetically is used.
            Example: ``"val/JaccardIndex"``.

    Example YAML:

    .. code-block:: yaml

        experiments_runner:
          seeds: [42, 101, 28]
          output_base_dir: outputs/reproducibility_study
          save_summary: true
          resume: true
          overwrite: [1]        # redo only run_idx 1 (e.g. seed 101 crashed)
          summary_metrics:
            - val/loss
            - val/F1Score
    """

    n_runs: Optional[int] = None
    seeds: Optional[List[int]] = None
    output_base_dir: str = "outputs/experiments_runner"
    save_summary: bool = True
    summary_metrics: List[str] = field(default_factory=lambda: ["val/loss"])
    resume: bool = True
    overwrite: Optional[Any] = None
    kfold: Optional[Any] = None
    representative_metric: Optional[str] = None
    optuna_search: Optional[Any] = None


def _register_configs() -> None:
    cs = ConfigStore.instance()
    cs.store(
        name="experiments_runner_config",
        node=ExperimentsRunnerConfig,
        group="experiments_runner",
    )


_register_configs()
