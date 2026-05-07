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
from typing import List, Optional

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
            to ``<output_base_dir>/run_<idx:02d>/``.
        save_summary: When ``True`` (default) a ``summary.csv`` with per-run
            metrics and mean ± std aggregation is written to
            ``output_base_dir`` after all runs finish.
        summary_metrics: Metric keys (as logged by Lightning, e.g.
            ``"val/loss"``) to highlight in log output.  All available
            metrics are always written to the CSV regardless of this list.

    Example YAML:

    .. code-block:: yaml

        experiments_runner:
          seeds: [42, 101, 28]
          output_base_dir: outputs/reproducibility_study
          save_summary: true
          summary_metrics:
            - val/loss
            - val/F1Score
    """

    n_runs: Optional[int] = None
    seeds: Optional[List[int]] = None
    output_base_dir: str = "outputs/experiments_runner"
    save_summary: bool = True
    summary_metrics: List[str] = field(default_factory=lambda: ["val/loss"])


def _register_configs() -> None:
    cs = ConfigStore.instance()
    cs.store(
        name="experiments_runner_config",
        node=ExperimentsRunnerConfig,
        group="experiments_runner",
    )


_register_configs()
