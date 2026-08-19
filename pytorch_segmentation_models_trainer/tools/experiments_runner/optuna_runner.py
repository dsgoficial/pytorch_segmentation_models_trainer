# -*- coding: utf-8 -*-
"""
/***************************************************************************
 pytorch_segmentation_models_trainer
                              -------------------
        begin                : 2026-08-13
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
import json
import logging
import os
import secrets
from typing import Any, Dict, List, Tuple

import optuna
from omegaconf import DictConfig, OmegaConf

from pytorch_segmentation_models_trainer.tools.experiments_runner.experiments_runner import (
    ExperimentsRunner,
)

logger = logging.getLogger(__name__)

_VALID_SAMPLERS = ("TPE", "GP", "CmaES", "Random", "Grid")


class OptunaRunner:
    """Runs Optuna hyperparameter search over an ExperimentsRunner configuration.

    Each Optuna trial builds a training config by applying the suggested
    hyperparameter values on top of the base config, then delegates to
    :meth:`ExperimentsRunner._run_single` (isolated subprocess).

    After the study finishes, the best trial config is saved as YAML and,
    when ``seeds`` / ``n_runs`` are present in the experiments_runner block,
    a standard seed loop is executed with that config (Mode B).

    Args:
        experiments_runner: A fully initialised :class:`ExperimentsRunner`
            whose ``runner_cfg`` contains an ``optuna_search`` block.

    Example::

        runner = ExperimentsRunner(cfg)  # cfg has experiments_runner.optuna_search
        study, seed_results = OptunaRunner(runner).run()
    """

    def __init__(self, experiments_runner: ExperimentsRunner) -> None:
        self.er: ExperimentsRunner = experiments_runner
        self.optuna_cfg: DictConfig = experiments_runner.runner_cfg.optuna_search
        self.output_dir: str = experiments_runner.runner_cfg.output_base_dir

    # ------------------------------------------------------------------
    # Study construction
    # ------------------------------------------------------------------

    def _build_sampler(self, name: str) -> optuna.samplers.BaseSampler:
        """Instantiate the Optuna sampler by name.

        Args:
            name: One of ``"TPE"``, ``"GP"``, ``"CmaES"``, ``"Random"``,
                ``"Grid"``.

        Returns:
            Configured sampler instance.

        Raises:
            ValueError: For unknown sampler names or ``"Grid"`` without a
                discrete search space.
        """
        if name not in _VALID_SAMPLERS:
            raise ValueError(
                f"Unknown sampler '{name}'. Valid options: {_VALID_SAMPLERS}"
            )
        if name == "TPE":
            return optuna.samplers.TPESampler()
        if name == "GP":
            return optuna.samplers.GPSampler()
        if name == "CmaES":
            return optuna.samplers.CmaEsSampler()
        if name == "Random":
            return optuna.samplers.RandomSampler()
        if name == "Grid":
            search_space = self._build_grid_search_space()
            if not search_space:
                raise ValueError(
                    "Grid sampler requires at least one categorical or int/float "
                    "search param with explicit choices."
                )
            return optuna.samplers.GridSampler(search_space)
        raise ValueError(f"Unknown sampler '{name}'.")

    def _build_grid_search_space(self) -> Dict[str, List[Any]]:
        """Build the discrete search space dict required by GridSampler."""
        grid: Dict[str, List[Any]] = {}
        for param in self.optuna_cfg.get("search_space", []):
            ptype = param["type"]
            key = param["key"]
            if ptype == "categorical":
                grid[key] = list(param["choices"])
            elif ptype == "config_block":
                grid[key] = [c["name"] for c in param["choices"]]
        return grid

    def _build_study(self) -> optuna.Study:
        """Create (or resume) an Optuna study.

        Uses ``load_if_exists=True`` so that pointing to an existing SQLite
        file automatically resumes the study.

        Returns:
            Optuna :class:`~optuna.Study` instance.
        """
        sampler_name = OmegaConf.select(self.optuna_cfg, "sampler", default="TPE")
        sampler = self._build_sampler(sampler_name)
        storage = OmegaConf.select(self.optuna_cfg, "storage", default=None)
        study_name = OmegaConf.select(
            self.optuna_cfg, "study_name", default="optuna_study"
        )
        direction = OmegaConf.select(self.optuna_cfg, "direction", default="maximize")
        return optuna.create_study(
            direction=direction,
            sampler=sampler,
            storage=storage,
            study_name=study_name,
            load_if_exists=True,
        )

    # ------------------------------------------------------------------
    # Search space suggestion
    # ------------------------------------------------------------------

    def _suggest_param(self, trial: optuna.Trial, param: Any) -> Any:
        """Suggest a value for one search parameter using the Optuna trial.

        Args:
            trial: Active Optuna trial.
            param: Search param config (DictConfig or dict with ``key``,
                ``type``, and type-specific fields).

        Returns:
            Suggested value. For ``config_block``, returns the block *name*
            (a string), not the block itself.

        Raises:
            ValueError: For unknown ``type`` values.
        """
        ptype = param["type"]
        key = param["key"]

        if ptype == "float":
            return trial.suggest_float(
                key,
                float(param["low"]),
                float(param["high"]),
                log=bool(param.get("log", False)),
            )
        if ptype == "int":
            step = param.get("step", None)
            kwargs = {}
            if step is not None:
                kwargs["step"] = int(step)
            return trial.suggest_int(
                key, int(param["low"]), int(param["high"]), **kwargs
            )
        if ptype == "categorical":
            return trial.suggest_categorical(key, list(param["choices"]))
        if ptype == "config_block":
            names = [c["name"] for c in param["choices"]]
            return trial.suggest_categorical(key, names)
        raise ValueError(
            f"Unknown search param type '{ptype}'. "
            "Valid types: float, int, categorical, config_block."
        )

    def _apply_trial_overrides(
        self, trial: optuna.Trial, base_cfg: DictConfig
    ) -> DictConfig:
        """Return a copy of ``base_cfg`` with trial suggestions applied.

        Does not mutate ``base_cfg``.

        Args:
            trial: Active Optuna trial.
            base_cfg: Base training config (without ``experiments_runner``).

        Returns:
            New :class:`~omegaconf.DictConfig` with all overrides applied.
        """
        cfg = OmegaConf.create(OmegaConf.to_container(base_cfg, resolve=True))
        OmegaConf.set_struct(cfg, False)

        for param in self.optuna_cfg.get("search_space", []):
            value = self._suggest_param(trial, param)

            if param["type"] == "config_block":
                # Find the matching block and merge its values subtree.
                block_values = next(
                    c["values"] for c in param["choices"] if c["name"] == value
                )
                block_dict = (
                    OmegaConf.to_container(block_values, resolve=True)
                    if isinstance(block_values, DictConfig)
                    else dict(block_values)
                )
                OmegaConf.update(
                    cfg, param["key"], OmegaConf.create(block_dict), merge=True
                )
            else:
                OmegaConf.update(cfg, param["key"], value, merge=True)

        return cfg

    # ------------------------------------------------------------------
    # Objective function
    # ------------------------------------------------------------------

    def _build_base_cfg(self) -> DictConfig:
        """Extract the base training config (strips ``experiments_runner``)."""
        keys = [k for k in self.er.cfg if k != "experiments_runner"]
        base = OmegaConf.masked_copy(self.er.cfg, keys)
        OmegaConf.set_struct(base, False)
        return base

    def _objective(self, trial: optuna.Trial) -> float:
        """Objective function executed for each Optuna trial.

        Applies trial suggestions to the base config, delegates to
        :meth:`ExperimentsRunner._run_single`, reads the target metric, and
        writes an incremental trial summary CSV entry.

        Args:
            trial: Active Optuna trial.

        Returns:
            Float value of the target metric for this trial.

        Raises:
            ValueError: If the target metric is absent from the trial's
                collected metrics.
        """
        run_idx = trial.number
        seed = secrets.randbelow(2**31)

        base_cfg = self._build_base_cfg()
        trial_cfg = self._apply_trial_overrides(trial, base_cfg)
        OmegaConf.update(trial_cfg, "seed", seed, merge=True)

        output_dir = os.path.join(self.output_dir, f"trial_{run_idx:03d}")
        os.makedirs(output_dir, exist_ok=True)

        result = self.er._run_single(
            run_idx=run_idx,
            seed=seed,
            run_cfg=trial_cfg,
            output_dir=output_dir,
        )

        all_metrics = {
            **result.train_metrics,
            **result.val_metrics,
            **result.test_metrics,
        }
        metric = OmegaConf.select(self.optuna_cfg, "metric", default="val/loss")
        if metric not in all_metrics:
            raise ValueError(
                f"Metric '{metric}' not found in trial {run_idx} results. "
                f"Available: {list(all_metrics.keys())}"
            )

        self._append_trial_summary_row(
            trial, all_metrics[metric], result.training_time_seconds
        )
        return all_metrics[metric]

    # ------------------------------------------------------------------
    # Persistence helpers
    # ------------------------------------------------------------------

    def _append_trial_summary_row(
        self, trial: optuna.Trial, value: float, duration_s: float
    ) -> None:
        """Append one row to the incremental trial_summary.csv."""
        os.makedirs(self.output_dir, exist_ok=True)
        path = os.path.join(self.output_dir, "trial_summary.csv")
        params = trial.params

        fieldnames = ["trial_number", "state", "value", "duration_s"] + sorted(
            params.keys()
        )
        row = {
            "trial_number": trial.number,
            "state": "COMPLETE",
            "value": f"{value:.6f}",
            "duration_s": f"{duration_s:.2f}",
            **{k: v for k, v in params.items()},
        }

        write_header = not os.path.exists(path)
        with open(path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            if write_header:
                writer.writeheader()
            writer.writerow(row)

    def _save_trial_summary(self, study: optuna.Study) -> None:
        """Write the final trial_summary.csv from the completed study.

        Overwrites any incremental version with a clean full-study table.

        Args:
            study: Completed Optuna study.
        """
        os.makedirs(self.output_dir, exist_ok=True)
        path = os.path.join(self.output_dir, "trial_summary.csv")

        trials = study.trials
        if not trials:
            return

        all_param_keys = sorted({k for t in trials for k in t.params.keys()})
        fieldnames = ["trial_number", "state", "value", "duration_s"] + all_param_keys

        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            for t in trials:
                duration = (
                    (t.datetime_complete - t.datetime_start).total_seconds()
                    if t.datetime_start and t.datetime_complete
                    else ""
                )
                row: Dict[str, Any] = {
                    "trial_number": t.number,
                    "state": t.state.name,
                    "value": f"{t.value:.6f}" if t.value is not None else "",
                    "duration_s": (
                        f"{duration:.2f}" if isinstance(duration, float) else ""
                    ),
                }
                row.update(t.params)
                writer.writerow(row)

    def _save_best_config(self, study: optuna.Study) -> None:
        """Save the best trial's training config as ``best_trial_config.yaml``.

        Reconstructs the full training config by applying the best trial's
        suggested parameter values to the base config.

        Args:
            study: Completed Optuna study with at least one successful trial.
        """
        os.makedirs(self.output_dir, exist_ok=True)
        best_trial = study.best_trial
        base_cfg = self._build_base_cfg()

        cfg = OmegaConf.create(OmegaConf.to_container(base_cfg, resolve=True))
        OmegaConf.set_struct(cfg, False)

        for param in self.optuna_cfg.get("search_space", []):
            key = param["key"]
            if key not in best_trial.params:
                continue
            value = best_trial.params[key]
            if param["type"] == "config_block":
                block_values = next(
                    c["values"] for c in param["choices"] if c["name"] == value
                )
                block_dict = (
                    OmegaConf.to_container(block_values, resolve=True)
                    if isinstance(block_values, DictConfig)
                    else dict(block_values)
                )
                OmegaConf.update(cfg, key, OmegaConf.create(block_dict), merge=True)
            else:
                OmegaConf.update(cfg, key, value, merge=True)

        out_path = os.path.join(self.output_dir, "best_trial_config.yaml")
        OmegaConf.save(cfg, out_path)
        logger.info("OptunaRunner — best trial config saved to %s", out_path)

    def _save_param_importances(self, study: optuna.Study) -> None:
        """Save fANOVA parameter importance scores as JSON.

        Skipped silently when fewer than 2 trials are completed (fANOVA
        requires at least 2 data points).

        Args:
            study: Completed Optuna study.
        """
        completed = [
            t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE
        ]
        if len(completed) < 2:
            logger.info(
                "OptunaRunner — skipping param importances (need ≥2 completed trials, got %d).",
                len(completed),
            )
            return

        try:
            importances = optuna.importance.get_param_importances(study)
        except Exception as exc:
            logger.warning(
                "OptunaRunner — could not compute param importances: %s", exc
            )
            return

        os.makedirs(self.output_dir, exist_ok=True)
        out_path = os.path.join(self.output_dir, "param_importances.json")
        with open(out_path, "w") as f:
            json.dump(dict(importances), f, indent=2)
        logger.info("OptunaRunner — param importances saved to %s", out_path)

    def _save_visualizations(self, study: optuna.Study) -> None:
        """Save Plotly HTML visualisation files.

        Only runs when ``save_visualizations`` is ``True`` in the config.
        Silently skips individual plots that fail (e.g. not enough trials).

        Output files in ``<output_base_dir>/plots/``:
        - ``optimization_history.html``
        - ``param_importances.html``
        - ``parallel_coordinates.html``
        - ``contour.html``

        Args:
            study: Completed Optuna study.
        """
        if not OmegaConf.select(self.optuna_cfg, "save_visualizations", default=True):
            return

        plots_dir = os.path.join(self.output_dir, "plots")
        os.makedirs(plots_dir, exist_ok=True)

        def _save(fig, name: str) -> None:
            try:
                fig.write_html(os.path.join(plots_dir, name))
            except Exception as exc:
                logger.warning("OptunaRunner — could not save %s: %s", name, exc)

        try:
            _save(
                optuna.visualization.plot_optimization_history(study),
                "optimization_history.html",
            )
        except Exception as exc:
            logger.warning("OptunaRunner — optimization_history plot failed: %s", exc)

        try:
            _save(
                optuna.visualization.plot_param_importances(study),
                "param_importances.html",
            )
        except Exception as exc:
            logger.warning("OptunaRunner — param_importances plot failed: %s", exc)

        try:
            _save(
                optuna.visualization.plot_parallel_coordinate(study),
                "parallel_coordinates.html",
            )
        except Exception as exc:
            logger.warning("OptunaRunner — parallel_coordinates plot failed: %s", exc)

        try:
            _save(
                optuna.visualization.plot_contour(study),
                "contour.html",
            )
        except Exception as exc:
            logger.warning("OptunaRunner — contour plot failed: %s", exc)

    # ------------------------------------------------------------------
    # Mode B — seed loop with best config
    # ------------------------------------------------------------------

    def _run_seed_loop_with_best(self, study: optuna.Study) -> List:
        """Run the standard seed loop using the best trial's config.

        Builds the best-config DictConfig, re-attaches the
        ``experiments_runner`` block (without ``optuna_search``), and
        delegates to a fresh :class:`ExperimentsRunner`.

        Args:
            study: Completed Optuna study.

        Returns:
            List of :class:`~experiments_runner.RunResult` from the seed loop.
        """
        best_trial = study.best_trial
        base_cfg = self._build_base_cfg()

        best_cfg = OmegaConf.create(OmegaConf.to_container(base_cfg, resolve=True))
        OmegaConf.set_struct(best_cfg, False)

        for param in self.optuna_cfg.get("search_space", []):
            key = param["key"]
            if key not in best_trial.params:
                continue
            value = best_trial.params[key]
            if param["type"] == "config_block":
                block_values = next(
                    c["values"] for c in param["choices"] if c["name"] == value
                )
                block_dict = (
                    OmegaConf.to_container(block_values, resolve=True)
                    if isinstance(block_values, DictConfig)
                    else dict(block_values)
                )
                OmegaConf.update(
                    best_cfg, key, OmegaConf.create(block_dict), merge=True
                )
            else:
                OmegaConf.update(best_cfg, key, value, merge=True)

        # Re-attach experiments_runner block (seeds/n_runs, no optuna_search).
        runner_keys = [k for k in self.er.runner_cfg if k != "optuna_search"]
        runner_block = OmegaConf.masked_copy(self.er.runner_cfg, runner_keys)
        seed_cfg = OmegaConf.merge(best_cfg, {"experiments_runner": runner_block})
        OmegaConf.set_struct(seed_cfg, False)

        seed_runner = ExperimentsRunner(seed_cfg)
        return seed_runner.run()

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def run(self) -> Tuple[optuna.Study, List]:
        """Execute the full Optuna HPO study.

        Steps:
        1. Build / resume the study.
        2. Run ``n_trials`` trials via :meth:`_objective`.
        3. Save best config YAML, trial summary CSV, param importances JSON.
        4. Save visualisation HTML files (when enabled).
        5. Run seed loop with best config when ``seeds`` / ``n_runs``
           are configured in the runner block (Mode B).

        Returns:
            Tuple of ``(study, seed_results)`` where ``seed_results`` is a
            list of :class:`~experiments_runner.RunResult` from the seed loop
            (empty list when Mode B is not configured).
        """
        n_trials = int(OmegaConf.select(self.optuna_cfg, "n_trials", default=10))
        metric = OmegaConf.select(self.optuna_cfg, "metric", default="val/loss")

        logger.info(
            "OptunaRunner — starting study '%s' | n_trials=%d | metric=%s | dir=%s",
            OmegaConf.select(self.optuna_cfg, "study_name", default="optuna_study"),
            n_trials,
            metric,
            self.output_dir,
        )

        study = self._build_study()
        study.optimize(self._objective, n_trials=n_trials)

        completed = [
            t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE
        ]
        if completed:
            self._save_best_config(study)
            self._save_trial_summary(study)
            if OmegaConf.select(
                self.optuna_cfg, "save_param_importances", default=True
            ):
                self._save_param_importances(study)
            if OmegaConf.select(self.optuna_cfg, "save_visualizations", default=True):
                self._save_visualizations(study)
        else:
            logger.warning("OptunaRunner — no completed trials; skipping output saves.")

        # Mode B: run seed loop with best config.
        has_seeds = self.er.runner_cfg.get("seeds", None) is not None
        has_n_runs = self.er.runner_cfg.get("n_runs", None) is not None
        seed_results: List = []
        if completed and (has_seeds or has_n_runs):
            logger.info("OptunaRunner — starting Mode B seed loop with best config.")
            seed_results = self._run_seed_loop_with_best(study)

        return study, seed_results
