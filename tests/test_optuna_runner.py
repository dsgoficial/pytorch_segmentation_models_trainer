# -*- coding: utf-8 -*-
"""Tests for tools/experiments_runner/optuna_runner.py."""

import json
import os
import secrets
import tempfile
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, call, patch

import optuna
import pytest
from omegaconf import DictConfig, OmegaConf

from pytorch_segmentation_models_trainer.tools.experiments_runner.experiments_runner import (
    ExperimentsRunner,
    RunResult,
)
from pytorch_segmentation_models_trainer.tools.experiments_runner.optuna_runner import (
    OptunaRunner,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_runner_cfg(
    output_base_dir: str,
    optuna_cfg: Optional[dict] = None,
    seeds: Optional[List[int]] = None,
    n_runs: Optional[int] = None,
) -> DictConfig:
    runner_block: dict = {"output_base_dir": output_base_dir}
    if seeds is not None:
        runner_block["seeds"] = seeds
    if n_runs is not None:
        runner_block["n_runs"] = n_runs
    # When optuna_search is absent, validation requires seeds or n_runs.
    if optuna_cfg is None and seeds is None and n_runs is None:
        runner_block["n_runs"] = 1
    if optuna_cfg is not None:
        runner_block["optuna_search"] = optuna_cfg
    return OmegaConf.create({"experiments_runner": runner_block, "seed": 42})


def _make_optuna_cfg(
    n_trials: int = 2,
    metric: str = "val/JaccardIndex",
    direction: str = "maximize",
    sampler: str = "TPE",
    storage: Optional[str] = None,
    study_name: str = "test_study",
    search_space: Optional[list] = None,
    save_visualizations: bool = False,
    save_param_importances: bool = True,
) -> dict:
    return {
        "n_trials": n_trials,
        "metric": metric,
        "direction": direction,
        "sampler": sampler,
        "storage": storage,
        "study_name": study_name,
        "search_space": search_space or [],
        "save_visualizations": save_visualizations,
        "save_param_importances": save_param_importances,
    }


def _make_run_result(val: float = 0.85, run_idx: int = 0) -> RunResult:
    return RunResult(
        run_idx=run_idx,
        seed=42,
        training_time_seconds=1.0,
        train_metrics={"train/loss": 0.1},
        val_metrics={"val/JaccardIndex": val},
        test_metrics={},
        output_dir="/tmp/run",
        epochs_trained=5,
        best_checkpoint_path="",
    )


def _make_er(cfg: DictConfig) -> ExperimentsRunner:
    return ExperimentsRunner(cfg)


# ---------------------------------------------------------------------------
# _build_sampler
# ---------------------------------------------------------------------------


class TestBuildSampler:
    def _runner(self, tmpdir):
        cfg = _make_runner_cfg(str(tmpdir), _make_optuna_cfg())
        return OptunaRunner(_make_er(cfg))

    def test_tpe(self, tmp_path):
        runner = self._runner(tmp_path)
        sampler = runner._build_sampler("TPE")
        assert isinstance(sampler, optuna.samplers.TPESampler)

    def test_gp(self, tmp_path):
        runner = self._runner(tmp_path)
        sampler = runner._build_sampler("GP")
        assert isinstance(sampler, optuna.samplers.GPSampler)

    def test_cmaes(self, tmp_path):
        runner = self._runner(tmp_path)
        sampler = runner._build_sampler("CmaES")
        assert isinstance(sampler, optuna.samplers.CmaEsSampler)

    def test_random(self, tmp_path):
        runner = self._runner(tmp_path)
        sampler = runner._build_sampler("Random")
        assert isinstance(sampler, optuna.samplers.RandomSampler)

    def test_grid_raises_without_search_space(self, tmp_path):
        cfg = _make_runner_cfg(
            str(tmp_path),
            _make_optuna_cfg(sampler="Grid", search_space=[]),
        )
        runner = OptunaRunner(_make_er(cfg))
        with pytest.raises(ValueError, match="Grid sampler requires"):
            runner._build_sampler("Grid")

    def test_invalid_name_raises(self, tmp_path):
        runner = self._runner(tmp_path)
        with pytest.raises(ValueError, match="Unknown sampler"):
            runner._build_sampler("Unknown")


# ---------------------------------------------------------------------------
# _build_study
# ---------------------------------------------------------------------------


class TestBuildStudy:
    def test_in_memory_study(self, tmp_path):
        cfg = _make_runner_cfg(str(tmp_path), _make_optuna_cfg(storage=None))
        runner = OptunaRunner(_make_er(cfg))
        study = runner._build_study()
        assert isinstance(study, optuna.Study)
        assert study.direction == optuna.study.StudyDirection.MAXIMIZE

    def test_minimize_direction(self, tmp_path):
        cfg = _make_runner_cfg(str(tmp_path), _make_optuna_cfg(direction="minimize"))
        runner = OptunaRunner(_make_er(cfg))
        study = runner._build_study()
        assert study.direction == optuna.study.StudyDirection.MINIMIZE

    def test_sqlite_storage(self, tmp_path):
        db = str(tmp_path / "study.db")
        cfg = _make_runner_cfg(
            str(tmp_path),
            _make_optuna_cfg(storage=f"sqlite:///{db}", study_name="test"),
        )
        runner = OptunaRunner(_make_er(cfg))
        study = runner._build_study()
        assert isinstance(study, optuna.Study)

    def test_load_if_exists_resumes(self, tmp_path):
        db = str(tmp_path / "study.db")
        storage_url = f"sqlite:///{db}"
        cfg = _make_runner_cfg(
            str(tmp_path),
            _make_optuna_cfg(storage=storage_url, study_name="resumable", n_trials=1),
        )
        runner = OptunaRunner(_make_er(cfg))
        study1 = runner._build_study()
        study1.add_trial(
            optuna.trial.create_trial(
                params={},
                distributions={},
                value=0.9,
            )
        )
        study2 = runner._build_study()
        assert len(study2.trials) == 1


# ---------------------------------------------------------------------------
# _suggest_param
# ---------------------------------------------------------------------------


class TestSuggestParam:
    def _runner(self, tmp_path):
        cfg = _make_runner_cfg(str(tmp_path), _make_optuna_cfg())
        return OptunaRunner(_make_er(cfg))

    def _make_trial(self):
        study = optuna.create_study()
        return study.ask()

    def test_float_basic(self, tmp_path):
        runner = self._runner(tmp_path)
        trial = self._make_trial()
        param = OmegaConf.create(
            {
                "key": "optimizer.lr",
                "type": "float",
                "low": 1e-5,
                "high": 1e-2,
                "log": False,
            }
        )
        val = runner._suggest_param(trial, param)
        assert 1e-5 <= val <= 1e-2

    def test_float_log_scale(self, tmp_path):
        runner = self._runner(tmp_path)
        trial = self._make_trial()
        param = OmegaConf.create(
            {
                "key": "optimizer.lr",
                "type": "float",
                "low": 1e-5,
                "high": 1e-2,
                "log": True,
            }
        )
        val = runner._suggest_param(trial, param)
        assert 1e-5 <= val <= 1e-2

    def test_int_basic(self, tmp_path):
        runner = self._runner(tmp_path)
        trial = self._make_trial()
        param = OmegaConf.create(
            {"key": "batch_size", "type": "int", "low": 8, "high": 64}
        )
        val = runner._suggest_param(trial, param)
        assert isinstance(val, int)
        assert 8 <= val <= 64

    def test_int_with_step(self, tmp_path):
        runner = self._runner(tmp_path)
        trial = self._make_trial()
        param = OmegaConf.create(
            {"key": "batch_size", "type": "int", "low": 8, "high": 64, "step": 8}
        )
        val = runner._suggest_param(trial, param)
        assert val % 8 == 0

    def test_categorical(self, tmp_path):
        runner = self._runner(tmp_path)
        trial = self._make_trial()
        param = OmegaConf.create(
            {
                "key": "model.encoder_name",
                "type": "categorical",
                "choices": ["resnet34", "resnet50", "efficientnet-b0"],
            }
        )
        val = runner._suggest_param(trial, param)
        assert val in ["resnet34", "resnet50", "efficientnet-b0"]

    def test_config_block_returns_name(self, tmp_path):
        runner = self._runner(tmp_path)
        trial = self._make_trial()
        param = OmegaConf.create(
            {
                "key": "loss",
                "type": "config_block",
                "choices": [
                    {"name": "ce", "values": {"_target_": "torch.nn.CrossEntropyLoss"}},
                    {
                        "name": "focal",
                        "values": {"_target_": "kornia.losses.FocalLoss", "alpha": 0.5},
                    },
                ],
            }
        )
        val = runner._suggest_param(trial, param)
        assert val in ["ce", "focal"]
        assert isinstance(val, str)

    def test_unknown_type_raises(self, tmp_path):
        runner = self._runner(tmp_path)
        trial = self._make_trial()
        param = OmegaConf.create({"key": "x", "type": "unknown"})
        with pytest.raises(ValueError, match="Unknown search param type"):
            runner._suggest_param(trial, param)


# ---------------------------------------------------------------------------
# _apply_trial_overrides
# ---------------------------------------------------------------------------


class TestApplyTrialOverrides:
    def _runner(self, tmp_path, search_space):
        cfg = _make_runner_cfg(
            str(tmp_path), _make_optuna_cfg(search_space=search_space)
        )
        return OptunaRunner(_make_er(cfg))

    def test_float_override(self, tmp_path):
        sp = [{"key": "optimizer.lr", "type": "float", "low": 1e-5, "high": 1e-2}]
        runner = self._runner(tmp_path, sp)
        base_cfg = OmegaConf.create({"optimizer": {"lr": 0.01, "weight_decay": 0.0}})
        study = optuna.create_study()
        trial = study.ask()
        result = runner._apply_trial_overrides(trial, base_cfg)
        assert "optimizer" in result
        assert 1e-5 <= result.optimizer.lr <= 1e-2

    def test_int_override(self, tmp_path):
        sp = [{"key": "train_dataset.batch_size", "type": "int", "low": 8, "high": 64}]
        runner = self._runner(tmp_path, sp)
        base_cfg = OmegaConf.create({"train_dataset": {"batch_size": 16}})
        study = optuna.create_study()
        trial = study.ask()
        result = runner._apply_trial_overrides(trial, base_cfg)
        assert 8 <= result.train_dataset.batch_size <= 64

    def test_categorical_override(self, tmp_path):
        sp = [
            {
                "key": "model.encoder_name",
                "type": "categorical",
                "choices": ["resnet34", "resnet50"],
            }
        ]
        runner = self._runner(tmp_path, sp)
        base_cfg = OmegaConf.create({"model": {"encoder_name": "resnet18"}})
        study = optuna.create_study()
        trial = study.ask()
        result = runner._apply_trial_overrides(trial, base_cfg)
        assert result.model.encoder_name in ["resnet34", "resnet50"]

    def test_config_block_merges_entire_subtree(self, tmp_path):
        sp = [
            {
                "key": "loss",
                "type": "config_block",
                "choices": [
                    {
                        "name": "ce",
                        "values": {"_target_": "torch.nn.CrossEntropyLoss"},
                    },
                    {
                        "name": "focal",
                        "values": {
                            "_target_": "kornia.losses.FocalLoss",
                            "alpha": 0.5,
                            "gamma": 2.0,
                        },
                    },
                ],
            }
        ]
        runner = self._runner(tmp_path, sp)
        base_cfg = OmegaConf.create({"loss": {"_target_": "torch.nn.CrossEntropyLoss"}})
        study = optuna.create_study()
        trial = study.ask()
        result = runner._apply_trial_overrides(trial, base_cfg)
        chosen = OmegaConf.to_container(result.loss)
        assert "_target_" in chosen
        assert chosen["_target_"] in [
            "torch.nn.CrossEntropyLoss",
            "kornia.losses.FocalLoss",
        ]

    def test_config_block_focal_has_alpha(self, tmp_path):
        sp = [
            {
                "key": "loss",
                "type": "config_block",
                "choices": [
                    {
                        "name": "focal",
                        "values": {"_target_": "kornia.losses.FocalLoss", "alpha": 0.5},
                    },
                ],
            }
        ]
        runner = self._runner(tmp_path, sp)
        base_cfg = OmegaConf.create({"loss": {}})
        study = optuna.create_study()
        trial = study.ask()
        result = runner._apply_trial_overrides(trial, base_cfg)
        assert result.loss._target_ == "kornia.losses.FocalLoss"
        assert result.loss.alpha == 0.5

    def test_nested_dotpath_override(self, tmp_path):
        sp = [{"key": "pl_trainer.max_epochs", "type": "int", "low": 10, "high": 50}]
        runner = self._runner(tmp_path, sp)
        base_cfg = OmegaConf.create({"pl_trainer": {"max_epochs": 20}})
        study = optuna.create_study()
        trial = study.ask()
        result = runner._apply_trial_overrides(trial, base_cfg)
        assert 10 <= result.pl_trainer.max_epochs <= 50

    def test_original_cfg_not_mutated(self, tmp_path):
        sp = [{"key": "optimizer.lr", "type": "float", "low": 0.001, "high": 0.01}]
        runner = self._runner(tmp_path, sp)
        base_cfg = OmegaConf.create({"optimizer": {"lr": 0.005}})
        original_lr = base_cfg.optimizer.lr
        study = optuna.create_study()
        trial = study.ask()
        runner._apply_trial_overrides(trial, base_cfg)
        assert base_cfg.optimizer.lr == original_lr


# ---------------------------------------------------------------------------
# _objective
# ---------------------------------------------------------------------------


class TestObjective:
    def _runner_with_mock(self, tmp_path, val_result=0.85):
        cfg = _make_runner_cfg(
            str(tmp_path),
            _make_optuna_cfg(
                search_space=[
                    {"key": "optimizer.lr", "type": "float", "low": 1e-5, "high": 1e-2}
                ]
            ),
        )
        er = _make_er(cfg)
        runner = OptunaRunner(er)
        runner.er._run_single = MagicMock(return_value=_make_run_result(val=val_result))
        return runner

    def test_calls_run_single(self, tmp_path):
        runner = self._runner_with_mock(tmp_path)
        study = optuna.create_study(direction="maximize")
        trial = study.ask()
        runner._objective(trial)
        runner.er._run_single.assert_called_once()

    def test_run_single_called_with_prebuilt_cfg(self, tmp_path):
        runner = self._runner_with_mock(tmp_path)
        study = optuna.create_study(direction="maximize")
        trial = study.ask()
        runner._objective(trial)
        kwargs = runner.er._run_single.call_args.kwargs
        assert "run_cfg" in kwargs
        assert "output_dir" in kwargs

    def test_returns_metric_value(self, tmp_path):
        runner = self._runner_with_mock(tmp_path, val_result=0.75)
        study = optuna.create_study(direction="maximize")
        trial = study.ask()
        value = runner._objective(trial)
        assert value == pytest.approx(0.75)

    def test_metric_missing_raises_value_error(self, tmp_path):
        cfg = _make_runner_cfg(
            str(tmp_path),
            _make_optuna_cfg(metric="val/NonExistentMetric"),
        )
        er = _make_er(cfg)
        runner = OptunaRunner(er)
        runner.er._run_single = MagicMock(return_value=_make_run_result())
        study = optuna.create_study()
        trial = study.ask()
        with pytest.raises(ValueError, match="val/NonExistentMetric"):
            runner._objective(trial)

    def test_output_dir_uses_trial_number(self, tmp_path):
        runner = self._runner_with_mock(tmp_path)
        study = optuna.create_study(direction="maximize")
        trial = study.ask()
        runner._objective(trial)
        kwargs = runner.er._run_single.call_args.kwargs
        assert f"trial_{trial.number:03d}" in kwargs["output_dir"]

    def test_trial_summary_csv_written(self, tmp_path):
        runner = self._runner_with_mock(tmp_path)
        study = optuna.create_study(direction="maximize")
        trial = study.ask()
        study.tell(trial, runner._objective(trial))
        summary_path = os.path.join(str(tmp_path), "trial_summary.csv")
        assert os.path.exists(summary_path)


# ---------------------------------------------------------------------------
# _save_best_config
# ---------------------------------------------------------------------------


class TestSaveBestConfig:
    def test_saves_yaml(self, tmp_path):
        cfg = _make_runner_cfg(
            str(tmp_path),
            _make_optuna_cfg(
                search_space=[
                    {"key": "optimizer.lr", "type": "float", "low": 1e-5, "high": 1e-2}
                ]
            ),
        )
        er = _make_er(cfg)
        runner = OptunaRunner(er)
        runner.er._run_single = MagicMock(return_value=_make_run_result())
        study = runner._build_study()
        study.optimize(runner._objective, n_trials=1)
        runner._save_best_config(study)
        best_cfg_path = os.path.join(str(tmp_path), "best_trial_config.yaml")
        assert os.path.exists(best_cfg_path)

    def test_best_config_contains_override(self, tmp_path):
        cfg = _make_runner_cfg(
            str(tmp_path),
            _make_optuna_cfg(
                search_space=[
                    {"key": "optimizer.lr", "type": "float", "low": 1e-5, "high": 1e-2}
                ]
            ),
        )
        er = _make_er(cfg)
        runner = OptunaRunner(er)
        runner.er._run_single = MagicMock(return_value=_make_run_result())
        study = runner._build_study()
        study.optimize(runner._objective, n_trials=1)
        runner._save_best_config(study)
        best_cfg_path = os.path.join(str(tmp_path), "best_trial_config.yaml")
        loaded = OmegaConf.load(best_cfg_path)
        assert "optimizer" in loaded
        assert "lr" in loaded.optimizer


# ---------------------------------------------------------------------------
# _save_trial_summary
# ---------------------------------------------------------------------------


class TestSaveTrialSummary:
    def test_csv_created(self, tmp_path):
        cfg = _make_runner_cfg(str(tmp_path), _make_optuna_cfg())
        er = _make_er(cfg)
        runner = OptunaRunner(er)
        runner.er._run_single = MagicMock(return_value=_make_run_result())
        study = runner._build_study()
        study.optimize(runner._objective, n_trials=1)
        runner._save_trial_summary(study)
        assert os.path.exists(os.path.join(str(tmp_path), "trial_summary.csv"))

    def test_csv_has_expected_columns(self, tmp_path):
        cfg = _make_runner_cfg(
            str(tmp_path),
            _make_optuna_cfg(
                search_space=[
                    {"key": "optimizer.lr", "type": "float", "low": 1e-5, "high": 1e-2}
                ]
            ),
        )
        er = _make_er(cfg)
        runner = OptunaRunner(er)
        runner.er._run_single = MagicMock(return_value=_make_run_result())
        study = runner._build_study()
        study.optimize(runner._objective, n_trials=1)
        runner._save_trial_summary(study)
        import csv

        with open(os.path.join(str(tmp_path), "trial_summary.csv")) as f:
            header = next(csv.reader(f))
        assert "trial_number" in header
        assert "value" in header
        assert "duration_s" in header
        assert "state" in header


# ---------------------------------------------------------------------------
# _save_param_importances
# ---------------------------------------------------------------------------


class TestSaveParamImportances:
    def test_json_created_with_two_trials(self, tmp_path):
        cfg = _make_runner_cfg(
            str(tmp_path),
            _make_optuna_cfg(
                n_trials=2,
                search_space=[
                    {"key": "optimizer.lr", "type": "float", "low": 1e-5, "high": 1e-2}
                ],
            ),
        )
        er = _make_er(cfg)
        runner = OptunaRunner(er)
        # Return varied values so fANOVA has signal to compute importances.
        side_effects = [_make_run_result(val=0.7), _make_run_result(val=0.9)]
        runner.er._run_single = MagicMock(side_effect=side_effects)
        study = runner._build_study()
        study.optimize(runner._objective, n_trials=2)
        runner._save_param_importances(study)
        path = os.path.join(str(tmp_path), "param_importances.json")
        assert os.path.exists(path)
        with open(path) as f:
            data = json.load(f)
        assert isinstance(data, dict)

    def test_skipped_with_single_trial(self, tmp_path):
        cfg = _make_runner_cfg(str(tmp_path), _make_optuna_cfg(n_trials=1))
        er = _make_er(cfg)
        runner = OptunaRunner(er)
        runner.er._run_single = MagicMock(return_value=_make_run_result())
        study = runner._build_study()
        study.optimize(runner._objective, n_trials=1)
        runner._save_param_importances(study)
        path = os.path.join(str(tmp_path), "param_importances.json")
        assert not os.path.exists(path)


# ---------------------------------------------------------------------------
# _save_visualizations
# ---------------------------------------------------------------------------


class TestSaveVisualizations:
    def _run_study(self, tmp_path, n_trials=2, search_space=None):
        sp = search_space or [
            {"key": "optimizer.lr", "type": "float", "low": 1e-5, "high": 1e-2}
        ]
        cfg = _make_runner_cfg(
            str(tmp_path),
            _make_optuna_cfg(
                n_trials=n_trials, search_space=sp, save_visualizations=True
            ),
        )
        er = _make_er(cfg)
        runner = OptunaRunner(er)
        runner.er._run_single = MagicMock(return_value=_make_run_result())
        study = runner._build_study()
        study.optimize(runner._objective, n_trials=n_trials)
        return runner, study

    def test_plots_dir_created(self, tmp_path):
        runner, study = self._run_study(tmp_path)
        runner._save_visualizations(study)
        assert os.path.isdir(os.path.join(str(tmp_path), "plots"))

    def test_optimization_history_html(self, tmp_path):
        runner, study = self._run_study(tmp_path)
        runner._save_visualizations(study)
        assert os.path.exists(
            os.path.join(str(tmp_path), "plots", "optimization_history.html")
        )

    def test_param_importances_html(self, tmp_path):
        # Need varied objective values so fANOVA has non-zero variance to plot.
        sp = [{"key": "optimizer.lr", "type": "float", "low": 1e-5, "high": 1e-2}]
        cfg = _make_runner_cfg(
            str(tmp_path),
            _make_optuna_cfg(n_trials=2, search_space=sp, save_visualizations=True),
        )
        er = _make_er(cfg)
        runner = OptunaRunner(er)
        runner.er._run_single = MagicMock(
            side_effect=[_make_run_result(val=0.7), _make_run_result(val=0.9)]
        )
        study = runner._build_study()
        study.optimize(runner._objective, n_trials=2)
        runner._save_visualizations(study)
        assert os.path.exists(
            os.path.join(str(tmp_path), "plots", "param_importances.html")
        )

    def test_parallel_coordinates_html(self, tmp_path):
        runner, study = self._run_study(tmp_path)
        runner._save_visualizations(study)
        assert os.path.exists(
            os.path.join(str(tmp_path), "plots", "parallel_coordinates.html")
        )

    def test_contour_html(self, tmp_path):
        sp = [
            {"key": "optimizer.lr", "type": "float", "low": 1e-5, "high": 1e-2},
            {"key": "train_dataset.batch_size", "type": "int", "low": 8, "high": 64},
        ]
        runner, study = self._run_study(tmp_path, search_space=sp)
        runner._save_visualizations(study)
        assert os.path.exists(os.path.join(str(tmp_path), "plots", "contour.html"))

    def test_visualizations_skipped_when_flag_false(self, tmp_path):
        cfg = _make_runner_cfg(
            str(tmp_path),
            _make_optuna_cfg(n_trials=2, save_visualizations=False),
        )
        er = _make_er(cfg)
        runner = OptunaRunner(er)
        runner.er._run_single = MagicMock(return_value=_make_run_result())
        study = runner._build_study()
        study.optimize(runner._objective, n_trials=2)
        runner._save_visualizations(study)
        assert not os.path.exists(os.path.join(str(tmp_path), "plots"))


# ---------------------------------------------------------------------------
# run() — full integration
# ---------------------------------------------------------------------------


class TestRunFull:
    def _setup(self, tmp_path, seeds=None, n_runs=None, extra_optuna_cfg=None):
        optuna_cfg = _make_optuna_cfg(
            n_trials=2,
            search_space=[
                {"key": "optimizer.lr", "type": "float", "low": 1e-5, "high": 1e-2}
            ],
            save_visualizations=False,
            save_param_importances=False,
            **(extra_optuna_cfg or {}),
        )
        cfg = _make_runner_cfg(str(tmp_path), optuna_cfg, seeds=seeds, n_runs=n_runs)
        er = _make_er(cfg)
        er._run_single = MagicMock(return_value=_make_run_result())
        runner = OptunaRunner(er)
        return runner

    def test_run_returns_study_and_results(self, tmp_path):
        runner = self._setup(tmp_path)
        study, results = runner.run()
        assert isinstance(study, optuna.Study)
        assert isinstance(results, list)

    def test_run_executes_n_trials(self, tmp_path):
        runner = self._setup(tmp_path)
        runner.run()
        assert runner.er._run_single.call_count == 2

    def test_run_saves_best_config_yaml(self, tmp_path):
        runner = self._setup(tmp_path)
        runner.run()
        assert os.path.exists(os.path.join(str(tmp_path), "best_trial_config.yaml"))

    def test_run_saves_trial_summary_csv(self, tmp_path):
        runner = self._setup(tmp_path)
        runner.run()
        assert os.path.exists(os.path.join(str(tmp_path), "trial_summary.csv"))

    def test_run_mode_b_triggers_seed_loop(self, tmp_path):
        runner = self._setup(tmp_path, seeds=[42, 101])
        with patch.object(runner, "_run_seed_loop_with_best") as mock_seed:
            mock_seed.return_value = []
            runner.run()
            mock_seed.assert_called_once()

    def test_run_no_seed_loop_without_seeds(self, tmp_path):
        cfg = _make_runner_cfg(
            str(tmp_path),
            _make_optuna_cfg(n_trials=2, save_visualizations=False),
        )
        # cfg has n_runs=1 by default from _make_runner_cfg but no seeds
        # Override to have neither seeds nor n_runs for the seed loop
        cfg2 = OmegaConf.create(
            {
                "experiments_runner": {
                    "output_base_dir": str(tmp_path),
                    "optuna_search": _make_optuna_cfg(
                        n_trials=2,
                        save_visualizations=False,
                        save_param_importances=False,
                    ),
                },
                "seed": 42,
            }
        )
        er = _make_er(cfg2)
        er._run_single = MagicMock(return_value=_make_run_result())
        runner = OptunaRunner(er)
        with patch.object(runner, "_run_seed_loop_with_best") as mock_seed:
            runner.run()
            mock_seed.assert_not_called()

    def test_run_mode_b_seed_results_returned(self, tmp_path):
        runner = self._setup(tmp_path, seeds=[42])
        seed_results = [_make_run_result(val=0.9, run_idx=0)]
        with patch.object(
            runner, "_run_seed_loop_with_best", return_value=seed_results
        ):
            _, results = runner.run()
        assert results == seed_results


# ---------------------------------------------------------------------------
# _run_seed_loop_with_best
# ---------------------------------------------------------------------------

_OPTUNA_RUNNER_MODULE = (
    "pytorch_segmentation_models_trainer.tools.experiments_runner.optuna_runner"
)


class TestRunSeedLoopWithBest:
    def test_seed_loop_runs_with_best_overrides(self, tmp_path):
        cfg = _make_runner_cfg(
            str(tmp_path),
            _make_optuna_cfg(
                n_trials=2,
                search_space=[
                    {"key": "optimizer.lr", "type": "float", "low": 1e-5, "high": 1e-2}
                ],
                save_visualizations=False,
                save_param_importances=False,
            ),
            seeds=[42, 101],
        )
        er = _make_er(cfg)
        er._run_single = MagicMock(return_value=_make_run_result())
        runner = OptunaRunner(er)
        study = runner._build_study()
        study.optimize(runner._objective, n_trials=2)

        with patch(f"{_OPTUNA_RUNNER_MODULE}.ExperimentsRunner") as MockER:
            mock_er_instance = MagicMock()
            mock_er_instance.run.return_value = [_make_run_result()]
            MockER.return_value = mock_er_instance
            results = runner._run_seed_loop_with_best(study)

        MockER.assert_called_once()
        mock_er_instance.run.assert_called_once()
        assert isinstance(results, list)

    def test_best_config_applied_to_new_er(self, tmp_path):
        cfg = _make_runner_cfg(
            str(tmp_path),
            _make_optuna_cfg(
                n_trials=2,
                search_space=[
                    {
                        "key": "optimizer.lr",
                        "type": "float",
                        "low": 1e-5,
                        "high": 1e-2,
                    }
                ],
                save_visualizations=False,
                save_param_importances=False,
            ),
            seeds=[42],
        )
        er = _make_er(cfg)
        er._run_single = MagicMock(return_value=_make_run_result())
        runner = OptunaRunner(er)
        study = runner._build_study()
        study.optimize(runner._objective, n_trials=2)

        captured_cfgs = []
        with patch(f"{_OPTUNA_RUNNER_MODULE}.ExperimentsRunner") as MockER:

            def capture_cfg(cfg_arg):
                captured_cfgs.append(cfg_arg)
                m = MagicMock()
                m.run.return_value = []
                return m

            MockER.side_effect = capture_cfg
            runner._run_seed_loop_with_best(study)

        assert len(captured_cfgs) == 1
        passed_cfg = captured_cfgs[0]
        best_params = study.best_trial.params
        for key, val in best_params.items():
            assert OmegaConf.select(passed_cfg, key) is not None


# ---------------------------------------------------------------------------
# ExperimentsRunner.run() dispatch
# ---------------------------------------------------------------------------


class TestExperimentsRunnerDispatch:
    def test_dispatches_to_optuna_runner_when_configured(self, tmp_path):
        cfg = _make_runner_cfg(
            str(tmp_path),
            _make_optuna_cfg(n_trials=1, save_visualizations=False),
        )
        er = ExperimentsRunner(cfg)
        # Patch OptunaRunner at the optuna_runner module level (where it is defined).
        with patch(f"{_OPTUNA_RUNNER_MODULE}.OptunaRunner") as MockOptunaRunner:
            mock_instance = MagicMock()
            mock_instance.run.return_value = (MagicMock(), [])
            MockOptunaRunner.return_value = mock_instance
            er.run()
            MockOptunaRunner.assert_called_once_with(er)
            mock_instance.run.assert_called_once()

    def test_does_not_dispatch_without_optuna_config(self, tmp_path):
        cfg = _make_runner_cfg(str(tmp_path), seeds=[42])
        er = ExperimentsRunner(cfg)
        with patch.object(er, "_run_seed_loop", return_value=[]) as mock_seed:
            er.run()
            mock_seed.assert_called_once()

    def test_kfold_still_dispatches_correctly(self, tmp_path):
        cfg = OmegaConf.create(
            {
                "experiments_runner": {
                    "output_base_dir": str(tmp_path),
                    "seeds": [42],
                    "kfold": {
                        "n_splits": 3,
                        "strategy": "by_image",
                        "input_csv_path": "/tmp/data.csv",
                        "save_fold_csvs_dir": "folds",
                    },
                },
                "seed": 42,
            }
        )
        er = ExperimentsRunner(cfg)
        with patch.object(er, "_run_kfold_loop", return_value=[]) as mock_kfold:
            er.run()
            mock_kfold.assert_called_once()


# ---------------------------------------------------------------------------
# ExperimentsRunner._run_single — prebuilt cfg
# ---------------------------------------------------------------------------


class TestRunSinglePrebuiltCfg:
    def test_accepts_prebuilt_cfg_and_skips_build(self, tmp_path):
        cfg = _make_runner_cfg(str(tmp_path), seeds=[42])
        er = ExperimentsRunner(cfg)

        prebuilt_cfg = OmegaConf.create({"seed": 99, "pl_trainer": {"max_epochs": 1}})
        prebuilt_output = str(tmp_path / "trial_000")
        os.makedirs(prebuilt_output)

        with patch.object(er, "_build_run_cfg") as mock_build:
            with patch(
                "pytorch_segmentation_models_trainer.tools.experiments_runner.experiments_runner.mp"
            ) as mock_mp:
                mock_proc = MagicMock()
                mock_proc.exitcode = 0
                mock_mp.get_context.return_value.Process.return_value = mock_proc

                result_data = {
                    "ok": True,
                    "train_metrics": {},
                    "val_metrics": {"val/loss": 0.1},
                    "test_metrics": {},
                    "epochs_trained": 1,
                    "best_checkpoint_path": "",
                }
                result_path = os.path.join(prebuilt_output, "_seed_result.json")
                with open(result_path, "w") as f:
                    json.dump(result_data, f)

                er._run_single(0, 42, run_cfg=prebuilt_cfg, output_dir=prebuilt_output)
                mock_build.assert_not_called()
