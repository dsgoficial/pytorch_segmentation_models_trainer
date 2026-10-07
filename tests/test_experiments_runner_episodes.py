# -*- coding: utf-8 -*-
"""Tests for the few-shot ``episodes`` axis of the ExperimentsRunner."""

import csv
import json
import os
from unittest.mock import patch

import pandas as pd
import pytest
from omegaconf import OmegaConf

from pytorch_segmentation_models_trainer.tools.experiments_runner.experiments_runner import (
    ExperimentsRunner,
    RunResult,
)

_RUN_SINGLE = (
    "pytorch_segmentation_models_trainer"
    ".tools.experiments_runner.experiments_runner.ExperimentsRunner._run_single"
)


@pytest.fixture
def episodes_csv(tmp_path):
    rows = []
    for draw in range(2):
        for k in (1, 2):
            for w in range(k):
                rows.append(
                    {
                        "mask_path": f"d{draw}_w{w}.tif",
                        "row_off": 0,
                        "col_off": 0,
                        "width": 8,
                        "height": 8,
                        "shots": k,
                        "draw": draw,
                        "novel_class": 5,
                        "novel_fraction": 0.5,
                    }
                )
    # same window selected for two novel classes -> must be deduplicated
    rows.append(dict(rows[-1], novel_class=6))
    path = tmp_path / "episodes.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def _cfg(tmp_path, episodes_csv, **runner_extra):
    runner = {
        "seeds": [42, 7],
        "output_base_dir": str(tmp_path / "out"),
        "save_summary": True,
        "resume": True,
        "episodes": {"csv": str(episodes_csv)},
    }
    runner.update(runner_extra)
    return OmegaConf.create(
        {
            "mode": "run-experiments",
            "experiments_runner": runner,
            "train_dataset": {"window_index_cache": "???"},
            "pl_trainer": {"default_root_dir": "/old"},
        }
    )


def _fake_run_single(calls, metric=lambda cfg: 0.5):
    def _se(
        run_idx, seed, fold_idx=None, fold_paths=None, run_cfg=None, output_dir=None
    ):
        calls.append(
            {"run_idx": run_idx, "seed": seed, "cfg": run_cfg, "out": output_dir}
        )
        return RunResult(
            run_idx=run_idx,
            seed=seed,
            training_time_seconds=1.0,
            train_metrics={},
            val_metrics={},
            test_metrics={"test/miou": metric(run_cfg)},
            output_dir=output_dir,
        )

    return _se


class TestEpisodesLoop:
    def test_one_run_per_seed_and_episode(self, tmp_path, episodes_csv):
        calls = []
        with patch(_RUN_SINGLE, side_effect=_fake_run_single(calls)):
            results = ExperimentsRunner(_cfg(tmp_path, episodes_csv)).run()
        assert len(results) == 2 * 4  # 2 seeds x (2 shots x 2 draws)
        assert [c["run_idx"] for c in calls] == list(range(8))
        assert [r.shots for r in results[:4]] == [1, 1, 2, 2]
        assert [r.draw for r in results[:4]] == [0, 1, 0, 1]
        assert {c["seed"] for c in calls[:4]} == {42}

    def test_support_csv_and_episode_injected(self, tmp_path, episodes_csv):
        calls = []
        with patch(_RUN_SINGLE, side_effect=_fake_run_single(calls)):
            ExperimentsRunner(_cfg(tmp_path, episodes_csv)).run()
        c = calls[3]  # seed 42, shots 2, draw 1
        cfg = c["cfg"]
        assert cfg.seed == 42
        assert cfg.episode.shots == 2 and cfg.episode.draw == 1
        assert "experiments_runner" not in cfg
        assert c["out"].endswith("ep_k02_d01_seed42")
        assert cfg.pl_trainer.default_root_dir == c["out"]
        support = pd.read_csv(cfg.train_dataset.window_index_cache)
        assert list(support.columns) == [
            "mask_path",
            "row_off",
            "col_off",
            "width",
            "height",
        ]
        assert sorted(support.mask_path) == ["d1_w0.tif", "d1_w1.tif"]  # deduplicated

    def test_custom_support_csv_key_and_filters(self, tmp_path, episodes_csv):
        calls = []
        cfg = _cfg(tmp_path, episodes_csv)
        cfg.experiments_runner.episodes = {
            "csv": str(episodes_csv),
            "shots": [2],
            "draws": [1],
            "support_csv_key": "support.csv_path",
        }
        with patch(_RUN_SINGLE, side_effect=_fake_run_single(calls)):
            results = ExperimentsRunner(cfg).run()
        assert len(results) == 2
        assert os.path.exists(calls[0]["cfg"].support.csv_path)

    def test_missing_episode_raises(self, tmp_path, episodes_csv):
        cfg = _cfg(tmp_path, episodes_csv)
        cfg.experiments_runner.episodes.shots = [3]
        with pytest.raises(ValueError, match="no episode"):
            ExperimentsRunner(cfg).run()

    def test_resume_skips_completed_and_keeps_episode_fields(
        self, tmp_path, episodes_csv
    ):
        calls = []
        with patch(_RUN_SINGLE, side_effect=_fake_run_single(calls)):
            ExperimentsRunner(_cfg(tmp_path, episodes_csv)).run()
        state = json.loads((tmp_path / "out" / "runner_state.json").read_text())
        assert state["completed_runs"][3]["shots"] == 2
        calls2 = []
        with patch(_RUN_SINGLE, side_effect=_fake_run_single(calls2)):
            results = ExperimentsRunner(_cfg(tmp_path, episodes_csv)).run()
        assert calls2 == [] and len(results) == 8
        assert results[3].shots == 2 and results[3].draw == 1

    def test_overwrite_forces_one_run(self, tmp_path, episodes_csv):
        calls = []
        with patch(_RUN_SINGLE, side_effect=_fake_run_single(calls)):
            ExperimentsRunner(_cfg(tmp_path, episodes_csv)).run()
        calls2 = []
        with patch(_RUN_SINGLE, side_effect=_fake_run_single(calls2)):
            results = ExperimentsRunner(
                _cfg(tmp_path, episodes_csv, overwrite=[5])
            ).run()
        assert [c["run_idx"] for c in calls2] == [5]
        assert len(results) == 8

    def test_kfold_and_episodes_are_exclusive(self, tmp_path, episodes_csv):
        cfg = _cfg(tmp_path, episodes_csv, kfold={"n_splits": 2})
        with pytest.raises(ValueError, match="kfold.*episodes"):
            ExperimentsRunner(cfg)


class TestSummaryGroupBy:
    def _read(self, tmp_path):
        with open(tmp_path / "out" / "summary.csv") as f:
            return list(csv.DictReader(f))

    def test_summary_has_episode_columns(self, tmp_path, episodes_csv):
        with patch(_RUN_SINGLE, side_effect=_fake_run_single([])):
            ExperimentsRunner(_cfg(tmp_path, episodes_csv)).run()
        rows = self._read(tmp_path)
        assert rows[0]["shots"] == "1" and rows[0]["draw"] == "0"
        assert [r["run"] for r in rows[-2:]] == ["mean", "std"]

    def test_group_by_shots_adds_mean_std_per_group(self, tmp_path, episodes_csv):
        metric = lambda cfg: float(cfg.episode.shots)  # noqa: E731
        with patch(_RUN_SINGLE, side_effect=_fake_run_single([], metric)):
            ExperimentsRunner(
                _cfg(tmp_path, episodes_csv, summary_group_by=["shots"])
            ).run()
        rows = {r["run"]: r for r in self._read(tmp_path)}
        assert float(rows["mean[shots=1]"]["test/miou"]) == 1.0
        assert float(rows["mean[shots=2]"]["test/miou"]) == 2.0
        assert float(rows["std[shots=2]"]["test/miou"]) == 0.0
        assert float(rows["mean"]["test/miou"]) == 1.5

    def test_group_by_unknown_column_raises(self, tmp_path, episodes_csv):
        with patch(_RUN_SINGLE, side_effect=_fake_run_single([])):
            with pytest.raises(ValueError, match="summary_group_by"):
                ExperimentsRunner(
                    _cfg(tmp_path, episodes_csv, summary_group_by=["bioma"])
                ).run()

    def test_plain_seed_loop_summary_unchanged(self, tmp_path):
        cfg = OmegaConf.create(
            {
                "experiments_runner": {
                    "seeds": [1],
                    "output_base_dir": str(tmp_path / "out"),
                    "save_summary": True,
                },
            }
        )
        with patch(_RUN_SINGLE, side_effect=_fake_run_single([])):
            ExperimentsRunner(cfg).run()
        assert "shots" not in self._read(tmp_path)[0]


def test_config_dataclasses_accept_episodes_and_group_by():
    from pytorch_segmentation_models_trainer.config_definitions.experiments_runner_config import (
        ExperimentsEpisodesConfig,
        ExperimentsRunnerConfig,
    )

    cfg = OmegaConf.merge(
        OmegaConf.structured(ExperimentsRunnerConfig),
        {"seeds": [1], "episodes": {"csv": "e.csv"}, "summary_group_by": ["shots"]},
    )
    assert cfg.episodes.support_csv_key == "train_dataset.window_index_cache"
    assert cfg.episodes.shots is None
    assert ExperimentsEpisodesConfig(csv="x").draws is None
