# -*- coding: utf-8 -*-
"""
Tests for ExperimentsRunner (TDD — written before implementation).

Run with:
    uv run pytest tests/test_experiments_runner.py -v --tb=short
"""
import os
import shutil
import tempfile
import unittest
from unittest.mock import MagicMock, patch, call

import pytorch_lightning as pl
from omegaconf import OmegaConf

from tests.utils import BasicTestCase


def _make_runner_cfg(
    seeds=None,
    n_runs=None,
    output_base_dir=None,
    save_summary=False,
    summary_metrics=None,
    extra=None,
):
    """Build a minimal DictConfig that ExperimentsRunner accepts."""
    runner_block = {"output_base_dir": output_base_dir or "/tmp/test_runs"}
    if seeds is not None:
        runner_block["seeds"] = seeds
    if n_runs is not None:
        runner_block["n_runs"] = n_runs
    runner_block["save_summary"] = save_summary
    runner_block["summary_metrics"] = summary_metrics or ["val/loss"]
    if extra:
        runner_block.update(extra)

    cfg = {
        "mode": "run-experiments",
        "experiments_runner": runner_block,
        "seed": 0,
        "hyperparameters": {"batch_size": 1, "epochs": 1, "max_lr": 0.1},
        "pl_trainer": {
            "max_epochs": 1,
            "accelerator": "cpu",
            "devices": 1,
            "default_root_dir": "/old/path",
        },
    }
    return OmegaConf.create(cfg)


def _make_mock_trainer(metrics=None):
    mock_trainer = MagicMock(spec=pl.Trainer)
    mock_trainer.callback_metrics = metrics or {"val/loss": 0.5, "train/loss": 0.3}
    return mock_trainer


class TestExperimentsRunnerValidation(BasicTestCase):
    """Config validation — no seeds and no n_runs must raise; conflicts must raise."""

    def _import(self):
        from pytorch_segmentation_models_trainer.tools.experiments_runner.experiments_runner import (
            ExperimentsRunner,
        )
        return ExperimentsRunner

    def test_raises_when_both_absent(self):
        ExperimentsRunner = self._import()
        cfg = _make_runner_cfg()  # neither seeds nor n_runs
        with self.assertRaises(ValueError):
            ExperimentsRunner(cfg)

    def test_raises_when_seeds_and_n_runs_conflict(self):
        ExperimentsRunner = self._import()
        cfg = _make_runner_cfg(seeds=[42, 101, 28], n_runs=1)
        with self.assertRaises(ValueError):
            ExperimentsRunner(cfg)

    def test_valid_with_only_seeds(self):
        ExperimentsRunner = self._import()
        cfg = _make_runner_cfg(seeds=[42, 101])
        runner = ExperimentsRunner(cfg)
        self.assertIsNotNone(runner)

    def test_valid_with_only_n_runs(self):
        ExperimentsRunner = self._import()
        cfg = _make_runner_cfg(n_runs=3)
        runner = ExperimentsRunner(cfg)
        self.assertIsNotNone(runner)

    def test_valid_when_seeds_and_n_runs_consistent(self):
        ExperimentsRunner = self._import()
        cfg = _make_runner_cfg(seeds=[42, 101, 28], n_runs=3)
        runner = ExperimentsRunner(cfg)
        self.assertIsNotNone(runner)


class TestResolveSeeds(BasicTestCase):
    """_resolve_seeds() must return the right list of seeds."""

    def _import(self):
        from pytorch_segmentation_models_trainer.tools.experiments_runner.experiments_runner import (
            ExperimentsRunner,
        )
        return ExperimentsRunner

    def test_explicit_seeds_returned_as_is(self):
        ExperimentsRunner = self._import()
        cfg = _make_runner_cfg(seeds=[42, 101, 28])
        seeds = ExperimentsRunner(cfg)._resolve_seeds()
        self.assertEqual(seeds, [42, 101, 28])

    def test_auto_seeds_have_correct_count(self):
        ExperimentsRunner = self._import()
        cfg = _make_runner_cfg(n_runs=5)
        seeds = ExperimentsRunner(cfg)._resolve_seeds()
        self.assertEqual(len(seeds), 5)

    def test_auto_seeds_are_in_valid_range(self):
        ExperimentsRunner = self._import()
        cfg = _make_runner_cfg(n_runs=10)
        seeds = ExperimentsRunner(cfg)._resolve_seeds()
        self.assertTrue(all(0 <= s < 2**31 for s in seeds))

    def test_auto_seeds_are_different_each_call(self):
        ExperimentsRunner = self._import()
        cfg = _make_runner_cfg(n_runs=4)
        runner = ExperimentsRunner(cfg)
        seeds1 = runner._resolve_seeds()
        seeds2 = runner._resolve_seeds()
        # Probability of collision: (2^31)^-4 ≈ 0
        self.assertNotEqual(seeds1, seeds2)

    def test_n_runs_derived_from_seeds_length(self):
        ExperimentsRunner = self._import()
        cfg = _make_runner_cfg(seeds=[1, 2, 3, 4])
        runner = ExperimentsRunner(cfg)
        self.assertEqual(runner._n_runs(), 4)


class TestBuildRunCfg(BasicTestCase):
    """_build_run_cfg() must return a clean config with correct seed and paths."""

    def _import(self):
        from pytorch_segmentation_models_trainer.tools.experiments_runner.experiments_runner import (
            ExperimentsRunner,
        )
        return ExperimentsRunner

    def test_seed_is_set_correctly(self):
        ExperimentsRunner = self._import()
        cfg = _make_runner_cfg(seeds=[42, 101])
        run_cfg, _ = ExperimentsRunner(cfg)._build_run_cfg(0, 42)
        self.assertEqual(run_cfg.seed, 42)

    def test_experiments_runner_block_removed(self):
        ExperimentsRunner = self._import()
        cfg = _make_runner_cfg(seeds=[42])
        run_cfg, _ = ExperimentsRunner(cfg)._build_run_cfg(0, 42)
        self.assertNotIn("experiments_runner", run_cfg)

    def test_mode_preserved(self):
        ExperimentsRunner = self._import()
        cfg = _make_runner_cfg(seeds=[42])
        run_cfg, _ = ExperimentsRunner(cfg)._build_run_cfg(0, 42)
        self.assertIn("mode", run_cfg)

    def test_output_dir_ends_with_correct_run_folder(self):
        ExperimentsRunner = self._import()
        cfg = _make_runner_cfg(seeds=[42, 101], output_base_dir="/base/runs")
        _, output_dir = ExperimentsRunner(cfg)._build_run_cfg(1, 101)
        self.assertTrue(output_dir.endswith("run_01"))

    def test_pl_trainer_default_root_dir_updated(self):
        ExperimentsRunner = self._import()
        cfg = _make_runner_cfg(seeds=[42], output_base_dir="/base/runs")
        run_cfg, output_dir = ExperimentsRunner(cfg)._build_run_cfg(0, 42)
        self.assertEqual(run_cfg.pl_trainer.default_root_dir, output_dir)

    def test_run_idx_zero_padding(self):
        ExperimentsRunner = self._import()
        cfg = _make_runner_cfg(seeds=[42], output_base_dir="/base")
        _, output_dir = ExperimentsRunner(cfg)._build_run_cfg(0, 42)
        self.assertIn("run_00", output_dir)

    def test_second_run_idx_formatted(self):
        ExperimentsRunner = self._import()
        cfg = _make_runner_cfg(seeds=[42, 99], output_base_dir="/base")
        _, output_dir = ExperimentsRunner(cfg)._build_run_cfg(9, 99)
        self.assertIn("run_09", output_dir)


class TestCollectMetrics(BasicTestCase):
    """_collect_metrics() must split trainer.callback_metrics by prefix."""

    def _import(self):
        from pytorch_segmentation_models_trainer.tools.experiments_runner.experiments_runner import (
            ExperimentsRunner,
        )
        return ExperimentsRunner

    def test_val_metrics_separated(self):
        ExperimentsRunner = self._import()
        cfg = _make_runner_cfg(seeds=[42])
        runner = ExperimentsRunner(cfg)
        trainer = _make_mock_trainer({"val/loss": 0.35, "train/loss": 0.2, "test/iou": 0.75})
        _, val, _ = runner._collect_metrics(trainer)
        self.assertIn("val/loss", val)
        self.assertNotIn("train/loss", val)

    def test_train_metrics_separated(self):
        ExperimentsRunner = self._import()
        cfg = _make_runner_cfg(seeds=[42])
        runner = ExperimentsRunner(cfg)
        trainer = _make_mock_trainer({"val/loss": 0.35, "train/loss": 0.2})
        train, _, _ = runner._collect_metrics(trainer)
        self.assertIn("train/loss", train)
        self.assertNotIn("val/loss", train)

    def test_test_metrics_separated(self):
        ExperimentsRunner = self._import()
        cfg = _make_runner_cfg(seeds=[42])
        runner = ExperimentsRunner(cfg)
        trainer = _make_mock_trainer({"test/iou": 0.75, "val/loss": 0.3})
        _, _, test = runner._collect_metrics(trainer)
        self.assertIn("test/iou", test)

    def test_metric_values_are_float(self):
        ExperimentsRunner = self._import()
        cfg = _make_runner_cfg(seeds=[42])
        runner = ExperimentsRunner(cfg)
        import torch
        trainer = _make_mock_trainer({"val/loss": torch.tensor(0.42)})
        _, val, _ = runner._collect_metrics(trainer)
        self.assertIsInstance(val["val/loss"], float)


_TRAIN_PATH = (
    "pytorch_segmentation_models_trainer"
    ".tools.experiments_runner.experiments_runner.train"
)


class TestRunMethod(BasicTestCase):
    """run() integration: correct number of calls, results, timing, CSV."""

    def setUp(self):
        super().setUp()
        self.tmp = self.make_temp_dir()

    def _make_cfg(self, seeds=None, n_runs=None, save_summary=False, metrics=None):
        return _make_runner_cfg(
            seeds=seeds,
            n_runs=n_runs,
            output_base_dir=self.tmp,
            save_summary=save_summary,
        )

    @patch(_TRAIN_PATH)
    def test_train_called_once_per_seed(self, mock_train):
        mock_train.return_value = _make_mock_trainer()
        from pytorch_segmentation_models_trainer.tools.experiments_runner.experiments_runner import ExperimentsRunner
        runner = ExperimentsRunner(self._make_cfg(seeds=[42, 101, 28]))
        runner.run()
        self.assertEqual(mock_train.call_count, 3)

    @patch(_TRAIN_PATH)
    def test_returns_one_result_per_run(self, mock_train):
        mock_train.return_value = _make_mock_trainer()
        from pytorch_segmentation_models_trainer.tools.experiments_runner.experiments_runner import ExperimentsRunner
        results = ExperimentsRunner(self._make_cfg(seeds=[42, 101])).run()
        self.assertEqual(len(results), 2)

    @patch(_TRAIN_PATH)
    def test_result_seeds_match_config(self, mock_train):
        mock_train.return_value = _make_mock_trainer()
        from pytorch_segmentation_models_trainer.tools.experiments_runner.experiments_runner import ExperimentsRunner
        results = ExperimentsRunner(self._make_cfg(seeds=[42, 101])).run()
        self.assertEqual(results[0].seed, 42)
        self.assertEqual(results[1].seed, 101)

    @patch(_TRAIN_PATH)
    def test_result_run_idx_sequential(self, mock_train):
        mock_train.return_value = _make_mock_trainer()
        from pytorch_segmentation_models_trainer.tools.experiments_runner.experiments_runner import ExperimentsRunner
        results = ExperimentsRunner(self._make_cfg(seeds=[10, 20, 30])).run()
        self.assertEqual([r.run_idx for r in results], [0, 1, 2])

    @patch(_TRAIN_PATH)
    def test_training_time_non_negative(self, mock_train):
        mock_train.return_value = _make_mock_trainer()
        from pytorch_segmentation_models_trainer.tools.experiments_runner.experiments_runner import ExperimentsRunner
        results = ExperimentsRunner(self._make_cfg(seeds=[42])).run()
        self.assertGreaterEqual(results[0].training_time_seconds, 0.0)

    @patch(_TRAIN_PATH)
    def test_val_metrics_in_result(self, mock_train):
        mock_train.return_value = _make_mock_trainer({"val/loss": 0.35, "train/loss": 0.2})
        from pytorch_segmentation_models_trainer.tools.experiments_runner.experiments_runner import ExperimentsRunner
        results = ExperimentsRunner(self._make_cfg(seeds=[42])).run()
        self.assertIn("val/loss", results[0].val_metrics)
        self.assertAlmostEqual(results[0].val_metrics["val/loss"], 0.35, places=4)

    @patch(_TRAIN_PATH)
    def test_train_metrics_in_result(self, mock_train):
        mock_train.return_value = _make_mock_trainer({"val/loss": 0.5, "train/loss": 0.2})
        from pytorch_segmentation_models_trainer.tools.experiments_runner.experiments_runner import ExperimentsRunner
        results = ExperimentsRunner(self._make_cfg(seeds=[42])).run()
        self.assertIn("train/loss", results[0].train_metrics)

    @patch(_TRAIN_PATH)
    def test_test_metrics_in_result(self, mock_train):
        mock_train.return_value = _make_mock_trainer({"test/iou": 0.75, "val/loss": 0.3})
        from pytorch_segmentation_models_trainer.tools.experiments_runner.experiments_runner import ExperimentsRunner
        results = ExperimentsRunner(self._make_cfg(seeds=[42])).run()
        self.assertIn("test/iou", results[0].test_metrics)

    @patch(_TRAIN_PATH)
    def test_correct_seed_passed_to_each_run_cfg(self, mock_train):
        mock_train.return_value = _make_mock_trainer()
        from pytorch_segmentation_models_trainer.tools.experiments_runner.experiments_runner import ExperimentsRunner
        ExperimentsRunner(self._make_cfg(seeds=[42, 101, 28])).run()
        seeds_used = [c.args[0].seed for c in mock_train.call_args_list]
        self.assertEqual(seeds_used, [42, 101, 28])

    @patch(_TRAIN_PATH)
    def test_experiments_runner_block_absent_in_train_call(self, mock_train):
        mock_train.return_value = _make_mock_trainer()
        from pytorch_segmentation_models_trainer.tools.experiments_runner.experiments_runner import ExperimentsRunner
        ExperimentsRunner(self._make_cfg(seeds=[42])).run()
        cfg_passed = mock_train.call_args.args[0]
        self.assertNotIn("experiments_runner", cfg_passed)

    @patch(_TRAIN_PATH)
    def test_n_runs_auto_seeds(self, mock_train):
        mock_train.return_value = _make_mock_trainer()
        from pytorch_segmentation_models_trainer.tools.experiments_runner.experiments_runner import ExperimentsRunner
        results = ExperimentsRunner(self._make_cfg(n_runs=3)).run()
        self.assertEqual(len(results), 3)
        self.assertEqual(mock_train.call_count, 3)

    @patch(_TRAIN_PATH)
    def test_summary_csv_created_when_enabled(self, mock_train):
        mock_train.return_value = _make_mock_trainer({"val/loss": 0.5})
        from pytorch_segmentation_models_trainer.tools.experiments_runner.experiments_runner import ExperimentsRunner
        ExperimentsRunner(self._make_cfg(seeds=[42, 101], save_summary=True)).run()
        self.assertTrue(os.path.exists(os.path.join(self.tmp, "summary.csv")))

    @patch(_TRAIN_PATH)
    def test_summary_csv_not_created_when_disabled(self, mock_train):
        mock_train.return_value = _make_mock_trainer()
        from pytorch_segmentation_models_trainer.tools.experiments_runner.experiments_runner import ExperimentsRunner
        ExperimentsRunner(self._make_cfg(seeds=[42], save_summary=False)).run()
        self.assertFalse(os.path.exists(os.path.join(self.tmp, "summary.csv")))

    @patch(_TRAIN_PATH)
    def test_summary_csv_has_mean_and_std_rows(self, mock_train):
        mock_train.return_value = _make_mock_trainer({"val/loss": 0.5})
        from pytorch_segmentation_models_trainer.tools.experiments_runner.experiments_runner import ExperimentsRunner
        ExperimentsRunner(self._make_cfg(seeds=[42, 101], save_summary=True)).run()
        with open(os.path.join(self.tmp, "summary.csv")) as f:
            content = f.read()
        self.assertIn("mean", content)
        self.assertIn("std", content)

    @patch(_TRAIN_PATH)
    def test_summary_csv_has_duration_column(self, mock_train):
        mock_train.return_value = _make_mock_trainer({"val/loss": 0.5})
        from pytorch_segmentation_models_trainer.tools.experiments_runner.experiments_runner import ExperimentsRunner
        ExperimentsRunner(self._make_cfg(seeds=[42, 101], save_summary=True)).run()
        with open(os.path.join(self.tmp, "summary.csv")) as f:
            header = f.readline()
        self.assertIn("duration_s", header)

    @patch(_TRAIN_PATH)
    def test_result_output_dir_matches_run_folder(self, mock_train):
        mock_train.return_value = _make_mock_trainer()
        from pytorch_segmentation_models_trainer.tools.experiments_runner.experiments_runner import ExperimentsRunner
        results = ExperimentsRunner(self._make_cfg(seeds=[42])).run()
        self.assertIn("run_00", results[0].output_dir)

    @patch(_TRAIN_PATH)
    def test_single_run_std_row_is_zero(self, mock_train):
        """With a single run, std must not raise and should be 0."""
        mock_train.return_value = _make_mock_trainer({"val/loss": 0.5})
        from pytorch_segmentation_models_trainer.tools.experiments_runner.experiments_runner import ExperimentsRunner
        # Must not raise even with a single result
        ExperimentsRunner(self._make_cfg(seeds=[42], save_summary=True)).run()
        with open(os.path.join(self.tmp, "summary.csv")) as f:
            content = f.read()
        self.assertIn("std", content)
