# -*- coding: utf-8 -*-
"""
Tests for ExperimentsRunner (TDD — written before implementation).

Run with:
    uv run pytest tests/test_experiments_runner.py -v --tb=short
"""

import json
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
    resume=False,
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
    runner_block["resume"] = resume
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


def _make_mock_trainer(metrics=None, current_epoch=4, best_checkpoint_path=""):
    mock_trainer = MagicMock(spec=pl.Trainer)
    mock_trainer.callback_metrics = metrics or {"val/loss": 0.5, "train/loss": 0.3}
    mock_trainer.current_epoch = current_epoch
    mock_trainer.checkpoint_callback = MagicMock()
    mock_trainer.checkpoint_callback.best_model_path = best_checkpoint_path
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

    def test_output_dir_contains_run_idx_and_seed(self):
        ExperimentsRunner = self._import()
        cfg = _make_runner_cfg(seeds=[42, 101], output_base_dir="/base/runs")
        _, output_dir = ExperimentsRunner(cfg)._build_run_cfg(1, 101)
        self.assertIn("run_01", output_dir)
        self.assertIn("seed101", output_dir)

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
        self.assertIn("seed42", output_dir)

    def test_second_run_idx_formatted(self):
        ExperimentsRunner = self._import()
        cfg = _make_runner_cfg(seeds=[42, 99], output_base_dir="/base")
        _, output_dir = ExperimentsRunner(cfg)._build_run_cfg(9, 99)
        self.assertIn("run_09", output_dir)
        self.assertIn("seed99", output_dir)


class TestLoggerInjection(BasicTestCase):
    """_build_run_cfg() must inject run identity into the configured logger."""

    def _import(self):
        from pytorch_segmentation_models_trainer.tools.experiments_runner.experiments_runner import (
            ExperimentsRunner,
        )

        return ExperimentsRunner

    def _make_cfg_with_logger(self, logger_block):
        base = _make_runner_cfg(seeds=[42])
        cfg_dict = OmegaConf.to_container(base, resolve=False)
        cfg_dict["logger"] = logger_block
        return OmegaConf.create(cfg_dict)

    def test_version_field_set_to_run_tag(self):
        ExperimentsRunner = self._import()
        cfg = self._make_cfg_with_logger(
            {
                "_target_": "pytorch_lightning.loggers.TensorBoardLogger",
                "save_dir": "logs",
                "name": "my_exp",
                "version": 0,
            }
        )
        run_cfg, _ = ExperimentsRunner(cfg)._build_run_cfg(0, 42)
        self.assertEqual(run_cfg.logger.version, "run_00_seed42")

    def test_name_appended_when_no_version(self):
        ExperimentsRunner = self._import()
        cfg = self._make_cfg_with_logger(
            {
                "_target_": "pytorch_lightning.loggers.WandbLogger",
                "name": "my_exp",
                "project": "segmentation",
            }
        )
        run_cfg, _ = ExperimentsRunner(cfg)._build_run_cfg(0, 42)
        self.assertIn("run_00_seed42", run_cfg.logger.name)
        self.assertTrue(run_cfg.logger.name.startswith("my_exp"))

    def test_no_logger_in_cfg_no_crash(self):
        ExperimentsRunner = self._import()
        cfg = _make_runner_cfg(seeds=[42])
        # Must not raise even without a logger block
        run_cfg, _ = ExperimentsRunner(cfg)._build_run_cfg(0, 42)
        self.assertNotIn("logger", run_cfg)

    def test_lightning_default_logger_true_no_crash(self):
        ExperimentsRunner = self._import()
        cfg = self._make_cfg_with_logger(True)
        run_cfg, _ = ExperimentsRunner(cfg)._build_run_cfg(0, 42)
        self.assertEqual(run_cfg.logger, True)


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
        trainer = _make_mock_trainer(
            {"val/loss": 0.35, "train/loss": 0.2, "test/iou": 0.75}
        )
        _, val, _ = runner._collect_metrics(trainer)
        self.assertIn("val/loss", val)
        self.assertNotIn("train/loss", val)

    def test_lightning_suffix_metric_convention_is_collected(self):
        ExperimentsRunner = self._import()
        cfg = _make_runner_cfg(seeds=[42])
        runner = ExperimentsRunner(cfg)
        trainer = _make_mock_trainer(
            {"loss/val": 0.35, "JaccardIndex/val": 0.6, "loss/train": 0.2}
        )
        train, val, _ = runner._collect_metrics(trainer)
        self.assertIn("loss/val", val)
        self.assertIn("JaccardIndex/val", val)
        self.assertIn("loss/train", train)

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

    def _make_cfg(self, seeds=None, n_runs=None, save_summary=False, resume=False):
        return _make_runner_cfg(
            seeds=seeds,
            n_runs=n_runs,
            output_base_dir=self.tmp,
            save_summary=save_summary,
            resume=resume,
        )

    @patch(_TRAIN_PATH)
    def test_train_called_once_per_seed(self, mock_train):
        mock_train.return_value = _make_mock_trainer()
        from pytorch_segmentation_models_trainer.tools.experiments_runner.experiments_runner import (
            ExperimentsRunner,
        )

        runner = ExperimentsRunner(self._make_cfg(seeds=[42, 101, 28]))
        runner.run()
        self.assertEqual(mock_train.call_count, 3)

    @patch(_TRAIN_PATH)
    def test_returns_one_result_per_run(self, mock_train):
        mock_train.return_value = _make_mock_trainer()
        from pytorch_segmentation_models_trainer.tools.experiments_runner.experiments_runner import (
            ExperimentsRunner,
        )

        results = ExperimentsRunner(self._make_cfg(seeds=[42, 101])).run()
        self.assertEqual(len(results), 2)

    @patch(_TRAIN_PATH)
    def test_result_seeds_match_config(self, mock_train):
        mock_train.return_value = _make_mock_trainer()
        from pytorch_segmentation_models_trainer.tools.experiments_runner.experiments_runner import (
            ExperimentsRunner,
        )

        results = ExperimentsRunner(self._make_cfg(seeds=[42, 101])).run()
        self.assertEqual(results[0].seed, 42)
        self.assertEqual(results[1].seed, 101)

    @patch(_TRAIN_PATH)
    def test_result_run_idx_sequential(self, mock_train):
        mock_train.return_value = _make_mock_trainer()
        from pytorch_segmentation_models_trainer.tools.experiments_runner.experiments_runner import (
            ExperimentsRunner,
        )

        results = ExperimentsRunner(self._make_cfg(seeds=[10, 20, 30])).run()
        self.assertEqual([r.run_idx for r in results], [0, 1, 2])

    @patch(_TRAIN_PATH)
    def test_training_time_non_negative(self, mock_train):
        mock_train.return_value = _make_mock_trainer()
        from pytorch_segmentation_models_trainer.tools.experiments_runner.experiments_runner import (
            ExperimentsRunner,
        )

        results = ExperimentsRunner(self._make_cfg(seeds=[42])).run()
        self.assertGreaterEqual(results[0].training_time_seconds, 0.0)

    @patch(_TRAIN_PATH)
    def test_val_metrics_in_result(self, mock_train):
        mock_train.return_value = _make_mock_trainer(
            {"val/loss": 0.35, "train/loss": 0.2}
        )
        from pytorch_segmentation_models_trainer.tools.experiments_runner.experiments_runner import (
            ExperimentsRunner,
        )

        results = ExperimentsRunner(self._make_cfg(seeds=[42])).run()
        self.assertIn("val/loss", results[0].val_metrics)
        self.assertAlmostEqual(results[0].val_metrics["val/loss"], 0.35, places=4)

    @patch(_TRAIN_PATH)
    def test_train_metrics_in_result(self, mock_train):
        mock_train.return_value = _make_mock_trainer(
            {"val/loss": 0.5, "train/loss": 0.2}
        )
        from pytorch_segmentation_models_trainer.tools.experiments_runner.experiments_runner import (
            ExperimentsRunner,
        )

        results = ExperimentsRunner(self._make_cfg(seeds=[42])).run()
        self.assertIn("train/loss", results[0].train_metrics)

    @patch(_TRAIN_PATH)
    def test_test_metrics_in_result(self, mock_train):
        mock_train.return_value = _make_mock_trainer(
            {"test/iou": 0.75, "val/loss": 0.3}
        )
        from pytorch_segmentation_models_trainer.tools.experiments_runner.experiments_runner import (
            ExperimentsRunner,
        )

        results = ExperimentsRunner(self._make_cfg(seeds=[42])).run()
        self.assertIn("test/iou", results[0].test_metrics)

    @patch(_TRAIN_PATH)
    def test_correct_seed_passed_to_each_run_cfg(self, mock_train):
        mock_train.return_value = _make_mock_trainer()
        from pytorch_segmentation_models_trainer.tools.experiments_runner.experiments_runner import (
            ExperimentsRunner,
        )

        ExperimentsRunner(self._make_cfg(seeds=[42, 101, 28])).run()
        seeds_used = [c.args[0].seed for c in mock_train.call_args_list]
        self.assertEqual(seeds_used, [42, 101, 28])

    @patch(_TRAIN_PATH)
    def test_experiments_runner_block_absent_in_train_call(self, mock_train):
        mock_train.return_value = _make_mock_trainer()
        from pytorch_segmentation_models_trainer.tools.experiments_runner.experiments_runner import (
            ExperimentsRunner,
        )

        ExperimentsRunner(self._make_cfg(seeds=[42])).run()
        cfg_passed = mock_train.call_args.args[0]
        self.assertNotIn("experiments_runner", cfg_passed)

    @patch(_TRAIN_PATH)
    def test_n_runs_auto_seeds(self, mock_train):
        mock_train.return_value = _make_mock_trainer()
        from pytorch_segmentation_models_trainer.tools.experiments_runner.experiments_runner import (
            ExperimentsRunner,
        )

        results = ExperimentsRunner(self._make_cfg(n_runs=3)).run()
        self.assertEqual(len(results), 3)
        self.assertEqual(mock_train.call_count, 3)

    @patch(_TRAIN_PATH)
    def test_summary_csv_created_when_enabled(self, mock_train):
        mock_train.return_value = _make_mock_trainer({"val/loss": 0.5})
        from pytorch_segmentation_models_trainer.tools.experiments_runner.experiments_runner import (
            ExperimentsRunner,
        )

        ExperimentsRunner(self._make_cfg(seeds=[42, 101], save_summary=True)).run()
        self.assertTrue(os.path.exists(os.path.join(self.tmp, "summary.csv")))

    @patch(_TRAIN_PATH)
    def test_summary_csv_not_created_when_disabled(self, mock_train):
        mock_train.return_value = _make_mock_trainer()
        from pytorch_segmentation_models_trainer.tools.experiments_runner.experiments_runner import (
            ExperimentsRunner,
        )

        ExperimentsRunner(self._make_cfg(seeds=[42], save_summary=False)).run()
        self.assertFalse(os.path.exists(os.path.join(self.tmp, "summary.csv")))

    @patch(_TRAIN_PATH)
    def test_summary_csv_has_mean_and_std_rows(self, mock_train):
        mock_train.return_value = _make_mock_trainer({"val/loss": 0.5})
        from pytorch_segmentation_models_trainer.tools.experiments_runner.experiments_runner import (
            ExperimentsRunner,
        )

        ExperimentsRunner(self._make_cfg(seeds=[42, 101], save_summary=True)).run()
        with open(os.path.join(self.tmp, "summary.csv")) as f:
            content = f.read()
        self.assertIn("mean", content)
        self.assertIn("std", content)

    @patch(_TRAIN_PATH)
    def test_summary_csv_has_duration_column(self, mock_train):
        mock_train.return_value = _make_mock_trainer({"val/loss": 0.5})
        from pytorch_segmentation_models_trainer.tools.experiments_runner.experiments_runner import (
            ExperimentsRunner,
        )

        ExperimentsRunner(self._make_cfg(seeds=[42, 101], save_summary=True)).run()
        with open(os.path.join(self.tmp, "summary.csv")) as f:
            header = f.readline()
        self.assertIn("duration_s", header)

    @patch(_TRAIN_PATH)
    def test_result_output_dir_contains_seed(self, mock_train):
        mock_train.return_value = _make_mock_trainer()
        from pytorch_segmentation_models_trainer.tools.experiments_runner.experiments_runner import (
            ExperimentsRunner,
        )

        results = ExperimentsRunner(self._make_cfg(seeds=[42])).run()
        self.assertIn("run_00", results[0].output_dir)
        self.assertIn("seed42", results[0].output_dir)

    @patch(_TRAIN_PATH)
    def test_single_run_std_row_is_zero(self, mock_train):
        """With a single run, std must not raise and should be 0."""
        mock_train.return_value = _make_mock_trainer({"val/loss": 0.5})
        from pytorch_segmentation_models_trainer.tools.experiments_runner.experiments_runner import (
            ExperimentsRunner,
        )

        # Must not raise even with a single result
        ExperimentsRunner(self._make_cfg(seeds=[42], save_summary=True)).run()
        with open(os.path.join(self.tmp, "summary.csv")) as f:
            content = f.read()
        self.assertIn("std", content)

    @patch(_TRAIN_PATH)
    def test_summary_written_after_each_run(self, mock_train):
        """summary.csv must exist after the first run, not just at the end."""
        summaries_seen = []

        def fake_train(cfg):
            summaries_seen.append(os.path.exists(os.path.join(self.tmp, "summary.csv")))
            return _make_mock_trainer({"val/loss": 0.5})

        mock_train.side_effect = fake_train
        from pytorch_segmentation_models_trainer.tools.experiments_runner.experiments_runner import (
            ExperimentsRunner,
        )

        ExperimentsRunner(self._make_cfg(seeds=[42, 101], save_summary=True)).run()
        # After run 0 completes, summary already exists when run 1 starts
        self.assertTrue(
            summaries_seen[1], "summary.csv was not present before run 1 started"
        )


class TestStateFile(BasicTestCase):
    """runner_state.json is written after each run and drives resume logic."""

    def setUp(self):
        super().setUp()
        self.tmp = self.make_temp_dir()

    def _make_cfg(self, seeds=None, n_runs=None, resume=False, save_summary=False):
        return _make_runner_cfg(
            seeds=seeds,
            n_runs=n_runs,
            output_base_dir=self.tmp,
            save_summary=save_summary,
            resume=resume,
        )

    @patch(_TRAIN_PATH)
    def test_state_file_created_after_first_run(self, mock_train):
        state_path = os.path.join(self.tmp, "runner_state.json")
        state_snapshots = []

        def fake_train(cfg):
            state_snapshots.append(os.path.exists(state_path))
            return _make_mock_trainer()

        mock_train.side_effect = fake_train
        from pytorch_segmentation_models_trainer.tools.experiments_runner.experiments_runner import (
            ExperimentsRunner,
        )

        ExperimentsRunner(self._make_cfg(seeds=[42, 101])).run()
        # State file must exist before run 1 starts (written after run 0)
        self.assertTrue(state_snapshots[1])

    @patch(_TRAIN_PATH)
    def test_state_file_contains_all_seeds(self, mock_train):
        mock_train.return_value = _make_mock_trainer()
        from pytorch_segmentation_models_trainer.tools.experiments_runner.experiments_runner import (
            ExperimentsRunner,
        )

        ExperimentsRunner(self._make_cfg(seeds=[42, 101, 28])).run()
        with open(os.path.join(self.tmp, "runner_state.json")) as f:
            state = json.load(f)
        self.assertEqual(state["all_seeds"], [42, 101, 28])

    @patch(_TRAIN_PATH)
    def test_state_file_records_completed_runs(self, mock_train):
        mock_train.return_value = _make_mock_trainer({"val/loss": 0.4})
        from pytorch_segmentation_models_trainer.tools.experiments_runner.experiments_runner import (
            ExperimentsRunner,
        )

        ExperimentsRunner(self._make_cfg(seeds=[42, 101])).run()
        with open(os.path.join(self.tmp, "runner_state.json")) as f:
            state = json.load(f)
        self.assertEqual(len(state["completed_runs"]), 2)
        self.assertEqual(state["completed_runs"][0]["seed"], 42)
        self.assertEqual(state["completed_runs"][1]["seed"], 101)

    @patch(_TRAIN_PATH)
    def test_state_file_preserves_auto_seeds_for_resume(self, mock_train):
        """With n_runs, state file must store the generated seeds so resume uses them."""
        mock_train.return_value = _make_mock_trainer()
        from pytorch_segmentation_models_trainer.tools.experiments_runner.experiments_runner import (
            ExperimentsRunner,
        )

        ExperimentsRunner(self._make_cfg(n_runs=3)).run()
        with open(os.path.join(self.tmp, "runner_state.json")) as f:
            state = json.load(f)
        self.assertEqual(len(state["all_seeds"]), 3)
        self.assertEqual(len(state["completed_runs"]), 3)

    @patch(_TRAIN_PATH)
    def test_resume_skips_already_completed_runs(self, mock_train):
        mock_train.return_value = _make_mock_trainer()
        from pytorch_segmentation_models_trainer.tools.experiments_runner.experiments_runner import (
            ExperimentsRunner,
            RunResult,
        )

        # Pre-write a state file with run 0 completed
        completed = RunResult(
            run_idx=0,
            seed=42,
            training_time_seconds=10.0,
            train_metrics={"train/loss": 0.2},
            val_metrics={"val/loss": 0.4},
            test_metrics={},
            output_dir=os.path.join(self.tmp, "run_00_seed42"),
        )
        import dataclasses

        state = {
            "all_seeds": [42, 101, 28],
            "completed_runs": [dataclasses.asdict(completed)],
        }
        os.makedirs(self.tmp, exist_ok=True)
        with open(os.path.join(self.tmp, "runner_state.json"), "w") as f:
            json.dump(state, f)

        cfg = self._make_cfg(seeds=[42, 101, 28], resume=True)
        results = ExperimentsRunner(cfg).run()

        # train() must only have been called for runs 1 and 2
        self.assertEqual(mock_train.call_count, 2)
        self.assertEqual(len(results), 3)

    @patch(_TRAIN_PATH)
    def test_resume_false_ignores_existing_state(self, mock_train):
        mock_train.return_value = _make_mock_trainer()
        from pytorch_segmentation_models_trainer.tools.experiments_runner.experiments_runner import (
            ExperimentsRunner,
            RunResult,
        )

        # Pre-write a complete state file
        import dataclasses

        completed = [
            dataclasses.asdict(
                RunResult(
                    run_idx=i,
                    seed=s,
                    training_time_seconds=10.0,
                    train_metrics={},
                    val_metrics={"val/loss": 0.5},
                    test_metrics={},
                    output_dir=f"/tmp/run_{i:02d}_seed{s}",
                )
            )
            for i, s in enumerate([42, 101])
        ]
        state = {"all_seeds": [42, 101], "completed_runs": completed}
        os.makedirs(self.tmp, exist_ok=True)
        with open(os.path.join(self.tmp, "runner_state.json"), "w") as f:
            json.dump(state, f)

        cfg = self._make_cfg(seeds=[42, 101], resume=False)
        ExperimentsRunner(cfg).run()

        # resume=False → all runs re-executed
        self.assertEqual(mock_train.call_count, 2)

    @patch(_TRAIN_PATH)
    def test_resume_uses_seeds_from_state_not_new_random(self, mock_train):
        """When n_runs is used and we resume, seeds come from state file."""
        mock_train.return_value = _make_mock_trainer()
        from pytorch_segmentation_models_trainer.tools.experiments_runner.experiments_runner import (
            ExperimentsRunner,
            RunResult,
        )
        import dataclasses

        fixed_seeds = [111111, 222222]
        completed = [
            dataclasses.asdict(
                RunResult(
                    run_idx=0,
                    seed=111111,
                    training_time_seconds=5.0,
                    train_metrics={},
                    val_metrics={"val/loss": 0.3},
                    test_metrics={},
                    output_dir=f"/tmp/run_00_seed111111",
                )
            )
        ]
        state = {"all_seeds": fixed_seeds, "completed_runs": completed}
        os.makedirs(self.tmp, exist_ok=True)
        with open(os.path.join(self.tmp, "runner_state.json"), "w") as f:
            json.dump(state, f)

        # Use n_runs (would generate new random seeds) but resume=True
        cfg = self._make_cfg(n_runs=2, resume=True)
        results = ExperimentsRunner(cfg).run()

        # Only run 1 should execute; its seed must be the one from state
        self.assertEqual(mock_train.call_count, 1)
        self.assertEqual(results[1].seed, 222222)

    @patch(_TRAIN_PATH)
    def test_resume_no_state_file_starts_fresh(self, mock_train):
        """resume=True with no state file on disk must not crash."""
        mock_train.return_value = _make_mock_trainer()
        from pytorch_segmentation_models_trainer.tools.experiments_runner.experiments_runner import (
            ExperimentsRunner,
        )

        cfg = self._make_cfg(seeds=[42, 101], resume=True)
        results = ExperimentsRunner(cfg).run()
        self.assertEqual(mock_train.call_count, 2)
        self.assertEqual(len(results), 2)


import dataclasses
import pandas as pd
from pathlib import Path

_SPLITTER_PATH = (
    "pytorch_segmentation_models_trainer"
    ".tools.experiments_runner.experiments_runner.SpatialKFoldSplitter"
)


def _make_kfold_runner_cfg(
    tmp_dir: str,
    csv_path: str,
    seeds=None,
    n_splits: int = 3,
    save_summary: bool = False,
    resume: bool = False,
):
    """Build a DictConfig with kfold block, train_dataset, and val_dataset."""
    cfg_dict = {
        "mode": "run-experiments",
        "experiments_runner": {
            "output_base_dir": tmp_dir,
            "seeds": seeds or [42, 101],
            "save_summary": save_summary,
            "summary_metrics": ["val/loss"],
            "resume": resume,
            "kfold": {
                "n_splits": n_splits,
                "seed": 42,
                "split_strategy": "by_image",
                "group_col": "image",
                "buffer_px": 0,
                "row_col": "row_off",
                "col_col": "col_off",
                "input_csv_path": csv_path,
                "save_fold_csvs_dir": "fold_splits",
            },
        },
        "seed": 0,
        "hyperparameters": {"batch_size": 1, "epochs": 1, "max_lr": 0.1},
        "pl_trainer": {
            "max_epochs": 1,
            "accelerator": "cpu",
            "devices": 1,
            "default_root_dir": "/old/path",
        },
        "train_dataset": {
            "_target_": "some.dataset.Class",
            "input_csv_path": csv_path,
        },
        "val_dataset": {
            "_target_": "some.dataset.Class",
            "input_csv_path": csv_path,
        },
    }
    return OmegaConf.create(cfg_dict)


class TestKFoldRunMethod(BasicTestCase):
    """run() with k-fold: correct folds × seeds, CSV injection, output structure."""

    def setUp(self):
        super().setUp()
        self.tmp = self.make_temp_dir()
        csv_path = os.path.join(self.tmp, "all_patches.csv")
        rows = [
            {
                "image": f"/data/img_{i}.tif",
                "mask": f"/data/mask_{i}.tif",
                "row_off": 0,
                "col_off": 0,
                "patch_size": 256,
            }
            for i in range(6)
        ]
        pd.DataFrame(rows).to_csv(csv_path, index=False)
        self.csv_path = csv_path

    def _make_cfg(self, seeds=None, n_splits=3, save_summary=False, resume=False):
        return _make_kfold_runner_cfg(
            tmp_dir=self.tmp,
            csv_path=self.csv_path,
            seeds=seeds,
            n_splits=n_splits,
            save_summary=save_summary,
            resume=resume,
        )

    @patch(_TRAIN_PATH)
    def test_no_kfold_keeps_original_behavior(self, mock_train):
        """Without a kfold block, run() behaves exactly as before."""
        mock_train.return_value = _make_mock_trainer()
        from pytorch_segmentation_models_trainer.tools.experiments_runner.experiments_runner import (
            ExperimentsRunner,
        )

        cfg = _make_runner_cfg(seeds=[42, 101], output_base_dir=self.tmp)
        results = ExperimentsRunner(cfg).run()
        self.assertEqual(mock_train.call_count, 2)
        self.assertEqual(len(results), 2)

    @patch(_TRAIN_PATH)
    def test_kfold_correct_total_runs(self, mock_train):
        """3 folds × 2 seeds → 6 train() calls and 6 RunResults."""
        mock_train.return_value = _make_mock_trainer()
        from pytorch_segmentation_models_trainer.tools.experiments_runner.experiments_runner import (
            ExperimentsRunner,
        )

        cfg = self._make_cfg(seeds=[42, 101], n_splits=3)
        results = ExperimentsRunner(cfg).run()
        self.assertEqual(mock_train.call_count, 6)
        self.assertEqual(len(results), 6)

    @patch(_TRAIN_PATH)
    def test_kfold_injects_train_csv_path(self, mock_train):
        """train_dataset.input_csv_path points to a fold_K_train.csv for every call."""
        mock_train.return_value = _make_mock_trainer()
        from pytorch_segmentation_models_trainer.tools.experiments_runner.experiments_runner import (
            ExperimentsRunner,
        )

        cfg = self._make_cfg(seeds=[42], n_splits=3)
        ExperimentsRunner(cfg).run()
        for c in mock_train.call_args_list:
            call_cfg = c.args[0]
            path = call_cfg.train_dataset.input_csv_path
            self.assertIn("fold_", path)
            self.assertIn("_train", path)

    @patch(_TRAIN_PATH)
    def test_kfold_injects_val_csv_path(self, mock_train):
        """val_dataset.input_csv_path points to a fold_K_val.csv for every call."""
        mock_train.return_value = _make_mock_trainer()
        from pytorch_segmentation_models_trainer.tools.experiments_runner.experiments_runner import (
            ExperimentsRunner,
        )

        cfg = self._make_cfg(seeds=[42], n_splits=3)
        ExperimentsRunner(cfg).run()
        for c in mock_train.call_args_list:
            call_cfg = c.args[0]
            path = call_cfg.val_dataset.input_csv_path
            self.assertIn("fold_", path)
            self.assertIn("_val", path)

    @patch(_TRAIN_PATH)
    def test_kfold_output_dir_contains_fold_tag(self, mock_train):
        """default_root_dir in each call contains a fold_XX tag."""
        mock_train.return_value = _make_mock_trainer()
        from pytorch_segmentation_models_trainer.tools.experiments_runner.experiments_runner import (
            ExperimentsRunner,
        )

        cfg = self._make_cfg(seeds=[42], n_splits=3)
        ExperimentsRunner(cfg).run()
        dirs = [
            c.args[0].pl_trainer.default_root_dir for c in mock_train.call_args_list
        ]
        for expected_tag in ("fold_00", "fold_01", "fold_02"):
            self.assertTrue(
                any(expected_tag in d for d in dirs),
                msg=f"Tag {expected_tag!r} not found in any output dir: {dirs}",
            )

    @patch(_TRAIN_PATH)
    def test_kfold_summary_has_fold_idx_column(self, mock_train):
        """summary.csv contains a fold_idx column when kfold is active."""
        mock_train.return_value = _make_mock_trainer({"val/loss": 0.5})
        from pytorch_segmentation_models_trainer.tools.experiments_runner.experiments_runner import (
            ExperimentsRunner,
        )

        cfg = self._make_cfg(seeds=[42], n_splits=3, save_summary=True)
        ExperimentsRunner(cfg).run()
        with open(os.path.join(self.tmp, "summary.csv")) as f:
            header = f.readline()
        self.assertIn("fold_idx", header)

    @patch(_TRAIN_PATH)
    def test_kfold_resume_skips_completed_fold_runs(self, mock_train):
        """With 3 of 6 runs pre-completed, only 3 train() calls are made."""
        mock_train.return_value = _make_mock_trainer()
        from pytorch_segmentation_models_trainer.tools.experiments_runner.experiments_runner import (
            ExperimentsRunner,
            RunResult,
        )

        completed = [
            dataclasses.asdict(
                RunResult(
                    run_idx=i,
                    seed=42,
                    training_time_seconds=5.0,
                    train_metrics={},
                    val_metrics={"val/loss": 0.3},
                    test_metrics={},
                    output_dir=f"/tmp/fold_{i:02d}_seed42",
                    fold_idx=i,
                )
            )
            for i in range(3)
        ]
        state = {"all_seeds": [42, 101], "completed_runs": completed}
        os.makedirs(self.tmp, exist_ok=True)
        with open(os.path.join(self.tmp, "runner_state.json"), "w") as f:
            json.dump(state, f)

        cfg = self._make_cfg(seeds=[42, 101], n_splits=3, resume=True)
        results = ExperimentsRunner(cfg).run()
        self.assertEqual(mock_train.call_count, 3)
        self.assertEqual(len(results), 6)

    @patch(_TRAIN_PATH)
    def test_kfold_fold_csvs_generated_once(self, mock_train):
        """generate_and_save_folds is called exactly once regardless of seed count."""
        mock_train.return_value = _make_mock_trainer()
        from pytorch_segmentation_models_trainer.tools.experiments_runner.experiments_runner import (
            ExperimentsRunner,
        )

        n_splits = 3
        fake_fold_paths = [
            (Path(f"/fake/fold_{k}_train.csv"), Path(f"/fake/fold_{k}_val.csv"))
            for k in range(n_splits)
        ]
        with patch(_SPLITTER_PATH) as MockSplitter:
            mock_instance = MockSplitter.return_value
            mock_instance.generate_and_save_folds.return_value = fake_fold_paths

            cfg = self._make_cfg(seeds=[42, 101], n_splits=n_splits)
            ExperimentsRunner(cfg).run()

            self.assertEqual(mock_instance.generate_and_save_folds.call_count, 1)

    @patch(_TRAIN_PATH)
    def test_kfold_run_result_has_fold_idx(self, mock_train):
        """RunResult.fold_idx contains the correct fold index for each run."""
        mock_train.return_value = _make_mock_trainer()
        from pytorch_segmentation_models_trainer.tools.experiments_runner.experiments_runner import (
            ExperimentsRunner,
        )

        cfg = self._make_cfg(seeds=[42], n_splits=3)
        results = ExperimentsRunner(cfg).run()
        fold_indices = sorted(r.fold_idx for r in results)
        self.assertEqual(fold_indices, [0, 1, 2])


class TestRunResultNewFields(BasicTestCase):
    """RunResult must expose epochs_trained and best_checkpoint_path fields."""

    def _make_minimal(self, **kwargs):
        from pytorch_segmentation_models_trainer.tools.experiments_runner.experiments_runner import (
            RunResult,
        )

        defaults = dict(
            run_idx=0,
            seed=42,
            training_time_seconds=1.0,
            train_metrics={},
            val_metrics={},
            test_metrics={},
            output_dir="/tmp",
        )
        defaults.update(kwargs)
        return RunResult(**defaults)

    def test_epochs_trained_defaults_to_none(self):
        r = self._make_minimal()
        self.assertIsNone(r.epochs_trained)

    def test_best_checkpoint_path_defaults_to_none(self):
        r = self._make_minimal()
        self.assertIsNone(r.best_checkpoint_path)

    def test_can_set_epochs_trained(self):
        r = self._make_minimal(epochs_trained=7)
        self.assertEqual(r.epochs_trained, 7)

    def test_can_set_best_checkpoint_path(self):
        r = self._make_minimal(best_checkpoint_path="/ckpt/best.ckpt")
        self.assertEqual(r.best_checkpoint_path, "/ckpt/best.ckpt")


class TestRunSingleExtractsTrainingInfo(BasicTestCase):
    """_run_single must capture epochs_trained and best_checkpoint_path."""

    def setUp(self):
        super().setUp()
        self.tmp = self.make_temp_dir()

    def _make_cfg(self, seeds=None):
        return _make_runner_cfg(seeds=seeds or [42], output_base_dir=self.tmp)

    @patch(_TRAIN_PATH)
    def test_epochs_trained_is_current_epoch_plus_one(self, mock_train):
        mock_train.return_value = _make_mock_trainer(current_epoch=6)
        from pytorch_segmentation_models_trainer.tools.experiments_runner.experiments_runner import (
            ExperimentsRunner,
        )

        results = ExperimentsRunner(self._make_cfg()).run()
        self.assertEqual(results[0].epochs_trained, 7)

    @patch(_TRAIN_PATH)
    def test_best_checkpoint_path_captured_from_callback(self, mock_train):
        mock_train.return_value = _make_mock_trainer(
            best_checkpoint_path="/ckpt/best.ckpt"
        )
        from pytorch_segmentation_models_trainer.tools.experiments_runner.experiments_runner import (
            ExperimentsRunner,
        )

        results = ExperimentsRunner(self._make_cfg()).run()
        self.assertEqual(results[0].best_checkpoint_path, "/ckpt/best.ckpt")

    @patch(_TRAIN_PATH)
    def test_best_checkpoint_path_empty_when_no_callback(self, mock_train):
        trainer = _make_mock_trainer()
        trainer.checkpoint_callback = None
        mock_train.return_value = trainer
        from pytorch_segmentation_models_trainer.tools.experiments_runner.experiments_runner import (
            ExperimentsRunner,
        )

        results = ExperimentsRunner(self._make_cfg()).run()
        self.assertEqual(results[0].best_checkpoint_path, "")


class TestSelectRepresentativeRun(BasicTestCase):
    """_select_representative_run returns the run closest to the mean metric."""

    def _import(self):
        from pytorch_segmentation_models_trainer.tools.experiments_runner.experiments_runner import (
            ExperimentsRunner,
            RunResult,
        )

        return ExperimentsRunner, RunResult

    def _make_results(self, metric_values, metric_key="val/iou"):
        from pytorch_segmentation_models_trainer.tools.experiments_runner.experiments_runner import (
            RunResult,
        )

        return [
            RunResult(
                run_idx=i,
                seed=i * 10,
                training_time_seconds=1.0,
                train_metrics={},
                val_metrics={metric_key: v},
                test_metrics={},
                output_dir=f"/tmp/run_{i:02d}",
                best_checkpoint_path=f"/ckpt/run_{i:02d}.ckpt",
            )
            for i, v in enumerate(metric_values)
        ]

    def test_closest_to_mean_selected(self):
        ExperimentsRunner, _ = self._import()
        cfg = _make_runner_cfg(seeds=[0, 10, 20])
        runner = ExperimentsRunner(cfg)
        # mean = (0.5 + 0.7 + 0.9) / 3 = 0.7 → run 1 is closest
        results = self._make_results([0.5, 0.7, 0.9], "val/iou")
        rep = runner._select_representative_run(results, "val/iou")
        self.assertEqual(rep.run_idx, 1)

    def test_single_run_returns_that_run(self):
        ExperimentsRunner, _ = self._import()
        cfg = _make_runner_cfg(seeds=[42])
        runner = ExperimentsRunner(cfg)
        results = self._make_results([0.65], "val/iou")
        rep = runner._select_representative_run(results, "val/iou")
        self.assertEqual(rep.run_idx, 0)

    def test_returns_none_when_metric_absent_in_all_runs(self):
        ExperimentsRunner, _ = self._import()
        cfg = _make_runner_cfg(seeds=[42])
        runner = ExperimentsRunner(cfg)
        results = self._make_results([0.65], "val/iou")
        rep = runner._select_representative_run(results, "val/loss")
        self.assertIsNone(rep)

    def test_tie_returns_a_valid_run(self):
        ExperimentsRunner, _ = self._import()
        cfg = _make_runner_cfg(seeds=[0, 10, 20, 30])
        runner = ExperimentsRunner(cfg)
        # mean = 0.7; |0.6-0.7|=|0.8-0.7|=0.1 → either run 1 or run 2 valid
        results = self._make_results([0.5, 0.6, 0.8, 0.9], "val/iou")
        rep = runner._select_representative_run(results, "val/iou")
        self.assertIn(rep.run_idx, [1, 2])

    def test_representative_metric_from_cfg_used(self):
        ExperimentsRunner, _ = self._import()
        cfg = _make_runner_cfg(
            seeds=[0, 10, 20],
            extra={"representative_metric": "val/iou"},
        )
        runner = ExperimentsRunner(cfg)
        metric_key = runner._resolve_representative_metric(
            self._make_results([0.5, 0.7, 0.9], "val/iou")
        )
        self.assertEqual(metric_key, "val/iou")


class TestSelectBestRun(BasicTestCase):
    """_select_best_run returns the run with the highest metric value."""

    def _import(self):
        from pytorch_segmentation_models_trainer.tools.experiments_runner.experiments_runner import (
            ExperimentsRunner,
            RunResult,
        )

        return ExperimentsRunner, RunResult

    def _make_results(self, metric_values, metric_key="val/iou"):
        from pytorch_segmentation_models_trainer.tools.experiments_runner.experiments_runner import (
            RunResult,
        )

        return [
            RunResult(
                run_idx=i,
                seed=i * 10,
                training_time_seconds=1.0,
                train_metrics={},
                val_metrics={metric_key: v},
                test_metrics={},
                output_dir=f"/tmp/run_{i:02d}",
                best_checkpoint_path=f"/ckpt/run_{i:02d}.ckpt",
            )
            for i, v in enumerate(metric_values)
        ]

    def test_best_run_has_highest_metric(self):
        ExperimentsRunner, _ = self._import()
        cfg = _make_runner_cfg(seeds=[0, 10, 20])
        runner = ExperimentsRunner(cfg)
        results = self._make_results([0.5, 0.9, 0.7], "val/iou")
        best = runner._select_best_run(results, "val/iou")
        self.assertEqual(best.run_idx, 1)

    def test_single_run_returns_that_run(self):
        ExperimentsRunner, _ = self._import()
        cfg = _make_runner_cfg(seeds=[42])
        runner = ExperimentsRunner(cfg)
        results = self._make_results([0.65], "val/iou")
        best = runner._select_best_run(results, "val/iou")
        self.assertEqual(best.run_idx, 0)

    def test_returns_none_when_metric_absent(self):
        ExperimentsRunner, _ = self._import()
        cfg = _make_runner_cfg(seeds=[42])
        runner = ExperimentsRunner(cfg)
        results = self._make_results([0.65], "val/iou")
        best = runner._select_best_run(results, "val/loss")
        self.assertIsNone(best)


class TestStateFileRepresentative(BasicTestCase):
    """runner_state.json must include representative and best_run entries."""

    def setUp(self):
        super().setUp()
        self.tmp = self.make_temp_dir()

    @patch(_TRAIN_PATH)
    def test_state_has_representative_key(self, mock_train):
        mock_train.return_value = _make_mock_trainer(
            metrics={"val/iou": 0.7}, best_checkpoint_path="/ckpt/best.ckpt"
        )
        from pytorch_segmentation_models_trainer.tools.experiments_runner.experiments_runner import (
            ExperimentsRunner,
        )

        ExperimentsRunner(
            _make_runner_cfg(seeds=[42, 101], output_base_dir=self.tmp)
        ).run()
        with open(os.path.join(self.tmp, "runner_state.json")) as f:
            state = json.load(f)
        self.assertIn("representative", state)
        self.assertIn("run_idx", state["representative"])
        self.assertIn("checkpoint_path", state["representative"])

    @patch(_TRAIN_PATH)
    def test_state_has_best_run_key(self, mock_train):
        mock_train.return_value = _make_mock_trainer(
            metrics={"val/iou": 0.7}, best_checkpoint_path="/ckpt/best.ckpt"
        )
        from pytorch_segmentation_models_trainer.tools.experiments_runner.experiments_runner import (
            ExperimentsRunner,
        )

        ExperimentsRunner(
            _make_runner_cfg(seeds=[42, 101], output_base_dir=self.tmp)
        ).run()
        with open(os.path.join(self.tmp, "runner_state.json")) as f:
            state = json.load(f)
        self.assertIn("best_run", state)
        self.assertIn("run_idx", state["best_run"])
        self.assertIn("checkpoint_path", state["best_run"])

    @patch(_TRAIN_PATH)
    def test_state_representative_is_none_when_no_val_metrics(self, mock_train):
        mock_train.return_value = _make_mock_trainer(metrics={"train/loss": 0.3})
        from pytorch_segmentation_models_trainer.tools.experiments_runner.experiments_runner import (
            ExperimentsRunner,
        )

        ExperimentsRunner(_make_runner_cfg(seeds=[42], output_base_dir=self.tmp)).run()
        with open(os.path.join(self.tmp, "runner_state.json")) as f:
            state = json.load(f)
        self.assertIsNone(state["representative"])
        self.assertIsNone(state["best_run"])


class TestSummaryCSVNewColumns(BasicTestCase):
    """summary.csv must include epochs_trained, best_checkpoint_path, representative, best_run."""

    def setUp(self):
        super().setUp()
        self.tmp = self.make_temp_dir()

    def _make_cfg(self, seeds):
        return _make_runner_cfg(
            seeds=seeds, output_base_dir=self.tmp, save_summary=True
        )

    @patch(_TRAIN_PATH)
    def test_summary_has_epochs_trained_column(self, mock_train):
        mock_train.return_value = _make_mock_trainer({"val/loss": 0.5}, current_epoch=4)
        from pytorch_segmentation_models_trainer.tools.experiments_runner.experiments_runner import (
            ExperimentsRunner,
        )

        ExperimentsRunner(self._make_cfg([42])).run()
        with open(os.path.join(self.tmp, "summary.csv")) as f:
            header = f.readline()
        self.assertIn("epochs_trained", header)

    @patch(_TRAIN_PATH)
    def test_summary_has_best_checkpoint_path_column(self, mock_train):
        mock_train.return_value = _make_mock_trainer({"val/loss": 0.5})
        from pytorch_segmentation_models_trainer.tools.experiments_runner.experiments_runner import (
            ExperimentsRunner,
        )

        ExperimentsRunner(self._make_cfg([42])).run()
        with open(os.path.join(self.tmp, "summary.csv")) as f:
            header = f.readline()
        self.assertIn("best_checkpoint_path", header)

    @patch(_TRAIN_PATH)
    def test_summary_has_representative_column(self, mock_train):
        mock_train.return_value = _make_mock_trainer({"val/loss": 0.5})
        from pytorch_segmentation_models_trainer.tools.experiments_runner.experiments_runner import (
            ExperimentsRunner,
        )

        ExperimentsRunner(self._make_cfg([42])).run()
        with open(os.path.join(self.tmp, "summary.csv")) as f:
            header = f.readline()
        self.assertIn("representative", header)

    @patch(_TRAIN_PATH)
    def test_summary_has_best_run_column(self, mock_train):
        mock_train.return_value = _make_mock_trainer({"val/loss": 0.5})
        from pytorch_segmentation_models_trainer.tools.experiments_runner.experiments_runner import (
            ExperimentsRunner,
        )

        ExperimentsRunner(self._make_cfg([42])).run()
        with open(os.path.join(self.tmp, "summary.csv")) as f:
            header = f.readline()
        self.assertIn("best_run", header)

    @patch(_TRAIN_PATH)
    def test_exactly_one_representative_marked(self, mock_train):
        import csv as csv_mod

        mock_train.side_effect = [
            _make_mock_trainer({"val/iou": 0.7}, best_checkpoint_path="/c0.ckpt"),
            _make_mock_trainer({"val/iou": 0.8}, best_checkpoint_path="/c1.ckpt"),
            _make_mock_trainer({"val/iou": 0.6}, best_checkpoint_path="/c2.ckpt"),
        ]
        from pytorch_segmentation_models_trainer.tools.experiments_runner.experiments_runner import (
            ExperimentsRunner,
        )

        ExperimentsRunner(self._make_cfg([42, 101, 28])).run()
        with open(os.path.join(self.tmp, "summary.csv")) as f:
            rows = list(csv_mod.DictReader(f))
        run_rows = [r for r in rows if r["run"] not in ("mean", "std")]
        marked = [r for r in run_rows if r.get("representative") == "*"]
        self.assertEqual(len(marked), 1)

    @patch(_TRAIN_PATH)
    def test_exactly_one_best_run_marked(self, mock_train):
        import csv as csv_mod

        mock_train.side_effect = [
            _make_mock_trainer({"val/iou": 0.7}, best_checkpoint_path="/c0.ckpt"),
            _make_mock_trainer({"val/iou": 0.8}, best_checkpoint_path="/c1.ckpt"),
            _make_mock_trainer({"val/iou": 0.6}, best_checkpoint_path="/c2.ckpt"),
        ]
        from pytorch_segmentation_models_trainer.tools.experiments_runner.experiments_runner import (
            ExperimentsRunner,
        )

        ExperimentsRunner(self._make_cfg([42, 101, 28])).run()
        with open(os.path.join(self.tmp, "summary.csv")) as f:
            rows = list(csv_mod.DictReader(f))
        run_rows = [r for r in rows if r["run"] not in ("mean", "std")]
        marked = [r for r in run_rows if r.get("best_run") == "*"]
        self.assertEqual(len(marked), 1)

    @patch(_TRAIN_PATH)
    def test_best_run_marks_highest_metric_run(self, mock_train):
        import csv as csv_mod

        mock_train.side_effect = [
            _make_mock_trainer({"val/iou": 0.7}, best_checkpoint_path="/c0.ckpt"),
            _make_mock_trainer({"val/iou": 0.9}, best_checkpoint_path="/c1.ckpt"),
            _make_mock_trainer({"val/iou": 0.6}, best_checkpoint_path="/c2.ckpt"),
        ]
        from pytorch_segmentation_models_trainer.tools.experiments_runner.experiments_runner import (
            ExperimentsRunner,
        )

        ExperimentsRunner(self._make_cfg([42, 101, 28])).run()
        with open(os.path.join(self.tmp, "summary.csv")) as f:
            rows = list(csv_mod.DictReader(f))
        run_rows = [r for r in rows if r["run"] not in ("mean", "std")]
        best_row = next(r for r in run_rows if r.get("best_run") == "*")
        self.assertEqual(int(best_row["run"]), 1)

    @patch(_TRAIN_PATH)
    def test_epochs_trained_value_correct(self, mock_train):
        import csv as csv_mod

        mock_train.return_value = _make_mock_trainer({"val/loss": 0.5}, current_epoch=9)
        from pytorch_segmentation_models_trainer.tools.experiments_runner.experiments_runner import (
            ExperimentsRunner,
        )

        ExperimentsRunner(self._make_cfg([42])).run()
        with open(os.path.join(self.tmp, "summary.csv")) as f:
            rows = list(csv_mod.DictReader(f))
        run_rows = [r for r in rows if r["run"] not in ("mean", "std")]
        self.assertEqual(run_rows[0]["epochs_trained"], "10")
