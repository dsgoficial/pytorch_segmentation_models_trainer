# -*- coding: utf-8 -*-
"""
Tests for PipelineRunner (TDD — written before implementation).

Run with:
    uv run pytest tests/test_pipeline_runner.py -v --tb=short
"""

import json
import os
from unittest.mock import MagicMock, patch

from omegaconf import OmegaConf

from tests.utils import BasicTestCase

_RUN_STEP_PATH = (
    "pytorch_segmentation_models_trainer.tools.experiments_runner"
    ".pipeline_runner.PipelineRunner._run_step"
)
_PREFLIGHT_PATH = (
    "pytorch_segmentation_models_trainer.tools.experiments_runner"
    ".pipeline_runner.PipelineRunner._run_preflight_check"
)


def _make_step(name, config_dir="conf/examples", config_name="some_config", **kw):
    d = {"name": name, "config_dir": config_dir, "config_name": config_name}
    d.update(kw)
    return d


def _make_pipeline_cfg(
    steps,
    output_base_dir,
    resume=True,
    overwrite=False,
    on_error="stop",
    preflight=False,
    resources=None,
):
    d = {
        "mode": "run-pipeline",
        "pipeline": {
            "steps": steps,
            "output_base_dir": output_base_dir,
            "resume": resume,
            "overwrite": overwrite,
            "on_error": on_error,
            "preflight": preflight,
        },
    }
    if resources is not None:
        d["pipeline"]["resources"] = resources
    return OmegaConf.create(d)


def _gpus(*vram_by_id):
    """Build a `resources.gpus` list: _gpus(24, 24) -> ids 0,1 with 24GB each."""
    return {
        "gpus": [{"id": i, "vram_gb": v} for i, v in enumerate(vram_by_id)],
        "max_parallel_cpu_steps": 1,
    }


def _ok(cp=None):
    """Fake subprocess.CompletedProcess-like success result."""
    m = MagicMock()
    m.returncode = 0
    m.stderr = ""
    return m


def _fail(stderr="boom"):
    m = MagicMock()
    m.returncode = 1
    m.stderr = stderr
    return m


class TestPipelineValidation(BasicTestCase):
    def _import(self):
        from pytorch_segmentation_models_trainer.tools.experiments_runner.pipeline_runner import (
            PipelineRunner,
        )

        return PipelineRunner

    def test_raises_on_empty_steps(self):
        PipelineRunner = self._import()
        cfg = _make_pipeline_cfg([], "/tmp/pipeline")
        with self.assertRaises(ValueError):
            PipelineRunner(cfg)

    def test_raises_on_duplicate_names(self):
        PipelineRunner = self._import()
        cfg = _make_pipeline_cfg([_make_step("a"), _make_step("a")], "/tmp/pipeline")
        with self.assertRaises(ValueError):
            PipelineRunner(cfg)

    def test_raises_on_missing_config_dir(self):
        PipelineRunner = self._import()
        step = _make_step("a")
        step["config_dir"] = ""
        cfg = _make_pipeline_cfg([step], "/tmp/pipeline")
        with self.assertRaises(ValueError):
            PipelineRunner(cfg)

    def test_raises_on_missing_config_name(self):
        PipelineRunner = self._import()
        step = _make_step("a")
        step["config_name"] = ""
        cfg = _make_pipeline_cfg([step], "/tmp/pipeline")
        with self.assertRaises(ValueError):
            PipelineRunner(cfg)

    def test_command_step_does_not_need_config_dir_or_name(self):
        PipelineRunner = self._import()
        step = {
            "name": "slico_gc",
            "command": ["pytorch-smt-tools", "build-slico-corrected-masks", "x.yaml"],
        }
        cfg = _make_pipeline_cfg([step], "/tmp/pipeline")
        runner = PipelineRunner(cfg)
        self.assertIsNotNone(runner)

    def test_raises_on_invalid_on_error(self):
        PipelineRunner = self._import()
        cfg = _make_pipeline_cfg([_make_step("a")], "/tmp/pipeline", on_error="retry")
        with self.assertRaises(ValueError):
            PipelineRunner(cfg)

    def test_valid_config_constructs(self):
        PipelineRunner = self._import()
        cfg = _make_pipeline_cfg([_make_step("a"), _make_step("b")], "/tmp/pipeline")
        runner = PipelineRunner(cfg)
        self.assertIsNotNone(runner)


class TestBuildCommand(BasicTestCase):
    def test_command_has_expected_shape(self):
        from pytorch_segmentation_models_trainer.tools.experiments_runner.pipeline_runner import (
            PipelineRunner,
        )

        cfg = _make_pipeline_cfg(
            [_make_step("a", overrides=["seed=42", "hyperparameters.epochs=5"])],
            "/tmp/pipeline",
        )
        runner = PipelineRunner(cfg)
        cmd = runner._build_command(cfg.pipeline.steps[0])
        self.assertEqual(
            cmd,
            [
                "pytorch-smt",
                "--config-dir",
                "conf/examples",
                "--config-name",
                "some_config",
                "seed=42",
                "hyperparameters.epochs=5",
            ],
        )

    def test_command_step_used_verbatim(self):
        """A `command` step (e.g. pytorch-smt-tools) bypasses the
        --config-dir/--config-name Hydra shape entirely."""
        from pytorch_segmentation_models_trainer.tools.experiments_runner.pipeline_runner import (
            PipelineRunner,
        )

        step = {
            "name": "slico_gc",
            "command": [
                "pytorch-smt-tools",
                "build-slico-corrected-masks",
                "conf/examples/slico_label_correction.yaml",
            ],
        }
        cfg = _make_pipeline_cfg([step], "/tmp/pipeline")
        runner = PipelineRunner(cfg)
        cmd = runner._build_command(cfg.pipeline.steps[0])
        self.assertEqual(
            cmd,
            [
                "pytorch-smt-tools",
                "build-slico-corrected-masks",
                "conf/examples/slico_label_correction.yaml",
            ],
        )

    def test_command_step_appends_overrides_verbatim(self):
        from pytorch_segmentation_models_trainer.tools.experiments_runner.pipeline_runner import (
            PipelineRunner,
        )

        step = {
            "name": "slico_gc",
            "command": ["pytorch-smt-tools", "build-slico-corrected-masks", "x.yaml"],
            "overrides": ["--start-idx", "0", "--end-idx", "999"],
        }
        cfg = _make_pipeline_cfg([step], "/tmp/pipeline")
        runner = PipelineRunner(cfg)
        cmd = runner._build_command(cfg.pipeline.steps[0])
        self.assertEqual(cmd[-4:], ["--start-idx", "0", "--end-idx", "999"])

    def test_custom_pytorch_smt_bin(self):
        from pytorch_segmentation_models_trainer.tools.experiments_runner.pipeline_runner import (
            PipelineRunner,
        )

        steps = [_make_step("a")]
        cfg = OmegaConf.create(
            {
                "pipeline": {
                    "steps": steps,
                    "output_base_dir": "/tmp/pipeline",
                    "pytorch_smt_bin": "/opt/venv/bin/pytorch-smt",
                }
            }
        )
        runner = PipelineRunner(cfg)
        cmd = runner._build_command(cfg.pipeline.steps[0])
        self.assertEqual(cmd[0], "/opt/venv/bin/pytorch-smt")


class TestRunSequential(BasicTestCase):
    def setUp(self):
        super().setUp()
        self.tmp = self.make_temp_dir()

    @patch(_RUN_STEP_PATH)
    def test_runs_all_steps_in_order(self, mock_run):
        mock_run.return_value = 0
        from pytorch_segmentation_models_trainer.tools.experiments_runner.pipeline_runner import (
            PipelineRunner,
        )

        cfg = _make_pipeline_cfg([_make_step("first"), _make_step("second")], self.tmp)
        results = PipelineRunner(cfg).run()

        self.assertEqual(mock_run.call_count, 2)
        self.assertEqual([r["name"] for r in results], ["first", "second"])
        self.assertTrue(all(r["status"] == "done" for r in results))

    @patch(_RUN_STEP_PATH)
    def test_disabled_step_is_skipped_entirely(self, mock_run):
        mock_run.return_value = 0
        from pytorch_segmentation_models_trainer.tools.experiments_runner.pipeline_runner import (
            PipelineRunner,
        )

        cfg = _make_pipeline_cfg(
            [_make_step("a", enabled=False), _make_step("b")], self.tmp
        )
        results = PipelineRunner(cfg).run()

        self.assertEqual(mock_run.call_count, 1)
        self.assertEqual([r["name"] for r in results], ["b"])

    @patch(_RUN_STEP_PATH)
    def test_pipeline_state_written_after_each_step(self, mock_run):
        mock_run.return_value = 0
        from pytorch_segmentation_models_trainer.tools.experiments_runner.pipeline_runner import (
            PipelineRunner,
        )

        cfg = _make_pipeline_cfg([_make_step("a"), _make_step("b")], self.tmp)
        PipelineRunner(cfg).run()

        with open(os.path.join(self.tmp, "pipeline_state.json")) as f:
            state = json.load(f)
        self.assertEqual(state["steps"]["a"]["status"], "done")
        self.assertEqual(state["steps"]["b"]["status"], "done")

    @patch(_RUN_STEP_PATH)
    def test_pipeline_log_records_steps(self, mock_run):
        mock_run.return_value = 0
        from pytorch_segmentation_models_trainer.tools.experiments_runner.pipeline_runner import (
            PipelineRunner,
        )

        cfg = _make_pipeline_cfg([_make_step("alpha")], self.tmp)
        PipelineRunner(cfg).run()

        with open(os.path.join(self.tmp, "pipeline.log")) as f:
            content = f.read()
        self.assertIn("alpha", content)


class TestResumeAndOverwrite(BasicTestCase):
    def setUp(self):
        super().setUp()
        self.tmp = self.make_temp_dir()

    def _write_state(self, steps_status):
        os.makedirs(self.tmp, exist_ok=True)
        state = {
            "steps": {
                name: {
                    "name": name,
                    "status": status,
                    "exit_code": 0 if status == "done" else 1,
                    "duration_s": 1.0,
                    "log_path": os.path.join(self.tmp, "logs", f"{name}.log"),
                }
                for name, status in steps_status.items()
            }
        }
        with open(os.path.join(self.tmp, "pipeline_state.json"), "w") as f:
            json.dump(state, f)

    @patch(_RUN_STEP_PATH)
    def test_resume_default_skips_done_step(self, mock_run):
        mock_run.return_value = 0
        from pytorch_segmentation_models_trainer.tools.experiments_runner.pipeline_runner import (
            PipelineRunner,
        )

        self._write_state({"a": "done"})
        cfg = _make_pipeline_cfg([_make_step("a"), _make_step("b")], self.tmp)
        results = PipelineRunner(cfg).run()

        self.assertEqual(mock_run.call_count, 1)  # only "b" executed
        self.assertEqual(len(results), 2)

    @patch(_RUN_STEP_PATH)
    def test_resume_false_reruns_everything(self, mock_run):
        mock_run.return_value = 0
        from pytorch_segmentation_models_trainer.tools.experiments_runner.pipeline_runner import (
            PipelineRunner,
        )

        self._write_state({"a": "done"})
        cfg = _make_pipeline_cfg(
            [_make_step("a"), _make_step("b")], self.tmp, resume=False
        )
        PipelineRunner(cfg).run()

        self.assertEqual(mock_run.call_count, 2)

    @patch(_RUN_STEP_PATH)
    def test_overwrite_true_forces_done_step(self, mock_run):
        mock_run.return_value = 0
        from pytorch_segmentation_models_trainer.tools.experiments_runner.pipeline_runner import (
            PipelineRunner,
        )

        self._write_state({"a": "done"})
        cfg = _make_pipeline_cfg([_make_step("a")], self.tmp, overwrite=True)
        PipelineRunner(cfg).run()

        self.assertEqual(mock_run.call_count, 1)

    @patch(_RUN_STEP_PATH)
    def test_overwrite_list_forces_only_named_step(self, mock_run):
        mock_run.return_value = 0
        from pytorch_segmentation_models_trainer.tools.experiments_runner.pipeline_runner import (
            PipelineRunner,
        )

        self._write_state({"a": "done", "b": "done"})
        cfg = _make_pipeline_cfg(
            [_make_step("a"), _make_step("b")], self.tmp, overwrite=["b"]
        )
        results = PipelineRunner(cfg).run()

        self.assertEqual(mock_run.call_count, 1)
        b_result = [r for r in results if r["name"] == "b"][0]
        self.assertEqual(b_result["status"], "done")

    @patch(_RUN_STEP_PATH)
    def test_per_step_overwrite_true_overrides_pipeline_false(self, mock_run):
        mock_run.return_value = 0
        from pytorch_segmentation_models_trainer.tools.experiments_runner.pipeline_runner import (
            PipelineRunner,
        )

        self._write_state({"a": "done"})
        cfg = _make_pipeline_cfg(
            [_make_step("a", overwrite=True)], self.tmp, overwrite=False
        )
        PipelineRunner(cfg).run()

        self.assertEqual(mock_run.call_count, 1)

    @patch(_RUN_STEP_PATH)
    def test_per_step_overwrite_false_blocks_pipeline_level_true(self, mock_run):
        mock_run.return_value = 0
        from pytorch_segmentation_models_trainer.tools.experiments_runner.pipeline_runner import (
            PipelineRunner,
        )

        self._write_state({"a": "done"})
        cfg = _make_pipeline_cfg(
            [_make_step("a", overwrite=False)], self.tmp, overwrite=True
        )
        PipelineRunner(cfg).run()

        self.assertEqual(mock_run.call_count, 0)  # step opts out of the force


class TestOnError(BasicTestCase):
    def setUp(self):
        super().setUp()
        self.tmp = self.make_temp_dir()

    @patch(_RUN_STEP_PATH)
    def test_stop_raises_and_skips_later_steps(self, mock_run):
        mock_run.side_effect = [1, 0]  # first step fails
        from pytorch_segmentation_models_trainer.tools.experiments_runner.pipeline_runner import (
            PipelineRunner,
        )

        cfg = _make_pipeline_cfg(
            [_make_step("a"), _make_step("b")], self.tmp, on_error="stop"
        )
        with self.assertRaises(RuntimeError):
            PipelineRunner(cfg).run()

        self.assertEqual(mock_run.call_count, 1)  # "b" never attempted

    @patch(_RUN_STEP_PATH)
    def test_continue_runs_later_steps_and_records_failure(self, mock_run):
        mock_run.side_effect = [1, 0]
        from pytorch_segmentation_models_trainer.tools.experiments_runner.pipeline_runner import (
            PipelineRunner,
        )

        cfg = _make_pipeline_cfg(
            [_make_step("a"), _make_step("b")], self.tmp, on_error="continue"
        )
        results = PipelineRunner(cfg).run()

        self.assertEqual(mock_run.call_count, 2)
        self.assertEqual(results[0]["status"], "failed")
        self.assertEqual(results[1]["status"], "done")

    @patch(_RUN_STEP_PATH)
    def test_failed_step_can_be_retried_via_resume(self, mock_run):
        """After a stop-on-error abort, re-running with resume retries only
        the failed step (and anything after it), not steps already done."""
        from pytorch_segmentation_models_trainer.tools.experiments_runner.pipeline_runner import (
            PipelineRunner,
        )

        mock_run.side_effect = [0, 1]
        cfg = _make_pipeline_cfg(
            [_make_step("a"), _make_step("b")], self.tmp, on_error="stop"
        )
        with self.assertRaises(RuntimeError):
            PipelineRunner(cfg).run()
        self.assertEqual(mock_run.call_count, 2)

        mock_run.side_effect = None
        mock_run.return_value = 0
        mock_run.reset_mock()
        results = PipelineRunner(cfg).run()

        self.assertEqual(mock_run.call_count, 1)  # only "b" retried
        self.assertEqual([r["status"] for r in results], ["done", "done"])


class TestPreflight(BasicTestCase):
    def setUp(self):
        super().setUp()
        self.tmp = self.make_temp_dir()

    @patch(_RUN_STEP_PATH)
    @patch(_PREFLIGHT_PATH)
    def test_preflight_runs_before_any_step(self, mock_preflight, mock_run):
        mock_preflight.return_value = _ok()
        mock_run.return_value = 0
        from pytorch_segmentation_models_trainer.tools.experiments_runner.pipeline_runner import (
            PipelineRunner,
        )

        cfg = _make_pipeline_cfg(
            [_make_step("a"), _make_step("b")], self.tmp, preflight=True
        )
        PipelineRunner(cfg).run()

        self.assertEqual(mock_preflight.call_count, 2)
        self.assertEqual(mock_run.call_count, 2)

    @patch(_RUN_STEP_PATH)
    @patch(_PREFLIGHT_PATH)
    def test_preflight_failure_blocks_all_execution(self, mock_preflight, mock_run):
        mock_preflight.side_effect = [_fail("bad key"), _ok()]
        from pytorch_segmentation_models_trainer.tools.experiments_runner.pipeline_runner import (
            PipelineRunner,
        )

        cfg = _make_pipeline_cfg(
            [_make_step("a"), _make_step("b")], self.tmp, preflight=True
        )
        with self.assertRaises(RuntimeError):
            PipelineRunner(cfg).run()

        self.assertEqual(mock_run.call_count, 0)

    @patch(_RUN_STEP_PATH)
    @patch(_PREFLIGHT_PATH)
    def test_preflight_disabled_by_default_flag_skips_check(
        self, mock_preflight, mock_run
    ):
        mock_run.return_value = 0
        from pytorch_segmentation_models_trainer.tools.experiments_runner.pipeline_runner import (
            PipelineRunner,
        )

        cfg = _make_pipeline_cfg([_make_step("a")], self.tmp, preflight=False)
        PipelineRunner(cfg).run()

        mock_preflight.assert_not_called()

    @patch(_RUN_STEP_PATH)
    @patch(_PREFLIGHT_PATH)
    def test_command_step_excluded_from_preflight(self, mock_preflight, mock_run):
        """pytorch-smt-tools steps don't understand mode=validate-config."""
        mock_preflight.return_value = _ok()
        mock_run.return_value = 0
        from pytorch_segmentation_models_trainer.tools.experiments_runner.pipeline_runner import (
            PipelineRunner,
        )

        steps = [
            {
                "name": "slico_gc",
                "command": [
                    "pytorch-smt-tools",
                    "build-slico-corrected-masks",
                    "x.yaml",
                ],
            },
            _make_step("b"),
        ]
        cfg = _make_pipeline_cfg(steps, self.tmp, preflight=True)
        PipelineRunner(cfg).run()

        self.assertEqual(mock_preflight.call_count, 1)  # only "b"
        self.assertEqual(mock_run.call_count, 2)  # both still actually run

    @patch(_RUN_STEP_PATH)
    @patch(_PREFLIGHT_PATH)
    def test_disabled_step_excluded_from_preflight(self, mock_preflight, mock_run):
        mock_preflight.return_value = _ok()
        mock_run.return_value = 0
        from pytorch_segmentation_models_trainer.tools.experiments_runner.pipeline_runner import (
            PipelineRunner,
        )

        cfg = _make_pipeline_cfg(
            [_make_step("a", enabled=False), _make_step("b")],
            self.tmp,
            preflight=True,
        )
        PipelineRunner(cfg).run()

        self.assertEqual(mock_preflight.call_count, 1)


class TestDagValidation(BasicTestCase):
    def _import(self):
        from pytorch_segmentation_models_trainer.tools.experiments_runner.pipeline_runner import (
            PipelineRunner,
        )

        return PipelineRunner

    def test_raises_on_unknown_depends_on(self):
        PipelineRunner = self._import()
        cfg = _make_pipeline_cfg(
            [_make_step("a", depends_on=["ghost"])], "/tmp/pipeline"
        )
        with self.assertRaises(ValueError):
            PipelineRunner(cfg)

    def test_raises_on_direct_cycle(self):
        PipelineRunner = self._import()
        cfg = _make_pipeline_cfg(
            [
                _make_step("a", depends_on=["b"]),
                _make_step("b", depends_on=["a"]),
            ],
            "/tmp/pipeline",
        )
        with self.assertRaises(ValueError):
            PipelineRunner(cfg)

    def test_raises_on_self_cycle(self):
        PipelineRunner = self._import()
        cfg = _make_pipeline_cfg([_make_step("a", depends_on=["a"])], "/tmp/pipeline")
        with self.assertRaises(ValueError):
            PipelineRunner(cfg)

    def test_valid_dag_constructs(self):
        PipelineRunner = self._import()
        cfg = _make_pipeline_cfg(
            [
                _make_step("a"),
                _make_step("b", depends_on=["a"]),
                _make_step("c", depends_on=["a", "b"]),
            ],
            "/tmp/pipeline",
        )
        runner = PipelineRunner(cfg)
        self.assertIsNotNone(runner)


class TestResourceValidation(BasicTestCase):
    def _import(self):
        from pytorch_segmentation_models_trainer.tools.experiments_runner.pipeline_runner import (
            PipelineRunner,
        )

        return PipelineRunner

    def test_raises_when_gpu_count_exceeds_pool(self):
        PipelineRunner = self._import()
        cfg = _make_pipeline_cfg(
            [_make_step("a", gpus=2)], "/tmp/pipeline", resources=_gpus(24)
        )
        with self.assertRaises(ValueError):
            PipelineRunner(cfg)

    def test_raises_when_vram_exceeds_every_gpu(self):
        PipelineRunner = self._import()
        cfg = _make_pipeline_cfg(
            [_make_step("a", gpus=1, vram_gb=100)],
            "/tmp/pipeline",
            resources=_gpus(24, 24),
        )
        with self.assertRaises(ValueError):
            PipelineRunner(cfg)

    def test_raises_when_gpu_requested_but_no_resources_configured(self):
        PipelineRunner = self._import()
        cfg = _make_pipeline_cfg([_make_step("a", gpus=1)], "/tmp/pipeline")
        with self.assertRaises(ValueError):
            PipelineRunner(cfg)

    def test_disabled_step_infeasible_request_does_not_raise(self):
        PipelineRunner = self._import()
        cfg = _make_pipeline_cfg(
            [_make_step("a", gpus=8, enabled=False), _make_step("b")],
            "/tmp/pipeline",
        )
        runner = PipelineRunner(cfg)
        self.assertIsNotNone(runner)

    def test_valid_resource_request_constructs(self):
        PipelineRunner = self._import()
        cfg = _make_pipeline_cfg(
            [_make_step("a", gpus=1, vram_gb=12), _make_step("b", gpus=2)],
            "/tmp/pipeline",
            resources=_gpus(24, 24),
        )
        runner = PipelineRunner(cfg)
        self.assertIsNotNone(runner)

    def test_cpu_only_step_never_needs_resources(self):
        PipelineRunner = self._import()
        cfg = _make_pipeline_cfg([_make_step("a")], "/tmp/pipeline")
        runner = PipelineRunner(cfg)
        self.assertIsNotNone(runner)


class TestCudaEnv(BasicTestCase):
    def test_none_for_cpu_reservation(self):
        from pytorch_segmentation_models_trainer.tools.experiments_runner.pipeline_runner import (
            PipelineRunner,
        )

        cfg = _make_pipeline_cfg([_make_step("a")], "/tmp/pipeline")
        runner = PipelineRunner(cfg)
        self.assertIsNone(runner._cuda_env_for({"type": "cpu"}))

    def test_single_gpu_reservation(self):
        from pytorch_segmentation_models_trainer.tools.experiments_runner.pipeline_runner import (
            PipelineRunner,
        )

        cfg = _make_pipeline_cfg([_make_step("a")], "/tmp/pipeline")
        runner = PipelineRunner(cfg)
        env = runner._cuda_env_for({"type": "gpu", "ids": [1], "full": True})
        self.assertEqual(env, {"CUDA_VISIBLE_DEVICES": "1"})

    def test_multi_gpu_reservation_joins_ids(self):
        from pytorch_segmentation_models_trainer.tools.experiments_runner.pipeline_runner import (
            PipelineRunner,
        )

        cfg = _make_pipeline_cfg([_make_step("a")], "/tmp/pipeline")
        runner = PipelineRunner(cfg)
        env = runner._cuda_env_for({"type": "gpu", "ids": [0, 2], "full": True})
        self.assertEqual(env, {"CUDA_VISIBLE_DEVICES": "0,2"})


class TestParallelScheduling(BasicTestCase):
    def setUp(self):
        super().setUp()
        self.tmp = self.make_temp_dir()

    def _concurrency_tracker(self, sleep_s=0.05):
        import threading as _threading
        import time as _time

        state = {"current": 0, "peak": 0}
        lock = _threading.Lock()

        def fake_run(cmd, log_path, env=None):
            with lock:
                state["current"] += 1
                state["peak"] = max(state["peak"], state["current"])
            _time.sleep(sleep_s)
            with lock:
                state["current"] -= 1
            return 0

        return state, fake_run

    @patch(_RUN_STEP_PATH)
    def test_independent_steps_run_concurrently_with_enough_cpu_slots(self, mock_run):
        state, fake_run = self._concurrency_tracker()
        mock_run.side_effect = fake_run
        from pytorch_segmentation_models_trainer.tools.experiments_runner.pipeline_runner import (
            PipelineRunner,
        )

        resources = {"gpus": [], "max_parallel_cpu_steps": 2}
        cfg = _make_pipeline_cfg(
            [_make_step("a"), _make_step("b")], self.tmp, resources=resources
        )
        results = PipelineRunner(cfg).run()

        self.assertEqual(state["peak"], 2)
        self.assertTrue(all(r["status"] == "done" for r in results))

    @patch(_RUN_STEP_PATH)
    def test_default_cpu_capacity_serializes_independent_steps(self, mock_run):
        """No `resources` configured -> default max_parallel_cpu_steps=1,
        so even independent (no depends_on) steps run one at a time —
        matches the original MVP's sequential behaviour exactly."""
        state, fake_run = self._concurrency_tracker()
        mock_run.side_effect = fake_run
        from pytorch_segmentation_models_trainer.tools.experiments_runner.pipeline_runner import (
            PipelineRunner,
        )

        cfg = _make_pipeline_cfg([_make_step("a"), _make_step("b")], self.tmp)
        PipelineRunner(cfg).run()

        self.assertEqual(state["peak"], 1)

    @patch(_RUN_STEP_PATH)
    def test_depends_on_blocks_start_until_dependency_done(self, mock_run):
        state, fake_run = self._concurrency_tracker()
        mock_run.side_effect = fake_run
        from pytorch_segmentation_models_trainer.tools.experiments_runner.pipeline_runner import (
            PipelineRunner,
        )

        resources = {"gpus": [], "max_parallel_cpu_steps": 2}
        cfg = _make_pipeline_cfg(
            [_make_step("a"), _make_step("b", depends_on=["a"])],
            self.tmp,
            resources=resources,
        )
        results = PipelineRunner(cfg).run()

        # Even with 2 CPU slots free, "b" must wait for "a" -> never overlap.
        self.assertEqual(state["peak"], 1)
        self.assertEqual([r["status"] for r in results], ["done", "done"])

    @patch(_RUN_STEP_PATH)
    def test_gpu_bin_packing_allows_two_small_steps_on_one_gpu(self, mock_run):
        state, fake_run = self._concurrency_tracker()
        mock_run.side_effect = fake_run
        from pytorch_segmentation_models_trainer.tools.experiments_runner.pipeline_runner import (
            PipelineRunner,
        )

        cfg = _make_pipeline_cfg(
            [
                _make_step("a", gpus=1, vram_gb=8),
                _make_step("b", gpus=1, vram_gb=8),
            ],
            self.tmp,
            resources=_gpus(24),
        )
        results = PipelineRunner(cfg).run()

        self.assertEqual(state["peak"], 2)
        self.assertTrue(all(r["status"] == "done" for r in results))

    @patch(_RUN_STEP_PATH)
    def test_gpu_vram_contention_serializes_steps_on_same_gpu(self, mock_run):
        state, fake_run = self._concurrency_tracker()
        mock_run.side_effect = fake_run
        from pytorch_segmentation_models_trainer.tools.experiments_runner.pipeline_runner import (
            PipelineRunner,
        )

        cfg = _make_pipeline_cfg(
            [
                _make_step("a", gpus=1, vram_gb=20),
                _make_step("b", gpus=1, vram_gb=20),
            ],
            self.tmp,
            resources=_gpus(24),  # only one fits at a time (20+20 > 24)
        )
        results = PipelineRunner(cfg).run()

        self.assertEqual(state["peak"], 1)
        self.assertTrue(all(r["status"] == "done" for r in results))

    @patch(_RUN_STEP_PATH)
    def test_multi_gpu_step_reserves_exclusively(self, mock_run):
        """A gpus=2 step on a 2-GPU pool must block a concurrent gpus=1
        step — it needs both GPUs entirely to itself."""
        state, fake_run = self._concurrency_tracker()
        mock_run.side_effect = fake_run
        from pytorch_segmentation_models_trainer.tools.experiments_runner.pipeline_runner import (
            PipelineRunner,
        )

        cfg = _make_pipeline_cfg(
            [
                _make_step("ddp", gpus=2),
                _make_step("solo", gpus=1),
            ],
            self.tmp,
            resources=_gpus(24, 24),
        )
        results = PipelineRunner(cfg).run()

        self.assertEqual(state["peak"], 1)
        self.assertTrue(all(r["status"] == "done" for r in results))

    @patch(_RUN_STEP_PATH)
    def test_two_gpu_steps_run_concurrently_on_distinct_gpus(self, mock_run):
        state, fake_run = self._concurrency_tracker()
        mock_run.side_effect = fake_run
        from pytorch_segmentation_models_trainer.tools.experiments_runner.pipeline_runner import (
            PipelineRunner,
        )

        cfg = _make_pipeline_cfg(
            [
                _make_step("a", gpus=1),
                _make_step("b", gpus=1),
            ],
            self.tmp,
            resources=_gpus(24, 24),
        )
        results = PipelineRunner(cfg).run()

        self.assertEqual(state["peak"], 2)
        self.assertTrue(all(r["status"] == "done" for r in results))

    @patch(_RUN_STEP_PATH)
    def test_dependent_step_on_failed_dependency_is_skipped(self, mock_run):
        mock_run.return_value = 1  # everything fails
        from pytorch_segmentation_models_trainer.tools.experiments_runner.pipeline_runner import (
            PipelineRunner,
        )

        resources = {"gpus": [], "max_parallel_cpu_steps": 2}
        cfg = _make_pipeline_cfg(
            [
                _make_step("a"),
                _make_step("b", depends_on=["a"]),
            ],
            self.tmp,
            on_error="continue",
            resources=resources,
        )
        results = PipelineRunner(cfg).run()

        by_name = {r["name"]: r for r in results}
        self.assertEqual(by_name["a"]["status"], "failed")
        self.assertEqual(by_name["b"]["status"], "skipped")
        self.assertEqual(mock_run.call_count, 1)  # "b" never actually launched

    @patch(_RUN_STEP_PATH)
    def test_independent_branch_still_completes_when_sibling_fails(self, mock_run):
        """on_error=stop halts new admissions, but a branch independent of
        the failed step that was already running gets to finish."""
        state, fake_run = self._concurrency_tracker(sleep_s=0.05)

        def mixed(cmd, log_path, env=None):
            if "_fail_" in log_path:
                return 1
            return fake_run(cmd, log_path, env=env)

        mock_run.side_effect = mixed
        from pytorch_segmentation_models_trainer.tools.experiments_runner.pipeline_runner import (
            PipelineRunner,
        )

        resources = {"gpus": [], "max_parallel_cpu_steps": 2}
        cfg = _make_pipeline_cfg(
            [
                {"name": "_fail_a", "config_dir": "c", "config_name": "n"},
                {"name": "ok_b", "config_dir": "c", "config_name": "n"},
            ],
            self.tmp,
            resources=resources,
        )
        with self.assertRaises(RuntimeError):
            PipelineRunner(cfg).run()

        with open(os.path.join(self.tmp, "pipeline_state.json")) as f:
            saved = json.load(f)
        self.assertEqual(saved["steps"]["_fail_a"]["status"], "failed")
        self.assertEqual(saved["steps"]["ok_b"]["status"], "done")
