# -*- coding: utf-8 -*-
"""Tests for resuming interrupted runs (``resumable: true``).

Covers ``utils.resume_utils`` (per-epoch ``last`` checkpoint, logger pinning,
AIO2 hand-off state), ``train._fit_single`` / ``train._fit_aio2`` resuming
real tiny CPU fits after a simulated crash, ``train()`` cleanup, and the
``ExperimentsRunner`` keeping an interrupted run's directory.
"""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import pytorch_lightning as pl
import torch
from omegaconf import OmegaConf
from pytorch_lightning import Callback
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader, TensorDataset

from pytorch_segmentation_models_trainer.tools.experiments_runner.experiments_runner import (
    ExperimentsRunner,
)
from pytorch_segmentation_models_trainer.train import _fit_aio2, _fit_single
from pytorch_segmentation_models_trainer.utils import resume_utils
from pytorch_segmentation_models_trainer.utils.resume_utils import (
    ResumePoint,
    clear_resume_state,
    has_resume_state,
    load_resume_checkpoint,
    logger_from_checkpoint,
    read_aio2_state,
    resume_checkpoint_callback,
    resume_dir,
    resume_logger,
    write_aio2_state,
)

from tests.test_aio2_two_phase import _ScriptedValMeanTeacher, _cfg, _val_best_cfg


class _Crash(Callback):
    """Raise at the start of ``epoch`` (0-indexed) — a simulated SIGTERM."""

    def __init__(self, epoch: int) -> None:
        self.epoch = epoch

    def on_train_epoch_start(self, trainer, pl_module):
        if trainer.current_epoch == self.epoch:
            raise RuntimeError(f"simulated crash at epoch {self.epoch}")


class _EpochRecorder(Callback):
    def __init__(self) -> None:
        self.epochs = []

    def on_train_epoch_start(self, trainer, pl_module):
        self.epochs.append(trainer.current_epoch)


class _TinyRegressor(pl.LightningModule):
    """Minimal module logging a scripted ``val/score`` (peak at epoch 1)."""

    curve = [0.2, 0.9, 0.5, 0.4, 0.3, 0.2]

    def __init__(self) -> None:
        super().__init__()
        self.layer = torch.nn.Linear(2, 1)

    def training_step(self, batch, batch_idx):
        x, y = batch
        return torch.nn.functional.mse_loss(self.layer(x), y)

    def validation_step(self, batch, batch_idx):
        x, y = batch
        return torch.nn.functional.mse_loss(self.layer(x), y)

    def on_validation_epoch_end(self):
        self.log("val/score", self.curve[self.current_epoch])

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=0.01)

    def _loader(self):
        g = torch.Generator().manual_seed(0)
        return DataLoader(
            TensorDataset(torch.randn(8, 2, generator=g), torch.randn(8, 1, generator=g)),
            batch_size=4,
        )

    def train_dataloader(self):
        return self._loader()

    def val_dataloader(self):
        return self._loader()


def _single_cfg(tmp_path, max_epochs=6, resumable=True):
    return OmegaConf.create(
        {
            "resumable": resumable,
            "pl_trainer": {
                "max_epochs": max_epochs,
                "accelerator": "cpu",
                "devices": 1,
                "default_root_dir": str(tmp_path),
                "enable_progress_bar": False,
                "enable_model_summary": False,
                "num_sanity_val_steps": 0,
            },
        }
    )


def _best_cb():
    return ModelCheckpoint(
        monitor="val/score", mode="max", save_top_k=1, filename="best-{epoch:02d}"
    )


def _pl_cfg(cfg):
    return OmegaConf.to_container(cfg.pl_trainer, resolve=True)


# ---------------------------------------------------------------------------
# resume_utils
# ---------------------------------------------------------------------------


class TestResumeUtils:
    def test_resume_checkpoint_callback_saves_only_last(self, tmp_path):
        cb = resume_checkpoint_callback(tmp_path, "warmup")
        assert isinstance(cb, ModelCheckpoint)
        assert Path(cb.dirpath) == resume_dir(tmp_path)
        assert resume_dir(tmp_path) == tmp_path / "resume_ckpt"
        assert cb.filename == "warmup_last"
        assert cb.save_top_k == 1
        assert cb.monitor is None
        assert not cb.save_last

    def test_load_missing_returns_none(self, tmp_path):
        assert load_resume_checkpoint(tmp_path, "fit") is None

    def test_load_keeps_only_callback_state(self, tmp_path):
        resume_dir(tmp_path).mkdir()
        path = resume_dir(tmp_path) / "fit_last.ckpt"
        torch.save({"state_dict": {"w": torch.ones(3)}, "callbacks": {"k": 1}}, path)
        point = load_resume_checkpoint(tmp_path, "fit")
        assert isinstance(point, ResumePoint)
        assert point.path == path
        assert point.state == {"callbacks": {"k": 1}}

    def test_load_corrupted_warns_and_returns_none(self, tmp_path, caplog):
        resume_dir(tmp_path).mkdir()
        (resume_dir(tmp_path) / "fit_last.ckpt").write_bytes(b"not a checkpoint")
        assert load_resume_checkpoint(tmp_path, "fit") is None
        assert "unreadable" in caplog.text

    def test_logger_from_checkpoint_pins_version_dir(self, tmp_path):
        state = {
            "callbacks": {
                "ModelCheckpoint{'monitor': None}": {"dirpath": str(resume_dir(tmp_path))},
                "EarlyStopping{'monitor': 'x'}": {"wait_count": 1},
                "ModelCheckpoint{'monitor': 'x'}": {
                    "dirpath": str(tmp_path / "lightning_logs" / "version_3" / "checkpoints")
                },
            }
        }
        tb = logger_from_checkpoint(state, tmp_path)
        assert isinstance(tb, TensorBoardLogger)
        assert Path(tb.log_dir) == tmp_path / "lightning_logs" / "version_3"

    def test_logger_from_checkpoint_none_without_version_dir(self, tmp_path):
        state = {
            "callbacks": {
                "ModelCheckpoint{'monitor': 'x'}": {"dirpath": str(tmp_path / "custom")},
                "ModelCheckpoint{'monitor': 'y'}": {"dirpath": None},
            }
        }
        assert logger_from_checkpoint(state, tmp_path) is None
        assert logger_from_checkpoint({}, tmp_path) is None

    def test_resume_logger_only_replaces_default_logger(self, tmp_path):
        state = {
            "callbacks": {
                "ModelCheckpoint{'monitor': 'x'}": {
                    "dirpath": str(tmp_path / "lightning_logs" / "version_0" / "checkpoints")
                }
            }
        }
        point = ResumePoint(tmp_path / "x.ckpt", state)
        assert resume_logger(True, None, tmp_path) is True
        custom = MagicMock()
        assert resume_logger(custom, point, tmp_path) is custom
        assert isinstance(resume_logger(True, point, tmp_path), TensorBoardLogger)
        empty = ResumePoint(tmp_path / "x.ckpt", {})
        assert resume_logger(True, empty, tmp_path) is True

    def test_aio2_state_roundtrip_and_has_state(self, tmp_path):
        assert read_aio2_state(tmp_path) is None
        assert has_resume_state(tmp_path) is False
        write_aio2_state(tmp_path, {"init_ckpt": "/a.ckpt", "info": {"x": 1.0}})
        assert read_aio2_state(tmp_path) == {"init_ckpt": "/a.ckpt", "info": {"x": 1.0}}
        assert has_resume_state(tmp_path) is True

    def test_has_state_with_last_checkpoint_only(self, tmp_path):
        resume_dir(tmp_path).mkdir()
        assert has_resume_state(tmp_path) is False
        (resume_dir(tmp_path) / "fit_last.ckpt").write_bytes(b"x")
        assert has_resume_state(tmp_path) is True

    def test_clear_resume_state(self, tmp_path):
        clear_resume_state(tmp_path)  # no-op when absent
        write_aio2_state(tmp_path, {})
        clear_resume_state(tmp_path)
        assert not resume_dir(tmp_path).exists()


# ---------------------------------------------------------------------------
# Single-phase training (train._fit_single)
# ---------------------------------------------------------------------------


class TestFitSingle:
    def test_not_resumable_is_a_plain_fit(self, tmp_path):
        cfg = _single_cfg(tmp_path, max_epochs=2, resumable=False)
        callbacks = [_best_cb()]
        trainer = _fit_single(cfg, _TinyRegressor(), _pl_cfg(cfg), True, callbacks)
        assert trainer.current_epoch == 2
        assert not resume_dir(tmp_path).exists()
        assert len(callbacks) == 1  # caller's list not mutated

    def test_resumes_after_crash_in_same_version_dir(self, tmp_path):
        cfg = _single_cfg(tmp_path)
        with pytest.raises(RuntimeError, match="simulated crash"):
            _fit_single(cfg, _TinyRegressor(), _pl_cfg(cfg), True, [_best_cb(), _Crash(3)])
        assert (resume_dir(tmp_path) / "fit_last.ckpt").exists()

        recorder = _EpochRecorder()
        trainer = _fit_single(
            cfg, _TinyRegressor(), _pl_cfg(cfg), True, [_best_cb(), recorder]
        )
        assert recorder.epochs == [3, 4, 5]  # epochs 0-2 not repeated
        assert trainer.current_epoch == 6
        best = trainer.checkpoint_callback
        # best epoch (1) came from before the crash and survived the resume:
        assert best.best_model_score == pytest.approx(0.9)
        assert Path(best.best_model_path).name == "best-epoch=01.ckpt"
        assert "version_0" in best.best_model_path
        assert not (tmp_path / "lightning_logs" / "version_1").exists()

    def test_custom_logger_is_kept_on_resume(self, tmp_path):
        cfg = _single_cfg(tmp_path)
        with pytest.raises(RuntimeError):
            _fit_single(cfg, _TinyRegressor(), _pl_cfg(cfg), True, [_best_cb(), _Crash(2)])
        custom = TensorBoardLogger(save_dir=str(tmp_path), name="custom", version="v")
        trainer = _fit_single(cfg, _TinyRegressor(), _pl_cfg(cfg), custom, [_best_cb()])
        assert trainer.logger is custom


# ---------------------------------------------------------------------------
# AIO2 (train._fit_aio2 -> fit_two_phase with a resume root)
# ---------------------------------------------------------------------------


class _CrashingValMeanTeacher(_ScriptedValMeanTeacher):
    """Crashes at the start of a given (phase, epoch); records epochs per phase."""

    crash_at = None  # (o2c_active, epoch)

    def __init__(self, cfg):
        super().__init__(cfg)
        self.warmup_epochs_run = []
        self.correction_epochs_run = []

    def on_train_epoch_start(self):
        if self.crash_at == (self.o2c_active, self.current_epoch):
            raise RuntimeError("simulated crash")
        (self.correction_epochs_run if self.o2c_active else self.warmup_epochs_run).append(
            self.current_epoch
        )


def _aio2_run(cfg, crash_at=None):
    model = _CrashingValMeanTeacher(cfg)
    model.crash_at = crash_at
    trainer, info = _fit_aio2(cfg, model, _pl_cfg(cfg), True)
    return model, trainer, info


class TestFitAio2Resume:
    """Scripted val curve: warm-up best epoch 2, stops after 5 epochs; phase 2
    restarts from 3 completed epochs (see ``_ScriptedValMeanTeacher``)."""

    def _reference(self, tmp_path):
        return _aio2_run(_val_best_cfg(tmp_path / "ref", resumable=True))

    def test_uninterrupted_run_matches_non_resumable(self, tmp_path):
        model, trainer, info = self._reference(tmp_path)
        assert info["aio2/warmup_epochs"] == 5.0
        assert info["aio2/resume_checkpoint_epoch"] == 3.0
        assert model.warmup_epochs_run == [0, 1, 2, 3, 4]
        assert min(model.correction_epochs_run) == 3
        assert "version_1" in trainer.checkpoint_callback.best_model_path
        assert read_aio2_state(tmp_path / "ref")["info"] == info

    def test_crash_in_warmup_resumes_warmup(self, tmp_path):
        cfg = _val_best_cfg(tmp_path, resumable=True)
        with pytest.raises(RuntimeError):
            _aio2_run(cfg, crash_at=(False, 4))
        assert read_aio2_state(tmp_path) is None  # warm-up never finished

        model, trainer, info = _aio2_run(cfg)
        assert model.warmup_epochs_run == [4]  # only the missing warm-up epoch
        assert info["aio2/warmup_epochs"] == 5.0
        assert info["aio2/resume_checkpoint_epoch"] == 3.0
        assert info["aio2/phase2_started"] == 1.0
        assert min(model.correction_epochs_run) == 3
        assert "version_1" in trainer.checkpoint_callback.best_model_path
        assert not (tmp_path / "lightning_logs" / "version_2").exists()

    def test_crash_in_correction_resumes_correction(self, tmp_path):
        cfg = _val_best_cfg(tmp_path, resumable=True)
        _, ref_trainer, ref_info = self._reference(tmp_path)
        with pytest.raises(RuntimeError):
            _aio2_run(cfg, crash_at=(True, 5))

        model, trainer, info = _aio2_run(cfg)
        assert model.warmup_epochs_run == []  # warm-up not redone
        assert model.correction_epochs_run[0] == 5
        assert model.o2c_active is True
        assert info == ref_info
        assert trainer.current_epoch == ref_trainer.current_epoch
        assert "version_1" in trainer.checkpoint_callback.best_model_path

    def test_crash_before_first_correction_epoch_restarts_phase2(self, tmp_path):
        cfg = _val_best_cfg(tmp_path, resumable=True)
        with pytest.raises(RuntimeError):
            _aio2_run(cfg, crash_at=(True, 3))  # no correction epoch saved yet
        assert not (resume_dir(tmp_path) / "correction_last.ckpt").exists()
        assert read_aio2_state(tmp_path) is not None

        model, _, info = _aio2_run(cfg)
        assert model.warmup_epochs_run == []
        assert model.correction_epochs_run[0] == 3  # phase 2 from the init ckpt
        assert info["aio2/resume_checkpoint_epoch"] == 3.0

    def test_act_trigger_is_never_resumed(self, tmp_path):
        """ACTTracker's history is not checkpointed, so ACT mode keeps the old
        behaviour: no resume state, the runner restarts the run from scratch."""
        cfg = _cfg(tmp_path, resumable=True)
        model = _CrashingValMeanTeacher(cfg)
        _fit_aio2(cfg, model, _pl_cfg(cfg), True)
        assert not resume_dir(tmp_path).exists()

    def test_not_resumable_writes_nothing(self, tmp_path):
        _aio2_run(_val_best_cfg(tmp_path))
        assert not resume_dir(tmp_path).exists()


# ---------------------------------------------------------------------------
# train(): resume state removed once the run (fit + test) is complete
# ---------------------------------------------------------------------------


class TestTrainClearsResumeState:
    @pytest.mark.parametrize("resumable", [True, False])
    def test_clear_after_success(self, tmp_path, resumable):
        from pytorch_segmentation_models_trainer import train as train_module

        write_aio2_state(tmp_path, {})
        cfg = OmegaConf.create(
            {
                "resumable": resumable,
                "hyperparameters": {},
                "pl_trainer": {"default_root_dir": str(tmp_path)},
                "add_final_metrics_callback": False,
            }
        )
        model = MagicMock()
        with patch.object(train_module, "Model", return_value=model), patch.object(
            train_module, "_fit_single", return_value=MagicMock()
        ) as fit:
            train_module.train(cfg)
        fit.assert_called_once()
        assert resume_dir(tmp_path).exists() is (not resumable)

    def test_resume_root_defaults_to_cwd(self, tmp_path, monkeypatch):
        from pytorch_segmentation_models_trainer.train import _resume_root

        monkeypatch.chdir(tmp_path)
        assert _resume_root({}) == tmp_path
        assert _resume_root({"default_root_dir": "/x"}) == Path("/x")


# ---------------------------------------------------------------------------
# ExperimentsRunner keeps an interrupted, resumable run's directory
# ---------------------------------------------------------------------------


def _runner(tmp_path, resumable):
    cfg = OmegaConf.create(
        {
            "resumable": resumable,
            "experiments_runner": {"seeds": [1], "output_base_dir": str(tmp_path)},
        }
    )
    return ExperimentsRunner(cfg)


class TestRunnerKeepsInterruptedRun:
    def _interrupted_run(self, tmp_path):
        out = tmp_path / "run_00_seed1"
        write_aio2_state(out, {})
        return out

    def test_keeps_dir_when_resumable(self, tmp_path):
        out = self._interrupted_run(tmp_path)
        _runner(tmp_path, True)._clean_output_dir(str(out))
        assert has_resume_state(out)

    def test_redo_always_cleans(self, tmp_path):
        out = self._interrupted_run(tmp_path)
        _runner(tmp_path, True)._clean_output_dir(str(out), redo=True)
        assert not out.exists()

    def test_not_resumable_cleans(self, tmp_path):
        out = self._interrupted_run(tmp_path)
        _runner(tmp_path, False)._clean_output_dir(str(out))
        assert not out.exists()

    def test_resumable_without_state_cleans(self, tmp_path):
        out = tmp_path / "run_00_seed1"
        out.mkdir()
        (out / "stale.txt").write_text("x")
        _runner(tmp_path, True)._clean_output_dir(str(out))
        assert not out.exists()

    @pytest.mark.parametrize(
        "completed, overwrite, redo",
        [([], None, False), ([0], True, True)],
    )
    def test_seed_loop_passes_redo_only_for_completed_runs(
        self, tmp_path, completed, overwrite, redo
    ):
        from pytorch_segmentation_models_trainer.tools.experiments_runner.experiments_runner import (
            RunResult,
        )

        runner = _runner(tmp_path, True)
        if overwrite is not None:
            runner.runner_cfg = OmegaConf.merge(runner.runner_cfg, {"overwrite": overwrite})
        prior = [
            RunResult(run_idx=i, seed=1, training_time_seconds=1.0, train_metrics={},
                      val_metrics={}, test_metrics={}, output_dir=str(tmp_path))
            for i in completed
        ]
        if prior:
            runner._save_state([1], prior)
        new = RunResult(run_idx=0, seed=1, training_time_seconds=1.0, train_metrics={},
                        val_metrics={}, test_metrics={}, output_dir=str(tmp_path))
        with patch.object(runner, "_clean_output_dir") as clean, patch.object(
            runner, "_run_single", return_value=new
        ):
            runner._run_seed_loop()
        assert clean.call_args.kwargs == {"redo": redo}
