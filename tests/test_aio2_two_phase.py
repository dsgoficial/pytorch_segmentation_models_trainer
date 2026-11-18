# -*- coding: utf-8 -*-
"""End-to-end tests for the AIO2 two-phase driver (warm-up -> O2C).

Runs real (tiny, CPU) Lightning fits of ``MeanTeacherModel`` through
``mean_teacher_model.fit_two_phase`` and ``train._fit_aio2``, with ACT's
trigger forced at a known epoch so the phase hand-off is deterministic.
"""

import json
from unittest.mock import MagicMock, patch

import pytest
import pytorch_lightning as pl
import torch
from hydra import compose, initialize
from omegaconf import OmegaConf
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from torch.utils.data import DataLoader, Dataset

from pytorch_segmentation_models_trainer.model_loader.mean_teacher_model import (
    MeanTeacherModel,
    fit_two_phase,
    prepare_phase2_checkpoint,
)
from pytorch_segmentation_models_trainer.tools.experiments_runner.experiments_runner import (
    _seed_subprocess_worker,
)
from pytorch_segmentation_models_trainer.train import _build_callbacks, _fit_aio2, train

from tests.utils import CustomTestCase

CLASSES, H, W, N = 3, 32, 32, 8


class _DictDataset(Dataset):
    def __init__(self, seed: int) -> None:
        g = torch.Generator().manual_seed(seed)
        self.images = torch.randn(N, 3, H, W, generator=g)
        self.masks = torch.randint(0, CLASSES, (N, H, W), generator=g)

    def __len__(self) -> int:
        return N

    def __getitem__(self, i):
        return {"image": self.images[i], "mask": self.masks[i]}


class _TinyMeanTeacher(MeanTeacherModel):
    """MeanTeacherModel with in-memory loaders and ACT forced to fire."""

    trigger_at_epoch = 3  # ACT "detects" at the end of this (0-indexed) epoch...
    # ...and reports I_r in ACTTracker's convention: a 1-indexed count of
    # completed epochs. I_r=3 -> state after 3 epochs -> file epoch_2.ckpt.
    i_r = 3

    def __init__(self, cfg):
        super().__init__(cfg)
        self.o2c_calls = 0
        self.epochs_seen_with_o2c = set()
        model = self

        class _ForcedACT:
            def update(self, epoch_iou):
                if model.current_epoch == model.trigger_at_epoch:
                    return model.i_r
                return None

        self.act_tracker = _ForcedACT()

    def train_dataloader(self):
        return DataLoader(_DictDataset(0), batch_size=4)

    def val_dataloader(self):
        return DataLoader(_DictDataset(1), batch_size=4)

    def _apply_o2c(self, batch):
        self.o2c_calls += 1
        self.epochs_seen_with_o2c.add(self.current_epoch)
        return super()._apply_o2c(batch)


def _cfg(tmp_path, **extra):
    base = {
        "model": {
            "_target_": "segmentation_models_pytorch.Unet",
            "encoder_name": "resnet18",
            "encoder_weights": None,
            "in_channels": 3,
            "classes": CLASSES,
        },
        "loss": {
            "_target_": "pytorch_segmentation_models_trainer.custom_losses.loss.WeightedDiceCrossEntropyLoss",
            "num_classes": CLASSES,
        },
        "optimizer": {"_target_": "torch.optim.AdamW", "lr": 1e-3},
        "metrics": [
            {
                "_target_": "torchmetrics.JaccardIndex",
                "task": "multiclass",
                "num_classes": CLASSES,
                "average": "macro",
            }
        ],
        "image_key": "image",
        "mask_key": "mask",
        "hyperparameters": {
            "batch_size": 4,
            "epochs": 8,
            "max_lr": 1e-3,
            "devices": 1,
            "accelerator": "cpu",
        },
        "classes_to_correct": [0, 1, 2],
        "alpha": 0.9,
        "checkpoint_every_n_epochs": 1,
        "callbacks": [
            {
                "_target_": "pytorch_lightning.callbacks.EarlyStopping",
                "monitor": "loss/val",
                "patience": 100,
                "mode": "min",
            },
            {
                "_target_": "pytorch_lightning.callbacks.ModelCheckpoint",
                "monitor": "loss/val",
                "mode": "min",
                "save_top_k": 1,
                "filename": "best-{epoch:02d}",
            },
        ],
        "pl_trainer": {
            "max_epochs": 8,
            "accelerator": "cpu",
            "devices": 1,
            "default_root_dir": str(tmp_path),
            "enable_progress_bar": False,
            "enable_model_summary": False,
        },
    }
    base.update(extra)
    return OmegaConf.create(base)


def _trainer_factory(cfg, record):
    pl_cfg = OmegaConf.to_container(cfg.pl_trainer, resolve=True)

    def make(phase, point=None):
        record.append(phase)
        return Trainer(
            **pl_cfg,
            logger=True,
            callbacks=_build_callbacks(
                cfg,
                drop_early_stopping=(
                    phase == "warmup" and cfg.get("phase2_trigger", "act") == "act"
                ),
            ),
        )

    return make


class _ScriptedValMeanTeacher(_TinyMeanTeacher):
    """Logs a scripted ``scripted/val`` curve so early stopping is deterministic.

    Peak at epoch 2: with ``patience=2`` the warm-up stops after epoch 4 and
    the best checkpoint is epoch 2 (state after 3 completed epochs).
    """

    curve = [0.1, 0.5, 0.9, 0.4, 0.3, 0.2, 0.15, 0.1, 0.05, 0.01]

    def on_validation_epoch_end(self):
        super().on_validation_epoch_end()
        self.log("scripted/val", self.curve[self.current_epoch])


def _val_best_cfg(tmp_path, **extra):
    cfg = _cfg(tmp_path, phase2_trigger="val_best", **extra)
    cfg.callbacks = [
        {
            "_target_": "pytorch_lightning.callbacks.EarlyStopping",
            "monitor": "scripted/val",
            "patience": 2,
            "mode": "max",
        },
        {
            "_target_": "pytorch_lightning.callbacks.ModelCheckpoint",
            "monitor": "scripted/val",
            "mode": "max",
            "save_top_k": 1,
            "filename": "best-{epoch:02d}",
        },
    ]
    cfg.pl_trainer.num_sanity_val_steps = 0
    return cfg


class TestBuildCallbacks:
    def test_warmup_drops_early_stopping_only(self, tmp_path):
        cfg = _cfg(tmp_path)
        warm = _build_callbacks(cfg, drop_early_stopping=True)
        corr = _build_callbacks(cfg, drop_early_stopping=False)
        assert not any(isinstance(c, EarlyStopping) for c in warm)
        assert any(isinstance(c, ModelCheckpoint) for c in warm)
        assert any(isinstance(c, EarlyStopping) for c in corr)

    def test_fresh_objects_per_call(self, tmp_path):
        cfg = _cfg(tmp_path)
        a = _build_callbacks(cfg)
        b = _build_callbacks(cfg)
        assert all(x is not y for x, y in zip(a, b))


class TestTwoPhaseFit:
    def test_handoff_resumes_at_checkpoint_with_o2c(self, tmp_path):
        cfg = _cfg(tmp_path)
        model = _TinyMeanTeacher(cfg)
        phases = []
        trainer, info = fit_two_phase(model, _trainer_factory(cfg, phases))

        assert phases == ["warmup", "correction"]
        assert info["aio2/act_triggered"] == 1.0
        assert info["aio2/resume_epoch"] == 3.0
        # Same convention as resume_epoch (completed epochs): epoch_2.ckpt.
        assert info["aio2/resume_checkpoint_epoch"] == 3.0
        # Warm-up stopped right after the trigger epoch (0..3 -> 4 epochs).
        assert info["aio2/warmup_epochs"] == 4.0
        assert model.o2c_active is True
        # Phase 2 trained with O2C, starting after the epoch-2 checkpoint,
        # and ran to max_epochs (patience 100 never fires).
        assert model.o2c_calls > 0
        assert min(model.epochs_seen_with_o2c) == 3
        assert trainer.current_epoch == cfg.pl_trainer.max_epochs
        # Per-seed default dir: checkpoints land under default_root_dir.
        assert model.act_checkpoint_dir == tmp_path / "act_warmup_checkpoints"
        assert (tmp_path / "act_warmup_checkpoints" / "epoch_2.ckpt").exists()

    def test_resume_never_after_i_r(self, tmp_path):
        """I_r=4 must resume from epoch_3.ckpt (4 done), not epoch_4 (5 done)."""
        cfg = _cfg(tmp_path)
        model = _TinyMeanTeacher(cfg)
        model.trigger_at_epoch, model.i_r = 5, 4
        _, info = fit_two_phase(model, _trainer_factory(cfg, []))
        assert info["aio2/resume_checkpoint_epoch"] == 4.0
        assert min(model.epochs_seen_with_o2c) == 4

    def test_phase2_best_checkpoint_is_from_phase2(self, tmp_path):
        cfg = _cfg(tmp_path)
        model = _TinyMeanTeacher(cfg)
        trainer, _ = fit_two_phase(model, _trainer_factory(cfg, []))
        best = trainer.checkpoint_callback.best_model_path
        assert best, "phase 2 ModelCheckpoint saved nothing"
        epoch = int(best.split("best-epoch=")[1].split(".ckpt")[0])
        assert epoch >= 3, f"best checkpoint {best} is a warm-up (pre-O2C) model"
        # Separate logger versions keep the phases' checkpoints apart.
        assert "version_1" in best

    def test_no_trigger_skips_phase2(self, tmp_path):
        cfg = _cfg(tmp_path)
        cfg.pl_trainer.max_epochs = 3
        model = _TinyMeanTeacher(cfg)
        model.trigger_at_epoch = 99  # never
        phases = []
        trainer, info = fit_two_phase(model, _trainer_factory(cfg, phases))
        assert phases == ["warmup"]
        assert info["aio2/act_triggered"] == 0.0
        assert info["aio2/phase2_started"] == 0.0
        assert model.o2c_active is False
        assert model.o2c_calls == 0
        assert float(trainer.callback_metrics["aio2/act_triggered"]) == 0.0

    def test_train_fit_aio2_wrapper(self, tmp_path):
        cfg = _cfg(tmp_path)
        model = _TinyMeanTeacher(cfg)
        pl_cfg = OmegaConf.to_container(cfg.pl_trainer, resolve=True)
        trainer, info = _fit_aio2(cfg, model, pl_cfg, True)
        assert info["aio2/act_triggered"] == 1.0
        assert any(isinstance(c, EarlyStopping) for c in trainer.callbacks)


class TestValBestTrigger:
    """phase2_trigger="val_best": phase 2 starts from warm-up's best-val checkpoint."""

    def test_handoff_from_best_val_checkpoint(self, tmp_path):
        cfg = _val_best_cfg(tmp_path)
        model = _ScriptedValMeanTeacher(cfg)  # forced ACT would fire at epoch 3
        phases = []
        trainer, info = fit_two_phase(model, _trainer_factory(cfg, phases))

        assert phases == ["warmup", "correction"]
        # ACT is ignored: warm-up ran past the forced-ACT epoch (3) and was
        # stopped by EarlyStopping (best epoch 2 + patience 2 -> 5 epochs).
        assert model.resume_epoch is None
        assert info["aio2/warmup_epochs"] == 5.0
        assert info["aio2/phase2_started"] == 1.0
        assert info["aio2/phase2_trigger_val_best"] == 1.0
        assert "aio2/act_triggered" not in info
        # best-epoch=02.ckpt = state after 3 completed epochs; O2C from epoch 3.
        assert info["aio2/resume_checkpoint_epoch"] == 3.0
        assert min(model.epochs_seen_with_o2c) == 3
        best = trainer.checkpoint_callback.best_model_path
        assert "version_1" in best
        # ACT warm-up checkpoints are not needed in this mode.
        assert not (tmp_path / "act_warmup_checkpoints").exists()

    def test_no_best_checkpoint_skips_phase2(self, tmp_path):
        cfg = _val_best_cfg(tmp_path)
        cfg.callbacks = [cfg.callbacks[0]]  # EarlyStopping only, no ModelCheckpoint
        cfg.pl_trainer.enable_checkpointing = False
        model = _ScriptedValMeanTeacher(cfg)
        phases = []
        trainer, info = fit_two_phase(model, _trainer_factory(cfg, phases))
        assert phases == ["warmup"]
        assert info["aio2/phase2_started"] == 0.0
        assert model.o2c_active is False
        assert float(trainer.callback_metrics["aio2/phase2_started"]) == 0.0

    def test_fit_aio2_keeps_early_stopping_in_warmup(self, tmp_path):
        cfg = _val_best_cfg(tmp_path)
        model = _ScriptedValMeanTeacher(cfg)
        pl_cfg = OmegaConf.to_container(cfg.pl_trainer, resolve=True)
        _, info = _fit_aio2(cfg, model, pl_cfg, True)
        # Without EarlyStopping in the warm-up it would run all 8 epochs.
        assert info["aio2/warmup_epochs"] == 5.0

    def test_invalid_trigger_rejected(self, tmp_path):
        with pytest.raises(ValueError, match="phase2_trigger"):
            _TinyMeanTeacher(_cfg(tmp_path, phase2_trigger="bogus"))

    def test_act_mode_reports_phase2_started(self, tmp_path):
        cfg = _cfg(tmp_path)
        _, info = fit_two_phase(_TinyMeanTeacher(cfg), _trainer_factory(cfg, []))
        assert info["aio2/phase2_started"] == 1.0
        assert info["aio2/phase2_trigger_val_best"] == 0.0


class TestPreparePhase2Checkpoint:
    def test_strips_callbacks_keeps_rest(self, tmp_path):
        ckpt = tmp_path / "epoch_5.ckpt"
        torch.save(
            {
                "state_dict": {"w": torch.ones(1)},
                "optimizer_states": [{"x": 1}],
                "epoch": 5,
                "global_step": 50,
                "callbacks": {"ModelCheckpoint{...}": {"best_model_score": 0.9}},
            },
            ckpt,
        )
        out = prepare_phase2_checkpoint(ckpt)
        assert out.name == "epoch_5_phase2_init.ckpt"
        state = torch.load(out, weights_only=False)
        assert "callbacks" not in state
        assert state["epoch"] == 5 and state["global_step"] == 50
        assert state["optimizer_states"] == [{"x": 1}]
        assert torch.equal(state["state_dict"]["w"], torch.ones(1))
        # original untouched
        assert "callbacks" in torch.load(ckpt, weights_only=False)


class TestTrainDispatch(CustomTestCase):
    """train() routes MeanTeacherModel through the two-phase driver."""

    @patch("pytorch_segmentation_models_trainer.train._fit_aio2")
    @patch("pytorch_segmentation_models_trainer.train.Trainer")
    @patch.object(MeanTeacherModel, "setup")
    def test_mean_teacher_uses_two_phase_and_keeps_diagnostics(
        self, _setup, MockTrainer, mock_fit_aio2
    ):
        phase_trainer = MagicMock(spec=pl.Trainer)
        phase_trainer.callback_metrics = {}
        info = {"aio2/act_triggered": 1.0, "aio2/resume_epoch": 30.0}
        mock_fit_aio2.return_value = (phase_trainer, info)

        with initialize(config_path="./test_configs"):
            cfg = compose(
                config_name="experiment_with_test.yaml",
                overrides=[
                    "train_dataset.input_csv_path=" + self.csv_ds_file,
                    "val_dataset.input_csv_path=" + self.csv_ds_file,
                    "test_dataset.input_csv_path=" + self.csv_ds_file,
                    "+pl_model._target_=pytorch_segmentation_models_trainer."
                    "model_loader.mean_teacher_model.MeanTeacherModel",
                ],
            )
            result = train(cfg)

        mock_fit_aio2.assert_called_once()
        assert isinstance(mock_fit_aio2.call_args.args[1], MeanTeacherModel)
        MockTrainer.assert_not_called()  # no single-phase Trainer built
        phase_trainer.test.assert_called_once()
        assert result is phase_trainer
        # diagnostics survive trainer.test() for the runner's JSON
        assert float(result.callback_metrics["aio2/act_triggered"]) == 1.0
        assert float(result.callback_metrics["aio2/resume_epoch"]) == 30.0


class TestSeedWorkerKeepsAio2Diagnostics:
    def test_aio2_keys_land_in_test_metrics(self, tmp_path):
        fake = MagicMock()
        fake.callback_metrics = {
            "MulticlassJaccardIndex/test": torch.tensor(0.5),
            "loss/val": torch.tensor(0.9),
            "aio2/act_triggered": torch.tensor(1.0),
            "aio2/resume_epoch": torch.tensor(30.0),
        }
        fake.checkpoint_callback.best_model_path = "/x/best.ckpt"
        fake.current_epoch = 41
        result_path = tmp_path / "r.json"
        with patch("pytorch_segmentation_models_trainer.train.train", return_value=fake):
            _seed_subprocess_worker("seed: 1\n", str(tmp_path / "out"), str(result_path))
        res = json.loads(result_path.read_text())
        assert res["ok"] is True
        assert res["test_metrics"] == {
            "MulticlassJaccardIndex/test": 0.5,
            "aio2/act_triggered": 1.0,
            "aio2/resume_epoch": 30.0,
        }
        assert res["val_metrics"] == {"loss/val": pytest.approx(0.9)}
