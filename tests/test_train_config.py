# -*- coding: utf-8 -*-
"""
Tests for train_config dataclasses.
"""

import pytest
from omegaconf import OmegaConf, MissingMandatoryValue

from pytorch_segmentation_models_trainer.config_definitions.train_config import (
    BackboneConfig,
    PLTrainerConfig,
    SegParams,
    OptimizerConfig,
    Hyperparameters,
    SchedulerConfig,
    SchedulerItemConfig,
    CallbackConfig,
    MetricConfig,
    TrainConfig,
)


class TestBackboneConfig:
    def test_defaults(self):
        cfg = OmegaConf.structured(BackboneConfig)
        assert cfg.name == "resnet152"
        assert cfg.input_width == 224
        assert cfg.input_height == 224


class TestPLTrainerConfig:
    def test_defaults(self):
        cfg = OmegaConf.structured(PLTrainerConfig)
        # Use to_container with resolve=False to check interpolations
        container = OmegaConf.to_container(cfg, resolve=False)
        assert container["max_epochs"] == "${hyperparameters.epochs}"
        assert container["devices"] == -1
        assert container["accelerator"] == "auto"
        assert container["precision"] == "32-true"


class TestSegParams:
    def test_defaults(self):
        cfg = OmegaConf.structured(SegParams)
        assert cfg.compute_interior is True
        assert cfg.compute_edge is True
        assert cfg.compute_vertex is False


class TestOptimizerConfig:
    def test_defaults(self):
        cfg = OmegaConf.structured(OptimizerConfig)
        container = OmegaConf.to_container(cfg, resolve=False)
        assert container["_target_"] == "torch.optim.AdamW"
        assert container["lr"] == "${hyperparameters.max_lr}"
        assert container["weight_decay"] == 1e-3


class TestHyperparameters:
    def test_defaults(self):
        cfg = OmegaConf.structured(Hyperparameters)
        assert cfg.model_name == "unet"
        assert cfg.batch_size == 16
        assert cfg.epochs == 10
        assert cfg.max_lr == 1e-2
        assert cfg.classes == 1
        with pytest.raises(MissingMandatoryValue):
            _ = cfg.backbone


class TestSchedulerConfig:
    def test_defaults(self):
        cfg = OmegaConf.structured(SchedulerConfig)
        container = OmegaConf.to_container(cfg, resolve=False)
        assert container["_target_"] == "torch.optim.lr_scheduler.OneCycleLR"
        assert container["max_lr"] == "${hyperparameters.max_lr}"
        assert container["steps_per_epoch"] == 5161
        assert container["epochs"] == "${hyperparameters.epochs}"


class TestSchedulerItemConfig:
    def test_defaults(self):
        cfg = OmegaConf.structured(SchedulerItemConfig)
        container = OmegaConf.to_container(cfg, resolve=False)
        assert (
            container["scheduler"]["_target_"] == "torch.optim.lr_scheduler.OneCycleLR"
        )
        assert container["name"] == "learning_rate"
        assert container["interval"] == "step"
        assert container["frequency"] == 1
        assert container["monitor"] == "avg_val_loss"


class TestCallbackConfig:
    def test_defaults(self):
        cfg = OmegaConf.structured(CallbackConfig)
        assert cfg._target_ == "pytorch_lightning.callbacks.LearningRateMonitor"


class TestMetricConfig:
    def test_target_is_required(self):
        cfg = OmegaConf.structured(MetricConfig)
        with pytest.raises(MissingMandatoryValue):
            _ = cfg._target_


class TestTrainConfig:
    def test_defaults(self):
        cfg = OmegaConf.structured(TrainConfig)
        assert cfg.compute_seg is True
        assert cfg.device == "cpu"
        assert cfg.seed is None
        assert cfg.deterministic_cudnn is False

        # Test mandatory fields
        with pytest.raises(MissingMandatoryValue):
            _ = cfg.metrics

        with pytest.raises(MissingMandatoryValue):
            _ = cfg.hyperparameters.backbone
