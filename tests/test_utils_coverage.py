# -*- coding: utf-8 -*-
import runpy
import sys
import warnings
from unittest.mock import patch

import pytest
import torch
import torch.nn as nn
from omegaconf import OmegaConf

from pytorch_segmentation_models_trainer.config_utils import validate_config
from pytorch_segmentation_models_trainer.custom_models.utils import (
    _SimpleSegmentationModel,
)
from pytorch_segmentation_models_trainer.utils.mc_dropout_utils import (
    compute_uncertainty,
    enable_mc_dropout,
    warn_if_no_dropout,
)


def test_validate_config_logs_yaml():
    cfg = OmegaConf.create({"model": {"name": "tiny"}})

    with patch("pytorch_segmentation_models_trainer.config_utils.logger") as logger:
        validate_config.__wrapped__(cfg)

    logger.info.assert_called_once()
    assert "model" in logger.info.call_args.args[1]


def test_config_utils_main_guard_runs_hydra_entrypoint(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["config_utils.py"])

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        runpy.run_module(
            "pytorch_segmentation_models_trainer.config_utils",
            run_name="__main__",
        )


def test_simple_segmentation_model_upsamples_classifier_output():
    class Backbone(nn.Module):
        def forward(self, x):
            return {"out": x[:, :1, ::2, ::2]}

    classifier = nn.Conv2d(1, 2, kernel_size=1)
    model = _SimpleSegmentationModel(Backbone(), classifier)
    x = torch.randn(2, 3, 8, 10)

    out = model(x)

    assert out.shape == (2, 2, 8, 10)


def test_enable_mc_dropout_only_sets_dropout_layers_to_train():
    model = nn.Sequential(
        nn.BatchNorm2d(3),
        nn.Dropout(),
        nn.Dropout2d(),
        nn.Dropout3d(),
    )
    model.eval()

    enable_mc_dropout(model)

    assert model[0].training is False
    assert model[1].training is True
    assert model[2].training is True
    assert model[3].training is True


def test_warn_if_no_dropout_reports_presence_and_absence():
    assert warn_if_no_dropout(nn.Sequential(nn.Dropout())) is True

    with pytest.warns(UserWarning, match="no Dropout"):
        assert warn_if_no_dropout(nn.Linear(2, 2)) is False


def test_compute_uncertainty_entropy_and_mutual_information():
    samples = torch.tensor(
        [
            [[[[0.8]], [[0.2]]]],
            [[[[0.2]], [[0.8]]]],
        ],
        dtype=torch.float32,
    )

    entropy = compute_uncertainty(samples, mode="entropy")
    mutual_information = compute_uncertainty(samples, mode="mutual_information")

    assert entropy.shape == (1, 1, 1, 1)
    assert mutual_information.shape == (1, 1, 1, 1)
    assert entropy.item() > 0
    assert mutual_information.item() > 0


def test_compute_uncertainty_rejects_unknown_mode():
    samples = torch.ones(1, 1, 2, 1, 1) / 2

    with pytest.raises(ValueError, match="uncertainty_mode"):
        compute_uncertainty(samples, mode="bad")
