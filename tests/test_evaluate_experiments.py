# -*- coding: utf-8 -*-
import logging
import runpy
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from omegaconf import OmegaConf

from pytorch_segmentation_models_trainer import evaluate_experiments


def test_setup_logging_uses_defaults_without_logging_config():
    cfg = OmegaConf.create({})

    with patch("logging.basicConfig") as mock_basic_config:
        evaluate_experiments.setup_logging(cfg)

    mock_basic_config.assert_called_once_with(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def test_setup_logging_uses_custom_level_without_file_handler():
    cfg = OmegaConf.create(
        {
            "logging": {
                "level": "debug",
                "save_to_file": False,
                "log_file": "custom.log",
            }
        }
    )

    with (
        patch("logging.basicConfig") as mock_basic_config,
        patch("logging.FileHandler") as mock_file_handler,
    ):
        evaluate_experiments.setup_logging(cfg)

    mock_basic_config.assert_called_once()
    assert mock_basic_config.call_args.kwargs["level"] == logging.DEBUG
    mock_file_handler.assert_not_called()


def test_setup_logging_adds_file_handler_in_output_dir(tmp_path):
    cfg = OmegaConf.create(
        {
            "logging": {
                "level": "warning",
                "save_to_file": True,
                "log_file": "evaluation-test.log",
            },
            "output": {"base_dir": str(tmp_path / "logs")},
        }
    )
    file_handler = MagicMock()

    with (
        patch("logging.basicConfig"),
        patch("logging.FileHandler", return_value=file_handler) as mock_file_handler,
        patch("logging.getLogger") as mock_get_logger,
    ):
        evaluate_experiments.setup_logging(cfg)

    expected_log_path = tmp_path / "logs" / "evaluation-test.log"
    mock_file_handler.assert_called_once_with(expected_log_path, mode="w")
    file_handler.setLevel.assert_called_once_with(logging.WARNING)
    file_handler.setFormatter.assert_called_once()
    mock_get_logger.return_value.addHandler.assert_called_once_with(file_handler)
    assert expected_log_path.parent.is_dir()


def test_setup_logging_uses_default_output_dir_when_missing_output_config(
    tmp_path, monkeypatch
):
    monkeypatch.chdir(tmp_path)
    cfg = OmegaConf.create(
        {"logging": {"save_to_file": True, "log_file": "fallback.log"}}
    )
    file_handler = MagicMock()

    with (
        patch("logging.basicConfig"),
        patch("logging.FileHandler", return_value=file_handler) as mock_file_handler,
        patch("logging.getLogger"),
    ):
        evaluate_experiments.setup_logging(cfg)

    expected_log_path = Path("evaluation_outputs/fallback.log")
    mock_file_handler.assert_called_once_with(expected_log_path, mode="w")
    assert (tmp_path / expected_log_path.parent).is_dir()


def test_evaluate_returns_pipeline_results():
    cfg = OmegaConf.create({"logging": {"save_to_file": False}})
    expected_results = {"experiment": {"iou": 0.75}}

    with (
        patch.object(evaluate_experiments, "setup_logging") as mock_setup,
        patch.object(evaluate_experiments, "EvaluationPipeline") as mock_pipeline_cls,
    ):
        mock_pipeline_cls.return_value.run.return_value = expected_results

        results = evaluate_experiments.evaluate(cfg)

    assert results == expected_results
    mock_setup.assert_called_once_with(cfg)
    mock_pipeline_cls.assert_called_once_with(cfg)
    mock_pipeline_cls.return_value.run.assert_called_once_with()


def test_evaluate_exits_when_pipeline_fails():
    cfg = OmegaConf.create({"logging": {"save_to_file": False}})

    with (
        patch.object(evaluate_experiments, "setup_logging"),
        patch.object(evaluate_experiments, "EvaluationPipeline") as mock_pipeline_cls,
        patch.object(evaluate_experiments.logger, "error"),
    ):
        mock_pipeline_cls.return_value.run.side_effect = RuntimeError("boom")

        with pytest.raises(SystemExit) as exc_info:
            evaluate_experiments.evaluate(cfg)

    assert exc_info.value.code == 1


def test_module_main_guard_runs_evaluate(monkeypatch):
    cfg = OmegaConf.create({"logging": {"save_to_file": False}})
    calls = []

    def fake_hydra_main(**_kwargs):
        def decorator(func):
            def wrapper():
                calls.append("evaluate")
                return func(cfg)

            return wrapper

        return decorator

    fake_hydra = types.SimpleNamespace(main=fake_hydra_main)
    fake_pipeline_module = types.ModuleType(
        "pytorch_segmentation_models_trainer.tools.evaluation.evaluation_pipeline"
    )
    fake_pipeline = MagicMock()
    fake_pipeline.return_value.run.return_value = {"ok": True}
    fake_pipeline_module.EvaluationPipeline = fake_pipeline

    monkeypatch.setitem(sys.modules, "hydra", fake_hydra)
    monkeypatch.setitem(
        sys.modules,
        "pytorch_segmentation_models_trainer.tools.evaluation.evaluation_pipeline",
        fake_pipeline_module,
    )
    monkeypatch.delitem(
        sys.modules,
        "pytorch_segmentation_models_trainer.evaluate_experiments",
        raising=False,
    )

    runpy.run_module(
        "pytorch_segmentation_models_trainer.evaluate_experiments",
        run_name="__main__",
    )

    assert calls == ["evaluate"]
