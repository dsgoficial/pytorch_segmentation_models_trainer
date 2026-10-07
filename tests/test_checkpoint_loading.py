# -*- coding: utf-8 -*-
"""Tests for pretrained-weight loading and runner-based checkpoint resolution."""

import json

import pytest
import torch
from torch import nn

from pytorch_segmentation_models_trainer.utils.checkpoint_loading import (
    load_pretrained_weights,
    resolve_checkpoint_from_runner,
)


def _net(out=3):
    return nn.Sequential(nn.Conv2d(2, 4, 1), nn.Conv2d(4, out, 1))


class TestLoadPretrainedWeights:
    def test_lightning_format_strips_prefix(self, tmp_path):
        src = _net()
        sd = {f"model.{k}": v for k, v in src.state_dict().items()}
        sd["other.weight"] = torch.zeros(1)  # non-model keys are ignored
        path = tmp_path / "a.ckpt"
        torch.save({"state_dict": sd}, path)
        dst = _net()
        load_pretrained_weights(
            dst, str(path), "pytorch_lightning", strict_loading=True
        )
        x = torch.randn(1, 2, 3, 3)
        torch.testing.assert_close(dst(x), src(x))

    def test_pytorch_format(self, tmp_path):
        src = _net()
        path = tmp_path / "a.pt"
        torch.save(src.state_dict(), path)
        dst = _net()
        load_pretrained_weights(dst, str(path), "pytorch", strict_loading=True)
        for a, b in zip(src.parameters(), dst.parameters()):
            torch.testing.assert_close(a, b)

    def test_strict_missing_key_raises(self, tmp_path):
        sd = _net().state_dict()
        sd.pop("1.bias")
        path = tmp_path / "a.pt"
        torch.save(sd, path)
        with pytest.raises(RuntimeError, match="Missing key"):
            load_pretrained_weights(_net(), str(path), "pytorch", strict_loading=True)

    def test_non_strict_logs_missing_and_unexpected(self, tmp_path, caplog):
        sd = _net().state_dict()
        sd.pop("1.bias")
        sd["extra"] = torch.zeros(1)
        path = tmp_path / "a.pt"
        torch.save(sd, path)
        with caplog.at_level("WARNING"):
            load_pretrained_weights(_net(), str(path), "pytorch", strict_loading=False)
        assert "missing keys" in caplog.text and "unexpected keys" in caplog.text

    def test_unknown_format_raises(self, tmp_path):
        path = tmp_path / "a.pt"
        torch.save({}, path)
        with pytest.raises(ValueError, match="source_format"):
            load_pretrained_weights(_net(), str(path), "onnx")


def _write_state(tmp_path, runs):
    (tmp_path / "runner_state.json").write_text(
        json.dumps({"all_seeds": [r["seed"] for r in runs], "completed_runs": runs})
    )


class TestResolveCheckpointFromRunner:
    def test_returns_best_checkpoint_of_matching_seed(self, tmp_path):
        ckpt = tmp_path / "run_01_seed123" / "best.ckpt"
        ckpt.parent.mkdir()
        ckpt.write_text("x")
        _write_state(
            tmp_path,
            [
                {"run_idx": 0, "seed": 42, "best_checkpoint_path": "/nope.ckpt"},
                {"run_idx": 1, "seed": 123, "best_checkpoint_path": str(ckpt)},
            ],
        )
        assert resolve_checkpoint_from_runner(str(tmp_path), 123) == str(ckpt)

    def test_missing_state_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="runner_state.json"):
            resolve_checkpoint_from_runner(str(tmp_path), 42)

    def test_seed_not_completed_raises(self, tmp_path):
        _write_state(
            tmp_path, [{"run_idx": 0, "seed": 42, "best_checkpoint_path": "x"}]
        )
        with pytest.raises(ValueError, match="seed 7.*completed seeds: \\[42\\]"):
            resolve_checkpoint_from_runner(str(tmp_path), 7)

    def test_empty_checkpoint_path_raises(self, tmp_path):
        _write_state(tmp_path, [{"run_idx": 0, "seed": 42, "best_checkpoint_path": ""}])
        with pytest.raises(ValueError, match="no best checkpoint"):
            resolve_checkpoint_from_runner(str(tmp_path), 42)

    def test_checkpoint_file_missing_raises(self, tmp_path):
        _write_state(
            tmp_path,
            [
                {
                    "run_idx": 0,
                    "seed": 42,
                    "best_checkpoint_path": str(tmp_path / "gone.ckpt"),
                }
            ],
        )
        with pytest.raises(FileNotFoundError, match="gone.ckpt"):
            resolve_checkpoint_from_runner(str(tmp_path), 42)

    def test_kfold_runs_are_ambiguous(self, tmp_path):
        ckpt = tmp_path / "a.ckpt"
        ckpt.write_text("x")
        _write_state(
            tmp_path,
            [
                {
                    "run_idx": 0,
                    "seed": 42,
                    "fold_idx": 0,
                    "best_checkpoint_path": str(ckpt),
                },
                {
                    "run_idx": 1,
                    "seed": 42,
                    "fold_idx": 1,
                    "best_checkpoint_path": str(ckpt),
                },
            ],
        )
        with pytest.raises(ValueError, match="2 completed runs"):
            resolve_checkpoint_from_runner(str(tmp_path), 42)
