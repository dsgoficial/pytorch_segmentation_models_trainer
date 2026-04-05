# -*- coding: utf-8 -*-
"""
Tests for fine-tuning strategies (Phase 5).

Covers freeze_backbone, linear_probe, lora (with peft mock), and the
generic set_encoder_trainable replacement in Model.
"""
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn as nn

from pytorch_segmentation_models_trainer.fine_tuning.lora_utils import (
    apply_fine_tuning_strategy,
    freeze_modules_by_name,
    get_trainable_parameter_count,
    is_peft_model,
    merge_lora_weights,
)


# ─── Helpers ─────────────────────────────────────────────────────────────────

class _SimpleSegModel(nn.Module):
    """Minimal model with encoder + decoder + head."""
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(16, 8, 3, padding=1),
            nn.ReLU(),
        )
        self.segmentation_head = nn.Conv2d(8, 2, 1)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return self.segmentation_head(x)


class _AttentionModel(nn.Module):
    """Minimal model with attention-like naming (query/value projections)."""
    def __init__(self):
        super().__init__()
        self.encoder = nn.ModuleDict({
            "query": nn.Linear(64, 64),
            "key":   nn.Linear(64, 64),
            "value": nn.Linear(64, 64),
        })
        self.segmentation_head = nn.Linear(64, 2)

    def forward(self, x):
        q = self.encoder["query"](x)
        k = self.encoder["key"](x)
        v = self.encoder["value"](x)
        return self.segmentation_head(q + k + v)


def _cfg(**kwargs):
    return SimpleNamespace(**kwargs)


# ─── Tests: get_trainable_parameter_count ─────────────────────────────────────

class TestGetTrainableParameterCount:
    def test_all_trainable(self):
        model = _SimpleSegModel()
        counts = get_trainable_parameter_count(model)
        assert counts["trainable"] == counts["total"]
        assert counts["frozen"] == 0

    def test_all_frozen(self):
        model = _SimpleSegModel()
        for p in model.parameters():
            p.requires_grad = False
        counts = get_trainable_parameter_count(model)
        assert counts["trainable"] == 0
        assert counts["frozen"] == counts["total"]


# ─── Tests: freeze_modules_by_name ────────────────────────────────────────────

class TestFreezeModulesByName:
    def test_encoder_frozen(self):
        model = _SimpleSegModel()
        freeze_modules_by_name(model, frozen_names=("encoder",), trainable_names=("decoder", "head"))
        for name, param in model.named_parameters():
            if "encoder" in name:
                assert not param.requires_grad, f"{name} should be frozen"

    def test_head_remains_trainable(self):
        model = _SimpleSegModel()
        freeze_modules_by_name(model, frozen_names=("encoder",), trainable_names=("decoder", "head"))
        for name, param in model.named_parameters():
            if "segmentation_head" in name or "decoder" in name:
                assert param.requires_grad, f"{name} should be trainable"

    def test_trainable_wins_over_frozen(self):
        """A module whose name matches both frozen and trainable should be trainable."""
        model = _SimpleSegModel()
        # 'encoder.0' matches both frozen=("encoder",) and trainable=("encoder.0",)
        freeze_modules_by_name(
            model,
            frozen_names=("encoder",),
            trainable_names=("encoder.0",),
        )
        for name, param in model.named_parameters():
            if "encoder.0" in name:
                assert param.requires_grad


# ─── Tests: apply_fine_tuning_strategy ────────────────────────────────────────

class TestFullStrategy:
    def test_all_params_trainable(self):
        model = _SimpleSegModel()
        for p in model.parameters():
            p.requires_grad = False
        result = apply_fine_tuning_strategy(model, _cfg(strategy="full"))
        counts = get_trainable_parameter_count(result)
        assert counts["trainable"] == counts["total"]


class TestFreezeBackboneStrategy:
    def test_encoder_frozen_decoder_trainable(self):
        model = _SimpleSegModel()
        cfg = _cfg(
            strategy="freeze_backbone",
            frozen_modules=["encoder"],
            trainable_modules=["decoder", "segmentation_head"],
        )
        result = apply_fine_tuning_strategy(model, cfg)
        for name, param in result.named_parameters():
            if "encoder" in name:
                assert not param.requires_grad, f"{name} should be frozen"
            if "decoder" in name or "segmentation_head" in name:
                assert param.requires_grad, f"{name} should be trainable"

    def test_reduces_trainable_count(self):
        model = _SimpleSegModel()
        total_before = get_trainable_parameter_count(model)["total"]
        cfg = _cfg(
            strategy="freeze_backbone",
            frozen_modules=["encoder"],
            trainable_modules=["decoder", "segmentation_head"],
        )
        result = apply_fine_tuning_strategy(model, cfg)
        counts = get_trainable_parameter_count(result)
        assert counts["trainable"] < total_before


class TestLinearProbeStrategy:
    def test_only_head_trainable(self):
        model = _SimpleSegModel()
        cfg = _cfg(
            strategy="linear_probe",
            trainable_modules=["segmentation_head"],
        )
        result = apply_fine_tuning_strategy(model, cfg)
        for name, param in result.named_parameters():
            if "segmentation_head" in name:
                assert param.requires_grad, f"{name} should be trainable"
            else:
                assert not param.requires_grad, f"{name} should be frozen"

    def test_warns_when_no_head_matched(self, caplog):
        import logging
        model = _SimpleSegModel()
        cfg = _cfg(
            strategy="linear_probe",
            trainable_modules=["nonexistent_layer"],
        )
        with caplog.at_level(logging.WARNING):
            apply_fine_tuning_strategy(model, cfg)
        assert "no parameters matched" in caplog.text.lower()


class TestLoraStrategy:
    def test_lora_requires_peft(self):
        with patch.dict("sys.modules", {"peft": None}):
            model = _SimpleSegModel()
            cfg = _cfg(
                strategy="lora",
                lora_config=_cfg(r=4, lora_alpha=8, lora_dropout=0.0, bias="none", target_modules=[]),
            )
            with pytest.raises(ImportError, match="peft"):
                apply_fine_tuning_strategy(model, cfg)

    def test_lora_requires_lora_config(self):
        try:
            import peft  # noqa: F401
        except ImportError:
            pytest.skip("peft not installed")
        model = _SimpleSegModel()
        cfg = _cfg(strategy="lora", lora_config=None)
        with pytest.raises(ValueError, match="lora_config"):
            apply_fine_tuning_strategy(model, cfg)

    def test_lora_applied_to_attention_model(self):
        try:
            from peft import PeftModel
        except ImportError:
            pytest.skip("peft not installed")
        model = _AttentionModel()
        cfg = _cfg(
            strategy="lora",
            lora_config=_cfg(
                r=4,
                lora_alpha=8,
                lora_dropout=0.0,
                bias="none",
                target_modules=["query", "value"],
            ),
        )
        result = apply_fine_tuning_strategy(model, cfg)
        assert is_peft_model(result)

    def test_lora_reduces_trainable_params(self):
        try:
            from peft import PeftModel
        except ImportError:
            pytest.skip("peft not installed")
        model = _AttentionModel()
        total_before = get_trainable_parameter_count(model)["total"]
        cfg = _cfg(
            strategy="lora",
            lora_config=_cfg(
                r=4,
                lora_alpha=8,
                lora_dropout=0.0,
                bias="none",
                target_modules=["query", "value"],
            ),
        )
        result = apply_fine_tuning_strategy(model, cfg)
        counts = get_trainable_parameter_count(result)
        assert counts["trainable"] < total_before

    def test_merge_lora_weights(self):
        try:
            from peft import PeftModel
        except ImportError:
            pytest.skip("peft not installed")
        model = _AttentionModel()
        cfg = _cfg(
            strategy="lora",
            lora_config=_cfg(
                r=4,
                lora_alpha=8,
                lora_dropout=0.0,
                bias="none",
                target_modules=["query", "value"],
            ),
        )
        lora_model = apply_fine_tuning_strategy(model, cfg)
        merged = merge_lora_weights(lora_model)
        # After merge, should no longer be a PeftModel
        assert not is_peft_model(merged)
        # Forward pass should still work
        x = torch.randn(2, 64)
        out = merged(x)
        assert out.shape == (2, 2)


class TestUnknownStrategy:
    def test_raises_value_error(self):
        model = _SimpleSegModel()
        with pytest.raises(ValueError, match="Unknown fine_tuning strategy"):
            apply_fine_tuning_strategy(model, _cfg(strategy="unknown_strategy"))


# ─── Tests: is_peft_model / merge_lora_weights ────────────────────────────────

class TestPeftHelpers:
    def test_is_peft_model_false_for_plain_model(self):
        assert not is_peft_model(_SimpleSegModel())

    def test_merge_lora_weights_no_op_for_plain_model(self):
        model = _SimpleSegModel()
        result = merge_lora_weights(model)
        assert result is model  # same object returned
