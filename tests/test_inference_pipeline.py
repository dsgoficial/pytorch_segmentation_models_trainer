# -*- coding: utf-8 -*-
"""
Tests for the inference pipeline (Phase 7).

Validates LoRA merge-before-inference, model extraction from checkpoints,
and that is_peft_model / merge_lora_weights behave correctly.
"""
from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn as nn

from pytorch_segmentation_models_trainer.fine_tuning.lora_utils import (
    is_peft_model,
    merge_lora_weights,
)


# ─── Helpers ─────────────────────────────────────────────────────────────────

class _SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(4, 2)

    def forward(self, x):
        return self.fc(x)


# ─── Tests: LoRA helpers ──────────────────────────────────────────────────────

class TestLoraInferencePrepHelpers:
    def test_is_peft_model_false_without_peft(self):
        with patch.dict("sys.modules", {"peft": None}):
            model = _SimpleModel()
            assert not is_peft_model(model)

    def test_is_peft_model_false_for_plain_model(self):
        assert not is_peft_model(_SimpleModel())

    def test_merge_lora_weights_returns_same_model_when_not_peft(self):
        model = _SimpleModel()
        result = merge_lora_weights(model)
        assert result is model

    def test_merge_lora_weights_noop_without_peft(self):
        with patch.dict("sys.modules", {"peft": None}):
            model = _SimpleModel()
            result = merge_lora_weights(model)
            assert result is model


class TestMergeLoraBeforeInference:
    def test_merge_called_when_is_peft_model(self):
        """Verify the predict.py path calls merge_and_unload on a PeftModel."""
        # Simulate a PeftModel by monkey-patching is_peft_model
        model = _SimpleModel()
        model.merge_and_unload = MagicMock(return_value=model)

        with patch(
            "pytorch_segmentation_models_trainer.fine_tuning.lora_utils.is_peft_model",
            return_value=True,
        ), patch(
            "pytorch_segmentation_models_trainer.fine_tuning.lora_utils.PeftModel",
            create=True,
        ):
            result = merge_lora_weights(model)
            model.merge_and_unload.assert_called_once()


# ─── Tests: inference model eval mode ─────────────────────────────────────────

class TestInstantiateModelFromCheckpoint:
    def test_model_is_in_eval_mode_after_load(self, tmp_path):
        """
        Smoke-test: load a fake checkpoint and confirm model.eval() was called.
        We mock the entire checkpoint load chain to avoid real file I/O.
        """
        from omegaconf import OmegaConf

        cfg = OmegaConf.create({
            "checkpoint_path": str(tmp_path / "fake.ckpt"),
            "pl_model": {
                "_target_": "pytorch_segmentation_models_trainer.model_loader.model.Model",
            },
            "pl_trainer": {"devices": [0]},
            "device": "cpu",
            "model": {
                "_target_": "segmentation_models_pytorch.Unet",
                "encoder_name": "resnet18",
                "encoder_weights": None,
                "in_channels": 3,
                "classes": 2,
            },
            "loss": {"_target_": "torch.nn.CrossEntropyLoss"},
            "hyperparameters": {"batch_size": 2, "devices": 1, "accelerator": "cpu"},
            "keep_lora_adapters": False,
        })

        inner_model = _SimpleModel()
        fake_pl_model = MagicMock()
        fake_pl_model.model = inner_model

        with patch(
            "pytorch_segmentation_models_trainer.predict.import_module_from_cfg"
        ) as mock_import, patch(
            "pytorch_segmentation_models_trainer.predict.is_peft_model",
            return_value=False,
        ):
            mock_cls = MagicMock()
            mock_cls.load_from_checkpoint.return_value = fake_pl_model
            mock_import.return_value = mock_cls

            from pytorch_segmentation_models_trainer.predict import (
                instantiate_model_from_checkpoint,
            )

            # Patch cuda to be unavailable so map_location = "cpu"
            with patch("torch.cuda.is_available", return_value=False):
                result = instantiate_model_from_checkpoint(cfg)

        # The returned model should be the inner model in eval mode
        assert not result.training

    def test_lora_merge_called_when_peft_model(self, tmp_path):
        from omegaconf import OmegaConf

        cfg = OmegaConf.create({
            "checkpoint_path": str(tmp_path / "fake.ckpt"),
            "pl_model": {"_target_": "pytorch_segmentation_models_trainer.model_loader.model.Model"},
            "pl_trainer": {"devices": [0]},
            "device": "cpu",
            "model": {
                "_target_": "segmentation_models_pytorch.Unet",
                "encoder_name": "resnet18",
                "encoder_weights": None,
                "in_channels": 3,
                "classes": 2,
            },
            "loss": {"_target_": "torch.nn.CrossEntropyLoss"},
            "hyperparameters": {"batch_size": 2, "devices": 1, "accelerator": "cpu"},
            "keep_lora_adapters": False,
        })

        inner_model = _SimpleModel()
        inner_model.merge_and_unload = MagicMock(return_value=inner_model)
        fake_pl_model = MagicMock()
        fake_pl_model.model = inner_model

        with patch(
            "pytorch_segmentation_models_trainer.predict.import_module_from_cfg"
        ) as mock_import, patch(
            "pytorch_segmentation_models_trainer.predict.is_peft_model",
            return_value=True,
        ), patch(
            "pytorch_segmentation_models_trainer.predict.merge_lora_weights",
            return_value=inner_model,
        ) as mock_merge, patch(
            "torch.cuda.is_available", return_value=False
        ):
            mock_cls = MagicMock()
            mock_cls.load_from_checkpoint.return_value = fake_pl_model
            mock_import.return_value = mock_cls

            from pytorch_segmentation_models_trainer.predict import (
                instantiate_model_from_checkpoint,
            )
            instantiate_model_from_checkpoint(cfg)

        mock_merge.assert_called_once()
