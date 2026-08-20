# -*- coding: utf-8 -*-
"""
Tests for HuggingFaceSegmentationWrapper (Phase 2).

All HuggingFace network calls are mocked so these tests run offline
without downloading any weights.

Skipped automatically when 'transformers' is not installed.  Install it
with:  pip install pytorch_segmentation_models_trainer[transformers]
"""

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
import torch.nn as nn

# Skip the entire module when transformers is not installed so that the main
# CI job (which does not install optional deps) reports SKIPPED instead of ERROR.
pytest.importorskip(
    "transformers",
    reason="transformers not installed – run: pip install '.[transformers]'",
)

from pytorch_segmentation_models_trainer.custom_models.huggingface_models import (
    HuggingFaceSegmentationWrapper,
)

# ─── Helpers ─────────────────────────────────────────────────────────────────

B, H, W = 2, 128, 128


def _make_fake_hf_model(num_labels: int, out_h: int = None, out_w: int = None):
    """Return a tiny nn.Module that mimics a HuggingFace segmentation model output."""
    _out_h = out_h or H // 4
    _out_w = out_w or W // 4

    class _FakeHFModel(nn.Module):
        def forward(self, pixel_values=None, **kwargs):
            logits = torch.randn(B, num_labels, _out_h, _out_w)
            return SimpleNamespace(logits=logits)

    return _FakeHFModel()


def _patch_auto_model(fake_model):
    """Context manager that patches AutoModelForSemanticSegmentation.from_pretrained."""
    return patch(
        "transformers.AutoModelForSemanticSegmentation.from_pretrained",
        return_value=fake_model,
    )


# ─── Tests ───────────────────────────────────────────────────────────────────


class TestHuggingFaceSegmentationWrapper:
    def test_instantiation(self):
        fake = _make_fake_hf_model(num_labels=2)
        with _patch_auto_model(fake):
            wrapper = HuggingFaceSegmentationWrapper(
                pretrained_model_name_or_path="fake/model",
                num_labels=2,
            )
        assert isinstance(wrapper, nn.Module)
        assert wrapper.num_labels == 2

    def test_forward_returns_tensor(self):
        fake = _make_fake_hf_model(num_labels=2)
        with _patch_auto_model(fake):
            wrapper = HuggingFaceSegmentationWrapper(
                pretrained_model_name_or_path="fake/model",
                num_labels=2,
            )
        x = torch.randn(B, 3, H, W)
        out = wrapper(x)
        assert isinstance(out, torch.Tensor)

    def test_output_upsampled_to_input_size(self):
        """Segformer outputs H/4 – wrapper must upsample to H."""
        fake = _make_fake_hf_model(num_labels=2, out_h=H // 4, out_w=W // 4)
        with _patch_auto_model(fake):
            wrapper = HuggingFaceSegmentationWrapper(
                pretrained_model_name_or_path="fake/model",
                num_labels=2,
                output_size="input",
            )
        x = torch.randn(B, 3, H, W)
        out = wrapper(x)
        assert out.shape == (B, 2, H, W)

    def test_output_shape_num_labels(self):
        for num_labels in [2, 5, 19]:
            fake = _make_fake_hf_model(num_labels=num_labels)
            with _patch_auto_model(fake):
                wrapper = HuggingFaceSegmentationWrapper(
                    pretrained_model_name_or_path="fake/model",
                    num_labels=num_labels,
                )
            x = torch.randn(B, 3, H, W)
            out = wrapper(x)
            assert out.shape[1] == num_labels, f"Failed for num_labels={num_labels}"

    def test_output_is_plain_tensor_not_dataclass(self):
        fake = _make_fake_hf_model(num_labels=2)
        with _patch_auto_model(fake):
            wrapper = HuggingFaceSegmentationWrapper(
                pretrained_model_name_or_path="fake/model",
                num_labels=2,
            )
        x = torch.randn(B, 3, H, W)
        out = wrapper(x)
        assert type(out) is torch.Tensor

    def test_import_error_when_transformers_missing(self):
        with patch.dict("sys.modules", {"transformers": None}):
            with pytest.raises(ImportError, match="transformers"):
                HuggingFaceSegmentationWrapper(
                    pretrained_model_name_or_path="fake/model",
                    num_labels=2,
                )

    def test_from_pretrained_called_with_correct_args(self):
        fake = _make_fake_hf_model(num_labels=3)
        with patch(
            "transformers.AutoModelForSemanticSegmentation.from_pretrained",
            return_value=fake,
        ) as mock_fp:
            HuggingFaceSegmentationWrapper(
                pretrained_model_name_or_path="some/path",
                num_labels=3,
                ignore_mismatched_sizes=True,
            )
            mock_fp.assert_called_once_with(
                "some/path",
                num_labels=3,
                ignore_mismatched_sizes=True,
            )
