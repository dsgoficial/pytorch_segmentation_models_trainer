# -*- coding: utf-8 -*-
"""
Tests for ModelOutputAdapter (Phase 1).
All tests run on CPU with random tensors – no pretrained weights needed.
"""
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from pytorch_segmentation_models_trainer.custom_models.transformer_adapters import (
    ModelOutputAdapter,
)


# ─── Fixtures ────────────────────────────────────────────────────────────────

B, C, H, W = 2, 4, 64, 64


def _random_input():
    return torch.randn(B, 3, H, W)


def _random_output(h=H, w=W):
    return torch.randn(B, C, h, w)


class _TensorModel(nn.Module):
    """Returns a plain tensor."""
    def forward(self, x):
        return _random_output()


class _DictOutModel(nn.Module):
    """Returns a dict with a configurable key."""
    def __init__(self, key="logits"):
        super().__init__()
        self.key = key

    def forward(self, x):
        return {self.key: _random_output()}


class _DataclassModel(nn.Module):
    """Returns a NamedTuple/dataclass-like object with .logits."""
    def forward(self, x):
        return SimpleNamespace(logits=_random_output())


class _TupleModel(nn.Module):
    """Returns a tuple (tensor, aux)."""
    def forward(self, x):
        return (_random_output(), torch.zeros(1))


class _LowResModel(nn.Module):
    """Returns logits at 1/4 resolution (e.g. Segformer)."""
    def forward(self, x):
        h, w = x.shape[-2] // 4, x.shape[-1] // 4
        return _random_output(h, w)


class _ListModel(nn.Module):
    """Returns a list of tensors (unsupported raw output)."""
    def forward(self, x):
        return [_random_output(), _random_output()]


# ─── Tests ───────────────────────────────────────────────────────────────────

class TestModelOutputAdapterPassthrough:
    def test_plain_tensor_passthrough(self):
        adapter = ModelOutputAdapter(model=_TensorModel(), output_size=None)
        out = adapter(_random_input())
        assert isinstance(out, torch.Tensor)
        assert out.shape == (B, C, H, W)

    def test_plain_tensor_is_not_copied(self):
        """When output is already correct, the same storage should be returned."""
        inner_out = _random_output()
        class _FixedModel(nn.Module):
            def forward(self, x):
                return inner_out
        adapter = ModelOutputAdapter(model=_FixedModel(), output_size=None)
        out = adapter(_random_input())
        assert out.data_ptr() == inner_out.data_ptr()


class TestModelOutputAdapterDictOutputs:
    def test_dict_logits_key(self):
        adapter = ModelOutputAdapter(model=_DictOutModel("logits"), output_size=None)
        out = adapter(_random_input())
        assert isinstance(out, torch.Tensor)
        assert out.shape[-2:] == (H, W)

    def test_dict_out_key(self):
        adapter = ModelOutputAdapter(model=_DictOutModel("out"), output_size=None)
        out = adapter(_random_input())
        assert isinstance(out, torch.Tensor)

    def test_dict_pred_key(self):
        adapter = ModelOutputAdapter(model=_DictOutModel("pred"), output_size=None)
        out = adapter(_random_input())
        assert isinstance(out, torch.Tensor)

    def test_dict_explicit_output_key(self):
        adapter = ModelOutputAdapter(
            model=_DictOutModel("custom_key"),
            output_key="custom_key",
            output_size=None,
        )
        out = adapter(_random_input())
        assert isinstance(out, torch.Tensor)

    def test_dict_wrong_explicit_key_raises(self):
        adapter = ModelOutputAdapter(
            model=_DictOutModel("logits"),
            output_key="missing_key",
            output_size=None,
        )
        with pytest.raises(KeyError, match="missing_key"):
            adapter(_random_input())

    def test_dict_no_known_key_raises(self):
        class _WeirdDictModel(nn.Module):
            def forward(self, x):
                return {"unknown_key": _random_output()}
        adapter = ModelOutputAdapter(model=_WeirdDictModel(), output_size=None)
        with pytest.raises(KeyError):
            adapter(_random_input())


class TestModelOutputAdapterDataclassOutput:
    def test_dataclass_logits_attribute(self):
        adapter = ModelOutputAdapter(model=_DataclassModel(), output_size=None)
        out = adapter(_random_input())
        assert isinstance(out, torch.Tensor)
        assert out.shape[-2:] == (H, W)


class TestModelOutputAdapterTupleOutput:
    def test_tuple_first_element(self):
        adapter = ModelOutputAdapter(model=_TupleModel(), output_size=None)
        out = adapter(_random_input())
        assert isinstance(out, torch.Tensor)
        assert out.shape[-2:] == (H, W)


class TestModelOutputAdapterUpsampling:
    def test_upsample_to_explicit_size(self):
        target = (128, 128)
        adapter = ModelOutputAdapter(model=_LowResModel(), output_size=target)
        out = adapter(_random_input())
        assert out.shape[-2:] == target

    def test_upsample_to_input_size(self):
        x = torch.randn(B, 3, H, W)
        adapter = ModelOutputAdapter(model=_LowResModel(), output_size="input")
        out = adapter(x)
        assert out.shape[-2:] == (H, W)

    def test_no_upsample_when_sizes_match(self):
        """When output size already matches, interpolation should be skipped."""
        adapter = ModelOutputAdapter(model=_TensorModel(), output_size="input")
        x = torch.randn(B, 3, H, W)
        out = adapter(x)
        assert out.shape[-2:] == (H, W)


class TestModelOutputAdapterErrors:
    def test_list_model_output_raises_type_error(self):
        adapter = ModelOutputAdapter(model=_ListModel(), output_size=None)
        # List of tensors is supported via the list branch – first element returned
        out = adapter(_random_input())
        assert isinstance(out, torch.Tensor)

    def test_unsupported_output_type_raises(self):
        class _BadModel(nn.Module):
            def forward(self, x):
                return 42  # int – unsupported
        adapter = ModelOutputAdapter(model=_BadModel(), output_size=None)
        with pytest.raises(TypeError, match="unsupported output type"):
            adapter(_random_input())
