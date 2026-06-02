# -*- coding: utf-8 -*-
"""
Tests for FeatureExtractorHook.
"""

import pytest
import torch
import torch.nn as nn

from pytorch_segmentation_models_trainer.domain_adaptation.feature_hooks import (
    FeatureExtractorHook,
)

# ---------------------------------------------------------------------------
# Helper model with named layers
# ---------------------------------------------------------------------------


class _SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(8, 16),
            nn.ReLU(),
        )
        self.decoder = nn.Linear(16, 4)

    def forward(self, x):
        feat = self.encoder(x)
        return self.decoder(feat)


class _NestedModel(nn.Module):
    """Two-level nesting: model.encoder.layer1"""

    def __init__(self):
        super().__init__()
        self.encoder = _Encoder()
        self.head = nn.Linear(16, 2)

    def forward(self, x):
        return self.head(self.encoder(x))


class _Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(8, 16)

    def forward(self, x):
        return self.layer1(x)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestFeatureExtractorHook:
    def test_captures_output_of_registered_layer(self):
        model = _SimpleModel()
        hook = FeatureExtractorHook(model, ["encoder"])
        x = torch.randn(2, 8)
        model(x)
        features = hook.get_features()

        assert "encoder" in features
        assert features["encoder"].shape == (2, 16)
        hook.remove()

    def test_captures_multiple_layers(self):
        model = _NestedModel()
        hook = FeatureExtractorHook(model, ["encoder", "head"])
        x = torch.randn(3, 8)
        model(x)
        features = hook.get_features()

        assert "encoder" in features
        assert "head" in features
        hook.remove()

    def test_captures_nested_layer_by_dot_path(self):
        model = _NestedModel()
        hook = FeatureExtractorHook(model, ["encoder.layer1"])
        x = torch.randn(2, 8)
        model(x)
        features = hook.get_features()

        assert "encoder.layer1" in features
        assert features["encoder.layer1"].shape == (2, 16)
        hook.remove()

    def test_get_features_returns_shallow_copy(self):
        model = _SimpleModel()
        hook = FeatureExtractorHook(model, ["encoder"])
        x = torch.randn(2, 8)
        model(x)
        f1 = hook.get_features()
        f2 = hook.get_features()
        # Different dict objects but same tensors
        assert f1 is not f2
        assert f1["encoder"] is f2["encoder"]
        hook.remove()

    def test_clear_empties_stored_features(self):
        model = _SimpleModel()
        hook = FeatureExtractorHook(model, ["encoder"])
        x = torch.randn(2, 8)
        model(x)
        assert "encoder" in hook.get_features()

        hook.clear()
        assert hook.get_features() == {}
        hook.remove()

    def test_features_updated_on_each_forward(self):
        model = _SimpleModel()
        hook = FeatureExtractorHook(model, ["encoder"])
        x1 = torch.zeros(2, 8)
        x2 = torch.ones(2, 8)

        model(x1)
        feat1 = hook.get_features()["encoder"].clone()

        hook.clear()
        model(x2)
        feat2 = hook.get_features()["encoder"].clone()

        assert not torch.allclose(feat1, feat2)
        hook.remove()

    def test_remove_deregisters_hooks(self):
        model = _SimpleModel()
        hook = FeatureExtractorHook(model, ["encoder"])
        hook.remove()

        # After removal, forward should not update features
        x = torch.randn(2, 8)
        model(x)
        assert hook.get_features() == {}

    def test_invalid_layer_name_raises_value_error(self):
        model = _SimpleModel()
        with pytest.raises(ValueError, match="not found"):
            FeatureExtractorHook(model, ["nonexistent_layer"])

    def test_invalid_nested_name_raises_value_error(self):
        model = _SimpleModel()
        with pytest.raises(ValueError, match="not found"):
            FeatureExtractorHook(model, ["encoder.nonexistent"])

    def test_non_module_attribute_raises_value_error(self):
        """Resolving to a non-Module attribute should raise."""

        class _ModelWithAttr(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(4, 4)
                self.stride = 2  # plain int, not a Module

            def forward(self, x):
                return self.linear(x)

        model = _ModelWithAttr()
        with pytest.raises(ValueError, match="expected nn.Module"):
            FeatureExtractorHook(model, ["stride"])
