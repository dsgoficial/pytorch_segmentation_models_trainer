# -*- coding: utf-8 -*-
"""
Tests for GradientReversalFunction and GradientReversalLayer.
"""

import pytest
import torch
import torch.nn as nn

from pytorch_segmentation_models_trainer.domain_adaptation.methods.gradient_reversal import (
    GradientReversalFunction,
    GradientReversalLayer,
)

# ---------------------------------------------------------------------------
# GradientReversalFunction
# ---------------------------------------------------------------------------


class TestGradientReversalFunction:
    def test_forward_is_identity(self):
        x = torch.randn(4, 8, 8, 8)
        out = GradientReversalFunction.apply(x, 1.0)
        assert torch.allclose(out, x)

    def test_forward_does_not_share_storage(self):
        """clone() means out and x are separate tensors."""
        x = torch.randn(2, 4)
        out = GradientReversalFunction.apply(x, 1.0)
        assert out.data_ptr() != x.data_ptr()

    def test_backward_reverses_gradient_lambda_one(self):
        x = torch.randn(3, 4, requires_grad=True)
        out = GradientReversalFunction.apply(x, 1.0)
        out.sum().backward()
        # Gradient of out.sum() w.r.t. x before GRL is all-ones.
        # After GRL with lambda=1: should be all -1.
        assert torch.allclose(x.grad, -torch.ones_like(x))

    def test_backward_scales_by_lambda(self):
        lambda_ = 2.5
        x = torch.randn(2, 3, requires_grad=True)
        out = GradientReversalFunction.apply(x, lambda_)
        out.sum().backward()
        assert torch.allclose(x.grad, -lambda_ * torch.ones_like(x))

    def test_backward_lambda_zero_blocks_gradient(self):
        """lambda_=0 means no gradient flows through the GRL."""
        x = torch.randn(2, 4, requires_grad=True)
        out = GradientReversalFunction.apply(x, 0.0)
        out.sum().backward()
        assert torch.allclose(x.grad, torch.zeros_like(x))

    def test_backward_negative_lambda_passes_positive_gradient(self):
        """Negative lambda inverts the inversion — gradient is positive."""
        x = torch.randn(2, 4, requires_grad=True)
        out = GradientReversalFunction.apply(x, -1.0)
        out.sum().backward()
        assert torch.allclose(x.grad, torch.ones_like(x))

    def test_works_inside_nn_graph(self):
        """GRL composes correctly with a linear layer."""
        linear = nn.Linear(4, 2)
        x = torch.randn(3, 4, requires_grad=True)
        grl_out = GradientReversalFunction.apply(x, 1.0)
        loss = linear(grl_out).sum()
        loss.backward()
        assert x.grad is not None
        assert x.grad.shape == x.shape


# ---------------------------------------------------------------------------
# GradientReversalLayer (nn.Module wrapper)
# ---------------------------------------------------------------------------


class TestGradientReversalLayer:
    def test_forward_is_identity(self):
        grl = GradientReversalLayer(lambda_=1.0)
        x = torch.randn(4, 8)
        assert torch.allclose(grl(x), x)

    def test_default_lambda_is_one(self):
        grl = GradientReversalLayer()
        assert grl.lambda_ == pytest.approx(1.0)

    def test_backward_reverses_gradient(self):
        grl = GradientReversalLayer(lambda_=1.0)
        x = torch.randn(2, 4, requires_grad=True)
        grl(x).sum().backward()
        assert torch.allclose(x.grad, -torch.ones_like(x))

    def test_set_lambda_updates_reversal_coefficient(self):
        grl = GradientReversalLayer(lambda_=1.0)
        grl.set_lambda(0.5)
        assert grl.lambda_ == pytest.approx(0.5)

    def test_set_lambda_affects_backward(self):
        grl = GradientReversalLayer(lambda_=0.0)
        x = torch.randn(2, 4, requires_grad=True)
        grl(x).sum().backward()
        # lambda=0 → zero gradient
        assert torch.allclose(x.grad, torch.zeros_like(x))

        x2 = torch.randn(2, 4, requires_grad=True)
        grl.set_lambda(3.0)
        grl(x2).sum().backward()
        assert torch.allclose(x2.grad, -3.0 * torch.ones_like(x2))

    def test_is_nn_module(self):
        assert isinstance(GradientReversalLayer(), nn.Module)

    def test_has_no_parameters(self):
        """GRL should not introduce any trainable parameters."""
        grl = GradientReversalLayer()
        assert list(grl.parameters()) == []

    def test_extra_repr_contains_lambda(self):
        grl = GradientReversalLayer(lambda_=0.1234)
        assert "lambda_" in repr(grl)

    def test_gradient_flows_to_upstream_layer(self):
        """Gradient should reach a linear layer placed before the GRL."""
        linear = nn.Linear(4, 4)
        grl = GradientReversalLayer(lambda_=1.0)
        x = torch.randn(2, 4)
        out = grl(linear(x))
        out.sum().backward()
        for p in linear.parameters():
            assert p.grad is not None
            assert p.grad.shape == p.shape

    def test_composable_in_sequential(self):
        """GRL can be used inside nn.Sequential without errors."""
        net = nn.Sequential(
            nn.Linear(8, 8),
            GradientReversalLayer(lambda_=1.0),
            nn.Linear(8, 2),
        )
        x = torch.randn(3, 8)
        loss = net(x).sum()
        loss.backward()
        for p in net.parameters():
            assert p.grad is not None
