# -*- coding: utf-8 -*-
"""Tests for the prototype-imprinting GFSS baseline."""

import pytest
import torch

from pytorch_segmentation_models_trainer.few_shot.hierarchy import ClassHierarchy
from pytorch_segmentation_models_trainer.few_shot.losses import novel_prototypes
from pytorch_segmentation_models_trainer.few_shot.methods.prototype import (
    PrototypeImprinting,
)


@pytest.fixture
def data():
    torch.manual_seed(0)
    h = ClassHierarchy({1: [1, 2], 0: [0, 3]}, num_base_classes=2)
    fs = torch.randn(2, 4, 3, 3)
    ms = torch.full((2, 3, 3), 254)
    ms[:, 0] = 2
    ms[:, 1] = 3
    return h, fs, ms


def _ready(h, fs, ms, scale):
    m = PrototypeImprinting(scale=scale)
    w = torch.randn(2, 4)
    m.setup(h, w, torch.tensor([0.1, -0.1]), 254)
    m.init_from_support(fs, ms)
    return m, w


@pytest.mark.parametrize("scale", ["base_norm", "unit", 2.5])
def test_imprinted_rows(data, scale):
    h, fs, ms = data
    m, w = _ready(h, fs, ms, scale)
    proto = novel_prototypes(fs, ms, h).T
    factor = {"base_norm": w.norm(dim=1).mean(), "unit": 1.0, 2.5: 2.5}[scale]
    torch.testing.assert_close(m.novel_weight, proto * factor)


def test_forward_keeps_base_logits(data):
    h, fs, ms = data
    m, w = _ready(h, fs, ms, "base_norm")
    x = torch.randn(1, 4, 3, 3)
    out = m(x)
    assert out.shape == (1, 4, 3, 3)
    torch.testing.assert_close(
        out[:, :2],
        torch.einsum("bfhw,cf->bchw", x, w)
        + torch.tensor([0.1, -0.1]).view(1, 2, 1, 1),
    )
    assert not m.transductive
    assert m.support_loss(fs, ms) is None


def test_invalid_scale():
    with pytest.raises(ValueError, match="scale"):
        PrototypeImprinting(scale="cosine")
