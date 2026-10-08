# -*- coding: utf-8 -*-
"""Tests for the fine-tuning GFSS baselines (B1a isolated head, B1b ± KD)."""

import pytest
import torch

from pytorch_segmentation_models_trainer.few_shot.hierarchy import ClassHierarchy
from pytorch_segmentation_models_trainer.few_shot.methods.finetune import FineTune

NN = 254


@pytest.fixture
def data():
    torch.manual_seed(0)
    h = ClassHierarchy({1: [1, 3]}, num_base_classes=3)
    w, b = torch.randn(3, 4), torch.tensor([0.1, 0.2, 0.3])
    f = torch.randn(2, 4, 5, 5)
    ms = torch.full((2, 5, 5), NN)
    ms[:, :2] = 3
    return h, w, b, f, ms


def _ready(data, **kw):
    h, w, b, f, ms = data
    m = FineTune(**kw)
    m.setup(h, w, b, NN)
    m.init_from_support(f, ms)
    return m


class TestInit:
    def test_mother_init_copies_mother_row(self, data):
        h, w, b, f, ms = data
        m = _ready(data)
        weight, bias = m.head()
        assert weight.shape == (4, 4) and bias.shape == (4,)
        torch.testing.assert_close(weight[:3], w)
        torch.testing.assert_close(weight[3], w[1])
        torch.testing.assert_close(bias[3], b[1])

    def test_prototype_init_has_mother_norm(self, data):
        h, w, b, f, ms = data
        m = _ready(data, init="prototype")
        weight, _ = m.head()
        assert weight[3].norm().item() == pytest.approx(w[1].norm().item(), rel=1e-5)
        assert not torch.allclose(weight[3], w[1])

    def test_forward_is_linear_head(self, data):
        h, w, b, f, ms = data
        m = _ready(data)
        weight, bias = m.head()
        expected = torch.einsum("bfhw,cf->bchw", f, weight) + bias.view(1, -1, 1, 1)
        torch.testing.assert_close(m(f), expected)


class TestTrainableRows:
    @pytest.mark.parametrize(
        "rows, trainable",
        [("children", {1, 3}), ("novel", {3}), ("all", {0, 1, 2, 3})],
    )
    def test_gradients_only_reach_selected_rows(self, data, rows, trainable):
        h, w, b, f, ms = data
        m = _ready(data, train_rows=rows)
        m.support_loss(f, ms).backward()
        weight, _ = m.head()
        changed = set()
        for p in m.parameters():
            assert p.grad is not None
        opt = torch.optim.SGD(m.parameters(), lr=1.0)
        opt.step()
        new_weight, _ = m.head()
        for c in range(4):
            if not torch.allclose(new_weight[c], weight[c].detach()):
                changed.add(c)
        assert changed == trainable


class TestLoss:
    def test_projected_ce_decreases(self, data):
        h, w, b, f, ms = data
        m = _ready(data, train_rows="novel")
        opt = torch.optim.SGD(m.parameters(), lr=0.5)
        first = m.support_loss(f, ms).item()
        for _ in range(20):
            loss = m.support_loss(f, ms)
            opt.zero_grad()
            loss.backward()
            opt.step()
        assert m.support_loss(f, ms).item() < first

    def test_kd_requires_snapshot(self, data):
        h, w, b, f, ms = data
        m = _ready(data, kd_weight=10.0)
        assert m.requires_snapshot
        with pytest.raises(ValueError, match="snapshot"):
            m.support_loss(f, ms)
        snap = torch.einsum("bfhw,cf->bchw", f, w) + b.view(1, -1, 1, 1)
        with_kd = m.support_loss(f, ms, snapshot_logits=snap)
        no_kd = _ready(data).support_loss(f, ms)
        # at init the novel row copies the mother: KD is not zero (mass split) but finite
        assert torch.isfinite(with_kd) and with_kd >= no_kd

    def test_no_snapshot_needed_without_kd(self, data):
        assert not _ready(data).requires_snapshot


@pytest.mark.parametrize(
    "kw, match",
    [({"init": "random"}, "init"), ({"train_rows": "decoder"}, "train_rows")],
)
def test_invalid_arguments(kw, match):
    with pytest.raises(ValueError, match=match):
        FineTune(**kw)
