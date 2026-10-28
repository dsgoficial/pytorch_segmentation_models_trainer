# -*- coding: utf-8 -*-
"""Tests for TripleGateEnsemble (average of 3 independently-gated HiSplit experts)."""

import pytest
import torch

from pytorch_segmentation_models_trainer.few_shot.hierarchy import ClassHierarchy
from pytorch_segmentation_models_trainer.few_shot.methods.hisplit import HiSplit
from pytorch_segmentation_models_trainer.few_shot.methods.q_strategies import (
    LinearAefGateStrategy,
    TransStrategy,
)
from pytorch_segmentation_models_trainer.few_shot.methods.triple_gate_ensemble import TripleGateEnsemble

NN = 254


def _data(seed=0):
    """Same shape/semantics as test_dual_gate_moe.py's own ``_data``."""
    g = torch.Generator().manual_seed(seed)
    f = torch.randn(2, 4, 6, 6, generator=g) * 0.1
    f[:, 0, :, 3:] += 3.0
    f[:, 1, :3, 3:] += 2.0
    f[:, 1, 3:, 3:] -= 2.0
    w = torch.zeros(3, 4)
    w[1, 0] = 2.0
    w[0, 0] = -2.0
    b = torch.tensor([0.0, -1.0, 0.5])
    masks = torch.full((2, 6, 6), NN)
    masks[:, :3, 3:] = 3
    return f, masks, w, b


def _hierarchy():
    return ClassHierarchy({1: [1, 3]}, num_base_classes=3)


def _gate_expert(aef_dim):
    return HiSplit(q=LinearAefGateStrategy(aef_gate_epochs=5, aef_gate_lr=1e-2), aef_dim=aef_dim)


def _ready_moe(f, masks, w, b, expert_a=None, expert_b=None, expert_c=None, **kw):
    m = TripleGateEnsemble(
        expert_a=expert_a or _gate_expert(8),
        expert_b=expert_b or _gate_expert(5),
        expert_c=expert_c or _gate_expert(3),
        **kw,
    )
    torch.manual_seed(42)
    m.setup(_hierarchy(), w, b, NN)
    g = torch.Generator().manual_seed(1)
    aux_a = torch.randn(2, 8, 6, 6, generator=g)
    aux_b = torch.randn(2, 5, 6, 6, generator=g)
    aux_c = torch.randn(2, 3, 6, 6, generator=g)
    m.set_aux({"aef": aux_a, "olmoearth": aux_b, "tessera": aux_c})
    m.init_from_support(f, masks)
    return m, aux_a, aux_b, aux_c


class TestConstruction:
    def test_builds_from_plain_hisplit_instances(self):
        m = TripleGateEnsemble(expert_a=_gate_expert(8), expert_b=_gate_expert(5), expert_c=_gate_expert(3))
        assert isinstance(m.expert_a, HiSplit)
        assert isinstance(m.expert_b, HiSplit)
        assert isinstance(m.expert_c, HiSplit)

    def test_non_hisplit_expert_raises_clear_error(self):
        with pytest.raises(TypeError, match="HiSplit"):
            TripleGateEnsemble(
                expert_a={
                    "_target_": "pytorch_segmentation_models_trainer.few_shot.methods.base_only.BaseOnly"
                },
                expert_b=_gate_expert(5),
                expert_c=_gate_expert(3),
            )

    def test_duplicate_aux_names_raises(self):
        with pytest.raises(ValueError, match="must all differ"):
            TripleGateEnsemble(
                expert_a=_gate_expert(8),
                expert_b=_gate_expert(5),
                expert_c=_gate_expert(3),
                aux_name_a="aef",
                aux_name_c="aef",
            )

    def test_aux_names_default_to_aef_olmoearth_tessera(self):
        m = TripleGateEnsemble(expert_a=_gate_expert(8), expert_b=_gate_expert(5), expert_c=_gate_expert(3))
        assert m.aux_names == ("aef", "olmoearth", "tessera")


class TestSetupAndAuxRouting:
    def test_setup_propagates_hierarchy_and_base_to_all_three_experts(self):
        f, masks, w, b = _data()
        m = TripleGateEnsemble(expert_a=_gate_expert(8), expert_b=_gate_expert(5), expert_c=_gate_expert(3))
        h = _hierarchy()
        m.setup(h, w, b, NN)
        assert m.expert_a.hierarchy is h and m.expert_b.hierarchy is h and m.expert_c.hierarchy is h
        torch.testing.assert_close(m.expert_c.base_weight, w)

    def test_set_aux_routes_each_name_to_the_right_expert(self):
        m = TripleGateEnsemble(expert_a=_gate_expert(8), expert_b=_gate_expert(5), expert_c=_gate_expert(3))
        m.setup(_hierarchy(), *_data()[2:], NN)
        aux_a, aux_b, aux_c = torch.randn(2, 8, 6, 6), torch.randn(2, 5, 6, 6), torch.randn(2, 3, 6, 6)
        m.set_aux({"aef": aux_a, "olmoearth": aux_b, "tessera": aux_c})
        assert m.expert_a._aux is aux_a
        assert m.expert_b._aux is aux_b
        assert m.expert_c._aux is aux_c

    def test_set_aux_missing_name_degrades_to_none(self):
        m = TripleGateEnsemble(expert_a=_gate_expert(8), expert_b=_gate_expert(5), expert_c=_gate_expert(3))
        m.setup(_hierarchy(), *_data()[2:], NN)
        m.set_aux({"aef": torch.randn(2, 8, 6, 6)})
        assert m.expert_a._aux is not None
        assert m.expert_b._aux is None
        assert m.expert_c._aux is None


class TestInitAndForward:
    def test_forward_shape(self):
        f, masks, w, b = _data()
        m, *_ = _ready_moe(f, masks, w, b)
        out = m(f)
        assert out.shape == (2, 4, 6, 6)

    def test_forward_is_the_average_of_all_three_experts_q_decoded_by_expert_a(self):
        f, masks, w, b = _data()
        m, *_ = _ready_moe(f, masks, w, b)
        q_a = m.expert_a._compute_q(f)
        q_b = m.expert_b._compute_q(f)
        q_c = m.expert_c._compute_q(f)
        expected = m.expert_a.forward(f, q=(q_a + q_b + q_c) / 3.0)
        torch.testing.assert_close(m.forward(f), expected)

    def test_forward_differs_from_any_single_expert_or_the_2_expert_average(self):
        f, masks, w, b = _data()
        m, *_ = _ready_moe(f, masks, w, b)
        out = m.forward(f)
        assert not torch.allclose(out, m.expert_a.forward(f))
        q_ab = 0.5 * (m.expert_a._compute_q(f) + m.expert_b._compute_q(f))
        two_way = m.expert_a.forward(f, q=q_ab)
        assert not torch.allclose(out, two_way)

    def test_decode_time_settings_come_from_expert_a_only(self):
        f, masks, w, b = _data()
        expert_a = HiSplit(q=LinearAefGateStrategy(aef_gate_epochs=5), aef_dim=8, decoding="flat")
        expert_b = HiSplit(q=LinearAefGateStrategy(aef_gate_epochs=5), aef_dim=5, decoding="hierarchical")
        expert_c = HiSplit(q=LinearAefGateStrategy(aef_gate_epochs=5), aef_dim=3, decoding="hierarchical")
        m, *_ = _ready_moe(f, masks, w, b, expert_a=expert_a, expert_b=expert_b, expert_c=expert_c)
        q = (m.expert_a._compute_q(f) + m.expert_b._compute_q(f) + m.expert_c._compute_q(f)) / 3.0
        expected_flat = torch.log(m.expert_a.split_probabilities(f, q=q) + 1e-10)
        torch.testing.assert_close(m.forward(f), expected_flat)


class TestSupportLossDelegation:
    def test_all_non_trainable_experts_yield_no_loss(self):
        f, masks, w, b = _data()
        m, *_ = _ready_moe(f, masks, w, b)
        assert m.support_loss(f, masks) is None

    def test_three_trainable_experts_returns_their_average(self):
        f, masks, w, b = _data()
        expert_a, expert_b, expert_c = HiSplit(q="linear"), HiSplit(q="linear"), HiSplit(q="linear")
        m = TripleGateEnsemble(expert_a=expert_a, expert_b=expert_b, expert_c=expert_c)
        m.setup(_hierarchy(), w, b, NN)
        m.init_from_support(f, masks)
        loss = m.support_loss(f, masks)
        expected = (
            expert_a.support_loss(f, masks)
            + expert_b.support_loss(f, masks)
            + expert_c.support_loss(f, masks)
        ) / 3.0
        torch.testing.assert_close(loss, expected)


class TestTransductiveDelegation:
    def test_transductive_true_when_any_expert_is(self):
        m = TripleGateEnsemble(
            expert_a=HiSplit(q=TransStrategy()), expert_b=_gate_expert(5), expert_c=_gate_expert(3)
        )
        assert m.transductive is True

    def test_transductive_false_when_none_is(self):
        m = TripleGateEnsemble(expert_a=_gate_expert(8), expert_b=_gate_expert(5), expert_c=_gate_expert(3))
        assert m.transductive is False

    def test_adapt_to_query_only_calls_the_transductive_expert(self):
        f, masks, w, b = _data()
        expert_a = HiSplit(q=TransStrategy(adapt_iter=2))
        expert_b, expert_c = _gate_expert(5), _gate_expert(3)
        m = TripleGateEnsemble(expert_a=expert_a, expert_b=expert_b, expert_c=expert_c)
        m.setup(_hierarchy(), w, b, NN)
        m.set_aux(
            {"aef": torch.randn(2, 8, 6, 6), "olmoearth": torch.randn(2, 5, 6, 6), "tessera": torch.randn(2, 3, 6, 6)}
        )
        m.init_from_support(f, masks)
        assert expert_a._task_params is None
        m.adapt_to_query(f, masks, f)
        assert expert_a._task_params is not None
