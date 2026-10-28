# -*- coding: utf-8 -*-
"""Tests for GateEnsemble (average of 2 independently-gated HiSplit experts)."""

import pytest
import torch

from pytorch_segmentation_models_trainer.few_shot.hierarchy import ClassHierarchy
from pytorch_segmentation_models_trainer.few_shot.methods.gate_ensemble import GateEnsemble
from pytorch_segmentation_models_trainer.few_shot.methods.hisplit import HiSplit
from pytorch_segmentation_models_trainer.few_shot.methods.q_strategies import (
    LinearAefGateStrategy,
    TransStrategy,
)

NN = 254


def _data(seed=0):
    """Same shape/semantics as test_few_shot_hisplit.py's own ``_data``:
    3 base classes (0, 1, 2); mother 1 -> children 1 (kept) and 3 (novel)."""
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
    return HiSplit(
        q=LinearAefGateStrategy(aef_gate_epochs=5, aef_gate_lr=1e-2),
        aef_dim=aef_dim,
    )


def _ready_moe(f, masks, w, b, expert_a=None, expert_b=None, **kw):
    m = GateEnsemble(
        expert_a=expert_a or _gate_expert(aef_dim=8),
        expert_b=expert_b or _gate_expert(aef_dim=5),
        **kw,
    )
    # HiSplit.setup() builds aef_gate_linear as a plain nn.Linear (PyTorch's
    # default, global-RNG-dependent init) -- seed the global RNG right
    # before it so this fixture is deterministic regardless of whatever
    # other tests ran earlier in the same process.
    torch.manual_seed(42)
    m.setup(_hierarchy(), w, b, NN)
    g = torch.Generator().manual_seed(1)
    aux_a = torch.randn(2, 8, 6, 6, generator=g)
    aux_b = torch.randn(2, 5, 6, 6, generator=g)
    m.set_aux({"aef": aux_a, "olmoearth": aux_b})
    m.init_from_support(f, masks)
    return m, aux_a, aux_b


class TestConstruction:
    def test_builds_from_plain_hisplit_instances(self):
        m = GateEnsemble(expert_a=_gate_expert(8), expert_b=_gate_expert(5))
        assert isinstance(m.expert_a, HiSplit)
        assert isinstance(m.expert_b, HiSplit)

    def test_builds_from_hydra_style_subconfigs(self):
        cfg_a = {
            "_target_": "pytorch_segmentation_models_trainer.few_shot.methods.hisplit.HiSplit",
            "q": {
                "_target_": "pytorch_segmentation_models_trainer.few_shot.methods.q_strategies.LinearAefGateStrategy",
                "aef_gate_epochs": 5,
            },
            "aef_dim": 8,
        }
        cfg_b = {**cfg_a, "aef_dim": 5}
        m = GateEnsemble(expert_a=cfg_a, expert_b=cfg_b)
        assert isinstance(m.expert_a, HiSplit)
        assert m.expert_a.aef_dim == 8
        assert m.expert_b.aef_dim == 5

    def test_non_hisplit_expert_raises_clear_error(self):
        from pytorch_segmentation_models_trainer.few_shot.methods.base_only import BaseOnly

        with pytest.raises(TypeError, match="HiSplit"):
            GateEnsemble(
                expert_a={
                    "_target_": "pytorch_segmentation_models_trainer.few_shot.methods.base_only.BaseOnly"
                },
                expert_b=_gate_expert(5),
            )

    def test_duplicate_aux_names_raises(self):
        with pytest.raises(ValueError, match="must differ"):
            GateEnsemble(
                expert_a=_gate_expert(8),
                expert_b=_gate_expert(5),
                aux_name_a="aef",
                aux_name_b="aef",
            )

    def test_aux_names_default_to_aef_and_olmoearth(self):
        m = GateEnsemble(expert_a=_gate_expert(8), expert_b=_gate_expert(5))
        assert m.aux_names == ("aef", "olmoearth")

    def test_custom_aux_names(self):
        m = GateEnsemble(
            expert_a=_gate_expert(8),
            expert_b=_gate_expert(5),
            aux_name_a="product_a",
            aux_name_b="product_b",
        )
        assert m.aux_names == ("product_a", "product_b")


class TestSetupAndAuxRouting:
    def test_setup_propagates_hierarchy_and_base_to_both_experts(self):
        f, masks, w, b = _data()
        m = GateEnsemble(expert_a=_gate_expert(8), expert_b=_gate_expert(5))
        h = _hierarchy()
        m.setup(h, w, b, NN)
        assert m.expert_a.hierarchy is h and m.expert_b.hierarchy is h
        torch.testing.assert_close(m.expert_a.base_weight, w)
        torch.testing.assert_close(m.expert_b.base_weight, w)

    def test_set_aux_routes_each_name_to_the_right_expert(self):
        m = GateEnsemble(
            expert_a=_gate_expert(8), expert_b=_gate_expert(5), aux_name_b="olmoearth"
        )
        m.setup(_hierarchy(), *_data()[2:], NN)
        aux_a = torch.randn(2, 8, 6, 6)
        aux_b = torch.randn(2, 5, 6, 6)
        m.set_aux({"aef": aux_a, "olmoearth": aux_b})
        assert m.expert_a._aux is aux_a
        assert m.expert_b._aux is aux_b

    def test_set_aux_missing_name_degrades_to_none(self):
        """A name in aux_names may be missing from aux (e.g. a missing cache
        tile) -- each expert must degrade gracefully, not raise."""
        m = GateEnsemble(expert_a=_gate_expert(8), expert_b=_gate_expert(5))
        m.setup(_hierarchy(), *_data()[2:], NN)
        m.set_aux({"aef": torch.randn(2, 8, 6, 6)})
        assert m.expert_a._aux is not None
        assert m.expert_b._aux is None


class TestInitAndForward:
    def test_init_from_support_fits_both_experts_independently(self):
        f, masks, w, b = _data()
        m, _, _ = _ready_moe(f, masks, w, b)
        # Different aux width per expert (8 vs 5) -> independently-shaped gates.
        assert m.expert_a.aef_gate_linear.weight.shape[1] == 8
        assert m.expert_b.aef_gate_linear.weight.shape[1] == 5
        assert torch.isfinite(m.expert_a.q_weight).all()
        assert torch.isfinite(m.expert_b.q_weight).all()

    def test_forward_shape(self):
        f, masks, w, b = _data()
        m, _, _ = _ready_moe(f, masks, w, b)
        out = m(f)
        assert out.shape == (2, 4, 6, 6)

    def test_forward_is_the_average_of_both_experts_q_decoded_by_expert_a(self):
        f, masks, w, b = _data()
        m, aux_a, aux_b = _ready_moe(f, masks, w, b)
        q_a = m.expert_a._compute_q(f)
        q_b = m.expert_b._compute_q(f)
        expected = m.expert_a.forward(f, q=0.5 * (q_a + q_b))
        torch.testing.assert_close(m.forward(f), expected)

    def test_forward_differs_from_either_expert_alone(self):
        """Proves this is actually averaging, not silently delegating to
        one expert -- the 2 experts see different aux content/width, so
        their own q's (and hence the single-expert decode) differ from the
        averaged decode."""
        f, masks, w, b = _data()
        m, _, _ = _ready_moe(f, masks, w, b)
        out = m.forward(f)
        assert not torch.allclose(out, m.expert_a.forward(f))
        assert not torch.allclose(out, m.expert_b.forward(f))

    def test_decode_time_settings_come_from_expert_a_only(self):
        """expert_b's own `decoding`/widen/etc. are never consulted --
        only its q. Here expert_a decodes `flat`, expert_b `hierarchical`;
        the composite's output must match a `flat` decode of the averaged
        q, not expert_b's hierarchical one."""
        f, masks, w, b = _data()
        expert_a = HiSplit(
            q=LinearAefGateStrategy(aef_gate_epochs=5), aef_dim=8, decoding="flat"
        )
        expert_b = HiSplit(
            q=LinearAefGateStrategy(aef_gate_epochs=5), aef_dim=5, decoding="hierarchical"
        )
        m, _, _ = _ready_moe(f, masks, w, b, expert_a=expert_a, expert_b=expert_b)
        q = 0.5 * (m.expert_a._compute_q(f) + m.expert_b._compute_q(f))
        expected_flat = torch.log(m.expert_a.split_probabilities(f, q=q) + 1e-10)
        torch.testing.assert_close(m.forward(f), expected_flat)


class TestSupportLossDelegation:
    def test_both_non_trainable_experts_yield_no_loss(self):
        f, masks, w, b = _data()
        m, _, _ = _ready_moe(f, masks, w, b)
        assert m.support_loss(f, masks) is None

    def test_one_trainable_expert_returns_its_own_loss(self):
        f, masks, w, b = _data()
        expert_a = HiSplit(q="linear")
        expert_b = _gate_expert(5)
        m, _, _ = _ready_moe(f, masks, w, b, expert_a=expert_a, expert_b=expert_b)
        loss = m.support_loss(f, masks)
        expected = expert_a.support_loss(f, masks)
        assert loss is not None
        torch.testing.assert_close(loss, expected)

    def test_two_trainable_experts_returns_their_average(self):
        f, masks, w, b = _data()
        expert_a = HiSplit(q="linear")
        expert_b = HiSplit(q="linear")
        m = GateEnsemble(expert_a=expert_a, expert_b=expert_b)
        m.setup(_hierarchy(), w, b, NN)
        m.init_from_support(f, masks)
        loss = m.support_loss(f, masks)
        expected = 0.5 * (expert_a.support_loss(f, masks) + expert_b.support_loss(f, masks))
        torch.testing.assert_close(loss, expected)


class TestTransductiveDelegation:
    def test_transductive_true_when_either_expert_is(self):
        m = GateEnsemble(expert_a=HiSplit(q=TransStrategy()), expert_b=_gate_expert(5))
        assert m.transductive is True

    def test_transductive_false_when_neither_is(self):
        m = GateEnsemble(expert_a=_gate_expert(8), expert_b=_gate_expert(5))
        assert m.transductive is False

    def test_adapt_to_query_only_calls_the_transductive_expert(self):
        f, masks, w, b = _data()
        expert_a = HiSplit(q=TransStrategy(adapt_iter=2))
        expert_b = _gate_expert(5)
        m = GateEnsemble(expert_a=expert_a, expert_b=expert_b)
        m.setup(_hierarchy(), w, b, NN)
        m.set_aux({"aef": torch.randn(2, 8, 6, 6), "olmoearth": torch.randn(2, 5, 6, 6)})
        m.init_from_support(f, masks)
        assert expert_a._task_params is None
        m.adapt_to_query(f, masks, f)
        assert expert_a._task_params is not None

    def test_adapt_to_query_works_when_expert_b_is_the_transductive_one(self):
        f, masks, w, b = _data()
        expert_a = _gate_expert(8)
        expert_b = HiSplit(q=TransStrategy(adapt_iter=2))
        m = GateEnsemble(expert_a=expert_a, expert_b=expert_b)
        m.setup(_hierarchy(), w, b, NN)
        m.set_aux({"aef": torch.randn(2, 8, 6, 6), "olmoearth": torch.randn(2, 5, 6, 6)})
        m.init_from_support(f, masks)
        assert expert_b._task_params is None
        m.adapt_to_query(f, masks, f)
        assert expert_b._task_params is not None
