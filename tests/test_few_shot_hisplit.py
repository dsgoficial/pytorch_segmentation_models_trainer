# -*- coding: utf-8 -*-
"""Tests for HiSplit (hierarchical split of a base class)."""

import pytest
import torch

from pytorch_segmentation_models_trainer.few_shot.hierarchy import ClassHierarchy
from pytorch_segmentation_models_trainer.few_shot.methods.hisplit import HiSplit

NN = 254


def _data(seed=0):
    """3 base classes (0, 1, 2); mother 1 -> children 1 (kept) and 3 (novel).

    Features: dim 0 drives the base classifier (class 1 where large), dim 1
    separates child 3 (positive) from child 1 (negative).
    """
    g = torch.Generator().manual_seed(seed)
    f = torch.randn(2, 4, 6, 6, generator=g) * 0.1
    f[:, 0, :, 3:] += 3.0  # right half -> base class 1 (mother)
    f[:, 1, :3, 3:] += 2.0  # top-right -> novel 3
    f[:, 1, 3:, 3:] -= 2.0  # bottom-right -> child 1
    w = torch.zeros(3, 4)
    w[1, 0] = 2.0
    w[0, 0] = -2.0
    b = torch.tensor([0.0, -1.0, 0.5])
    masks = torch.full((2, 6, 6), NN)
    masks[:, :3, 3:] = 3  # S-novel: only the novel class labelled
    return f, masks, w, b


def _ready(f, masks, w, b, hierarchy=None, **kw):
    h = hierarchy or ClassHierarchy({1: [1, 3]}, num_base_classes=3)
    m = HiSplit(**kw)
    m.setup(h, w, b, NN)
    m.init_from_support(f, masks)
    return m


class TestTargetsAndInit:
    def test_free_negatives_from_base_prediction(self):
        f, masks, w, b = _data()
        m = _ready(f, masks, w, b)
        t = m._targets(f, masks)
        assert m.child_classes == [1, 3]
        assert (t[:, :3, 3:] == 1).all()  # novel -> index of child 3
        assert (t[:, 3:, 3:] == 0).all()  # not-novel predicted as mother -> child 1
        assert (t[:, :, :3] == -1).all()  # not-novel predicted as class 0 -> unused

    def test_full_label_regime_uses_mother_label(self):
        f, masks, w, b = _data()
        full = masks.clone()
        full[:, 3:, 3:] = 1
        full[:, :, :3] = 0
        m = _ready(f, full, w, b)
        t = m._targets(f, full)
        assert (t[:, 3:, 3:] == 0).all() and (t[:, :, :3] == -1).all()

    def test_missing_child_warns(self, caplog):
        f, masks, w, b = _data()
        masks[:] = NN
        masks[:, 0, 0] = 3
        f[:, 0] = -5.0  # base never predicts the mother -> no negatives
        with caplog.at_level("WARNING"):
            _ready(f, masks, w, b)
        assert "no support pixel for class 1" in caplog.text


class TestProbabilities:
    @pytest.mark.parametrize("q", ["proto", "proto_prob", "linear"])
    def test_split_preserves_untouched_and_sums_to_mother(self, q):
        f, masks, w, b = _data()
        m = _ready(f, masks, w, b, q=q)
        p = m.split_probabilities(f)
        p_base = m._base_probs(f)
        assert p.shape == (2, 4, 6, 6)
        torch.testing.assert_close(p[:, [0, 2]], p_base[:, [0, 2]])
        torch.testing.assert_close(p[:, 1] + p[:, 3], p_base[:, 1])
        torch.testing.assert_close(p.sum(1), torch.ones(2, 6, 6))

    @pytest.mark.parametrize("q", ["proto", "proto_prob"])
    @pytest.mark.parametrize("variance", ["shared", "per_class"])
    def test_separable_children_are_recovered(self, q, variance):
        f, masks, w, b = _data()
        m = _ready(f, masks, w, b, q=q, variance=variance)
        pred = m(f).argmax(1)
        assert (pred[:, :3, 3:] == 3).all()
        assert (pred[:, 3:, 3:] == 1).all()

    def test_hierarchical_decoding_keeps_base_predictions_elsewhere(self):
        f, masks, w, b = _data()
        f = f + torch.randn_like(f)  # noisy, includes near ties
        m = _ready(f, masks, w, b, q="proto")
        base = m._base_probs(f).argmax(1)
        pred = m(f).argmax(1)
        outside = base != 1
        assert torch.equal(pred[outside], base[outside])
        assert set(pred[~outside].unique().tolist()) <= {1, 3}

    def test_flat_decoding_returns_log_probabilities(self):
        f, masks, w, b = _data()
        m = _ready(f, masks, w, b, decoding="flat")
        torch.testing.assert_close(
            m(f).exp(), m.split_probabilities(f), atol=1e-6, rtol=0
        )

    def test_multiple_mothers(self):
        f, masks, w, b = _data()
        masks[:, 0, 0] = 4
        h = ClassHierarchy({1: [1, 3], 0: [0, 4]}, num_base_classes=3)
        m = _ready(f, masks, w, b, hierarchy=h)
        p = m.split_probabilities(f)
        assert p.shape[1] == 5
        torch.testing.assert_close(p.sum(1), torch.ones(2, 6, 6))
        torch.testing.assert_close(p[:, 0] + p[:, 4], m._base_probs(f)[:, 0])


class TestTraining:
    def test_proto_has_no_loss_or_parameters(self):
        f, masks, w, b = _data()
        m = _ready(f, masks, w, b, q="proto")
        assert m.support_loss(f, masks) is None
        assert not any(p.requires_grad for p in m.parameters())
        assert not m.transductive

    def test_linear_loss_decreases(self):
        f, masks, w, b = _data()
        m = _ready(f, masks, w, b, q="linear")
        m.q_weight.data.zero_()  # uniform q: the synthetic data is separable at init
        opt = torch.optim.SGD([m.q_weight, m.q_bias], lr=0.1)
        first = m.support_loss(f, masks)
        for _ in range(20):
            loss = m.support_loss(f, masks)
            opt.zero_grad()
            loss.backward()
            opt.step()
        assert m.support_loss(f, masks) < first

    def test_trans_adapts_per_map(self):
        f, masks, w, b = _data()
        m = _ready(f, masks, w, b, q="trans", adapt_iter=12, lr=0.05)
        m.q_weight.data.zero_()
        assert m.transductive
        before = m(f)
        m.adapt_to_query(f, masks, f)
        assert m._task_params[0].shape == (2, 2, 4)
        assert not torch.allclose(before, m(f))
        torch.testing.assert_close(
            m(f[:1]), before[:1]
        )  # other batch size -> base params


@pytest.mark.parametrize(
    "kw, match",
    [
        ({"q": "knn"}, "q must"),
        ({"variance": "full"}, "variance"),
        ({"decoding": "x"}, "decoding"),
    ],
)
def test_invalid_arguments(kw, match):
    with pytest.raises(ValueError, match=match):
        HiSplit(**kw)


class TestEvidential:
    def test_edl_q_is_dirichlet_mean_and_trains(self):
        f, masks, w, b = _data()
        m = _ready(f, masks, w, b, q="edl", kl_anneal_steps=4)
        assert m.q_weight.requires_grad
        scores = m._scores(f, m.q_weight, m.q_bias)
        alpha = torch.nn.functional.softplus(scores) + 1
        torch.testing.assert_close(m._q(scores), alpha / alpha.sum(1, keepdim=True))
        m.q_weight.data.zero_()
        opt = torch.optim.SGD([m.q_weight, m.q_bias], lr=0.05)
        first = m.support_loss(f, masks)
        for _ in range(10):
            loss = m.support_loss(f, masks)
            opt.zero_grad()
            loss.backward()
            opt.step()
        assert m._edl_step == 11
        assert m.support_loss(f, masks) < first

    def test_edl_loss_skips_mothers_without_targets(self):
        f, masks, w, b = _data()
        masks[:] = 255
        m = _ready(f, masks, w, b, q="edl")
        assert m.support_loss(f, masks).item() == 0.0

    def test_uncertainty_softmax_base(self):
        f, masks, w, b = _data()
        m = _ready(f, masks, w, b, q="edl")
        u = m.uncertainty(f)
        assert set(u) == {"split_entropy", "split_vacuity"}
        base_pred = m._base_probs(f).argmax(1)
        assert (u["split_entropy"][base_pred != 1] == 0).all()
        assert ((u["split_vacuity"] > 0) == (base_pred == 1)).all()
        assert all(((v >= 0) & (v <= 1)).all() for v in u.values())

    def test_evidential_base_split_and_uncertainty(self):
        f, masks, w, b = _data()
        h = ClassHierarchy({1: [1, 3]}, num_base_classes=3)
        m = HiSplit(q="proto")
        m.setup(h, w, b, NN, base_output="evidential")
        m.init_from_support(f, masks)
        logits = torch.einsum("bfhw,cf->bchw", f, w) + b.view(1, -1, 1, 1)
        alpha = torch.nn.functional.softplus(logits) + 1
        p = m.split_probabilities(f)
        q = m._q(m._scores(f, m.q_weight, m.q_bias))
        strength = alpha.sum(1)
        torch.testing.assert_close(
            p[:, 3], alpha[:, 1] * q[:, 1] / strength
        )  # α_c = α_m q_c
        u = m.uncertainty(f)
        assert set(u) == {"split_entropy", "base_vacuity", "dissonance"}
        torch.testing.assert_close(u["base_vacuity"], 3 / strength)
        assert ((u["dissonance"] >= 0) & (u["dissonance"] <= 1)).all()


def _second_choice_data():
    """Left half: base predicts class 2 with the mother (1) second, and the
    features look like the novel class."""
    f, masks, w, b = _data()
    f[:, 0, :, :3] += 0.6  # mother logit 0.2 vs class 2 logit 0.5
    f[:, 1, :, :3] += 2.0  # q prefers the novel child
    return f, masks, w, b


class TestBoundaryP2:
    def test_leak_preserves_normalisation_and_moves_mass(self):
        f, masks, w, b = _data()
        m = _ready(f, masks, w, b, leak=True, leak_init=-1.0)
        p = m.split_probabilities(f)
        torch.testing.assert_close(p.sum(1), torch.ones(2, 6, 6))
        t = m.leak_coefficients()
        assert t.shape == (2, 3)
        assert (
            (t[0] == 0).all() and t[1, 1] == 0 and t[1, 0] > 0
        )  # only novel row, not from mother
        p_base = m._base_probs(f)
        torch.testing.assert_close(p[:, 0], p_base[:, 0] * (1 - t[1, 0]))

    def test_leak_is_trained_on_novel_pixels_predicted_elsewhere(self):
        f, masks, w, b = _data()
        masks[:, :3, :2] = 3  # cropland inside a region the base predicts as class 0
        m = _ready(f, masks, w, b, leak=True)
        assert m.leak_logit.requires_grad and m.support_loss(f, masks) is not None
        opt = torch.optim.SGD([m.leak_logit], lr=1.0)
        before = m.leak_coefficients()[1, 0].item()
        for _ in range(30):
            loss = m.support_loss(f, masks)
            opt.zero_grad()
            loss.backward()
            opt.step()
        assert m.leak_coefficients()[1, 0].item() > before

    def test_leak_with_trainable_q_sums_losses(self):
        f, masks, w, b = _data()
        m = _ready(f, masks, w, b, q="linear", leak=True)
        both = m.support_loss(f, masks)
        m.leak = False
        assert both > m.support_loss(f, masks)

    def test_leak_full_label_regime(self):
        f, masks, w, b = _data()
        full = masks.clone()
        full[full == NN] = 0
        m = _ready(f, full, w, b, leak=True)
        assert torch.isfinite(m.support_loss(f, full))

    def test_hierarchical_decoding_with_leak_can_take_pixels_from_neighbours(self):
        f, masks, w, b = _data()
        m = _ready(f, masks, w, b, leak=True)
        with torch.no_grad():
            m.leak_logit.fill_(4.0)  # almost all mass of class 0/2 leaks into cropland
        pred = m(f).argmax(1)
        base = m._base_probs(f).argmax(1)
        assert ((base == 2) & (pred == 3)).any()

    def test_widen_prob_switches_only_second_choice_mother_pixels(self):
        f, masks, w, b = _second_choice_data()
        m = _ready(f, masks, w, b, widen="prob", widen_threshold=0.0)
        ref = _ready(f, masks, w, b)
        level = m._base_probs(f)
        top2 = level.topk(2, 1).indices
        pred, before = m(f).argmax(1), ref(f).argmax(1)
        changed = pred != before
        assert changed.any()
        assert (top2[:, 1][changed] == 1).all() and (pred[changed] == 3).all()
        # threshold 1 -> nothing widened
        m_hi = _ready(f, masks, w, b, widen="prob", widen_threshold=1.0)
        assert torch.equal(m_hi(f).argmax(1), before)

    def test_widen_dissonance_needs_evidential_base(self):
        f, masks, w, b = _data()
        with pytest.raises(ValueError, match="evidential"):
            _ready(f, masks, w, b, widen="dissonance")
        h = ClassHierarchy({1: [1, 3]}, num_base_classes=3)
        m = HiSplit(widen="dissonance", widen_threshold=0.0)
        m.setup(h, w, b, NN, base_output="evidential")
        m.init_from_support(f, masks)
        assert m(f).shape == (2, 4, 6, 6)

    def test_novel_prior_weight_increases_novel_share(self):
        f, masks, w, b = _data()
        f = f + torch.randn_like(f)
        m1 = _ready(f, masks, w, b)
        m2 = _ready(f, masks, w, b, novel_prior_weight=5.0)
        p1, p2 = m1.split_probabilities(f), m2.split_probabilities(f)
        assert (p2[:, 3] >= p1[:, 3] - 1e-6).all() and (p2[:, 3] > p1[:, 3]).any()
        torch.testing.assert_close(p2[:, 1] + p2[:, 3], p1[:, 1] + p1[:, 3])

    def test_sweep_variants(self):
        f, masks, w, b = _second_choice_data()
        m = _ready(f, masks, w, b, widen="prob", sweep=[0.0, 1.0])
        assert m.variant_names() == ["widen_0", "widen_1"]
        v = m.decode_variants(f)
        ref = _ready(f, masks, w, b)(f).argmax(1)
        assert torch.equal(v["widen_1"].argmax(1), ref)
        assert not torch.equal(v["widen_0"].argmax(1), ref)

    @pytest.mark.parametrize(
        "kw, match",
        [
            ({"widen": "maybe"}, "widen must"),
            ({"widen": "prob", "decoding": "flat"}, "hierarchical"),
            ({"sweep": [0.1]}, "sweep requires"),
        ],
    )
    def test_invalid_p2_arguments(self, kw, match):
        with pytest.raises(ValueError, match=match):
            HiSplit(**kw)


def test_leak_never_swaps_classes_outside_the_superclass():
    """Uneven leak fractions must not reorder classes outside the superclass:
    p0 = 0.50, p2 = 0.45, p_vb = 0.05 and half of class 0 leaks into the
    novel child -> p''(0) = 0.25 < p2 = 0.45. Before the fix the pixel became
    class 2; now it may only keep class 0 or join the superclass (here the
    superclass wins: 0.05 + 0.25 = 0.30 > 0.25)."""
    h = ClassHierarchy({1: [1, 3]}, num_base_classes=3)
    m = HiSplit(q="proto", leak=True)
    m.setup(h, torch.eye(3), torch.zeros(3), NN)  # logits = features
    support = (
        torch.log(torch.tensor([0.1, 0.8, 0.1])).view(1, 3, 1, 1).repeat(1, 1, 2, 2)
    )
    support_masks = torch.tensor([[[3, NN], [NN, NN]]])
    support[0, 2, 0, 0] += 1.0  # the novel pixel looks different
    m.init_from_support(support, support_masks)
    with torch.no_grad():
        m.leak_logit.fill_(-30.0)
        m.leak_logit[1, 0] = 0.0  # t = 0.5 from class 0
    query = torch.log(torch.tensor([0.50, 0.05, 0.45])).view(1, 3, 1, 1)
    assert m._base_probs(query).argmax(1).item() == 0
    assert m(query).argmax(1).item() in (1, 3)
    # with a smaller leak (t = 0.2) neither neighbour nor superclass wins:
    # p''(0) = 0.40 > p''(vb) = 0.15 -> the base decision (0) is kept, not 2
    with torch.no_grad():
        m.leak_logit[1, 0] = torch.logit(torch.tensor(0.2))
    assert m(query).argmax(1).item() == 0


def test_leak_can_still_take_pixels_into_the_superclass():
    h = ClassHierarchy({1: [1, 3]}, num_base_classes=3)
    m = HiSplit(q="proto", leak=True)
    m.setup(h, torch.eye(3), torch.zeros(3), NN)
    support = (
        torch.log(torch.tensor([0.1, 0.8, 0.1])).view(1, 3, 1, 1).repeat(1, 1, 2, 2)
    )
    support[0, 0, 0, 0] += 2.0
    m.init_from_support(support, torch.tensor([[[3, NN], [NN, NN]]]))
    with torch.no_grad():
        m.leak_logit.fill_(-30.0)
        m.leak_logit[1, 0] = 4.0  # ~98% of class 0 leaks
    query = torch.log(torch.tensor([0.50, 0.05, 0.45])).view(1, 3, 1, 1)
    assert m(query).argmax(1).item() in (1, 3)
