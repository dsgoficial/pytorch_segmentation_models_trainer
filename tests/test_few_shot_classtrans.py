# -*- coding: utf-8 -*-
"""Tests for the ClassTrans GFSS method, including parity with the official code.

The reference tensors in ``tests/testing_data/few_shot/classtrans_reference.pt``
were produced by the official ``TransitionClassifier``
(https://github.com/earth-insights/ClassTrans) on small random inputs (8 base
+ 4 novel classes, the sizes its hard-coded LDAM counts require); the
generator script lives outside the library (research repository,
``scripts/gfss_parity/make_reference_tensors.py``).
"""

from pathlib import Path

import pytest
import torch

from pytorch_segmentation_models_trainer.few_shot.hierarchy import ClassHierarchy
from pytorch_segmentation_models_trainer.few_shot.methods.classtrans import (
    ClassTrans,
    sinkhorn,
)

REF = Path(__file__).parent / "testing_data" / "few_shot" / "classtrans_reference.pt"
NOT_NOVEL = 254


@pytest.fixture(scope="module")
def ref():
    return torch.load(REF, weights_only=False)


def _from_official(ref):
    n_novel = ref["gt_s"].shape[0]
    masks = ref["gt_s"].flatten(0, 1).clone()
    masks[masks == 0] = NOT_NOVEL
    n_base = ref["n_base"]
    hierarchy = ClassHierarchy(
        {0: [0] + list(range(n_base, n_base + n_novel))}, num_base_classes=n_base
    )
    return hierarchy, ref["features_s"].flatten(0, 1), masks, ref["features_q"][:, 0]


class TestParityWithOfficialCode:
    def test_init_and_logits_match(self, ref):
        a = ref["args"]
        hierarchy, fs, ms, fq = _from_official(ref)
        m = ClassTrans(
            weights=a["weights"],
            adapt_iter=a["adapt_iter"],
            lr=a["cls_lr"],
            ldam_start_iter=a["epoch_LDAM"],
            pi_update_at=a["pi_update_at"],
            fine_tune_base_classifier=a["fine_tune_base_classifier"],
            class_counts=ref["class_counts"].tolist(),
        )
        m.setup(hierarchy, ref["base_weight"].T, ref["base_bias"], NOT_NOVEL)
        torch.manual_seed(2)  # same RNG state as the official init
        m.init_from_support(fs, ms)
        torch.testing.assert_close(m.novel_weight, ref["init"]["novel_weight"])
        torch.testing.assert_close(m.row_weight, ref["init"]["transition_row_weight"])
        torch.testing.assert_close(
            m.col_weight, ref["init"]["transition_column_weight"]
        )
        m.adapt_to_query(fs, ms, fq)
        torch.testing.assert_close(m(fq), ref["logits_q"][:, 0], rtol=1e-4, atol=1e-5)


@pytest.fixture
def small():
    torch.manual_seed(0)
    h = ClassHierarchy({1: [1, 2]}, num_base_classes=2)
    fs = torch.randn(3, 4, 5, 5)
    ms = torch.full((3, 5, 5), NOT_NOVEL)
    ms[:, :2] = 2
    fq = torch.randn(2, 4, 5, 5)
    return h, fs, ms, fq


def _ready(h, fs, ms, **kw):
    m = ClassTrans(**kw)
    m.setup(h, torch.randn(2, 4), torch.zeros(2), NOT_NOVEL)
    m.init_from_support(fs, ms)
    return m


class TestBehaviour:
    def test_initial_forward_equals_classification_branch(self, small):
        h, fs, ms, fq = small
        m = _ready(h, fs, ms)
        out = m(fq)
        assert out.shape == (2, 3, 5, 5)
        torch.testing.assert_close(
            out[:, :2], torch.einsum("bfhw,cf->bchw", fq, m.base_weight)
        )  # layer_scale starts at 0

    def test_adaptation_with_ldam_per_task(self, small):
        h, fs, ms, fq = small
        m = _ready(h, fs, ms, adapt_iter=6, ldam_start_iter=3, pi_update_at=[2])
        before = m(fq)
        m.adapt_to_query(fs, ms, fq)
        after = m(fq)
        assert torch.isfinite(after).all() and not torch.allclose(before, after)
        assert m._task_params["layer_scale"].shape == (2, 3)
        torch.testing.assert_close(m(fq[:1]), before[:1])

    def test_frozen_base_rows(self, small):
        h, fs, ms, fq = small
        m = _ready(h, fs, ms, adapt_iter=2, fine_tune_base_classifier=False)
        m.adapt_to_query(fs, ms, fq)
        torch.testing.assert_close(m._task_params["base_w"][0], m.base_weight.T)

    def test_support_counts_default(self, small):
        h, fs, ms, fq = small
        m = _ready(h, fs, ms)
        n_novel = float((ms == 2).sum())
        n_rest = float((ms == NOT_NOVEL).sum())
        expected = torch.tensor([n_rest, n_rest, n_novel]) ** -0.25
        expected = expected * 6.0 / expected.max()
        torch.testing.assert_close(m.ldam_margins, expected)

    def test_wrong_number_of_counts_raises(self, small):
        h, fs, ms, fq = small
        with pytest.raises(ValueError, match="class_counts"):
            _ready(h, fs, ms, class_counts=[1, 2])

    def test_explicit_not_novel_weight(self, small):
        h, fs, ms, fq = small
        m = _ready(h, fs, ms, adapt_iter=2, not_novel_weight=1.0)
        m.adapt_to_query(fs, ms, fq)
        assert torch.isfinite(m(fq)).all()


def test_sinkhorn_marginals():
    cost = torch.rand(3, 5, dtype=torch.float64)
    target = torch.full((3,), 1 / 3, dtype=torch.float64)
    source = torch.full((5,), 1 / 5, dtype=torch.float64)
    plan = sinkhorn(cost, target, source, lam=0.1)
    torch.testing.assert_close(plan.sum(0), source)
    torch.testing.assert_close(plan.sum(1), target, atol=1e-6, rtol=0)


def test_sinkhorn_stops_at_max_iter():
    cost = torch.rand(2, 2)
    plan = sinkhorn(
        cost,
        torch.tensor([0.5, 0.5]),
        torch.tensor([0.5, 0.5]),
        lam=0.1,
        eps=-1,
        max_iter=3,
    )
    assert torch.isfinite(plan).all()


@pytest.mark.parametrize("masked", [True, False])
def test_hierarchical_mask_feeds_novel_only_from_mother(small, masked):
    h, fs, ms, fq = small  # mother 1 -> novel 2; class 0 is not the mother
    m = _ready(h, fs, ms, hierarchical_mask=masked)
    p = m._initial_params(2)
    p["layer_scale"].fill_(1.0)
    f5 = fq.unsqueeze(1)
    before = m._transition(f5, p)[:, :, 2]
    m.base_weight[0] += 1.0  # change the snapshot logit of a non-mother class
    after = m._transition(f5, p)[:, :, 2]
    assert torch.allclose(before, after) == masked
