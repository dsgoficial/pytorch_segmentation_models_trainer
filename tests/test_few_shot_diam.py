# -*- coding: utf-8 -*-
"""Tests for the DIaM GFSS method, including parity with the official code.

The reference tensors in ``tests/testing_data/few_shot/diam_reference.pt``
were produced by the official ``Classifier`` (https://github.com/sinahmr/DIaM)
on small random inputs; the generator script lives outside the library
(research repository, ``scripts/gfss_parity/make_reference_tensors.py``).
"""

from pathlib import Path

import pytest
import torch

from pytorch_segmentation_models_trainer.few_shot.hierarchy import ClassHierarchy
from pytorch_segmentation_models_trainer.few_shot.methods.diam import DIaM

REF = Path(__file__).parent / "testing_data" / "few_shot" / "diam_reference.pt"
NOT_NOVEL = 254


def _from_official(ref):
    """Official layout -> GFSSModel layout (background label 0 -> not-novel)."""
    n_novel, shot = ref["gt_s"].shape[:2]
    masks = ref["gt_s"].flatten(0, 1).clone()
    masks[masks == 0] = NOT_NOVEL
    feats_s = ref["features_s"].flatten(0, 1)
    feats_q = ref["features_q"][:, 0]
    n_base = ref["n_base"]
    hierarchy = ClassHierarchy(
        {0: [0] + list(range(n_base, n_base + n_novel))}, num_base_classes=n_base
    )
    return hierarchy, feats_s, masks, feats_q


@pytest.fixture(scope="module")
def ref():
    return torch.load(REF, weights_only=False)


def _method(ref, **overrides):
    a = ref["args"]
    kwargs = dict(
        weights=a["weights"],
        adapt_iter=a["adapt_iter"],
        lr=a["cls_lr"],
        pi_estimation=a["pi_estimation_strategy"],
        pi_update_at=a["pi_update_at"],
        fine_tune_base_classifier=a["fine_tune_base_classifier"],
    )
    kwargs.update(overrides)
    return DIaM(**kwargs)


class TestParityWithOfficialCode:
    def test_prototypes_and_logits_match(self, ref):
        hierarchy, fs, ms, fq = _from_official(ref)
        m = _method(ref)
        m.setup(hierarchy, ref["base_weight"].T, ref["base_bias"], NOT_NOVEL)
        m.init_from_support(fs, ms)
        torch.testing.assert_close(m.novel_weight, ref["init_novel_weight"])
        m.adapt_to_query(fs, ms, fq)
        torch.testing.assert_close(m(fq), ref["logits_q"][:, 0], rtol=1e-4, atol=1e-5)


@pytest.fixture
def small():
    torch.manual_seed(0)
    h = ClassHierarchy({1: [1, 2]}, num_base_classes=2)
    fs = torch.randn(3, 4, 5, 5)
    ms = torch.full((3, 5, 5), NOT_NOVEL)
    ms[:, :2] = 2
    ms[0, 4, 4] = 255
    fq = torch.randn(2, 4, 5, 5)
    return h, fs, ms, fq


def _ready(h, fs, ms, **kw):
    m = DIaM(**kw)
    m.setup(h, torch.randn(2, 4), torch.zeros(2), NOT_NOVEL)
    m.init_from_support(fs, ms)
    return m


class TestBehaviour:
    def test_transductive_flag_and_forward_before_adaptation(self, small):
        h, fs, ms, fq = small
        m = _ready(h, fs, ms)
        assert m.transductive
        out = m(fq)
        assert out.shape == (2, 3, 5, 5)
        expected = torch.einsum("bfhw,cf->bchw", fq, m.base_weight)
        torch.testing.assert_close(out[:, :2], expected)

    def test_adaptation_changes_logits_and_is_per_task(self, small):
        h, fs, ms, fq = small
        m = _ready(h, fs, ms, adapt_iter=5)
        before = m(fq)
        m.adapt_to_query(fs, ms, fq)
        after = m(fq)
        assert not torch.allclose(before, after)
        assert m._task_weight.shape == (2, 4, 3)
        assert not m._task_weight.requires_grad
        # a batch of another size falls back to the initial classifier
        torch.testing.assert_close(m(fq[:1]), before[:1])

    def test_frozen_base_rows_when_not_fine_tuned(self, small):
        h, fs, ms, fq = small
        m = _ready(h, fs, ms, adapt_iter=3, fine_tune_base_classifier=False)
        m.adapt_to_query(fs, ms, fq)
        torch.testing.assert_close(m._task_weight[0, :, :2], m.base_weight.T)

    def test_uniform_prior_and_explicit_weight(self, small):
        h, fs, ms, fq = small
        m = _ready(
            h, fs, ms, adapt_iter=12, pi_estimation="uniform", not_novel_weight=0.5
        )
        m.adapt_to_query(fs, ms, fq)
        assert torch.isfinite(m(fq)).all()

    def test_full_label_regime(self, small):
        h, fs, ms, fq = small
        full = ms.clone()
        full[full == NOT_NOVEL] = 0
        m = _ready(h, fs, full, adapt_iter=3)
        m.adapt_to_query(fs, full, fq)
        assert torch.isfinite(m(fq)).all()

    def test_invalid_pi_estimation(self):
        with pytest.raises(ValueError, match="pi_estimation"):
            DIaM(pi_estimation="upperbound")
