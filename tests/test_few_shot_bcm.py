# -*- coding: utf-8 -*-
"""Tests for the BCM GFSS method, including parity with the official code.

``tests/testing_data/few_shot/bcm_reference.pt`` was produced by the official
``Classifier`` of https://github.com/IBM/BCM (``src/bcm.py``) on small random
inputs, with ``sklearnex`` replaced by ``sklearn`` (same API) and the NumPy
global generator seeded; the generator script lives outside the library
(research repository, ``scripts/gfss_parity/make_reference_tensors.py``).
"""

from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from pytorch_segmentation_models_trainer.few_shot.hierarchy import ClassHierarchy
from pytorch_segmentation_models_trainer.few_shot.methods.bcm import BCM

REF = Path(__file__).parent / "testing_data" / "few_shot" / "bcm_reference.pt"
REF_ENS = (
    Path(__file__).parent / "testing_data" / "few_shot" / "bcm_ensemble_reference.pt"
)
NN = 254


@pytest.fixture(scope="module")
def ref():
    return torch.load(REF, weights_only=False)


def _upsample(logits, size):
    return F.interpolate(logits, size=size, mode="bilinear", align_corners=True)


def _from_official(ref):
    n_novel = ref["gt_s"].shape[0]
    n_base = ref["n_base"]
    masks = ref["gt_s"].flatten(0, 1).clone()
    masks[masks == 0] = NN
    fs = ref["features_s"].flatten(0, 1)
    h = ClassHierarchy(
        {0: [0] + list(range(n_base, n_base + n_novel))}, num_base_classes=n_base
    )
    return h, fs, masks, ref["features_q"][:, 0]


class TestParityWithOfficialCode:
    def test_table_and_predictions_match(self, ref):
        h, fs, ms, fq = _from_official(ref)
        a = ref["args"]
        m = BCM(
            mapping="mined",
            top_k=a["top_k"],
            sampling=a["sampling"],
            beta=a["beta"],
            seed=ref["numpy_seed"],
        )
        m.setup(h, ref["base_weight"].T, ref["base_bias"], NN)
        small = (
            F.interpolate(ms.unsqueeze(1).float(), size=fs.shape[-2:], mode="nearest")
            .squeeze(1)
            .long()
        )
        m.init_from_support(fs, small)
        assert m.table == {int(k): list(v) for k, v in ref["table"].items()}
        logits = m.decode_to(fq, ref["out_size"], _upsample)
        assert torch.equal(logits.argmax(1), ref["pred_q"][:, 0])


def test_parity_shot_wise_ensemble_with_tukey():
    """Paper setting (§4.5): Tukey tau = 0.5 and shot-wise ensemble (5 one-shot
    models weight 1 + the 5-shot model weight 5)."""
    ref = torch.load(REF_ENS, weights_only=False)
    h, fs, ms, fq = _from_official(ref)
    a = ref["args"]
    m = BCM(beta=a["beta"], ensemble=a["ensemble"], seed=ref["numpy_seed"])
    m.setup(h, ref["base_weight"].T, ref["base_bias"], NN)
    small = (
        F.interpolate(ms.unsqueeze(1).float(), size=fs.shape[-2:], mode="nearest")
        .squeeze(1)
        .long()
    )
    m.init_from_support(fs, small)
    assert len(m.models[1]) == 6
    logits = m.decode_to(fq, ref["out_size"], _upsample)
    assert torch.equal(logits.argmax(1), ref["pred_q"][:, 0])


@pytest.fixture
def toy():
    torch.manual_seed(0)
    h = ClassHierarchy({1: [1, 3]}, num_base_classes=3)
    w = torch.zeros(3, 4)
    w[1, 0] = 2.0
    w[0, 0] = -2.0
    b = torch.tensor([0.0, -1.0, 0.5])
    f = torch.randn(4, 4, 8, 8) * 0.1
    f[:, 0, :, 4:] += 3.0  # right half: base predicts class 1 (mother)
    f[:, 1, :4, 4:] += 2.0  # top-right: novel
    ms = torch.full((4, 8, 8), NN)
    ms[:, :4, 4:] = 3
    return h, w, b, f, ms


def _ready(toy, **kw):
    h, w, b, f, ms = toy
    m = BCM(**kw)
    m.setup(h, w, b, NN)
    m.init_from_support(f, ms)
    return m


class TestBehaviour:
    def test_hierarchy_mapping_uses_the_mother(self, toy):
        m = _ready(toy, mapping="hierarchy")
        assert m.table == {1: [3]}

    def test_mined_mapping_on_toy(self, toy):
        assert _ready(toy, mapping="mined").table == {1: [3]}

    def test_overrides_only_pixels_of_the_mapped_base_class(self, toy):
        h, w, b, f, ms = toy
        m = _ready(toy)
        size = (16, 16)
        logits = m.decode_to(f, size, _upsample)
        pred = logits.argmax(1)
        logits = torch.einsum("bfhw,cf->bchw", f, w) + b.view(1, -1, 1, 1)
        base = _upsample(torch.softmax(logits, 1), size).argmax(
            1
        )  # official: upsampled probas
        changed = pred != base
        assert changed.any()
        assert (base[changed] == 1).all() and (pred[changed] == 3).all()

    def test_forward_feature_resolution_logits(self, toy):
        h, w, b, f, ms = toy
        m = _ready(toy)
        out = m(f)
        assert out.shape == (4, 4, 8, 8)
        assert (out.argmax(1)[:, :4, 4:] == 3).float().mean() > 0.9

    @pytest.mark.parametrize("sampling", ["us", "os", "bg"])
    def test_sampling_options(self, toy, sampling):
        assert 1 in _ready(toy, sampling=sampling).models

    def test_beta_power_on_non_negative_features(self, toy):
        h, w, b, f, ms = toy
        m = _ready((h, w, b, f.abs(), ms), beta=0.5)
        assert 1 in m.models
        assert np.isfinite(m._features_np(f.abs())).all()

    def test_full_label_regime(self, toy):
        h, w, b, f, ms = toy
        full = ms.clone()
        full[full == NN] = 0
        m = _ready((h, w, b, f, full))
        assert m.table == {1: [3]}

    def test_no_novel_pixels_gives_empty_table(self, toy):
        h, w, b, f, ms = toy
        empty = torch.full_like(ms, NN)
        m = _ready((h, w, b, f, empty))
        assert m.table == {}
        logits = m.decode_to(f, (8, 8), _upsample)
        assert (logits.argmax(1) != 3).all()

    @pytest.mark.parametrize(
        "kw, match",
        [({"mapping": "cooc"}, "mapping"), ({"sampling": "smote"}, "sampling")],
    )
    def test_invalid_arguments(self, kw, match):
        with pytest.raises(ValueError, match=match):
            BCM(**kw)


def test_balance_oversamples_smaller_classes():
    from pytorch_segmentation_models_trainer.few_shot.methods.bcm import _balance

    X = np.arange(10).reshape(-1, 1)
    y = np.array([0, 0, 0, 0, 0, 1, 1, 1, 2, 2])
    Xb, yb = _balance(X, y, "os", np.random.RandomState(0))
    assert np.bincount(yb).tolist() == [3, 3, 3]


def test_mapped_base_class_absent_from_query(toy):
    h, w, b, f, ms = toy
    m = _ready(toy)
    f_other = f.clone()
    f_other[:, 0] = -5.0  # base never predicts the mother here
    pred = m.decode_to(f_other, (8, 8), _upsample).argmax(1)
    assert (pred != 3).all()
