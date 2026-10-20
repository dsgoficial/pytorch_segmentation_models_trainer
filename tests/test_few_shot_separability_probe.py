# -*- coding: utf-8 -*-
"""Tests for the separability probe of the frozen base features (E2)."""

import numpy as np
import pandas as pd
import pytest
import segmentation_models_pytorch as smp
import torch
from omegaconf import OmegaConf
from torch.utils.data import Dataset

from pytorch_segmentation_models_trainer.tools.few_shot.separability_probe import (
    alignment_scores,
    collect_pair_features,
    probe_scores,
    run_separability_probe,
)

_MODEL = {
    "_target_": "segmentation_models_pytorch.UPerNet",
    "encoder_name": "resnet18",
    "encoder_weights": None,
    "classes": 4,
    "decoder_channels": 16,
}


class _CsvTiny(Dataset):
    """Windows listed in a CSV (column ``idx``); left half class 3 (dark),
    right half class 5 (bright) when idx is even, all class 0 otherwise."""

    def __init__(self, window_index_cache, **kwargs):
        self.idx = pd.read_csv(window_index_cache)["idx"].tolist()

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, i):
        g = torch.Generator().manual_seed(int(self.idx[i]))
        img = torch.rand(3, 32, 32, generator=g) * 0.1
        mask = torch.zeros(32, 32, dtype=torch.long)
        if self.idx[i] % 2 == 0:
            mask[:, :16] = 3
            mask[:, 16:] = 5
            img[:, :, 16:] += 2.0
        mask[0, 0] = 255
        return {"image": img, "mask": mask}


def _segmenter():
    from pytorch_segmentation_models_trainer.few_shot.backbone import (
        FrozenLinearHeadSegmenter,
    )

    torch.manual_seed(0)
    return FrozenLinearHeadSegmenter(
        smp.UPerNet(**{k: v for k, v in _MODEL.items() if k != "_target_"})
    )


def _index(tmp_path, ids, name="idx.csv"):
    p = tmp_path / name
    pd.DataFrame({"idx": ids}).to_csv(p, index=False)
    return str(p)


class TestProbeScores:
    def test_separable_data_scores_high(self):
        rng = np.random.default_rng(0)
        X = np.vstack(
            [rng.normal([4, 0, 0, 0], 1, (50, 4)), rng.normal([0, 4, 0, 0], 1, (50, 4))]
        )
        y = np.r_[np.zeros(50), np.ones(50)]
        out = probe_scores(X, y, X, y, C=1.0)
        assert out["auroc_logreg"] > 0.99 and out["balacc_logreg"] > 0.95
        assert out["auroc_proto"] > 0.99 and out["balacc_proto"] > 0.95
        assert out["ap_logreg"] > 0.99

    def test_missing_class_gives_nan(self):
        X = np.zeros((4, 2))
        out = probe_scores(X, np.zeros(4), X, np.r_[0, 0, 1, 1], C=1.0)
        assert all(np.isnan(v) for v in out.values())
        out = probe_scores(X, np.r_[0, 0, 1, 1], X, np.zeros(4), C=1.0)
        assert all(np.isnan(v) for v in out.values())


class TestAlignmentScores:
    @staticmethod
    def _two_blobs(n=60):
        rng = np.random.default_rng(0)
        X = np.vstack(
            [rng.normal([5, 0, 0], 0.3, (n, 3)), rng.normal([0, 5, 0], 0.3, (n, 3))]
        )
        return X, np.r_[np.zeros(n), np.ones(n)].astype(int)

    def test_aligned_subprototypes_score_one(self):
        X, y = self._two_blobs()
        protos = np.array([[1.0, 0, 0], [0, 1.0, 0], [0, 0, 1.0]])
        out = alignment_scores(X, y, protos, kmeans_clusters=2, seed=0)
        assert out["nmi_subproto"] == pytest.approx(1.0)
        assert out["purity_subproto"] == pytest.approx(1.0)
        assert out["used_subproto"] == 2 and out["num_subproto"] == 3
        assert out["subproto_pos_frac"] == "0.000;1.000;nan"
        assert out["subproto_share"] == "0.500;0.500;0.000"
        assert out["nmi_kmeans"] > 0.99 and out["purity_kmeans"] > 0.99
        assert out["kmeans_clusters"] == 2

    def test_misaligned_subprototypes_score_zero(self):
        X, y = self._two_blobs()
        X[:, 2] = np.tile([3.0, -3.0], len(y) // 2)  # axis unrelated to y
        protos = np.array([[0, 0, 1.0], [0, 0, -1.0]])
        out = alignment_scores(X, y, protos, kmeans_clusters=2, seed=0)
        assert out["nmi_subproto"] == pytest.approx(0.0, abs=1e-9)
        assert out["purity_subproto"] == pytest.approx(0.5)

    def test_without_subprototypes_only_kmeans(self):
        X, y = self._two_blobs()
        out = alignment_scores(X, y, None, kmeans_clusters=4, seed=0)
        assert np.isnan(out["nmi_subproto"]) and np.isnan(out["purity_subproto"])
        assert out["num_subproto"] == 0 and out["subproto_pos_frac"] == ""
        assert 0.0 <= out["nmi_kmeans"] <= 1.0 and out["kmeans_clusters"] == 4

    def test_missing_class_gives_nan(self):
        X = np.ones((6, 3))
        out = alignment_scores(X, np.zeros(6, int), np.eye(3), kmeans_clusters=2)
        assert np.isnan(out["nmi_subproto"]) and np.isnan(out["nmi_kmeans"])
        assert np.isnan(out["purity_kmeans"])

    def test_kmeans_clusters_capped_by_pixels(self):
        X = np.array([[1.0, 0], [0, 1.0], [1.0, 0.1]])
        out = alignment_scores(X, np.array([0, 1, 0]), None, kmeans_clusters=10)
        assert out["kmeans_clusters"] == 3


class TestCollect:
    def test_only_pair_pixels_with_cap(self, tmp_path):
        ds = _CsvTiny(_index(tmp_path, [0, 1, 2]))
        X, y = collect_pair_features(
            _segmenter(),
            ds,
            negative_class=3,
            positive_class=5,
            batch_size=2,
            keep_fraction=1.0,
            max_per_class=10,
            rng=np.random.default_rng(0),
        )
        assert X.shape[1] == 16 and set(np.unique(y)) == {0, 1}
        assert (y == 0).sum() == 10 and (y == 1).sum() == 10

    def test_keep_fraction_subsamples(self, tmp_path):
        ds = _CsvTiny(_index(tmp_path, [0]))
        kw = dict(negative_class=3, positive_class=5, rng=np.random.default_rng(0))
        _, y_all = collect_pair_features(_segmenter(), ds, **kw)
        _, y_half = collect_pair_features(_segmenter(), ds, keep_fraction=0.5, **kw)
        assert 0 < len(y_half) < len(y_all)

    def test_no_pair_pixels(self, tmp_path):
        X, y = collect_pair_features(
            _segmenter(),
            _CsvTiny(_index(tmp_path, [1, 3])),
            negative_class=3,
            positive_class=5,
            rng=np.random.default_rng(0),
        )
        assert X.shape == (0, 16) and y.shape == (0,)


def _cfg(tmp_path, ckpt, **probe):
    episodes = tmp_path / "episodes.csv"
    pd.DataFrame(
        {
            "shots": [1, 2, 2],
            "draw": [0, 0, 0],
            "novel_class": [5, 5, 5],
            "idx": [2, 4, 1],
        }
    ).to_csv(episodes, index=False)
    ds = {"_target_": "tests.test_few_shot_separability_probe._CsvTiny"}
    node = {
        "checkpoint": {"path": str(ckpt)},
        "episodes_csv": str(episodes),
        "support_dataset": dict(ds),
        "test_dataset": dict(ds, window_index_cache=_index(tmp_path, [6, 7, 8])),
        "negative_class": 3,
        "positive_class": 5,
        "output_csv": str(tmp_path / "out" / "probe.csv"),
        "test_keep_fraction": 1.0,
    }
    node.update(probe)
    return OmegaConf.create(
        {"mode": "separability-probe", "model": _MODEL, "separability_probe": node}
    )


@pytest.fixture
def ckpt(tmp_path):
    torch.manual_seed(0)
    model = smp.UPerNet(**{k: v for k, v in _MODEL.items() if k != "_target_"})
    path = tmp_path / "base.ckpt"
    torch.save(
        {"state_dict": {f"model.{k}": v for k, v in model.state_dict().items()}}, path
    )
    return path


class TestRun:
    def test_writes_one_row_per_episode(self, tmp_path, ckpt):
        out = run_separability_probe(_cfg(tmp_path, ckpt))
        df = pd.read_csv(out)
        assert list(df["shots"]) == [1, 2] and list(df["draw"]) == [0, 0]
        assert (df["n_support_pos"] > 0).all() and (df["n_test_pos"] > 0).all()
        assert df["auroc_logreg"].between(0, 1).all()
        assert {"base_seed", "auroc_proto", "balacc_logreg", "ap_logreg"} <= set(
            df.columns
        )

    def test_filters_shots(self, tmp_path, ckpt):
        out = run_separability_probe(_cfg(tmp_path, ckpt, shots=[2]))
        assert list(pd.read_csv(out)["shots"]) == [2]

    def test_filters_draws(self, tmp_path, ckpt):
        out = run_separability_probe(_cfg(tmp_path, ckpt, draws=[0], shots=[2]))
        assert list(pd.read_csv(out)["shots"]) == [2]
        with pytest.raises(ValueError, match="no episode"):
            run_separability_probe(_cfg(tmp_path, ckpt, draws=[1]))

    def test_from_runner_uses_each_seed(self, tmp_path, ckpt, monkeypatch):
        import pytorch_segmentation_models_trainer.tools.few_shot.separability_probe as sp

        seen = []

        def fake(runner, seed):
            seen.append((runner, seed))
            return str(ckpt)

        monkeypatch.setattr(sp, "resolve_checkpoint_from_runner", fake)
        cfg = _cfg(tmp_path, ckpt, seeds=[1, 2], shots=[1])
        cfg.separability_probe.checkpoint = {"from_runner": "runs/r2"}
        df = pd.read_csv(run_separability_probe(cfg))
        assert seen == [("runs/r2", 1), ("runs/r2", 2)]
        assert list(df["base_seed"]) == [1, 2]

    def test_checkpoint_needs_one_source(self, tmp_path, ckpt):
        cfg = _cfg(tmp_path, ckpt)
        cfg.separability_probe.checkpoint = {}
        with pytest.raises(ValueError):
            run_separability_probe(cfg)


_REFINE = {
    "_target_": "pytorch_segmentation_models_trainer.custom_models.refinement_aware.RefinementAwareWrapper",
    "superclass_index": 3,
    "edge_weight": 0.0,
    "subprototype_weight": 0.1,
    "num_subprototypes": 3,
    "model": _MODEL,
}


@pytest.fixture
def refine_ckpt(tmp_path):
    from pytorch_segmentation_models_trainer.custom_models.refinement_aware import (
        RefinementAwareWrapper,
    )

    torch.manual_seed(0)
    plain = smp.UPerNet(**{k: v for k, v in _MODEL.items() if k != "_target_"})
    model = RefinementAwareWrapper(
        plain, superclass_index=3, edge_weight=0.0, num_subprototypes=3
    )
    path = tmp_path / "refine.ckpt"
    torch.save(
        {"state_dict": {f"model.{k}": v for k, v in model.state_dict().items()}}, path
    )
    return path


class TestAlignmentRun:
    def test_refine_model_writes_subprototype_alignment(self, tmp_path, refine_ckpt):
        align = tmp_path / "out" / "align.csv"
        cfg = _cfg(tmp_path, refine_ckpt, shots=[1], alignment_csv=str(align))
        cfg.model = _REFINE
        run_separability_probe(cfg)
        df = pd.read_csv(align)
        assert len(df) == 1 and df["base_seed"].isna().all()
        assert df.loc[0, "num_subproto"] == 3
        assert 0.0 <= df.loc[0, "nmi_subproto"] <= 1.0
        assert df.loc[0, "n_neg"] > 0 and df.loc[0, "n_pos"] > 0
        assert df.loc[0, "kmeans_clusters"] == 3  # default = number of sub-prototypes

    def test_plain_model_writes_kmeans_only(self, tmp_path, ckpt):
        align = tmp_path / "out" / "align.csv"
        cfg = _cfg(
            tmp_path,
            ckpt,
            shots=[1],
            alignment_csv=str(align),
            alignment_kmeans_clusters=2,
        )
        run_separability_probe(cfg)
        df = pd.read_csv(align)
        assert df.loc[0, "num_subproto"] == 0 and np.isnan(df.loc[0, "nmi_subproto"])
        assert df.loc[0, "kmeans_clusters"] == 2
        assert 0.0 <= df.loc[0, "nmi_kmeans"] <= 1.0

    def test_plain_model_default_kmeans_clusters(self, tmp_path, ckpt):
        align = tmp_path / "out" / "align.csv"
        run_separability_probe(
            _cfg(tmp_path, ckpt, shots=[1], alignment_csv=str(align))
        )
        assert pd.read_csv(align).loc[0, "kmeans_clusters"] == 4

    def test_no_alignment_csv_by_default(self, tmp_path, ckpt):
        run_separability_probe(_cfg(tmp_path, ckpt, shots=[1]))
        assert sorted(p.name for p in (tmp_path / "out").iterdir()) == ["probe.csv"]


def test_main_dispatches(tmp_path, ckpt):
    from pytorch_segmentation_models_trainer.main import main

    out = main.__wrapped__(_cfg(tmp_path, ckpt, shots=[1]))
    assert out.endswith("probe.csv")
