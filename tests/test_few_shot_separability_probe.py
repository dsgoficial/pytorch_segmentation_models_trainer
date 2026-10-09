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


def test_main_dispatches(tmp_path, ckpt):
    from pytorch_segmentation_models_trainer.main import main

    out = main.__wrapped__(_cfg(tmp_path, ckpt, shots=[1]))
    assert out.endswith("probe.csv")
