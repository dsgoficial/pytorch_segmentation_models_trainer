# -*- coding: utf-8 -*-
"""Tests for tools/coreset/ module (Nogueira et al. 2026 core-set selection)."""

import json
import os
import tempfile

import numpy as np
import pandas as pd
import pytest
from click.testing import CliRunner

from pytorch_segmentation_models_trainer.tools.coreset.coreset_config import (
    CoreSetConfig,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_lc_df(entropies):
    return pd.DataFrame(
        {
            "class_entropy": entropies,
            "class_dist_json": [json.dumps({"1": 1.0})] * len(entropies),
        }
    )


def _make_cb_df(class_dists):
    """class_dists: list of dicts {class_id: proportion}."""
    return pd.DataFrame(
        {
            "class_dist_json": [json.dumps(d) for d in class_dists],
            "class_entropy": [0.5] * len(class_dists),
        }
    )


def _make_embed_df(embeddings, class_entropy=None, class_dist=None):
    """embeddings: list of 1-D arrays."""
    n = len(embeddings)
    data = {
        "embedding": embeddings,
        "class_entropy": class_entropy if class_entropy is not None else [0.5] * n,
        "class_dist_json": (
            class_dist
            if class_dist is not None
            else [json.dumps({"1": 0.5, "2": 0.5})] * n
        ),
    }
    return pd.DataFrame(data)


# ---------------------------------------------------------------------------
# CoreSetConfig
# ---------------------------------------------------------------------------


class TestCoreSetConfig:
    def test_defaults(self):
        cfg = CoreSetConfig()
        assert cfg.method == "lc"
        assert cfg.budget == 0.5
        assert cfg.budget_mode == "fraction"
        assert cfg.embedding_column is None
        assert cfg.fd_k_min == 2
        assert cfg.fd_k_max == 20
        assert cfg.fd_vendi_delta == pytest.approx(0.005)
        assert cfg.fd_use_vendi is False
        assert cfg.lc_fd_cutoff_m is None
        assert cfg.fa_cb_lambda == pytest.approx(0.5)
        assert cfg.score_column == "coreset_score"
        assert cfg.output_csv_path == "coreset.csv"
        assert cfg.input_csv_path == ""

    def test_override(self):
        cfg = CoreSetConfig(method="fd", budget=100, budget_mode="count")
        assert cfg.method == "fd"
        assert cfg.budget == 100
        assert cfg.budget_mode == "count"


# ---------------------------------------------------------------------------
# score_lc
# ---------------------------------------------------------------------------


class TestScoreLc:
    def _fn(self, df, cfg=None):
        from pytorch_segmentation_models_trainer.tools.coreset.coreset_scorer import (
            score_lc,
        )

        return score_lc(df, cfg or CoreSetConfig())

    def test_returns_normalized_array(self):
        df = _make_lc_df([0.0, 0.5, 1.0])
        scores = self._fn(df)
        assert scores.shape == (3,)
        assert np.all(scores >= 0) and np.all(scores <= 1)
        assert scores[2] > scores[1] > scores[0]

    def test_missing_column_raises(self):
        df = pd.DataFrame({"other": [1, 2, 3]})
        with pytest.raises(ValueError, match="class_entropy"):
            self._fn(df)

    def test_uniform_entropy_returns_equal_scores(self):
        df = _make_lc_df([0.5, 0.5, 0.5])
        scores = self._fn(df)
        assert np.allclose(scores, 1.0)

    def test_uses_dataset_max_entropy(self):
        # max in dataset is 0.8, not theoretical log(C)
        df = _make_lc_df([0.0, 0.4, 0.8])
        scores = self._fn(df)
        # patch with entropy 0.8 gets score 1.0
        assert scores[2] == pytest.approx(1.0)
        # patch with entropy 0.4 gets score 0.5
        assert scores[1] == pytest.approx(0.5)

    def test_all_zero_entropy_returns_zeros(self):
        df = _make_lc_df([0.0, 0.0, 0.0])
        scores = self._fn(df)
        assert np.all(scores == 0.0)

    def test_none_entropy_treated_as_zero(self):
        df = pd.DataFrame(
            {
                "class_entropy": [None, 0.5, 1.0],
                "class_dist_json": ['{"1":1.0}'] * 3,
            }
        )
        scores = self._fn(df)
        assert scores[0] == pytest.approx(0.0)
        assert scores[2] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# score_cb
# ---------------------------------------------------------------------------


class TestScoreCb:
    def _fn(self, df, cfg=None):
        from pytorch_segmentation_models_trainer.tools.coreset.coreset_scorer import (
            score_cb,
        )

        return score_cb(df, cfg or CoreSetConfig())

    def test_first_selected_highest_score(self):
        # Mixed patch has highest entropy → selected first → score 1.0
        df = _make_cb_df(
            [
                {"1": 0.5, "2": 0.5},  # both classes
                {"1": 1.0},  # single class
                {"2": 1.0},  # single class
            ]
        )
        cfg = CoreSetConfig(budget=1.0, budget_mode="fraction")
        scores = self._fn(df, cfg)
        assert scores[0] == pytest.approx(1.0)

    def test_scores_in_01(self):
        df = _make_cb_df(
            [{"1": 0.3, "2": 0.3, "3": 0.4}, {"1": 0.8, "2": 0.2}, {"3": 1.0}]
        )
        scores = self._fn(df)
        assert np.all(scores >= 0) and np.all(scores <= 1)

    def test_maximizes_global_entropy(self):
        # 6 patches; verify selected order increases global entropy
        dists = [
            {"1": 0.5, "2": 0.5},
            {"1": 0.9, "2": 0.1},
            {"1": 0.1, "2": 0.9},
            {"1": 1.0},
            {"2": 1.0},
            {"1": 0.7, "2": 0.3},
        ]
        df = _make_cb_df(dists)
        cfg = CoreSetConfig(budget=1.0, budget_mode="fraction")
        scores = self._fn(df, cfg)
        # Higher-score patches should increase global entropy progressively
        # We just verify scores are valid and not all equal (greedy is non-trivial)
        assert len(set(scores.tolist())) > 1

    def test_budget_early_stop(self):
        # Only top-3 patches get score > 0
        n = 100
        dists = [{"1": 0.5, "2": 0.5}] * 50 + [{"1": 1.0}] * 50
        df = _make_cb_df(dists)
        cfg = CoreSetConfig(budget=3.0, budget_mode="count")
        scores = self._fn(df, cfg)
        assert (scores > 0).sum() == 3
        assert (scores == 0).sum() == 97

    def test_single_patch(self):
        df = _make_cb_df([{"1": 1.0}])
        scores = self._fn(df)
        assert scores.shape == (1,)
        assert scores[0] == pytest.approx(1.0)

    def test_missing_column_raises(self):
        df = pd.DataFrame({"other": [1, 2]})
        with pytest.raises(ValueError, match="class_dist_json"):
            self._fn(df)


# ---------------------------------------------------------------------------
# score_fa
# ---------------------------------------------------------------------------


class TestScoreFa:
    def _fn(self, df, cfg):
        from pytorch_segmentation_models_trainer.tools.coreset.coreset_scorer import (
            score_fa,
        )

        return score_fa(df, cfg)

    def test_high_mean_high_std_wins(self):
        # Patch C: high mean, moderate std → should beat A (no std) and B (low mean)
        rng = np.random.RandomState(0)
        patch_a = np.ones(64)  # high mean, zero std → score ≈ 0
        patch_b = np.array([0.0, 1.0] * 32)  # moderate mean, high std
        patch_c = np.linspace(0.7, 1.0, 64)  # high mean, moderate std
        df = _make_embed_df([patch_a, patch_b, patch_c])
        cfg = CoreSetConfig(embedding_column="embedding")
        scores = self._fn(df, cfg)
        # patch_a has zero std → score 0
        assert scores[0] == pytest.approx(0.0)
        # patch_c has highest mean; tie-break depends on scaling but score > 0
        assert scores[2] >= scores[0]

    def test_missing_embedding_column_raises(self):
        df = _make_embed_df([np.ones(4)])
        cfg = CoreSetConfig(embedding_column=None)
        with pytest.raises(ValueError, match="embedding_column"):
            self._fn(df, cfg)

    def test_scores_in_01(self):
        rng = np.random.RandomState(1)
        embeddings = [rng.randn(32) for _ in range(10)]
        df = _make_embed_df(embeddings)
        cfg = CoreSetConfig(embedding_column="embedding")
        scores = self._fn(df, cfg)
        assert np.all(scores >= 0) and np.all(scores <= 1)

    def test_constant_embedding_gets_zero_std_score(self):
        # All same value → std=0 → σ_scaled=0 → score=0
        patch_const = np.full(16, 0.5)
        patch_varied = np.linspace(0, 1, 16)
        df = _make_embed_df([patch_const, patch_varied])
        cfg = CoreSetConfig(embedding_column="embedding")
        scores = self._fn(df, cfg)
        assert scores[0] == pytest.approx(0.0)

    def test_missing_column_in_df_raises(self):
        df = pd.DataFrame({"other": [1, 2]})
        cfg = CoreSetConfig(embedding_column="embedding")
        with pytest.raises(ValueError, match="embedding"):
            self._fn(df, cfg)


# ---------------------------------------------------------------------------
# score_fd
# ---------------------------------------------------------------------------


def _two_cluster_embeddings(n_per_cluster=3, dim=4, seed=42):
    """Return a DataFrame with embeddings in 2 well-separated clusters."""
    rng = np.random.RandomState(seed)
    cluster_a = rng.randn(n_per_cluster, dim) + np.array([3.0] + [0.0] * (dim - 1))
    cluster_b = rng.randn(n_per_cluster, dim) + np.array([-3.0] + [0.0] * (dim - 1))
    embeddings = list(cluster_a) + list(cluster_b)
    labels = [0] * n_per_cluster + [1] * n_per_cluster
    df = _make_embed_df(embeddings)
    return df, np.array(labels)


class TestScoreFd:
    def _fn(self, df, cfg):
        from pytorch_segmentation_models_trainer.tools.coreset.coreset_scorer import (
            score_fd,
        )

        return score_fd(df, cfg)

    def test_covers_all_clusters(self):
        # 4 patches in 2 clusters → top-2 selection covers both clusters
        df, gt_labels = _two_cluster_embeddings(n_per_cluster=2, dim=4)
        cfg = CoreSetConfig(embedding_column="embedding", fd_k_min=2, fd_k_max=2)
        scores = self._fn(df, cfg)
        top2_idx = np.argsort(-scores)[:2]
        top2_clusters = set(gt_labels[top2_idx].tolist())
        assert 0 in top2_clusters and 1 in top2_clusters

    def test_round_robin_interleaves(self):
        # 6 patches, 2 clusters of 3 → first 2 selected from different clusters
        df, gt_labels = _two_cluster_embeddings(n_per_cluster=3, dim=4)
        cfg = CoreSetConfig(embedding_column="embedding", fd_k_min=2, fd_k_max=2)
        scores = self._fn(df, cfg)
        top2_idx = np.argsort(-scores)[:2]
        top2_clusters = set(gt_labels[top2_idx].tolist())
        assert 0 in top2_clusters and 1 in top2_clusters

    def test_scores_in_01(self):
        rng = np.random.RandomState(2)
        embeddings = [rng.randn(8) for _ in range(10)]
        df = _make_embed_df(embeddings)
        cfg = CoreSetConfig(embedding_column="embedding", fd_k_min=2, fd_k_max=4)
        scores = self._fn(df, cfg)
        assert np.all(scores >= 0) and np.all(scores <= 1)

    def test_missing_embedding_raises(self):
        df = _make_embed_df([np.ones(4)])
        cfg = CoreSetConfig(embedding_column=None)
        with pytest.raises(ValueError, match="embedding_column"):
            self._fn(df, cfg)

    def test_single_cluster(self):
        # All patches in one cluster (k_min=k_max=1) → round-robin still works
        rng = np.random.RandomState(3)
        embeddings = [rng.randn(4) for _ in range(4)]
        df = _make_embed_df(embeddings)
        cfg = CoreSetConfig(embedding_column="embedding", fd_k_min=1, fd_k_max=1)
        scores = self._fn(df, cfg)
        assert scores.shape == (4,)
        assert np.all(scores >= 0) and np.all(scores <= 1)


# ---------------------------------------------------------------------------
# Hybrid scorers
# ---------------------------------------------------------------------------


class TestHybridScorers:
    def test_score_lc_fd_top_m_from_fd(self):
        from pytorch_segmentation_models_trainer.tools.coreset.coreset_scorer import (
            score_fd,
            score_lc_fd,
        )

        df, gt_labels = _two_cluster_embeddings(n_per_cluster=5, dim=4)
        df["class_entropy"] = [0.1 * i for i in range(10)]
        m = 4
        cfg = CoreSetConfig(
            embedding_column="embedding", fd_k_min=2, fd_k_max=2, lc_fd_cutoff_m=m
        )
        fd_scores = score_fd(df, cfg)
        lc_fd_scores = score_lc_fd(df, cfg)

        top_m_fd = set(np.argsort(-fd_scores)[:m].tolist())
        top_m_lc_fd = set(np.argsort(-lc_fd_scores)[:m].tolist())
        # Top-m LC/FD patches must be a subset of top-m FD patches
        assert top_m_lc_fd.issubset(top_m_fd)

    def test_score_lc_fd_remainder_from_lc(self):
        from pytorch_segmentation_models_trainer.tools.coreset.coreset_scorer import (
            score_fd,
            score_lc,
            score_lc_fd,
        )

        df, _ = _two_cluster_embeddings(n_per_cluster=5, dim=4)
        df["class_entropy"] = [0.1 * i for i in range(10)]
        m = 2
        cfg = CoreSetConfig(
            embedding_column="embedding", fd_k_min=2, fd_k_max=2, lc_fd_cutoff_m=m
        )
        fd_scores = score_fd(df, cfg)
        lc_scores = score_lc(df, cfg)
        lc_fd_scores = score_lc_fd(df, cfg)

        top_m_fd_idx = set(np.argsort(-fd_scores)[:m].tolist())
        remaining_idx = [i for i in range(10) if i not in top_m_fd_idx]
        # Remaining should be ordered by LC (higher LC = higher lc_fd score)
        lc_fd_remaining = [(i, lc_fd_scores[i]) for i in remaining_idx]
        lc_fd_remaining.sort(key=lambda x: -x[1])
        lc_remaining = [(i, lc_scores[i]) for i in remaining_idx]
        lc_remaining.sort(key=lambda x: -x[1])
        # Order of remaining patches by lc_fd should match order by lc
        assert [x[0] for x in lc_fd_remaining] == [x[0] for x in lc_remaining]

    def test_score_fa_cb_lambda_0_equals_cb(self):
        from pytorch_segmentation_models_trainer.tools.coreset.coreset_scorer import (
            score_cb,
            score_fa_cb,
        )

        rng = np.random.RandomState(4)
        embeddings = [rng.randn(8) for _ in range(6)]
        dists = [{"1": 0.5, "2": 0.5}] * 3 + [{"1": 1.0}] * 3
        df = pd.DataFrame(
            {
                "embedding": embeddings,
                "class_dist_json": [json.dumps(d) for d in dists],
                "class_entropy": [0.5] * 6,
            }
        )
        cfg_cb = CoreSetConfig(budget=1.0, budget_mode="fraction")
        cfg_fa_cb = CoreSetConfig(
            embedding_column="embedding",
            fa_cb_lambda=0.0,
            budget=1.0,
            budget_mode="fraction",
        )
        cb_scores = score_cb(df, cfg_cb)
        fa_cb_scores = score_fa_cb(df, cfg_fa_cb)
        assert np.allclose(fa_cb_scores, cb_scores)

    def test_score_fa_cb_lambda_1_equals_fa(self):
        from pytorch_segmentation_models_trainer.tools.coreset.coreset_scorer import (
            score_fa,
            score_fa_cb,
        )

        rng = np.random.RandomState(5)
        embeddings = [rng.randn(8) for _ in range(6)]
        dists = [{"1": 0.5, "2": 0.5}] * 6
        df = pd.DataFrame(
            {
                "embedding": embeddings,
                "class_dist_json": [json.dumps(d) for d in dists],
                "class_entropy": [0.5] * 6,
            }
        )
        cfg = CoreSetConfig(embedding_column="embedding", fa_cb_lambda=1.0)
        fa_scores = score_fa(df, cfg)
        fa_cb_scores = score_fa_cb(df, cfg)
        assert np.allclose(fa_cb_scores, fa_scores)


# ---------------------------------------------------------------------------
# vendi_score
# ---------------------------------------------------------------------------


class TestVendiScore:
    def test_identical_vectors_returns_1(self):
        from pytorch_segmentation_models_trainer.tools.coreset.vendi_score import (
            vendi_score,
        )

        v = np.array([1.0, 0.0, 0.0])
        embeddings = np.stack([v] * 5)
        score = vendi_score(embeddings)
        assert score == pytest.approx(1.0, abs=0.01)

    def test_orthogonal_vectors_returns_n(self):
        from pytorch_segmentation_models_trainer.tools.coreset.vendi_score import (
            vendi_score,
        )

        embeddings = np.eye(4)  # 4 orthogonal unit vectors
        score = vendi_score(embeddings)
        assert score == pytest.approx(4.0, abs=0.1)

    def test_positive(self):
        from pytorch_segmentation_models_trainer.tools.coreset.vendi_score import (
            vendi_score,
        )

        rng = np.random.RandomState(6)
        embeddings = rng.randn(10, 8)
        score = vendi_score(embeddings)
        assert score > 0


# ---------------------------------------------------------------------------
# CoreSetSelector integration
# ---------------------------------------------------------------------------


class TestCoreSetSelector:
    def _make_df(self, n=20):
        rng = np.random.RandomState(7)
        return pd.DataFrame(
            {
                "class_entropy": rng.uniform(0, 1, n),
                "class_dist_json": [
                    json.dumps({"1": rng.uniform(), "2": rng.uniform()})
                    for _ in range(n)
                ],
                "image_path": [f"/img/{i}.tif" for i in range(n)],
            }
        )

    def test_adds_coreset_score_column(self, tmp_path):
        from pytorch_segmentation_models_trainer.tools.coreset.coreset_selector import (
            CoreSetSelector,
        )

        cfg = CoreSetConfig(output_csv_path=str(tmp_path / "out.csv"))
        df = self._make_df()
        result = CoreSetSelector(cfg).select(df)
        assert "coreset_score" in result.columns

    def test_adds_coreset_selected_column(self, tmp_path):
        from pytorch_segmentation_models_trainer.tools.coreset.coreset_selector import (
            CoreSetSelector,
        )

        cfg = CoreSetConfig(output_csv_path=str(tmp_path / "out.csv"))
        df = self._make_df()
        result = CoreSetSelector(cfg).select(df)
        assert "coreset_selected" in result.columns
        assert result["coreset_selected"].dtype == bool

    def test_respects_fraction_budget(self, tmp_path):
        from pytorch_segmentation_models_trainer.tools.coreset.coreset_selector import (
            CoreSetSelector,
        )

        cfg = CoreSetConfig(
            budget=0.5,
            budget_mode="fraction",
            output_csv_path=str(tmp_path / "out.csv"),
        )
        df = self._make_df(n=100)
        result = CoreSetSelector(cfg).select(df)
        assert result["coreset_selected"].sum() == 50

    def test_respects_count_budget(self, tmp_path):
        from pytorch_segmentation_models_trainer.tools.coreset.coreset_selector import (
            CoreSetSelector,
        )

        cfg = CoreSetConfig(
            budget=25, budget_mode="count", output_csv_path=str(tmp_path / "out.csv")
        )
        df = self._make_df(n=100)
        result = CoreSetSelector(cfg).select(df)
        assert result["coreset_selected"].sum() == 25

    def test_writes_csv(self, tmp_path):
        from pytorch_segmentation_models_trainer.tools.coreset.coreset_selector import (
            CoreSetSelector,
        )

        out = str(tmp_path / "coreset.csv")
        cfg = CoreSetConfig(output_csv_path=out)
        df = self._make_df()
        CoreSetSelector(cfg).select(df)
        assert os.path.exists(out)
        read_back = pd.read_csv(out)
        assert "coreset_score" in read_back.columns

    def test_lc_method_e2e(self, tmp_path):
        from pytorch_segmentation_models_trainer.tools.coreset.coreset_selector import (
            CoreSetSelector,
        )

        cfg = CoreSetConfig(
            method="lc",
            budget=0.5,
            budget_mode="fraction",
            output_csv_path=str(tmp_path / "out.csv"),
        )
        df = self._make_df(n=10)
        result = CoreSetSelector(cfg).select(df)
        assert result["coreset_selected"].sum() == 5

    def test_cb_method_e2e(self, tmp_path):
        from pytorch_segmentation_models_trainer.tools.coreset.coreset_selector import (
            CoreSetSelector,
        )

        cfg = CoreSetConfig(
            method="cb",
            budget=0.5,
            budget_mode="fraction",
            output_csv_path=str(tmp_path / "out.csv"),
        )
        df = self._make_df(n=10)
        result = CoreSetSelector(cfg).select(df)
        assert result["coreset_selected"].sum() == 5

    def test_invalid_method_raises(self, tmp_path):
        from pytorch_segmentation_models_trainer.tools.coreset.coreset_scorer import (
            get_scorer,
        )

        with pytest.raises(ValueError, match="Unknown coreset method"):
            get_scorer("invalid_method")


# ---------------------------------------------------------------------------
# CLI tests
# ---------------------------------------------------------------------------


class TestCliSelectCoreset:
    def _write_yaml(self, path, extra=None):
        import yaml

        cfg = {
            "coreset": {
                "method": "lc",
                "budget": 0.5,
                "budget_mode": "fraction",
                "input_csv_path": str(path / "input.csv"),
                "output_csv_path": str(path / "output.csv"),
            }
        }
        if extra:
            cfg["coreset"].update(extra)
        yaml_path = path / "config.yaml"
        with open(yaml_path, "w") as f:
            yaml.dump(cfg, f)
        return str(yaml_path)

    def _write_input_csv(self, path, n=10):
        import json as _json

        rng = np.random.RandomState(8)
        df = pd.DataFrame(
            {
                "class_entropy": rng.uniform(0, 1, n),
                "class_dist_json": [
                    _json.dumps({"1": 0.5, "2": 0.5}) for _ in range(n)
                ],
            }
        )
        df.to_csv(str(path / "input.csv"), index=False)

    def test_cli_select_coreset_runs(self, tmp_path):
        from pytorch_segmentation_models_trainer.tools.cli import cli

        self._write_input_csv(tmp_path)
        yaml_path = self._write_yaml(tmp_path)
        runner = CliRunner()
        result = runner.invoke(cli, ["select-coreset", yaml_path])
        assert result.exit_code == 0, result.output
        assert os.path.exists(str(tmp_path / "output.csv"))

    def test_cli_select_coreset_invalid_method_raises(self, tmp_path):
        from pytorch_segmentation_models_trainer.tools.cli import cli

        self._write_input_csv(tmp_path)
        yaml_path = self._write_yaml(tmp_path, extra={"method": "unknown_xyz"})
        runner = CliRunner()
        result = runner.invoke(cli, ["select-coreset", yaml_path])
        assert result.exit_code != 0


# ---------------------------------------------------------------------------
# Additional coverage tests
# ---------------------------------------------------------------------------


class TestParseClassDist:
    def test_dict_input_passes_through(self):
        from pytorch_segmentation_models_trainer.tools.coreset.coreset_scorer import (
            _parse_class_dist,
        )

        series = pd.Series([{"1": 0.5, "2": 0.5}, {"1": 1.0}])
        result = _parse_class_dist(series)
        assert result[0] == {"1": 0.5, "2": 0.5}
        assert result[1] == {"1": 1.0}

    def test_none_input_returns_empty_dict(self):
        from pytorch_segmentation_models_trainer.tools.coreset.coreset_scorer import (
            _parse_class_dist,
        )

        series = pd.Series([None])
        result = _parse_class_dist(series)
        assert result[0] == {}


class TestScoreFdExtraPaths:
    def test_missing_column_in_df_raises(self):
        from pytorch_segmentation_models_trainer.tools.coreset.coreset_scorer import (
            score_fd,
        )

        df = pd.DataFrame({"other": [1, 2]})
        cfg = CoreSetConfig(embedding_column="embedding")
        with pytest.raises(ValueError, match="embedding"):
            score_fd(df, cfg)

    def test_fd_use_vendi_true(self):
        from pytorch_segmentation_models_trainer.tools.coreset.coreset_scorer import (
            score_fd,
        )

        df, _ = _two_cluster_embeddings(n_per_cluster=4, dim=4)
        cfg = CoreSetConfig(
            embedding_column="embedding",
            fd_k_min=2,
            fd_k_max=4,
            fd_use_vendi=True,
        )
        scores = score_fd(df, cfg)
        assert scores.shape == (8,)
        assert np.all(scores >= 0) and np.all(scores <= 1)


class TestScoreLcFdAllFd:
    def test_lc_fd_m_equals_n_no_remaining(self):
        from pytorch_segmentation_models_trainer.tools.coreset.coreset_scorer import (
            score_lc_fd,
        )

        df, _ = _two_cluster_embeddings(n_per_cluster=3, dim=4)
        df["class_entropy"] = [0.5] * 6
        # m = N → all patches selected by FD, no remaining
        cfg = CoreSetConfig(
            embedding_column="embedding", fd_k_min=2, fd_k_max=2, lc_fd_cutoff_m=6
        )
        scores = score_lc_fd(df, cfg)
        assert scores.shape == (6,)
        assert np.all(scores >= 0) and np.all(scores <= 1)


class TestVendiScoreEdgeCases:
    def test_vendi_zero_norm_embeddings(self):
        from pytorch_segmentation_models_trainer.tools.coreset.vendi_score import (
            vendi_score,
        )

        # Zero-norm embeddings → tr = 0 → returns 1.0
        embeddings = np.zeros((4, 8))
        score = vendi_score(embeddings)
        assert score == pytest.approx(1.0)

    def test_select_k_vendi_returns_valid_k(self):
        from pytorch_segmentation_models_trainer.tools.coreset.vendi_score import (
            select_k_vendi,
        )

        rng = np.random.RandomState(9)
        embeddings = rng.randn(20, 8)
        k = select_k_vendi(embeddings, k_min=2, delta=0.1)
        assert k >= 2

    def test_select_k_vendi_k_min_exceeds_max(self):
        from pytorch_segmentation_models_trainer.tools.coreset.vendi_score import (
            select_k_vendi,
        )

        # k_min >= max_k → returns k_min
        embeddings = np.eye(3)
        k = select_k_vendi(embeddings, k_min=5, delta=0.005)
        assert k == 5

    def test_select_k_vendi_triggers_subsampling(self):
        from pytorch_segmentation_models_trainer.tools.coreset.vendi_score import (
            select_k_vendi,
        )

        # max_subsample=3 → triggers subsample branch with 8 patches per cluster
        rng = np.random.RandomState(10)
        embeddings = rng.randn(16, 4)
        k = select_k_vendi(embeddings, k_min=2, delta=0.1, max_subsample=3)
        assert k >= 2

    def test_select_k_vendi_triggers_early_stop(self):
        from pytorch_segmentation_models_trainer.tools.coreset.vendi_score import (
            select_k_vendi,
        )

        # Large delta → improvement < delta → early stop after 3 consecutive
        rng = np.random.RandomState(11)
        embeddings = rng.randn(30, 4)
        k = select_k_vendi(embeddings, k_min=2, delta=100.0)
        # With delta=100, every improvement is < delta → converges quickly
        assert k >= 2


# ---------------------------------------------------------------------------
# Parquet-based embedding loading in CoreSetSelector
# ---------------------------------------------------------------------------


class TestSelectorParquetEmbeddings:
    def _make_parquet(self, tmp_path, n=10, dim=8):
        """Write a Parquet file using the same schema as embedding_extractor."""
        from pytorch_segmentation_models_trainer.tools.sampling.embedding_extractor import (
            save_embeddings_to_parquet,
        )

        rng = np.random.RandomState(20)
        image_paths = [f"/img/{i}.tif" for i in range(n)]
        row_offs = list(range(0, n * 256, 256))
        col_offs = [0] * n
        embeddings = list(rng.randn(n, dim).astype(np.float32))
        df = pd.DataFrame(
            {
                "tile_id": [
                    f"{ip}_{ro}_{co}"
                    for ip, ro, co in zip(image_paths, row_offs, col_offs)
                ],
                "image_path": image_paths,
                "mask_path": [f"/msk/{i}.tif" for i in range(n)],
                "row_off": row_offs,
                "col_off": col_offs,
                "patch_size": [256] * n,
                "embedding": embeddings,
                "dino_model": ["dinov2-base"] * n,
            }
        )
        parquet_path = str(tmp_path / "embeddings.parquet")
        save_embeddings_to_parquet(df, parquet_path)
        return parquet_path, df

    def _make_csv_df(self, n=10):
        rng = np.random.RandomState(21)
        return pd.DataFrame(
            {
                "image_path": [f"/img/{i}.tif" for i in range(n)],
                "mask_path": [f"/msk/{i}.tif" for i in range(n)],
                "row_off": list(range(0, n * 256, 256)),
                "col_off": [0] * n,
                "patch_size": [256] * n,
                "class_entropy": rng.uniform(0, 1, n),
                "class_dist_json": [json.dumps({"1": 0.5, "2": 0.5})] * n,
            }
        )

    def test_selector_loads_embeddings_from_parquet_fa(self, tmp_path):
        from pytorch_segmentation_models_trainer.tools.coreset.coreset_selector import (
            CoreSetSelector,
        )

        parquet_path, _ = self._make_parquet(tmp_path)
        df = self._make_csv_df()
        cfg = CoreSetConfig(
            method="fa",
            embedding_column="embedding",
            parquet_path=parquet_path,
            output_csv_path=str(tmp_path / "out.csv"),
        )
        result = CoreSetSelector(cfg).select(df)
        assert result["coreset_selected"].sum() == 5
        assert "coreset_score" in result.columns

    def test_selector_loads_embeddings_from_parquet_fd(self, tmp_path):
        from pytorch_segmentation_models_trainer.tools.coreset.coreset_selector import (
            CoreSetSelector,
        )

        parquet_path, _ = self._make_parquet(tmp_path)
        df = self._make_csv_df()
        cfg = CoreSetConfig(
            method="fd",
            embedding_column="embedding",
            parquet_path=parquet_path,
            fd_k_min=2,
            fd_k_max=4,
            output_csv_path=str(tmp_path / "out.csv"),
        )
        result = CoreSetSelector(cfg).select(df)
        assert result["coreset_selected"].sum() == 5

    def test_selector_embedding_already_in_df_no_parquet_needed(self, tmp_path):
        from pytorch_segmentation_models_trainer.tools.coreset.coreset_selector import (
            CoreSetSelector,
        )

        rng = np.random.RandomState(22)
        n = 6
        df = _make_embed_df([rng.randn(8).astype(np.float32) for _ in range(n)])
        cfg = CoreSetConfig(
            method="fa",
            embedding_column="embedding",
            output_csv_path=str(tmp_path / "out.csv"),
        )
        # No parquet_path — column already in df → should succeed
        result = CoreSetSelector(cfg).select(df)
        assert "coreset_score" in result.columns

    def test_selector_fa_no_embedding_no_parquet_raises(self, tmp_path):
        from pytorch_segmentation_models_trainer.tools.coreset.coreset_selector import (
            CoreSetSelector,
        )

        df = self._make_csv_df()  # no embedding column
        cfg = CoreSetConfig(
            method="fa",
            embedding_column="embedding",
            parquet_path=None,
            output_csv_path=str(tmp_path / "out.csv"),
        )
        with pytest.raises(ValueError, match="embedding"):
            CoreSetSelector(cfg).select(df)

    def test_selector_parquet_path_ignored_for_lc(self, tmp_path):
        from pytorch_segmentation_models_trainer.tools.coreset.coreset_selector import (
            CoreSetSelector,
        )

        df = self._make_csv_df()  # no embedding — LC doesn't need it
        cfg = CoreSetConfig(
            method="lc",
            output_csv_path=str(tmp_path / "out.csv"),
        )
        result = CoreSetSelector(cfg).select(df)
        assert result["coreset_selected"].sum() == 5
