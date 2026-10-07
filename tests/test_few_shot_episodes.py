# -*- coding: utf-8 -*-
"""Tests for the few-shot support episode generator."""

import numpy as np
import pandas as pd
import pytest
import rasterio
from omegaconf import OmegaConf
from rasterio.transform import from_origin

from pytorch_segmentation_models_trainer.config_definitions.few_shot_config import (
    FewShotEpisodesConfig,
)
from pytorch_segmentation_models_trainer.tools.few_shot.episodes import (
    build_fewshot_episodes,
    compute_class_fractions,
    sample_episodes,
)


def _records(df, base):
    return [dict(r, mask_path=base / r["mask_path"]) for r in df.to_dict("records")]


def _write_mask(path, array):
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=array.shape[0],
        width=array.shape[1],
        count=1,
        dtype="uint8",
        transform=from_origin(0, 0, 1, 1),
    ) as dst:
        dst.write(array.astype(np.uint8), 1)


@pytest.fixture
def mask_dir(tmp_path):
    """Six 10x10 masks; tile i has i*10 pixels of class 5 (0..50%), rest 3."""
    d = tmp_path / "masks"
    d.mkdir()
    rows = []
    for i in range(6):
        m = np.full((10, 10), 3, dtype=np.uint8)
        m.flat[: i * 10] = 5
        _write_mask(d / f"{i}.tif", m)
        rows.append(
            {
                "mask_path": f"{i}.tif",
                "row_off": 0,
                "col_off": 0,
                "width": 10,
                "height": 10,
            }
        )
    # tile 6: 50% class 5, 50% NODATA -> fraction over valid pixels = 1.0
    m = np.full((10, 10), 255, dtype=np.uint8)
    m.flat[:50] = 5
    _write_mask(d / "6.tif", m)
    rows.append(
        {"mask_path": "6.tif", "row_off": 0, "col_off": 0, "width": 10, "height": 10}
    )
    csv = tmp_path / "train.csv"
    pd.DataFrame(rows).to_csv(csv, index=False)
    return d, csv


class TestComputeClassFractions:
    def test_fractions_over_valid_pixels(self, mask_dir):
        d, csv = mask_dir
        df = pd.read_csv(csv)
        frac = compute_class_fractions(_records(df, d), [5])
        assert frac.shape == (7, 1)
        assert frac.dtype == np.float64
        np.testing.assert_allclose(frac[:6, 0], [0, 0.1, 0.2, 0.3, 0.4, 0.5])
        assert frac[6, 0] == pytest.approx(1.0)

    def test_window_offsets_are_respected(self, tmp_path):
        m = np.zeros((4, 8), dtype=np.uint8)
        m[:, 4:] = 5
        _write_mask(tmp_path / "a.tif", m)
        df = pd.DataFrame(
            [
                {
                    "mask_path": "a.tif",
                    "row_off": 0,
                    "col_off": 0,
                    "width": 4,
                    "height": 4,
                },
                {
                    "mask_path": "a.tif",
                    "row_off": 0,
                    "col_off": 4,
                    "width": 4,
                    "height": 4,
                },
            ]
        )
        frac = compute_class_fractions(_records(df, tmp_path), [5, 0])
        np.testing.assert_allclose(frac, [[0.0, 1.0], [1.0, 0.0]])

    def test_all_ignore_window_has_zero_fraction(self, tmp_path):
        _write_mask(tmp_path / "a.tif", np.full((4, 4), 255))
        df = pd.DataFrame(
            [
                {
                    "mask_path": "a.tif",
                    "row_off": 0,
                    "col_off": 0,
                    "width": 4,
                    "height": 4,
                }
            ]
        )
        frac = compute_class_fractions(_records(df, tmp_path), [5])
        assert frac[0, 0] == 0.0


class TestSampleEpisodes:
    def test_nested_supports_and_eligibility(self):
        frac = np.array([[0.0], [0.1], [0.2], [0.3], [0.4], [0.5]])
        eps = sample_episodes(
            frac, [5], shots=[1, 3], n_draws=4, min_fraction=0.15, seed=7
        )
        assert set(eps.columns) == {"shots", "draw", "novel_class", "window_idx"}
        for draw in range(4):
            k1 = eps.query("draw == @draw and shots == 1").window_idx.tolist()
            k3 = eps.query("draw == @draw and shots == 3").window_idx.tolist()
            assert len(k1) == 1 and len(k3) == 3
            assert k1 == k3[:1]  # nested: K=1 is a prefix of K=3
            assert set(k3) <= {2, 3, 4, 5}

    def test_deterministic_given_seed(self):
        frac = np.random.default_rng(0).random((30, 1))
        a = sample_episodes(
            frac, [5], shots=[1, 5], n_draws=3, min_fraction=0.2, seed=1
        )
        b = sample_episodes(
            frac, [5], shots=[1, 5], n_draws=3, min_fraction=0.2, seed=1
        )
        pd.testing.assert_frame_equal(a, b)
        c = sample_episodes(
            frac, [5], shots=[1, 5], n_draws=3, min_fraction=0.2, seed=2
        )
        assert not a.equals(c)

    def test_draws_differ(self):
        frac = np.full((50, 1), 0.5)
        eps = sample_episodes(frac, [5], shots=[5], n_draws=2, min_fraction=0.1, seed=0)
        d0 = eps.query("draw == 0").window_idx.tolist()
        d1 = eps.query("draw == 1").window_idx.tolist()
        assert d0 != d1

    def test_one_support_set_per_novel_class(self):
        frac = np.array([[0.5, 0.0], [0.5, 0.0], [0.0, 0.5], [0.0, 0.5]])
        eps = sample_episodes(
            frac, [5, 6], shots=[2], n_draws=1, min_fraction=0.1, seed=0
        )
        assert sorted(eps.query("novel_class == 5").window_idx) == [0, 1]
        assert sorted(eps.query("novel_class == 6").window_idx) == [2, 3]

    def test_not_enough_eligible_windows_raises(self):
        frac = np.array([[0.5], [0.0]])
        with pytest.raises(ValueError, match="only 1 eligible"):
            sample_episodes(frac, [5], shots=[2], n_draws=1, min_fraction=0.1, seed=0)


class TestBuildFewshotEpisodes:
    def test_writes_csv_with_original_columns(self, mask_dir, tmp_path):
        d, csv = mask_dir
        out = tmp_path / "out" / "episodes.csv"
        cfg = OmegaConf.structured(
            FewShotEpisodesConfig(
                window_index_cache=str(csv),
                mask_base_path=str(d),
                novel_classes=[5],
                shots=[1, 2],
                n_draws=3,
                min_novel_fraction=0.25,
                seed=3,
                output_csv=str(out),
            )
        )
        result = build_fewshot_episodes(OmegaConf.create({"fewshot_episodes": cfg}))
        assert result == str(out)
        eps = pd.read_csv(out)
        assert {"mask_path", "row_off", "col_off", "width", "height"} <= set(
            eps.columns
        )
        assert {"shots", "draw", "novel_class", "novel_fraction"} <= set(eps.columns)
        assert len(eps) == 3 * (1 + 2)
        assert set(eps.mask_path) <= {"3.tif", "4.tif", "5.tif", "6.tif"}
        assert (eps.novel_fraction >= 0.25).all()

    def test_missing_output_dir_is_created(self, mask_dir, tmp_path):
        d, csv = mask_dir
        out = tmp_path / "a" / "b" / "e.csv"
        cfg = OmegaConf.create(
            {
                "fewshot_episodes": {
                    "window_index_cache": str(csv),
                    "mask_base_path": str(d),
                    "novel_classes": [5],
                    "shots": [1],
                    "n_draws": 1,
                    "output_csv": str(out),
                }
            }
        )
        build_fewshot_episodes(cfg)
        assert out.exists()


class TestConfig:
    def test_defaults(self):
        cfg = FewShotEpisodesConfig(window_index_cache="a.csv", output_csv="b.csv")
        assert cfg.shots == [1, 3, 5, 10]
        assert cfg.n_draws == 5
        assert cfg.min_novel_fraction == 0.05
        assert cfg.ignore_index == 255
        assert cfg.window_index_mask_path_key == "mask_path"


def test_main_dispatches_build_fewshot_episodes(monkeypatch):
    from pytorch_segmentation_models_trainer import main as main_module
    from pytorch_segmentation_models_trainer.tools.few_shot import episodes

    monkeypatch.setattr(episodes, "build_fewshot_episodes", lambda cfg: "called")
    cfg = OmegaConf.create({"mode": "build-fewshot-episodes"})
    assert main_module.main.__wrapped__(cfg) == "called"


def test_interpolations_to_the_root_config_are_resolved(mask_dir, tmp_path):
    d, csv = mask_dir
    cfg = OmegaConf.create(
        {
            "paths": {"masks": str(d), "out": str(tmp_path)},
            "fewshot_episodes": {
                "window_index_cache": str(csv),
                "mask_base_path": "${paths.masks}",
                "novel_classes": [5],
                "shots": [1],
                "n_draws": 1,
                "output_csv": "${paths.out}/e.csv",
            },
        }
    )
    assert build_fewshot_episodes(cfg) == str(tmp_path / "e.csv")
