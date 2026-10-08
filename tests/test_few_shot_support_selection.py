# -*- coding: utf-8 -*-
"""Tests for diversity/uncertainty-driven support selection."""

import numpy as np
import pytest
import segmentation_models_pytorch as smp
import torch
from torch.utils.data import Dataset

from pytorch_segmentation_models_trainer.few_shot.backbone import (
    FrozenLinearHeadSegmenter,
)
from pytorch_segmentation_models_trainer.tools.few_shot.support_selection import (
    support_diversity,
    tile_descriptors,
    weighted_kcenter,
)


class TestWeightedKCenter:
    def test_farthest_point_order_from_given_start(self):
        emb = np.array([[0.0, 0.0], [10.0, 0.0], [0.1, 0.0], [5.0, 5.0]])
        order = weighted_kcenter(emb, k=3, first=0)
        assert order == [0, 1, 3]

    def test_uncertainty_weight_changes_the_choice(self):
        emb = np.array([[0.0, 0.0], [10.0, 0.0], [6.0, 0.0]])
        u = np.array([1.0, 0.1, 1.0])
        assert weighted_kcenter(emb, 2, first=0)[1] == 1
        assert weighted_kcenter(emb, 2, first=0, uncertainty=u, gamma=2.0)[1] == 2
        # gamma = 0 ignores the uncertainty
        assert weighted_kcenter(emb, 2, first=0, uncertainty=u, gamma=0.0)[1] == 1

    def test_nested_and_k_capped(self):
        emb = np.random.default_rng(0).random((6, 3))
        assert weighted_kcenter(emb, 4, first=2)[:2] == weighted_kcenter(
            emb, 2, first=2
        )
        assert len(weighted_kcenter(emb, 10, first=0)) == 6


class TestDiversity:
    def test_identical_vs_orthogonal(self):
        same = np.ones((3, 4))
        ortho = np.eye(3)
        assert support_diversity(same) == pytest.approx(1.0)
        assert support_diversity(ortho) == pytest.approx(3.0)
        assert support_diversity(ortho[:1]) == pytest.approx(1.0)


class _Images(Dataset):
    def __init__(self, n=3, **kwargs):
        g = torch.Generator().manual_seed(0)
        self.x = torch.randn(n, 3, 64, 64, generator=g)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, i):
        return {"image": self.x[i], "mask": torch.zeros(64, 64, dtype=torch.long)}


class TestTileDescriptors:
    def _seg(self, evidential=False):
        torch.manual_seed(0)
        m = smp.UPerNet(
            encoder_name="resnet18",
            encoder_weights=None,
            classes=3,
            decoder_channels=16,
        )
        if evidential:
            from pytorch_segmentation_models_trainer.custom_models.edl_wrapper import (
                EvidentialWrapper,
            )

            m = EvidentialWrapper(m)
        return FrozenLinearHeadSegmenter(m)

    def test_shapes_pooling_and_uncertainty_range(self):
        seg = self._seg()
        emb, unc = tile_descriptors(seg, _Images(), mothers=[1], batch_size=2)
        assert emb.shape == (3, 16) and emb.dtype == np.float64
        assert set(unc) == {"base_entropy"}
        assert unc["base_entropy"].shape == (3,)
        assert ((unc["base_entropy"] >= 0) & (unc["base_entropy"] <= 1)).all()

    def test_tile_pooling_differs_from_superclass_pooling(self):
        seg = self._seg()
        a, _ = tile_descriptors(seg, _Images(), mothers=[1], pooling="superclass")
        b, _ = tile_descriptors(seg, _Images(), mothers=[1], pooling="tile")
        assert not np.allclose(a, b)

    def test_tile_without_superclass_falls_back_to_tile_mean(self):
        seg = self._seg()
        a, unc = tile_descriptors(
            seg, _Images(1), mothers=[7]
        )  # mother never predicted
        b, _ = tile_descriptors(seg, _Images(1), mothers=[7], pooling="tile")
        np.testing.assert_allclose(a, b)
        assert unc["base_entropy"][0] == 0.0

    def test_evidential_base_adds_vacuity_and_dissonance(self):
        seg = self._seg(evidential=True)
        _, unc = tile_descriptors(seg, _Images(), mothers=[1])
        assert set(unc) == {"base_entropy", "base_vacuity", "base_dissonance"}

    def test_invalid_pooling(self):
        with pytest.raises(ValueError, match="pooling"):
            tile_descriptors(self._seg(), _Images(), mothers=[1], pooling="max")


def _episodes_cfg(tmp_path, mask_dir_csv, method, **sel):
    from omegaconf import OmegaConf

    d, csv = mask_dir_csv
    torch.manual_seed(0)
    model = smp.UPerNet(
        encoder_name="resnet18", encoder_weights=None, classes=6, decoder_channels=16
    )
    ckpt = tmp_path / "base.ckpt"
    torch.save(
        {"state_dict": {f"model.{k}": v for k, v in model.state_dict().items()}}, ckpt
    )
    selection = {
        "method": method,
        "mothers": [3],
        "model": {
            "_target_": "segmentation_models_pytorch.UPerNet",
            "encoder_name": "resnet18",
            "encoder_weights": None,
            "classes": 6,
            "decoder_channels": 16,
        },
        "base_checkpoint": {"path": str(ckpt)},
        "dataset": {
            "_target_": "tests.test_few_shot_support_selection._Images",
            "n": 7,
        },
        "batch_size": 4,
    }
    selection.update(sel)
    return OmegaConf.create(
        {
            "fewshot_episodes": {
                "window_index_cache": str(csv),
                "mask_base_path": str(d),
                "novel_classes": [5],
                "shots": [1, 2],
                "n_draws": 2,
                "min_novel_fraction": 0.15,
                "seed": 3,
                "output_csv": str(tmp_path / "ep.csv"),
                "selection": selection,
            }
        }
    )


class TestEpisodesWithSelection:
    @pytest.fixture
    def pool(self, tmp_path):
        import pandas as pd
        import rasterio
        from rasterio.transform import from_origin

        d = tmp_path / "masks"
        d.mkdir()
        rows = []
        for i in range(7):
            m = np.full((10, 10), 3, dtype=np.uint8)
            m.flat[: (i % 6) * 10] = 5
            with rasterio.open(
                d / f"{i}.tif",
                "w",
                driver="GTiff",
                height=10,
                width=10,
                count=1,
                dtype="uint8",
                transform=from_origin(0, 0, 1, 1),
            ) as dst:
                dst.write(m, 1)
            rows.append(
                {
                    "mask_path": f"{i}.tif",
                    "row_off": 0,
                    "col_off": 0,
                    "width": 10,
                    "height": 10,
                }
            )
        csv = tmp_path / "pool.csv"
        pd.DataFrame(rows).to_csv(csv, index=False)
        return d, csv

    def _run(self, tmp_path, pool, method, **sel):
        import pandas as pd

        from pytorch_segmentation_models_trainer.tools.few_shot.episodes import (
            build_fewshot_episodes,
        )

        cfg = _episodes_cfg(tmp_path, pool, method, **sel)
        return pd.read_csv(build_fewshot_episodes(cfg))

    def test_random_selection_adds_diversity_column(self, tmp_path, pool):
        eps = self._run(tmp_path, pool, "random")
        assert "diversity" in eps.columns and "selection" in eps.columns
        assert (eps.selection == "random").all()
        k1 = eps[eps.shots == 1]
        assert np.allclose(k1.diversity, 1.0)

    def test_kcenter_is_nested_and_draws_start_differently(self, tmp_path, pool):
        eps = self._run(tmp_path, pool, "kcenter")
        assert (eps.selection == "kcenter").all()
        for d in (0, 1):
            k1 = eps.query("draw == @d and shots == 1").mask_path.tolist()
            k2 = eps.query("draw == @d and shots == 2").mask_path.tolist()
            assert k1 == k2[:1]
        assert set(eps.mask_path) <= {f"{i}.tif" for i in (2, 3, 4, 5)}  # eligible only

    def test_uncertainty_weighted_kcenter(self, tmp_path, pool):
        eps = self._run(
            tmp_path, pool, "kcenter", uncertainty="base_entropy", gamma=2.0
        )
        assert (eps.selection == "kcenter+base_entropy").all()
        assert "uncertainty" in eps.columns

    def test_unknown_uncertainty_raises(self, tmp_path, pool):
        with pytest.raises(KeyError, match="base_vacuity"):
            self._run(tmp_path, pool, "kcenter", uncertainty="base_vacuity")

    def test_dataset_length_must_match_index(self, tmp_path, pool):
        cfg_kw = {
            "dataset": {
                "_target_": "tests.test_few_shot_support_selection._Images",
                "n": 3,
            }
        }
        with pytest.raises(ValueError, match="windows"):
            self._run(tmp_path, pool, "random", **cfg_kw)

    def test_invalid_method(self, tmp_path, pool):
        with pytest.raises(ValueError, match="selection.method"):
            self._run(tmp_path, pool, "coreset")


def test_kcenter_episodes_needs_enough_eligible():
    from pytorch_segmentation_models_trainer.tools.few_shot.episodes import (
        kcenter_episodes,
    )

    with pytest.raises(ValueError, match="only 1 eligible"):
        kcenter_episodes(
            np.array([[0.5], [0.0]]), [5], [2], 1, 0.1, 0, embeddings=np.eye(2)
        )


def test_selection_from_runner_and_gpu_path(tmp_path, monkeypatch):
    import json

    import pandas as pd

    from pytorch_segmentation_models_trainer.few_shot.backbone import (
        FrozenLinearHeadSegmenter,
    )
    from pytorch_segmentation_models_trainer.tools.few_shot.episodes import (
        build_fewshot_episodes,
    )

    pool = TestEpisodesWithSelection.pool.__wrapped__(None, tmp_path)
    cfg = _episodes_cfg(tmp_path, pool, "kcenter")
    runner = tmp_path / "runner"
    runner.mkdir()
    (runner / "runner_state.json").write_text(
        json.dumps(
            {
                "completed_runs": [
                    {
                        "run_idx": 0,
                        "seed": 7,
                        "best_checkpoint_path": str(tmp_path / "base.ckpt"),
                    }
                ]
            }
        )
    )
    cfg.fewshot_episodes.selection.base_checkpoint = {
        "from_runner": str(runner),
        "seed": 7,
    }
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(FrozenLinearHeadSegmenter, "cuda", lambda self: self)
    eps = pd.read_csv(build_fewshot_episodes(cfg))
    assert (eps.selection == "kcenter").all()
