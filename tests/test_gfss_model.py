# -*- coding: utf-8 -*-
"""Tests for GFSSModel (LightningModule of generalized few-shot segmentation)."""

import json
from unittest.mock import MagicMock

import pytest
import segmentation_models_pytorch as smp
import torch
from omegaconf import OmegaConf
from torch.utils.data import Dataset

from pytorch_segmentation_models_trainer.few_shot.base_method import BaseGFSSMethod
from pytorch_segmentation_models_trainer.few_shot.methods.base_only import BaseOnly
from pytorch_segmentation_models_trainer.model_loader.gfss_model import GFSSModel

_MODEL = {
    "_target_": "segmentation_models_pytorch.UPerNet",
    "encoder_name": "resnet18",
    "encoder_weights": None,
    "classes": 3,
    "decoder_channels": 16,
}


class TinySegDataset(Dataset):
    """4 images 64x64; mask: left half class 0, right half ``right`` class,
    top-right quadrant class ``corner``."""

    def __init__(self, n=4, right=1, corner=2, seed=None, **kwargs):
        g = torch.Generator().manual_seed(0)
        self.images = torch.randn(n, 3, 64, 64, generator=g)
        mask = torch.zeros(64, 64, dtype=torch.long)
        mask[:, 32:] = right
        mask[:32, 32:] = corner
        self.mask = mask

    def __len__(self):
        return len(self.images)

    def __getitem__(self, i):
        return {"image": self.images[i], "mask": self.mask.clone()}


class TrainableProbe(BaseGFSSMethod):
    """Minimal trainable method used to exercise the training path."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.scale = torch.nn.Parameter(torch.ones(1))
        self.init_calls = 0
        self.seen_masks = None

    def init_from_support(self, features, masks):
        self.init_calls += 1
        self.seen_masks = masks

    def support_loss(self, features, masks):
        return (self.scale * features.mean()) ** 2

    def forward(self, features):
        base = torch.einsum("bfhw,cf->bchw", features, self.base_weight)
        base = base + self.base_bias.view(1, -1, 1, 1)
        novel = base[:, [2]] * self.scale
        return torch.cat([base, novel], dim=1)


class TransductiveProbe(TrainableProbe):
    transductive = True

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.adapt_calls = 0

    def adapt_to_query(self, support_features, support_masks, query_features):
        assert torch.is_grad_enabled()
        self.adapt_calls += 1


def _ds(**kw):
    node = {
        "_target_": "tests.test_gfss_model.TinySegDataset",
        "data_loader": {"num_workers": 0, "shuffle": False, "drop_last": False},
    }
    node.update(kw)
    return node


@pytest.fixture(scope="module")
def base_ckpt(tmp_path_factory):
    # One checkpoint for the whole module (~45 MB each; read-only in the tests).
    torch.manual_seed(0)
    model = smp.UPerNet(**{k: v for k, v in _MODEL.items() if k != "_target_"})
    path = tmp_path_factory.mktemp("base") / "base.ckpt"
    torch.save(
        {"state_dict": {f"model.{k}": v for k, v in model.state_dict().items()}}, path
    )
    return path, model


def _cfg(
    ckpt_path,
    method="pytorch_segmentation_models_trainer.few_shot.methods.base_only.BaseOnly",
    **extra,
):
    cfg = {
        "model": dict(_MODEL),
        "gfss": {
            "hierarchy": {2: [2, 3]},
            "base_checkpoint": {"path": str(ckpt_path)},
            "method": {"_target_": method},
        },
        "hyperparameters": {"batch_size": 2},
        "optimizer": {"_target_": "torch.optim.SGD", "lr": 0.1},
        "train_dataset": _ds(right=1, corner=3),
        "test_dataset": _ds(right=1, corner=3),
    }
    cfg.update(extra)
    return OmegaConf.create(cfg)


class TestConstruction:
    def test_builds_frozen_base_and_method(self, base_ckpt):
        path, base = base_ckpt
        m = GFSSModel(_cfg(path))
        assert isinstance(m.method, BaseOnly)
        assert m.hierarchy.num_classes == 4
        assert not any(p.requires_grad for p in m.model.parameters())
        torch.testing.assert_close(
            m.method.base_weight, base.segmentation_head[0].weight.detach().flatten(1)
        )
        assert m.loss_function is None

    def test_forward_shape_and_floor_matches_base(self, base_ckpt):
        path, base = base_ckpt
        m = GFSSModel(_cfg(path)).eval()
        x = torch.randn(2, 3, 64, 64)
        logits = m(x)
        assert logits.shape == (2, 4, 64, 64)
        torch.testing.assert_close(logits[:, :3], base.eval()(x))

    def test_checkpoint_from_runner_uses_seed(self, base_ckpt, tmp_path):
        path, _ = base_ckpt
        runner = tmp_path / "runner"
        runner.mkdir()
        (runner / "runner_state.json").write_text(
            json.dumps(
                {
                    "completed_runs": [
                        {"run_idx": 0, "seed": 7, "best_checkpoint_path": str(path)}
                    ]
                }
            )
        )
        cfg = _cfg(path, seed=7)
        cfg.gfss.base_checkpoint = {"from_runner": str(runner)}
        assert GFSSModel(cfg).hierarchy.num_base_classes == 3

    def test_checkpoint_path_interpolated_from_root(self, base_ckpt):
        path, _ = base_ckpt
        cfg = _cfg(path, paths={"ckpt": str(path)})
        cfg.gfss.base_checkpoint = {"path": "${paths.ckpt}"}
        assert GFSSModel(cfg).hierarchy.num_base_classes == 3

    def test_from_runner_without_seed_raises(self, base_ckpt, tmp_path):
        path, _ = base_ckpt
        cfg = _cfg(path)
        cfg.gfss.base_checkpoint = {"from_runner": str(tmp_path)}
        with pytest.raises(ValueError, match="seed"):
            GFSSModel(cfg)

    @pytest.mark.parametrize("ckpt", [{}, {"path": "a", "from_runner": "b"}])
    def test_exactly_one_checkpoint_source(self, base_ckpt, ckpt):
        path, _ = base_ckpt
        cfg = _cfg(path)
        cfg.gfss.base_checkpoint = ckpt
        with pytest.raises(ValueError, match="exactly one"):
            GFSSModel(cfg)

    def test_class_names_from_class_definitions(self, base_ckpt):
        path, _ = base_ckpt
        cfg = _cfg(path, class_definitions={"names": ["a", "b", "c", "d"]})
        assert GFSSModel(cfg).test_gfss_metrics.class_names == ["a", "b", "c", "d"]

    def test_class_names_from_gfss_node_win(self, base_ckpt):
        path, _ = base_ckpt
        cfg = _cfg(path, class_definitions={"names": ["a", "b", "c", "d"]})
        cfg.gfss.class_names = ["w", "x", "y", "z"]
        assert GFSSModel(cfg).test_gfss_metrics.class_names == ["w", "x", "y", "z"]


class TestOptimizers:
    def test_no_trainable_parameters_returns_none(self, base_ckpt):
        m = GFSSModel(_cfg(base_ckpt[0]))
        assert m.configure_optimizers() is None

    def test_only_method_parameters_are_optimized(self, base_ckpt):
        m = GFSSModel(_cfg(base_ckpt[0], method="tests.test_gfss_model.TrainableProbe"))
        opt = m.configure_optimizers()[0][0]
        params = [p for g in opt.param_groups for p in g["params"]]
        assert len(params) == 1 and params[0] is m.method.scale


class TestSteps:
    def test_support_masks_downsampled_and_init_once(self, base_ckpt):
        m = GFSSModel(_cfg(base_ckpt[0], method="tests.test_gfss_model.TrainableProbe"))
        m.on_fit_start()
        m.on_fit_start()
        assert m.method.init_calls == 1
        assert m.method.seen_masks.shape == (4, 16, 16)
        assert set(m.method.seen_masks.unique().tolist()) == {0, 1, 3}

    def test_training_step_returns_loss_and_logs(self, base_ckpt):
        m = GFSSModel(_cfg(base_ckpt[0], method="tests.test_gfss_model.TrainableProbe"))
        m.log = MagicMock()
        batch = next(iter(m.train_dataloader()))
        loss = m.training_step(batch, 0)
        assert loss.requires_grad
        loss.backward()
        assert m.method.scale.grad is not None
        m.log.assert_called()

    def test_training_step_without_loss_returns_none(self, base_ckpt):
        m = GFSSModel(_cfg(base_ckpt[0]))
        assert m.training_step(next(iter(m.train_dataloader())), 0) is None

    def test_test_step_accumulates_and_epoch_end_logs(self, base_ckpt):
        m = GFSSModel(_cfg(base_ckpt[0]))
        m.log_dict = MagicMock()
        for i, batch in enumerate(m.test_dataloader()):
            m.test_step(batch, i)
        assert m.test_gfss_metrics.confmat.sum() == 4 * 64 * 64
        m.on_test_epoch_end()
        logged = m.log_dict.call_args[0][0]
        assert logged["test/iou/3"] == 0.0
        assert "test/split_ceiling/3" in logged and "test/locality" in logged
        assert m.test_gfss_metrics.confmat.sum() == 0

    def test_validation_uses_val_metrics(self, base_ckpt):
        cfg = _cfg(base_ckpt[0], val_dataset=_ds(right=1, corner=3))
        m = GFSSModel(cfg)
        m.log_dict = MagicMock()
        m.validation_step(next(iter(m.val_dataloader())), 0)
        m.on_validation_epoch_end()
        assert "val/miou" in m.log_dict.call_args[0][0]

    def test_boundary_width_adds_boundary_metrics(self, base_ckpt):
        cfg = _cfg(base_ckpt[0])
        cfg.gfss.boundary_width = 2
        m = GFSSModel(cfg)
        assert m.test_gfss_metrics.boundary_width == 2
        assert m.val_gfss_metrics.boundary_width == 2
        m.log_dict = MagicMock()
        for i, batch in enumerate(m.test_dataloader()):
            m.test_step(batch, i)
        m.on_test_epoch_end()
        logged = m.log_dict.call_args[0][0]
        assert "test/boundary/miou_novel" in logged
        assert "test/boundary/iou/3" in logged

    def test_transductive_method_adapts_per_batch_with_grad(self, base_ckpt):
        m = GFSSModel(
            _cfg(base_ckpt[0], method="tests.test_gfss_model.TransductiveProbe")
        )
        with torch.no_grad():
            for i, batch in enumerate(m.test_dataloader()):
                m.test_step(batch, i)
        assert m.method.adapt_calls == 2
        assert m.method.init_calls == 1


class TestTrainIntegration:
    @pytest.mark.parametrize(
        "method, steps, extra",
        [
            ("tests.test_gfss_model.TrainableProbe", 3, {}),
            (
                "pytorch_segmentation_models_trainer.few_shot.methods.base_only.BaseOnly",
                0,
                {},
            ),
            ("tests.test_gfss_model.TransductiveProbe", 0, {"inference_mode": False}),
            (
                "pytorch_segmentation_models_trainer.few_shot.methods.prototype.PrototypeImprinting",
                0,
                {},
            ),
            (
                "pytorch_segmentation_models_trainer.few_shot.methods.diam.DIaM",
                0,
                {"inference_mode": False},
            ),
            (
                "pytorch_segmentation_models_trainer.few_shot.methods.classtrans.ClassTrans",
                0,
                {"inference_mode": False},
            ),
            ("pytorch_segmentation_models_trainer.few_shot.methods.bcm.BCM", 0, {}),
        ],
    )
    def test_train_entrypoint_runs_fit_and_test(
        self, base_ckpt, tmp_path, method, steps, extra
    ):
        from pytorch_segmentation_models_trainer.train import train

        cfg = _cfg(
            base_ckpt[0],
            method=method,
            pl_model={
                "_target_": "pytorch_segmentation_models_trainer.model_loader.gfss_model.GFSSModel"
            },
            pl_trainer={
                "max_steps": steps,
                "accelerator": "cpu",
                "enable_checkpointing": False,
                "enable_progress_bar": False,
                "default_root_dir": str(tmp_path),
                **extra,
            },
        )
        trainer = train(cfg)
        assert trainer.global_step == steps
        assert "test/miou" in trainer.callback_metrics
        assert "test/oem_score" in trainer.callback_metrics


def test_example_configs_match_dataclasses():
    from pathlib import Path

    from pytorch_segmentation_models_trainer.config_definitions.experiments_runner_config import (
        ExperimentsRunnerConfig,
    )
    from pytorch_segmentation_models_trainer.config_definitions.few_shot_config import (
        FewShotEpisodesConfig,
        GFSSConfig,
    )
    from pytorch_segmentation_models_trainer.few_shot.hierarchy import ClassHierarchy

    root = Path(__file__).resolve().parents[1] / "conf" / "examples"
    gfss = OmegaConf.load(root / "gfss_base_only.yaml")
    merged = OmegaConf.merge(OmegaConf.structured(GFSSConfig), gfss.gfss)
    ClassHierarchy(merged.hierarchy, num_base_classes=gfss.model.classes)
    OmegaConf.merge(
        OmegaConf.structured(ExperimentsRunnerConfig), gfss.experiments_runner
    )
    episodes = OmegaConf.load(root / "build_fewshot_episodes.yaml")
    OmegaConf.merge(
        OmegaConf.structured(FewShotEpisodesConfig), episodes.fewshot_episodes
    )


@pytest.mark.parametrize(
    "name, target",
    [
        ("gfss_prototype", "PrototypeImprinting"),
        ("gfss_diam", "DIaM"),
        ("gfss_classtrans", "ClassTrans"),
        ("gfss_hisplit", "HiSplit"),
        ("gfss_finetune", "FineTune"),
        ("gfss_bcm", "BCM"),
    ],
)
def test_method_example_configs_compose_and_instantiate(name, target):
    from pathlib import Path

    from hydra import compose, initialize_config_dir
    from hydra.utils import instantiate

    root = Path(__file__).resolve().parents[1] / "conf" / "examples"
    with initialize_config_dir(config_dir=str(root), version_base="1.2"):
        cfg = compose(config_name=name)
    method = instantiate(cfg.gfss.method, _recursive_=False)
    assert type(method).__name__ == target
    assert cfg.gfss.hierarchy == {3: [3, 5]}
    assert cfg.experiments_runner.episodes.csv == "outputs/episodes/pampa.csv"


@pytest.mark.parametrize(
    "q, steps, extra",
    [("proto", 0, {}), ("linear", 3, {}), ("trans", 2, {"inference_mode": False})],
)
def test_hisplit_runs_through_train(base_ckpt, tmp_path, q, steps, extra):
    from pytorch_segmentation_models_trainer.train import train

    cfg = _cfg(
        base_ckpt[0],
        method="pytorch_segmentation_models_trainer.few_shot.methods.hisplit.HiSplit",
        pl_model={
            "_target_": "pytorch_segmentation_models_trainer.model_loader.gfss_model.GFSSModel"
        },
        pl_trainer={
            "max_steps": steps,
            "accelerator": "cpu",
            "enable_checkpointing": False,
            "enable_progress_bar": False,
            "default_root_dir": str(tmp_path),
            **extra,
        },
    )
    cfg.gfss.method.q = q
    cfg.gfss.method.adapt_iter = 2
    trainer = train(cfg)
    assert trainer.global_step == steps
    assert "test/locality" in trainer.callback_metrics


class TestUncertaintyEvaluation:
    def test_methods_without_uncertainty_have_no_unc_metrics(self, base_ckpt):
        m = GFSSModel(_cfg(base_ckpt[0]))
        assert m.test_unc_metrics is None and m.val_unc_metrics is None

    def test_hisplit_edl_logs_uncertainty_metrics_through_train(
        self, base_ckpt, tmp_path
    ):
        from pytorch_segmentation_models_trainer.train import train

        cfg = _cfg(
            base_ckpt[0],
            method="pytorch_segmentation_models_trainer.few_shot.methods.hisplit.HiSplit",
            pl_model={
                "_target_": "pytorch_segmentation_models_trainer.model_loader.gfss_model.GFSSModel"
            },
            pl_trainer={
                "max_steps": 3,
                "accelerator": "cpu",
                "enable_checkpointing": False,
                "enable_progress_bar": False,
                "default_root_dir": str(tmp_path),
            },
        )
        cfg.gfss.method.q = "edl"
        trainer = train(cfg)
        keys = set(trainer.callback_metrics)
        for name in ("split_entropy", "split_vacuity"):
            assert f"test/unc/aurc/{name}" in keys
            assert f"test/unc/tile_spearman/{name}" in keys
            assert f"test/unc/tile_mean/{name}" in keys

    def test_refinement_aware_base_checkpoint(self, tmp_path):
        from pytorch_segmentation_models_trainer.custom_models.refinement_aware import (
            RefinementAwareWrapper,
        )

        torch.manual_seed(0)
        inner = smp.UPerNet(**{k: v for k, v in _MODEL.items() if k != "_target_"})
        wrapper = RefinementAwareWrapper(inner, superclass_index=2)
        path = tmp_path / "r2c.ckpt"
        torch.save(
            {"state_dict": {f"model.{k}": v for k, v in wrapper.state_dict().items()}},
            path,
        )
        cfg = _cfg(path)
        cfg.model = {
            "_target_": "pytorch_segmentation_models_trainer.custom_models.refinement_aware.RefinementAwareWrapper",
            "superclass_index": 2,
            "model": dict(_MODEL),
        }
        m = GFSSModel(cfg)
        assert not m.model.evidential
        torch.testing.assert_close(
            m.method.base_weight, inner.segmentation_head[0].weight.flatten(1)
        )

    def test_evidential_base_checkpoint(self, tmp_path):
        from pytorch_segmentation_models_trainer.custom_models.edl_wrapper import (
            EvidentialWrapper,
        )

        torch.manual_seed(0)
        inner = smp.UPerNet(**{k: v for k, v in _MODEL.items() if k != "_target_"})
        wrapper = EvidentialWrapper(inner)
        path = tmp_path / "edl.ckpt"
        torch.save(
            {"state_dict": {f"model.{k}": v for k, v in wrapper.state_dict().items()}},
            path,
        )
        cfg = _cfg(
            path,
            method="pytorch_segmentation_models_trainer.few_shot.methods.hisplit.HiSplit",
        )
        cfg.model = {
            "_target_": "pytorch_segmentation_models_trainer.custom_models.edl_wrapper.EvidentialWrapper",
            "model": dict(_MODEL),
        }
        m = GFSSModel(cfg)
        assert m.model.evidential and m.method.base_output == "evidential"
        assert m.test_unc_metrics.names == [
            "split_entropy",
            "base_vacuity",
            "dissonance",
        ]
        m.log_dict = MagicMock()
        for i, batch in enumerate(m.test_dataloader()):
            m.test_step(batch, i)
        m.on_test_epoch_end()
        logged = {}
        for call in m.log_dict.call_args_list:
            logged.update(call[0][0])
        assert "test/unc/aurc/dissonance" in logged and "test/miou" in logged


def test_decoding_variants_are_evaluated_and_logged(base_ckpt, tmp_path):
    from pytorch_segmentation_models_trainer.train import train

    cfg = _cfg(
        base_ckpt[0],
        method="pytorch_segmentation_models_trainer.few_shot.methods.hisplit.HiSplit",
        pl_model={
            "_target_": "pytorch_segmentation_models_trainer.model_loader.gfss_model.GFSSModel"
        },
        pl_trainer={
            "max_steps": 2,
            "accelerator": "cpu",
            "enable_checkpointing": False,
            "enable_progress_bar": False,
            "default_root_dir": str(tmp_path),
        },
    )
    cfg.gfss.method.widen = "prob"
    cfg.gfss.method.sweep = [0.1, 0.5]
    cfg.gfss.method.leak = True
    trainer = train(cfg)
    keys = set(trainer.callback_metrics)
    assert {"test/var/widen_0.1/miou", "test/var/widen_0.5/locality"} <= keys
    assert trainer.global_step == 2  # leak parameters trained


def test_no_variant_metrics_by_default(base_ckpt):
    m = GFSSModel(_cfg(base_ckpt[0]))
    assert len(m.variant_metrics) == 0


def test_uncertainty_evaluation_options_from_config(base_ckpt):
    cfg = _cfg(
        base_ckpt[0],
        method="pytorch_segmentation_models_trainer.few_shot.methods.hisplit.HiSplit",
    )
    cfg.gfss.uncertainty_eval = {
        "abstain_thresholds": [0.3],
        "ece_bins": 5,
        "n_bins": 100,
    }
    m = GFSSModel(cfg)
    assert m.test_unc_metrics.abstain_thresholds == [0.3]
    assert m.test_unc_metrics.ece_bins == 5 and m.val_unc_metrics.n_bins == 100


class TestTrainableBackbone:
    def _cfg(self, base_ckpt, tmp_path, trainable, kd):
        cfg = _cfg(
            base_ckpt[0],
            method="pytorch_segmentation_models_trainer.few_shot.methods.finetune.FineTune",
            pl_model={
                "_target_": "pytorch_segmentation_models_trainer.model_loader.gfss_model.GFSSModel"
            },
            pl_trainer={
                "max_steps": 2,
                "accelerator": "cpu",
                "enable_checkpointing": False,
                "enable_progress_bar": False,
                "default_root_dir": str(tmp_path),
            },
        )
        cfg.gfss.backbone = {"trainable": trainable}
        cfg.gfss.method.train_rows = "all"
        cfg.gfss.method.kd_weight = kd
        return cfg

    def test_frozen_reference_and_optimizer_params(self, base_ckpt, tmp_path):
        m = GFSSModel(self._cfg(base_ckpt, tmp_path, "decoder", 1.0))
        assert m.base_reference is not None
        assert not any(p.requires_grad for p in m.base_reference.parameters())
        opt = m.configure_optimizers()[0][0]
        n_opt = sum(p.numel() for g in opt.param_groups for p in g["params"])
        n_dec = sum(p.numel() for p in m.model.model.decoder.parameters())
        assert n_opt > n_dec
        assert GFSSModel(_cfg(base_ckpt[0])).base_reference is None

    def test_full_finetune_with_kd_through_train(self, base_ckpt, tmp_path):
        from pytorch_segmentation_models_trainer.train import train

        cfg = self._cfg(base_ckpt, tmp_path, "all", 1.0)
        before = torch.load(base_ckpt[0], weights_only=False)["state_dict"]
        trainer = train(cfg)
        assert trainer.global_step == 2
        assert "test/locality" in trainer.callback_metrics
        model = trainer.lightning_module
        tuned = model.model.model.state_dict()
        ref = model.base_reference.model.state_dict()
        keys = [k for k in tuned if k.startswith("decoder.") and k.endswith("weight")]
        assert keys
        # the backbone was fine-tuned, the reference stayed equal to the checkpoint
        assert any(not torch.equal(tuned[k], before[f"model.{k}"]) for k in keys)
        assert all(torch.equal(ref[k], before[f"model.{k}"]) for k in keys)

    def test_invalid_trainable_value(self, base_ckpt, tmp_path):
        with pytest.raises(ValueError, match="trainable"):
            GFSSModel(self._cfg(base_ckpt, tmp_path, "head", 0.0))


class TestTTA:
    def test_base_only_with_d8_reproduces_base_model_tta(self, base_ckpt):
        from pytorch_segmentation_models_trainer.tools.tta.tta import apply_tta

        cfg = _cfg(base_ckpt[0], tta_mode="d8")
        m = GFSSModel(cfg).eval()
        batch = next(iter(m.test_dataloader()))
        m.test_step(batch, 0)
        images, masks = batch["image"], batch["mask"]
        augs = m._get_tta_augmentations()
        assert len(augs) == 8
        expected = apply_tta(m.model.base_logits, images, augs).argmax(1)
        cm = m.test_gfss_metrics.confmat
        # BaseOnly never predicts the novel class and agrees with the base under TTA
        assert cm[:, 3].sum() == 0
        out = m.test_gfss_metrics.compute()
        assert out["test/locality"] == pytest.approx(1.0)
        m.test_gfss_metrics.reset()
        m.test_gfss_metrics.update(expected, masks.long(), expected)
        torch.testing.assert_close(cm, m.test_gfss_metrics.confmat)

    def test_tta_changes_predictions_and_is_only_used_in_test(self, base_ckpt):
        cfg = _cfg(
            base_ckpt[0],
            method="pytorch_segmentation_models_trainer.few_shot.methods.prototype.PrototypeImprinting",
            tta_mode="d8",
            val_dataset=_ds(right=1, corner=3),
        )
        m = GFSSModel(cfg).eval()
        calls = []
        orig = m._predict_logits

        def spy(images, augmentations):
            calls.append(augmentations)
            return orig(images, augmentations)

        m._predict_logits = spy
        batch = next(iter(m.test_dataloader()))
        m.test_step(batch, 0)
        m.validation_step(batch, 0)
        assert calls[0] is not None and len(calls[0]) == 8
        assert calls[1] is None

    def test_transductive_and_variants_under_tta(self, base_ckpt, tmp_path):
        from pytorch_segmentation_models_trainer.train import train

        cfg = _cfg(
            base_ckpt[0],
            method="pytorch_segmentation_models_trainer.few_shot.methods.hisplit.HiSplit",
            tta_mode="flip",
            pl_model={
                "_target_": "pytorch_segmentation_models_trainer.model_loader.gfss_model.GFSSModel"
            },
            pl_trainer={
                "max_steps": 0,
                "accelerator": "cpu",
                "enable_checkpointing": False,
                "enable_progress_bar": False,
                "inference_mode": False,
                "default_root_dir": str(tmp_path),
            },
        )
        cfg.gfss.method.q = "trans"
        cfg.gfss.method.adapt_iter = 2
        cfg.gfss.method.widen = "prob"
        cfg.gfss.method.sweep = [0.2]
        trainer = train(cfg)
        assert "test/var/widen_0.2/miou" in trainer.callback_metrics
        assert "test/unc/aurc/split_entropy" in trainer.callback_metrics
