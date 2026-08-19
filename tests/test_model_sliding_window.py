# -*- coding: utf-8 -*-
"""
Tests for sliding-window test-phase methods added to Model:
  - test_step routing (SW vs. original)
  - _test_step_sliding_window
  - on_test_epoch_end
  - _build_sw_core
  - _save_test_prediction
  - SlidingWindowTestConfig dataclass
"""

import tempfile
from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn as nn
from omegaconf import OmegaConf

from pytorch_segmentation_models_trainer.model_loader.model import Model
from pytorch_segmentation_models_trainer.tools.inference.sliding_window import (
    SlidingWindowCore,
    SlidingWindowOutput,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

B, C, H, W = 2, 2, 32, 32


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _sw_cfg_dict(**extra):
    return {
        "use_sliding_window_test": True,
        "sliding_window_test": {
            "tile_size": [H, W],
            "tile_step": [H // 2, W // 2],
            "num_classes": C,
            "batch_size": 2,
            "tile_weight": "mean",
        },
        **extra,
    }


def _make_model(cfg_overrides=None):
    base_cfg = OmegaConf.create(
        {
            "model": {
                "_target_": "segmentation_models_pytorch.Unet",
                "encoder_name": "resnet18",
                "encoder_weights": None,
                "in_channels": 3,
                "classes": C,
            },
            "loss": {"_target_": "torch.nn.CrossEntropyLoss"},
            "hyperparameters": {"batch_size": B, "devices": 1, "accelerator": "cpu"},
        }
    )
    if cfg_overrides:
        cfg = OmegaConf.merge(base_cfg, OmegaConf.create(cfg_overrides))
    else:
        cfg = base_cfg

    mock_inner = MagicMock()
    with patch.object(Model, "get_model", return_value=mock_inner):
        model = Model(cfg, inference_mode=True)

    model.loss_function = nn.CrossEntropyLoss()
    model.use_compound_loss = False
    model.should_normalize = False
    model.gpu_test_transform = None
    return model


def _make_batch(with_image_path=False):
    batch = {
        "image": torch.randn(B, 3, H, W),
        "mask": torch.zeros(B, H, W, dtype=torch.long),
    }
    if with_image_path:
        batch["image_path"] = ["/fake/img1.tif", "/fake/img2.tif"]
    return batch


def _make_sw_output(tta_unc=False, mc_unc=False):
    return SlidingWindowOutput(
        prediction=torch.randn(1, C, H, W),
        tta_uncertainty=torch.zeros(1, 1, H, W) if tta_unc else None,
        mc_uncertainty=torch.zeros(1, 1, H, W) if mc_unc else None,
    )


def _mock_sw_core(**kwargs):
    mock = MagicMock(spec=SlidingWindowCore)
    mock.predict.return_value = _make_sw_output(**kwargs)
    return mock


# ===========================================================================
# test_step routing
# ===========================================================================


class TestTestStepRouting:
    def test_sw_disabled_uses_original_path(self):
        """test_step must NOT call _test_step_sliding_window when flag is off."""
        model = _make_model()
        model.log = MagicMock()
        model.model.return_value = torch.randn(B, C, H, W)
        with patch.object(model, "_test_step_sliding_window") as mock_sw:
            model.test_step(_make_batch(), 0)
        mock_sw.assert_not_called()

    def test_sw_enabled_delegates_to_sw_step(self):
        """test_step must delegate entirely to _test_step_sliding_window."""
        model = _make_model(_sw_cfg_dict())
        model.log = MagicMock()
        batch = _make_batch()
        sentinel = torch.tensor(0.42)
        with patch.object(
            model, "_test_step_sliding_window", return_value=sentinel
        ) as mock_sw:
            result = model.test_step(batch, 7)
        mock_sw.assert_called_once_with(batch, 7)
        assert result is sentinel


# ===========================================================================
# _test_step_sliding_window
# ===========================================================================


class TestTestStepSlidingWindow:
    def _run(self, model, batch=None, core=None):
        model.log = MagicMock()
        if core is None:
            core = _mock_sw_core()
        with patch.object(model, "_build_sw_core", return_value=core):
            return model._test_step_sliding_window(batch or _make_batch(), 0)

    def test_returns_tensor(self):
        model = _make_model(_sw_cfg_dict())
        result = self._run(model)
        assert isinstance(result, torch.Tensor)

    def test_returns_scalar(self):
        model = _make_model(_sw_cfg_dict())
        result = self._run(model)
        assert result.ndim == 0

    def test_logs_loss_test_key(self):
        model = _make_model(_sw_cfg_dict())
        self._run(model)
        logged_keys = [c.args[0] for c in model.log.call_args_list]
        assert "loss/test" in logged_keys

    def test_calls_predict_once_per_image(self):
        model = _make_model(_sw_cfg_dict())
        core = _mock_sw_core()
        self._run(model, core=core)
        assert core.predict.call_count == B

    def test_metrics_update_called_not_compute(self):
        """SW mode: must call metric.update() per image, never metric()."""
        model = _make_model(_sw_cfg_dict())
        mock_metrics = MagicMock()
        model.test_metrics = mock_metrics
        self._run(model)
        assert mock_metrics.update.call_count == B
        mock_metrics.assert_not_called()  # __call__ = update+compute must NOT run

    def test_save_called_when_image_path_present(self):
        model = _make_model(_sw_cfg_dict())
        batch = _make_batch(with_image_path=True)
        with (
            patch.object(model, "_save_test_prediction") as mock_save,
            patch.object(model, "_build_sw_core", return_value=_mock_sw_core()),
        ):
            model.log = MagicMock()
            model._test_step_sliding_window(batch, 0)
        assert mock_save.call_count == B

    def test_save_not_called_when_no_image_path(self):
        model = _make_model(_sw_cfg_dict())
        with (
            patch.object(model, "_save_test_prediction") as mock_save,
            patch.object(model, "_build_sw_core", return_value=_mock_sw_core()),
        ):
            model.log = MagicMock()
            model._test_step_sliding_window(_make_batch(), 0)
        mock_save.assert_not_called()

    def test_gpu_test_transform_applied(self):
        model = _make_model(_sw_cfg_dict())
        transform = MagicMock(side_effect=lambda x: x)
        model.gpu_test_transform = transform
        self._run(model)
        assert transform.called

    def test_loss_is_mean_over_batch(self):
        """The returned loss must equal the mean of per-image losses."""
        model = _make_model(_sw_cfg_dict())
        fixed_loss = torch.tensor(1.0)
        with (
            patch.object(model, "_compute_loss", return_value=(fixed_loss, {}, {})),
            patch.object(model, "_build_sw_core", return_value=_mock_sw_core()),
        ):
            model.log = MagicMock()
            result = model._test_step_sliding_window(_make_batch(), 0)
        assert result.item() == pytest.approx(fixed_loss.item())


# ===========================================================================
# on_test_epoch_end
# ===========================================================================


class TestOnTestEpochEnd:
    def test_noop_when_sw_disabled(self):
        model = _make_model()  # no use_sliding_window_test
        mock_metrics = MagicMock()
        model.test_metrics = mock_metrics
        model.on_test_epoch_end()
        mock_metrics.compute.assert_not_called()

    def test_noop_when_no_test_metrics(self):
        model = _make_model(_sw_cfg_dict())
        model.log_dict = MagicMock()
        model.on_test_epoch_end()
        model.log_dict.assert_not_called()

    def test_compute_then_log_then_reset(self):
        model = _make_model(_sw_cfg_dict())
        mock_metrics = MagicMock()
        mock_metrics.compute.return_value = {"f1/test": torch.tensor(0.9)}
        model.test_metrics = mock_metrics
        model.log_dict = MagicMock()
        model.on_test_epoch_end()
        mock_metrics.compute.assert_called_once()
        model.log_dict.assert_called_once()
        mock_metrics.reset.assert_called_once()

    def test_compute_before_reset(self):
        """compute() must be called before reset() (order matters for metrics)."""
        call_order = []
        model = _make_model(_sw_cfg_dict())
        mock_metrics = MagicMock()
        mock_metrics.compute.side_effect = lambda: call_order.append("compute") or {}
        mock_metrics.reset.side_effect = lambda: call_order.append("reset")
        model.test_metrics = mock_metrics
        model.log_dict = MagicMock()
        model.on_test_epoch_end()
        assert call_order == ["compute", "reset"]


# ===========================================================================
# _build_sw_core
# ===========================================================================


class TestBuildSwCore:
    def _model_with_sw_cfg(self, **sw_extra):
        return _make_model(
            _sw_cfg_dict(
                **{
                    "sliding_window_test": {
                        "tile_size": [H, W],
                        "tile_step": [H // 2, W // 2],
                        "num_classes": C,
                        "batch_size": 2,
                        "tile_weight": "mean",
                        **sw_extra,
                    }
                }
            )
        )

    def test_returns_sliding_window_core(self):
        model = self._model_with_sw_cfg()
        core = model._build_sw_core()
        assert isinstance(core, SlidingWindowCore)

    def test_cached_second_call_same_instance(self):
        model = self._model_with_sw_cfg()
        core1 = model._build_sw_core()
        core2 = model._build_sw_core()
        assert core1 is core2

    def test_num_classes_passed_correctly(self):
        model = self._model_with_sw_cfg()
        core = model._build_sw_core()
        assert core.num_classes == C

    def test_tile_size_and_step_correct(self):
        model = self._model_with_sw_cfg()
        core = model._build_sw_core()
        assert core.tile_size == (H, W)
        assert core.tile_step == (H // 2, W // 2)

    def test_tta_augmentations_passed(self):
        model = _make_model(
            _sw_cfg_dict(
                **{
                    "sliding_window_test": {
                        "tile_size": [H, W],
                        "tile_step": [H // 2, W // 2],
                        "num_classes": C,
                        "tile_weight": "mean",
                        "tta_augmentations": ["rot0", "rot90"],
                    }
                }
            )
        )
        core = model._build_sw_core()
        assert core.tta_augmentations == ["rot0", "rot90"]

    def test_no_tta_augmentations_gives_empty_list(self):
        model = self._model_with_sw_cfg()
        core = model._build_sw_core()
        assert core.tta_augmentations == []


# ===========================================================================
# _save_test_prediction
# ===========================================================================


class TestSaveTestPrediction:
    def _model(self, output_dir=None):
        sw_dict = {
            "tile_size": [H, W],
            "tile_step": [H // 2, W // 2],
            "num_classes": C,
            "tile_weight": "mean",
        }
        if output_dir is not None:
            sw_dict["output_dir"] = output_dir
        return _make_model(_sw_cfg_dict(**{"sliding_window_test": sw_dict}))

    def test_skips_when_rasterio_not_installed(self):
        model = self._model(output_dir="/some/dir")
        sw_out = _make_sw_output()
        import builtins

        real_import = builtins.__import__

        def _no_rasterio(name, *args, **kwargs):
            if name == "rasterio":
                raise ImportError("mocked")
            return real_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=_no_rasterio):
            model._save_test_prediction(sw_out, "/fake/img.tif")
        # No exception and no files written — just a silent skip

    def test_skips_when_output_dir_not_set(self):
        model = self._model(output_dir=None)
        sw_out = _make_sw_output()
        with patch("rasterio.open") as mock_open:
            model._save_test_prediction(sw_out, "/fake/img.tif")
        mock_open.assert_not_called()

    def test_writes_prediction_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            model = self._model(output_dir=tmpdir)
            sw_out = _make_sw_output()

            mock_file = MagicMock()
            mock_file.__enter__ = MagicMock(return_value=mock_file)
            mock_file.__exit__ = MagicMock(return_value=False)
            mock_file.profile = {"driver": "GTiff", "dtype": "uint8", "count": 3}

            with patch("rasterio.open", return_value=mock_file) as mock_open:
                model._save_test_prediction(sw_out, "/fake/img.tif")

            # First call reads the source profile; subsequent calls write files.
            write_paths = [
                str(c.args[0])
                for c in mock_open.call_args_list
                if c.args[1:] == ("w",) or (len(c.args) > 1 and c.args[1] == "w")
            ]
            assert any("img_prediction" in p for p in write_paths)

    def test_writes_tta_uncertainty_file_when_present(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            model = self._model(output_dir=tmpdir)
            sw_out = _make_sw_output(tta_unc=True)

            mock_file = MagicMock()
            mock_file.__enter__ = MagicMock(return_value=mock_file)
            mock_file.__exit__ = MagicMock(return_value=False)
            mock_file.profile = {"driver": "GTiff", "dtype": "uint8", "count": 3}

            with patch("rasterio.open", return_value=mock_file) as mock_open:
                model._save_test_prediction(sw_out, "/fake/img.tif")

            all_paths = [str(c.args[0]) for c in mock_open.call_args_list]
            assert any("tta_uncertainty" in p for p in all_paths)

    def test_writes_mc_uncertainty_file_when_present(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            model = self._model(output_dir=tmpdir)
            sw_out = _make_sw_output(mc_unc=True)

            mock_file = MagicMock()
            mock_file.__enter__ = MagicMock(return_value=mock_file)
            mock_file.__exit__ = MagicMock(return_value=False)
            mock_file.profile = {"driver": "GTiff", "dtype": "uint8", "count": 3}

            with patch("rasterio.open", return_value=mock_file) as mock_open:
                model._save_test_prediction(sw_out, "/fake/img.tif")

            all_paths = [str(c.args[0]) for c in mock_open.call_args_list]
            assert any("mc_uncertainty" in p for p in all_paths)

    def test_no_extra_files_without_uncertainties(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            model = self._model(output_dir=tmpdir)
            sw_out = _make_sw_output()  # no uncertainties

            mock_file = MagicMock()
            mock_file.__enter__ = MagicMock(return_value=mock_file)
            mock_file.__exit__ = MagicMock(return_value=False)
            mock_file.profile = {"driver": "GTiff", "dtype": "uint8", "count": 3}

            with patch("rasterio.open", return_value=mock_file) as mock_open:
                model._save_test_prediction(sw_out, "/fake/img.tif")

            all_paths = [str(c.args[0]) for c in mock_open.call_args_list]
            assert not any("tta_uncertainty" in p for p in all_paths)
            assert not any("mc_uncertainty" in p for p in all_paths)


# ===========================================================================
# SlidingWindowTestConfig
# ===========================================================================


class TestSlidingWindowTestConfig:
    def test_defaults(self):
        from pytorch_segmentation_models_trainer.config_definitions.sliding_window_test_config import (
            SlidingWindowTestConfig,
        )

        cfg = SlidingWindowTestConfig(num_classes=5)
        assert cfg.tile_size == [512, 512]
        assert cfg.tile_step == [256, 256]
        assert cfg.batch_size == 4
        assert cfg.tile_weight == "gaussian"
        assert cfg.tta_augmentations == []
        assert cfg.compute_tta_uncertainty is False
        assert cfg.use_mc_dropout is False
        assert cfg.n_mc_samples == 10
        assert cfg.compute_mc_uncertainty is False
        assert cfg.uncertainty_mode == "entropy"
        assert cfg.output_dir is None

    def test_num_classes_set(self):
        from pytorch_segmentation_models_trainer.config_definitions.sliding_window_test_config import (
            SlidingWindowTestConfig,
        )

        cfg = SlidingWindowTestConfig(num_classes=7)
        assert cfg.num_classes == 7

    def test_structured_config_roundtrip(self):
        """OmegaConf.structured must accept SlidingWindowTestConfig and preserve values."""
        from omegaconf import OmegaConf
        from pytorch_segmentation_models_trainer.config_definitions.sliding_window_test_config import (
            SlidingWindowTestConfig,
        )

        cfg = OmegaConf.structured(SlidingWindowTestConfig(num_classes=3))
        assert cfg.num_classes == 3
        assert cfg.tile_weight == "gaussian"
        assert cfg.uncertainty_mode == "entropy"

    def test_config_store_group_registered(self):
        """SlidingWindowTestConfig must be present in the Hydra ConfigStore."""
        from hydra.core.config_store import ConfigStore
        import pytorch_segmentation_models_trainer.config_definitions.sliding_window_test_config  # noqa: F401

        cs = ConfigStore.instance()
        assert "sliding_window_test" in cs.repo
        # ConfigStore stores nodes with .yaml suffix
        assert "default.yaml" in cs.repo["sliding_window_test"]
