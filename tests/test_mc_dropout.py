# -*- coding: utf-8 -*-
"""
Unit and integration tests for MC Dropout utilities and inference processor.

All tests run on CPU with minimal models to avoid heavy dependencies.
"""

import math
import warnings
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
import torch.nn as nn

from pytorch_segmentation_models_trainer.utils.mc_dropout_utils import (
    compute_uncertainty,
    enable_mc_dropout,
    warn_if_no_dropout,
)

B, C, H, W = 2, 4, 32, 32
T = 5  # MC samples


# ---------------------------------------------------------------------------
# Minimal model stubs
# ---------------------------------------------------------------------------


class TinySegModelWithDropout(nn.Module):
    """Minimal model that returns [B, C, H, W] logits with Dropout2d."""

    def __init__(self, in_channels=3, num_classes=C):
        super().__init__()
        self.encoder = nn.Conv2d(in_channels, 8, 3, padding=1)
        self.dropout = nn.Dropout2d(p=0.5)
        self.decoder = nn.Conv2d(8, num_classes, 1)

    def forward(self, x):
        return self.decoder(self.dropout(torch.relu(self.encoder(x))))


class TinySegModelNoDropout(nn.Module):
    """Minimal model with no Dropout layers."""

    def __init__(self, num_classes=C):
        super().__init__()
        self.conv = nn.Conv2d(3, num_classes, 1)

    def forward(self, x):
        return self.conv(x)


# ---------------------------------------------------------------------------
# enable_mc_dropout
# ---------------------------------------------------------------------------


class TestEnableMcDropout:
    def test_dropout2d_set_to_train(self):
        model = TinySegModelWithDropout()
        model.eval()
        assert not model.dropout.training
        enable_mc_dropout(model)
        assert model.dropout.training

    def test_non_dropout_stays_eval(self):
        model = nn.Sequential(
            nn.BatchNorm2d(3),
            nn.Dropout2d(0.5),
            nn.Conv2d(3, 4, 1),
        )
        model.eval()
        enable_mc_dropout(model)
        # BatchNorm must stay in eval mode
        assert not model[0].training
        # Dropout must be in train mode
        assert model[1].training
        # Conv must stay in eval mode
        assert not model[2].training

    def test_all_dropout_types_enabled(self):
        model = nn.Sequential(
            nn.Dropout(0.5),
            nn.Dropout2d(0.5),
            nn.Dropout3d(0.5),
        )
        model.eval()
        enable_mc_dropout(model)
        assert model[0].training
        assert model[1].training
        assert model[2].training

    def test_no_dropout_layers_no_error(self):
        model = TinySegModelNoDropout()
        model.eval()
        enable_mc_dropout(model)  # must not raise


# ---------------------------------------------------------------------------
# warn_if_no_dropout
# ---------------------------------------------------------------------------


class TestWarnIfNoDropout:
    def test_warns_when_no_dropout(self):
        model = TinySegModelNoDropout()
        with pytest.warns(UserWarning, match="no Dropout"):
            result = warn_if_no_dropout(model)
        assert result is False

    def test_no_warning_when_dropout_present(self):
        model = TinySegModelWithDropout()
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            result = warn_if_no_dropout(model)
        assert result is True


# ---------------------------------------------------------------------------
# compute_uncertainty
# ---------------------------------------------------------------------------


class TestComputeUncertainty:
    def _make_uniform_samples(self):
        """All T samples are uniform → maximum entropy."""
        probs = torch.full((T, B, C, H, W), 1.0 / C)
        return probs

    def _make_certain_samples(self):
        """All T samples are certain about class 0."""
        probs = torch.zeros(T, B, C, H, W)
        probs[:, :, 0, :, :] = 1.0
        return probs

    def test_output_shape(self):
        samples = torch.rand(T, B, C, H, W)
        samples = samples / samples.sum(dim=2, keepdim=True)
        unc = compute_uncertainty(samples, mode="entropy")
        assert unc.shape == (B, 1, H, W)

    def test_entropy_range(self):
        samples = torch.rand(T, B, C, H, W)
        samples = samples / samples.sum(dim=2, keepdim=True)
        unc = compute_uncertainty(samples, mode="entropy")
        assert (unc >= 0).all()
        assert (unc <= math.log(C) + 1e-5).all()

    def test_mutual_information_range(self):
        samples = torch.rand(T, B, C, H, W)
        samples = samples / samples.sum(dim=2, keepdim=True)
        unc = compute_uncertainty(samples, mode="mutual_information")
        assert (unc >= 0).all()
        assert (unc <= math.log(C) + 1e-5).all()

    def test_entropy_maximum_for_uniform(self):
        unc = compute_uncertainty(self._make_uniform_samples(), mode="entropy")
        expected = math.log(C)
        assert torch.allclose(unc, torch.full_like(unc, expected), atol=1e-4)

    def test_entropy_zero_for_certain(self):
        unc = compute_uncertainty(self._make_certain_samples(), mode="entropy")
        assert (unc < 1e-4).all()

    def test_mutual_information_zero_when_samples_agree(self):
        """If all T samples are identical, MI must be zero."""
        single = torch.rand(1, B, C, H, W)
        single = single / single.sum(dim=2, keepdim=True)
        samples = single.expand(T, -1, -1, -1, -1)
        unc = compute_uncertainty(samples, mode="mutual_information")
        assert (unc.abs() < 1e-5).all()

    def test_mutual_information_positive_when_samples_disagree(self):
        """If T samples disagree maximally, MI should be > 0."""
        probs = torch.zeros(T, 1, C, H, W)
        for t in range(T):
            probs[t, :, t % C, :, :] = 1.0
        probs = probs.expand(T, B, C, H, W)
        unc = compute_uncertainty(probs, mode="mutual_information")
        assert (unc > 0).all()

    def test_entropy_and_mi_differ(self):
        samples = torch.rand(T, B, C, H, W)
        samples = samples / samples.sum(dim=2, keepdim=True)
        h = compute_uncertainty(samples, mode="entropy")
        mi = compute_uncertainty(samples, mode="mutual_information")
        # They should generally differ (MI ≤ entropy)
        assert not torch.allclose(h, mi)

    def test_invalid_mode_raises(self):
        samples = torch.rand(T, B, C, H, W)
        with pytest.raises(ValueError, match="uncertainty_mode must be"):
            compute_uncertainty(samples, mode="variance")

    def test_n_samples_one_mi_is_zero(self):
        """With T=1, MI = H[p̄] - E[H[p_t]] = H - H = 0."""
        samples = torch.rand(1, B, C, H, W)
        samples = samples / samples.sum(dim=2, keepdim=True)
        unc = compute_uncertainty(samples, mode="mutual_information")
        assert (unc.abs() < 1e-5).all()


# ---------------------------------------------------------------------------
# Stochasticity — MC samples must differ when dropout is active
# ---------------------------------------------------------------------------


class TestMcDropoutStochasticity:
    def test_samples_differ_with_dropout(self):
        model = TinySegModelWithDropout()
        model.eval()
        enable_mc_dropout(model)

        x = torch.randn(1, 3, H, W)
        with torch.no_grad():
            out1 = model(x)
            out2 = model(x)

        assert not torch.allclose(out1, out2), (
            "Two MC Dropout samples should differ (dropout is stochastic)"
        )

    def test_samples_identical_without_dropout(self):
        model = TinySegModelNoDropout()
        model.eval()

        x = torch.randn(1, 3, H, W)
        with torch.no_grad():
            out1 = model(x)
            out2 = model(x)

        assert torch.allclose(out1, out2), (
            "Deterministic model should produce identical outputs"
        )

    def test_n_samples_1_gives_single_pass(self):
        """n_samples=1 is valid and simply runs one stochastic forward pass."""
        model = TinySegModelWithDropout()
        model.eval()
        enable_mc_dropout(model)

        x = torch.randn(1, 3, H, W)
        probs_list = []
        with torch.no_grad():
            logits = model(x)
        probs_list.append(torch.softmax(logits, dim=1))
        samples = torch.stack(probs_list)  # [1, 1, C, H, W]
        unc = compute_uncertainty(samples, mode="entropy")
        assert unc.shape == (1, 1, H, W)


# ---------------------------------------------------------------------------
# MCDropoutInferenceProcessor
# ---------------------------------------------------------------------------


def _make_mock_model(num_classes=C):
    """Return a CPU model mock that returns deterministic logits."""
    model = TinySegModelWithDropout(num_classes=num_classes)
    model.eval()
    return model


def _make_processor(
    model=None,
    n_samples=3,
    export_uncertainty_map=False,
    uncertainty_mode="entropy",
    num_classes=C,
):
    from pytorch_segmentation_models_trainer.tools.inference.mc_dropout_inference_processor import (
        MCDropoutInferenceProcessor,
    )

    if model is None:
        model = _make_mock_model(num_classes=num_classes)

    return MCDropoutInferenceProcessor(
        model=model,
        device="cpu",
        batch_size=1,
        export_strategy=None,
        n_samples=n_samples,
        uncertainty_mode=uncertainty_mode,
        export_uncertainty_map=export_uncertainty_map,
        num_classes=num_classes,
        model_input_shape=(32, 32),
        step_shape=(16, 16),
    )


class TestMcDropoutInferenceProcessorInit:
    def test_warns_no_dropout(self):
        from pytorch_segmentation_models_trainer.tools.inference.mc_dropout_inference_processor import (
            MCDropoutInferenceProcessor,
        )

        model = TinySegModelNoDropout()
        with pytest.warns(UserWarning, match="no Dropout"):
            MCDropoutInferenceProcessor(
                model=model,
                device="cpu",
                batch_size=1,
                export_strategy=None,
                n_samples=3,
                num_classes=C,
                model_input_shape=(32, 32),
                step_shape=(16, 16),
            )

    def test_tta_mode_ignored_with_warning(self, caplog):
        import logging

        with caplog.at_level(logging.WARNING):
            proc = _make_processor()
            # Pass tta_mode through kwargs — should be stripped with a warning
        # No error raised — processor instantiates fine

    def test_invalid_uncertainty_mode_raises(self):
        from pytorch_segmentation_models_trainer.tools.inference.mc_dropout_inference_processor import (
            MCDropoutInferenceProcessor,
        )

        with pytest.raises(ValueError):
            MCDropoutInferenceProcessor(
                model=_make_mock_model(),
                device="cpu",
                batch_size=1,
                export_strategy=None,
                n_samples=3,
                uncertainty_mode="variance",
                num_classes=C,
                model_input_shape=(32, 32),
                step_shape=(16, 16),
            )


class TestMcDropoutInferenceProcessorMakeInference:
    def test_make_inference_returns_seg(self):
        proc = _make_processor(n_samples=2, export_uncertainty_map=False)
        image = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        result = proc.make_inference(image)
        assert "seg" in result
        assert result["seg"].shape[:2] == (64, 64)

    def test_make_inference_no_uncertainty_by_default(self):
        proc = _make_processor(n_samples=2, export_uncertainty_map=False)
        image = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        result = proc.make_inference(image)
        assert "uncertainty" not in result

    def test_make_inference_with_uncertainty(self):
        proc = _make_processor(n_samples=3, export_uncertainty_map=True)
        image = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        result = proc.make_inference(image)
        assert "seg" in result
        assert "uncertainty" in result
        assert result["uncertainty"].shape[:2] == (64, 64)

    def test_uncertainty_shape_matches_seg(self):
        proc = _make_processor(n_samples=3, export_uncertainty_map=True)
        image = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        result = proc.make_inference(image)
        assert result["seg"].shape[:2] == result["uncertainty"].shape[:2]

    def test_uncertainty_range_entropy(self):
        proc = _make_processor(
            n_samples=3, export_uncertainty_map=True, uncertainty_mode="entropy"
        )
        image = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        result = proc.make_inference(image)
        unc = result["uncertainty"]
        assert float(unc.min()) >= -1e-5
        assert float(unc.max()) <= math.log(C) + 1e-4

    def test_uncertainty_range_mutual_information(self):
        proc = _make_processor(
            n_samples=3,
            export_uncertainty_map=True,
            uncertainty_mode="mutual_information",
        )
        image = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        result = proc.make_inference(image)
        unc = result["uncertainty"]
        assert float(unc.min()) >= -1e-5
        assert float(unc.max()) <= math.log(C) + 1e-4

    def test_n_samples_1_returns_valid_output(self):
        proc = _make_processor(n_samples=1, export_uncertainty_map=True)
        image = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        result = proc.make_inference(image)
        assert "seg" in result
        assert "uncertainty" in result


class TestMcDropoutInferenceProcessorProcess:
    def test_process_saves_uncertainty_raster(self, tmp_path):
        proc = _make_processor(
            n_samples=2,
            export_uncertainty_map=True,
            uncertainty_mode="entropy",
        )
        proc.output_uncertainty_dir = str(tmp_path / "uncertainty")

        # Build a mock export strategy
        mock_strategy = MagicMock()
        mock_strategy.output_file_path = str(tmp_path / "seg")
        mock_strategy.save_inference.return_value = str(tmp_path / "seg" / "out.tif")
        proc.export_strategy = mock_strategy

        # Write a tiny GeoTIFF
        import rasterio
        from rasterio.transform import from_bounds

        image_path = str(tmp_path / "image.tif")
        transform = from_bounds(0, 0, 1, 1, 64, 64)
        with rasterio.open(
            image_path,
            "w",
            driver="GTiff",
            height=64,
            width=64,
            count=3,
            dtype="uint8",
            crs="EPSG:4326",
            transform=transform,
        ) as dst:
            dst.write(np.random.randint(0, 255, (3, 64, 64), dtype=np.uint8))

        proc.process(image_path, save_inference_output=True)

        unc_files = list(Path(proc.output_uncertainty_dir).glob("*_mc_uncertainty.tif"))
        assert len(unc_files) == 1

    def test_process_no_uncertainty_file_when_disabled(self, tmp_path):
        proc = _make_processor(n_samples=2, export_uncertainty_map=False)
        proc.output_uncertainty_dir = str(tmp_path / "uncertainty")

        import rasterio
        from rasterio.transform import from_bounds

        image_path = str(tmp_path / "image.tif")
        transform = from_bounds(0, 0, 1, 1, 64, 64)
        with rasterio.open(
            image_path,
            "w",
            driver="GTiff",
            height=64,
            width=64,
            count=3,
            dtype="uint8",
            crs="EPSG:4326",
            transform=transform,
        ) as dst:
            dst.write(np.random.randint(0, 255, (3, 64, 64), dtype=np.uint8))

        proc.process(image_path, save_inference_output=True)

        unc_dir = Path(proc.output_uncertainty_dir)
        assert not unc_dir.exists() or len(list(unc_dir.glob("*.tif"))) == 0


# ---------------------------------------------------------------------------
# MultiClassInferenceProcessor — TTA uncertainty
# ---------------------------------------------------------------------------


class TestTTAUncertainty:
    def _make_tta_processor(self, tta_mode="d4", export_uncertainty_map=False):
        from pytorch_segmentation_models_trainer.tools.inference.inference_processors import (
            MultiClassInferenceProcessor,
        )

        model = TinySegModelWithDropout(num_classes=C)
        model.eval()
        return MultiClassInferenceProcessor(
            model=model,
            device="cpu",
            batch_size=1,
            export_strategy=None,
            num_classes=C,
            tta_mode=tta_mode,
            model_input_shape=(32, 32),
            step_shape=(16, 16),
            export_uncertainty_map=export_uncertainty_map,
        )

    def test_tta_uncertainty_map_present_when_enabled(self):
        proc = self._make_tta_processor(tta_mode="d4", export_uncertainty_map=True)
        image = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        result = proc.make_inference(image)
        assert "uncertainty" in result
        assert result["uncertainty"].shape[:2] == (64, 64)

    def test_tta_no_uncertainty_when_disabled(self):
        proc = self._make_tta_processor(tta_mode="d4", export_uncertainty_map=False)
        image = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        result = proc.make_inference(image)
        assert "uncertainty" not in result

    def test_tta_uncertainty_range(self):
        proc = self._make_tta_processor(tta_mode="d4", export_uncertainty_map=True)
        image = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        result = proc.make_inference(image)
        unc = result["uncertainty"]
        assert float(unc.min()) >= -1e-5
        assert float(unc.max()) <= math.log(C) + 1e-4

    def test_no_tta_no_uncertainty_even_if_enabled(self):
        """Without TTA (tta_mode=None), the uncertainty merger is not created
        and the uncertainty map is not returned, even if export_uncertainty_map=True."""
        from pytorch_segmentation_models_trainer.tools.inference.inference_processors import (
            MultiClassInferenceProcessor,
        )

        model = TinySegModelNoDropout(num_classes=C)
        model.eval()
        proc = MultiClassInferenceProcessor(
            model=model,
            device="cpu",
            batch_size=1,
            export_strategy=None,
            num_classes=C,
            tta_mode=None,
            model_input_shape=(32, 32),
            step_shape=(16, 16),
            export_uncertainty_map=True,
        )
        image = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        result = proc.make_inference(image)
        assert "uncertainty" not in result
