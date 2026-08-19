# -*- coding: utf-8 -*-
"""
Unit tests for EvidentialInferenceProcessor.

Tests cover:
  - integrate_batch with EDL dict output
  - save_uncertainty_raster: output shape, CRS/transform preservation,
    single band, values in range, nodata
  - Tiled inference smoke test (no file I/O needed for most tests)
"""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import rasterio
import rasterio.transform
import torch

from pytorch_segmentation_models_trainer.tools.inference.edl_inference_processor import (
    EvidentialInferenceProcessor,
)

B, K, H, W = 1, 4, 64, 64


# ---------------------------------------------------------------------------
# Minimal processor factory (no real model needed for unit tests)
# ---------------------------------------------------------------------------


def _make_processor(num_classes=K, output_uncertainty_path=None):
    """Build a processor with a dummy model and no export strategy."""
    import torch.nn as nn
    from pytorch_segmentation_models_trainer.custom_models.edl_wrapper import (
        EvidentialWrapper,
    )

    class TinyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(3, num_classes, 1)

        def forward(self, x):
            return self.conv(x)

    model = EvidentialWrapper(TinyModel())
    return EvidentialInferenceProcessor(
        model=model,
        device="cpu",
        batch_size=1,
        export_strategy=None,
        output_uncertainty_path=output_uncertainty_path,
        num_classes=num_classes,
    )


# ---------------------------------------------------------------------------
# save_uncertainty_raster
# ---------------------------------------------------------------------------


class TestSaveUncertaintyRaster:
    def _make_profile(self, H=64, W=64):
        return {
            "driver": "GTiff",
            "dtype": "uint8",
            "width": W,
            "height": H,
            "count": 3,
            "crs": rasterio.crs.CRS.from_epsg(4326),
            "transform": rasterio.transform.from_bounds(
                west=-43.0,
                south=-23.0,
                east=-42.0,
                north=-22.0,
                width=W,
                height=H,
            ),
        }

    def test_creates_file(self):
        proc = _make_processor()
        unc = np.random.rand(H, W).astype(np.float32)
        profile = self._make_profile()
        with tempfile.NamedTemporaryFile(suffix=".tif", delete=False) as f:
            out_path = f.name
        proc.save_uncertainty_raster(unc, profile, out_path)
        assert Path(out_path).exists()

    def test_single_band(self):
        proc = _make_processor()
        unc = np.random.rand(H, W).astype(np.float32)
        profile = self._make_profile()
        with tempfile.NamedTemporaryFile(suffix=".tif", delete=False) as f:
            out_path = f.name
        proc.save_uncertainty_raster(unc, profile, out_path)
        with rasterio.open(out_path) as ds:
            assert ds.count == 1

    def test_preserves_crs(self):
        proc = _make_processor()
        unc = np.random.rand(H, W).astype(np.float32)
        profile = self._make_profile()
        with tempfile.NamedTemporaryFile(suffix=".tif", delete=False) as f:
            out_path = f.name
        proc.save_uncertainty_raster(unc, profile, out_path)
        with rasterio.open(out_path) as ds:
            assert ds.crs == rasterio.crs.CRS.from_epsg(4326)

    def test_preserves_transform(self):
        proc = _make_processor()
        unc = np.random.rand(H, W).astype(np.float32)
        profile = self._make_profile()
        expected_transform = profile["transform"]
        with tempfile.NamedTemporaryFile(suffix=".tif", delete=False) as f:
            out_path = f.name
        proc.save_uncertainty_raster(unc, profile, out_path)
        with rasterio.open(out_path) as ds:
            assert ds.transform == expected_transform

    def test_float32_dtype(self):
        proc = _make_processor()
        unc = np.random.rand(H, W).astype(np.float32)
        profile = self._make_profile()
        with tempfile.NamedTemporaryFile(suffix=".tif", delete=False) as f:
            out_path = f.name
        proc.save_uncertainty_raster(unc, profile, out_path)
        with rasterio.open(out_path) as ds:
            data = ds.read(1)
            assert data.dtype == np.float32

    def test_values_preserved(self):
        proc = _make_processor()
        unc = np.ones((H, W), dtype=np.float32) * 0.42
        profile = self._make_profile()
        with tempfile.NamedTemporaryFile(suffix=".tif", delete=False) as f:
            out_path = f.name
        proc.save_uncertainty_raster(unc, profile, out_path)
        with rasterio.open(out_path) as ds:
            data = ds.read(1)
            assert np.allclose(data, 0.42, atol=1e-4)

    def test_accepts_hwc_array(self):
        """Input shape [H, W, 1] should work (output of merge_masks)."""
        proc = _make_processor()
        unc = np.random.rand(H, W, 1).astype(np.float32)
        profile = self._make_profile()
        with tempfile.NamedTemporaryFile(suffix=".tif", delete=False) as f:
            out_path = f.name
        proc.save_uncertainty_raster(unc, profile, out_path)
        with rasterio.open(out_path) as ds:
            assert ds.count == 1


# ---------------------------------------------------------------------------
# integrate_batch
# ---------------------------------------------------------------------------


class TestIntegrateBatch:
    def test_dict_output_integrated(self):
        from pytorch_toolbelt.inference.tiles import ImageSlicer

        proc = _make_processor(num_classes=K)
        tiler = ImageSlicer(
            (H, W, 3), tile_size=(32, 32), tile_step=(32, 32), weight="mean"
        )
        merger_dict = proc.get_merger_dict(tiler)

        probs = torch.rand(1, K, 32, 32)
        uncertainty = torch.rand(1, 1, 32, 32)
        pred = {"probs": probs, "uncertainty": uncertainty, "alpha": probs + 1.0}
        coords = torch.tensor([[0, 0, 32, 32]])

        proc.integrate_batch(pred, coords, merger_dict)
        # Should not raise; uncertainty merger should have data
        merged_unc = merger_dict["uncertainty"].merge()
        assert merged_unc is not None

    def test_plain_tensor_fallback(self):
        """When a plain tensor is passed, probs merger is used, unc is zeros."""
        from pytorch_toolbelt.inference.tiles import ImageSlicer

        proc = _make_processor(num_classes=K)
        tiler = ImageSlicer(
            (H, W, 3), tile_size=(32, 32), tile_step=(32, 32), weight="mean"
        )
        merger_dict = proc.get_merger_dict(tiler)

        plain = torch.rand(1, K, 32, 32)
        coords = torch.tensor([[0, 0, 32, 32]])
        proc.integrate_batch(plain, coords, merger_dict)
        # Should not raise


# ---------------------------------------------------------------------------
# get_merger_dict
# ---------------------------------------------------------------------------


class TestGetMergerDict:
    def test_returns_probs_and_uncertainty(self):
        from pytorch_toolbelt.inference.tiles import ImageSlicer

        proc = _make_processor(num_classes=K)
        tiler = ImageSlicer(
            (H, W, 3), tile_size=(32, 32), tile_step=(32, 32), weight="mean"
        )
        mergers = proc.get_merger_dict(tiler)
        assert "probs" in mergers
        assert "uncertainty" in mergers

    def test_alpha_merger_when_export_alpha(self):
        from pytorch_toolbelt.inference.tiles import ImageSlicer
        import torch.nn as nn
        from pytorch_segmentation_models_trainer.custom_models.edl_wrapper import (
            EvidentialWrapper,
        )

        class TinyModel(nn.Module):
            def forward(self, x):
                return torch.zeros(x.shape[0], K, x.shape[2], x.shape[3])

        proc = EvidentialInferenceProcessor(
            model=EvidentialWrapper(TinyModel()),
            device="cpu",
            batch_size=1,
            export_strategy=None,
            num_classes=K,
            export_alpha=True,
        )
        tiler = ImageSlicer(
            (H, W, 3), tile_size=(32, 32), tile_step=(32, 32), weight="mean"
        )
        mergers = proc.get_merger_dict(tiler)
        assert "alpha" in mergers


class TestMergeAndSave:
    def _make_profile(self, H=8, W=8):
        return {
            "driver": "GTiff",
            "dtype": "uint8",
            "width": W,
            "height": H,
            "count": 3,
            "crs": rasterio.crs.CRS.from_epsg(4326),
            "transform": rasterio.transform.from_origin(0, H, 1, 1),
            "blockxsize": 16,
            "blockysize": 16,
            "tiled": True,
        }

    def test_integrate_batch_writes_alpha_when_enabled(self):
        proc = _make_processor(num_classes=K)
        proc.export_alpha = True
        merger_dict = {
            "probs": MagicMock(),
            "uncertainty": MagicMock(),
            "alpha": MagicMock(),
        }
        pred = {
            "probs": torch.rand(1, K, 4, 4),
            "uncertainty": torch.rand(1, 1, 4, 4),
            "alpha": torch.rand(1, K, 4, 4),
        }

        proc.integrate_batch(pred, torch.tensor([[0, 0, 4, 4]]), merger_dict)

        merger_dict["alpha"].integrate_batch.assert_called_once()

    def test_merge_masks_crops_alpha(self):
        class FakeMerger:
            def __init__(self, tensor):
                self.tensor = tensor

            def merge(self):
                return self.tensor

        class FakeTiler:
            def crop_to_orignal_size(self, array):
                return array[:4, :4]

        proc = _make_processor(num_classes=K)
        proc.export_alpha = True
        output = proc.merge_masks(
            FakeTiler(),
            {
                "probs": FakeMerger(torch.ones(K, 5, 5)),
                "uncertainty": FakeMerger(torch.ones(1, 5, 5)),
                "alpha": FakeMerger(torch.ones(K, 5, 5) * 2),
            },
        )

        assert output["seg"].shape == (4, 4, K)
        assert output["uncertainty"].shape == (4, 4, 1)
        assert output["alpha"].shape == (4, 4, K)

    def test_save_inference_exports_uncertainty_alpha_and_parent(self, tmp_path):
        proc = _make_processor(
            num_classes=K,
            output_uncertainty_path=str(tmp_path / "unc.tif"),
        )
        proc.export_alpha = True
        profile = self._make_profile()
        inference = {
            "seg": np.ones((8, 8, K), dtype=np.float32),
            "uncertainty": np.ones((8, 8, 1), dtype=np.float32) * 0.3,
            "alpha": np.ones((8, 8, K), dtype=np.float32) * 2,
        }

        with patch(
            "pytorch_segmentation_models_trainer.tools.inference.inference_processors.SingleImageInfereceProcessor.save_inference"
        ) as parent_save:
            proc.save_inference(
                "image.tif", 0.5, profile, inference, {}, apply_threshold=True
            )

        assert (tmp_path / "unc.tif").exists()
        assert (tmp_path / "unc_alpha.tif").exists()
        parent_save.assert_called_once()

    def test_save_multiband_accepts_2d_array(self, tmp_path):
        proc = _make_processor()
        out_path = tmp_path / "single_as_multiband.tif"
        proc._save_multiband_raster(
            np.ones((8, 8), dtype=np.float32),
            self._make_profile(),
            out_path,
        )

        with rasterio.open(out_path) as ds:
            assert ds.count == 1

    def test_predict_with_uncertainty_temporarily_overrides_path(self, tmp_path):
        proc = _make_processor(output_uncertainty_path=str(tmp_path / "initial.tif"))
        original = proc.output_uncertainty_path
        proc.process = MagicMock(return_value={"seg": np.zeros((2, 2, K))})

        output = proc.predict_with_uncertainty(
            "image.tif",
            output_uncertainty_path=tmp_path / "override.tif",
            threshold=0.25,
        )

        assert output["seg"].shape == (2, 2, K)
        assert proc.output_uncertainty_path == original
        proc.process.assert_called_once_with(
            image_path="image.tif",
            threshold=0.25,
            save_inference_output=True,
        )
