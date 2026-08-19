# -*- coding: utf-8 -*-
"""
Unit and integration tests for SlidingWindowCore and FullImageSegmentationDataset.

All tests run on CPU with minimal dummy models and synthetic tensors.
"""

import os
import tempfile
from pathlib import Path
import warnings

import numpy as np
import pytest
import torch
import torch.nn as nn

from pytorch_segmentation_models_trainer.tools.inference.sliding_window import (
    SlidingWindowCore,
    SlidingWindowOutput,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

IN_C = 3
NUM_CLASSES = 4
TILE_H, TILE_W = 32, 32
STEP_H, STEP_W = 16, 16
IMG_H, IMG_W = 64, 64

DEVICE = torch.device("cpu")


# ---------------------------------------------------------------------------
# Minimal model stubs
# ---------------------------------------------------------------------------


class TinySegModel(nn.Module):
    """Maps [B, IN_C, H, W] → [B, NUM_CLASSES, H, W] with no Dropout."""

    def __init__(self, in_c=IN_C, num_classes=NUM_CLASSES):
        super().__init__()
        self.conv = nn.Conv2d(in_c, num_classes, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class TinySegModelWithDropout(nn.Module):
    """Same as TinySegModel but with Dropout2d for MC Dropout tests."""

    def __init__(self, in_c=IN_C, num_classes=NUM_CLASSES):
        super().__init__()
        self.conv = nn.Conv2d(in_c, num_classes, kernel_size=1)
        self.dropout = nn.Dropout2d(p=0.5)

    def forward(self, x):
        return self.dropout(self.conv(x))


def make_image(h=IMG_H, w=IMG_W, c=IN_C):
    """Normalised float tensor [C, H, W]."""
    return torch.randn(c, h, w)


def make_core(model=None, **kwargs):
    """Build a SlidingWindowCore with sensible defaults."""
    if model is None:
        model = TinySegModel().eval()
    defaults = dict(
        model_fn=model,
        device=DEVICE,
        tile_size=(TILE_H, TILE_W),
        tile_step=(STEP_H, STEP_W),
        num_classes=NUM_CLASSES,
        batch_size=2,
    )
    defaults.update(kwargs)
    return SlidingWindowCore(**defaults)


# ===========================================================================
# SlidingWindowOutput
# ===========================================================================


class TestSlidingWindowOutput:
    def test_only_prediction(self):
        pred = torch.zeros(1, NUM_CLASSES, IMG_H, IMG_W)
        out = SlidingWindowOutput(prediction=pred)
        assert out.tta_uncertainty is None
        assert out.mc_uncertainty is None

    def test_with_uncertainties(self):
        pred = torch.zeros(1, NUM_CLASSES, IMG_H, IMG_W)
        tta = torch.zeros(1, 1, IMG_H, IMG_W)
        mc = torch.zeros(1, 1, IMG_H, IMG_W)
        out = SlidingWindowOutput(
            prediction=pred, tta_uncertainty=tta, mc_uncertainty=mc
        )
        assert out.tta_uncertainty is not None
        assert out.mc_uncertainty is not None


# ===========================================================================
# SlidingWindowCore.__init__ — validation
# ===========================================================================


class TestSlidingWindowCoreInit:
    def test_valid_construction(self):
        core = make_core()
        assert core.num_classes == NUM_CLASSES

    def test_invalid_tta_augmentation_raises(self):
        with pytest.raises(ValueError, match="Unknown TTA augmentations"):
            make_core(tta_augmentations=["invalid_aug"])

    def test_invalid_uncertainty_mode_raises(self):
        with pytest.raises(ValueError, match="uncertainty_mode"):
            make_core(uncertainty_mode="bad_mode")

    def test_invalid_tile_weight_raises(self):
        with pytest.raises(ValueError, match="tile_weight"):
            make_core(tile_weight="bad_weight")

    def test_tile_weight_mean(self):
        core = make_core(tile_weight="mean")
        assert core._tile_weight_arr == "mean"

    def test_tile_weight_pyramid(self):
        core = make_core(tile_weight="pyramid")
        assert core._tile_weight_arr == "pyramid"

    def test_tile_weight_gaussian_returns_array(self):
        core = make_core(tile_weight="gaussian")
        assert isinstance(core._tile_weight_arr, np.ndarray)
        assert core._tile_weight_arr.shape == (TILE_H, TILE_W)

    def test_tile_weight_custom_array(self):
        arr = np.ones((TILE_H, TILE_W), dtype=np.float32)
        core = make_core(tile_weight=arr)
        assert core._tile_weight_arr is arr

    def test_tta_augmentations_stored(self):
        augs = ["rot0", "rot90"]
        core = make_core(tta_augmentations=augs)
        assert core.tta_augmentations == augs

    def test_empty_tta_augmentations(self):
        core = make_core(tta_augmentations=[])
        assert core.tta_augmentations == []

    def test_mc_dropout_warns_when_no_dropout(self):
        model_no_dropout = TinySegModel().eval()
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            make_core(model=model_no_dropout, use_mc_dropout=True)
        assert any("dropout" in str(warning.message).lower() for warning in w)


# ===========================================================================
# _gaussian_importance_map
# ===========================================================================


class TestGaussianImportanceMap:
    def test_shape(self):
        arr = SlidingWindowCore._gaussian_importance_map((32, 48))
        assert arr.shape == (32, 48)

    def test_dtype_float32(self):
        arr = SlidingWindowCore._gaussian_importance_map((16, 16))
        assert arr.dtype == np.float32

    def test_max_at_center(self):
        arr = SlidingWindowCore._gaussian_importance_map((32, 32))
        max_idx = np.unravel_index(np.argmax(arr), arr.shape)
        center = (15, 15)
        assert abs(max_idx[0] - center[0]) <= 1
        assert abs(max_idx[1] - center[1]) <= 1

    def test_all_positive(self):
        arr = SlidingWindowCore._gaussian_importance_map((16, 16))
        assert (arr > 0).all()

    def test_max_near_one(self):
        arr = SlidingWindowCore._gaussian_importance_map((32, 32))
        assert arr.max() > 0.9


# ===========================================================================
# predict — output contracts
# ===========================================================================


class TestPredictOutputContracts:
    def test_output_is_sliding_window_output(self):
        core = make_core()
        out = core.predict(make_image())
        assert isinstance(out, SlidingWindowOutput)

    def test_prediction_shape(self):
        core = make_core()
        out = core.predict(make_image())
        assert out.prediction.shape == (1, NUM_CLASSES, IMG_H, IMG_W)

    def test_prediction_dtype_float(self):
        core = make_core()
        out = core.predict(make_image())
        assert out.prediction.is_floating_point()

    def test_no_uncertainty_by_default(self):
        core = make_core()
        out = core.predict(make_image())
        assert out.tta_uncertainty is None
        assert out.mc_uncertainty is None

    def test_accepts_4d_input(self):
        """Input [1, C, H, W] should be squeezed to [C, H, W] internally."""
        core = make_core()
        img = make_image().unsqueeze(0)  # [1, C, H, W]
        out = core.predict(img)
        assert out.prediction.shape == (1, NUM_CLASSES, IMG_H, IMG_W)

    def test_non_square_image(self):
        core = make_core()
        img = make_image(h=48, w=80)
        out = core.predict(img)
        assert out.prediction.shape == (1, NUM_CLASSES, 48, 80)

    def test_image_smaller_than_tile(self):
        """Image smaller than tile_size — padding must keep dims intact."""
        core = make_core()
        img = make_image(h=20, w=20)
        out = core.predict(img)
        assert out.prediction.shape == (1, NUM_CLASSES, 20, 20)

    def test_output_on_correct_device(self):
        core = make_core()
        out = core.predict(make_image())
        assert out.prediction.device.type == "cpu"

    def test_different_batch_sizes_same_result_shape(self):
        model = TinySegModel().eval()
        core1 = make_core(model=model, batch_size=1)
        core2 = make_core(model=model, batch_size=4)
        img = make_image()
        assert (
            core1.predict(img).prediction.shape == core2.predict(img).prediction.shape
        )


# ===========================================================================
# predict — TTA
# ===========================================================================


class TestPredictTTA:
    def test_tta_enabled_no_uncertainty(self):
        core = make_core(tta_augmentations=["rot0", "rot90"])
        out = core.predict(make_image())
        assert out.prediction.shape == (1, NUM_CLASSES, IMG_H, IMG_W)
        assert out.tta_uncertainty is None

    def test_tta_uncertainty_shape(self):
        core = make_core(
            tta_augmentations=["rot0", "rot90"],
            compute_tta_uncertainty=True,
        )
        out = core.predict(make_image())
        assert out.tta_uncertainty is not None
        assert out.tta_uncertainty.shape == (1, 1, IMG_H, IMG_W)

    def test_tta_uncertainty_non_negative(self):
        core = make_core(
            tta_augmentations=["rot0", "rot90"],
            compute_tta_uncertainty=True,
        )
        out = core.predict(make_image())
        assert (out.tta_uncertainty >= 0).all()

    def test_single_augmentation_no_tta_uncertainty(self):
        """Only one augmentation → no std possible → uncertainty stays None."""
        core = make_core(
            tta_augmentations=["rot0"],
            compute_tta_uncertainty=True,
        )
        out = core.predict(make_image())
        assert out.tta_uncertainty is None

    def test_all_d4_augmentations(self):
        core = make_core(
            tta_augmentations=[
                "rot0",
                "rot90",
                "rot180",
                "rot270",
                "flip_h",
                "flip_v",
                "flip_h_rot90",
                "flip_v_rot90",
            ],
        )
        out = core.predict(make_image())
        assert out.prediction.shape == (1, NUM_CLASSES, IMG_H, IMG_W)

    def test_mc_uncertainty_absent_when_tta_only(self):
        core = make_core(tta_augmentations=["rot0", "rot90"])
        out = core.predict(make_image())
        assert out.mc_uncertainty is None


# ===========================================================================
# predict — MC Dropout
# ===========================================================================


class TestPredictMCDropout:
    def _make_mc_core(self, **kwargs):
        model = TinySegModelWithDropout().eval()
        defaults = dict(
            model=model,
            use_mc_dropout=True,
            n_mc_samples=5,
        )
        defaults.update(kwargs)
        return make_core(**defaults)

    def test_mc_only_prediction_shape(self):
        core = self._make_mc_core()
        out = core.predict(make_image())
        assert out.prediction.shape == (1, NUM_CLASSES, IMG_H, IMG_W)

    def test_mc_uncertainty_shape(self):
        core = self._make_mc_core(compute_mc_uncertainty=True)
        out = core.predict(make_image())
        assert out.mc_uncertainty is not None
        assert out.mc_uncertainty.shape == (1, 1, IMG_H, IMG_W)

    def test_mc_uncertainty_non_negative(self):
        core = self._make_mc_core(compute_mc_uncertainty=True)
        out = core.predict(make_image())
        assert (out.mc_uncertainty >= 0).all()

    def test_mc_uncertainty_absent_when_not_requested(self):
        core = self._make_mc_core(compute_mc_uncertainty=False)
        out = core.predict(make_image())
        assert out.mc_uncertainty is None

    def test_mc_mutual_information_mode(self):
        core = self._make_mc_core(
            compute_mc_uncertainty=True,
            uncertainty_mode="mutual_information",
        )
        out = core.predict(make_image())
        assert out.mc_uncertainty is not None
        assert out.mc_uncertainty.shape == (1, 1, IMG_H, IMG_W)

    def test_tta_uncertainty_absent_when_mc_only(self):
        core = self._make_mc_core()
        out = core.predict(make_image())
        assert out.tta_uncertainty is None

    def test_model_restored_to_eval_after_mc(self):
        """Dropout layers must return to eval after MC sampling."""
        model = TinySegModelWithDropout().eval()
        core = make_core(model=model, use_mc_dropout=True, n_mc_samples=3)
        core.predict(make_image())
        assert not model.training


# ===========================================================================
# predict — TTA + MC Dropout combined
# ===========================================================================


class TestPredictTTAPlusMC:
    def test_both_uncertainties_present(self):
        model = TinySegModelWithDropout().eval()
        core = make_core(
            model=model,
            tta_augmentations=["rot0", "rot90"],
            compute_tta_uncertainty=True,
            use_mc_dropout=True,
            n_mc_samples=3,
            compute_mc_uncertainty=True,
        )
        out = core.predict(make_image())
        assert out.tta_uncertainty is not None
        assert out.mc_uncertainty is not None
        assert out.tta_uncertainty.shape == (1, 1, IMG_H, IMG_W)
        assert out.mc_uncertainty.shape == (1, 1, IMG_H, IMG_W)

    def test_prediction_shape_combined(self):
        model = TinySegModelWithDropout().eval()
        core = make_core(
            model=model,
            tta_augmentations=["rot0", "rot90"],
            use_mc_dropout=True,
            n_mc_samples=3,
        )
        out = core.predict(make_image())
        assert out.prediction.shape == (1, NUM_CLASSES, IMG_H, IMG_W)


# ===========================================================================
# _crop_merged — dimension correctness
# ===========================================================================


class TestCropMerged:
    def _make_tiler(self, img_h, img_w):
        from pytorch_toolbelt.inference.tiles import ImageSlicer

        dummy = np.zeros((img_h, img_w, IN_C), dtype=np.float32)
        return ImageSlicer(
            dummy.shape, tile_size=(TILE_H, TILE_W), tile_step=(STEP_H, STEP_W)
        )

    def test_crop_returns_original_dimensions(self):
        core = make_core()
        img = make_image(h=IMG_H, w=IMG_W)
        out = core.predict(img)
        assert out.prediction.shape[-2:] == (IMG_H, IMG_W)

    def test_crop_non_power_of_two_height(self):
        core = make_core()
        img = make_image(h=50, w=64)
        out = core.predict(img)
        assert out.prediction.shape[-2:] == (50, 64)

    def test_crop_width_not_divisible(self):
        core = make_core()
        img = make_image(h=64, w=55)
        out = core.predict(img)
        assert out.prediction.shape[-2:] == (64, 55)


# ===========================================================================
# FullImageSegmentationDataset
# ===========================================================================

# Paths to existing test data — avoids creating dummy image files.
_TESTS_DIR = Path(__file__).parent
_DATA_DIR = _TESTS_DIR / "testing_data" / "data"
_IMAGES_DIR = _DATA_DIR / "images"
_LABELS_DIR = _DATA_DIR / "labels"


def _make_csv(tmpdir: str) -> str:
    """Write a minimal CSV using existing test images."""
    from tests.utils import get_file_list, create_csv_file

    images = get_file_list(str(_IMAGES_DIR), ".png")[:3]
    labels = get_file_list(str(_LABELS_DIR), ".png")[:3]
    return create_csv_file(os.path.join(tmpdir, "test_full_image.csv"), images, labels)


class TestFullImageSegmentationDataset:
    """Tests for FullImageSegmentationDataset.image_path key contract."""

    def _build_dataset(self, csv_path: str):
        from pytorch_segmentation_models_trainer.dataset_loader.dataset import (
            FullImageSegmentationDataset,
        )

        return FullImageSegmentationDataset(input_csv_path=csv_path)

    def test_image_path_key_present(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            ds = self._build_dataset(_make_csv(tmpdir))
            sample = ds[0]
            assert "image_path" in sample

    def test_image_path_is_string(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            ds = self._build_dataset(_make_csv(tmpdir))
            sample = ds[0]
            assert isinstance(sample["image_path"], str)

    def test_image_path_points_to_existing_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            ds = self._build_dataset(_make_csv(tmpdir))
            sample = ds[0]
            assert Path(sample["image_path"]).exists()

    def test_image_and_mask_still_present(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            ds = self._build_dataset(_make_csv(tmpdir))
            sample = ds[0]
            assert "image" in sample
            assert "mask" in sample

    def test_image_path_matches_csv_entry(self):
        """image_path must equal the raw CSV image path (not a transformed copy)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            ds = self._build_dataset(_make_csv(tmpdir))
            sample = ds[0]
            expected = str(ds.get_path(0, key=ds.image_key))
            assert sample["image_path"] == expected

    def test_all_items_have_image_path(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            ds = self._build_dataset(_make_csv(tmpdir))
            for i in range(len(ds)):
                assert "image_path" in ds[i]
