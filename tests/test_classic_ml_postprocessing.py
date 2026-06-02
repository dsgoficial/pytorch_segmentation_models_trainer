# -*- coding: utf-8 -*-
"""Tests for pytorch_segmentation_models_trainer.classic_ml.postprocessing."""

import sys
from unittest.mock import MagicMock, call, patch

import numpy as np
import pytest

from pytorch_segmentation_models_trainer.classic_ml.postprocessing import (
    DenseCRFPostprocessor,
    GraphCutsPostprocessor,
    PostprocessingPipeline,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

H, W, C = 8, 8, 3
N_CLASSES = 3


def _probabilities():
    p = np.random.rand(N_CLASSES, H, W).astype(np.float32)
    p /= p.sum(axis=0, keepdims=True)
    return p


def _image():
    return (np.random.rand(H, W, C) * 255).astype(np.uint8)


def _mock_pydensecrf():
    """Return (mock_pydensecrf_pkg, mock_dcrf_module, mock_utils_module, d_instance).

    ``import pydensecrf.densecrf as dcrf`` uses the IMPORT_FROM bytecode opcode,
    which resolves to ``getattr(sys.modules["pydensecrf"], "densecrf")``.  We
    therefore create a parent mock whose ``.densecrf`` attribute is the same
    object we put in ``sys.modules["pydensecrf.densecrf"]``.
    """
    mock_dcrf = MagicMock()
    mock_utils = MagicMock()

    # DenseCRF2D instance
    d_instance = MagicMock()
    fake_Q = np.zeros((N_CLASSES, H * W), dtype=np.float32)
    fake_Q[0, :] = 1.0
    d_instance.inference.return_value = fake_Q
    mock_dcrf.DenseCRF2D.return_value = d_instance

    mock_utils.unary_from_softmax.return_value = np.zeros(
        (N_CLASSES, H * W), dtype=np.float32
    )

    # Parent package mock — attribute must match sys.modules entry
    mock_pkg = MagicMock()
    mock_pkg.densecrf = mock_dcrf
    mock_pkg.utils = mock_utils

    return mock_pkg, mock_dcrf, mock_utils, d_instance


def _mock_pygco():
    """Return a mock pygco module."""
    mock_gco = MagicMock()
    labels = np.zeros(H * W, dtype=np.int32)
    mock_gco.cut_grid_graph.return_value = labels
    return mock_gco


# ---------------------------------------------------------------------------
# DenseCRFPostprocessor — ImportError
# ---------------------------------------------------------------------------


class TestDenseCRFPostprocessorImportError:
    def test_raises_import_error_when_pydensecrf_missing(self):
        with patch.dict(
            sys.modules,
            {"pydensecrf": None, "pydensecrf.densecrf": None, "pydensecrf.utils": None},
        ):
            with pytest.raises(ImportError, match="pydensecrf"):
                DenseCRFPostprocessor()


# ---------------------------------------------------------------------------
# DenseCRFPostprocessor — happy path
# ---------------------------------------------------------------------------


def _pydensecrf_patch(mock_pkg, mock_dcrf, mock_utils):
    """Return a patch.dict context that correctly wires the pydensecrf mocks.

    ``import pydensecrf.densecrf as dcrf`` uses the IMPORT_FROM bytecode
    opcode (``getattr(parent, "densecrf")``), so the parent package mock must
    have its ``.densecrf`` attribute explicitly pointing to ``mock_dcrf``.
    """
    return patch.dict(
        sys.modules,
        {
            "pydensecrf": mock_pkg,
            "pydensecrf.densecrf": mock_dcrf,
            "pydensecrf.utils": mock_utils,
        },
    )


class TestDenseCRFPostprocessor:
    def _make(self, **kwargs):
        mock_pkg, mock_dcrf, mock_utils, d_instance = _mock_pydensecrf()
        with _pydensecrf_patch(mock_pkg, mock_dcrf, mock_utils):
            proc = DenseCRFPostprocessor(**kwargs)
        return proc, mock_pkg, mock_dcrf, mock_utils, d_instance

    def test_defaults(self):
        proc, *_ = self._make()
        assert proc.n_iterations == 5
        assert proc.bilateral_sxy == 80.0
        assert proc.gaussian_sxy == 3.0

    def test_custom_params(self):
        proc, *_ = self._make(n_iterations=10, bilateral_sxy=50.0)
        assert proc.n_iterations == 10
        assert proc.bilateral_sxy == 50.0

    def test_refine_output_shape(self):
        proc, mock_pkg, mock_dcrf, mock_utils, d_instance = self._make()
        proba = _probabilities()
        image = _image()

        with _pydensecrf_patch(mock_pkg, mock_dcrf, mock_utils):
            result = proc.refine(proba, image)

        assert result.shape == (H, W)

    def test_refine_calls_densecrf2d(self):
        proc, mock_pkg, mock_dcrf, mock_utils, d_instance = self._make()
        proba = _probabilities()
        image = _image()

        with _pydensecrf_patch(mock_pkg, mock_dcrf, mock_utils):
            proc.refine(proba, image)

        mock_dcrf.DenseCRF2D.assert_called_once_with(W, H, N_CLASSES)
        d_instance.setUnaryEnergy.assert_called_once()
        d_instance.addPairwiseBilateral.assert_called_once()
        d_instance.addPairwiseGaussian.assert_called_once()
        d_instance.inference.assert_called_once_with(proc.n_iterations)

    def test_extra_kwargs_ignored(self):
        proc, *_ = self._make(extra_param="ignored")
        assert proc.n_iterations == 5


# ---------------------------------------------------------------------------
# GraphCutsPostprocessor — ImportError
# ---------------------------------------------------------------------------


class TestGraphCutsPostprocessorImportError:
    def test_raises_import_error_when_pygco_missing(self):
        with patch.dict(sys.modules, {"pygco": None}):
            with pytest.raises(ImportError, match="pygco"):
                GraphCutsPostprocessor()


# ---------------------------------------------------------------------------
# GraphCutsPostprocessor — happy path
# ---------------------------------------------------------------------------


class TestGraphCutsPostprocessor:
    def _make(self, **kwargs):
        mock_gco = _mock_pygco()
        with patch.dict(sys.modules, {"pygco": mock_gco}):
            proc = GraphCutsPostprocessor(**kwargs)
        return proc, mock_gco

    def test_defaults(self):
        proc, _ = self._make()
        assert proc.unary_scale == 10.0
        assert proc.pairwise_weight == 1.0
        assert proc.n_iter == -1
        assert proc.algorithm == "swap"

    def test_custom_params(self):
        proc, _ = self._make(unary_scale=5.0, algorithm="expansion")
        assert proc.unary_scale == 5.0
        assert proc.algorithm == "expansion"

    def test_refine_output_shape(self):
        proc, mock_gco = self._make()
        proba = _probabilities()
        image = _image()

        with patch.dict(sys.modules, {"pygco": mock_gco}):
            result = proc.refine(proba, image)

        assert result.shape == (H, W)

    def test_refine_calls_cut_grid_graph(self):
        proc, mock_gco = self._make()
        proba = _probabilities()
        image = _image()

        with patch.dict(sys.modules, {"pygco": mock_gco}):
            proc.refine(proba, image)

        mock_gco.cut_grid_graph.assert_called_once()

    def test_refine_with_greyscale_image(self):
        """Greyscale (H, W) image should not raise."""
        proc, mock_gco = self._make()
        proba = _probabilities()
        grey = (np.random.rand(H, W) * 255).astype(np.uint8)

        with patch.dict(sys.modules, {"pygco": mock_gco}):
            result = proc.refine(proba, grey)

        assert result.shape == (H, W)

    def test_extra_kwargs_ignored(self):
        proc, _ = self._make(unused=True)
        assert proc.unary_scale == 10.0


# ---------------------------------------------------------------------------
# PostprocessingPipeline
# ---------------------------------------------------------------------------


class TestPostprocessingPipeline:
    def _make_mock_processor(self, return_value=None):
        proc = MagicMock()
        if return_value is None:
            return_value = np.zeros((H, W), dtype=np.int32)
        proc.refine.return_value = return_value
        return proc

    def test_single_postprocessor(self):
        labels = np.ones((H, W), dtype=np.int32)
        proc = self._make_mock_processor(return_value=labels)
        pipeline = PostprocessingPipeline(postprocessors=[proc])
        result = pipeline.refine(_probabilities(), _image())
        np.testing.assert_array_equal(result, labels)

    def test_multiple_postprocessors_returns_last(self):
        labels_1 = np.zeros((H, W), dtype=np.int32)
        labels_2 = np.ones((H, W), dtype=np.int32)
        proc1 = self._make_mock_processor(return_value=labels_1)
        proc2 = self._make_mock_processor(return_value=labels_2)
        pipeline = PostprocessingPipeline(postprocessors=[proc1, proc2])
        proba = _probabilities()
        image = _image()
        result = pipeline.refine(proba, image)
        np.testing.assert_array_equal(result, labels_2)

    def test_each_processor_receives_original_probabilities(self):
        """Each postprocessor should receive the original proba, not modified output."""
        proc1 = self._make_mock_processor()
        proc2 = self._make_mock_processor()
        pipeline = PostprocessingPipeline(postprocessors=[proc1, proc2])
        proba = _probabilities()
        image = _image()
        pipeline.refine(proba, image)

        # Both processors should have been called with the same proba + image
        for proc in [proc1, proc2]:
            call_args = proc.refine.call_args
            np.testing.assert_array_equal(call_args[0][0], proba)

    def test_extra_kwargs_ignored(self):
        proc = self._make_mock_processor()
        pipeline = PostprocessingPipeline(postprocessors=[proc], unused=True)
        pipeline.refine(_probabilities(), _image())
        proc.refine.assert_called_once()

    def test_empty_list_raises_value_error(self):
        with pytest.raises(ValueError, match="at least one"):
            PostprocessingPipeline(postprocessors=[])
