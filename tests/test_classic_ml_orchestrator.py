# -*- coding: utf-8 -*-
"""Tests for pytorch_segmentation_models_trainer.classic_ml.orchestrator."""

import pickle
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

from pytorch_segmentation_models_trainer.classic_ml.orchestrator import (
    ClassicMLOrchestrator,
)

# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------

H, W, C = 8, 8, 3
N_CLASSES = 3


def _image():
    return (np.random.rand(H, W, C) * 255).astype(np.uint8)


def _mask():
    return np.random.randint(0, N_CLASSES, (H, W))


def _mock_pipeline(n_features: int = 5):
    """Feature pipeline that returns (H*W, n_features) arrays."""
    pipeline = MagicMock()
    pipeline.transform.return_value = np.random.rand(H * W, n_features).astype(
        np.float32
    )
    return pipeline


def _mock_classifier(n_features: int = 5):
    """Classifier with predict_proba returning (H*W, N_CLASSES)."""
    clf = MagicMock()
    clf.fit.return_value = clf
    proba = np.random.rand(H * W, N_CLASSES).astype(np.float32)
    proba /= proba.sum(axis=1, keepdims=True)
    clf.predict_proba.return_value = proba
    return clf


def _mock_postprocessor():
    proc = MagicMock()
    proc.refine.return_value = np.zeros((H, W), dtype=np.int32)
    return proc


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


class TestClassicMLOrchestratorConstruction:
    def test_stores_components(self):
        pipeline = _mock_pipeline()
        clf = _mock_classifier()
        orch = ClassicMLOrchestrator(feature_pipeline=pipeline, classifier=clf)
        assert orch.feature_pipeline is pipeline
        assert orch.classifier is clf
        assert orch.postprocessor is None

    def test_stores_postprocessor(self):
        proc = _mock_postprocessor()
        orch = ClassicMLOrchestrator(
            feature_pipeline=_mock_pipeline(),
            classifier=_mock_classifier(),
            postprocessor=proc,
        )
        assert orch.postprocessor is proc

    def test_extra_kwargs_ignored(self):
        orch = ClassicMLOrchestrator(
            feature_pipeline=_mock_pipeline(),
            classifier=_mock_classifier(),
            unused_param=True,
        )
        assert orch.classifier is not None


# ---------------------------------------------------------------------------
# fit
# ---------------------------------------------------------------------------


class TestClassicMLOrchestratorFit:
    def test_calls_pipeline_transform_for_each_image(self):
        pipeline = _mock_pipeline()
        clf = _mock_classifier()
        orch = ClassicMLOrchestrator(feature_pipeline=pipeline, classifier=clf)

        images = [_image(), _image()]
        masks = [_mask(), _mask()]
        orch.fit(images, masks)

        assert pipeline.transform.call_count == 2

    def test_calls_classifier_fit_once(self):
        pipeline = _mock_pipeline()
        clf = _mock_classifier()
        orch = ClassicMLOrchestrator(feature_pipeline=pipeline, classifier=clf)

        orch.fit([_image()], [_mask()])

        clf.fit.assert_called_once()

    def test_fit_concatenates_features_and_labels(self):
        pipeline = _mock_pipeline(n_features=5)
        clf = _mock_classifier()
        orch = ClassicMLOrchestrator(feature_pipeline=pipeline, classifier=clf)

        n_images = 3
        orch.fit([_image()] * n_images, [_mask()] * n_images)

        X_passed, y_passed = clf.fit.call_args[0]
        assert X_passed.shape == (n_images * H * W, 5)
        assert y_passed.shape == (n_images * H * W,)

    def test_fit_returns_self(self):
        pipeline = _mock_pipeline()
        clf = _mock_classifier()
        orch = ClassicMLOrchestrator(feature_pipeline=pipeline, classifier=clf)
        result = orch.fit([_image()], [_mask()])
        assert result is orch


# ---------------------------------------------------------------------------
# predict
# ---------------------------------------------------------------------------


class TestClassicMLOrchestratorPredict:
    def _fitted_orch(self, postprocessor=None):
        pipeline = _mock_pipeline()
        clf = _mock_classifier()
        orch = ClassicMLOrchestrator(
            feature_pipeline=pipeline,
            classifier=clf,
            postprocessor=postprocessor,
        )
        orch.fit([_image()], [_mask()])
        return orch

    def test_predict_output_shape(self):
        orch = self._fitted_orch()
        labels = orch.predict(_image())
        assert labels.shape == (H, W)

    def test_predict_calls_pipeline_and_classifier(self):
        pipeline = _mock_pipeline()
        clf = _mock_classifier()
        orch = ClassicMLOrchestrator(feature_pipeline=pipeline, classifier=clf)

        image = _image()
        orch.predict(image)

        pipeline.transform.assert_called_with(image)
        clf.predict_proba.assert_called_once()

    def test_predict_argmax_when_no_postprocessor(self):
        pipeline = _mock_pipeline()
        clf = _mock_classifier()
        orch = ClassicMLOrchestrator(feature_pipeline=pipeline, classifier=clf)

        # Make one class dominate uniformly
        proba = np.zeros((H * W, N_CLASSES), dtype=np.float32)
        proba[:, 2] = 1.0
        clf.predict_proba.return_value = proba

        labels = orch.predict(_image())
        assert (labels == 2).all()

    def test_predict_uses_postprocessor_when_set(self):
        proc = _mock_postprocessor()
        orch = self._fitted_orch(postprocessor=proc)

        orch.predict(_image())
        proc.refine.assert_called_once()

    def test_return_probabilities_false(self):
        orch = self._fitted_orch()
        result = orch.predict(_image(), return_probabilities=False)
        assert isinstance(result, np.ndarray)
        assert result.shape == (H, W)

    def test_return_probabilities_true(self):
        orch = self._fitted_orch()
        labels, proba_map = orch.predict(_image(), return_probabilities=True)
        assert labels.shape == (H, W)
        assert proba_map.shape == (N_CLASSES, H, W)

    def test_return_probabilities_true_with_postprocessor(self):
        proc = _mock_postprocessor()
        orch = self._fitted_orch(postprocessor=proc)
        labels, proba_map = orch.predict(_image(), return_probabilities=True)
        assert labels.shape == (H, W)
        assert proba_map.shape == (N_CLASSES, H, W)


# ---------------------------------------------------------------------------
# save / load  (use real sklearn objects — MagicMock is not picklable)
# ---------------------------------------------------------------------------

from pytorch_segmentation_models_trainer.classic_ml.estimators import (
    GPUAcceleratedRandomForest,
)
from pytorch_segmentation_models_trainer.classic_ml.feature_engineering import (
    FeatureEngineeringPipeline,
    GradientExtractor,
)


class _ConstantPostprocessor:
    """Picklable stub postprocessor that always returns an all-zero label map."""

    def refine(self, proba, image):
        return np.zeros(image.shape[:2], dtype=np.int32)


def _real_orch(with_postprocessor=False):
    """Return a fitted orchestrator with real (picklable) components."""
    pipeline = FeatureEngineeringPipeline(extractors=[GradientExtractor()])
    clf = GPUAcceleratedRandomForest(n_estimators=3, random_state=0)
    postprocessor = _ConstantPostprocessor() if with_postprocessor else None
    orch = ClassicMLOrchestrator(
        feature_pipeline=pipeline, classifier=clf, postprocessor=postprocessor
    )
    images = [_image()]
    masks = [_mask()]
    orch.fit(images, masks)
    return orch


class TestClassicMLOrchestratorPersistence:
    def test_save_creates_file(self):
        orch = _real_orch()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "model.pkl"
            orch.save(path)
            assert path.exists()

    def test_load_restores_components(self):
        orch = _real_orch()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "model.pkl"
            orch.save(path)
            loaded = ClassicMLOrchestrator.load(path)

        assert type(loaded.feature_pipeline) is type(orch.feature_pipeline)
        assert type(loaded.classifier) is type(orch.classifier)
        assert loaded.postprocessor is None

    def test_save_load_with_postprocessor(self):
        orch = _real_orch(with_postprocessor=True)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "model.pkl"
            orch.save(path)
            loaded = ClassicMLOrchestrator.load(path)
        assert loaded.postprocessor is not None

    def test_save_accepts_string_path(self):
        orch = _real_orch()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = str(Path(tmpdir) / "model.pkl")
            orch.save(path)
            loaded = ClassicMLOrchestrator.load(path)
        assert type(loaded.classifier) is type(orch.classifier)
