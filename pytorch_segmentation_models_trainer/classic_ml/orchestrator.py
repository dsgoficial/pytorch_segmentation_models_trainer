# -*- coding: utf-8 -*-
"""End-to-end classic ML segmentation orchestrator.

:class:`ClassicMLOrchestrator` is **not** a ``pl.LightningModule`` — classic
ML models do not use iterative backpropagation.  The orchestrator glues
together a :class:`~feature_engineering.FeatureEngineeringPipeline`, a
sklearn-compatible classifier, and an optional
:class:`~postprocessing.BasePostprocessor`.
"""

import pickle
from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np


class ClassicMLOrchestrator:
    """Orchestrate feature extraction, fitting, and prediction.

    The full prediction pipeline is:

    1. **Feature extraction** — :class:`~feature_engineering.FeatureEngineeringPipeline`
       maps each image pixel to a feature vector.
    2. **Classification** — a sklearn-compatible classifier assigns class
       probabilities to every pixel feature vector.
    3. **Post-processing** (optional) — a
       :class:`~postprocessing.BasePostprocessor` refines the probability
       map into a clean label map.

    Args:
        feature_pipeline: A fitted or unfitted
            :class:`~feature_engineering.FeatureEngineeringPipeline`.
        classifier: Any object with ``fit(X, y)``,  ``predict(X)``, and
            ``predict_proba(X)`` methods (e.g.
            :class:`~estimators.GPUAcceleratedRandomForest`).
        postprocessor: Optional
            :class:`~postprocessing.BasePostprocessor`.  When ``None``,
            the label map is produced by argmax over the probability map.
        **kwargs: Ignored extra Hydra arguments.

    Example YAML:
        orchestrator:
          _target_: pytorch_segmentation_models_trainer.classic_ml.orchestrator.ClassicMLOrchestrator
          feature_pipeline:
            _target_: pytorch_segmentation_models_trainer.classic_ml.feature_engineering.FeatureEngineeringPipeline
            extractors:
              - _target_: pytorch_segmentation_models_trainer.classic_ml.feature_engineering.GaborFilterExtractor
                frequencies: [0.1, 0.25, 0.4]
                num_orientations: 4
          classifier:
            _target_: pytorch_segmentation_models_trainer.classic_ml.estimators.GPUAcceleratedRandomForest
            n_estimators: 100
            random_state: 42
    """

    def __init__(
        self,
        feature_pipeline,
        classifier,
        postprocessor=None,
        **kwargs,
    ):
        self.feature_pipeline = feature_pipeline
        self.classifier = classifier
        self.postprocessor = postprocessor

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def fit(
        self,
        images: List[np.ndarray],
        masks: List[np.ndarray],
    ) -> "ClassicMLOrchestrator":
        """Fit the classifier on a list of images and corresponding masks.

        Feature extraction is performed for each image; all pixel features
        and labels are concatenated before passing to the classifier.

        Args:
            images: List of images, each of shape ``(H, W)`` or ``(H, W, C)``.
            masks: List of ground-truth label maps, each of shape ``(H, W)``
                with integer class IDs.

        Returns:
            ``self``.
        """
        X_parts = []
        y_parts = []
        for img, mask in zip(images, masks):
            X_parts.append(self.feature_pipeline.transform(img))
            y_parts.append(np.asarray(mask).ravel())
        X = np.concatenate(X_parts, axis=0)
        y = np.concatenate(y_parts, axis=0)
        self.classifier.fit(X, y)
        return self

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict(
        self,
        image: np.ndarray,
        return_probabilities: bool = False,
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """Predict a segmentation mask for a single image.

        Args:
            image: Input image of shape ``(H, W)`` or ``(H, W, C)``.
            return_probabilities: When ``True``, also return the
                ``(n_classes, H, W)`` probability map.

        Returns:
            * If ``return_probabilities=False``: label map ``(H, W)``.
            * If ``return_probabilities=True``: tuple
              ``(labels (H, W), probabilities (n_classes, H, W))``.
        """
        H, W = image.shape[:2]
        X = self.feature_pipeline.transform(image)
        proba_flat = self.classifier.predict_proba(X)  # (H*W, n_classes)
        n_classes = proba_flat.shape[1]
        proba_map = proba_flat.reshape(H, W, n_classes).transpose(
            2, 0, 1
        )  # (n_classes, H, W)

        if self.postprocessor is not None:
            labels = self.postprocessor.refine(proba_map, image)
        else:
            labels = proba_map.argmax(axis=0)

        if return_probabilities:
            return labels, proba_map
        return labels

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: Union[str, Path]) -> None:
        """Serialise the orchestrator state to a pickle file.

        Args:
            path: Destination file path (``str`` or ``Path``).
        """
        state = {
            "feature_pipeline": self.feature_pipeline,
            "classifier": self.classifier,
            "postprocessor": self.postprocessor,
        }
        with open(path, "wb") as fh:
            pickle.dump(state, fh)

    @classmethod
    def load(cls, path: Union[str, Path]) -> "ClassicMLOrchestrator":
        """Load an orchestrator from a pickle file.

        Args:
            path: Path to a file previously saved by :meth:`save`.

        Returns:
            A :class:`ClassicMLOrchestrator` with restored state.
        """
        with open(path, "rb") as fh:
            state = pickle.load(fh)
        obj = cls.__new__(cls)
        obj.feature_pipeline = state["feature_pipeline"]
        obj.classifier = state["classifier"]
        obj.postprocessor = state["postprocessor"]
        return obj
