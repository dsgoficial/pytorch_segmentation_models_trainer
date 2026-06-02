# -*- coding: utf-8 -*-
"""Hydra dataclass configurations for the classic ML pipeline."""

from dataclasses import dataclass, field
from typing import Any, List, Optional

from hydra.core.config_store import ConfigStore
from omegaconf import MISSING

# ---------------------------------------------------------------------------
# Feature Engineering
# ---------------------------------------------------------------------------


@dataclass
class GaborFilterExtractorConfig:
    """Configuration for :class:`~classic_ml.feature_engineering.GaborFilterExtractor`.

    Example YAML:
        - _target_: pytorch_segmentation_models_trainer.classic_ml.feature_engineering.GaborFilterExtractor
          frequencies: [0.1, 0.25, 0.4]
          num_orientations: 4
    """

    _target_: str = (
        "pytorch_segmentation_models_trainer.classic_ml"
        ".feature_engineering.GaborFilterExtractor"
    )
    frequencies: List[float] = field(default_factory=lambda: [0.1, 0.25, 0.4])
    num_orientations: int = 4


@dataclass
class GradientExtractorConfig:
    """Configuration for :class:`~classic_ml.feature_engineering.GradientExtractor`.

    Example YAML:
        - _target_: pytorch_segmentation_models_trainer.classic_ml.feature_engineering.GradientExtractor
    """

    _target_: str = (
        "pytorch_segmentation_models_trainer.classic_ml"
        ".feature_engineering.GradientExtractor"
    )


@dataclass
class MultiscaleExtractorConfig:
    """Configuration for :class:`~classic_ml.feature_engineering.MultiscaleExtractor`.

    Example YAML:
        - _target_: pytorch_segmentation_models_trainer.classic_ml.feature_engineering.MultiscaleExtractor
          sigmas: [1.0, 2.0, 4.0]
    """

    _target_: str = (
        "pytorch_segmentation_models_trainer.classic_ml"
        ".feature_engineering.MultiscaleExtractor"
    )
    sigmas: List[float] = field(default_factory=lambda: [1.0, 2.0, 4.0])


@dataclass
class FeatureEngineeringPipelineConfig:
    """Configuration for :class:`~classic_ml.feature_engineering.FeatureEngineeringPipeline`.

    Example YAML:
        feature_pipeline:
          _target_: pytorch_segmentation_models_trainer.classic_ml.feature_engineering.FeatureEngineeringPipeline
          extractors:
            - _target_: ...GaborFilterExtractor
              frequencies: [0.1, 0.25, 0.4]
              num_orientations: 4
    """

    _target_: str = (
        "pytorch_segmentation_models_trainer.classic_ml"
        ".feature_engineering.FeatureEngineeringPipeline"
    )
    extractors: List[Any] = MISSING


# ---------------------------------------------------------------------------
# Estimators
# ---------------------------------------------------------------------------


@dataclass
class GPUAcceleratedRandomForestConfig:
    """Configuration for :class:`~classic_ml.estimators.GPUAcceleratedRandomForest`.

    Example YAML:
        classifier:
          _target_: pytorch_segmentation_models_trainer.classic_ml.estimators.GPUAcceleratedRandomForest
          n_estimators: 100
          max_depth: null
          random_state: 42
    """

    _target_: str = (
        "pytorch_segmentation_models_trainer.classic_ml.estimators.GPUAcceleratedRandomForest"
    )
    n_estimators: int = 100
    max_depth: Optional[int] = None
    random_state: Optional[int] = None


@dataclass
class GPUAcceleratedSVMConfig:
    """Configuration for :class:`~classic_ml.estimators.GPUAcceleratedSVM`.

    Example YAML:
        classifier:
          _target_: pytorch_segmentation_models_trainer.classic_ml.estimators.GPUAcceleratedSVM
          C: 1.0
          kernel: rbf
    """

    _target_: str = (
        "pytorch_segmentation_models_trainer.classic_ml.estimators.GPUAcceleratedSVM"
    )
    C: float = 1.0
    kernel: str = "rbf"
    random_state: Optional[int] = None


@dataclass
class GPUAcceleratedKMeansConfig:
    """Configuration for :class:`~classic_ml.estimators.GPUAcceleratedKMeans`.

    Example YAML:
        classifier:
          _target_: pytorch_segmentation_models_trainer.classic_ml.estimators.GPUAcceleratedKMeans
          n_clusters: 8
          max_iter: 300
          random_state: 42
    """

    _target_: str = (
        "pytorch_segmentation_models_trainer.classic_ml.estimators.GPUAcceleratedKMeans"
    )
    n_clusters: int = 8
    max_iter: int = 300
    random_state: Optional[int] = None


# ---------------------------------------------------------------------------
# Postprocessing
# ---------------------------------------------------------------------------


@dataclass
class DenseCRFPostprocessorConfig:
    """Configuration for :class:`~classic_ml.postprocessing.DenseCRFPostprocessor`.

    Example YAML:
        postprocessor:
          _target_: pytorch_segmentation_models_trainer.classic_ml.postprocessing.DenseCRFPostprocessor
          n_iterations: 5
          bilateral_sxy: 80.0
          bilateral_srgb: 13.0
          bilateral_compat: 10.0
          gaussian_sxy: 3.0
          gaussian_compat: 3.0
    """

    _target_: str = (
        "pytorch_segmentation_models_trainer.classic_ml"
        ".postprocessing.DenseCRFPostprocessor"
    )
    n_iterations: int = 5
    bilateral_sxy: float = 80.0
    bilateral_srgb: float = 13.0
    bilateral_compat: float = 10.0
    gaussian_sxy: float = 3.0
    gaussian_compat: float = 3.0


@dataclass
class GraphCutsPostprocessorConfig:
    """Configuration for :class:`~classic_ml.postprocessing.GraphCutsPostprocessor`.

    Example YAML:
        postprocessor:
          _target_: pytorch_segmentation_models_trainer.classic_ml.postprocessing.GraphCutsPostprocessor
          unary_scale: 10.0
          pairwise_weight: 1.0
    """

    _target_: str = (
        "pytorch_segmentation_models_trainer.classic_ml"
        ".postprocessing.GraphCutsPostprocessor"
    )
    unary_scale: float = 10.0
    pairwise_weight: float = 1.0
    n_iter: int = -1
    algorithm: str = "swap"


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


@dataclass
class ClassicMLOrchestratorConfig:
    """Configuration for :class:`~classic_ml.orchestrator.ClassicMLOrchestrator`.

    Example YAML:
        orchestrator:
          _target_: pytorch_segmentation_models_trainer.classic_ml.orchestrator.ClassicMLOrchestrator
    """

    _target_: str = (
        "pytorch_segmentation_models_trainer.classic_ml"
        ".orchestrator.ClassicMLOrchestrator"
    )
    feature_pipeline: Any = MISSING
    classifier: Any = MISSING
    postprocessor: Optional[Any] = None


# ---------------------------------------------------------------------------
# ConfigStore registration
# ---------------------------------------------------------------------------

cs = ConfigStore.instance()

cs.store(
    group="classic_ml/feature_extractor",
    name="gabor",
    node=GaborFilterExtractorConfig,
)
cs.store(
    group="classic_ml/feature_extractor",
    name="gradient",
    node=GradientExtractorConfig,
)
cs.store(
    group="classic_ml/feature_extractor",
    name="multiscale",
    node=MultiscaleExtractorConfig,
)
cs.store(
    group="classic_ml/feature_pipeline",
    name="default",
    node=FeatureEngineeringPipelineConfig,
)
cs.store(
    group="classic_ml/classifier",
    name="random_forest",
    node=GPUAcceleratedRandomForestConfig,
)
cs.store(
    group="classic_ml/classifier",
    name="svm",
    node=GPUAcceleratedSVMConfig,
)
cs.store(
    group="classic_ml/classifier",
    name="kmeans",
    node=GPUAcceleratedKMeansConfig,
)
cs.store(
    group="classic_ml/postprocessor",
    name="dense_crf",
    node=DenseCRFPostprocessorConfig,
)
cs.store(
    group="classic_ml/postprocessor",
    name="graph_cuts",
    node=GraphCutsPostprocessorConfig,
)
cs.store(
    group="classic_ml/orchestrator",
    name="default",
    node=ClassicMLOrchestratorConfig,
)
