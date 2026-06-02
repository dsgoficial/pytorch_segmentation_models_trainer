# -*- coding: utf-8 -*-
"""Classic ML pipeline with GPU acceleration via NVIDIA RAPIDS.

Submodules
----------
feature_engineering
    GPU/CPU agnostic feature extractors (Gabor, gradients, multi-scale).
estimators
    RandomForest, SVM, and KMeans wrappers with transparent cuml acceleration.
postprocessing
    Dense CRF and graph-cuts mask refinement.
orchestrator
    End-to-end classic ML pipeline orchestrator.
"""

from pytorch_segmentation_models_trainer.classic_ml.feature_engineering import (
    FeatureEngineeringPipeline,
    GaborFilterExtractor,
    GradientExtractor,
    MultiscaleExtractor,
)
from pytorch_segmentation_models_trainer.classic_ml.estimators import (
    GPUAcceleratedKMeans,
    GPUAcceleratedRandomForest,
    GPUAcceleratedSVM,
    enable_gpu_acceleration,
)
from pytorch_segmentation_models_trainer.classic_ml.postprocessing import (
    DenseCRFPostprocessor,
    GraphCutsPostprocessor,
    PostprocessingPipeline,
)
from pytorch_segmentation_models_trainer.classic_ml.orchestrator import (
    ClassicMLOrchestrator,
)

__all__ = [
    "FeatureEngineeringPipeline",
    "GaborFilterExtractor",
    "GradientExtractor",
    "MultiscaleExtractor",
    "GPUAcceleratedRandomForest",
    "GPUAcceleratedSVM",
    "GPUAcceleratedKMeans",
    "enable_gpu_acceleration",
    "DenseCRFPostprocessor",
    "GraphCutsPostprocessor",
    "PostprocessingPipeline",
    "ClassicMLOrchestrator",
]
