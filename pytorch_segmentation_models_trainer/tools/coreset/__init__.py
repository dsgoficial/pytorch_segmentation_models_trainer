# -*- coding: utf-8 -*-
"""Core-set selection tools for data-efficient segmentation training.

Implements 6 model-agnostic methods from Nogueira et al. 2026 (IEEE Access):
LC, CB, FA, FD, LC/FD hybrid, FA/CB hybrid.
"""

from pytorch_segmentation_models_trainer.tools.coreset.coreset_config import (
    CoreSetConfig,
)
from pytorch_segmentation_models_trainer.tools.coreset.coreset_selector import (
    CoreSetSelector,
)
from pytorch_segmentation_models_trainer.tools.coreset.hybrid_coreset_selector import (
    EmbeddingSelectionStep,
    HybridVectorCoresetConfig,
    HybridVectorCoresetSelector,
    VectorSelectionStep,
)
from pytorch_segmentation_models_trainer.tools.coreset.vector_selector import (
    compute_intersection_areas,
    entropy_sweep_select,
    fd_embedding_select,
    lc_fd_select,
    select_by_vector_intersection,
)

__all__ = [
    "CoreSetConfig",
    "CoreSetSelector",
    "HybridVectorCoresetConfig",
    "HybridVectorCoresetSelector",
    "VectorSelectionStep",
    "EmbeddingSelectionStep",
    "compute_intersection_areas",
    "select_by_vector_intersection",
    "fd_embedding_select",
    "lc_fd_select",
    "entropy_sweep_select",
]
