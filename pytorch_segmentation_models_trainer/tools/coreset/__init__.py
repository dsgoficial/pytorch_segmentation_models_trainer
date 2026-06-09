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

__all__ = ["CoreSetConfig", "CoreSetSelector"]
