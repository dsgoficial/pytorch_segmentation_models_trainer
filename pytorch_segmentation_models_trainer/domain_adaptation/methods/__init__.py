# -*- coding: utf-8 -*-
"""
Concrete domain adaptation method implementations.

Add your method implementations here as subclasses of
``BaseDomainAdaptationMethod``. Each file should contain one method class.
"""

from pytorch_segmentation_models_trainer.domain_adaptation.methods.gradient_reversal import (
    GradientReversalLayer,
)
from pytorch_segmentation_models_trainer.domain_adaptation.methods.dann import (
    DANNMethod,
    DomainClassifier,
)

__all__ = [
    "GradientReversalLayer",
    "DANNMethod",
    "DomainClassifier",
]
