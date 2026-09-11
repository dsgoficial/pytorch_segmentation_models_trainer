# -*- coding: utf-8 -*-
"""Shared region-based label correction core.

Used by both ``tools.sam_correction`` (SAM AMG) and ``tools.slico_correction``
(SLICO superpixels) — the two differ only in how they partition the image into
regions; the majority-vote correction rule itself lives here.
"""

from pytorch_segmentation_models_trainer.tools.region_correction.correction import (
    apply_region_correction,
    chunk_cache_key,
    parse_correction_targets,
)
from pytorch_segmentation_models_trainer.tools.region_correction.segment_cache import (
    LabelMapCache,
    SegmentListCache,
)

__all__ = [
    "apply_region_correction",
    "chunk_cache_key",
    "parse_correction_targets",
    "LabelMapCache",
    "SegmentListCache",
]
