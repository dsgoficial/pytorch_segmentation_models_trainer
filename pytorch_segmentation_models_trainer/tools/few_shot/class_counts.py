# -*- coding: utf-8 -*-
"""
/***************************************************************************
 pytorch_segmentation_models_trainer
                              -------------------
        begin                : 2026-10-07
        copyright            : (C) 2026 by Philipe Borba - Cartographic Engineer
                                                            @ Brazilian Army
        email                : philipeborba at gmail dot com
 ***************************************************************************/
/***************************************************************************
 *                                                                         *
 *   This program is free software; you can redistribute it and/or modify  *
 *   it under the terms of the GNU General Public License as published by  *
 *   the Free Software Foundation; either version 2 of the License, or     *
 *   (at your option) any later version.                                   *
 *                                                                         *
 ****
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np
import rasterio
from omegaconf import DictConfig, OmegaConf
from rasterio.windows import Window
from tqdm import tqdm

from pytorch_segmentation_models_trainer.config_definitions.few_shot_config import (
    ClassCountsConfig,
)
from pytorch_segmentation_models_trainer.dataset_loader.mask_class_mapping import (
    apply_mask_class_lut,
    build_mask_class_lut,
)
from pytorch_segmentation_models_trainer.dataset_loader.mbtiles_mask_dataset import (
    MBTilesMaskWindowedDataset,
)

logger = logging.getLogger(__name__)


def count_class_pixels(
    records: Sequence[Dict[str, Any]],
    num_classes: int,
    mask_class_mapping: Optional[Mapping] = None,
    ignore_index: int = 255,
) -> List[int]:
    """Pixel count of each class 0 .. num_classes - 1 over mask windows.

    Args:
        records: Window records (resolved ``mask_path``, pixel offsets/size).
        num_classes: Number of classes counted.
        mask_class_mapping: Optional ``{source: target}`` remapping applied
            first (e.g. ``{5: 3}`` for a base model trained with merged
            classes).
        ignore_index: Excluded value; values ``>= num_classes`` are excluded too.

    Returns:
        List of ``num_classes`` integers.
    """
    lut = build_mask_class_lut(mask_class_mapping)
    counts = np.zeros(num_classes, dtype=np.int64)
    for rec in tqdm(records, desc="Class pixel counts", leave=False):
        window = Window(rec["col_off"], rec["row_off"], rec["width"], rec["height"])
        with rasterio.open(rec["mask_path"]) as src:
            mask = src.read(1, window=window)
        if lut is not None:
            mask = apply_mask_class_lut(mask, lut)
        values = mask[(mask != ignore_index) & (mask < num_classes)]
        counts += np.bincount(values.ravel(), minlength=num_classes)[:num_classes]
    return [int(c) for c in counts]


def count_class_pixels_from_config(cfg: DictConfig) -> str:
    """Entry point of ``mode: count-class-pixels``; writes a JSON file
    ``{"num_classes": n, "counts": [...]}`` and returns its path."""
    cc = OmegaConf.merge(
        OmegaConf.structured(ClassCountsConfig),
        OmegaConf.to_container(cfg.class_counts, resolve=True),
    )
    records = MBTilesMaskWindowedDataset._read_window_index_cache(
        Path(cc.window_index_cache),
        mask_path_key=cc.window_index_mask_path_key,
        coordinate_mode=cc.window_index_coordinate_mode,
        mask_base_path=Path(cc.mask_base_path) if cc.mask_base_path else None,
    )
    mapping = (
        OmegaConf.to_container(cc.mask_class_mapping)
        if cc.mask_class_mapping is not None
        else None
    )
    counts = count_class_pixels(records, cc.num_classes, mapping, cc.ignore_index)
    output = Path(cc.output_json)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({"num_classes": cc.num_classes, "counts": counts}))
    logger.info("Class pixel counts %s -> %s", counts, output)
    return str(output)
