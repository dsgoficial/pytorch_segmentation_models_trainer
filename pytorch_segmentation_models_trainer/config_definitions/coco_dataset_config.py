# -*- coding: utf-8 -*-
"""
/***************************************************************************
 pytorch_segmentation_models_trainer
                              -------------------
        begin                : 2021-08-02
        git sha              : $Format:%H$
        copyright            : (C) 2021 by Philipe Borba - Cartographic Engineer 
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
import dataclasses
from dataclasses import dataclass, field
from typing import Any, Dict, List

import hydra
import torch
from hydra.core.config_store import ConfigStore
from hydra.utils import instantiate
from omegaconf import MISSING, DictConfig, OmegaConf
from pytorch_segmentation_models_trainer.model_loader.model import Model
from pytorch_segmentation_models_trainer.utils.os_utils import \
    import_module_from_cfg


@dataclass(frozen=True)
class CocoDatasetInfoConfig:
    description: str = MISSING
    url: str = MISSING
    version: str = MISSING
    year: int = MISSING
    contributor: str = MISSING
    date_created: str = MISSING

    

@dataclass(frozen=True)
class LicenseConfig:
    id: str = MISSING
    url: str = MISSING
    name: str = MISSING

@dataclass(frozen=True)
class ImageConfig:
    license: int = MISSING
    file_name: str = MISSING
    coco_url: str = MISSING
    height: int = MISSING
    width: int = MISSING
    date_captured: str = MISSING
    flickr_url: str = MISSING
    id: int = MISSING

@dataclass(frozen=True)
class CategoryConfig:
    supercategory: str = MISSING
    id: int = MISSING
    name: str = MISSING

@dataclass(frozen=True)
class AnnotationConfig:
    id: int = MISSING
    category_id: int = MISSING
    image_id: int = MISSING
    segmentation: List[List[float]] = field(default_factory=lambda: [[]])
    bbox: List[float] = field(default_factory=lambda: [0, 0, 0, 0])
    area: float = MISSING
    iscrowd: bool = MISSING

@dataclass
class CocoDatasetConfig:
    info: CocoDatasetInfoConfig = field(default_factory=lambda: CocoDatasetInfoConfig())
    licenses: List[LicenseConfig] = field(default_factory=lambda: [LicenseConfig()])
    images: List[ImageConfig] = field(default_factory=lambda: [ImageConfig()])
    categories: List[CategoryConfig] = field(default_factory=lambda: [CategoryConfig()])
    annotations: List[AnnotationConfig] = field(default_factory=lambda: [AnnotationConfig()])
