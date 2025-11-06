# -*- coding: utf-8 -*-
"""
/***************************************************************************
 pytorch_segmentation_models_trainer
                              -------------------
        begin                : 2025-11-06
        copyright            : (C) 2025 by Philipe Borba
        email                : philipeborba at gmail dot com
 ***************************************************************************/
/***************************************************************************
 *                                                                         *
 *   This program is free software; you can redistribute it and/or modify  *
 *   it under the terms of the GNU General Public License as published by  *
 *   the Free Software Foundation; either version 2 of the License, or     *
 *   (at your option) any later version.                                   *
 *                                                                         *
 ***************************************************************************/
"""
from dataclasses import dataclass, field
from typing import Optional
from omegaconf import MISSING
from hydra.core.config_store import ConfigStore


@dataclass
class DataLoaderConfig:
    """
    Configuração do DataLoader para inferência.
    """
    num_workers: int = 4
    prefetch_factor: int = 2


@dataclass
class BuildCSVFromFolderConfig:
    """
    Configuração para construir CSV automaticamente de pasta de imagens.
    Usado pelo predict_from_batch.py.
    """
    enabled: bool = False
    images_folder: str = MISSING
    image_pattern: str = "*.tif"
    recursive: bool = False
    masks_folder: Optional[str] = None
    mask_pattern: Optional[str] = None
    mask_suffix: str = "_mask"
    root_dir: Optional[str] = None
    output_csv_path: Optional[str] = None
    force_rebuild: bool = False


@dataclass
class InferenceDatasetConfig:
    """
    Configuração do dataset para inferência.
    Usada pelo predict_from_batch.py.
    
    Suporta CSV existente ou construção automática.
    """
    # Modo 1: CSV existente
    input_csv_path: str = MISSING
    root_dir: Optional[str] = None
    
    # Modo 2: Construir CSV de pasta
    build_csv_from_folder: BuildCSVFromFolderConfig = field(
        default_factory=BuildCSVFromFolderConfig
    )
    
    # DataLoader settings
    data_loader: DataLoaderConfig = field(
        default_factory=DataLoaderConfig
    )
    
    # Opcional: limitar número de linhas
    n_first_rows_to_read: Optional[int] = None


# Registrar configurações com Hydra
cs = ConfigStore.instance()
cs.store(name="inference_dataset", node=InferenceDatasetConfig)
cs.store(name="build_csv_from_folder", node=BuildCSVFromFolderConfig)
cs.store(name="data_loader", node=DataLoaderConfig)
