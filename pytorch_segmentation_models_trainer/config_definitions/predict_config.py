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
from typing import Any, Optional, List
from omegaconf import MISSING
from hydra.core.config_store import ConfigStore


@dataclass
class InferenceImageReaderConfig:
    """
    Configuração do leitor de imagens para inferência.
    Usada pelo predict.py via get_images().
    """
    _target_: str = MISSING
    input_csv_path: str = MISSING
    key: str = "image"
    root_dir: Optional[str] = None
    n_first_rows_to_read: Optional[int] = None


@dataclass
class InferenceProcessorConfig:
    """
    Configuração do processador de inferência single-image.
    Usada pelo predict.py via instantiate_inference_processor().
    
    Nota: model, polygonizer, export_strategy, device, batch_size e mask_bands
    são injetados automaticamente pelo código, não precisam estar no YAML.
    """
    _target_: str = MISSING
    model_input_shape: Optional[List[int]] = None
    step_shape: Optional[List[int]] = None


@dataclass
class ExportStrategyConfig:
    """
    Configuração da estratégia de exportação de inferência.
    Usada pelo predict.py (opcional).
    """
    _target_: str = MISSING
    # Campos dependem do tipo de estratégia


@dataclass
class PolygonizerConfig:
    """
    Configuração do polygonizer.
    Usada pelo predict.py via instantiate_polygonizer().
    """
    _target_: str = MISSING
    config: Optional[Any] = None
    data_writer: Optional[Any] = None


@dataclass
class PredictSingleImageConfig:
    """
    Configuração completa para predict.py (single image processor).
    
    Este config é usado pelo script predict.py que processa imagens
    uma por vez usando inference_processor.process().
    
    Campos injetados automaticamente pelo código:
    - inference_processor.model (de checkpoint)
    - inference_processor.polygonizer (de polygonizer config)
    - inference_processor.export_strategy (de export_strategy config)
    - inference_processor.device (de device)
    - inference_processor.batch_size (de hyperparameters.batch_size)
    - inference_processor.mask_bands (de seg_params)
    """
    # Checkpoint e modelo
    checkpoint_path: str = MISSING
    device: str = "cuda:0"
    
    # Herda de train config (pl_model, hyperparameters, seg_params)
    pl_model: Any = MISSING
    hyperparameters: Any = MISSING
    seg_params: Optional[Any] = None
    
    # Leitor de imagens
    inference_image_reader: InferenceImageReaderConfig = field(
        default_factory=InferenceImageReaderConfig
    )
    
    # Processador de inferência
    inference_processor: InferenceProcessorConfig = field(
        default_factory=InferenceProcessorConfig
    )
    
    # Polygonizer (opcional)
    polygonizer: Optional[PolygonizerConfig] = None
    
    # Export strategy (opcional)
    export_strategy: Optional[ExportStrategyConfig] = None
    
    # Parâmetros de inferência
    inference_threshold: float = 0.5
    save_inference: bool = True


# Registrar configurações com Hydra
cs = ConfigStore.instance()
cs.store(name="predict_single_image_config", node=PredictSingleImageConfig)
cs.store(name="inference_image_reader", node=InferenceImageReaderConfig)
cs.store(name="inference_processor_single", node=InferenceProcessorConfig)
cs.store(name="polygonizer", node=PolygonizerConfig)
cs.store(name="export_strategy", node=ExportStrategyConfig)