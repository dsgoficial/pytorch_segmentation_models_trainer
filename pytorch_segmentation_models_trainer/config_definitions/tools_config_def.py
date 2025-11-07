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
from typing import Any, List, Optional
from omegaconf import MISSING
from hydra.core.config_store import ConfigStore


# ============================================================================
# INFERENCE PROCESSORS
# ============================================================================

@dataclass
class SingleImageInfereceProcessorConfig:
    """
    Configuração para SingleImageInfereceProcessor.
    Processa uma imagem por vez usando tiles.
    
    Parâmetros injetados automaticamente:
    - model, device, batch_size, export_strategy, polygonizer
    """
    _target_: str = "pytorch_segmentation_models_trainer.tools.inference.inference_processors.SingleImageInfereceProcessor"
    model_input_shape: Optional[List[int]] = None  # Ex: [448, 448]
    step_shape: Optional[List[int]] = None  # Ex: [224, 224]
    mask_bands: int = 1
    config: Optional[Any] = None
    group_output_by_image_basename: bool = False


@dataclass
class SingleImageFromFrameFieldProcessorConfig:
    """
    Configuração para SingleImageFromFrameFieldProcessor.
    Específico para Frame Field models (seg + crossfield).
    """
    _target_: str = "pytorch_segmentation_models_trainer.tools.inference.inference_processors.SingleImageFromFrameFieldProcessor"
    model_input_shape: Optional[List[int]] = None
    step_shape: Optional[List[int]] = None
    mask_bands: int = 1
    config: Optional[Any] = None
    group_output_by_image_basename: bool = False


@dataclass
class MultiClassInferenceProcessorConfig:
    """
    Configuração para MultiClassInferenceProcessor.
    Para segmentação multi-classe com argmax.
    """
    _target_: str = "pytorch_segmentation_models_trainer.tools.inference.inference_processors.MultiClassInferenceProcessor"
    model_input_shape: Optional[List[int]] = None
    step_shape: Optional[List[int]] = None
    num_classes: int = MISSING
    config: Optional[Any] = None
    group_output_by_image_basename: bool = False


@dataclass
class ObjectDetectionInferenceProcessorConfig:
    """
    Configuração para ObjectDetectionInferenceProcessor.
    """
    _target_: str = "pytorch_segmentation_models_trainer.tools.inference.inference_processors.ObjectDetectionInferenceProcessor"
    model_input_shape: Optional[List[int]] = None
    config: Optional[Any] = None


@dataclass
class PolygonRNNInferenceProcessorConfig:
    """
    Configuração para PolygonRNNInferenceProcessor.
    """
    _target_: str = "pytorch_segmentation_models_trainer.tools.inference.inference_processors.PolygonRNNInferenceProcessor"
    sequence_length: int = 60
    config: Optional[Any] = None
    group_output_by_image_basename: bool = False


# ============================================================================
# DATA HANDLERS - RASTER READERS
# ============================================================================

@dataclass
class SingleImageReaderProcessorConfig:
    """
    Configuração para leitura de uma única imagem.
    """
    _target_: str = "pytorch_segmentation_models_trainer.tools.data_handlers.raster_reader.SingleImageReaderProcessor"
    file_name: str = MISSING


@dataclass
class FolderImageReaderProcessorConfig:
    """
    Configuração para leitura de pasta de imagens.
    """
    _target_: str = "pytorch_segmentation_models_trainer.tools.data_handlers.raster_reader.FolderImageReaderProcessor"
    folder_name: str = MISSING
    recursive: bool = True
    image_extension: str = "tif"


@dataclass
class CSVImageReaderProcessorConfig:
    """
    Configuração para leitura de imagens via CSV.
    """
    _target_: str = "pytorch_segmentation_models_trainer.tools.data_handlers.raster_reader.CSVImageReaderProcessor"
    input_csv_path: str = MISSING
    key: str = "image"
    root_dir: Optional[str] = None
    n_first_rows_to_read: Optional[int] = None


# ============================================================================
# DATA HANDLERS - DATA WRITERS
# ============================================================================

@dataclass
class VectorFileDataWriterConfig:
    """
    Configuração para escrita de vetores em arquivo.
    """
    _target_: str = "pytorch_segmentation_models_trainer.tools.data_handlers.data_writer.VectorFileDataWriter"
    output_file_folder: str = MISSING
    output_file_name: str = "output.geojson"
    driver: str = "GeoJSON"
    mode: str = "w"


@dataclass
class RasterDataWriterConfig:
    """
    Configuração para escrita de raster.
    """
    _target_: str = "pytorch_segmentation_models_trainer.tools.data_handlers.data_writer.RasterDataWriter"
    output_file_path: str = MISSING
    output_profile: Optional[Any] = None


@dataclass
class VectorDatabaseDataWriterConfig:
    """
    Configuração para escrita de vetores em banco de dados.
    """
    _target_: str = "pytorch_segmentation_models_trainer.tools.data_handlers.data_writer.VectorDatabaseDataWriter"
    user: str = MISSING
    password: str = MISSING
    database: str = MISSING
    sql: str = MISSING
    host: str = "localhost"
    port: int = 5432
    table_name: str = "buildings"
    geometry_column: str = "geom"


# ============================================================================
# EXPORT STRATEGIES
# ============================================================================

@dataclass
class RasterExportInferenceStrategyConfig:
    """
    Configuração para exportação de inferência como raster único.
    """
    _target_: str = "pytorch_segmentation_models_trainer.tools.inference.export_inference.RasterExportInferenceStrategy"
    output_file_path: str = MISSING
    output_profile: Optional[Any] = None


@dataclass
class MultipleRasterExportInferenceStrategyConfig:
    """
    Configuração para exportação de múltiplos rasters (ex: seg, crossfield).
    """
    _target_: str = "pytorch_segmentation_models_trainer.tools.inference.export_inference.MultipleRasterExportInferenceStrategy"
    output_folder: str = MISSING
    output_basename: str = "inference.tif"
    output_profile: Optional[Any] = None


@dataclass
class VectorFileExportInferenceStrategyConfig:
    """
    Configuração para exportação de inferência como vetor.
    """
    _target_: str = "pytorch_segmentation_models_trainer.tools.inference.export_inference.VectorFileExportInferenceStrategy"
    output_file_path: str = MISSING
    driver: str = "GeoJSON"


@dataclass
class VectorDatabaseExportInferenceStrategyConfig:
    """
    Configuração para exportação de inferência para banco de dados.
    """
    _target_: str = "pytorch_segmentation_models_trainer.tools.inference.export_inference.VectorDatabaseExportInferenceStrategy"
    user: str = MISSING
    password: str = MISSING
    database: str = MISSING
    sql: str = MISSING
    host: str = "localhost"
    port: int = 5432
    table_name: str = "buildings"
    geometry_column: str = "geom"


@dataclass
class ObjectDetectionExportInferenceStrategyConfig:
    """
    Configuração para exportação de detecção de objetos.
    """
    _target_: str = "pytorch_segmentation_models_trainer.tools.inference.export_inference.ObjectDetectionExportInferenceStrategy"
    output_file_path: str = MISSING


# ============================================================================
# POLYGONIZERS
# ============================================================================

@dataclass
class InnerPolylinesParamsConfig:
    """
    Configuração de parâmetros para polylines internas.
    """
    enable: bool = False
    max_traces: int = 1000
    seed_threshold: float = 0.5
    low_threshold: float = 0.1
    min_width: int = 2
    max_width: int = 8
    step_size: int = 1


@dataclass
class SimplePolConfigConfig:
    """
    Configuração para SimplePolygonizer.
    """
    data_level: float = 0.5
    tolerance: float = 1.0
    seg_threshold: float = 0.5
    min_area: float = 10


@dataclass
class SimplePolygonizerProcessorConfig:
    """
    Configuração para SimplePolygonizerProcessor.
    """
    _target_: str = "pytorch_segmentation_models_trainer.tools.polygonization.polygonizer.SimplePolygonizerProcessor"
    config: SimplePolConfigConfig = field(default_factory=SimplePolConfigConfig)
    data_writer: Optional[Any] = None


@dataclass
class ACMConfigConfig:
    """
    Configuração para Active Contours Model (ACM).
    """
    indicator_add_edge: bool = False
    steps: int = 500
    data_level: float = 0.5
    data_coef: float = 0.1
    length_coef: float = 0.4
    crossfield_coef: float = 0.5
    poly_lr: float = 0.01
    warmup_iters: int = 100
    warmup_factor: float = 0.1
    device: str = "cpu"
    tolerance: float = 0.5
    seg_threshold: float = 0.5
    min_area: int = 1
    inner_polylines_params: InnerPolylinesParamsConfig = field(
        default_factory=InnerPolylinesParamsConfig
    )


@dataclass
class ACMPolygonizerProcessorConfig:
    """
    Configuração para ACMPolygonizerProcessor.
    """
    _target_: str = "pytorch_segmentation_models_trainer.tools.polygonization.polygonizer.ACMPolygonizerProcessor"
    config: ACMConfigConfig = field(default_factory=ACMConfigConfig)
    data_writer: Optional[Any] = None


@dataclass
class LossParamsCoefsConfig:
    """
    Configuração de coeficientes para ASM loss.
    """
    step_thresholds: List[int] = field(default_factory=lambda: [0, 100, 200, 300])
    data: List[float] = field(default_factory=lambda: [1.0, 0.1, 0.0, 0.0])
    crossfield: List[float] = field(default_factory=lambda: [0.0, 0.05, 0.0, 0.0])
    length: List[float] = field(default_factory=lambda: [0.1, 0.01, 0.0, 0.0])
    curvature: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0, 0.0])
    corner: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0, 0.0])
    junction: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0, 0.0])


@dataclass
class LossParamsConfig:
    """
    Configuração de parâmetros de loss para ASM.
    """
    coefs: LossParamsCoefsConfig = field(default_factory=LossParamsCoefsConfig)
    curvature_dissimilarity_threshold: int = 15
    corner_angles: List[int] = field(default_factory=lambda: [45, 90, 135])
    corner_angle_threshold: float = 22.5
    junction_angles: List[int] = field(default_factory=lambda: [0, 45, 90, 135])
    junction_angle_weights: List[float] = field(default_factory=lambda: [1, 0.01, 0.1, 0.01])
    junction_angle_threshold: float = 22.5


@dataclass
class ASMConfigConfig:
    """
    Configuração para Active Skeletons Model (ASM).
    """
    init_method: str = "skeleton"  # skeleton ou marching_squares
    data_level: float = 0.5
    loss_params: LossParamsConfig = field(default_factory=LossParamsConfig)
    lr: float = 0.001
    gamma: float = 0.0001
    device: str = "cpu"
    tolerance: float = 22
    seg_threshold: float = 0.5
    min_area: float = 12


@dataclass
class ASMPolygonizerProcessorConfig:
    """
    Configuração para ASMPolygonizerProcessor.
    """
    _target_: str = "pytorch_segmentation_models_trainer.tools.polygonization.polygonizer.ASMPolygonizerProcessor"
    config: ASMConfigConfig = field(default_factory=ASMConfigConfig)
    data_writer: Optional[Any] = None


@dataclass
class PolygonRNNConfigConfig:
    """
    Configuração para PolygonRNN polygonizer.
    """
    tolerance: float = 0.0
    grid_size: int = 28
    min_area: float = 10


@dataclass
class PolygonRNNPolygonizerProcessorConfig:
    """
    Configuração para PolygonRNNPolygonizerProcessor.
    """
    _target_: str = "pytorch_segmentation_models_trainer.tools.polygonization.polygonizer.PolygonRNNPolygonizerProcessor"
    config: PolygonRNNConfigConfig = field(default_factory=PolygonRNNConfigConfig)
    data_writer: Optional[Any] = None


# ============================================================================
# REGISTRAR CONFIGURAÇÕES COM HYDRA
# ============================================================================

cs = ConfigStore.instance()

# Inference Processors
cs.store(group="inference_processor", name="single_image", node=SingleImageInfereceProcessorConfig)
cs.store(group="inference_processor", name="frame_field", node=SingleImageFromFrameFieldProcessorConfig)
cs.store(group="inference_processor", name="multiclass", node=MultiClassInferenceProcessorConfig)
cs.store(group="inference_processor", name="object_detection", node=ObjectDetectionInferenceProcessorConfig)
cs.store(group="inference_processor", name="polygon_rnn", node=PolygonRNNInferenceProcessorConfig)

# Raster Readers
cs.store(group="image_reader", name="single", node=SingleImageReaderProcessorConfig)
cs.store(group="image_reader", name="folder", node=FolderImageReaderProcessorConfig)
cs.store(group="image_reader", name="csv", node=CSVImageReaderProcessorConfig)

# Data Writers
cs.store(group="data_writer", name="vector_file", node=VectorFileDataWriterConfig)
cs.store(group="data_writer", name="raster", node=RasterDataWriterConfig)
cs.store(group="data_writer", name="vector_database", node=VectorDatabaseDataWriterConfig)

# Export Strategies
cs.store(group="export_strategy", name="raster", node=RasterExportInferenceStrategyConfig)
cs.store(group="export_strategy", name="multiple_raster", node=MultipleRasterExportInferenceStrategyConfig)
cs.store(group="export_strategy", name="vector_file", node=VectorFileExportInferenceStrategyConfig)
cs.store(group="export_strategy", name="vector_database", node=VectorDatabaseExportInferenceStrategyConfig)
cs.store(group="export_strategy", name="object_detection", node=ObjectDetectionExportInferenceStrategyConfig)

# Polygonizers
cs.store(group="polygonizer", name="simple", node=SimplePolygonizerProcessorConfig)
cs.store(group="polygonizer", name="acm", node=ACMPolygonizerProcessorConfig)
cs.store(group="polygonizer", name="asm", node=ASMPolygonizerProcessorConfig)
cs.store(group="polygonizer", name="polygon_rnn", node=PolygonRNNPolygonizerProcessorConfig)

# Sub-configs
cs.store(name="simple_pol_config", node=SimplePolConfigConfig)
cs.store(name="acm_config", node=ACMConfigConfig)
cs.store(name="asm_config", node=ASMConfigConfig)
cs.store(name="polygon_rnn_config", node=PolygonRNNConfigConfig)
cs.store(name="inner_polylines_params", node=InnerPolylinesParamsConfig)
cs.store(name="loss_params", node=LossParamsConfig)
cs.store(name="loss_params_coefs", node=LossParamsCoefsConfig)