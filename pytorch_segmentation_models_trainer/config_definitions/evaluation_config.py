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
from typing import Any, Dict, List, Optional
from omegaconf import MISSING
from hydra.core.config_store import ConfigStore


@dataclass
class ExperimentConfig:
    """
    Configuração de um experimento individual para avaliação.
    Usada pelo evaluation_pipeline.py.
    """
    name: str = MISSING
    predict_config: str = MISSING
    checkpoint_path: str = MISSING
    output_folder: str = MISSING
    
    # Opcionais
    overrides: Optional[Dict[str, Any]] = None
    precomputed_predictions_folder: Optional[str] = None


@dataclass
class BuildCSVFromFoldersConfig:
    """
    Configuração para construir CSV de pastas de imagens e máscaras.
    Usada pelo DatasetCSVBuilder.
    """
    enabled: bool = False
    images_folder: str = MISSING
    masks_folder: str = MISSING
    image_pattern: str = "*.tif"
    mask_pattern: str = "*.tif"
    # Campos validados pelo validate_evaluation_config.py
    matching_strategy: Optional[str] = None
    recursive: Optional[bool] = None


@dataclass
class EvaluationDatasetConfig:
    """
    Configuração do dataset para avaliação.
    Usada pelo evaluation_pipeline.py.
    """
    input_csv_path: str = MISSING
    
    # Construir CSV automaticamente
    build_csv_from_folders: BuildCSVFromFoldersConfig = field(
        default_factory=BuildCSVFromFoldersConfig
    )


@dataclass
class MetricsConfig:
    """
    Configuração de métricas de segmentação.
    Usada pelo MetricsCalculator.
    """
    num_classes: int = MISSING
    class_names: List[str] = MISSING
    
    # Lista de métricas (torchmetrics)
    segmentation_metrics: List[Any] = field(default_factory=list)


@dataclass
class OutputConfig:
    """
    Configuração de saída de resultados.
    Usada pelo evaluation_pipeline.py.
    """
    base_dir: str = "./evaluation_results"
    
    # Campos adicionais que podem ser usados
    structure: Optional[Any] = None
    files: Optional[Any] = None


@dataclass
class ComparisonPlotsConfig:
    """
    Configuração de gráficos de comparação.
    """
    enabled: bool = True


@dataclass
class VisualizationConfig:
    """
    Configuração de visualizações.
    Usada pelo evaluation_pipeline.py.
    """
    comparison_plots: ComparisonPlotsConfig = field(
        default_factory=ComparisonPlotsConfig
    )


@dataclass
class ParallelInferenceConfig:
    """
    Configuração para inferência paralela.
    Usada pelo GPUDistributor.
    """
    enabled: bool = False


@dataclass
class LoadPredictionsConfig:
    """
    Configuração para carregar predições existentes.
    Usada pelo evaluation_pipeline.py.
    """
    enabled: bool = False
    base_folder: Optional[str] = None


@dataclass
class PipelineOptionsConfig:
    """
    Opções gerais do pipeline de avaliação.
    Usada pelo evaluation_pipeline.py.
    """
    parallel_inference: ParallelInferenceConfig = field(
        default_factory=ParallelInferenceConfig
    )
    
    load_predictions_from_folder: LoadPredictionsConfig = field(
        default_factory=LoadPredictionsConfig
    )
    
    # Campos adicionais que podem ser usados
    skip_existing_predictions: Optional[bool] = None
    skip_existing_metrics: Optional[bool] = None


@dataclass
class EvaluationPipelineConfig:
    """
    Configuração completa do pipeline de avaliação.
    Usada pelo evaluate_experiments.py e evaluation_pipeline.py.
    """
    experiments: List[ExperimentConfig] = field(default_factory=list)
    
    evaluation_dataset: EvaluationDatasetConfig = field(
        default_factory=EvaluationDatasetConfig
    )
    
    metrics: MetricsConfig = field(default_factory=MetricsConfig)
    
    output: OutputConfig = field(default_factory=OutputConfig)
    
    visualization: VisualizationConfig = field(default_factory=VisualizationConfig)
    
    pipeline_options: PipelineOptionsConfig = field(
        default_factory=PipelineOptionsConfig
    )


# Registrar configurações com Hydra
cs = ConfigStore.instance()
cs.store(name="evaluation_pipeline_config", node=EvaluationPipelineConfig)
cs.store(name="experiment_config", node=ExperimentConfig)
cs.store(name="evaluation_dataset", node=EvaluationDatasetConfig)
cs.store(name="build_csv_from_folders_eval", node=BuildCSVFromFoldersConfig)
cs.store(name="metrics_config", node=MetricsConfig)
cs.store(name="output_config", node=OutputConfig)
cs.store(name="visualization_config", node=VisualizationConfig)
cs.store(name="pipeline_options", node=PipelineOptionsConfig)
cs.store(name="parallel_inference", node=ParallelInferenceConfig)
cs.store(name="load_predictions", node=LoadPredictionsConfig)
