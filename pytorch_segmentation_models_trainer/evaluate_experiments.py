#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
/***************************************************************************
 pytorch_segmentation_models_trainer
                              -------------------
        begin                : 2025-10-15
        git sha              : $Format:%H$
        copyright            : (C) 2025 by Philipe Borba - Cartographic Engineer
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

import logging
import sys
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

# Adicionar diretório raiz ao path
sys.path.insert(0, str(Path(__file__).parent.parent))

from pytorch_segmentation_models_trainer.tools.evaluation.evaluation_pipeline import (
    EvaluationPipeline
)

logger = logging.getLogger(__name__)


def setup_logging(config: DictConfig):
    """
    Configura logging do pipeline.
    
    Args:
        config: DictConfig com configurações de logging
    """
    # Valores padrão
    log_level = logging.INFO
    save_to_file = False
    log_file = "evaluation.log"
    
    # Tentar obter configurações customizadas
    if hasattr(config, 'logging'):
        if hasattr(config.logging, 'level'):
            log_level = getattr(logging, config.logging.level.upper())
        if hasattr(config.logging, 'save_to_file'):
            save_to_file = config.logging.save_to_file
        if hasattr(config.logging, 'log_file'):
            log_file = config.logging.log_file
    
    # Configurar logger raiz
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Se salvar em arquivo
    if save_to_file:
        if hasattr(config, 'output') and hasattr(config.output, 'base_dir'):
            log_dir = Path(config.output.base_dir)
        else:
            log_dir = Path("./evaluation_outputs")
        
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file_path = log_dir / log_file
        
        file_handler = logging.FileHandler(log_file_path, mode='w')
        file_handler.setLevel(log_level)
        file_handler.setFormatter(
            logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
        )
        
        logging.getLogger().addHandler(file_handler)
        logger.info(f"Logging to file: {log_file_path}")


@hydra.main(
    version_base=None,
    config_path="../configs/evaluation",
    config_name="pipeline_config"
)
def evaluate(cfg: DictConfig):
    """
    Script principal para executar pipeline de avaliação.
    
    Este script:
    1. Carrega configuração com Hydra
    2. Prepara dataset (constrói CSV se necessário)
    3. Executa predições para cada experimento
    4. Calcula métricas
    5. Agrega resultados
    6. Gera visualizações
    
    Usage:
        # Usar config padrão
        python evaluate_experiments.py
        
        # Com overrides
        python evaluate_experiments.py \
            experiments[0].checkpoint_path=/new/path.ckpt \
            pipeline_options.parallel_inference.enabled=false
            
        # Construir CSV automaticamente
        python evaluate_experiments.py \
            evaluation_dataset.build_csv_from_folders.enabled=true \
            evaluation_dataset.build_csv_from_folders.images_folder=/data/images \
            evaluation_dataset.build_csv_from_folders.masks_folder=/data/masks
    """
    # Setup logging
    setup_logging(cfg)
    
    logger.info("="*80)
    logger.info("EVALUATION PIPELINE")
    logger.info("="*80)
    logger.info(f"\nConfiguration:\n{OmegaConf.to_yaml(cfg)}")
    
    try:
        # Criar e executar pipeline
        pipeline = EvaluationPipeline(cfg)
        results = pipeline.run()
        
        logger.info("\n" + "="*80)
        logger.info("PIPELINE COMPLETED SUCCESSFULLY")
        logger.info("="*80)
        
        return results
        
    except Exception as e:
        logger.error("\n" + "="*80)
        logger.error("PIPELINE FAILED")
        logger.error("="*80)
        logger.error(f"Error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    evaluate()
