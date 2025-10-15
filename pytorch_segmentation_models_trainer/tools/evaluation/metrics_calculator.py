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

import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import rasterio
import torch
import torchmetrics
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

logger = logging.getLogger(__name__)


class MetricsCalculator:
    """
    Calcula métricas de segmentação usando torchmetrics.
    
    Features:
    - Extrai automaticamente num_classes e class_names do training config
    - Suporta métricas customizadas via DictConfig
    - Calcula métricas por imagem e agregadas
    - Sempre calcula matriz de confusão
    - Lida com métricas que retornam arrays (average='none')
    - Salva resultados em múltiplos formatos (CSV, JSON, NPY)
    """
    
    def __init__(self, config: DictConfig, experiment_config: DictConfig):
        """
        Args:
            config: Config geral do pipeline
            experiment_config: Config específico do experimento
        """
        self.config = config
        self.experiment_config = experiment_config
        
        # Extrair informações de classes do training config
        self.num_classes, self.class_names = self._extract_class_info()
        
        # Instanciar métricas
        self.metrics = self._instantiate_metrics()
        
        # Matriz de confusão (sempre calculada)
        self.confusion_matrix = torchmetrics.ConfusionMatrix(
            task="multiclass",
            num_classes=self.num_classes
        )
        
        logger.info(f"MetricsCalculator initialized for {experiment_config.name}")
        logger.info(f"  Num classes: {self.num_classes}")
        logger.info(f"  Class names: {self.class_names}")
        logger.info(f"  Num metrics: {len(self.metrics)}")
    
    def _extract_class_info(self) -> Tuple[int, List[str]]:
        """
        Extrai num_classes e class_names do training config.
        
        Procura em:
        1. experiment_config.model.classes (via training config)
        2. experiment_config.class_definitions.names (via training config)
        
        Returns:
            Tupla (num_classes, class_names)
        """
        logger.info("Extracting class information from training config...")
        
        # Carregar training config referenciado no predict_config
        train_cfg = self._load_training_config()
        
        # Extrair num_classes
        num_classes = OmegaConf.select(train_cfg, "model.classes")
        if num_classes is None:
            raise ValueError(
                f"Could not find 'model.classes' in training config for experiment "
                f"'{self.experiment_config.name}'. "
                f"Make sure the predict config inherits from the training config using 'defaults'."
            )
        
        # Extrair class_names
        class_names = OmegaConf.select(train_cfg, "class_definitions.names")
        if class_names is None:
            logger.warning(
                f"Could not find 'class_definitions.names' in training config. "
                f"Using generic names: ['class_0', 'class_1', ...]"
            )
            class_names = [f"class_{i}" for i in range(num_classes)]
        else:
            # Converter para lista se for ListConfig
            class_names = list(class_names)
        
        # Validar consistência
        if len(class_names) != num_classes:
            logger.warning(
                f"Mismatch: model.classes={num_classes} but "
                f"class_definitions.names has {len(class_names)} names. "
                f"Will use first {num_classes} names or pad with generic names."
            )
            if len(class_names) < num_classes:
                # Pad com nomes genéricos
                class_names.extend([
                    f"class_{i}" for i in range(len(class_names), num_classes)
                ])
            else:
                # Truncar
                class_names = class_names[:num_classes]
        
        logger.info(f"Extracted: num_classes={num_classes}, class_names={class_names}")
        return num_classes, class_names
    
    def _load_training_config(self) -> DictConfig:
        """
        Carrega o training config referenciado no predict_config.
        
        O predict_config tem 'defaults' que inclui o training config:
        defaults:
          - train_resnet34_unet_6classes
        
        Returns:
            DictConfig com configuração completa
        """
        predict_cfg_path = self.experiment_config.predict_config
        
        if not os.path.exists(predict_cfg_path):
            raise FileNotFoundError(
                f"Predict config not found: {predict_cfg_path}"
            )
        
        predict_cfg_path = Path(predict_cfg_path).resolve()
        config_dir = str(predict_cfg_path.parent)
        config_name = predict_cfg_path.stem
        
        logger.debug(f"Loading config from: {config_dir}/{config_name}")
        
        # Usar Hydra para carregar com resolução de defaults
        with initialize_config_dir(
            config_dir=config_dir,
            version_base=None
        ):
            cfg = compose(
                config_name=config_name,
                overrides=[],
                return_hydra_config=False
            )
        
        return cfg
    
    def _instantiate_metrics(self) -> List[torchmetrics.Metric]:
        """
        Instancia métricas substituindo ${num_classes} por valor real.
        
        Returns:
            Lista de métricas torchmetrics instanciadas
        """
        logger.info("Instantiating metrics...")
        
        metrics_list = []
        
        for idx, metric_cfg in enumerate(self.config.metrics.segmentation_metrics):
            try:
                # Criar cópia para não modificar config original
                metric_cfg_yaml = OmegaConf.to_yaml(metric_cfg)
                metric_cfg_copy = OmegaConf.create(metric_cfg_yaml)
                
                # Substituir ${num_classes} por valor real
                # Primeiro criar um DictConfig com a variável
                resolver_cfg = OmegaConf.create({
                    "num_classes": self.num_classes
                })
                
                # Merge configs
                metric_cfg_resolved = OmegaConf.merge(
                    resolver_cfg,
                    metric_cfg_copy
                )
                
                # Resolver todas as interpolações
                OmegaConf.resolve(metric_cfg_resolved)
                
                # Instanciar métrica
                metric = instantiate(metric_cfg_resolved)
                metrics_list.append(metric)
                
                logger.debug(
                    f"Instantiated metric {idx + 1}: {metric.__class__.__name__}"
                )
                
            except Exception as e:
                logger.error(
                    f"Failed to instantiate metric {idx + 1}: {e}",
                    exc_info=True
                )
                raise
        
        logger.info(f"Successfully instantiated {len(metrics_list)} metrics")
        return metrics_list
    
    def calculate_metrics(
        self,
        predictions_folder: str,
        ground_truth_csv: str,
        experiment_name: str
    ) -> Dict:
        """
        Calcula todas as métricas para um experimento.
        
        Args:
            predictions_folder: pasta com predições (TIF com índices de classe)
            ground_truth_csv: CSV com colunas 'image' e 'mask'
            experiment_name: nome do experimento
            
        Returns:
            Dict com:
                - 'per_image': DataFrame com métricas por imagem
                - 'aggregated': Dict com métricas agregadas
                - 'confusion_matrix': np.ndarray com matriz de confusão
                - 'num_classes': int
                - 'class_names': List[str]
                - 'output_dir': str
        """
        logger.info("="*60)
        logger.info(f"Calculating metrics for: {experiment_name}")
        logger.info("="*60)
        
        # 1. Carregar ground truth CSV
        gt_df = pd.read_csv(ground_truth_csv)
        logger.info(f"Loaded {len(gt_df)} images from ground truth CSV")
        
        # 2. Inicializar estruturas
        per_image_results = []
        
        # Reset metrics
        for metric in self.metrics:
            metric.reset()
        self.confusion_matrix.reset()
        
        # 3. Iterar sobre imagens
        successful = 0
        failed = 0
        
        for idx, row in tqdm(
            gt_df.iterrows(), 
            total=len(gt_df),
            desc=f"Evaluating {experiment_name}",
            unit="img"
        ):
            try:
                # Carregar predição e ground truth
                pred_mask = self._load_prediction(predictions_folder, row)
                gt_mask = self._load_ground_truth(row)
                
                # Validar shapes
                if pred_mask.shape != gt_mask.shape:
                    logger.error(
                        f"Shape mismatch for {row['image']}: "
                        f"pred={pred_mask.shape}, gt={gt_mask.shape}. Skipping."
                    )
                    failed += 1
                    continue
                
                # Converter para tensors
                pred_tensor = torch.from_numpy(pred_mask).long()
                gt_tensor = torch.from_numpy(gt_mask).long()
                
                # Calcular métricas por imagem
                image_metrics = self._calculate_per_image_metrics(
                    pred_tensor, gt_tensor, row['image']
                )
                per_image_results.append(image_metrics)
                
                # Atualizar métricas agregadas
                self._update_aggregated_metrics(pred_tensor, gt_tensor)
                
                successful += 1
                
            except FileNotFoundError as e:
                logger.error(f"File not found: {e}")
                failed += 1
                continue
                
            except Exception as e:
                logger.error(
                    f"Error processing image {row['image']}: {e}",
                    exc_info=True
                )
                failed += 1
                continue
        
        logger.info(f"Processing completed: {successful} successful, {failed} failed")
        
        if successful == 0:
            raise ValueError(
                "No images were successfully processed! Check logs for errors."
            )
        
        # 4. Computar métricas finais
        logger.info("Computing aggregated metrics...")
        aggregated_metrics = self._compute_aggregated_metrics()
        
        logger.info("Computing confusion matrix...")
        confusion_mat = self.confusion_matrix.compute().cpu().numpy()
        
        # 5. Preparar diretório de saída
        output_dir = self._get_output_dir(experiment_name)
        
        # 6. Salvar resultados
        logger.info("Saving results...")
        self._save_results(
            per_image_results,
            aggregated_metrics,
            confusion_mat,
            experiment_name,
            output_dir
        )
        
        logger.info(f"Metrics calculation completed for {experiment_name}")
        logger.info(f"Results saved to: {output_dir}")
        
        return {
            'per_image': pd.DataFrame(per_image_results),
            'aggregated': aggregated_metrics,
            'confusion_matrix': confusion_mat,
            'num_classes': self.num_classes,
            'class_names': self.class_names,
            'output_dir': output_dir
        }
    
    def _load_prediction(
        self, 
        predictions_folder: str, 
        row: pd.Series
    ) -> np.ndarray:
        """
        Carrega predição de uma imagem.
        
        Predições são TIF com 1 banda, valores uint8 = índices de classe.
        Pattern esperado: seg_{image_stem}_output.tif
        
        Args:
            predictions_folder: pasta com predições
            row: linha do DataFrame com coluna 'image'
            
        Returns:
            np.ndarray [H, W] com índices de classe
        """
        image_stem = Path(row['image']).stem
        pred_filename = f"seg_{image_stem}_output.tif"
        pred_path = os.path.join(predictions_folder, pred_filename)
        
        if not os.path.exists(pred_path):
            raise FileNotFoundError(f"Prediction not found: {pred_path}")
        
        # Carregar com rasterio
        with rasterio.open(pred_path) as src:
            pred_mask = src.read(1)  # Ler primeira (única) banda
        
        return pred_mask
    
    def _load_ground_truth(self, row: pd.Series) -> np.ndarray:
        """
        Carrega ground truth de uma imagem.
        
        Ground truth é TIF com 1 banda, valores uint8 = índices de classe.
        
        Args:
            row: linha do DataFrame com coluna 'mask'
            
        Returns:
            np.ndarray [H, W] com índices de classe
        """
        mask_path = row['mask']
        
        if not os.path.exists(mask_path):
            raise FileNotFoundError(f"Ground truth not found: {mask_path}")
        
        # Carregar com rasterio
        with rasterio.open(mask_path) as src:
            gt_mask = src.read(1)  # Ler primeira banda
        
        return gt_mask
    
    def _calculate_per_image_metrics(
        self, 
        pred: torch.Tensor, 
        gt: torch.Tensor,
        image_name: str
    ) -> Dict:
        """
        Calcula métricas para uma única imagem.
        
        Args:
            pred: Tensor [H, W] com predições
            gt: Tensor [H, W] com ground truth
            image_name: nome da imagem
            
        Returns:
            Dict com métricas da imagem
        """
        results = {'image': image_name}
        
        for metric in self.metrics:
            # Clonar métrica para não afetar agregada
            metric_copy = metric.clone()
            
            # Flatten para métricas globais
            pred_flat = pred.flatten()
            gt_flat = gt.flatten()
            
            metric_copy.update(pred_flat, gt_flat)
            value = metric_copy.compute()
            
            # Lidar com métricas que retornam array (average='none')
            metric_name = metric.__class__.__name__
            
            if isinstance(value, torch.Tensor) and value.numel() > 1:
                # Métrica por classe
                for i, v in enumerate(value):
                    class_name = (
                        self.class_names[i] 
                        if i < len(self.class_names) 
                        else f"class_{i}"
                    )
                    results[f'{metric_name}_{class_name}'] = v.item()
            else:
                # Métrica global
                results[metric_name] = (
                    value.item() if isinstance(value, torch.Tensor) else value
                )
        
        return results
    
    def _update_aggregated_metrics(
        self, 
        pred: torch.Tensor, 
        gt: torch.Tensor
    ):
        """
        Atualiza métricas agregadas com dados de uma imagem.
        
        Args:
            pred: Tensor [H, W] com predições
            gt: Tensor [H, W] com ground truth
        """
        pred_flat = pred.flatten()
        gt_flat = gt.flatten()
        
        for metric in self.metrics:
            metric.update(pred_flat, gt_flat)
        
        self.confusion_matrix.update(pred_flat, gt_flat)
    
    def _compute_aggregated_metrics(self) -> Dict:
        """
        Computa métricas finais agregadas.
        
        Returns:
            Dict com métricas agregadas
        """
        results = {}
        
        for metric in self.metrics:
            value = metric.compute()
            metric_name = metric.__class__.__name__
            
            if isinstance(value, torch.Tensor) and value.numel() > 1:
                # Métrica por classe
                for i, v in enumerate(value):
                    class_name = (
                        self.class_names[i] 
                        if i < len(self.class_names) 
                        else f"class_{i}"
                    )
                    results[f'{metric_name}_{class_name}'] = v.item()
            else:
                # Métrica global
                results[metric_name] = (
                    value.item() if isinstance(value, torch.Tensor) else value
                )
        
        return results
    
    def _get_output_dir(self, experiment_name: str) -> str:
        """
        Retorna diretório de saída para o experimento.
        
        Args:
            experiment_name: nome do experimento
            
        Returns:
            Path do diretório de saída
        """
        base_dir = self.config.output.base_dir
        
        # Adicionar timestamp se configurado
        if self.config.output.timestamp_folders:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            base_dir = os.path.join(base_dir, timestamp)
        
        output_dir = os.path.join(
            base_dir,
            self.config.output.structure.experiments_folder,
            experiment_name,
            "metrics"
        )
        
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        return output_dir
    
    def _save_results(
        self,
        per_image_results: List[Dict],
        aggregated_metrics: Dict,
        confusion_mat: np.ndarray,
        experiment_name: str,
        output_dir: str
    ):
        """
        Salva resultados em arquivos.
        
        Args:
            per_image_results: lista de dicts com métricas por imagem
            aggregated_metrics: dict com métricas agregadas
            confusion_mat: matriz de confusão
            experiment_name: nome do experimento
            output_dir: diretório de saída
        """
        # 1. Per-image metrics CSV
        per_image_df = pd.DataFrame(per_image_results)
        per_image_csv = os.path.join(
            output_dir,
            self.config.output.files.per_image_metrics_pattern.format(
                experiment_name=experiment_name
            )
        )
        per_image_df.to_csv(per_image_csv, index=False)
        logger.info(f"  Saved per-image metrics: {per_image_csv}")
        
        # 2. Aggregated metrics JSON
        aggregated_json = os.path.join(output_dir, "aggregated_metrics.json")
        with open(aggregated_json, 'w') as f:
            json.dump(aggregated_metrics, f, indent=2)
        logger.info(f"  Saved aggregated metrics: {aggregated_json}")
        
        # 3. Confusion matrix NPY
        confusion_npy = os.path.join(
            output_dir,
            self.config.output.files.confusion_matrix_data_pattern.format(
                experiment_name=experiment_name
            )
        )
        np.save(confusion_npy, confusion_mat)
        logger.info(f"  Saved confusion matrix: {confusion_npy}")
        
        # 4. Metadata JSON
        metadata = {
            'experiment_name': experiment_name,
            'num_classes': self.num_classes,
            'class_names': self.class_names,
            'num_images_evaluated': len(per_image_results),
            'timestamp': datetime.now().isoformat()
        }
        metadata_json = os.path.join(output_dir, "metadata.json")
        with open(metadata_json, 'w') as f:
            json.dump(metadata, f, indent=2)
        logger.info(f"  Saved metadata: {metadata_json}")
