import os
import json
from pathlib import Path
from typing import Dict, List
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import rasterio
from tqdm import tqdm
from omegaconf import DictConfig, OmegaConf
from hydra.utils import instantiate

from pytorch_segmentation_models_trainer.tools.logging_config import logger


class MetricsCalculator:
    """
    Calcula métricas de segmentação comparando predições com ground truth.
    
    Suporta matching flexível de arquivos por nome.
    """
    
    def __init__(self, pipeline_config: DictConfig, experiment_config: DictConfig):
        """
        Args:
            pipeline_config: Config do pipeline (métricas, output, etc.)
            experiment_config: Config do experimento específico
        """
        self.config = pipeline_config
        self.experiment_config = experiment_config
        
        # Extrair configurações de métricas
        self.num_classes = self.config.metrics.num_classes
        self.class_names = self.config.metrics.class_names
        
        # Instanciar métricas
        self.metrics = self._instantiate_metrics()
        
        # Confusion Matrix
        self.confusion_matrix = self._instantiate_confusion_matrix()
        
        logger.info(f"MetricsCalculator initialized")
        logger.info(f"  Num classes: {self.num_classes}")
        logger.info(f"  Class names: {self.class_names}")
        logger.info(f"  Num metrics: {len(self.metrics)}")
    
    def _instantiate_confusion_matrix(self):
        """Instancia ConfusionMatrix do torchmetrics."""
        from torchmetrics import ConfusionMatrix
        
        return ConfusionMatrix(
            task='multiclass',
            num_classes=self.num_classes
        )
    
    def _instantiate_metrics(self) -> List:
        """
        Instancia todas as métricas da config.
        
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
        
        # 2. Construir mapeamento de predições disponíveis
        prediction_map = self._build_prediction_map(predictions_folder)
        logger.info(f"Found {len(prediction_map)} prediction files")
        
        # 3. Inicializar estruturas
        per_image_results = []
        
        # Reset metrics
        for metric in self.metrics:
            metric.reset()
        self.confusion_matrix.reset()
        
        # 4. Iterar sobre imagens
        successful = 0
        failed = 0
        not_found = 0
        
        for idx, row in tqdm(
            gt_df.iterrows(), 
            total=len(gt_df),
            desc=f"Evaluating {experiment_name}",
            unit="img"
        ):
            try:
                # Carregar predição e ground truth
                pred_mask, pred_path = self._load_prediction_flexible(
                    predictions_folder, 
                    row,
                    prediction_map
                )
                
                if pred_mask is None:
                    logger.warning(
                        f"Prediction not found for {row['image']}. "
                        f"Tried multiple patterns. Skipping."
                    )
                    not_found += 1
                    continue
                
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
                image_metrics['prediction_file'] = pred_path  # Adicionar info do arquivo usado
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
        
        logger.info(
            f"Processing completed: {successful} successful, "
            f"{failed} failed, {not_found} not found"
        )
        
        if successful == 0:
            raise ValueError(
                "No images were successfully processed! Check logs for errors."
            )
        
        # 5. Computar métricas finais
        logger.info("Computing aggregated metrics...")
        aggregated_metrics = self._compute_aggregated_metrics()
        
        logger.info("Computing confusion matrix...")
        confusion_mat = self.confusion_matrix.compute().cpu().numpy()
        
        # 6. Preparar diretório de saída
        output_dir = self._get_output_dir(experiment_name)
        
        # 7. Salvar resultados
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
    
    def _build_prediction_map(self, predictions_folder: str) -> Dict[str, str]:
        """
        Constrói mapeamento de stems de arquivo para paths completos.
        
        Isso permite matching flexível de arquivos de predição.
        
        Args:
            predictions_folder: pasta com predições
            
        Returns:
            Dict mapeando stem -> full_path
        """
        prediction_map = {}
        
        # Buscar todos arquivos TIF
        pred_folder_path = Path(predictions_folder)
        
        if not pred_folder_path.exists():
            logger.warning(f"Predictions folder does not exist: {predictions_folder}")
            return prediction_map
        
        # Padrões de busca (em ordem de prioridade)
        patterns = [
            "*.tif",
            "*.tiff",
            "*.TIF",
            "*.TIFF"
        ]
        
        all_files = []
        for pattern in patterns:
            all_files.extend(pred_folder_path.glob(pattern))
        
        logger.debug(f"Found {len(all_files)} TIF files in predictions folder")
        
        # Construir mapeamento
        for file_path in all_files:
            # Obter stem (nome sem extensão)
            stem = file_path.stem
            
            # Tentar remover prefixos/sufixos comuns
            clean_stems = self._generate_stem_variants(stem)
            
            for clean_stem in clean_stems:
                if clean_stem not in prediction_map:
                    prediction_map[clean_stem] = str(file_path)
        
        logger.debug(f"Built prediction map with {len(prediction_map)} entries")
        
        return prediction_map
    
    def _generate_stem_variants(self, stem: str) -> List[str]:
        """
        Gera variantes do stem removendo prefixos/sufixos comuns.
        
        Exemplos:
            "seg_image001_output" -> ["seg_image001_output", "image001", "seg_image001"]
            "image001_pred" -> ["image001_pred", "image001"]
            "mi_001" -> ["mi_001"]
        
        Args:
            stem: nome do arquivo sem extensão
            
        Returns:
            Lista de variantes do stem
        """
        variants = [stem]  # Sempre incluir o original
        
        # Remover prefixos comuns
        prefixes_to_remove = ['seg_', 'pred_', 'output_', 'mask_']
        temp_stem = stem
        
        for prefix in prefixes_to_remove:
            if temp_stem.startswith(prefix):
                temp_stem = temp_stem[len(prefix):]
                variants.append(temp_stem)
                break
        
        # Remover sufixos comuns
        suffixes_to_remove = ['_output', '_pred', '_prediction', '_seg', '_mask']
        temp_stem = stem
        
        for suffix in suffixes_to_remove:
            if temp_stem.endswith(suffix):
                temp_stem = temp_stem[:-len(suffix)]
                if temp_stem not in variants:
                    variants.append(temp_stem)
                break
        
        # Tentar remover prefixo E sufixo
        temp_stem = stem
        for prefix in prefixes_to_remove:
            if temp_stem.startswith(prefix):
                temp_stem = temp_stem[len(prefix):]
                break
        
        for suffix in suffixes_to_remove:
            if temp_stem.endswith(suffix):
                temp_stem = temp_stem[:-len(suffix)]
                break
        
        if temp_stem not in variants:
            variants.append(temp_stem)
        
        return variants
    
    def _load_prediction_flexible(
        self, 
        predictions_folder: str, 
        row: pd.Series,
        prediction_map: Dict[str, str]
    ) -> tuple:
        """
        Carrega predição com matching flexível por nome.
        
        Tenta múltiplas estratégias para encontrar o arquivo:
        1. Match exato do stem
        2. Match após remover prefixos/sufixos comuns
        3. Match com variantes do ground truth
        
        Args:
            predictions_folder: pasta com predições
            row: linha do DataFrame com coluna 'image'
            prediction_map: mapeamento de stems para paths
            
        Returns:
            Tupla (mask_array, file_path) ou (None, None) se não encontrado
        """
        # Obter stem do ground truth
        gt_path = Path(row['image'])
        gt_stem = gt_path.stem
        
        # Gerar variantes do stem do ground truth
        gt_variants = self._generate_stem_variants(gt_stem)
        
        # Tentar encontrar match
        matched_path = None
        
        for variant in gt_variants:
            if variant in prediction_map:
                matched_path = prediction_map[variant]
                logger.debug(f"Matched {gt_stem} -> {Path(matched_path).name} (variant: {variant})")
                break
        
        if matched_path is None:
            # Fallback: tentar match direto por nome de arquivo
            pred_filename = gt_path.name
            direct_path = os.path.join(predictions_folder, pred_filename)
            
            if os.path.exists(direct_path):
                matched_path = direct_path
                logger.debug(f"Matched {gt_stem} -> {pred_filename} (direct match)")
        
        if matched_path is None:
            return None, None
        
        # Carregar arquivo
        try:
            with rasterio.open(matched_path) as src:
                pred_mask = src.read(1)  # Ler primeira banda
            
            return pred_mask, matched_path
            
        except Exception as e:
            logger.error(f"Failed to load prediction from {matched_path}: {e}")
            return None, None
    
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
        if hasattr(self.config.output, 'timestamp_folders') and self.config.output.timestamp_folders:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            base_dir = os.path.join(base_dir, timestamp)
        
        # Estrutura de pastas
        if hasattr(self.config.output, 'structure'):
            output_dir = os.path.join(
                base_dir,
                self.config.output.structure.experiments_folder,
                experiment_name,
                "metrics"
            )
        else:
            # Fallback para estrutura simples
            output_dir = os.path.join(
                base_dir,
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
        
        # Nome do arquivo
        if hasattr(self.config.output, 'files') and hasattr(self.config.output.files, 'per_image_metrics_pattern'):
            csv_filename = self.config.output.files.per_image_metrics_pattern.format(
                experiment_name=experiment_name
            )
        else:
            csv_filename = f"{experiment_name}_per_image_metrics.csv"
        
        per_image_csv = os.path.join(output_dir, csv_filename)
        per_image_df.to_csv(per_image_csv, index=False)
        logger.info(f"  Saved per-image metrics: {per_image_csv}")
        
        # 2. Aggregated metrics JSON
        aggregated_json = os.path.join(output_dir, "aggregated_metrics.json")
        with open(aggregated_json, 'w') as f:
            json.dump(aggregated_metrics, f, indent=2)
        logger.info(f"  Saved aggregated metrics: {aggregated_json}")
        
        # 3. Confusion matrix NPY
        if hasattr(self.config.output, 'files') and hasattr(self.config.output.files, 'confusion_matrix_data_pattern'):
            cm_filename = self.config.output.files.confusion_matrix_data_pattern.format(
                experiment_name=experiment_name
            )
        else:
            cm_filename = f"{experiment_name}_confusion_matrix.npy"
        
        confusion_npy = os.path.join(output_dir, cm_filename)
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
