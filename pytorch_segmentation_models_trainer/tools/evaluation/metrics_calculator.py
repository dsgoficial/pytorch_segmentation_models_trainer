import os
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import rasterio
from rasterio.windows import Window, from_bounds
from rasterio.coords import BoundingBox
from tqdm import tqdm
from omegaconf import DictConfig, OmegaConf
from hydra.utils import instantiate
from pytorch_segmentation_models_trainer.tools.evaluation.image_processing_worker import (
    process_single_image_worker
)

import logging
logger = logging.getLogger(__name__)


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
        
        logger.info(f"MetricsCalculator initialized")
        logger.info(f"  Num classes: {self.num_classes}")
        logger.info(f"  Class names: {self.class_names}")
    
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
                # Converter para dict Python (resolve=True resolve todas as interpolações)
                metric_dict = OmegaConf.to_container(metric_cfg, resolve=True)
                
                # Substituir manualmente qualquer ${} que possa ter sobrado
                # Converter para string JSON e substituir
                import json
                metric_json = json.dumps(metric_dict)
                
                # Substituir interpolações conhecidas
                metric_json = metric_json.replace('${metrics.num_classes}', str(self.num_classes))
                metric_json = metric_json.replace('${hyperparameters.num_classes}', str(self.num_classes))
                
                # Recriar dict
                metric_dict = json.loads(metric_json)
                
                # Instanciar métrica diretamente do dict
                metric = instantiate(metric_dict)
                metrics_list.append(metric)
                
                logger.debug(
                    f"Instantiated metric {idx + 1}: {metric.__class__.__name__}"
                )
                
            except Exception as e:
                logger.error(
                    f"Failed to instantiate metric {idx}: {e}",
                    exc_info=True
                )
                # Continuar com próxima métrica
                continue
        
        if len(metrics_list) == 0:
            logger.warning("No metrics could be instantiated!")
        else:
            logger.info(f"Successfully instantiated {len(metrics_list)} metrics")
        
        return metrics_list
    
    def _get_spatial_overlap(self, pred_path: str, gt_path: str) -> tuple:
        """
        Calcula a área de overlap espacial entre dois rasters georreferenciados.
        
        Args:
            pred_path: Caminho da predição
            gt_path: Caminho do ground truth
            
        Returns:
            (pred_window, gt_window, matched_shape) ou None se não houver overlap
        """
        
        with rasterio.open(pred_path) as pred_src, rasterio.open(gt_path) as gt_src:
            # Obter bounds de cada raster
            pred_bounds = pred_src.bounds
            gt_bounds = gt_src.bounds
            
            # Calcular intersecção das bounding boxes
            intersection = BoundingBox(
                left=max(pred_bounds.left, gt_bounds.left),
                bottom=max(pred_bounds.bottom, gt_bounds.bottom),
                right=min(pred_bounds.right, gt_bounds.right),
                top=min(pred_bounds.top, gt_bounds.top)
            )
            
            # Verificar se há overlap
            if intersection.left >= intersection.right or intersection.bottom >= intersection.top:
                logger.error(f"No spatial overlap between {pred_path} and {gt_path}")
                return None
            
            # Converter bounds para windows em cada raster
            pred_window = from_bounds(
                intersection.left, intersection.bottom,
                intersection.right, intersection.top,
                pred_src.transform
            )
            
            gt_window = from_bounds(
                intersection.left, intersection.bottom,
                intersection.right, intersection.top,
                gt_src.transform
            )
            
            # Arredondar para pixels inteiros
            pred_window = Window(
                col_off=round(pred_window.col_off),
                row_off=round(pred_window.row_off),
                width=round(pred_window.width),
                height=round(pred_window.height)
            )
            
            gt_window = Window(
                col_off=round(gt_window.col_off),
                row_off=round(gt_window.row_off),
                width=round(gt_window.width),
                height=round(gt_window.height)
            )
            
            # A shape final deve ser a mesma (ou muito próxima)
            matched_shape = (int(pred_window.height), int(pred_window.width))
            
            # Ajustar se houver pequenas diferenças devido a arredondamento
            if abs(pred_window.height - gt_window.height) > 1 or \
               abs(pred_window.width - gt_window.width) > 1:
                logger.warning(
                    f"Window size mismatch after spatial alignment: "
                    f"pred={pred_window.height}x{pred_window.width}, "
                    f"gt={gt_window.height}x{gt_window.width}"
                )
                # Usar o menor tamanho
                matched_shape = (
                    min(int(pred_window.height), int(gt_window.height)),
                    min(int(pred_window.width), int(gt_window.width))
                )
                pred_window = Window(
                    pred_window.col_off, pred_window.row_off,
                    matched_shape[1], matched_shape[0]
                )
                gt_window = Window(
                    gt_window.col_off, gt_window.row_off,
                    matched_shape[1], matched_shape[0]
                )
            
            return pred_window, gt_window, matched_shape
    
    def calculate_metrics(
        self,
        predictions_folder: str,
        ground_truth_csv: str,
        experiment_name: str,
        parallel: bool = True,
        num_workers: int = None
    ) -> Dict:
        """
        Calcula todas as métricas para um experimento.
        
        Args:
            predictions_folder: pasta com predições (TIF com índices de classe)
            ground_truth_csv: CSV com colunas 'image' e 'mask'
            experiment_name: nome do experimento
            parallel: Se True, processa imagens em paralelo
            num_workers: Número de workers (None = usar CPU count)
            
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
        
        # 1. Carregar CSV e encontrar predições
        gt_df = pd.read_csv(ground_truth_csv)
        logger.info(f"Loaded {len(gt_df)} images from ground truth CSV")
        
        # 2. Encontrar arquivos de predição
        pred_files = self._find_prediction_files(predictions_folder)
        logger.info(f"Found {len(pred_files)} prediction files")
        
        # 3. Criar lista de tarefas (pares de pred/gt)
        tasks = self._create_tasks(gt_df, pred_files)
        logger.info(f"Created {len(tasks)} prediction-groundtruth pairs")
        
        if len(tasks) == 0:
            raise ValueError("No prediction-groundtruth pairs found!")
        
        # 4. Processar imagens (paralelo ou sequencial)
        if parallel and len(tasks) > 1:
            image_results = self._process_images_parallel(tasks, num_workers)
        else:
            image_results = self._process_images_sequential(tasks)
        
        # 5. Calcular métricas a partir dos resultados
        results_dict = self._compute_metrics_from_results(image_results, experiment_name)
        
        return results_dict
    
    def _find_prediction_files(self, predictions_folder: str) -> Dict[str, str]:
        """
        Encontra todos os arquivos de predição.
        
        Returns:
            Dict mapeando basename -> full_path
        """
        pred_folder = Path(predictions_folder)
        pred_files = {}
        
        # Buscar arquivos .tif, .tiff, .png
        for ext in ['*.tif', '*.tiff', '*.TIF', '*.TIFF']:
            for pred_path in pred_folder.glob(ext):
                basename = pred_path.stem
                pred_files[basename] = str(pred_path)
        
        return pred_files
    
    def _create_tasks(self, gt_df: pd.DataFrame, pred_files: Dict[str, str]) -> list:
        """
        Cria lista de tarefas (pares pred/gt) para processar.
        
        Returns:
            Lista de dicts com pred_path, gt_path, image_name, index
        """
        tasks = []
        
        for idx, row in gt_df.iterrows():
            gt_path = row['mask']
            image_name = Path(gt_path).stem
            
            # Encontrar predição correspondente
            pred_path = self._find_matching_prediction(image_name, pred_files)
            
            if pred_path:
                tasks.append({
                    'image_name': image_name,
                    'pred_path': pred_path,
                    'gt_path': gt_path,
                    'index': idx
                })
            else:
                logger.warning(f"No prediction found for {image_name}")
        
        return tasks
    
    def _find_matching_prediction(self, image_name: str, pred_files: Dict[str, str]) -> Optional[str]:
        """
        Encontra o arquivo de predição correspondente.
        
        Tenta diferentes estratégias de matching.
        """
        # Estratégia 1: Match exato
        if image_name in pred_files:
            return pred_files[image_name]
        
        # Estratégia 2: Remover prefixos comuns (mask_, gt_, etc.)
        clean_name = image_name.replace('mask_', '').replace('gt_', '')
        if clean_name in pred_files:
            return pred_files[clean_name]
        
        # Estratégia 3: Buscar por substring
        for pred_name, pred_path in pred_files.items():
            if clean_name in pred_name or pred_name in clean_name:
                return pred_path
        
        return None
    
    def _process_images_parallel(
        self, 
        tasks: list, 
        num_workers: int = None
    ) -> list:
        """
        Processa imagens em paralelo usando ThreadPoolExecutor.
        
        Args:
            tasks: Lista de tarefas (dicts com pred_path, gt_path, etc.)
            num_workers: Número de workers
            
        Returns:
            Lista de resultados (um dict por imagem processada)
        """
        if num_workers is None:
            num_workers = min(32, len(tasks), os.cpu_count() or 1)
        
        logger.info(f"Processing {len(tasks)} images with {num_workers} workers")
        
        results = []
        
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            # Submeter todas as tarefas
            future_to_task = {
                executor.submit(
                    process_single_image_worker,
                    task,
                    self.num_classes,
                    list(self.class_names)  # Converter para lista Python
                ): task
                for task in tasks
            }
            
            # Coletar resultados
            with tqdm(
                total=len(tasks), 
                desc=f"Evaluating {self.experiment_config.name}", 
                unit="img"
            ) as pbar:
                for future in as_completed(future_to_task):
                    task = future_to_task[future]
                    
                    try:
                        result = future.result()
                        if result is not None:
                            results.append(result)
                    except Exception as e:
                        logger.error(f"Failed to process {task['image_name']}: {e}")
                    
                    pbar.update(1)
        
        logger.info(
            f"Processing completed: {len(results)} successful, "
            f"{len(tasks) - len(results)} failed"
        )
        
        return results
    
    def _process_images_sequential(self, tasks: list) -> list:
        """
        Processa imagens sequencialmente (fallback).
        
        Args:
            tasks: Lista de tarefas
            
        Returns:
            Lista de resultados
        """
        logger.info(f"Processing {len(tasks)} images sequentially")
        
        results = []
        
        for task in tqdm(
            tasks, 
            desc=f"Evaluating {self.experiment_config.name}", 
            unit="img"
        ):
            try:
                # Importar função worker e executar localmente
                from pytorch_segmentation_models_trainer.tools.evaluation.image_processing_worker import (
                    process_single_image_worker
                )
                
                result = process_single_image_worker(
                    task,
                    self.num_classes,
                    list(self.class_names)
                )
                
                if result is not None:
                    results.append(result)
                    
            except Exception as e:
                logger.error(f"Failed to process {task['image_name']}: {e}")
        
        logger.info(
            f"Processing completed: {len(results)} successful, "
            f"{len(tasks) - len(results)} failed"
        )
        
        return results
    
    def _compute_metrics_from_results(
        self, 
        image_results: list, 
        experiment_name: str
    ) -> Dict:
        """
        Calcula métricas agregadas e por imagem a partir dos resultados.
        OTIMIZADO: Usa apenas confusion matrices, sem torchmetrics no loop.
        """
        logger.info("Computing metrics from results...")
        
        if len(image_results) == 0:
            raise ValueError("No images were successfully processed! Check logs for errors.")
        
        # Inicializar confusion matrix global acumuladora
        cm_global = np.zeros((self.num_classes, self.num_classes), dtype=np.int64)
        
        # Processar cada imagem
        per_image_metrics = []
        
        logger.info(f"Computing per-image metrics for {len(image_results)} images...")
        
        for result in tqdm(image_results, desc="Computing metrics", unit="img"):
            image_name = result['image_name']
            
            # Converter de volta para tensors
            pred_flat = torch.from_numpy(result['pred_flat']).long()
            gt_flat = torch.from_numpy(result['gt_flat']).long()
            
            # Calcular confusion matrix por imagem
            cm_per_image = self._compute_confusion_matrix_fast(pred_flat, gt_flat)
            
            # Derivar métricas da confusion matrix
            image_metrics = self._metrics_from_confusion_matrix(cm_per_image, image_name)
            per_image_metrics.append(image_metrics)
            
            # ACUMULAR confusion matrix global (soma simples!)
            cm_global += cm_per_image
        
        # Calcular métricas agregadas da confusion matrix global
        logger.info("Computing global metrics from accumulated confusion matrix...")
        
        # Derivar métricas agregadas da CM global
        aggregated_metrics = self._metrics_from_confusion_matrix_aggregated(cm_global)
        
        # Converter CM global para formato esperado
        confusion_matrix_np = cm_global
        
        # Criar DataFrame
        per_image_df = pd.DataFrame(per_image_metrics)
        
        # Preparar e salvar resultados
        output_dir = self._prepare_output_directory(experiment_name)
        
        logger.info("Saving results...")
        self._save_results(
            per_image_df,
            aggregated_metrics,
            confusion_matrix_np,
            output_dir,
            experiment_name
        )
        
        # Retornar estrutura completa
        return {
            'per_image': per_image_df,
            'aggregated': aggregated_metrics,
            'confusion_matrix': confusion_matrix_np,
            'num_classes': int(self.num_classes),
            'class_names': list(self.class_names),
            'output_dir': str(output_dir),
            'experiment_name': experiment_name
        }
    
    def _metrics_from_confusion_matrix_aggregated(self, cm: np.ndarray) -> Dict[str, float]:
        """
        Calcula métricas agregadas (sem nome de imagem) a partir da confusion matrix.
        
        Args:
            cm: Confusion matrix global [num_classes, num_classes]
            
        Returns:
            Dict com métricas agregadas (sem 'image_name' e sem métricas por classe)
        """
        # Calcular somas
        tp = np.diag(cm)  # True positives por classe
        fp = cm.sum(axis=0) - tp  # False positives por classe
        fn = cm.sum(axis=1) - tp  # False negatives por classe
        tn = cm.sum() - (tp + fp + fn)  # True negatives por classe
        
        # Evitar divisão por zero
        epsilon = 1e-10
        
        metrics = {}
        
        # 1. Accuracy (macro average)
        accuracy_per_class = (tp + tn) / (tp + tn + fp + fn + epsilon)
        metrics['Accuracy'] = float(np.mean(accuracy_per_class))
        
        # 2. IoU / Jaccard (macro average)
        iou_per_class = tp / (tp + fp + fn + epsilon)
        metrics['JaccardIndex'] = float(np.mean(iou_per_class))
        
        # 3. Dice (macro average)
        f1score_per_class = (2 * tp) / (2 * tp + fp + fn + epsilon)
        
        # 4. F1-Score (macro average)
        metrics['F1Score'] = float(np.mean(f1score_per_class))
        
        # 5. Precision (macro average)
        precision_per_class = tp / (tp + fp + epsilon)
        metrics['Precision'] = float(np.mean(precision_per_class))
        
        # 6. Recall (macro average)
        recall_per_class = tp / (tp + fn + epsilon)
        metrics['Recall'] = float(np.mean(recall_per_class))
        
        # Adicionar métricas por classe
        for i, class_name in enumerate(self.class_names):
            metrics[f'IoU_{class_name}'] = float(iou_per_class[i])
            metrics[f'F1_{class_name}'] = float(f1score_per_class[i])
            metrics[f'Precision_{class_name}'] = float(precision_per_class[i])
            metrics[f'Recall_{class_name}'] = float(recall_per_class[i])
            metrics[f'Accuracy_{class_name}'] = float(accuracy_per_class[i])
        
        return metrics
    
    def _compute_confusion_matrix_fast(
        self, 
        pred: torch.Tensor, 
        gt: torch.Tensor
    ) -> np.ndarray:
        """
        Calcula confusion matrix de forma rápida usando bincount.
        Muito mais rápido que torchmetrics para uma única imagem.
        
        Args:
            pred: Tensor 1D com predições [N]
            gt: Tensor 1D com ground truth [N]
            
        Returns:
            Confusion matrix [num_classes, num_classes]
        """
        # Validar entrada
        assert pred.dim() == 1 and gt.dim() == 1
        assert pred.shape == gt.shape
        
        # Criar índices combinados: gt * num_classes + pred
        indices = gt * self.num_classes + pred
        
        # Usar bincount para contar (muito rápido!)
        cm_flat = torch.bincount(
            indices, 
            minlength=self.num_classes ** 2
        )
        
        # Reshape para matriz
        cm = cm_flat.reshape(self.num_classes, self.num_classes)
        
        return cm.cpu().numpy()


    def _metrics_from_confusion_matrix(
        self, 
        cm: np.ndarray, 
        image_name: str
    ) -> Dict[str, float]:
        """
        Calcula todas as métricas a partir da confusion matrix.
        Muito mais rápido que usar torchmetrics.
        
        Args:
            cm: Confusion matrix [num_classes, num_classes]
            image_name: Nome da imagem
            
        Returns:
            Dict com todas as métricas
        """
        metrics = {'image_name': image_name}
        
        # Calcular somas
        tp = np.diag(cm)  # True positives por classe
        fp = cm.sum(axis=0) - tp  # False positives por classe
        fn = cm.sum(axis=1) - tp  # False negatives por classe
        tn = cm.sum() - (tp + fp + fn)  # True negatives por classe
        
        # Evitar divisão por zero
        epsilon = 1e-10
        
        # 1. Accuracy (macro)
        accuracy_per_class = (tp + tn) / (tp + tn + fp + fn + epsilon)
        metrics['Accuracy'] = float(np.mean(accuracy_per_class))
        
        # 2. IoU / Jaccard (macro)
        iou_per_class = tp / (tp + fp + fn + epsilon)
        metrics['JaccardIndex'] = float(np.mean(iou_per_class))
        
        # 3. Dice (macro)
        f1score_per_class = (2 * tp) / (2 * tp + fp + fn + epsilon)
        
        # 4. F1-Score (macro)
        metrics['F1Score'] = float(np.mean(f1score_per_class))
        
        # 5. Precision (macro)
        precision_per_class = tp / (tp + fp + epsilon)
        metrics['Precision'] = float(np.mean(precision_per_class))
        
        # 6. Recall (macro)
        recall_per_class = tp / (tp + fn + epsilon)
        metrics['Recall'] = float(np.mean(recall_per_class))
        
        # Adicionar métricas por classe (opcional, mas útil)
        for i, class_name in enumerate(self.class_names):
            metrics[f'IoU_{class_name}'] = float(iou_per_class[i])
            metrics[f'F1_{class_name}'] = float(f1score_per_class[i])
            metrics[f'Precision_{class_name}'] = float(precision_per_class[i])
            metrics[f'Recall_{class_name}'] = float(recall_per_class[i])
        
        return metrics
    
    def _prepare_output_directory(self, experiment_name: str) -> Path:
        """Cria estrutura de diretórios para salvar resultados."""
        # Base output directory
        base_dir = Path(self.config.output.base_dir)
        
        # Experiment-specific directory
        exp_dir = base_dir / self.config.output.structure.experiments_folder / experiment_name / "metrics"
        exp_dir.mkdir(parents=True, exist_ok=True)
        
        return exp_dir
    
    def _save_results(
        self,
        per_image_df: pd.DataFrame,
        aggregated_metrics: Dict,
        confusion_matrix_np: np.ndarray,
        output_dir: Path,
        experiment_name: str
    ):
        """Salva todos os resultados em arquivos."""
        
        # 1. Salvar métricas por imagem (CSV)
        per_image_file = output_dir / self.config.output.files.per_image_metrics_pattern.format(
            experiment_name=experiment_name
        )
        per_image_df.to_csv(per_image_file, index=False)
        logger.info(f"  Saved per-image metrics: {per_image_file}")
        
        # 2. Salvar métricas agregadas (JSON)
        aggregated_file = output_dir / "aggregated_metrics.json"
        with open(aggregated_file, 'w') as f:
            json.dump(aggregated_metrics, f, indent=2)
        logger.info(f"  Saved aggregated metrics: {aggregated_file}")
        
        # 3. Salvar confusion matrix (NumPy)
        cm_file = output_dir / self.config.output.files.confusion_matrix_data_pattern.format(
            experiment_name=experiment_name
        )
        np.save(cm_file, confusion_matrix_np)
        logger.info(f"  Saved confusion matrix: {cm_file}")
    
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
        pred_mask: torch.Tensor, 
        gt_mask: torch.Tensor,
    ) -> Dict[str, float]:
        """
        Calcula métricas para um par predição/ground-truth.
        
        Args:
            pred_mask: Máscara predita (H, W) com valores 0 até num_classes-1
            gt_mask: Máscara ground truth (H, W) com valores 0 até num_classes-1
            
        Returns:
            Dict com nome_métrica -> valor
        """
        # Converter para tensors PyTorch
        # CRÍTICO: garantir tipo inteiro e valores válidos
        pred_tensor = pred_mask.long()  # Garantir long (int64)
        gt_tensor = gt_mask.long()
        
        # Validar valores
        # Clipar valores para estar no range válido [0, num_classes-1]
        pred_tensor = torch.clamp(pred_tensor, 0, self.num_classes - 1)
        gt_tensor = torch.clamp(gt_tensor, 0, self.num_classes - 1)
        
        # Flatten para 1D
        pred_flat = pred_tensor.flatten()
        gt_flat = gt_tensor.flatten()
        
        # Verificações de sanidade
        assert pred_flat.dim() == 1, f"pred_flat deve ser 1D, mas é {pred_flat.dim()}D"
        assert gt_flat.dim() == 1, f"gt_flat deve ser 1D, mas é {gt_flat.dim()}D"
        assert pred_flat.dtype in [torch.int32, torch.int64], f"pred_flat deve ser int, mas é {pred_flat.dtype}"
        assert gt_flat.dtype in [torch.int32, torch.int64], f"gt_flat deve ser int, mas é {gt_flat.dtype}"
        assert pred_flat.min() >= 0, f"pred_flat tem valores negativos: {pred_flat.min()}"
        assert gt_flat.min() >= 0, f"gt_flat tem valores negativos: {gt_flat.min()}"
        assert pred_flat.max() < self.num_classes, f"pred_flat tem valores >= num_classes: {pred_flat.max()}"
        assert gt_flat.max() < self.num_classes, f"gt_flat tem valores >= num_classes: {gt_flat.max()}"
        
        # Calcular métricas
        metrics_dict = {}
    
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

def _process_single_image_worker(
    task: Dict,
    num_classes: int,
    class_names: List[str]
) -> Dict:
    """
    Worker function para processar uma imagem.
    Executado em processo separado.
    
    Args:
        task: Dict com pred_path, gt_path, image_name
        num_classes: Número de classes
        class_names: Lista de nomes das classes
        
    Returns:
        Dict com resultados ou None se falhar
    """
    import numpy as np
    import torch
    import rasterio
    from rasterio.windows import Window, from_bounds
    from rasterio.coords import BoundingBox
    
    try:
        pred_path = task['pred_path']
        gt_path = task['gt_path']
        image_name = task['image_name']
        
        # 1. Ler e alinhar rasters (copiar código de _read_aligned_rasters)
        pred_mask, gt_mask = _read_aligned_rasters_worker(
            pred_path, gt_path, num_classes
        )
        
        if pred_mask is None or gt_mask is None:
            return None
        
        # 2. Calcular métricas (versão simplificada sem torchmetrics)
        # Apenas retornar os dados para cálculo posterior
        pred_tensor = torch.from_numpy(pred_mask).long().flatten()
        gt_tensor = torch.from_numpy(gt_mask).long().flatten()
        
        return {
            'image_name': image_name,
            'pred_flat': pred_tensor.numpy(),
            'gt_flat': gt_tensor.numpy()
        }
        
    except Exception as e:
        import logging
        logger = logging.getLogger(__name__)
        logger.error(f"Worker error for {task.get('image_name', 'unknown')}: {e}")
        return None


def _read_aligned_rasters_worker(
    pred_path: str,
    gt_path: str,
    num_classes: int
) -> tuple:
    """
    Lê dois rasters garantindo que as áreas lidas correspondem espacialmente.
    
    Args:
        pred_path: Caminho da predição
        gt_path: Caminho do ground truth
        
    Returns:
        (pred_array, gt_array) com mesma shape, ou (None, None) se falhar
    """
    try:
        # Calcular overlap espacial
        overlap_result = self._get_spatial_overlap(pred_path, gt_path)
        
        if overlap_result is None:
            return None, None
        
        pred_window, gt_window, expected_shape = overlap_result
        
        # Ler apenas a área de overlap de cada raster
        with rasterio.open(pred_path) as pred_src:
            pred_array = pred_src.read(1, window=pred_window)
        
        with rasterio.open(gt_path) as gt_src:
            gt_array = gt_src.read(1, window=gt_window)
        
        # Verificação final de shapes
        if pred_array.shape != gt_array.shape:
            logger.warning(
                f"Shape mismatch after spatial alignment: "
                f"pred={pred_array.shape}, gt={gt_array.shape}. "
                f"Cropping to common size."
            )
            # Fallback: crop para o menor tamanho
            min_h = min(pred_array.shape[0], gt_array.shape[0])
            min_w = min(pred_array.shape[1], gt_array.shape[1])
            pred_array = pred_array[:min_h, :min_w]
            gt_array = gt_array[:min_h, :min_w]
        
        logger.debug(
            f"Aligned rasters: shape={pred_array.shape}, "
            f"pred_window={pred_window}, gt_window={gt_window}"
        )
        
        # Clipar valores para o range válido
        pred_array = np.clip(pred_array, 0, self.num_classes - 1)
        gt_array = np.clip(gt_array, 0, self.num_classes - 1)
        
        # Verificar se há NaN ou valores inválidos
        if np.isnan(pred_array).any():
            logger.warning(f"NaN values found in prediction {pred_path}, replacing with 0")
            pred_array = np.nan_to_num(pred_array, nan=0.0).astype(np.int32)
        
        if np.isnan(gt_array).any():
            logger.warning(f"NaN values found in ground truth {gt_path}, replacing with 0")
            gt_array = np.nan_to_num(gt_array, nan=0.0).astype(np.int32)
        
        # Log de estatísticas para debug
        logger.debug(
            f"Aligned rasters: shape={pred_array.shape}, "
            f"pred_range=[{pred_array.min()}, {pred_array.max()}], "
            f"gt_range=[{gt_array.min()}, {gt_array.max()}]"
        )
        
        return pred_array, gt_array
        
    except Exception as e:
        logger.error(f"Error aligning rasters: {e}", exc_info=True)
        return None, None
