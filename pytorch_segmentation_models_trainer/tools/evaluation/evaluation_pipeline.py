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
import multiprocessing as mp
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from pytorch_segmentation_models_trainer.tools.evaluation.csv_builder import (
    DatasetCSVBuilder
)
from pytorch_segmentation_models_trainer.tools.evaluation.gpu_distributor import (
    GPUDistributor
)
from pytorch_segmentation_models_trainer.tools.evaluation.metrics_calculator import (
    MetricsCalculator
)
from pytorch_segmentation_models_trainer.tools.visualization.confusion_matrix_plots import (
    ConfusionMatrixPlotter
)
from pytorch_segmentation_models_trainer.tools.visualization.comparison_plots import (
    ComparisonPlotter
)
from pytorch_segmentation_models_trainer.tools.evaluation.results_aggregator import (
    ResultsAggregator
)

logger = logging.getLogger(__name__)


class EvaluationPipeline:
    """
    Pipeline principal de avaliação de experimentos.
    
    Fluxo:
    1. Preparar dataset (construir CSV se necessário)
    2. Executar/Carregar predições para cada experimento
    3. Calcular métricas para cada experimento
    4. Agregar resultados
    5. Gerar visualizações (Fase 3)
    
    Suporta:
    - Múltiplos experimentos
    - Skip de predições/avaliações existentes
    - Carregamento de predições pré-computadas
    - Execução sequencial ou paralela (Fase 2)
    """
    
    def __init__(self, config: DictConfig):
        """
        Args:
            config: DictConfig do pipeline completo
        """
        self.config = config
        self.experiments = config.experiments
        
        # GPU Distributor (se paralelização habilitada)
        self.gpu_distributor = None
        if config.pipeline_options.parallel_inference.enabled:
            self.gpu_distributor = GPUDistributor(config)
        
        # Results Aggregator
        self.results_aggregator = ResultsAggregator(config)
        
        # Visualization (Fase 3)
        self.confusion_matrix_plotter = None
        self.comparison_plotter = None
        
        if config.visualization.comparison_plots.enabled:
            self.confusion_matrix_plotter = ConfusionMatrixPlotter(config)
            self.comparison_plotter = ComparisonPlotter(config)
        
        logger.info("EvaluationPipeline initialized")
        logger.info(f"Number of experiments: {len(self.experiments)}")
        logger.info(
            f"Parallel inference: "
            f"{'enabled' if config.pipeline_options.parallel_inference.enabled else 'disabled'}"
        )
        logger.info(
            f"Visualizations: "
            f"{'enabled' if config.visualization.comparison_plots.enabled else 'disabled'}"
        )
    
    def run(self) -> Dict:
        """
        Executa pipeline completo.
        
        Returns:
            Dict com todos os resultados agregados
        """
        logger.info("="*80)
        logger.info("EVALUATION PIPELINE STARTED")
        logger.info("="*80)
        
        start_time = time.time()
        
        try:
            # 1. Preparar dataset
            dataset_csv = self._prepare_dataset()
            
            # 2. Executar/Carregar predições para cada experimento
            predictions = self._run_predictions(dataset_csv)
            
            # 3. Calcular métricas para cada experimento
            all_results = self._evaluate_all_experiments(predictions, dataset_csv)
            
            # 4. Agregar resultados
            aggregated = self.results_aggregator.aggregate(all_results)
            
            # 5. Gerar visualizações (Fase 3)
            if self.config.visualization.comparison_plots.enabled:
                self._generate_visualizations(aggregated)
            
            # 6. Salvar relatório final
            self._save_summary_report(aggregated, time.time() - start_time)
            
            elapsed_time = time.time() - start_time
            logger.info("="*80)
            logger.info("PIPELINE COMPLETED SUCCESSFULLY")
            logger.info(f"Total time: {elapsed_time:.2f}s ({elapsed_time/60:.2f} minutes)")
            logger.info("="*80)
            
            return aggregated
            
        except Exception as e:
            logger.error(f"Pipeline failed: {e}", exc_info=True)
            raise
    
    def _prepare_dataset(self) -> str:
        """
        Prepara CSV do dataset.
        
        Suporta 3 modos:
        1. CSV existente (input_csv_path)
        2. Construir de pastas com imagens+máscaras (build_csv_from_folders)
        3. NOVO: Avaliação direta de pastas (direct_folder_evaluation)
        
        Returns:
            Path do CSV do dataset
        """
        logger.info("\n" + "="*60)
        logger.info("STEP 1: PREPARING DATASET")
        logger.info("="*60)
        
        # MODO 3: Direct folder evaluation (NOVO)
        if (hasattr(self.config.evaluation_dataset, 'direct_folder_evaluation') and
            self.config.evaluation_dataset.direct_folder_evaluation.enabled):
            
            logger.info("Using DIRECT FOLDER EVALUATION mode (no CSV needed)")
            
            from pytorch_segmentation_models_trainer.tools.evaluation.direct_folder_evaluator import (
                prepare_evaluation_csv_from_folders
            )
            
            config = self.config.evaluation_dataset.direct_folder_evaluation
            
            csv_path = os.path.join(
                self.config.output.base_dir,
                "direct_evaluation_dataset.csv"
            )
            
            # Criar pasta se não existe
            Path(csv_path).parent.mkdir(parents=True, exist_ok=True)
            
            # Verificar se deve regenerar
            if os.path.exists(csv_path) and not config.get('force_rebuild', False):
                logger.info(f"Using existing CSV: {csv_path}")
                return csv_path
            
            # Criar CSV diretamente das pastas
            prepare_evaluation_csv_from_folders(
                ground_truth_folder=config.ground_truth_folder,
                predictions_folder=config.predictions_folder if hasattr(config, 'predictions_folder') else None,
                output_csv_path=csv_path,
                gt_pattern=config.get('gt_pattern', '*.tif'),
                pred_pattern=config.get('pred_pattern', '*.tif')
            )
            
            logger.info(f"Dataset CSV built (direct mode): {csv_path}")
            return csv_path
        
        # MODO 2: Build from folders (ORIGINAL)
        if self.config.evaluation_dataset.build_csv_from_folders.enabled:
            logger.info("Building CSV from folders...")
            
            from pytorch_segmentation_models_trainer.tools.evaluation.csv_builder import (
                DatasetCSVBuilder
            )
            
            builder = DatasetCSVBuilder(
                self.config.evaluation_dataset.build_csv_from_folders
            )
            
            csv_path = os.path.join(
                self.config.output.base_dir, 
                "generated_dataset.csv"
            )
            
            # Criar pasta se não existe
            Path(csv_path).parent.mkdir(parents=True, exist_ok=True)
            
            builder.build_csv(csv_path)
            logger.info(f"Dataset CSV built: {csv_path}")
            
            return csv_path
        
        # MODO 1: CSV existente (ORIGINAL)
        else:
            csv_path = self.config.evaluation_dataset.input_csv_path
            
            if not os.path.exists(csv_path):
                raise FileNotFoundError(
                    f"Dataset CSV not found: {csv_path}. "
                    f"Either provide a valid input_csv_path, enable "
                    f"build_csv_from_folders, or use direct_folder_evaluation."
                )
            
            logger.info(f"Using existing dataset CSV: {csv_path}")
            return csv_path
    
    def _run_predictions(self, dataset_csv: str) -> Dict:
        """
        Executa ou carrega predições para todos experimentos.
        
        Se load_predictions_from_folder.enabled=True, carrega predições existentes.
        Se skip_existing_predictions=True, pula predições que já existem.
        Caso contrário, executa predições.
        
        Args:
            dataset_csv: path do CSV com dataset
            
        Returns:
            Dict mapeando experiment_name -> info sobre predições
        """
        logger.info("\n" + "="*60)
        logger.info("STEP 2: RUNNING/LOADING PREDICTIONS")
        logger.info("="*60)
        
        # NOVO: Verificar se deve carregar predições de pasta
        if (hasattr(self.config.pipeline_options, 'load_predictions_from_folder') and 
            self.config.pipeline_options.load_predictions_from_folder.enabled):
            return self._load_existing_predictions()
        
        # Comportamento original: executar predições
        if (self.config.pipeline_options.parallel_inference.enabled and 
            self.gpu_distributor is not None):
            return self._run_predictions_parallel(dataset_csv)
        else:
            return self._run_predictions_sequential(dataset_csv)
    
    def _load_existing_predictions(self) -> Dict:
        """
        Carrega predições já existentes de pastas especificadas.
        
        Suporta dois modos:
        1. Base folder centralizada (pipeline_options.load_predictions_from_folder.base_folder)
        2. Por experimento (experiment.precomputed_predictions_folder)
        
        Returns:
            Dict com info de predições carregadas
        """
        logger.info("Loading existing predictions from folders")
        
        predictions_info = {}
        base_folder = None
        
        # Verificar se há base_folder centralizada
        if (hasattr(self.config.pipeline_options.load_predictions_from_folder, 'base_folder') and
            self.config.pipeline_options.load_predictions_from_folder.base_folder):
            base_folder = self.config.pipeline_options.load_predictions_from_folder.base_folder
            logger.info(f"Using centralized base folder: {base_folder}")
        
        for exp in self.experiments:
            logger.info(f"\n--- Experiment: {exp.name} ---")
            
            # Determinar pasta de predições
            predictions_folder = None
            
            # 1. Verificar se experimento tem pasta específica
            if hasattr(exp, 'precomputed_predictions_folder') and exp.precomputed_predictions_folder:
                predictions_folder = exp.precomputed_predictions_folder
                logger.info(f"Using experiment-specific folder: {predictions_folder}")
            
            # 2. Usar base_folder + nome do experimento
            elif base_folder:
                predictions_folder = os.path.join(base_folder, exp.name)
                logger.info(f"Using base folder structure: {predictions_folder}")
            
            # 3. Usar output_folder padrão
            else:
                predictions_folder = exp.output_folder
                logger.info(f"Using default output folder: {predictions_folder}")
            
            # Validar pasta
            validation_result = self._validate_predictions_folder(
                predictions_folder, 
                exp.name
            )
            
            if validation_result['valid']:
                predictions_info[exp.name] = {
                    'output_folder': predictions_folder,
                    'skipped': False,
                    'loaded': True,
                    'gpu_id': -1,
                    'num_predictions': validation_result['num_files']
                }
                logger.info(
                    f"✓ Loaded {validation_result['num_files']} predictions "
                    f"for {exp.name}"
                )
            else:
                error_msg = (
                    f"✗ Invalid predictions folder for {exp.name}: "
                    f"{validation_result['error']}"
                )
                logger.error(error_msg)
                predictions_info[exp.name] = {
                    'output_folder': predictions_folder,
                    'skipped': False,
                    'loaded': False,
                    'gpu_id': -1,
                    'error': validation_result['error']
                }
        
        # Resumo
        loaded_count = sum(1 for p in predictions_info.values() if p.get('loaded', False))
        failed_count = sum(1 for p in predictions_info.values() if 'error' in p)
        logger.info(f"\nSummary: {loaded_count} experiments loaded, {failed_count} failed")
        
        return predictions_info
    
    def _validate_predictions_folder(
        self, 
        folder: str, 
        experiment_name: str
    ) -> Dict:
        """
        Valida se uma pasta contém predições válidas.
        
        ATUALIZADO: Aceita qualquer arquivo TIF, sem depender de padrões específicos.
        
        Args:
            folder: Pasta a validar
            experiment_name: Nome do experimento
            
        Returns:
            Dict com resultado da validação:
            {
                'valid': bool,
                'num_files': int,
                'error': str (se invalid)
            }
        """
        # Verificar se pasta existe
        if not os.path.exists(folder):
            return {
                'valid': False,
                'num_files': 0,
                'error': f"Folder does not exist: {folder}"
            }
        
        if not os.path.isdir(folder):
            return {
                'valid': False,
                'num_files': 0,
                'error': f"Path is not a directory: {folder}"
            }
        
        # Buscar TODOS os arquivos TIF (sem padrão específico)
        prediction_files = []
        
        # Padrões de extensão a buscar
        patterns = ["*.tif", "*.tiff", "*.TIF", "*.TIFF"]
        
        for pattern in patterns:
            prediction_files.extend(list(Path(folder).glob(pattern)))
        
        if len(prediction_files) == 0:
            return {
                'valid': False,
                'num_files': 0,
                'error': "No TIF files found in folder"
            }
        
        logger.debug(f"Found {len(prediction_files)} TIF files in {folder}")
        
        # Validação adicional: verificar se arquivos têm tamanho > 0
        empty_files = [f for f in prediction_files if os.path.getsize(f) == 0]
        if empty_files:
            logger.warning(f"Found {len(empty_files)} empty files in {folder}")
        
        # Contabilizar apenas arquivos não vazios
        valid_files = [f for f in prediction_files if os.path.getsize(f) > 0]
        
        if len(valid_files) == 0:
            return {
                'valid': False,
                'num_files': 0,
                'error': "All TIF files are empty (0 bytes)"
            }
        
        return {
            'valid': True,
            'num_files': len(valid_files),
            'error': None
        }
    
    def _run_predictions_sequential(self, dataset_csv: str) -> Dict:
        """
        Executa predições sequencialmente.
        
        Args:
            dataset_csv: path do CSV com dataset
            
        Returns:
            Dict com info de predições
        """
        logger.info("Running predictions SEQUENTIALLY")
        
        device_id = -1  # Default: CPU
        if self.gpu_distributor and len(self.gpu_distributor.available_gpus) > 0:
            device_id = self.gpu_distributor.available_gpus[0]
            logger.info(f"Using GPU {device_id} for sequential execution")
        else:
            logger.info("Using CPU for sequential execution")
        
        predictions_info = {}
        
        for exp in self.experiments:
            logger.info(f"\n--- Experiment: {exp.name} ---")
            
            # Verificar se deve skip
            if self._should_skip_prediction(exp):
                logger.info(f"Skipping prediction (already exists)")
                predictions_info[exp.name] = {
                    'output_folder': exp.output_folder,
                    'skipped': True,
                    'gpu_id': device_id,
                }
                continue
            
            # Executar predição
            try:
                self._run_single_prediction(exp, dataset_csv, gpu_id=-1)
                predictions_info[exp.name] = {
                    'output_folder': exp.output_folder,
                    'skipped': False,
                    'gpu_id': device_id
                }
                logger.info(f"Prediction completed for {exp.name}")
                
            except Exception as e:
                logger.error(
                    f"Failed to run prediction for {exp.name}: {e}",
                    exc_info=True
                )
                raise
        
        return predictions_info
    
    def _run_predictions_parallel(self, dataset_csv: str) -> Dict:
        """
        Executa predições em paralelo, distribuindo entre GPUs.
        
        Args:
            dataset_csv: path do CSV com dataset
            
        Returns:
            Dict com info de predições
        """
        logger.info("Running predictions IN PARALLEL")
        
        # Distribuir experimentos entre GPUs
        gpu_assignments = self.gpu_distributor.assign_experiments(self.experiments)
        
        logger.info(f"GPU assignments: {gpu_assignments}")
        
        # Preparar lista de tarefas
        tasks = []
        
        for gpu_id, experimentList in gpu_assignments.items():
            for exp in self.experiments:
                # Verificar se deve skip
                if self._should_skip_prediction(exp):
                    logger.info(f"Skipping {exp.name} (already exists)")
                    continue
                tasks.append((exp, gpu_id, dataset_csv, self.config))
        
        if len(tasks) == 0:
            logger.warning("No experiments to run (all skipped)")
            return {
                exp.name: {
                    'output_folder': exp.output_folder,
                    'skipped': True,
                    'gpu_id': -1
                }
                for exp in self.experiments
            }
        
        logger.info(f"Running {len(tasks)} predictions in parallel")
        
        # Executar em paralelo
        predictions_info = {}
        
        # Adicionar experimentos que foram skipados
        for exp in self.experiments:
            if exp.name not in [t[0].name for t in tasks]:
                predictions_info[exp.name] = {
                    'output_folder': exp.output_folder,
                    'skipped': True,
                    'gpu_id': -1
                }
        
        with ThreadPoolExecutor(max_workers=len(tasks)) as executor:
            # Submeter tarefas
            future_to_exp = {
                executor.submit(_run_prediction_worker, *task): task[0].name
                for task in tasks
            }
            
            # Coletar resultados com progress bar
            with tqdm(
                total=len(future_to_exp),
                desc="Parallel prediction",
                unit="exp",
                position=0,
                leave=True,
                dynamic_ncols=True,
            ) as pbar:
                for future in as_completed(future_to_exp):
                    exp_name = future_to_exp[future]
                    
                    try:
                        result = future.result()
                        predictions_info[exp_name] = result
                        logger.info(f"✓ {exp_name} completed on GPU {result['gpu_id']}")
                        
                    except Exception as e:
                        logger.error(f"✗ {exp_name} failed: {e}")
                        # Não falhar todo pipeline por um experimento
                        predictions_info[exp_name] = {
                            'output_folder': None,
                            'skipped': False,
                            'gpu_id': -1,
                            'error': str(e)
                        }
                    
                    pbar.update(1)
        
        return predictions_info
    
    def _should_skip_prediction(self, experiment: DictConfig) -> bool:
        """
        Verifica se deve pular predição de um experimento.
        
        Args:
            experiment: config do experimento
            
        Returns:
            True se deve pular
        """
        if not self.config.pipeline_options.skip_existing_predictions:
            return False
        
        output_folder = experiment.output_folder
        
        if not os.path.exists(output_folder):
            return False
        
        # Verificar se há pelo menos um arquivo de predição
        prediction_files = list(Path(output_folder).glob("*.tif"))
        
        if len(prediction_files) > 0:
            logger.info(
                f"Found {len(prediction_files)} prediction files in {output_folder}"
            )
            return True
        
        return False
    
    def _run_single_prediction(
        self, 
        experiment: DictConfig, 
        dataset_csv: str,
        gpu_id: int = -1
    ):
        """
        Executa predição para um único experimento.
        
        Chama o script predict.py como subprocesso.
        
        Args:
            experiment: config do experimento
            dataset_csv: path do CSV com dataset
            gpu_id: ID da GPU a usar (-1 para CPU)
        """
        logger.info(f"Running prediction for {experiment.name} on GPU {gpu_id}...")
        
        # Preparar overrides para Hydra
        overrides = [
            f"checkpoint_path={experiment.checkpoint_path}",
            # f"inference_dataset.input_csv_path={dataset_csv}",
        ]
        
        # Adicionar device override
        if gpu_id >= 0:
            overrides.append(f"device=cuda:{gpu_id}")
            overrides.append(f"pl_trainer.devices=[{gpu_id}]")
        else:
            overrides.append("device=cpu")
            overrides.append("pl_trainer.accelerator=cpu")
            overrides.append("pl_trainer.devices=1")
        
        # Adicionar overrides específicos do experimento
        if "overrides" in experiment and experiment.overrides:
            for key, value in experiment.overrides.items():
                overrides.append(f"{key}={value}")
        
        # Construir comando
        script_path = os.path.join(
            os.path.dirname(__file__),
            "..",
            "..",
            "predict.py"
        )
        
        if not os.path.exists(script_path):
            raise FileNotFoundError(
                f"predict.py not found at {script_path}"
            )
        
        cmd = [
            sys.executable,
            script_path,
            "--config-dir", str(Path(experiment.predict_config).parent),
            "--config-name", Path(experiment.predict_config).stem,
        ] + overrides
        
        logger.debug(f"Command: {' '.join(cmd)}")
        
        # Executar
        result = subprocess.run(cmd)
        
        if result.returncode != 0:
            logger.error(f"Prediction failed with return code {result.returncode}")
            logger.error(f"STDOUT:\n{result.stdout}")
            logger.error(f"STDERR:\n{result.stderr}")
            raise RuntimeError(
                f"Prediction failed for {experiment.name}. "
                f"Check logs for details."
            )
        
        logger.info(f"Prediction completed successfully for {experiment.name}")
    
    def _evaluate_all_experiments_sequential(self, predictions: Dict, dataset_csv: str) -> Dict:
        """
        Calcula métricas sequencialmente (um experimento por vez).
        Mas cada experimento processa imagens em paralelo.
        """
        logger.info("Evaluating experiments SEQUENTIALLY (images in parallel)")
        
        all_results = {}
        
        # Obter configuração de paralelização por imagem
        parallel_config = self.config.pipeline_options.get('parallel_image_processing', {})
        use_parallel = parallel_config.get('enabled', True)
        num_workers = parallel_config.get('num_workers', None)
        
        for exp_name, pred_info in predictions.items():
            logger.info(f"\n{'='*60}")
            logger.info(f"Evaluating experiment: {exp_name}")
            logger.info(f"{'='*60}")
            
            # Verificar se houve erro na predição/carregamento
            if 'error' in pred_info:
                logger.error(
                    f"Skipping evaluation (prediction/loading failed): "
                    f"{pred_info['error']}"
                )
                continue
            
            # Log se predições foram carregadas
            if pred_info.get('loaded', False):
                logger.info(
                    f"Using precomputed predictions "
                    f"({pred_info['num_predictions']} files)"
                )
            
            # Verificar se deve skip avaliação
            if self._should_skip_evaluation(exp_name):
                logger.info(f"Skipping evaluation (already exists)")
                continue
            
            # Encontrar config do experimento
            exp_config = next(
                (exp for exp in self.experiments if exp.name == exp_name),
                None
            )
            
            if exp_config is None:
                logger.error(f"Config not found for experiment: {exp_name}")
                continue
            
            try:
                # Criar MetricsCalculator
                metrics_calculator = MetricsCalculator(self.config, exp_config)
                
                # Calcular métricas (com paralelização por imagem)
                results = metrics_calculator.calculate_metrics(
                    predictions_folder=pred_info['output_folder'],
                    ground_truth_csv=dataset_csv,
                    experiment_name=exp_name,
                    parallel=use_parallel,
                    num_workers=num_workers
                )
                
                all_results[exp_name] = results
                
            except Exception as e:
                logger.error(
                    f"Failed to calculate metrics for {exp_name}: {e}",
                    exc_info=True
                )
                continue
        
        return all_results
    
    def _evaluate_all_experiments(self, predictions: Dict, dataset_csv: str) -> Dict:
        """
        Calcula métricas para todos experimentos.
        Sempre sequencial por experimento, paralelo por imagem.
        """
        logger.info("\n" + "="*60)
        logger.info("STEP 3: CALCULATING METRICS")
        logger.info("="*60)
        
        # Sempre usar sequential (que processa imagens em paralelo)
        return self._evaluate_all_experiments_sequential(predictions, dataset_csv)
    
    def _evaluate_all_experiments_sequential(
        self, 
        predictions: Dict, 
        dataset_csv: str
    ) -> Dict:
        """
        Calcula métricas sequencialmente.
        
        Args:
            predictions: info sobre predições
            dataset_csv: path do CSV
            
        Returns:
            Dict com resultados
        """
        logger.info("Evaluating experiments SEQUENTIALLY")
        
        all_results = {}
        
        for exp_name, pred_info in predictions.items():
            logger.info(f"\n{'='*60}")
            logger.info(f"Evaluating experiment: {exp_name}")
            logger.info(f"{'='*60}")
            
            # Verificar se houve erro na predição/carregamento
            if 'error' in pred_info:
                logger.error(
                    f"Skipping evaluation (prediction/loading failed): "
                    f"{pred_info['error']}"
                )
                continue
            
            # NOVO: Log se predições foram carregadas
            if pred_info.get('loaded', False):
                logger.info(
                    f"Using precomputed predictions "
                    f"({pred_info['num_predictions']} files)"
                )
            
            # Verificar se deve skip avaliação
            if self._should_skip_evaluation(exp_name):
                logger.info(f"Skipping evaluation (already exists)")
                # TODO: carregar resultados existentes
                continue
            
            # Encontrar config do experimento
            exp_config = next(
                (exp for exp in self.experiments if exp.name == exp_name),
                None
            )
            
            if exp_config is None:
                logger.error(f"Config not found for experiment: {exp_name}")
                continue
            
            try:
                # Criar MetricsCalculator
                metrics_calculator = MetricsCalculator(self.config, exp_config)
                
                # Calcular métricas
                results = metrics_calculator.calculate_metrics(
                    predictions_folder=pred_info['output_folder'],
                    ground_truth_csv=dataset_csv,
                    experiment_name=exp_name
                )
                
                all_results[exp_name] = results
                
            except Exception as e:
                logger.error(
                    f"Failed to calculate metrics for {exp_name}: {e}",
                    exc_info=True
                )
                # Continuar com outros experimentos
                continue
        
        return all_results
    
    def _should_skip_evaluation(self, experiment_name: str) -> bool:
        """
        Verifica se deve pular avaliação de um experimento.
        
        Args:
            experiment_name: nome do experimento
            
        Returns:
            True se deve pular
        """
        if not self.config.pipeline_options.skip_existing_evaluations:
            return False
        
        # TODO: verificar se resultados já existem
        # Por enquanto, sempre avalia
        return False
    
    def _generate_visualizations(self, aggregated: Dict):
        """
        Gera todas as visualizações (Fase 3).
        
        Args:
            aggregated: Resultados agregados
        """
        logger.info("\n" + "="*60)
        logger.info("STEP 4: GENERATING VISUALIZATIONS")
        logger.info("="*60)
        
        all_results = aggregated['experiments']
        if len(all_results) == 0:
            logger.warning("No results to visualize")
            return
        
        # Diretório de saída para visualizações
        # Usar valor padrão se comparison_folder não existir
        comparison_folder = getattr(
            self.config.output.structure, 
            'comparison_folder', 
            'comparisons'  # Valor padrão
        )
            
        # Diretório de saída para visualizações
        output_dir = os.path.join(
            self.config.output.base_dir,
            comparison_folder,
        )
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # 1. Confusion Matrices
        if self.config.visualization.confusion_matrix.save_individual:
            logger.info("Plotting individual confusion matrices...")
            for exp_name, results in all_results.items():
                self.confusion_matrix_plotter.plot_single_experiment(
                    confusion_matrix=results['confusion_matrix'],
                    class_names=results['class_names'],
                    experiment_name=exp_name,
                    output_dir=os.path.join(output_dir, "confusion_matrices"),
                    normalize='all'
                )
        
        if self.config.visualization.confusion_matrix.save_comparison:
            logger.info("Plotting confusion matrices comparison...")
            self.confusion_matrix_plotter.plot_comparison_grid(
                experiments_data=all_results,
                output_dir=output_dir
            )
        
        # 2. Comparison Plots
        # if self.config.visualization.comparison_plots.enabled:
        #     logger.info("Generating comparison plots...")
        #     self.comparison_plotter.plot_all(
        #         experiments_data=all_results,
        #         output_dir=output_dir
        #     )
        
        logger.info("Visualizations completed")
    
    def _save_summary_report(self, aggregated: Dict, elapsed_time: float):
        """
        Salva relatório resumo em JSON.
        
        Args:
            aggregated: resultados agregados
            elapsed_time: tempo total de execução
        """
        import json
        from datetime import datetime
        
        logger.info("\n" + "="*60)
        logger.info("STEP 5: SAVING SUMMARY REPORT")
        logger.info("="*60)
        
        summary = {
            'pipeline_config': OmegaConf.to_container(self.config, resolve=True),
            'execution_time_seconds': elapsed_time,
            'execution_time_formatted': f"{elapsed_time/60:.2f} minutes",
            'timestamp': datetime.now().isoformat(),
            'num_experiments': aggregated['num_experiments'],
            'experiments': {}
        }
        
        # Adicionar info de cada experimento
        for exp_name, results in aggregated['experiments'].items():
            summary['experiments'][exp_name] = {
                'status': 'success',
                'num_classes': results['num_classes'],
                'class_names': results['class_names'],
                'num_images': len(results['per_image']),
                'output_dir': results['output_dir'],
                'metrics': results['aggregated']
            }
        
        # Salvar
        output_dir = self.config.output.base_dir
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        json_path = os.path.join(
            output_dir,
            self.config.output.files.summary_report
        )
        
        with open(json_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"Summary report saved: {json_path}")
        
        # Log resumo no console
        logger.info("\n" + "="*60)
        logger.info("SUMMARY")
        logger.info("="*60)
        for exp_name, exp_info in summary['experiments'].items():
            logger.info(f"\n{exp_name}:")
            logger.info(f"  Status: {exp_info['status']}")
            logger.info(f"  Images evaluated: {exp_info['num_images']}")
            logger.info(f"  Metrics:")
            for metric_name, value in exp_info['metrics'].items():
                if isinstance(value, float):
                    logger.info(f"    {metric_name}: {value:.4f}")
                else:
                    logger.info(f"    {metric_name}: {value}")


# ============================================================================
# WORKER FUNCTIONS FOR PARALLEL PROCESSING
# ============================================================================

def _run_prediction_worker(
    experiment: DictConfig,
    gpu_id: int,
    dataset_csv: str,
    config: DictConfig
) -> Dict:
    """
    Worker function para executar predição de um experimento.
    
    Esta função é executada em um processo separado.
    
    Args:
        experiment: Config do experimento
        gpu_id: ID da GPU
        dataset_csv: Path do CSV
        config: Config geral do pipeline
        
    Returns:
        Dict com info do resultado
    """
    import os
    import subprocess
    import sys
    from pathlib import Path
    
    # Preparar overrides
    overrides = [
        f"checkpoint_path={experiment.checkpoint_path}",
        # f"inference_dataset.input_csv_path={dataset_csv}",
    ]
    
    # Device override
    if gpu_id >= 0:
        overrides.append(f"device=cuda:{gpu_id}")
        overrides.append(f"pl_trainer.devices=[{gpu_id}]")
        # Setar variável de ambiente para garantir que usa a GPU correta
        os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    else:
        overrides.append("device=cpu")
        overrides.append("pl_trainer.accelerator=cpu")
        overrides.append("pl_trainer.devices=1")
    
    # Overrides do experimento
    if "overrides" in experiment and experiment.overrides:
        for key, value in experiment.overrides.items():
            overrides.append(f"{key}={value}")
    
    # Script path
    script_path = os.path.join(
        os.path.dirname(__file__),
        "..",
        "..",
        "predict.py"
    )
    
    # Comando
    cmd = [
        sys.executable,
        script_path,
        "--config-dir", str(Path(experiment.predict_config).parent),
        "--config-name", Path(experiment.predict_config).stem,
    ] + overrides
    
    # Executar
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True
    )
    
    if result.returncode != 0:
        raise RuntimeError(
            f"Prediction failed for {experiment.name}. "
            f"Return code: {result.returncode}\n"
            f"STDERR: {result.stderr}"
        )
    
    return {
        'output_folder': experiment.output_folder,
        'skipped': False,
        'gpu_id': gpu_id
    }


def _evaluate_experiment_worker(
    experiment_config: DictConfig,
    predictions_folder: str,
    dataset_csv: str,
    pipeline_config: DictConfig
) -> Tuple[str, Dict]:
    """
    Worker function para avaliar um experimento.
    
    Esta função é executada em um processo separado.
    
    Args:
        experiment_config: Config do experimento
        predictions_folder: Pasta com predições
        dataset_csv: CSV com ground truth
        pipeline_config: Config geral do pipeline
        
    Returns:
        Tupla (experiment_name, results_dict)
    """
    from pytorch_segmentation_models_trainer.tools.evaluation.metrics_calculator import (
        MetricsCalculator
    )
    
    metrics_calculator = MetricsCalculator(pipeline_config, experiment_config)
    
    results = metrics_calculator.calculate_metrics(
        predictions_folder=predictions_folder,
        ground_truth_csv=dataset_csv,
        experiment_name=experiment_config.name
    )
    
    return experiment_config.name, results
