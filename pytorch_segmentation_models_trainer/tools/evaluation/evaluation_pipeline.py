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
from concurrent.futures import ProcessPoolExecutor, as_completed
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
    2. Executar predições para cada experimento
    3. Calcular métricas para cada experimento
    4. Agregar resultados
    5. Gerar visualizações (Fase 3)
    
    Suporta:
    - Múltiplos experimentos
    - Skip de predições/avaliações existentes
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
            
            # 2. Executar predições para cada experimento
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
        
        Se build_csv_from_folders.enabled=True, constrói CSV automaticamente.
        Caso contrário, usa input_csv_path existente.
        
        Returns:
            Path do CSV do dataset
        """
        logger.info("\n" + "="*60)
        logger.info("STEP 1: PREPARING DATASET")
        logger.info("="*60)
        
        if self.config.evaluation_dataset.build_csv_from_folders.enabled:
            logger.info("Building CSV from folders...")
            
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
            
        else:
            csv_path = self.config.evaluation_dataset.input_csv_path
            
            if not os.path.exists(csv_path):
                raise FileNotFoundError(
                    f"Dataset CSV not found: {csv_path}. "
                    f"Either provide a valid input_csv_path or enable "
                    f"build_csv_from_folders."
                )
            
            logger.info(f"Using existing dataset CSV: {csv_path}")
            return csv_path
    
    def _run_predictions(self, dataset_csv: str) -> Dict:
        """
        Executa predições para todos experimentos.
        
        Se paralelização estiver habilitada, distribui entre GPUs e executa em paralelo.
        Caso contrário, executa sequencialmente.
        
        Args:
            dataset_csv: path do CSV com dataset
            
        Returns:
            Dict mapeando experiment_name -> info sobre predições
        """
        logger.info("\n" + "="*60)
        logger.info("STEP 2: RUNNING PREDICTIONS")
        logger.info("="*60)
        
        if (self.config.pipeline_options.parallel_inference.enabled and 
            self.gpu_distributor is not None):
            return self._run_predictions_parallel(dataset_csv)
        else:
            return self._run_predictions_sequential(dataset_csv)
    
    def _run_predictions_sequential(self, dataset_csv: str) -> Dict:
        """
        Executa predições sequencialmente.
        
        Args:
            dataset_csv: path do CSV com dataset
            
        Returns:
            Dict com info de predições
        """
        logger.info("Running predictions SEQUENTIALLY")
        
        predictions_info = {}
        
        for exp in self.experiments:
            logger.info(f"\n--- Experiment: {exp.name} ---")
            
            # Verificar se deve skip
            if self._should_skip_prediction(exp):
                logger.info(f"Skipping prediction (already exists)")
                predictions_info[exp.name] = {
                    'output_folder': exp.output_folder,
                    'skipped': True,
                    'gpu_id': -1
                }
                continue
            
            # Executar predição
            try:
                self._run_single_prediction(exp, dataset_csv, gpu_id=-1)
                predictions_info[exp.name] = {
                    'output_folder': exp.output_folder,
                    'skipped': False,
                    'gpu_id': -1
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
        assignments = self.gpu_distributor.assign_experiments(self.experiments)
        
        # Filtrar experimentos que precisam de predição
        experiments_to_run = []
        skipped_info = {}
        
        for gpu_id, exps in assignments.items():
            for exp in exps:
                if self._should_skip_prediction(exp):
                    logger.info(f"Skipping {exp.name} (already exists)")
                    skipped_info[exp.name] = {
                        'output_folder': exp.output_folder,
                        'skipped': True,
                        'gpu_id': gpu_id
                    }
                else:
                    experiments_to_run.append((exp, gpu_id, dataset_csv))
        
        if len(experiments_to_run) == 0:
            logger.info("No predictions to run (all skipped)")
            return skipped_info
        
        logger.info(
            f"Running {len(experiments_to_run)} predictions in parallel "
            f"across {len(assignments)} device(s)"
        )
        
        # Executar em paralelo
        predictions_info = skipped_info.copy()
        
        # Usar ProcessPoolExecutor para paralelização
        max_workers = len(assignments)
        
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            # Submeter todas as tarefas
            future_to_exp = {
                executor.submit(
                    _run_prediction_worker,
                    exp,
                    gpu_id,
                    dataset_csv,
                    self.config
                ): exp.name
                for exp, gpu_id, dataset_csv in experiments_to_run
            }
            
            # Coletar resultados com barra de progresso
            with tqdm(
                total=len(future_to_exp),
                desc="Parallel predictions",
                unit="exp"
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
        prediction_files = list(Path(output_folder).glob("seg_*_output.tif"))
        
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
        
        Chama o script predict_from_batch.py como subprocesso.
        
        Args:
            experiment: config do experimento
            dataset_csv: path do CSV com dataset
            gpu_id: ID da GPU a usar (-1 para CPU)
        """
        logger.info(f"Running prediction for {experiment.name} on GPU {gpu_id}...")
        
        # Preparar overrides para Hydra
        overrides = [
            f"checkpoint_path={experiment.checkpoint_path}",
            f"inference_dataset.input_csv_path={dataset_csv}",
        ]
        
        # Adicionar device override
        if gpu_id >= 0:
            overrides.append(f"device=cuda:{gpu_id}")
            overrides.append(f"pl_trainer.devices=[{gpu_id}]")
        else:
            overrides.append("device=cpu")
            overrides.append("pl_trainer.accelerator=cpu")
        
        # Adicionar overrides específicos do experimento
        if "overrides" in experiment and experiment.overrides:
            for key, value in experiment.overrides.items():
                overrides.append(f"{key}={value}")
        
        # Construir comando
        script_path = os.path.join(
            os.path.dirname(__file__),
            "..",
            "..",
            "predict_from_batch.py"
        )
        
        if not os.path.exists(script_path):
            raise FileNotFoundError(
                f"predict_from_batch.py not found at {script_path}"
            )
        
        cmd = [
            sys.executable,
            script_path,
            "--config-dir", str(Path(experiment.predict_config).parent),
            "--config-name", Path(experiment.predict_config).stem,
        ] + overrides
        
        logger.debug(f"Command: {' '.join(cmd)}")
        
        # Executar
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True
        )
        
        if result.returncode != 0:
            logger.error(f"Prediction failed with return code {result.returncode}")
            logger.error(f"STDOUT:\n{result.stdout}")
            logger.error(f"STDERR:\n{result.stderr}")
            raise RuntimeError(
                f"Prediction failed for {experiment.name}. "
                f"Check logs for details."
            )
        
        logger.info(f"Prediction completed successfully for {experiment.name}")
    
    def _evaluate_all_experiments(
        self, 
        predictions: Dict, 
        dataset_csv: str
    ) -> Dict:
        """
        Calcula métricas para todos experimentos.
        
        Se paralelização estiver habilitada, executa em paralelo.
        
        Args:
            predictions: info sobre predições
            dataset_csv: path do CSV com dataset
            
        Returns:
            Dict mapeando experiment_name -> resultados
        """
        logger.info("\n" + "="*60)
        logger.info("STEP 3: CALCULATING METRICS")
        logger.info("="*60)
        
        if self.config.pipeline_options.parallel_evaluation.enabled:
            return self._evaluate_all_experiments_parallel(predictions, dataset_csv)
        else:
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
            
            # Verificar se houve erro na predição
            if 'error' in pred_info:
                logger.error(f"Skipping evaluation (prediction failed): {pred_info['error']}")
                continue
            
            # Verificar se deve skip
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
    
    def _evaluate_all_experiments_parallel(
        self, 
        predictions: Dict, 
        dataset_csv: str
    ) -> Dict:
        """
        Calcula métricas em paralelo usando multiprocessing.
        
        Args:
            predictions: info sobre predições
            dataset_csv: path do CSV
            
        Returns:
            Dict com resultados
        """
        logger.info("Evaluating experiments IN PARALLEL")
        
        # Preparar lista de tarefas
        tasks = []
        
        for exp_name, pred_info in predictions.items():
            # Verificar se houve erro na predição
            if 'error' in pred_info:
                logger.error(f"Skipping {exp_name} (prediction failed)")
                continue
            
            # Verificar se deve skip
            if self._should_skip_evaluation(exp_name):
                logger.info(f"Skipping {exp_name} (already exists)")
                continue
            
            # Encontrar config
            exp_config = next(
                (exp for exp in self.experiments if exp.name == exp_name),
                None
            )
            
            if exp_config is None:
                logger.error(f"Config not found for {exp_name}")
                continue
            
            tasks.append((
                exp_config,
                pred_info['output_folder'],
                dataset_csv,
                self.config
            ))
        
        if len(tasks) == 0:
            logger.warning("No experiments to evaluate")
            return {}
        
        logger.info(f"Evaluating {len(tasks)} experiments in parallel")
        
        # Executar em paralelo
        all_results = {}
        num_workers = min(
            len(tasks),
            self.config.pipeline_options.parallel_evaluation.num_workers
        )
        
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            # Submeter tarefas
            future_to_exp = {
                executor.submit(_evaluate_experiment_worker, *task): task[0].name
                for task in tasks
            }
            
            # Coletar resultados
            with tqdm(
                total=len(future_to_exp),
                desc="Parallel evaluation",
                unit="exp"
            ) as pbar:
                for future in as_completed(future_to_exp):
                    exp_name = future_to_exp[future]
                    
                    try:
                        name, results = future.result()
                        all_results[name] = results
                        logger.info(f"✓ {name} evaluation completed")
                        
                    except Exception as e:
                        logger.error(f"✗ {exp_name} evaluation failed: {e}")
                    
                    pbar.update(1)
        
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
    
    def _simple_aggregate(self, all_results: Dict) -> Dict:
        """
        Agregação simples dos resultados (DEPRECATED - usar results_aggregator).
        
        Args:
            all_results: resultados de todos experimentos
            
        Returns:
            Dict com resultados agregados
        """
        logger.info("\n" + "="*60)
        logger.info("STEP 4: AGGREGATING RESULTS")
        logger.info("="*60)
        
        # Por enquanto, apenas retorna os resultados
        # Fase 3 terá ResultsAggregator com funcionalidades avançadas
        
        aggregated = {
            'experiments': all_results,
            'num_experiments': len(all_results)
        }
        
        # Criar CSV agregado simples
        self._create_aggregated_csv(all_results)
        
        return aggregated
    
    def _generate_visualizations(self, aggregated: Dict):
        """
        Gera todas as visualizações (Fase 3).
        
        Args:
            aggregated: Resultados agregados
        """
        logger.info("\n" + "="*60)
        logger.info("STEP 5: GENERATING VISUALIZATIONS")
        logger.info("="*60)
        
        all_results = aggregated['experiments']
        
        # Diretório de saída para visualizações
        output_dir = os.path.join(
            self.config.output.base_dir,
            self.config.output.structure.comparison_folder
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
                    output_dir=os.path.join(output_dir, "confusion_matrices")
                )
        
        if self.config.visualization.confusion_matrix.save_comparison:
            logger.info("Plotting confusion matrices comparison...")
            self.confusion_matrix_plotter.plot_comparison_grid(
                experiments_data=all_results,
                output_dir=output_dir
            )
        
        # 2. Per-class accuracy
        if self.config.visualization.per_class_analysis.enabled:
            logger.info("Plotting per-class accuracy...")
            self.confusion_matrix_plotter.plot_per_class_accuracy(
                experiments_data=all_results,
                output_dir=output_dir
            )
        
        # 3. Comparison plots
        if "bar_chart" in self.config.visualization.comparison_plots.plot_types:
            logger.info("Plotting metrics bar chart...")
            self.comparison_plotter.plot_metrics_bar_chart(
                experiments_data=all_results,
                output_dir=output_dir
            )
        
        if "box_plot" in self.config.visualization.comparison_plots.plot_types:
            logger.info("Plotting box plots...")
            # Box plot para cada métrica configurada
            for metric in self.config.visualization.comparison_plots.metrics_to_compare:
                try:
                    self.comparison_plotter.plot_per_image_boxplot(
                        experiments_data=all_results,
                        output_dir=output_dir,
                        metric=metric
                    )
                except Exception as e:
                    logger.warning(f"Could not plot box plot for {metric}: {e}")
        
        if "radar_chart" in self.config.visualization.comparison_plots.plot_types:
            logger.info("Plotting radar chart...")
            self.comparison_plotter.plot_radar_chart(
                experiments_data=all_results,
                output_dir=output_dir
            )
        
        # 4. Per-class comparison
        if self.config.visualization.per_class_analysis.enabled:
            for plot_type in self.config.visualization.per_class_analysis.plot_types:
                if plot_type == "iou_per_class":
                    logger.info("Plotting IoU per class...")
                    self.comparison_plotter.plot_per_class_comparison(
                        experiments_data=all_results,
                        output_dir=output_dir,
                        metric_prefix="JaccardIndex"
                    )
                elif plot_type == "accuracy_per_class":
                    logger.info("Plotting accuracy per class...")
                    self.comparison_plotter.plot_per_class_comparison(
                        experiments_data=all_results,
                        output_dir=output_dir,
                        metric_prefix="Accuracy"
                    )
        
        # 5. Best/Worst images analysis
        if self.config.visualization.per_image_analysis.enabled:
            logger.info("Analyzing best/worst images...")
            self.comparison_plotter.plot_best_worst_images(
                experiments_data=all_results,
                output_dir=output_dir,
                metric=self.config.visualization.per_image_analysis.criteria,
                n_images=self.config.visualization.per_image_analysis.save_worst_n
            )
        
        # 6. Distribution plots
        logger.info("Plotting metric distributions...")
        for metric in self.config.visualization.comparison_plots.metrics_to_compare:
            try:
                self.comparison_plotter.plot_metrics_distribution(
                    experiments_data=all_results,
                    output_dir=output_dir,
                    metric=metric
                )
            except Exception as e:
                logger.warning(f"Could not plot distribution for {metric}: {e}")
        
        logger.info(f"All visualizations saved to: {output_dir}")
    
    def _create_aggregated_csv(self, all_results: Dict):
        """
        Cria CSV agregado com métricas de todos experimentos.
        
        Args:
            all_results: resultados de todos experimentos
        """
        import pandas as pd
        
        rows = []
        
        for exp_name, results in all_results.items():
            row = {
                'experiment': exp_name,
                'num_classes': results['num_classes'],
            }
            row.update(results['aggregated'])
            rows.append(row)
        
        df = pd.DataFrame(rows)
        
        # Salvar
        output_dir = self.config.output.base_dir
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        csv_path = os.path.join(
            output_dir,
            self.config.output.files.aggregated_metrics
        )
        
        df.to_csv(csv_path, index=False)
        logger.info(f"Aggregated metrics saved: {csv_path}")
    
    def _save_summary_report(self, aggregated: Dict, elapsed_time: float):
        """
        Salva relatório resumido do pipeline.
        
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
        f"inference_dataset.input_csv_path={dataset_csv}",
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
    
    # Overrides do experimento
    if "overrides" in experiment and experiment.overrides:
        for key, value in experiment.overrides.items():
            overrides.append(f"{key}={value}")
    
    # Script path
    script_path = os.path.join(
        os.path.dirname(__file__),
        "..",
        "..",
        "predict_from_batch.py"
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
