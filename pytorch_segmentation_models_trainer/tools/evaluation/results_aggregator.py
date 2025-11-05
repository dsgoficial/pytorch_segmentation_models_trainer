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
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from omegaconf import DictConfig

logger = logging.getLogger(__name__)


class ResultsAggregator:
    """
    Agrega e organiza resultados de múltiplos experimentos.
    
    Features:
    - Agregação de métricas por experimento
    - Ranking de experimentos
    - Estatísticas descritivas
    - Identificação de best/worst
    - Exportação em múltiplos formatos
    """
    
    def __init__(self, config: DictConfig):
        """
        Args:
            config: DictConfig do pipeline
        """
        self.config = config
        logger.info("ResultsAggregator initialized")
    
    def aggregate(self, all_results: Dict) -> Dict:
        """
        Agrega resultados de múltiplos experimentos.
        """
        logger.info(f"Aggregating results from {len(all_results)} experiments")
        
        # DEBUG: Mostrar o que chegou
        if len(all_results) == 0:
            logger.warning("Received empty all_results dict!")
        else:
            logger.info(f"Experiments received: {list(all_results.keys())}")
            for exp_name, results in all_results.items():
                if results is None:
                    logger.warning(f"  {exp_name}: results is None")
                else:
                    logger.info(f"  {exp_name}: keys={list(results.keys())}")
        
        # Verificar se há resultados válidos
        valid_results = {k: v for k, v in all_results.items() if v is not None}
        
        if len(valid_results) == 0:
            logger.warning("No valid results to aggregate")
            return {
                'experiments': {},
                'rankings': {},
                'statistics': {},
                'num_experiments': 0
            }
        
        # Criar estrutura agregada
        aggregated = {
            'experiments': valid_results,
            'rankings': self._create_rankings(valid_results),
            'statistics': self._calculate_statistics(valid_results),
            'num_experiments': len(valid_results)
        }
        
        # Salvar CSV agregado
        self._save_aggregated_csv(valid_results)
        
        return aggregated
    
    def _create_summary(self, all_results: Dict) -> Dict:
        """
        Cria resumo geral dos resultados.
        
        Args:
            all_results: Resultados de todos experimentos
            
        Returns:
            Dict com resumo
        """
        summary = {
            'num_experiments': len(all_results),
            'total_images_evaluated': 0,
            'experiments_summary': {}
        }
        
        for exp_name, results in all_results.items():
            num_images = len(results['per_image'])
            summary['total_images_evaluated'] += num_images
            
            summary['experiments_summary'][exp_name] = {
                'num_images': num_images,
                'num_classes': results['num_classes'],
                'class_names': results['class_names'],
                'key_metrics': {
                    k: v for k, v in results['aggregated'].items()
                    if not k.endswith('_class_0') and not k.endswith('_class_1')
                    and not '_class_' in k
                }
            }
        
        return summary
    
    def _create_rankings(self, all_results: Dict) -> Dict:
        """
        Cria rankings de experimentos por métrica.
        """
        if len(all_results) == 0:
            logger.warning("No results to create rankings from")
            return {}
        
        logger.info("Creating rankings")
        
        # Obter lista de métricas do primeiro experimento
        first_exp = next(iter(all_results.values()))
        
        if first_exp is None or 'aggregated' not in first_exp:
            logger.warning("First experiment has no aggregated metrics")
            return {}
        
        metric_names = list(first_exp['aggregated'].keys())
        
        rankings = {}
        
        for metric_name in metric_names:
            # Coletar valores de todos os experimentos
            values = []
            for exp_name, results in all_results.items():
                if results and 'aggregated' in results and metric_name in results['aggregated']:
                    value = results['aggregated'][metric_name]
                    if value is not None:  # Ignorar valores None
                        values.append((exp_name, value))
            
            # Ordenar (maior valor = melhor)
            values.sort(key=lambda x: x[1], reverse=True)
            
            rankings[metric_name] = [
                {'rank': i+1, 'experiment': name, 'value': value}
                for i, (name, value) in enumerate(values)
            ]
        
        return rankings
    
    def _calculate_statistics(self, all_results: Dict) -> Dict:
        """
        Calcula estatísticas descritivas.
        
        Args:
            all_results: Resultados de todos experimentos
            
        Returns:
            Dict com estatísticas
        """
        logger.info("Calculating statistics")
        
        statistics = {}
        
        # Para cada experimento, calcular estatísticas das métricas por imagem
        for exp_name, results in all_results.items():
            per_image_df = results['per_image']
            
            # Selecionar colunas numéricas (métricas)
            metric_cols = [
                col for col in per_image_df.columns
                if col != 'image' and per_image_df[col].dtype in [np.float64, np.float32, np.int64]
            ]
            
            exp_stats = {}
            
            for metric in metric_cols:
                values = per_image_df[metric].dropna()
                
                if len(values) == 0:
                    continue
                
                exp_stats[metric] = {
                    'mean': float(values.mean()),
                    'std': float(values.std()),
                    'min': float(values.min()),
                    'max': float(values.max()),
                    'median': float(values.median()),
                    'q25': float(values.quantile(0.25)),
                    'q75': float(values.quantile(0.75))
                }
            
            statistics[exp_name] = exp_stats
        
        return statistics
    
    def _find_best_experiment(
        self, 
        all_results: Dict,
        metric: str = "JaccardIndex"
    ) -> Dict:
        """
        Encontra o melhor experimento baseado em uma métrica.
        
        Args:
            all_results: Resultados de todos experimentos
            metric: Métrica para comparação
            
        Returns:
            Dict com info do melhor experimento
        """
        best_exp = None
        best_score = -1
        
        for exp_name, results in all_results.items():
            score = results['aggregated'].get(metric, 0.0)
            
            if score > best_score:
                best_score = score
                best_exp = exp_name
        
        if best_exp:
            return {
                'name': best_exp,
                'metric': metric,
                'score': best_score
            }
        
        return {}
    
    def _find_worst_experiment(
        self, 
        all_results: Dict,
        metric: str = "JaccardIndex"
    ) -> Dict:
        """
        Encontra o pior experimento baseado em uma métrica.
        
        Args:
            all_results: Resultados de todos experimentos
            metric: Métrica para comparação
            
        Returns:
            Dict com info do pior experimento
        """
        worst_exp = None
        worst_score = float('inf')
        
        for exp_name, results in all_results.items():
            score = results['aggregated'].get(metric, 0.0)
            
            if score < worst_score:
                worst_score = score
                worst_exp = exp_name
        
        if worst_exp:
            return {
                'name': worst_exp,
                'metric': metric,
                'score': worst_score
            }
        
        return {}
    
    def _save_aggregated_csv(self, all_results: Dict):
        """
        Salva CSV com métricas agregadas de todos os experimentos.
        """
        logger.info("Saving aggregated metrics CSV")
        
        # Verificar se há resultados
        if len(all_results) == 0:
            logger.warning("No results to save to CSV")
            # Criar CSV vazio com colunas padrão
            empty_df = pd.DataFrame(columns=['experiment', 'num_classes', 'num_images'])
            output_path = os.path.join(
                self.config.output.base_dir,
                "aggregated_metrics.csv"
            )
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            empty_df.to_csv(output_path, index=False)
            logger.info(f"Empty CSV saved: {output_path}")
            return
        
        # Construir lista de linhas
        rows = []
        for exp_name, results in all_results.items():
            try:
                row = {
                    'experiment': exp_name,
                    'num_classes': results.get('num_classes', 0),
                    'num_images': len(results.get('per_image', []))
                }
                # Adicionar métricas agregadas
                if 'aggregated' in results and results['aggregated']:
                    row.update(results['aggregated'])
                rows.append(row)
            except Exception as e:
                logger.error(f"Error processing results for {exp_name}: {e}")
                continue
        
        # Verificar se há linhas
        if len(rows) == 0:
            logger.warning("No valid rows to save to CSV")
            empty_df = pd.DataFrame(columns=['experiment', 'num_classes', 'num_images'])
            output_path = os.path.join(
                self.config.output.base_dir,
                "aggregated_metrics.csv"
            )
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            empty_df.to_csv(output_path, index=False)
            logger.info(f"Empty CSV saved: {output_path}")
            return
        
        # Criar DataFrame
        df = pd.DataFrame(rows)
        
        # Reorganizar colunas
        cols = ['experiment', 'num_classes', 'num_images']
        # Filtrar apenas colunas que existem
        cols = [c for c in cols if c in df.columns]
        other_cols = [c for c in df.columns if c not in cols]
        
        if len(cols) > 0:
            df = df[cols + other_cols]
        
        # Salvar
        output_path = os.path.join(
            self.config.output.base_dir,
            "aggregated_metrics.csv"
        )
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
        
        logger.info(f"Aggregated metrics CSV saved: {output_path}")
        logger.info(f"  Experiments: {len(df)}")
        if len(other_cols) > 0:
            logger.info(f"  Metrics: {len(other_cols)}")
    
    def _save_rankings_csv(self, rankings: Dict):
        """
        Salva rankings em CSV.
        
        Args:
            rankings: Dict com rankings por métrica
        """
        logger.info("Saving rankings CSV")
        
        for metric, ranking in rankings.items():
            df = pd.DataFrame(ranking)
            
            # Salvar
            output_dir = os.path.join(
                self.config.output.base_dir,
                self.config.output.structure.reports_folder
            )
            Path(output_dir).mkdir(parents=True, exist_ok=True)
            
            csv_path = os.path.join(
                output_dir,
                f"ranking_{metric}.csv"
            )
            
            df.to_csv(csv_path, index=False)
            logger.debug(f"Saved ranking for {metric}: {csv_path}")
    
    def _save_statistics_json(self, statistics: Dict):
        """
        Salva estatísticas em JSON.
        
        Args:
            statistics: Dict com estatísticas
        """
        logger.info("Saving statistics JSON")
        
        output_dir = os.path.join(
            self.config.output.base_dir,
            self.config.output.structure.reports_folder
        )
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        json_path = os.path.join(output_dir, "statistics.json")
        
        with open(json_path, 'w') as f:
            json.dump(statistics, f, indent=2)
        
        logger.info(f"Saved statistics: {json_path}")
    
    def create_comparison_table(
        self, 
        all_results: Dict,
        metrics: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Cria tabela de comparação formatada.
        
        Args:
            all_results: Resultados de todos experimentos
            metrics: Métricas a incluir (None = todas)
            
        Returns:
            DataFrame formatado
        """
        rows = []
        
        for exp_name, results in all_results.items():
            row = {'Experiment': exp_name}
            
            aggregated = results['aggregated']
            
            if metrics:
                for metric in metrics:
                    if metric in aggregated:
                        row[metric] = f"{aggregated[metric]:.4f}"
            else:
                # Todas as métricas globais
                for metric, value in aggregated.items():
                    if '_class_' not in metric.lower() or metric.endswith('_class_0'):
                        continue  # Pular métricas por classe
                    row[metric] = f"{value:.4f}"
            
            rows.append(row)
        
        df = pd.DataFrame(rows)
        return df
    
    def get_improvement_matrix(
        self,
        all_results: Dict,
        baseline_exp: str,
        metric: str = "JaccardIndex"
    ) -> Dict:
        """
        Calcula matriz de melhorias relativas ao baseline.
        
        Args:
            all_results: Resultados de todos experimentos
            baseline_exp: Nome do experimento baseline
            metric: Métrica para comparação
            
        Returns:
            Dict com melhorias
        """
        if baseline_exp not in all_results:
            logger.error(f"Baseline experiment '{baseline_exp}' not found")
            return {}
        
        baseline_score = all_results[baseline_exp]['aggregated'].get(metric, 0.0)
        
        improvements = {}
        
        for exp_name, results in all_results.items():
            if exp_name == baseline_exp:
                improvements[exp_name] = {
                    'absolute': 0.0,
                    'relative': 0.0,
                    'percentage': 0.0
                }
                continue
            
            score = results['aggregated'].get(metric, 0.0)
            absolute_improvement = score - baseline_score
            relative_improvement = absolute_improvement / baseline_score if baseline_score != 0 else 0
            percentage_improvement = relative_improvement * 100
            
            improvements[exp_name] = {
                'absolute': absolute_improvement,
                'relative': relative_improvement,
                'percentage': percentage_improvement,
                'score': score,
                'baseline_score': baseline_score
            }
        
        return improvements
