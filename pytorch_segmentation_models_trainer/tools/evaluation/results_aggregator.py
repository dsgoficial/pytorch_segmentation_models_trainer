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
    
    def aggregate(self, all_results: Dict[str, Dict]) -> Dict:
        """
        Agrega resultados de todos os experimentos.
        
        Args:
            all_results: Dict {exp_name: results}
            
        Returns:
            Dict com resultados agregados
        """
        logger.info(f"Aggregating results from {len(all_results)} experiments")
        
        aggregated = {
            'experiments': all_results,
            'summary': self._create_summary(all_results),
            'rankings': self._create_rankings(all_results),
            'statistics': self._calculate_statistics(all_results),
            'best_experiment': self._find_best_experiment(all_results),
            'worst_experiment': self._find_worst_experiment(all_results)
        }
        
        # Salvar agregações
        self._save_aggregated_csv(all_results)
        self._save_rankings_csv(aggregated['rankings'])
        self._save_statistics_json(aggregated['statistics'])
        
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
    
    def _create_rankings(
        self, 
        all_results: Dict,
        metrics_to_rank: Optional[List[str]] = None
    ) -> Dict:
        """
        Cria rankings de experimentos para diferentes métricas.
        
        Args:
            all_results: Resultados de todos experimentos
            metrics_to_rank: Lista de métricas para ranquear (None = todas)
            
        Returns:
            Dict com rankings
        """
        logger.info("Creating rankings")
        
        rankings = {}
        
        # Obter métricas globais
        first_exp = next(iter(all_results.values()))
        aggregated = first_exp['aggregated']
        
        # Filtrar métricas globais (sem "_class_")
        global_metrics = [
            k for k in aggregated.keys()
            if '_class_' not in k.lower() or k.endswith('_class_0')  # Evitar métricas por classe
        ]
        
        # Limpar nomes de métricas
        global_metrics = [k for k in global_metrics if not any(
            k.endswith(f'_class_{i}') for i in range(20)
        )]
        
        if metrics_to_rank:
            global_metrics = [m for m in global_metrics if m in metrics_to_rank]
        
        # Ranquear para cada métrica
        for metric in global_metrics:
            ranking = []
            
            for exp_name, results in all_results.items():
                value = results['aggregated'].get(metric, 0.0)
                ranking.append({
                    'experiment': exp_name,
                    'score': value
                })
            
            # Ordenar (maior é melhor)
            ranking = sorted(ranking, key=lambda x: x['score'], reverse=True)
            
            # Adicionar posição
            for i, item in enumerate(ranking, 1):
                item['rank'] = i
            
            rankings[metric] = ranking
        
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
        Salva CSV com métricas agregadas de todos experimentos.
        
        Args:
            all_results: Resultados de todos experimentos
        """
        logger.info("Saving aggregated metrics CSV")
        
        rows = []
        
        for exp_name, results in all_results.items():
            row = {
                'experiment': exp_name,
                'num_classes': results['num_classes'],
                'num_images': len(results['per_image'])
            }
            row.update(results['aggregated'])
            rows.append(row)
        
        df = pd.DataFrame(rows)
        
        # Reordenar colunas
        cols = ['experiment', 'num_classes', 'num_images']
        other_cols = [c for c in df.columns if c not in cols]
        df = df[cols + other_cols]
        
        # Salvar
        output_dir = self.config.output.base_dir
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        csv_path = os.path.join(
            output_dir,
            self.config.output.files.aggregated_metrics
        )
        
        df.to_csv(csv_path, index=False)
        logger.info(f"Saved aggregated metrics: {csv_path}")
    
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
