# -*- coding: utf-8 -*-
"""
/***************************************************************************
 pytorch_segmentation_models_trainer
                              -------------------
        begin                : 2025-01-15
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
import os
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import patches
from omegaconf import DictConfig

logger = logging.getLogger(__name__)


class ComparisonPlotter:
    """
    Gera plots de comparação entre experimentos.
    
    Features:
    - Bar charts de métricas agregadas
    - Box plots de métricas por imagem
    - Radar charts comparativos
    - Line plots de distribuição
    - Análise de best/worst images
    """
    
    def __init__(self, config: DictConfig):
        """
        Args:
            config: DictConfig do pipeline com visualization.comparison_plots
        """
        self.config = config
        self.viz_config = config.visualization.comparison_plots
        
        # Configurações com valores padrão
        self.dpi = getattr(self.viz_config, 'dpi', 300)
        self.save_format = getattr(self.viz_config, 'save_format', 'png')
        
        # Métricas a comparar
        if hasattr(self.viz_config, 'metrics_to_compare'):
            self.metrics_to_compare = self.viz_config.metrics_to_compare
        elif hasattr(self.viz_config, 'metrics_to_plot'):
            # Fallback para metrics_to_plot
            self.metrics_to_compare = self.viz_config.metrics_to_plot
        else:
            # Default para as métricas mais comuns
            self.metrics_to_compare = ['Accuracy', 'JaccardIndex', 'Dice', 'F1Score']
        
        # Estilo
        if hasattr(self.viz_config, 'style'):
            plt.style.use(self.viz_config.style)
        else:
            plt.style.use('seaborn-v0_8-darkgrid')
        
        logger.info("ComparisonPlotter initialized")
    
    def plot_metrics_bar_chart(
        self,
        experiments_data: Dict[str, Dict],
        output_dir: str
    ) -> str:
        """
        Plota bar chart comparando métricas globais entre experimentos.
        
        Args:
            experiments_data: Dict {exp_name: {'aggregated': {...}}}
            output_dir: Diretório de saída
            
        Returns:
            Path do arquivo salvo
        """
        logger.info(f"Plotting metrics bar chart for {len(experiments_data)} experiments")
        
        # Extrair métricas a comparar
        metrics_to_compare = self.metrics_to_compare
        
        # Preparar dados
        exp_names = []
        metrics_data = {metric: [] for metric in metrics_to_compare}
        
        for exp_name, data in experiments_data.items():
            exp_names.append(exp_name)
            aggregated = data['aggregated']
            
            for metric in metrics_to_compare:
                # Buscar métrica no dicionário
                value = aggregated.get(metric, 0.0)
                metrics_data[metric].append(value)
        
        # Criar figura
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Configurar posições
        x = np.arange(len(exp_names))
        width = 0.8 / len(metrics_to_compare)
        
        # Cores
        colors = plt.cm.Set3(np.linspace(0, 1, len(metrics_to_compare)))
        
        # Plotar barras
        for i, (metric, values) in enumerate(metrics_data.items()):
            offset = (i - len(metrics_to_compare)/2 + 0.5) * width
            bars = ax.bar(
                x + offset,
                values,
                width,
                label=metric,
                alpha=0.8,
                color=colors[i]
            )
            
            # Adicionar valores nas barras
            for bar in bars:
                height = bar.get_height()
                ax.text(
                    bar.get_x() + bar.get_width()/2.,
                    height,
                    f'{height:.3f}',
                    ha='center',
                    va='bottom',
                    fontsize=8
                )
        
        # Configurar gráfico
        ax.set_xlabel('Experiments', fontsize=14, fontweight='bold')
        ax.set_ylabel('Score', fontsize=14, fontweight='bold')
        ax.set_title('Metrics Comparison Across Experiments', fontsize=16, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(exp_names, rotation=45, ha='right')
        ax.legend(loc='upper left', framealpha=0.9)
        ax.grid(axis='y', alpha=0.3)
        ax.set_ylim([0, 1])
        
        plt.tight_layout()
        
        # Salvar
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        output_path = os.path.join(
            output_dir,
            f"metrics_bar_chart.{self.save_format}"
        )
        plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved bar chart: {output_path}")
        return output_path
    
    def plot_per_image_boxplot(
        self,
        experiments_data: Dict[str, Dict],
        output_dir: str,
        metric: str = "JaccardIndex"
    ) -> str:
        """
        Plota box plot de uma métrica por imagem para cada experimento.
        
        Args:
            experiments_data: Dict {exp_name: {'per_image': DataFrame}}
            output_dir: Diretório de saída
            metric: Métrica a plotar
            
        Returns:
            Path do arquivo salvo
        """
        logger.info(f"Plotting box plot for {metric}")
        
        # Preparar dados
        plot_data = []
        
        for exp_name, data in experiments_data.items():
            per_image_df = data['per_image']
            
            if metric in per_image_df.columns:
                values = per_image_df[metric].values
                for value in values:
                    plot_data.append({
                        'Experiment': exp_name,
                        'Score': value
                    })
        
        if len(plot_data) == 0:
            logger.warning(f"No data found for metric {metric}")
            return None
        
        df = pd.DataFrame(plot_data)
        
        # Criar figura
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Box plot
        sns.boxplot(
            data=df,
            x='Experiment',
            y='Score',
            ax=ax,
            palette='Set3'
        )
        
        # Adicionar pontos individuais (strip plot)
        sns.stripplot(
            data=df,
            x='Experiment',
            y='Score',
            ax=ax,
            color='black',
            alpha=0.3,
            size=3
        )
        
        # Configurar
        ax.set_xlabel('Experiments', fontsize=14, fontweight='bold')
        ax.set_ylabel(metric, fontsize=14, fontweight='bold')
        ax.set_title(f'{metric} Distribution Across Images', fontsize=16, fontweight='bold')
        plt.xticks(rotation=45, ha='right')
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        
        # Salvar
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        output_path = os.path.join(
            output_dir,
            f"boxplot_{metric}.{self.save_format}"
        )
        plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved box plot: {output_path}")
        return output_path
    
    def plot_radar_chart(
        self,
        experiments_data: Dict[str, Dict],
        output_dir: str
    ) -> str:
        """
        Plota radar chart comparando múltiplas métricas.
        
        Args:
            experiments_data: Dict com dados dos experimentos
            output_dir: Diretório de saída
            
        Returns:
            Path do arquivo salvo
        """
        logger.info(f"Plotting radar chart for {len(experiments_data)} experiments")
        
        # Métricas a comparar
        metrics = self.viz_config.metrics_to_compare[:6]  # Máximo 6 métricas
        
        if len(metrics) < 3:
            logger.warning("Radar chart needs at least 3 metrics. Skipping.")
            return None
        
        # Preparar dados
        exp_names = []
        values_list = []
        
        for exp_name, data in experiments_data.items():
            exp_names.append(exp_name)
            aggregated = data['aggregated']
            
            values = [aggregated.get(metric, 0.0) for metric in metrics]
            values_list.append(values)
        
        # Configurar ângulos
        num_vars = len(metrics)
        angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
        angles += angles[:1]  # Fechar o círculo
        
        # Criar figura
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
        
        # Cores
        colors = plt.cm.Set3(np.linspace(0, 1, len(exp_names)))
        
        # Plotar cada experimento
        for i, (exp_name, values) in enumerate(zip(exp_names, values_list)):
            values += values[:1]  # Fechar o círculo
            ax.plot(angles, values, 'o-', linewidth=2, label=exp_name, color=colors[i])
            ax.fill(angles, values, alpha=0.15, color=colors[i])
        
        # Configurar labels
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(metrics, fontsize=12)
        ax.set_ylim(0, 1)
        ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=10)
        ax.grid(True, alpha=0.3)
        
        # Título e legenda
        plt.title('Metrics Comparison - Radar Chart', fontsize=16, fontweight='bold', pad=20)
        plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
        
        plt.tight_layout()
        
        # Salvar
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        output_path = os.path.join(
            output_dir,
            f"radar_chart.{self.save_format}"
        )
        plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved radar chart: {output_path}")
        return output_path
    
    def plot_best_worst_images(
        self,
        experiments_data: Dict[str, Dict],
        output_dir: str,
        metric: str = "JaccardIndex",
        n_images: int = 5
    ) -> Dict[str, str]:
        """
        Identifica e plota análise das melhores e piores imagens.
        
        Args:
            experiments_data: Dict com dados dos experimentos
            output_dir: Diretório de saída
            metric: Métrica para ranking
            n_images: Número de imagens para mostrar
            
        Returns:
            Dict com paths dos arquivos salvos
        """
        logger.info(f"Analyzing best/worst images by {metric}")
        
        output_paths = {}
        
        for exp_name, data in experiments_data.items():
            per_image_df = data['per_image']
            
            if metric not in per_image_df.columns:
                logger.warning(f"Metric {metric} not found for {exp_name}")
                continue
            
            # Ordenar por métrica
            sorted_df = per_image_df.sort_values(metric)
            
            # Piores e melhores
            worst = sorted_df.head(n_images)
            best = sorted_df.tail(n_images)
            
            # Criar figura
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
            
            # Plot worst
            ax1.barh(range(len(worst)), worst[metric].values, color='#ff6b6b')
            ax1.set_yticks(range(len(worst)))
            ax1.set_yticklabels([Path(img).name for img in worst['image'].values], fontsize=9)
            ax1.set_xlabel(metric, fontsize=12, fontweight='bold')
            ax1.set_title(f'Worst {n_images} Images', fontsize=14, fontweight='bold', color='#ff6b6b')
            ax1.grid(axis='x', alpha=0.3)
            ax1.invert_yaxis()
            
            # Plot best
            ax2.barh(range(len(best)), best[metric].values, color='#51cf66')
            ax2.set_yticks(range(len(best)))
            ax2.set_yticklabels([Path(img).name for img in best['image'].values], fontsize=9)
            ax2.set_xlabel(metric, fontsize=12, fontweight='bold')
            ax2.set_title(f'Best {n_images} Images', fontsize=14, fontweight='bold', color='#51cf66')
            ax2.grid(axis='x', alpha=0.3)
            ax2.invert_yaxis()
            
            # Título geral
            fig.suptitle(
                f'{exp_name} - Best/Worst Images Analysis',
                fontsize=16,
                fontweight='bold'
            )
            
            plt.tight_layout()
            
            # Salvar
            exp_output_dir = os.path.join(output_dir, exp_name)
            Path(exp_output_dir).mkdir(parents=True, exist_ok=True)
            
            output_path = os.path.join(
                exp_output_dir,
                f"best_worst_analysis.{self.save_format}"
            )
            plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
            plt.close()
            
            output_paths[exp_name] = output_path
            logger.info(f"Saved best/worst analysis for {exp_name}: {output_path}")
        
        return output_paths
    
    def plot_metrics_distribution(
        self,
        experiments_data: Dict[str, Dict],
        output_dir: str,
        metric: str = "JaccardIndex"
    ) -> str:
        """
        Plota distribuição (histograma + KDE) de uma métrica.
        
        Args:
            experiments_data: Dict com dados
            output_dir: Diretório de saída
            metric: Métrica a plotar
            
        Returns:
            Path do arquivo salvo
        """
        logger.info(f"Plotting distribution for {metric}")
        
        # Criar figura
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Plotar para cada experimento
        for exp_name, data in experiments_data.items():
            per_image_df = data['per_image']
            
            if metric in per_image_df.columns:
                values = per_image_df[metric].dropna()
                
                # Histogram + KDE
                sns.histplot(
                    values,
                    kde=True,
                    label=exp_name,
                    alpha=0.4,
                    ax=ax
                )
        
        # Configurar
        ax.set_xlabel(metric, fontsize=14, fontweight='bold')
        ax.set_ylabel('Frequency', fontsize=14, fontweight='bold')
        ax.set_title(f'{metric} Distribution Comparison', fontsize=16, fontweight='bold')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        
        # Salvar
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        output_path = os.path.join(
            output_dir,
            f"distribution_{metric}.{self.save_format}"
        )
        plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved distribution plot: {output_path}")
        return output_path
    
    def plot_per_class_comparison(
        self,
        experiments_data: Dict[str, Dict],
        output_dir: str,
        metric_prefix: str = "JaccardIndex"
    ) -> str:
        """
        Plota comparação de métrica por classe entre experimentos.
        
        Args:
            experiments_data: Dict com dados
            output_dir: Diretório de saída
            metric_prefix: Prefixo da métrica (ex: "JaccardIndex")
            
        Returns:
            Path do arquivo salvo
        """
        logger.info(f"Plotting per-class comparison for {metric_prefix}")
        
        # Encontrar métricas por classe
        first_exp = next(iter(experiments_data.values()))
        aggregated = first_exp['aggregated']
        class_names = first_exp['class_names']
        
        # Filtrar métricas por classe
        per_class_metrics = [
            key for key in aggregated.keys()
            if key.startswith(metric_prefix + '_')
        ]
        
        if len(per_class_metrics) == 0:
            logger.warning(f"No per-class metrics found for {metric_prefix}")
            return None
        
        # Extrair nomes de classes das métricas
        class_names_from_metrics = [
            metric.replace(metric_prefix + '_', '')
            for metric in per_class_metrics
        ]
        
        # Preparar dados
        exp_names = list(experiments_data.keys())
        values_matrix = []
        
        for exp_name in exp_names:
            values = [
                experiments_data[exp_name]['aggregated'].get(metric, 0.0)
                for metric in per_class_metrics
            ]
            values_matrix.append(values)
        
        values_matrix = np.array(values_matrix)
        
        # Criar figura
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Configurar posições
        x = np.arange(len(class_names_from_metrics))
        width = 0.8 / len(exp_names)
        
        # Cores
        colors = plt.cm.Set3(np.linspace(0, 1, len(exp_names)))
        
        # Plotar barras
        for i, (exp_name, values) in enumerate(zip(exp_names, values_matrix)):
            offset = (i - len(exp_names)/2 + 0.5) * width
            ax.bar(x + offset, values, width, label=exp_name, alpha=0.8, color=colors[i])
        
        # Configurar
        ax.set_xlabel('Classes', fontsize=14, fontweight='bold')
        ax.set_ylabel(metric_prefix, fontsize=14, fontweight='bold')
        ax.set_title(f'{metric_prefix} Per-Class Comparison', fontsize=16, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(class_names_from_metrics, rotation=45, ha='right')
        ax.legend(loc='lower right')
        ax.grid(axis='y', alpha=0.3)
        ax.set_ylim([0, 1])
        
        plt.tight_layout()
        
        # Salvar
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        output_path = os.path.join(
            output_dir,
            f"per_class_{metric_prefix}.{self.save_format}"
        )
        plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved per-class comparison: {output_path}")
        return output_path
