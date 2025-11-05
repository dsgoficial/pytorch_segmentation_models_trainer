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
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.gridspec import GridSpec
from omegaconf import DictConfig
from datetime import datetime

logger = logging.getLogger(__name__)


class ConfusionMatrixPlotter:
    """
    Gera plots de matrizes de confusão.
    
    Features:
    - Plot individual por experimento
    - Grid comparando múltiplos experimentos
    - Normalização configurável
    - Estilos customizáveis
    - Exportação em alta resolução
    """
    
    def __init__(self, config: DictConfig):
        """
        Args:
            config: DictConfig do pipeline com visualization.confusion_matrix
        """
        self.config = config
        self.viz_config = config.visualization.confusion_matrix
        
        # Tentar obter metrics_config, ou usar viz_config como fallback
        if hasattr(config.metrics, 'confusion_matrix'):
            self.metrics_config = config.metrics.confusion_matrix
        else:
            # Usar visualization config como fallback
            self.metrics_config = self.viz_config
        
        # Configurações de plot com valores padrão
        if hasattr(self.metrics_config, 'plot_config'):
            self.figsize = tuple(self.metrics_config.plot_config.figsize)
            self.cmap = self.metrics_config.plot_config.cmap
            self.dpi = self.metrics_config.plot_config.dpi
            self.save_format = self.metrics_config.plot_config.save_format
            self.annot_fontsize = self.metrics_config.plot_config.annot_fontsize
        else:
            # Valores padrão
            self.figsize = (10, 8)
            self.cmap = 'Blues'
            self.dpi = 300
            self.save_format = 'png'
            self.annot_fontsize = 10
        
        # Configurar estilo matplotlib
        plt.style.use('seaborn-v0_8-darkgrid')
    
    logger.info("ConfusionMatrixPlotter initialized")
    
    def plot_single_experiment(
        self,
        confusion_matrix: np.ndarray,
        class_names: List[str],
        experiment_name: str,
        output_dir: str,
        normalize: Optional[str] = None
    ) -> str:
        """
        Plota matriz de confusão de um único experimento.
        
        Args:
            confusion_matrix: Matriz de confusão [num_classes, num_classes]
            class_names: Lista de nomes de classes
            experiment_name: Nome do experimento
            output_dir: Diretório de saída
            normalize: Como normalizar ("true", "pred", "all", None)
            
        Returns:
            Path do arquivo salvo
        """
        logger.info(f"Plotting confusion matrix for {experiment_name}")
        
        # Normalizar se necessário
        cm = confusion_matrix.copy()
        normalize = normalize or self.metrics_config.normalize
        
        if normalize == 'true':
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            cm = np.nan_to_num(cm)
            fmt = '.2f'
            title_suffix = ' (Normalized by True Labels)'
        elif normalize == 'pred':
            cm = cm.astype('float') / cm.sum(axis=0)[np.newaxis, :]
            cm = np.nan_to_num(cm)
            fmt = '.2f'
            title_suffix = ' (Normalized by Predictions)'
        elif normalize == 'all':
            cm = cm.astype('float') / cm.sum() * 100
            fmt = '.2f'
            title_suffix = ' (Normalized by All - Percentage)'
        else:
            fmt = 'd'
            title_suffix = ''
        
        # Criar figura
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # Plot heatmap
        sns.heatmap(
            cm,
            annot=True,
            fmt=fmt,
            cmap=self.cmap,
            xticklabels=class_names,
            yticklabels=class_names,
            cbar_kws={'shrink': 0.8},
            ax=ax,
            annot_kws={'fontsize': self.annot_fontsize}
        )
        
        # Títulos e labels
        ax.set_title(
            f'Confusion Matrix - {experiment_name}{title_suffix}',
            fontsize=16,
            pad=20,
            fontweight='bold'
        )
        ax.set_xlabel('Predicted Labels', fontsize=14, fontweight='bold')
        ax.set_ylabel('True Labels', fontsize=14, fontweight='bold')
        
        # Rotacionar labels
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        
        plt.tight_layout()
        
        # Salvar
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        output_path = os.path.join(
            output_dir,
            f"confusion_matrix_{experiment_name}.{self.save_format}"
        )
        plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved confusion matrix plot: {output_path}")
        return output_path
    
    def plot_comparison_grid(
        self,
        experiments_data: Dict[str, Dict],
        output_dir: str,
        normalize: Optional[str] = None
    ) -> str:
        """
        Plota grid com múltiplas matrizes de confusão para comparação.
        
        Args:
            experiments_data: Dict {exp_name: {'confusion_matrix': ..., 'class_names': ...}}
            output_dir: Diretório de saída
            normalize: Como normalizar
            
        Returns:
            Path do arquivo salvo
        """
        logger.info(f"Plotting confusion matrix comparison grid for {len(experiments_data)} experiments")
        
        num_experiments = len(experiments_data)
        
        # Calcular layout do grid
        if num_experiments <= 2:
            nrows, ncols = 1, num_experiments
        elif num_experiments <= 4:
            nrows, ncols = 2, 2
        elif num_experiments <= 6:
            nrows, ncols = 2, 3
        elif num_experiments <= 9:
            nrows, ncols = 3, 3
        else:
            nrows = int(np.ceil(np.sqrt(num_experiments)))
            ncols = int(np.ceil(num_experiments / nrows))
        
        # Criar figura
        fig_width = ncols * 8
        fig_height = nrows * 7
        fig = plt.figure(figsize=(fig_width, fig_height))
        
        # Grid layout
        gs = GridSpec(nrows, ncols, figure=fig, hspace=0.3, wspace=0.3)
        
        # Normalização
        normalize = normalize or self.metrics_config.normalize
        
        # Plotar cada experimento
        for idx, (exp_name, data) in enumerate(experiments_data.items()):
            row = idx // ncols
            col = idx % ncols
            
            ax = fig.add_subplot(gs[row, col])
            
            # Obter matriz e nomes
            cm = data['confusion_matrix'].copy()
            class_names = data['class_names']
            
            # Normalizar
            if normalize == 'true':
                cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
                cm = np.nan_to_num(cm)
                fmt = '.2f'
            elif normalize == 'pred':
                cm = cm.astype('float') / cm.sum(axis=0)[np.newaxis, :]
                cm = np.nan_to_num(cm)
                fmt = '.2f'
            elif normalize == 'all':
                cm = cm.astype('float') / cm.sum()
                fmt = '.2f'
            else:
                fmt = 'd'
            
            # Plot
            sns.heatmap(
                cm,
                annot=True,
                fmt=fmt,
                cmap=self.cmap,
                xticklabels=class_names,
                yticklabels=class_names,
                cbar=True,
                ax=ax,
                annot_kws={'fontsize': max(6, self.annot_fontsize - 2)}
            )
            
            # Título
            ax.set_title(exp_name, fontsize=12, fontweight='bold')
            ax.set_xlabel('Predicted', fontsize=10)
            ax.set_ylabel('True', fontsize=10)
            
            # Rotacionar labels
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', fontsize=8)
            ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=8)
        
        # Título geral
        title_suffix = ''
        if normalize == 'true':
            title_suffix = ' (Normalized by True Labels)'
        elif normalize == 'pred':
            title_suffix = ' (Normalized by Predictions)'
        elif normalize == 'all':
            title_suffix = ' (Normalized by All)'
        
        fig.suptitle(
            f'Confusion Matrices Comparison{title_suffix}',
            fontsize=18,
            fontweight='bold',
            y=0.98
        )
        
        # Salvar
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        output_path = os.path.join(
            output_dir,
            f'confusion_matrices_comparison_{datetime.now().strftime("%Y%m%d_%H%M%S")}.{self.save_format}'
        )
        plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved comparison grid: {output_path}")
        return output_path
    
    def plot_difference_matrix(
        self,
        experiment1_data: Dict,
        experiment2_data: Dict,
        output_dir: str
    ) -> str:
        """
        Plota matriz de diferença entre dois experimentos.
        
        Args:
            experiment1_data: Dados do experimento 1
            experiment2_data: Dados do experimento 2
            output_dir: Diretório de saída
            
        Returns:
            Path do arquivo salvo
        """
        logger.info(
            f"Plotting difference matrix: {experiment1_data['name']} vs {experiment2_data['name']}"
        )
        
        # Normalizar ambas as matrizes
        cm1 = experiment1_data['confusion_matrix'].astype('float')
        cm1 = cm1 / cm1.sum(axis=1)[:, np.newaxis]
        cm1 = np.nan_to_num(cm1)
        
        cm2 = experiment2_data['confusion_matrix'].astype('float')
        cm2 = cm2 / cm2.sum(axis=1)[:, np.newaxis]
        cm2 = np.nan_to_num(cm2)
        
        # Calcular diferença
        diff = cm1 - cm2
        
        # Criar figura
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # Plot heatmap com diverging colormap
        sns.heatmap(
            diff,
            annot=True,
            fmt='.2f',
            cmap='RdBu_r',  # Diverging colormap
            xticklabels=experiment1_data['class_names'],
            yticklabels=experiment1_data['class_names'],
            center=0,  # Centro em 0
            cbar_kws={'shrink': 0.8, 'label': 'Difference'},
            ax=ax,
            vmin=-1,
            vmax=1
        )
        
        # Título
        ax.set_title(
            f"Difference: {experiment1_data['name']} - {experiment2_data['name']}",
            fontsize=16,
            pad=20,
            fontweight='bold'
        )
        ax.set_xlabel('Predicted Labels', fontsize=14)
        ax.set_ylabel('True Labels', fontsize=14)
        
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        
        # Salvar
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        output_path = os.path.join(
            output_dir,
            f"confusion_matrix_diff_{experiment1_data['name']}_vs_{experiment2_data['name']}.{self.save_format}"
        )
        plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved difference matrix: {output_path}")
        return output_path
    
    def plot_per_class_accuracy(
        self,
        experiments_data: Dict[str, Dict],
        output_dir: str
    ) -> str:
        """
        Plota acurácia por classe para cada experimento.
        
        Args:
            experiments_data: Dict com dados dos experimentos
            output_dir: Diretório de saída
            
        Returns:
            Path do arquivo salvo
        """
        logger.info(f"Plotting per-class accuracy for {len(experiments_data)} experiments")
        
        # Calcular acurácia por classe para cada experimento
        exp_names = []
        class_names = None
        accuracies = []
        
        for exp_name, data in experiments_data.items():
            cm = data['confusion_matrix']
            if class_names is None:
                class_names = data['class_names']
            
            # Acurácia por classe = diagonal / total por classe
            class_acc = np.diag(cm) / cm.sum(axis=1)
            class_acc = np.nan_to_num(class_acc)
            
            exp_names.append(exp_name)
            accuracies.append(class_acc)
        
        accuracies = np.array(accuracies)
        
        # Criar figura
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Configurar posições das barras
        x = np.arange(len(class_names))
        width = 0.8 / len(exp_names)
        
        # Plotar barras para cada experimento
        for i, (exp_name, acc) in enumerate(zip(exp_names, accuracies)):
            offset = (i - len(exp_names)/2 + 0.5) * width
            ax.bar(x + offset, acc, width, label=exp_name, alpha=0.8)
        
        # Configurar gráfico
        ax.set_xlabel('Classes', fontsize=14, fontweight='bold')
        ax.set_ylabel('Accuracy', fontsize=14, fontweight='bold')
        ax.set_title('Per-Class Accuracy Comparison', fontsize=16, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(class_names, rotation=45, ha='right')
        ax.legend(loc='lower right')
        ax.grid(axis='y', alpha=0.3)
        ax.set_ylim([0, 1])
        
        plt.tight_layout()
        
        # Salvar
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        output_path = os.path.join(
            output_dir,
            f"per_class_accuracy.{self.save_format}"
        )
        plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved per-class accuracy plot: {output_path}")
        return output_path
