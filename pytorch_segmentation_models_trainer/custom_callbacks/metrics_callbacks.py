# -*- coding: utf-8 -*-
"""
/***************************************************************************
 pytorch_segmentation_models_trainer
                              -------------------
        begin                : 2025-09-29
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
import datetime
import os
from pathlib import Path
from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
import seaborn as sns
import torch
from pytorch_lightning.utilities.rank_zero import rank_zero_only
import torchmetrics


class ConfusionMatrixCallback(pl.callbacks.Callback):
    def __init__(
        self,
        num_classes: int,
        class_names: Optional[List[str]] = None,
        normalize: str = 'true',
        log_every_n_epochs: int = 10,
        figsize: tuple = (12, 10),
        output_path: str = None,
    ) -> None:
        """
        Callback para plotar matriz de confusão durante validação.
        
        Args:
            num_classes: Número de classes
            class_names: Lista com nomes das classes. Se None, usa índices.
            normalize: Como normalizar a matriz ('true', 'pred', 'all', None)
            log_every_n_epochs: A cada quantas épocas plotar a matriz
            figsize: Tamanho da figura
            output_path: Caminho para salvar as imagens. Se None, usa log_dir do trainer.
        """
        super().__init__()
        self.num_classes = num_classes
        self.class_names = class_names or [f'Class {i}' for i in range(num_classes)]
        self.normalize = normalize
        self.log_every_n_epochs = log_every_n_epochs
        self.figsize = figsize
        self.output_path = output_path
        self.save_outputs = False
        
        # Use TorchMetrics for efficient memory usage
        self.confmat = None

    def on_sanity_check_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        """Configurar o path de saída após sanity check."""
        self.save_outputs = True
        if self.output_path is None:
            self.output_path = os.path.join(trainer.log_dir, "confusion_matrices")
        if not os.path.exists(self.output_path):
            Path(self.output_path).mkdir(parents=True, exist_ok=True)

    def on_validation_epoch_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        """Inicializar/resetar a matriz de confusão no início da validação."""
        self.confmat = torchmetrics.ConfusionMatrix(
            task="multiclass",
            num_classes=self.num_classes
        ).to(pl_module.device)

    def on_validation_batch_end(
        self, 
        trainer: pl.Trainer, 
        pl_module: pl.LightningModule, 
        outputs,
        batch, 
        batch_idx: int, 
        dataloader_idx: int = 0
    ):
        """Atualizar matriz de confusão durante validação."""
        if self.confmat is None:
            return
            
        if hasattr(batch, 'keys'):
            # Para dataset que retorna dict
            images, targets = batch['image'], batch['mask']
        else:
            # Para dataset que retorna tuple
            images, targets = batch
        
        # Obter predições do modelo
        with torch.no_grad():
            predictions = pl_module(images)
            
        # Converter para classes preditas
        if predictions.dim() > 2:
            predictions = torch.argmax(predictions, dim=1)
        if targets.dim() == 4:  # [B, C, H, W]
            if targets.shape[1] > 1:  # Multi-class one-hot
                targets = torch.argmax(targets, dim=1)
            else:  # Single channel [B, 1, H, W] -> [B, H, W]
                targets = targets.squeeze(1)
        elif targets.dim() == 3:  # Already [B, H, W]
            pass
        
        # Flatten tensors - permanece na GPU para eficiência
        predictions_flat = predictions.flatten()
        targets_flat = targets.flatten()
        
        # Atualizar matriz de confusão (operação eficiente na GPU)
        self.confmat.update(predictions_flat, targets_flat)

    @rank_zero_only
    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        """Plotar matriz de confusão no final da época de validação."""
        if not self.save_outputs or self.confmat is None:
            return
        
        current_epoch = trainer.current_epoch
        
        # Verificar se deve plotar nesta época
        if current_epoch % self.log_every_n_epochs != 0:
            return
        
        try:
            # Computar matriz de confusão final e mover para CPU
            cm = self.confmat.compute().cpu().numpy()
            
            # Normalizar se solicitado
            if self.normalize == 'true':
                cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
                cm = np.nan_to_num(cm)  # Substituir NaN por 0
                fmt = '.2f'
                title = 'Normalized Confusion Matrix (by true labels)'
            elif self.normalize == 'pred':
                cm = cm.astype('float') / cm.sum(axis=0)[np.newaxis, :]
                cm = np.nan_to_num(cm)
                fmt = '.2f'
                title = 'Normalized Confusion Matrix (by predictions)'
            elif self.normalize == 'all':
                cm = cm.astype('float') / cm.sum()
                fmt = '.2f'
                title = 'Normalized Confusion Matrix (by all)'
            else:
                fmt = 'd'
                title = 'Confusion Matrix'
            
            # Criar figura
            plt.figure(figsize=self.figsize)
            sns.heatmap(
                cm,
                annot=True,
                fmt=fmt,
                cmap='Blues',
                xticklabels=self.class_names,
                yticklabels=self.class_names,
                cbar_kws={'shrink': 0.8}
            )
            
            plt.title(f'{title} - Epoch {current_epoch}', fontsize=16, pad=20)
            plt.xlabel('Predicted Labels', fontsize=14)
            plt.ylabel('True Labels', fontsize=14)
            plt.xticks(rotation=45, ha='right')
            plt.yticks(rotation=0)
            plt.tight_layout()
            
            # Salvar figura
            if self.save_outputs:
                timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
                filename = f"confusion_matrix_epoch_{current_epoch:03d}_{timestamp}.png"
                filepath = os.path.join(self.output_path, filename)
                plt.savefig(filepath, dpi=300, bbox_inches='tight')
                
                # Log para TensorBoard se disponível
                if trainer.logger:
                    # Converter figura para imagem numpy
                    fig = plt.gcf()
                    fig.canvas.draw()
                    # Use buffer_rgba() and convert to RGB
                    img_array = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
                    img_array = img_array.reshape(fig.canvas.get_width_height()[::-1] + (4,))
                    img_array = img_array[:, :, :3]  # Drop alpha channel to get RGB
                    img_tensor = torch.from_numpy(img_array).permute(2, 0, 1)
                    
                    trainer.logger.experiment.add_image(
                        f"confusion_matrix/epoch_{current_epoch}",
                        img_tensor,
                        current_epoch
                    )
            
            plt.close()
            
            # Calcular métricas adicionais por classe
            # Usar a matriz não-normalizada para cálculos
            if self.normalize is None:
                cm_raw = cm
            else:
                # Recomputar matriz sem normalização para métricas
                cm_raw = self.confmat.compute().cpu().numpy()
            
            class_accuracy = cm_raw.diagonal() / cm_raw.sum(axis=1)
            class_precision = cm_raw.diagonal() / cm_raw.sum(axis=0)
            
            # Log métricas por classe
            for i, class_name in enumerate(self.class_names):
                if trainer.logger and not np.isnan(class_accuracy[i]):
                    trainer.logger.experiment.add_scalar(
                        f"metrics_by_class/accuracy_{class_name}",
                        class_accuracy[i],
                        current_epoch
                    )
                if trainer.logger and not np.isnan(class_precision[i]):
                    trainer.logger.experiment.add_scalar(
                        f"metrics_by_class/precision_{class_name}",
                        class_precision[i],
                        current_epoch
                    )
        
        except Exception as e:
            print(f"Erro ao plotar matriz de confusão: {e}")
            import traceback
            traceback.print_exc()

class ClassificationReportCallback(pl.callbacks.Callback):
    """
    Callback adicional para gerar relatório de classificação detalhado.
    """
    def __init__(
        self,
        num_classes: int,
        class_names: Optional[List[str]] = None,
        log_every_n_epochs: int = 10,
        output_path: str = None,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.class_names = class_names or [f'Class {i}' for i in range(num_classes)]
        self.log_every_n_epochs = log_every_n_epochs
        self.output_path = output_path
        self.save_outputs = False
        
        self.val_predictions = []
        self.val_targets = []

    def on_sanity_check_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        self.save_outputs = True
        if self.output_path is None:
            self.output_path = os.path.join(trainer.log_dir, "classification_reports")
        if not os.path.exists(self.output_path):
            Path(self.output_path).mkdir(parents=True, exist_ok=True)

    def on_validation_batch_end(
        self, 
        trainer: pl.Trainer, 
        pl_module: pl.LightningModule,
        outputs, 
        batch, 
        batch_idx: int, 
        dataloader_idx: int = 0
    ):
        if hasattr(batch, 'keys'):
            images, targets =batch['image'], batch['mask']
        else:
            images, targets = batch
        
        with torch.no_grad():
            predictions = pl_module(images)
            
        if predictions.dim() > 2:
            predictions = torch.argmax(predictions, dim=1)
        if targets.dim() > 2:
            targets = torch.argmax(targets, dim=1) if targets.shape[1] > 1 else targets.squeeze(1)
        
        predictions_flat = predictions.flatten().cpu().numpy()
        targets_flat = targets.flatten().cpu().numpy()
        assert predictions_flat.shape == targets_flat.shape, \
            f"Shape mismatch: predictions {predictions_flat.shape} vs targets {targets_flat.shape}"
    
        
        self.val_predictions.extend(predictions_flat)
        self.val_targets.extend(targets_flat)

    @rank_zero_only
    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        if not self.save_outputs:
            return
        
        if len(self.val_predictions) != len(self.val_targets):
            print(f"WARNING: Prediction count ({len(self.val_predictions)}) != "
                f"Target count ({len(self.val_targets)}). Skipping confusion matrix.")
            self.val_predictions = []
            self.val_targets = []
            return
        
        current_epoch = trainer.current_epoch
        
        if current_epoch % self.log_every_n_epochs != 0:
            self.val_predictions = []
            self.val_targets = []
            return
        
        if len(self.val_predictions) == 0 or len(self.val_targets) == 0:
            return
        
        try:
            from sklearn.metrics import classification_report
            
            predictions = np.array(self.val_predictions)
            targets = np.array(self.val_targets)
            
            # Gerar relatório
            report = classification_report(
                targets,
                predictions,
                target_names=self.class_names,
                labels=list(range(self.num_classes)),
                digits=4
            )
            
            # Salvar relatório
            if self.save_outputs:
                timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
                filename = f"classification_report_epoch_{current_epoch:03d}_{timestamp}.txt"
                filepath = os.path.join(self.output_path, filename)
                
                with open(filepath, 'w') as f:
                    f.write(f"Classification Report - Epoch {current_epoch}\n")
                    f.write("=" * 50 + "\n\n")
                    f.write(report)
                    f.write(f"\n\nGenerated at: {datetime.datetime.now()}\n")
                
                print(f"\nClassification Report - Epoch {current_epoch}")
                print("=" * 50)
                print(report)
        
        except Exception as e:
            print(f"Erro ao gerar relatório de classificação: {e}")
        
        finally:
            self.val_predictions = []
            self.val_targets = []