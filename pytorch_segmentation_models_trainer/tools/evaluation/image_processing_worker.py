# -*- coding: utf-8 -*-
# -*- coding: utf-8 -*-
"""
/***************************************************************************
 pytorch_segmentation_models_trainer
                              -------------------
        begin                : 2025-11-05
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
from pathlib import Path
from typing import Dict, Tuple, Optional

import numpy as np
import torch
import rasterio
from rasterio.windows import Window, from_bounds
from rasterio.coords import BoundingBox

logger = logging.getLogger(__name__)


def process_single_image_worker(
    task: Dict,
    num_classes: int,
    class_names: list
) -> Optional[Dict]:
    """
    Worker function para processar uma única imagem.
    Executado em processo separado.
    
    Args:
        task: Dict com pred_path, gt_path, image_name, index
        num_classes: Número de classes
        class_names: Lista de nomes das classes
        
    Returns:
        Dict com resultados ou None se falhar
    """
    try:
        pred_path = task['pred_path']
        gt_path = task['gt_path']
        image_name = task['image_name']
        
        # Ler e alinhar rasters
        pred_mask, gt_mask = read_aligned_rasters_worker(
            pred_path, gt_path, num_classes
        )
        
        if pred_mask is None or gt_mask is None:
            return None
        
        # Converter para tensors e preparar para agregação
        pred_tensor = torch.from_numpy(pred_mask).long().flatten()
        gt_tensor = torch.from_numpy(gt_mask).long().flatten()
        
        # Retornar dados brutos para cálculo de métricas no processo principal
        return {
            'image_name': image_name,
            'index': task.get('index', 0),
            'pred_flat': pred_tensor.numpy(),  # NumPy array para serialização
            'gt_flat': gt_tensor.numpy(),
            'shape': pred_mask.shape,
            'num_pixels': pred_mask.size
        }
        
    except Exception as e:
        logger.error(f"Worker error for {task.get('image_name', 'unknown')}: {e}")
        return None


def read_aligned_rasters_worker(
    pred_path: str,
    gt_path: str,
    num_classes: int
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Lê dois rasters garantindo que as áreas lidas correspondem espacialmente.
    Versão standalone para workers.
    
    Args:
        pred_path: Caminho da predição
        gt_path: Caminho do ground truth
        num_classes: Número de classes para validação
        
    Returns:
        (pred_array, gt_array) com mesma shape, ou (None, None) se falhar
    """
    try:
        # Calcular overlap espacial
        overlap_result = get_spatial_overlap_worker(pred_path, gt_path)
        
        if overlap_result is None:
            return None, None
        
        pred_window, gt_window, expected_shape = overlap_result
        
        # Ler apenas a área de overlap de cada raster
        with rasterio.open(pred_path) as pred_src:
            pred_array = pred_src.read(1, window=pred_window)
        
        with rasterio.open(gt_path) as gt_src:
            gt_array = gt_src.read(1, window=gt_window)
        
        # Verificação de shapes
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
        
        # Validar e limpar os dados
        pred_array = pred_array.astype(np.int32)
        gt_array = gt_array.astype(np.int32)
        
        # Clipar valores para o range válido
        pred_array = np.clip(pred_array, 0, num_classes - 1)
        gt_array = np.clip(gt_array, 0, num_classes - 1)
        
        # Verificar NaN
        if np.isnan(pred_array).any():
            logger.warning(f"NaN values in prediction {pred_path}, replacing with 0")
            pred_array = np.nan_to_num(pred_array, nan=0.0).astype(np.int32)
        
        if np.isnan(gt_array).any():
            logger.warning(f"NaN values in ground truth {gt_path}, replacing with 0")
            gt_array = np.nan_to_num(gt_array, nan=0.0).astype(np.int32)
        
        return pred_array, gt_array
        
    except Exception as e:
        logger.error(f"Error aligning rasters: {e}")
        return None, None


def get_spatial_overlap_worker(
    pred_path: str,
    gt_path: str
) -> Optional[Tuple[Window, Window, Tuple[int, int]]]:
    """
    Calcula a área de overlap espacial entre dois rasters.
    Versão standalone para workers.
    
    Args:
        pred_path: Caminho da predição
        gt_path: Caminho do ground truth
        
    Returns:
        (pred_window, gt_window, matched_shape) ou None se não houver overlap
    """
    try:
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
            
    except Exception as e:
        logger.error(f"Error calculating spatial overlap: {e}")
        return None