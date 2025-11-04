# -*- coding: utf-8 -*-
"""
/***************************************************************************
 pytorch_segmentation_models_trainer
                              -------------------
        begin                : 2025-11-04
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

import os
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import rasterio
from tqdm import tqdm

import logging
logger = logging.getLogger(__name__)


class DirectFolderEvaluator:
    """
    Avalia predições comparando diretamente duas pastas.
    
    Não requer CSV - faz matching automático por nome de arquivo.
    """
    
    def __init__(
        self,
        ground_truth_folder: str,
        predictions_folder: str,
        gt_pattern: str = "*.tif",
        pred_pattern: str = "*.tif"
    ):
        """
        Args:
            ground_truth_folder: Pasta com máscaras ground truth
            predictions_folder: Pasta com predições
            gt_pattern: Padrão para buscar ground truth (default: "*.tif")
            pred_pattern: Padrão para buscar predições (default: "*.tif")
        """
        self.gt_folder = Path(ground_truth_folder)
        self.pred_folder = Path(predictions_folder)
        self.gt_pattern = gt_pattern
        self.pred_pattern = pred_pattern
        
        if not self.gt_folder.exists():
            raise FileNotFoundError(f"Ground truth folder not found: {ground_truth_folder}")
        
        if not self.pred_folder.exists():
            raise FileNotFoundError(f"Predictions folder not found: {predictions_folder}")
    
    def build_matching_pairs(self) -> List[Dict[str, str]]:
        """
        Constrói lista de pares (ground_truth, prediction) com matching por nome.
        
        Returns:
            Lista de dicts com keys: 'name', 'gt_path', 'pred_path'
        """
        logger.info("Building matching pairs...")
        
        # 1. Buscar todos ground truth files
        gt_files = list(self.gt_folder.glob(self.gt_pattern))
        logger.info(f"Found {len(gt_files)} ground truth files")
        
        if len(gt_files) == 0:
            raise ValueError(f"No ground truth files found with pattern {self.gt_pattern}")
        
        # 2. Criar mapa de predições por stem
        pred_map = self._build_prediction_map()
        logger.info(f"Found {len(pred_map)} prediction files")
        
        # 3. Fazer matching
        pairs = []
        not_found = []
        
        for gt_path in gt_files:
            gt_stem = gt_path.stem
            
            # Tentar variantes do stem
            variants = self._generate_stem_variants(gt_stem)
            
            pred_path = None
            matched_variant = None
            
            for variant in variants:
                if variant in pred_map:
                    pred_path = pred_map[variant]
                    matched_variant = variant
                    break
            
            if pred_path:
                pairs.append({
                    'name': gt_stem,
                    'gt_path': str(gt_path),
                    'pred_path': pred_path,
                    'matched_as': matched_variant
                })
            else:
                not_found.append(gt_stem)
        
        # 4. Log resultados
        logger.info(f"Matched {len(pairs)} pairs")
        
        if not_found:
            logger.warning(f"Could not find predictions for {len(not_found)} ground truth files:")
            for name in not_found[:5]:
                logger.warning(f"  - {name}")
            if len(not_found) > 5:
                logger.warning(f"  ... and {len(not_found) - 5} more")
        
        if len(pairs) == 0:
            raise ValueError("No matching pairs found! Check file names.")
        
        return pairs
    
    def _build_prediction_map(self) -> Dict[str, str]:
        """
        Constrói mapa de stems -> paths para predições.
        
        Returns:
            Dict mapeando stem -> full_path
        """
        pred_map = {}
        
        # Buscar todos arquivos
        patterns = [self.pred_pattern, "*.tiff", "*.TIF", "*.TIFF"]
        all_files = []
        
        for pattern in patterns:
            all_files.extend(self.pred_folder.glob(pattern))
        
        # Remover duplicatas
        all_files = list(set(all_files))
        
        # Construir mapa com variantes
        for file_path in all_files:
            stem = file_path.stem
            variants = self._generate_stem_variants(stem)
            
            for variant in variants:
                if variant not in pred_map:
                    pred_map[variant] = str(file_path)
        
        return pred_map
    
    def _generate_stem_variants(self, stem: str) -> List[str]:
        """
        Gera variantes do stem removendo prefixos/sufixos comuns.
        """
        variants = [stem]
        
        prefixes = ['seg_', 'pred_', 'output_', 'mask_']
        suffixes = ['_output', '_pred', '_prediction', '_seg', '_mask']
        
        # Remover prefixo
        temp = stem
        for prefix in prefixes:
            if temp.startswith(prefix):
                temp = temp[len(prefix):]
                variants.append(temp)
                break
        
        # Remover sufixo
        temp = stem
        for suffix in suffixes:
            if temp.endswith(suffix):
                temp = temp[:-len(suffix)]
                if temp not in variants:
                    variants.append(temp)
                break
        
        # Remover ambos
        temp = stem
        for prefix in prefixes:
            if temp.startswith(prefix):
                temp = temp[len(prefix):]
                break
        for suffix in suffixes:
            if temp.endswith(suffix):
                temp = temp[:-len(suffix)]
                break
        if temp not in variants:
            variants.append(temp)
        
        return variants
    
    def create_evaluation_csv(self, output_path: str) -> pd.DataFrame:
        """
        Cria CSV para avaliação a partir dos pares matched.
        
        Este CSV pode ser usado pelo MetricsCalculator existente.
        
        Args:
            output_path: Onde salvar o CSV
            
        Returns:
            DataFrame com colunas: image, mask, width, height
        """
        pairs = self.build_matching_pairs()
        
        rows = []
        for pair in tqdm(pairs, desc="Creating CSV"):
            # Carregar dimensões do ground truth
            with rasterio.open(pair['gt_path']) as src:
                width = src.width
                height = src.height
            
            rows.append({
                'image': pair['gt_path'],  # Usar GT como "image" também
                'mask': pair['gt_path'],
                'width': width,
                'height': height
            })
        
        df = pd.DataFrame(rows)
        
        # Salvar
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
        
        logger.info(f"Evaluation CSV created: {output_path}")
        logger.info(f"  Total pairs: {len(df)}")
        
        return df
    
    def load_pair(self, pair: Dict) -> Tuple[np.ndarray, np.ndarray]:
        """
        Carrega um par de ground truth e predição.
        
        Args:
            pair: Dict com 'gt_path' e 'pred_path'
            
        Returns:
            (gt_mask, pred_mask) como numpy arrays
        """
        with rasterio.open(pair['gt_path']) as src:
            gt_mask = src.read(1)
        
        with rasterio.open(pair['pred_path']) as src:
            pred_mask = src.read(1)
        
        return gt_mask, pred_mask


# ============================================================================
# FUNÇÃO HELPER PARA INTEGRATION NO PIPELINE
# ============================================================================

def prepare_evaluation_csv_from_folders(
    ground_truth_folder: str,
    predictions_folder: str,
    output_csv_path: str,
    gt_pattern: str = "*.tif",
    pred_pattern: str = "*.tif"
) -> str:
    """
    Função helper que cria CSV de avaliação diretamente de pastas.
    
    Pode ser chamada no início do evaluation pipeline.
    
    Args:
        ground_truth_folder: Pasta com ground truth
        predictions_folder: Pasta com predições
        output_csv_path: Onde salvar CSV
        gt_pattern: Padrão de busca para GT
        pred_pattern: Padrão de busca para predições
        
    Returns:
        Path do CSV criado
    """
    evaluator = DirectFolderEvaluator(
        ground_truth_folder=ground_truth_folder,
        predictions_folder=predictions_folder,
        gt_pattern=gt_pattern,
        pred_pattern=pred_pattern
    )
    
    df = evaluator.create_evaluation_csv(output_csv_path)
    
    return output_csv_path


if __name__ == "__main__":
    # Exemplo de uso standalone
    import argparse
    
    parser = argparse.ArgumentParser(description="Direct Folder Evaluation")
    parser.add_argument("--gt-folder", required=True, help="Ground truth folder")
    parser.add_argument("--pred-folder", required=True, help="Predictions folder")
    parser.add_argument("--output-csv", required=True, help="Output CSV path")
    parser.add_argument("--gt-pattern", default="*.tif", help="GT file pattern")
    parser.add_argument("--pred-pattern", default="*.tif", help="Pred file pattern")
    
    args = parser.parse_args()
    
    prepare_evaluation_csv_from_folders(
        ground_truth_folder=args.gt_folder,
        predictions_folder=args.pred_folder,
        output_csv_path=args.output_csv,
        gt_pattern=args.gt_pattern,
        pred_pattern=args.pred_pattern
    )
    
    print(f"✓ CSV created: {args.output_csv}")