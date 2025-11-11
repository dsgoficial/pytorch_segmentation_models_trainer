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
import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import rasterio
from omegaconf import DictConfig
from tqdm import tqdm

logger = logging.getLogger(__name__)


class DatasetCSVBuilder:
    """
    Constrói CSV de dataset a partir de pastas de imagens e máscaras.
    
    O CSV gerado contém as colunas:
    - image: caminho da imagem
    - mask: caminho da máscara correspondente
    - width: largura da imagem
    - height: altura da imagem
    
    Suporta diferentes estratégias de matching:
    - same_basename: imagem e máscara têm o mesmo nome base
    - prefix_suffix: usa prefixos/sufixos diferentes
    - custom_regex: usa regex customizado para extrair ID comum
    """
    
    def __init__(self, config: DictConfig):
        """
        Args:
            config: DictConfig com campos:
                - images_folder: pasta com imagens
                - masks_folder: pasta com máscaras
                - image_pattern: padrão glob para imagens (ex: "*.tif")
                - mask_pattern: padrão glob para máscaras
                - matching_strategy: "same_basename", "prefix_suffix", ou "custom_regex" (opcional)
                - image_prefix: prefixo das imagens (opcional)
                - mask_prefix: prefixo das máscaras (opcional)
                - image_suffix: sufixo das imagens (opcional)
                - mask_suffix: sufixo das máscaras (opcional)
                - regex_pattern: padrão regex (opcional)
                - recursive: buscar recursivamente (opcional)
        """
        self.config = config
        self.images_folder = Path(config.images_folder)
        self.masks_folder = Path(config.masks_folder)
        self.image_pattern = getattr(config, 'image_pattern', '*.tif')
        self.mask_pattern = getattr(config, 'mask_pattern', '*.tif')
        
        # Matching strategy com valor padrão
        self.matching_strategy = getattr(config, 'matching_strategy', 'same_basename')
        
        # Recursive search (padrão: False)
        self.recursive = getattr(config, 'recursive', False)
        
        # Parâmetros opcionais para matching
        self.image_prefix = getattr(config, 'image_prefix', '')
        self.mask_prefix = getattr(config, 'mask_prefix', '')
        self.image_suffix = getattr(config, 'image_suffix', '')
        self.mask_suffix = getattr(config, 'mask_suffix', '')
        self.regex_pattern = getattr(config, 'regex_pattern', None)
        
        # Validar pastas
        if not self.images_folder.exists():
            raise FileNotFoundError(f"Images folder not found: {self.images_folder}")
        if not self.masks_folder.exists():
            raise FileNotFoundError(f"Masks folder not found: {self.masks_folder}")
        
        logger.info(f"DatasetCSVBuilder initialized:")
        logger.info(f"  Images folder: {self.images_folder}")
        logger.info(f"  Masks folder: {self.masks_folder}")
        logger.info(f"  Matching strategy: {self.matching_strategy}")
        logger.info(f"  Recursive: {self.recursive}")
    
    def build_csv(self, output_path: str) -> pd.DataFrame:
        """
        Constrói CSV com colunas: image, mask, width, height.
        
        Args:
            output_path: caminho onde salvar o CSV
            
        Returns:
            DataFrame com o dataset construído
        """
        logger.info("Building dataset CSV...")
        
        # 1. Encontrar todas as imagens
        image_files = self._find_files(self.images_folder, self.image_pattern)
        logger.info(f"Found {len(image_files)} images")
        
        if len(image_files) == 0:
            raise ValueError(
                f"No images found in {self.images_folder} "
                f"with pattern {self.image_pattern}"
            )
        
        # 2. Encontrar máscaras correspondentes
        dataset_rows = []
        not_found = []
        
        for image_path in tqdm(image_files, desc="Matching images to masks"):
            mask_path = self.match_image_to_mask(image_path)
            
            if mask_path is None:
                not_found.append(image_path)
                continue
            
            # Extrair dimensões
            width, height = self._get_image_dimensions(image_path)
            
            dataset_rows.append({
                'image': str(image_path),
                'mask': str(mask_path),
                'width': width,
                'height': height
            })
        
        # 3. Validar resultados
        if len(not_found) > 0:
            logger.warning(
                f"{len(not_found)} images without corresponding masks found:"
            )
            for img in not_found[:5]:  # Mostrar só os primeiros 5
                logger.warning(f"  - {img}")
            if len(not_found) > 5:
                logger.warning(f"  ... and {len(not_found) - 5} more")
        
        if len(dataset_rows) == 0:
            raise ValueError("No valid image-mask pairs found!")
        
        # 4. Criar DataFrame
        df = pd.DataFrame(dataset_rows)
        
        # 5. Validar dataset
        self.validate_dataset(df)
        
        # 6. Salvar CSV
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
        
        logger.info(f"Dataset CSV saved: {output_path}")
        logger.info(f"Total pairs: {len(df)}")
        logger.info(f"Unique dimensions: {df[['width', 'height']].drop_duplicates().shape[0]}")
        
        return df
    
    def match_image_to_mask(self, image_path: Path) -> Optional[Path]:
        """
        Encontra máscara correspondente para uma imagem.
        
        Args:
            image_path: caminho da imagem
            
        Returns:
            Path da máscara ou None se não encontrada
        """
        if self.matching_strategy == "same_basename":
            return self._match_same_basename(image_path)
        
        elif self.matching_strategy == "prefix_suffix":
            return self._match_prefix_suffix(image_path)
        
        elif self.matching_strategy == "custom_regex":
            return self._match_custom_regex(image_path)
        
        else:
            raise ValueError(
                f"Unknown matching strategy: {self.matching_strategy}"
            )
    
    def _match_same_basename(self, image_path: Path) -> Optional[Path]:
        """
        Matching simples: mesmo nome base (stem).
        
        Exemplo:
            image_001.tif -> mask_001.tif
            (desde que ambos tenham stem = "001" ou "image_001" vs "mask_001")
        """
        # Tentar com extensão da máscara
        mask_extension = self.mask_pattern.replace("*", "")
        mask_path = self.masks_folder / f"{image_path.stem}{mask_extension}"
        
        if mask_path.exists():
            return mask_path
        
        # Tentar buscar qualquer arquivo com mesmo stem
        for mask_file in self.masks_folder.glob(f"{image_path.stem}.*"):
            return mask_file
        
        return None
    
    def _match_prefix_suffix(self, image_path: Path) -> Optional[Path]:
        """
        Matching com prefixos/sufixos diferentes.
        
        Exemplo:
            image_prefix="image_", mask_prefix="mask_"
            image_001.tif -> mask_001.tif
        """
        image_prefix = self.config.get("image_prefix", "")
        mask_prefix = self.config.get("mask_prefix", "")
        image_suffix = self.config.get("image_suffix", "")
        mask_suffix = self.config.get("mask_suffix", "")
        
        # Remover prefixo e sufixo da imagem
        stem = image_path.stem
        if image_prefix and stem.startswith(image_prefix):
            stem = stem[len(image_prefix):]
        if image_suffix and stem.endswith(image_suffix):
            stem = stem[:-len(image_suffix)]
        
        # Construir nome da máscara
        mask_stem = f"{mask_prefix}{stem}{mask_suffix}"
        mask_extension = self.mask_pattern.replace("*", "")
        mask_path = self.masks_folder / f"{mask_stem}{mask_extension}"
        
        if mask_path.exists():
            return mask_path
        
        return None
    
    def _match_custom_regex(self, image_path: Path) -> Optional[Path]:
        """
        Matching usando regex customizado.
        
        O regex deve ter um grupo nomeado 'id' que captura o identificador comum.
        
        Exemplo:
            regex_pattern="(?P<id>\\d+)"
            image_abc_123.tif -> mask_xyz_123.tif (ambos capturam "123")
        """
        regex_pattern = self.config.get("regex_pattern")
        if not regex_pattern:
            raise ValueError(
                "custom_regex strategy requires 'regex_pattern' in config"
            )
        
        # Extrair ID da imagem
        match = re.search(regex_pattern, image_path.stem)
        if not match:
            logger.warning(
                f"Regex pattern '{regex_pattern}' did not match image: {image_path}"
            )
            return None
        
        image_id = match.group("id")
        
        # Buscar máscara com mesmo ID
        for mask_file in self.masks_folder.glob(self.mask_pattern):
            mask_match = re.search(regex_pattern, mask_file.stem)
            if mask_match and mask_match.group("id") == image_id:
                return mask_file
        
        return None
    
    def _find_files(self, folder: Path, pattern: str) -> List[Path]:
        """
        Busca arquivos em uma pasta.
        
        Args:
            folder: pasta onde buscar
            pattern: padrão glob (ex: "*.tif")
            
        Returns:
            Lista de Paths encontrados
        """
        if self.recursive:
            return sorted(folder.rglob(pattern))
        else:
            return sorted(folder.glob(pattern))
    
    def _get_image_dimensions(self, image_path: Path) -> Tuple[int, int]:
        """
        Extrai dimensões (width, height) de uma imagem.
        
        Args:
            image_path: caminho da imagem
            
        Returns:
            Tupla (width, height)
        """
        try:
            with rasterio.open(image_path) as src:
                return src.width, src.height
        except Exception as e:
            logger.error(f"Could not read dimensions from {image_path}: {e}")
            return 0, 0
    
    def validate_dataset(self, df: pd.DataFrame) -> bool:
        """
        Valida se o dataset está consistente.
        
        Args:
            df: DataFrame do dataset
            
        Returns:
            True se válido
            
        Raises:
            ValueError se houver problemas
        """
        logger.info("Validating dataset...")
        
        # 1. Verificar se todas as imagens existem
        missing_images = []
        for img_path in df['image']:
            if not Path(img_path).exists():
                missing_images.append(img_path)
        
        if missing_images:
            raise ValueError(
                f"{len(missing_images)} images not found. Examples:\n"
                + "\n".join(missing_images[:3])
            )
        
        # 2. Verificar se todas as máscaras existem
        missing_masks = []
        for mask_path in df['mask']:
            if not Path(mask_path).exists():
                missing_masks.append(mask_path)
        
        if missing_masks:
            raise ValueError(
                f"{len(missing_masks)} masks not found. Examples:\n"
                + "\n".join(missing_masks[:3])
            )
        
        # 3. Verificar se dimensões são válidas
        invalid_dims = df[(df['width'] == 0) | (df['height'] == 0)]
        if len(invalid_dims) > 0:
            logger.warning(
                f"{len(invalid_dims)} images with invalid dimensions (0x0)"
            )
        
        # 4. Verificar duplicatas
        duplicates = df[df.duplicated(subset=['image'], keep=False)]
        if len(duplicates) > 0:
            logger.warning(
                f"{len(duplicates)} duplicate images found"
            )
        
        logger.info("Dataset validation completed!")
        return True
