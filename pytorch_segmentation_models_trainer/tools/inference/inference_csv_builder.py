# -*- coding: utf-8 -*-
"""
/***************************************************************************
 pytorch_segmentation_models_trainer
                              -------------------
        begin                : 2025-11-03
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
from typing import Optional, List, Tuple

import pandas as pd
import rasterio
from tqdm import tqdm

logger = logging.getLogger(__name__)


class InferenceCSVBuilder:
    """
    Constrói CSV de inferência a partir de pasta de imagens.
    
    Suporta:
    - Busca automática de imagens em pasta
    - Busca opcional de máscaras correspondentes
    - Detecção automática de dimensões
    - Múltiplos padrões de arquivo
    """
    
    def __init__(
        self,
        images_folder: str,
        image_pattern: str = "*.tif",
        recursive: bool = False,
        masks_folder: Optional[str] = None,
        mask_pattern: Optional[str] = None,
        mask_suffix: Optional[str] = "",
        root_dir: Optional[str] = None
    ):
        """
        Args:
            images_folder: Pasta com imagens para inferência
            image_pattern: Padrão glob para buscar imagens (e.g., "*.tif", "*.png")
            recursive: Se True, busca recursivamente em subpastas
            masks_folder: Pasta com máscaras (opcional, para validação)
            mask_pattern: Padrão glob para buscar máscaras
            mask_suffix: Sufixo esperado nas máscaras (e.g., "_mask")
            root_dir: Diretório raiz para paths relativos (opcional)
        """
        self.images_folder = Path(images_folder)
        self.image_pattern = image_pattern
        self.recursive = recursive
        self.masks_folder = Path(masks_folder) if masks_folder else None
        self.mask_pattern = mask_pattern or image_pattern
        self.mask_suffix = mask_suffix
        self.root_dir = Path(root_dir) if root_dir else None
        
        # Validações
        if not self.images_folder.exists():
            raise FileNotFoundError(f"Images folder not found: {images_folder}")
        
        if self.masks_folder and not self.masks_folder.exists():
            raise FileNotFoundError(f"Masks folder not found: {masks_folder}")
    
    def find_images(self) -> List[Path]:
        """
        Encontra todas as imagens na pasta.
        
        Returns:
            Lista de paths de imagens
        """
        if self.recursive:
            images = list(self.images_folder.rglob(self.image_pattern))
        else:
            images = list(self.images_folder.glob(self.image_pattern))
        
        # Ordenar para consistência
        images = sorted(images)
        
        logger.info(f"Found {len(images)} images in {self.images_folder}")
        
        return images
    
    def find_corresponding_mask(self, image_path: Path) -> Optional[Path]:
        """
        Encontra máscara correspondente a uma imagem.
        
        Args:
            image_path: Path da imagem
            
        Returns:
            Path da máscara ou None se não encontrada
        """
        if not self.masks_folder:
            return None
        
        # Estratégia 1: Nome exato com sufixo
        # image.tif -> image_mask.tif
        mask_name = f"{image_path.stem}{self.mask_suffix}{image_path.suffix}"
        mask_path = self.masks_folder / mask_name
        
        if mask_path.exists():
            return mask_path
        
        # Estratégia 2: Nome exato sem sufixo
        # image.tif -> image.tif
        mask_path = self.masks_folder / image_path.name
        
        if mask_path.exists():
            return mask_path
        
        # Estratégia 3: Buscar por padrão
        # Procurar qualquer arquivo que comece com o stem da imagem
        candidates = list(self.masks_folder.glob(f"{image_path.stem}*{image_path.suffix}"))
        
        if len(candidates) == 1:
            return candidates[0]
        elif len(candidates) > 1:
            logger.warning(
                f"Multiple mask candidates found for {image_path.name}: "
                f"{[c.name for c in candidates]}. Using first one."
            )
            return candidates[0]
        
        return None
    
    def get_image_dimensions(self, image_path: Path) -> Tuple[int, int]:
        """
        Obtém dimensões de uma imagem.
        
        Args:
            image_path: Path da imagem
            
        Returns:
            Tupla (width, height)
        """
        try:
            with rasterio.open(image_path) as src:
                width = src.width
                height = src.height
            return width, height
        except Exception as e:
            logger.warning(f"Could not read dimensions of {image_path}: {e}")
            return 0, 0
    
    def build_csv(
        self,
        output_csv_path: str,
        validate_images: bool = True
    ) -> pd.DataFrame:
        """
        Constrói CSV de inferência.
        
        Args:
            output_csv_path: Path de saída do CSV
            validate_images: Se True, valida que imagens podem ser abertas
            
        Returns:
            DataFrame construído
        """
        logger.info("Building inference CSV from folder...")
        logger.info(f"  Images folder: {self.images_folder}")
        logger.info(f"  Image pattern: {self.image_pattern}")
        logger.info(f"  Recursive: {self.recursive}")
        
        if self.masks_folder:
            logger.info(f"  Masks folder: {self.masks_folder}")
            logger.info(f"  Mask pattern: {self.mask_pattern}")
        
        # Encontrar imagens
        images = self.find_images()
        
        if len(images) == 0:
            raise ValueError(
                f"No images found in {self.images_folder} with pattern {self.image_pattern}"
            )
        
        # Construir DataFrame
        data = []
        
        for idx, image_path in enumerate(tqdm(images, desc="Processing images")):
            # Path da imagem
            if self.root_dir:
                try:
                    image_str = str(image_path.relative_to(self.root_dir))
                except ValueError:
                    # Se não conseguir fazer relativo, usa absoluto
                    image_str = str(image_path.absolute())
            else:
                image_str = str(image_path.absolute())
            
            # Path da máscara (se houver)
            mask_path = self.find_corresponding_mask(image_path)
            
            if mask_path:
                if self.root_dir:
                    try:
                        mask_str = str(mask_path.relative_to(self.root_dir))
                    except ValueError:
                        mask_str = str(mask_path.absolute())
                else:
                    mask_str = str(mask_path.absolute())
            else:
                mask_str = ""
            
            # Dimensões
            width, height = self.get_image_dimensions(image_path)
            
            if width == 0 or height == 0:
                logger.warning(f"Skipping invalid image: {image_path}")
                continue
        
            data.append({
                'id': idx,
                'image': image_str,
                'mask': mask_str,
                'width': width,
                'height': height
            })
        
        # Criar DataFrame
        df = pd.DataFrame(data)
        
        # Salvar CSV
        output_path = Path(output_csv_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        df.to_csv(output_csv_path, index=False)
        
        logger.info(f"CSV built successfully: {output_csv_path}")
        logger.info(f"  Total images: {len(df)}")
        
        if self.masks_folder:
            masks_found = df['mask'].astype(bool).sum()
            logger.info(f"  Masks found: {masks_found} ({masks_found/len(df)*100:.1f}%)")
        
        if 'width' in df.columns and 'height' in df.columns:
            unique_dims = df.groupby(['width', 'height']).size()
            logger.info(f"  Unique dimensions: {len(unique_dims)}")
            for (w, h), count in unique_dims.items():
                logger.info(f"    {w}x{h}: {count} images")
        
        return df
    
    def load_or_build_csv(
        self,
        output_csv_path: str,
        force_rebuild: bool = False
    ) -> pd.DataFrame:
        """
        Carrega CSV existente ou constrói novo.
        
        Args:
            output_csv_path: Path do CSV
            force_rebuild: Se True, reconstrói mesmo se existir
            
        Returns:
            DataFrame
        """
        csv_path = Path(output_csv_path)
        
        if csv_path.exists() and not force_rebuild:
            logger.info(f"Loading existing CSV: {output_csv_path}")
            df = pd.read_csv(output_csv_path)
            logger.info(f"  Loaded {len(df)} images")
            return df
        
        logger.info(f"Building new CSV: {output_csv_path}")
        return self.build_csv(output_csv_path)


def build_inference_csv_from_config(config) -> str:
    """
    Constrói CSV de inferência a partir de config do Hydra.
    
    Args:
        config: DictConfig com seção build_csv_from_folder
        
    Returns:
        Path do CSV construído
    """
    builder = InferenceCSVBuilder(
        images_folder=config.images_folder,
        image_pattern=config.get('image_pattern', '*.tif'),
        recursive=config.get('recursive', False),
        masks_folder=config.get('masks_folder'),
        mask_pattern=config.get('mask_pattern'),
        mask_suffix=config.get('mask_suffix', ''),
        root_dir=config.get('root_dir')
    )
    
    # Determinar path de saída
    if 'output_csv_path' in config and config.output_csv_path:
        output_csv_path = config.output_csv_path
    else:
        # MUDANÇA: Adicionar hash único para evitar conflitos em paralelo
        import hashlib
        import time
        
        # Criar identificador único baseado na pasta e timestamp
        folder_hash = hashlib.md5(
            config.images_folder.encode()
        ).hexdigest()[:8]
        
        timestamp = str(int(time.time() * 1000))  # milliseconds
        
        # Gerar path com identificador único
        output_csv_path = os.path.join(
            config.images_folder,
            f"inference_dataset_{folder_hash}_{timestamp}.csv"
        )
    
    # Construir ou carregar
    force_rebuild = config.get('force_rebuild', False)
    
    df = builder.load_or_build_csv(output_csv_path, force_rebuild)
    
    return output_csv_path
