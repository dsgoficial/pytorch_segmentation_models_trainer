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

import os
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from omegaconf import OmegaConf
from rasterio.transform import from_bounds

from pytorch_segmentation_models_trainer.tools.evaluation.csv_builder import (
    DatasetCSVBuilder
)


class TestDatasetCSVBuilder(unittest.TestCase):
    """
    Testes para DatasetCSVBuilder.
    """
    
    def setUp(self):
        """Setup para cada teste."""
        self.temp_dir = tempfile.mkdtemp()
        self.images_folder = Path(self.temp_dir) / "images"
        self.masks_folder = Path(self.temp_dir) / "masks"
        
        self.images_folder.mkdir(parents=True)
        self.masks_folder.mkdir(parents=True)
        
        # Criar imagens e máscaras de teste
        self._create_test_data()
    
    def tearDown(self):
        """Cleanup após cada teste."""
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def _create_test_data(self):
        """Cria imagens e máscaras de teste."""
        for i in range(5):
            # Criar imagem
            image_path = self.images_folder / f"image_{i:03d}.tif"
            self._create_dummy_tif(image_path, width=256, height=256, bands=3)
            
            # Criar máscara
            mask_path = self.masks_folder / f"mask_{i:03d}.tif"
            self._create_dummy_tif(mask_path, width=256, height=256, bands=1)
    
    def _create_dummy_tif(
        self, 
        path: Path, 
        width: int = 256, 
        height: int = 256, 
        bands: int = 1
    ):
        """Cria arquivo TIF dummy."""
        data = np.random.randint(0, 255, (bands, height, width), dtype=np.uint8)
        
        transform = from_bounds(0, 0, width, height, width, height)
        
        with rasterio.open(
            path,
            'w',
            driver='GTiff',
            height=height,
            width=width,
            count=bands,
            dtype=rasterio.uint8,
            transform=transform
        ) as dst:
            for i in range(bands):
                dst.write(data[i], i + 1)
    
    def test_csv_builder_same_basename(self):
        """Testa construção de CSV com matching same_basename."""
        # Criar imagens e máscaras com mesmo basename
        test_images = self.images_folder / "test_images"
        test_masks = self.masks_folder / "test_masks"
        test_images.mkdir()
        test_masks.mkdir()
        
        for i in range(3):
            image_path = test_images / f"{i:03d}.tif"
            mask_path = test_masks / f"{i:03d}.tif"
            self._create_dummy_tif(image_path)
            self._create_dummy_tif(mask_path)
        
        # Config
        config = OmegaConf.create({
            'images_folder': str(test_images),
            'masks_folder': str(test_masks),
            'image_pattern': '*.tif',
            'mask_pattern': '*.tif',
            'matching_strategy': 'same_basename'
        })
        
        # Construir CSV
        builder = DatasetCSVBuilder(config)
        csv_path = Path(self.temp_dir) / "dataset.csv"
        df = builder.build_csv(str(csv_path))
        
        # Verificações
        self.assertTrue(csv_path.exists())
        self.assertEqual(len(df), 3)
        self.assertIn('image', df.columns)
        self.assertIn('mask', df.columns)
        self.assertIn('width', df.columns)
        self.assertIn('height', df.columns)
        
        # Verificar que todos os arquivos existem
        for _, row in df.iterrows():
            self.assertTrue(Path(row['image']).exists())
            self.assertTrue(Path(row['mask']).exists())
    
    def test_csv_builder_prefix_suffix(self):
        """Testa construção de CSV com matching prefix_suffix."""
        # Criar imagens e máscaras com prefixos diferentes
        test_images = self.images_folder / "test_prefix"
        test_masks = self.masks_folder / "test_prefix"
        test_images.mkdir()
        test_masks.mkdir()
        
        for i in range(3):
            image_path = test_images / f"image_{i:03d}.tif"
            mask_path = test_masks / f"mask_{i:03d}.tif"
            self._create_dummy_tif(image_path)
            self._create_dummy_tif(mask_path)
        
        # Config
        config = OmegaConf.create({
            'images_folder': str(test_images),
            'masks_folder': str(test_masks),
            'image_pattern': 'image_*.tif',
            'mask_pattern': 'mask_*.tif',
            'matching_strategy': 'prefix_suffix',
            'image_prefix': 'image_',
            'mask_prefix': 'mask_'
        })
        
        # Construir CSV
        builder = DatasetCSVBuilder(config)
        csv_path = Path(self.temp_dir) / "dataset_prefix.csv"
        df = builder.build_csv(str(csv_path))
        
        # Verificações
        self.assertEqual(len(df), 3)
        
        # Verificar matching correto
        for _, row in df.iterrows():
            image_stem = Path(row['image']).stem
            mask_stem = Path(row['mask']).stem
            
            # Remover prefixos
            image_id = image_stem.replace('image_', '')
            mask_id = mask_stem.replace('mask_', '')
            
            self.assertEqual(image_id, mask_id)
    
    def test_csv_builder_custom_regex(self):
        """Testa construção de CSV com matching custom_regex."""
        # Criar imagens e máscaras com IDs numéricos
        test_images = self.images_folder / "test_regex"
        test_masks = self.masks_folder / "test_regex"
        test_images.mkdir()
        test_masks.mkdir()
        
        for i in range(3):
            image_path = test_images / f"abc_{i:03d}_xyz.tif"
            mask_path = test_masks / f"mask_{i:03d}_end.tif"
            self._create_dummy_tif(image_path)
            self._create_dummy_tif(mask_path)
        
        # Config
        config = OmegaConf.create({
            'images_folder': str(test_images),
            'masks_folder': str(test_masks),
            'image_pattern': '*.tif',
            'mask_pattern': '*.tif',
            'matching_strategy': 'custom_regex',
            'regex_pattern': r'(?P<id>\d{3})'
        })
        
        # Construir CSV
        builder = DatasetCSVBuilder(config)
        csv_path = Path(self.temp_dir) / "dataset_regex.csv"
        df = builder.build_csv(str(csv_path))
        
        # Verificações
        self.assertEqual(len(df), 3)
    
    def test_csv_builder_no_matches(self):
        """Testa comportamento quando não há matches."""
        # Criar apenas imagens, sem máscaras
        test_images = self.images_folder / "test_no_match"
        test_masks = self.masks_folder / "test_no_match"
        test_images.mkdir()
        test_masks.mkdir()
        
        for i in range(3):
            image_path = test_images / f"image_{i:03d}.tif"
            self._create_dummy_tif(image_path)
        
        # Config
        config = OmegaConf.create({
            'images_folder': str(test_images),
            'masks_folder': str(test_masks),
            'image_pattern': '*.tif',
            'mask_pattern': '*.tif',
            'matching_strategy': 'same_basename'
        })
        
        # Construir CSV deve falhar
        builder = DatasetCSVBuilder(config)
        csv_path = Path(self.temp_dir) / "dataset_no_match.csv"
        
        with self.assertRaises(ValueError):
            builder.build_csv(str(csv_path))
    
    def test_validate_dataset(self):
        """Testa validação de dataset."""
        # Criar CSV válido
        df = pd.DataFrame({
            'image': [str(self.images_folder / f"image_{i:03d}.tif") for i in range(5)],
            'mask': [str(self.masks_folder / f"mask_{i:03d}.tif") for i in range(5)],
            'width': [256] * 5,
            'height': [256] * 5
        })
        
        config = OmegaConf.create({
            'images_folder': str(self.images_folder),
            'masks_folder': str(self.masks_folder),
            'image_pattern': '*.tif',
            'mask_pattern': '*.tif',
            'matching_strategy': 'same_basename'
        })
        
        builder = DatasetCSVBuilder(config)
        
        # Validação deve passar
        self.assertTrue(builder.validate_dataset(df))


class TestMetricsCalculatorIntegration(unittest.TestCase):
    """
    Testes de integração para MetricsCalculator.
    
    Nota: Estes testes requerem um ambiente completo com modelos treinados.
    Por isso, são marcados como skip por padrão.
    """
    
    @unittest.skip("Requires full environment with trained models")
    def test_metrics_calculator_basic(self):
        """Teste básico do MetricsCalculator."""
        # Este teste seria implementado quando houver um ambiente de teste completo
        pass


if __name__ == '__main__':
    unittest.main()
