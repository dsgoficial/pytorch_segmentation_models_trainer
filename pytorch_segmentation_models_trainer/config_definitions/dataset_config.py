# -*- coding: utf-8 -*-
"""
/***************************************************************************
 pytorch_segmentation_models_trainer
                              -------------------
        begin                : 2021-07-19
        git sha              : $Format:%H$
        copyright            : (C) 2021 by Philipe Borba - Cartographic Engineer
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

from dataclasses import dataclass, field
from typing import List, Optional

from omegaconf import MISSING


@dataclass
class DataLoaderConfig:
    shuffle: bool = True
    num_workers: int = 6
    pin_memory: bool = True
    drop_last: bool = True
    prefetch_factor: str = "${hyperparameters.batch_size}"


@dataclass
class DatasetConfig:
    _target_: str = (
        "pytorch_segmentation_models_trainer.dataset_loader.dataset.SegmentationDataset"
    )
    input_csv_path: str = MISSING
    root_dir: str = MISSING
    gpu_augmentation_list: List = field(default_factory=list)
    augmentation_list: List = field(default_factory=list)
    data_loader: DataLoaderConfig = field(default_factory=DataLoaderConfig)
    image_dtype: str = "uint8"


@dataclass
class AutoencoderDatasetConfig(DatasetConfig):
    """Configuração para AutoencoderDataset baseado em CSV.

    Exemplo de YAML::

        train_dataset:
          _target_: pytorch_segmentation_models_trainer.dataset_loader.image_dataset.AutoencoderDataset
          input_csv_path: /data/images.csv
          root_dir: /data
          corruption_augmentation_list:
            - _target_: albumentations.GaussNoise
              p: 0.5
    """

    _target_: str = (
        "pytorch_segmentation_models_trainer.dataset_loader"
        ".image_dataset.AutoencoderDataset"
    )
    corruption_augmentation_list: List = field(default_factory=list)


@dataclass
class AutoencoderRandomCropDatasetConfig:
    """Configuração para AutoencoderRandomCropDataset.

    Lê imagens inteiras de um CSV ou varre ``image_dir`` recursivamente e
    amostra crops aleatórios on-the-fly para treino de reconstrução.

    Exemplo de YAML::

        train_dataset:
          _target_: pytorch_segmentation_models_trainer.dataset_loader.image_dataset.AutoencoderRandomCropDataset
          image_dir: /data/unlabeled
          split: train
          crop_size: [256, 256]
          samples_per_epoch: 20000
    """

    _target_: str = (
        "pytorch_segmentation_models_trainer.dataset_loader"
        ".image_dataset.AutoencoderRandomCropDataset"
    )
    image_dir: str = MISSING
    input_csv_path: Optional[str] = None
    root_dir: Optional[str] = None
    image_extensions: Optional[List[str]] = None
    split: str = "all"
    val_fraction: float = 0.2
    split_seed: int = 42
    crop_size: List[int] = field(default_factory=lambda: [256, 256])
    samples_per_epoch: int = 10000
    selected_bands: Optional[List[int]] = None
    image_dtype: str = "uint8"
    file_cache_maxsize: int = 0
    corruption_augmentation_list: List = field(default_factory=list)
    reset_augmentation_function: bool = False
    gpu_augmentation_list: List = field(default_factory=list)
    augmentation_list: List = field(default_factory=list)
    data_loader: DataLoaderConfig = field(default_factory=DataLoaderConfig)
    image_key: str = "image"
    n_first_rows_to_read: Optional[int] = None
    max_retries: int = 10


@dataclass
class RasterPatchDatasetConfig:
    """Configuração para RasterPatchDataset (janela deslizante determinística).

    O dataset varre dois diretórios recursivamente, emparelha imagens e máscaras
    por caminho relativo, e expõe cada patch possível como um item independente.
    O total de items em ``__len__`` é a soma de patches de todas as imagens — não
    o número de imagens.

    Exemplo de YAML::

        train_dataset:
          _target_: pytorch_segmentation_models_trainer.dataset_loader.raster_patch_dataset.RasterPatchDataset
          image_dir: /data/images
          mask_dir: /data/masks
          extension: .tif
          patch_size: 256
          stride: 128
    """

    _target_: str = (
        "pytorch_segmentation_models_trainer.dataset_loader"
        ".raster_patch_dataset.RasterPatchDataset"
    )
    image_dir: str = MISSING
    mask_dir: str = MISSING
    extension: str = ".tif"
    patch_size: int = 256
    stride: int = 128
    mask_extension: Optional[str] = None
    augmentation_list: List = field(default_factory=list)
    data_loader: DataLoaderConfig = field(default_factory=DataLoaderConfig)
    selected_bands: Optional[List[int]] = None
    image_dtype: str = "uint8"
    n_classes: int = 2
    reset_augmentation_function: bool = False


@dataclass
class CSVWindowedDatasetConfig(DatasetConfig):
    """Configuração para CSVWindowedSegmentationDataset.

    Lê patches específicos de imagens maiores baseando-se em offsets e tamanho
    de patch definidos em um arquivo CSV. Útil para datasets onde os patches
    foram pré-selecionados.

    Exemplo de YAML::

        train_dataset:
          _target_: pytorch_segmentation_models_trainer.dataset_loader.dataset.CSVWindowedSegmentationDataset
          input_csv_path: patches.csv
          image_key: image
          mask_key: mask
          row_off_key: row_off
          col_off_key: col_off
          patch_size_key: patch_size
    """

    _target_: str = (
        "pytorch_segmentation_models_trainer.dataset_loader.dataset.CSVWindowedSegmentationDataset"
    )
    image_key: str = "image"
    mask_key: str = "mask"
    row_off_key: str = "row_off"
    col_off_key: str = "col_off"
    patch_size_key: str = "patch_size"
    n_classes: int = 2
    selected_bands: Optional[List[int]] = None
    use_rasterio: bool = True
    reset_augmentation_function: bool = False


@dataclass
class CSVWindowedImageDatasetConfig(DatasetConfig):
    """Configuração para CSVWindowedImageDataset.

    Lê patches específicos de imagens maiores para tarefas de imagem pura
    (sem máscara), baseando-se em offsets e tamanho de patch definidos em um CSV.

    Exemplo de YAML::

        train_dataset:
          _target_: pytorch_segmentation_models_trainer.dataset_loader.image_dataset.CSVWindowedImageDataset
          input_csv_path: patches.csv
          image_key: image
          row_off_key: row_off
          col_off_key: col_off
          patch_size_key: patch_size
    """

    _target_: str = (
        "pytorch_segmentation_models_trainer.dataset_loader.image_dataset.CSVWindowedImageDataset"
    )
    image_key: str = "image"
    row_off_key: str = "row_off"
    col_off_key: str = "col_off"
    patch_size_key: str = "patch_size"
    selected_bands: Optional[List[int]] = None
    use_rasterio: bool = True
    reset_augmentation_function: bool = False
