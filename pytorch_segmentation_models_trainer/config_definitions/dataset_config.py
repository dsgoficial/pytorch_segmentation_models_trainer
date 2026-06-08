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
from typing import Any, List, Optional

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
class WindowedImageAutoencoderDatasetConfig:
    """Configuração para WindowedImageAutoencoderDataset.

    Lê recortes determinísticos de imagens inteiras para validação, teste ou
    treino de autoencoders. Quando ``verify_windows`` é habilitado, recortes
    ilegíveis são removidos da indexação durante a inicialização.

    Exemplo de YAML::

        val_dataset:
          _target_: pytorch_segmentation_models_trainer.dataset_loader.image_dataset.WindowedImageAutoencoderDataset
          image_dir: /data/validation/images
          crop_size: [224, 224]
          stride: 224
          verify_windows: true
          window_index_cache: /data/validation/cache/window_index.json
          serialize_rasterio_reads: true
          reopen_rasterio_on_read: true
    """

    _target_: str = (
        "pytorch_segmentation_models_trainer.dataset_loader"
        ".image_dataset.WindowedImageAutoencoderDataset"
    )
    image_dir: str = MISSING
    input_csv_path: Optional[str] = None
    root_dir: Optional[str] = None
    image_extensions: Optional[List[str]] = None
    split: str = "all"
    val_fraction: float = 0.2
    split_seed: int = 42
    crop_size: List[int] = field(default_factory=lambda: [256, 256])
    stride: Any = None
    selected_bands: Optional[List[int]] = None
    image_dtype: str = "uint8"
    file_cache_maxsize: int = 0
    verify_windows: bool = False
    window_index_cache: Optional[str] = None
    serialize_rasterio_reads: bool = False
    rasterio_lock_dir: Optional[str] = None
    reopen_rasterio_on_read: bool = False
    corruption_augmentation_list: List = field(default_factory=list)
    gpu_augmentation_list: List = field(default_factory=list)
    augmentation_list: List = field(default_factory=list)
    data_loader: DataLoaderConfig = field(default_factory=DataLoaderConfig)
    image_key: str = "image"
    n_first_rows_to_read: Optional[int] = None


@dataclass
class IterableWindowedImageAutoencoderDatasetConfig(
    WindowedImageAutoencoderDatasetConfig
):
    """Configuração para IterableWindowedImageAutoencoderDataset.

    Variante iterável do dataset windowed de autoencoder. Em DataLoaders com
    múltiplos workers, cada worker recebe imagens inteiras distintas, evitando
    leituras concorrentes do mesmo GeoTIFF.

    Exemplo de YAML::

        val_dataset:
          _target_: pytorch_segmentation_models_trainer.dataset_loader.image_dataset.IterableWindowedImageAutoencoderDataset
          image_dir: /data/validation/images
          crop_size: [224, 224]
          stride: 224
          verify_windows: true
          data_loader:
            shuffle: false
            num_workers: 4
    """

    _target_: str = (
        "pytorch_segmentation_models_trainer.dataset_loader"
        ".image_dataset.IterableWindowedImageAutoencoderDataset"
    )
    data_loader: DataLoaderConfig = field(
        default_factory=lambda: DataLoaderConfig(shuffle=False)
    )


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
class MBTilesMaskWindowedDatasetConfig:
    """Configuração para MBTilesMaskWindowedDataset.

    A máscara GeoTIFF define o grid de treinamento. A imagem MBTiles é
    reprojetada/reamostrada para cada janela da máscara em tempo de leitura.

    Exemplo de YAML::

        train_dataset:
          _target_: pytorch_segmentation_models_trainer.dataset_loader.mbtiles_mask_dataset.MBTilesMaskWindowedDataset
          mbtiles_path: /data/source.mbtiles
          mask_dir: /data/masks
          patch_size: 512
          stride: 512
    """

    _target_: str = (
        "pytorch_segmentation_models_trainer.dataset_loader"
        ".mbtiles_mask_dataset.MBTilesMaskWindowedDataset"
    )
    mbtiles_path: str = MISSING
    mask_paths: Optional[List[str]] = None
    mask_dir: Optional[str] = None
    mask_extension: str = ".tif"
    patch_size: int = 512
    stride: Optional[int] = None
    selected_bands: Optional[List[int]] = None
    image_dtype: str = "uint8"
    image_resampling: str = "bilinear"
    n_classes: int = 2
    augmentation_list: List = field(default_factory=list)
    data_loader: DataLoaderConfig = field(default_factory=DataLoaderConfig)
    return_metadata: bool = True
    window_index_cache: Optional[str] = None


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


@dataclass
class SoftLabelDatasetConfig:
    """Configuration for ``SoftLabelDataset``.

    Example YAML:
        train_dataset:
          _target_: pytorch_segmentation_models_trainer.dataset_loader.soft_label_dataset.SoftLabelDataset
          input_csv_path: /data/train.csv
          image_key: image_path
          p_soft_key: p_soft_path
          w_conf_key: w_conf_path
          augmentation_list:
            - _target_: albumentations.HorizontalFlip
              p: 0.5
            - _target_: albumentations.VerticalFlip
              p: 0.5
            - _target_: albumentations.RandomRotate90
              p: 0.5
            - _target_: albumentations.Normalize
              mean: [0.485, 0.456, 0.406]
              std: [0.229, 0.224, 0.225]
            - _target_: albumentations.pytorch.ToTensorV2
          data_loader:
            batch_size: 16
            num_workers: 8
            shuffle: true
    """

    _target_: str = (
        "pytorch_segmentation_models_trainer.dataset_loader.soft_label_dataset.SoftLabelDataset"
    )
    input_csv_path: Optional[str] = None
    root_dir: Optional[str] = None
    augmentation_list: List = field(default_factory=list)
    data_loader: DataLoaderConfig = field(default_factory=DataLoaderConfig)
    image_key: str = "image_path"
    p_soft_key: str = "p_soft_path"
    w_conf_key: str = "w_conf_path"
    n_first_rows_to_read: Optional[int] = None
    seed: Optional[int] = None


@dataclass
class SoftLabelWindowedDatasetConfig:
    """Configuration for ``SoftLabelWindowedDataset``.

    Example YAML:
        train_dataset:
          _target_: pytorch_segmentation_models_trainer.dataset_loader.soft_label_windowed_dataset.SoftLabelWindowedDataset
          input_csv_path: /data/patches.csv
          image_key: image_path
          p_soft_key: p_soft_path
          w_conf_key: w_conf_path
          row_off_key: row_off
          col_off_key: col_off
          patch_size_key: patch_size
          augmentation_list:
            - _target_: albumentations.HorizontalFlip
              p: 0.5
            - _target_: albumentations.Normalize
              mean: [0.485, 0.456, 0.406]
              std: [0.229, 0.224, 0.225]
            - _target_: albumentations.pytorch.ToTensorV2
    """

    _target_: str = (
        "pytorch_segmentation_models_trainer.dataset_loader.soft_label_windowed_dataset.SoftLabelWindowedDataset"
    )
    input_csv_path: Optional[str] = None
    root_dir: Optional[str] = None
    augmentation_list: List = field(default_factory=list)
    data_loader: DataLoaderConfig = field(default_factory=DataLoaderConfig)
    image_key: str = "image_path"
    p_soft_key: str = "p_soft_path"
    w_conf_key: str = "w_conf_path"
    row_off_key: str = "row_off"
    col_off_key: str = "col_off"
    patch_size_key: str = "patch_size"
    n_first_rows_to_read: Optional[int] = None
    seed: Optional[int] = None


@dataclass
class LulcInputDatasetConfig:
    """Configuration for ``LulcInputDataset``.

    Example YAML:
        train_dataset:
          _target_: pytorch_segmentation_models_trainer.dataset_loader.lulc_input_dataset.LulcInputDataset
          input_csv_path: /data/splits/train.csv
          image_key: image_path
          mask_key: mask_path
          lulc_keys:
            - mapbiomas_path
            - esri_path
            - dw_path
          num_classes: 6
          augmentation_list:
            - _target_: albumentations.HorizontalFlip
              p: 0.5
            - _target_: albumentations.Normalize
              mean: [0.485, 0.456, 0.406]
              std: [0.229, 0.224, 0.225]
            - _target_: albumentations.pytorch.ToTensorV2
          data_loader:
            batch_size: 16
            num_workers: 8
            shuffle: true
            pin_memory: true
    """

    _target_: str = (
        "pytorch_segmentation_models_trainer.dataset_loader"
        ".lulc_input_dataset.LulcInputDataset"
    )
    input_csv_path: Optional[str] = None
    root_dir: Optional[str] = None
    lulc_keys: List[str] = field(default_factory=list)
    num_classes: int = 6
    ignore_lulc_index: int = 255
    image_key: str = "image_path"
    mask_key: str = "mask_path"
    augmentation_list: List = field(default_factory=list)
    data_loader: DataLoaderConfig = field(default_factory=DataLoaderConfig)
    n_first_rows_to_read: Optional[int] = None
    seed: Optional[int] = None


@dataclass
class LulcInputWindowedDatasetConfig(LulcInputDatasetConfig):
    """Configuration for ``LulcInputWindowedDataset``.

    Example YAML:
        train_dataset:
          _target_: pytorch_segmentation_models_trainer.dataset_loader.lulc_input_dataset.LulcInputWindowedDataset
          input_csv_path: /data/splits/train.csv
          lulc_keys: [mapbiomas_path, esri_path, dw_path]
          num_classes: 6
          row_off_key: row_off
          col_off_key: col_off
          patch_size_key: patch_size
    """

    _target_: str = (
        "pytorch_segmentation_models_trainer.dataset_loader"
        ".lulc_input_dataset.LulcInputWindowedDataset"
    )
    row_off_key: str = "row_off"
    col_off_key: str = "col_off"
    patch_size_key: str = "patch_size"


@dataclass
class MBTilesCropsGeoTifMaskDatasetConfig:
    """Configuration for ``MBTilesCropsGeoTifMaskDataset``.

    Loads pre-selected crop windows (from a CSV/Parquet file or a vector file)
    and pairs each window with a spatially-aligned mask patch read from any
    rasterio-readable source (VRT, GeoTIFF, MBTile, …).  Mask resampling is
    always nearest-neighbour to preserve class-index integrity.

    Example YAML (CSV windows, VRT mask)::

        train_dataset:
          _target_: pytorch_segmentation_models_trainer.dataset_loader.mbtiles_crops_dataset.MBTilesCropsGeoTifMaskDataset
          image_mbtiles_path: /data/imagery.mbtiles
          mask_path: /data/masks/mask.vrt
          crops_path: /data/crops.csv
          patch_size: 256
          color_map:
            - [255, 0, 0, 1]
            - [0, 255, 0, 2]
          augmentation_list:
            - _target_: albumentations.HorizontalFlip
              p: 0.5
            - _target_: albumentations.Normalize
              mean: [0.485, 0.456, 0.406]
              std: [0.229, 0.224, 0.225]
            - _target_: albumentations.pytorch.ToTensorV2

    Example YAML (vector windows, single-band GeoTIFF mask)::

        train_dataset:
          _target_: pytorch_segmentation_models_trainer.dataset_loader.mbtiles_crops_dataset.MBTilesCropsGeoTifMaskDataset
          image_mbtiles_path: /data/imagery.mbtiles
          mask_path: /data/masks/mask.tif
          crops_path: /data/crops.gpkg
          patch_size: 256
          n_classes: 2
    """

    _target_: str = (
        "pytorch_segmentation_models_trainer.dataset_loader"
        ".mbtiles_crops_dataset.MBTilesCropsGeoTifMaskDataset"
    )
    image_mbtiles_path: str = MISSING
    mask_path: str = MISSING
    crops_path: str = MISSING
    patch_size: int = 256
    color_map: Optional[List] = None
    n_classes: int = 2
    selected_bands: Optional[List[int]] = None
    image_dtype: str = "uint8"
    image_resampling: str = "bilinear"
    crops_layer: Optional[str] = None
    col_off_key: str = "col_off"
    row_off_key: str = "row_off"
    augmentation_list: List = field(default_factory=list)
    data_loader: DataLoaderConfig = field(default_factory=DataLoaderConfig)
    return_metadata: bool = False
    window_index_cache: Optional[str] = None
