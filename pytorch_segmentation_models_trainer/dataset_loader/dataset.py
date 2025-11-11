# -*- coding: utf-8 -*-
"""
/***************************************************************************
 pytorch_segmentation_models_trainer
                              -------------------
        begin                : 2021-02-25
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
import itertools
import json
import math
import os
import rasterio
from copy import deepcopy
from abc import abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import albumentations as A
import numpy as np
import pandas as pd
import shapely.wkt
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf
from PIL import Image
from albumentations.pytorch import ToTensorV2
from pytorch_segmentation_models_trainer.utils import polygonrnn_utils
from pytorch_segmentation_models_trainer.utils.object_detection_utils import (
    bbox_xywh_to_xyxy,
    bbox_xyxy_to_xywh,
)
from shapely.geometry import Polygon
from shapely.geometry.base import BaseGeometry
from torch.utils.data import Dataset
import copy

from pytorch_toolbelt.inference.tiles import ImageSlicer, TileMerger
from pytorch_toolbelt.utils.torch_utils import (
    image_to_tensor,
    tensor_from_rgb_image,
    to_numpy,
)
import kornia as K
import gc


def load_augmentation_object(input_list, bbox_params=None):
    if isinstance(input_list, A.Compose):
        return input_list
    try:
        aug_list = [instantiate(i, _recursive_=False) for i in input_list]
    except:
        aug_list = input_list
    if bbox_params is None:
        return A.Compose(aug_list)
    if isinstance(bbox_params, A.BboxParams):
        return A.Compose(aug_list, bbox_params=bbox_params)
    return A.Compose(aug_list, bbox_params=OmegaConf.to_container(bbox_params))


class AbstractDataset(Dataset):
    def __init__(
        self,
        input_csv_path: Path = None,
        df: pd.DataFrame = None,
        root_dir=None,
        augmentation_list=None,
        data_loader=None,
        image_key=None,
        mask_key=None,
        n_first_rows_to_read=None,
    ) -> None:
        if input_csv_path is None and df is None:
            raise ValueError("Must provide either input_csv_path or df")
        self.input_csv_path = input_csv_path
        self.root_dir = root_dir
        if df is not None:
            self.df = df
        else:
            self.df = (
                pd.read_csv(input_csv_path)
                if n_first_rows_to_read is None
                else pd.read_csv(input_csv_path, nrows=n_first_rows_to_read)
            )
        self.len = len(self.df)
        self.transform = (
            None
            if augmentation_list is None
            else load_augmentation_object(augmentation_list)
        )
        self.data_loader = data_loader
        self.image_key = image_key if image_key is not None else "image"
        self.mask_key = mask_key if mask_key is not None else "mask"

    def update_df(self, new_df):
        self.df = new_df
        self.len = len(self.df)

    def __len__(self) -> int:
        return self.len

    @abstractmethod
    def __getitem__(
        self, idx: int
    ) -> Union[Dict[str, Any], Tuple[torch.Tensor, Any], Tuple[torch.Tensor, Any, Any]]:
        """Item getter. Must be reimplemented in each subclass.

        Args:
            idx (int): index of the item to be returned

        Returns:
            Dict[str, Any]: Loaded item.
        """
        pass

    def get_path(self, idx: int, key: str = None, add_root_dir: bool = True):
        key = self.image_key if key is None else key
        image_path = str(self.df.iloc[idx][key])
        if self.root_dir is not None and add_root_dir:
            return self._add_root_dir_to_path(image_path)
        return image_path

    def _add_root_dir_to_path(self, image_path):
        return os.path.join(
            self.root_dir,
            image_path if not image_path.startswith(os.path.sep) else image_path[1::],
        )

    def load_image(self, idx, key=None, is_mask=False, force_rgb=False, is_binary_mask=True):
        key = self.image_key if key is None else key
        image_path = self.get_path(idx, key=key)
        return self.load_image_from_path(
            image_path, is_mask=is_mask, force_rgb=force_rgb, is_binary_mask=is_binary_mask
        )

    def load_image_from_path(self, image_path, is_mask=False, force_rgb=False, is_binary_mask=True):
        image = (
            Image.open(image_path)
            if not is_mask
            else Image.open(image_path).convert("L")
        )
        if force_rgb:
            image = image.convert("RGB")
        image = np.array(image)
        if not is_mask:
            return image
        if is_binary_mask:
            return (image > 0).astype(np.uint8)
        return image.astype(np.uint8)

    def to_tensor(self, x):
        return x if isinstance(x, torch.Tensor) else torch.from_numpy(x)


class ImageDataset(AbstractDataset):
    def __init__(
        self,
        input_csv_path: Path = None,
        df=None,
        root_dir=None,
        augmentation_list=None,
        data_loader=None,
        image_key=None,
        n_first_rows_to_read=None,
    ) -> None:
        super(ImageDataset, self).__init__(
            input_csv_path=input_csv_path,
            df=df,
            root_dir=root_dir,
            augmentation_list=augmentation_list,
            data_loader=data_loader,
            image_key=image_key,
            n_first_rows_to_read=n_first_rows_to_read,
        )

    @classmethod
    def get_grouped_datasets(
        cls,
        df,
        group_by_keys: List[str],
        root_dir=None,
        augmentation_list=None,
        image_key=None,
        n_first_rows_to_read=None,
        **kwargs,
    ):
        return {
            k: cls(
                df=pd.DataFrame(df.iloc[v]).reset_index(),
                root_dir=root_dir,
                augmentation_list=augmentation_list,
                image_key=image_key,
                n_first_rows_to_read=n_first_rows_to_read,
                **kwargs,
            )
            for k, v in df.groupby(group_by_keys).groups.items()
        }

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        idx = idx % self.len

        image = self.load_image(idx, key=self.image_key)
        result = (
            {"image": image} if self.transform is None else self.transform(image=image)
        )
        result.update({"path": self.get_path(idx, key=self.image_key)})
        return result


class TiledInferenceImageDataset(ImageDataset):
    def __init__(
        self,
        input_csv_path: Path = None,
        df=None,
        root_dir=None,
        augmentation_list=None,
        data_loader=None,
        image_key=None,
        normalize_output=True,
        n_first_rows_to_read=None,
        pad_if_needed=False,
        model_input_shape=None,
        step_shape=None,
    ) -> None:
        super(TiledInferenceImageDataset, self).__init__(
            input_csv_path=input_csv_path,
            df=df,
            root_dir=root_dir,
            augmentation_list=None,
            data_loader=data_loader,
            image_key=image_key,
            n_first_rows_to_read=n_first_rows_to_read,
        )
        if pad_if_needed and model_input_shape is None:
            raise ValueError("Must provide model_input_shape if pad_if_needed is True")
        self.pad_if_needed = pad_if_needed
        self.model_input_shape = model_input_shape
        self.step_shape = (224, 224) if step_shape is None else step_shape
        self.transform = A.Normalize() if normalize_output else None
        self.tiler_dict = dict()
        self.shape_dict = dict()
        self.pad_func = A.PadIfNeeded(
            math.ceil(self.df["width"].max() / self.model_input_shape[0])
            * self.model_input_shape[0],
            math.ceil(self.df["height"].max() / self.model_input_shape[1])
            * self.model_input_shape[1],
        )

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        idx = idx % self.len

        image = self.load_image(idx, key=self.image_key)
        self.shape_dict[idx] = image.shape[0:2]
        result = (
            {"image": image} if self.transform is None else self.transform(image=image)
        )
        if self.pad_if_needed:
            result = self.pad_func(**result)
        result.update({"path": self.get_path(idx, key=self.image_key)})
        tiler = ImageSlicer(
            result["image"].shape,
            tile_size=self.model_input_shape,
            tile_step=self.step_shape,
        )
        tiles = [image_to_tensor(tile) for tile in tiler.split(result.pop("image"))]
        result.update({"tiles": torch.stack(tiles)})
        result.update(
            {"tile_image_idx": idx * torch.ones(len(tiles), dtype=torch.int64)}
        )
        result.update({"tiler_object": tiler})
        result.update({"original_shape": tuple(self.df[["width", "height"]].iloc[idx])})
        return result

    @staticmethod
    def collate_fn(batch: List) -> Dict[str, Union[torch.Tensor, List[str]]]:
        """
        :param batch: an iterable of N sets from __getitem__()
        :return: a tensor of images, lists of varying-size tensors of bounding boxes, labels, and difficulties
        """
        paths = [item["path"] for item in batch]
        tiles = torch.cat([item["tiles"] for item in batch], dim=0)
        indexes = torch.cat([item["tile_image_idx"] for item in batch], dim=0)
        tiler_object_list = [item["tiler_object"] for item in batch]
        original_shape_list = [item["original_shape"] for item in batch]
        return {
            "path": paths,
            "tiles": tiles,
            "tile_image_idx": indexes,
            "tiler_object_list": tiler_object_list,
            "original_shape": original_shape_list,
        }

class SegmentationDataset(AbstractDataset):
    """Dataset para segmentação com suporte a seleção de bandas.
    
    Args:
        input_csv_path (Path): Caminho para o arquivo CSV com os dados
        root_dir: Diretório raiz dos dados
        augmentation_list: Lista de augmentações do Albumentations
        data_loader: Configuração do DataLoader
        image_key: Nome da coluna com os caminhos das imagens
        mask_key: Nome da coluna com os caminhos das máscaras
        n_first_rows_to_read: Número de linhas a ler do CSV
        n_classes: Número de classes para segmentação
        selected_bands (Optional[List[int]]): Lista com os índices das bandas a serem 
            carregadas (base 1). Ex: [1, 2, 3] carrega as 3 primeiras bandas.
            Se None, carrega todas as bandas disponíveis.
        use_rasterio (bool): Se True, usa rasterio para carregar imagens (recomendado 
            para imagens multiespectrais). Se False, usa PIL (padrão para RGB).
    """

    def __init__(
        self,
        input_csv_path: Path,
        root_dir=None,
        augmentation_list=None,
        data_loader=None,
        image_key=None,
        mask_key=None,
        n_first_rows_to_read=None,
        n_classes=2,
        selected_bands: Optional[List[int]] = None,
        use_rasterio: bool = False,
        reset_augmentation_function: bool = False,
    ) -> None:
        super(SegmentationDataset, self).__init__(
            input_csv_path=input_csv_path,
            root_dir=root_dir,
            augmentation_list=augmentation_list,
            data_loader=data_loader,
            image_key=image_key,
            mask_key=mask_key,
            n_first_rows_to_read=n_first_rows_to_read,
        )
        self.n_classes = n_classes
        self.selected_bands = selected_bands
        self.use_rasterio = use_rasterio
        self.reset_augmentation_function = reset_augmentation_function
        
        # Validação das bandas selecionadas
        if self.selected_bands is not None:
            if not all(isinstance(b, int) and b > 0 for b in self.selected_bands):
                raise ValueError("selected_bands deve conter apenas inteiros positivos (base 1)")

    def load_image(self, idx: int, key: str = None, is_mask: bool = False, 
                   force_rgb: bool = False, is_binary_mask: bool = True) -> np.ndarray:
        """Carrega imagem com suporte a seleção de bandas.
        
        Args:
            idx: Índice da imagem no dataset
            key: Chave da coluna no DataFrame
            is_mask: Se True, carrega como máscara
            force_rgb: Se True, força conversão para RGB
            is_binary_mask: Se True, converte máscara para binária
            
        Returns:
            np.ndarray: Imagem carregada com shape (H, W, C) ou (H, W) para máscaras
        """
        key = self.image_key if key is None else key
        image_path = self.get_path(idx, key=key)
        
        # Máscaras sempre usam o método original
        if is_mask:
            return self.load_image_from_path(
                image_path, 
                is_mask=True, 
                force_rgb=force_rgb,
                is_binary_mask=is_binary_mask
            )
        
        # Se usar rasterio OU se houver seleção de bandas, usa método com rasterio
        if self.use_rasterio or self.selected_bands is not None:
            return self._load_image_with_rasterio(image_path)
        else:
            # Método padrão para imagens RGB
            return self.load_image_from_path(image_path, is_mask=False, force_rgb=force_rgb)

    def _load_image_with_rasterio(self, image_path: str) -> np.ndarray:
        with rasterio.open(image_path, 'r') as src:
            data = src.read() if self.selected_bands is None \
                else src.read(self.selected_bands)
            image = np.transpose(data, (1, 2, 0)).copy()
            del data
            src.close()
        return np.array(image, dtype=np.uint8)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Retorna item do dataset com suporte a bandas personalizadas."""
        idx = idx % self.len

        image = self.load_image(idx, key=self.image_key)
        mask = self.load_image(
            idx, 
            key=self.mask_key, 
            is_mask=True, 
            is_binary_mask=(self.n_classes == 2)
        )
        if self.transform is None:
            image = torch.from_numpy(image).float()
            image = image.permute(2, 0, 1)  # (H,W,C) → (C,H,W) para PyTorch
            image = image / 255.0
            mask = torch.from_numpy(mask).long()
            return {
                "image": image,
                "mask": mask
            }
        # This copy is to avoid data leaks from albumentations.
        # Albumentations stores caches and sometimes do not purge them from the memory, 
        # inducing crashes on the server while training, due to lack of available memory.
        # We added a deepcopy to copy the function, perform the transformation, copy the 
        # result and delete any albumentations related object before returning the evaluated data.
        if not self.reset_augmentation_function:
            return self.transform(image=image, mask=mask)
        transform_func = deepcopy(self.transform)
        output = transform_func(image=image, mask=mask)
        output_dict = {
            "image": output["image"].clone().detach() if torch.is_tensor(output["image"]) else output["image"].copy(),
            "mask": output["mask"].clone().detach() if torch.is_tensor(output["mask"]) else output["mask"].copy(),
        }
        del output
        del transform_func
        if idx % 100 == 0:
            gc.collect()
        return output_dict


class FrameFieldSegmentationDataset(SegmentationDataset):
    def __init__(
        self,
        input_csv_path: Path,
        root_dir=None,
        augmentation_list=None,
        data_loader=None,
        image_key=None,
        mask_key=None,
        multi_band_mask=False,
        boundary_mask_key=None,
        return_boundary_mask=True,
        vertex_mask_key=None,
        return_vertex_mask=True,
        n_first_rows_to_read=None,
        return_crossfield_mask=True,
        crossfield_mask_key=None,
        return_distance_mask=True,
        distance_mask_key=None,
        return_size_mask=True,
        size_mask_key=None,
        image_width=224,
        image_height=224,
        gpu_augmentation_list=None,
    ) -> None:
        mask_key = "polygon_mask" if mask_key is None else mask_key
        super(FrameFieldSegmentationDataset, self).__init__(
            input_csv_path,
            root_dir=root_dir,
            augmentation_list=augmentation_list,
            data_loader=data_loader,
            image_key=image_key,
            mask_key=mask_key,
            n_first_rows_to_read=n_first_rows_to_read,
        )
        self.multi_band_mask = multi_band_mask
        self.boundary_mask_key = (
            boundary_mask_key if boundary_mask_key is not None else "boundary_mask"
        )
        self.vertex_mask_key = (
            vertex_mask_key if vertex_mask_key is not None else "vertex_mask"
        )
        self.return_crossfield_mask = return_crossfield_mask
        self.return_distance_mask = return_distance_mask
        self.return_size_mask = return_size_mask
        self.crossfield_mask_key = (
            crossfield_mask_key
            if crossfield_mask_key is not None
            else "crossfield_mask"
        )
        self.distance_mask_key = (
            distance_mask_key if distance_mask_key is not None else "distance_mask"
        )
        self.size_mask_key = size_mask_key if size_mask_key is not None else "size_mask"
        self.alternative_transform = A.Compose(
            [A.Resize(image_height, image_width), A.Normalize(), ToTensorV2()]
        )
        self.masks_to_load_dict = {
            mask_key: True,
            self.boundary_mask_key: return_boundary_mask,
            self.vertex_mask_key: return_vertex_mask,
        }

    def load_masks(self, idx):
        if self.multi_band_mask:
            multi_band_mask = self.load_image(idx, key=self.mask_key, is_mask=True)
            mask_dict = {
                self.mask_key: multi_band_mask[:, :, 0],
                self.boundary_mask_key: multi_band_mask[:, :, 1],
                self.vertex_mask_key: multi_band_mask[:, :, 2],
            }
        else:
            mask_dict = {
                mask_key: self.load_image(idx, key=mask_key, is_mask=True)
                for mask_key, load_mask in self.masks_to_load_dict.items()
                if load_mask
            }
        if self.return_crossfield_mask:
            mask_dict[self.crossfield_mask_key] = self.load_image(
                idx, key=self.crossfield_mask_key, is_mask=False
            )
        if self.return_distance_mask:
            mask_dict[self.distance_mask_key] = self.load_image(
                idx, key=self.distance_mask_key, is_mask=False
            )
        if self.return_size_mask:
            mask_dict[self.size_mask_key] = self.load_image(
                idx, key=self.size_mask_key, is_mask=False
            )
        return mask_dict

    def compute_class_freq(self, gt_polygons_image):
        pass

    def get_mean_axis(self, mask):
        if len(mask.shape) > 2:
            return tuple(
                [
                    idx
                    for idx, shape in enumerate(mask.shape)
                    if shape != min(mask.shape)
                ]
            )
        elif len(mask.shape) == 2:
            return (0, 1)
        else:
            return 0

    def is_valid_crop(self, ds_item_dict):
        gt_polygons_mask = (0 < ds_item_dict["gt_polygons_image"]).float()
        background_freq = 1 - torch.sum(ds_item_dict["class_freq"], dim=0)
        pixel_class_freq = (
            gt_polygons_mask * ds_item_dict["class_freq"][:, None, None]
            + (1 - gt_polygons_mask) * background_freq[None, None, None]
        )
        if pixel_class_freq.min() == 0 or (
            "sizes" in ds_item_dict and ds_item_dict["sizes"].min() == 0
        ):
            return False
        return True

    def build_ds_item_dict(self, idx, transformed):
        gt_polygons_image = self.to_tensor(
            np.stack(
                [
                    transformed["masks"][0],
                    transformed["masks"][1],
                    transformed["masks"][2],
                ],
                axis=0,
            )
        ).float()
        ds_item_dict = {
            "idx": idx,
            "path": self.get_path(idx),
            "image": self.to_tensor(transformed["image"]),
            "gt_polygons_image": gt_polygons_image,
            "class_freq": torch.mean(gt_polygons_image, axis=(1, 2)),
        }
        return ds_item_dict

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        if self.multi_band_mask:
            return super().__getitem__(idx)
        image = self.load_image(idx, force_rgb=True)
        mask_dict = self.load_masks(idx)
        if self.transform is None:
            ds_item_dict = {
                "idx": idx,
                "path": self.get_path(idx),
                "image": self.to_tensor(image),
                "gt_polygons_image": self.to_tensor(
                    np.stack(
                        [
                            mask_dict[key]
                            for key, load_key in self.masks_to_load_dict.items()
                            if load_key
                        ],
                        axis=-1,
                    )
                ),
                "class_freq": self.to_tensor(
                    np.mean(
                        mask_dict[self.mask_key],
                        axis=self.get_mean_axis(mask_dict[self.mask_key]),
                    )
                    / 255
                    if "class_freq" not in self.df.columns
                    else self.to_tensor(
                        np.fromstring(
                            self.df.iloc[idx]["class_freq"]
                            .replace("[", "")
                            .replace("]", ""),
                            sep=" ",
                        )
                    )
                ),
            }
            if self.return_crossfield_mask:
                ds_item_dict["gt_crossfield_angle"] = (
                    self.to_tensor(mask_dict[self.crossfield_mask_key])
                    .float()
                    .unsqueeze(0)
                )
            if self.return_distance_mask:
                ds_item_dict["distances"] = (
                    self.to_tensor(mask_dict[self.distance_mask_key])
                    .float()
                    .unsqueeze(0)
                )
            if self.return_size_mask:
                ds_item_dict["sizes"] = (
                    self.to_tensor(mask_dict[self.size_mask_key]).float().unsqueeze(0)
                )
            return ds_item_dict
        transformed = self.transform(image=image, masks=list(mask_dict.values()))
        ds_item_dict = self.build_ds_item_dict(idx, transformed)
        if not self.is_valid_crop(ds_item_dict):
            transformed = self.alternative_transform(
                image=image, masks=list(mask_dict.values())
            )
            ds_item_dict = self.build_ds_item_dict(idx, transformed)

        mask_idx = sum(self.masks_to_load_dict.values())
        if self.return_crossfield_mask:
            ds_item_dict["gt_crossfield_angle"] = (
                self.to_tensor(transformed["masks"][mask_idx]).float().unsqueeze(0)
            )
            mask_idx += 1
        if self.return_distance_mask:
            ds_item_dict["distances"] = (
                self.to_tensor(transformed["masks"][mask_idx]).float().unsqueeze(0)
            )
            mask_idx += 1
        if self.return_size_mask:
            ds_item_dict["sizes"] = (
                self.to_tensor(transformed["masks"][mask_idx]).float().unsqueeze(0)
            )
        return ds_item_dict


class PolygonRNNDataset(AbstractDataset):
    def __init__(
        self,
        input_csv_path: Path,
        sequence_length: int = 60,
        root_dir=None,
        augmentation_list=None,
        data_loader=None,
        image_key: str = None,
        mask_key: str = None,
        image_width_key: str = None,
        image_height_key: str = None,
        scale_h_key: str = None,
        scale_w_key: str = None,
        min_col_key: str = None,
        min_row_key: str = None,
        original_image_path_key: str = None,
        original_polygon_key: str = None,
        n_first_rows_to_read: str = None,
        dataset_type: str = "train",
        grid_size=28,
    ) -> None:
        super(PolygonRNNDataset, self).__init__(
            input_csv_path=input_csv_path,
            root_dir=root_dir,
            augmentation_list=augmentation_list,
            data_loader=data_loader,
            image_key=image_key,
            mask_key=mask_key,
            n_first_rows_to_read=n_first_rows_to_read,
        )
        self.sequence_length = sequence_length
        self.scale_h_key = scale_h_key if scale_h_key is not None else "scale_h"
        self.scale_w_key = scale_w_key if scale_w_key is not None else "scale_w"
        self.min_col_key = min_col_key if min_col_key is not None else "min_col"
        self.min_row_key = min_row_key if min_row_key is not None else "min_row"
        self.original_image_path_key = (
            original_image_path_key
            if original_image_path_key is not None
            else "original_image_path"
        )
        self.original_polygon_key = (
            original_polygon_key
            if original_polygon_key is not None
            else "original_polygon_wkt"
        )
        if dataset_type not in ["train", "val"]:
            raise NotImplemented
        self.dataset_type = dataset_type
        if dataset_type == "val":
            self.unique_image_path_list = self.df[self.original_image_path_key].unique()
        self.grid_size = grid_size

    def load_polygon(self, idx: int) -> Tuple[np.ndarray, int]:
        mask_name = os.path.join(self.root_dir, self.df.iloc[idx][self.mask_key])
        return self._load_polygon_from_path(mask_name)

    def _load_polygon_from_path(self, mask_name: str) -> Tuple[np.ndarray, int]:
        with open(mask_name, "r") as f:
            json_file = json.load(f)
        polygon = np.array(json_file["polygon"])
        point_num = len(json_file["polygon"])
        return polygon, point_num

    def __getitem__(
        self,
        index: int,
        load_images: bool = True,
        get_bbox: bool = False,
        original_image_bounds: Optional[Tuple] = None,
    ) -> Dict[str, Any]:
        polygon, num_vertexes = self.load_polygon(index)
        label_array, label_index_array = polygonrnn_utils.build_arrays(
            polygon, num_vertexes, self.sequence_length, grid_size=self.grid_size
        )
        output_dict: Dict[str, Any] = {
            "x1": self.to_tensor(label_array[2]).float(),
            "x2": self.to_tensor(label_array[:-2]).float(),
            "x3": self.to_tensor(label_array[1:-1]).float(),
            "ta": self.to_tensor(label_index_array[2:]).long(),
        }
        if get_bbox:
            img_bounds = (
                (300, 300) if original_image_bounds is None else original_image_bounds
            )
            bbox = polygonrnn_utils.get_extended_bounds_from_np_array_polygon(
                polygon, img_bounds, extend_factor=0.1
            )
            output_dict.update({"bbox": torch.floor(torch.tensor(bbox))})
        if self.dataset_type == "val":
            output_dict.update(
                {
                    "polygon_wkt": self.df.iloc[index][self.original_polygon_key],
                    "scale_h": self.df.iloc[index][self.scale_h_key],
                    "scale_w": self.df.iloc[index][self.scale_w_key],
                    "min_col": self.df.iloc[index][self.min_col_key],
                    "min_row": self.df.iloc[index][self.min_row_key],
                    "original_image_path": self.get_path(
                        index, key=self.original_image_path_key
                    ),
                }
            )
        if load_images:
            image = self.load_image(
                index, key=self.image_key, is_mask=False, force_rgb=True
            )
            if self.transform is not None:
                augmented = self.transform(image=image)
                image = augmented["image"]
            output_dict.update({"image": self.to_tensor(image).float()})
        return output_dict

    def get_images_from_image_path(self, image_path: str) -> Dict[str, Any]:
        entries, items = self.get_training_images_from_image_path(image_path)
        return {
            "original_image": self.load_image(
                entries.index.tolist()[0], key=self.original_image_path_key
            ),
            "shapely_polygon_list": [
                shapely.wkt.loads(item["polygon_wkt"]) for item in items
            ],
            "croped_images": torch.stack([item["image"] for item in items]),
            "scale_h": torch.tensor([item["scale_h"] for item in items]),
            "scale_w": torch.tensor([item["scale_w"] for item in items]),
            "min_col": torch.tensor([item["min_col"] for item in items]),
            "min_row": torch.tensor([item["min_row"] for item in items]),
            "ta": torch.stack([item["ta"] for item in items]),
        }

    def get_entries_from_image_path(self, image_path: str) -> pd.DataFrame:
        return self.df.loc[self.df[self.original_image_path_key] == image_path]

    def get_training_images_from_image_path(
        self, image_path: str
    ) -> Tuple[pd.DataFrame, List[Dict[str, Any]]]:
        entries = self.get_entries_from_image_path(image_path)
        items = [self.__getitem__(idx) for idx in entries.index.tolist()]
        return entries, items

    def get_n_image_path_lists(self, n: int) -> List:
        return list(
            itertools.chain.from_iterable(
                self.get_images_from_image_path(image_path)
                for image_path in self.unique_image_path_list[:n]
            )
        )

    def get_n_image_path_dict_list(self, n: int) -> Dict[str, Any]:
        output_dict = {
            image_path: self.get_images_from_image_path(image_path)
            for image_path in self.unique_image_path_list[:n]
        }
        return output_dict


class ObjectDetectionDataset(AbstractDataset):
    def __init__(
        self,
        input_csv_path: Path,
        root_dir=None,
        augmentation_list=None,
        data_loader=None,
        image_key=None,
        mask_key=None,
        bounding_box_key=None,
        n_first_rows_to_read=None,
        bbox_format="xywh",
        bbox_output_format="xyxy",
        bbox_params=None,
    ) -> None:
        super(ObjectDetectionDataset, self).__init__(
            input_csv_path=input_csv_path,
            root_dir=root_dir,
            augmentation_list=None,
            data_loader=data_loader,
            image_key=image_key,
            mask_key=mask_key,
            n_first_rows_to_read=n_first_rows_to_read,
        )
        self.transform = (
            None
            if augmentation_list is None
            else load_augmentation_object(augmentation_list, bbox_params=bbox_params)
        )
        self.bounding_box_key = (
            "bounding_boxes" if bounding_box_key is None else bounding_box_key
        )
        self.bbox_format = bbox_format
        self.bbox_output_format = bbox_output_format

    def convert_bbox(self, bbox: List) -> List:
        if self.bbox_format == self.bbox_output_format:
            return bbox
        elif self.bbox_format == "xywh" and self.bbox_output_format == "xyxy":
            return bbox_xywh_to_xyxy(bbox)
        elif self.bbox_format == "xyxy" and self.bbox_output_format == "xywh":
            return bbox_xyxy_to_xywh(bbox)
        else:
            raise NotImplementedError

    def load_bounding_boxes_and_labels(
        self, idx: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        bbox_path = self.get_path(idx, key=self.bounding_box_key)
        with open(bbox_path, "r") as f:
            json_file = json.load(f)
        bbox_list, label_list = [], []
        for box_item in json_file:
            bbox_list.append(box_item["bbox"])
            label_list.append(box_item["class"])
        return (
            torch.as_tensor(bbox_list, dtype=torch.float32),
            torch.as_tensor(label_list, dtype=torch.int64),
        )

    def __getitem__(
        self, index: int
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], int]:
        image = self.load_image(
            index, key=self.image_key, is_mask=False, force_rgb=True
        )
        bbox_list, label_list = self.load_bounding_boxes_and_labels(index)
        ds_item_dict = {"image": image, "bboxes": bbox_list, "labels": label_list}
        if self.transform is not None:
            ds_item_dict = self.transform(**ds_item_dict)
        image = ds_item_dict.pop("image")
        ds_item_dict["boxes"] = torch.as_tensor(
            [self.convert_bbox(bbox) for bbox in ds_item_dict.pop("bboxes")],
            dtype=torch.float32,
        )
        ds_item_dict["labels"] = torch.as_tensor(
            ds_item_dict["labels"], dtype=torch.int64
        )
        return image, ds_item_dict, index

    @staticmethod
    def collate_fn(
        batch: List,
    ) -> Tuple[torch.Tensor, List[Dict[str, torch.Tensor]], torch.Tensor]:
        """
        :param batch: an iterable of N sets from __getitem__()
        :return: a tensor of images, lists of varying-size tensors of bounding boxes, labels, and difficulties
        """

        images, targets, indexes = tuple(zip(*batch))
        images = torch.stack(images, dim=0)
        indexes = torch.tensor(indexes, dtype=torch.int64)
        return images, list(targets), indexes


class InstanceSegmentationDataset(ObjectDetectionDataset):
    def __init__(
        self,
        input_csv_path: Path,
        root_dir=None,
        augmentation_list=None,
        data_loader=None,
        image_key=None,
        mask_key=None,
        bounding_box_key=None,
        keypoint_key=None,
        n_first_rows_to_read=None,
        bbox_format="xywh",
        bbox_output_format="xyxy",
        return_mask=True,
        return_keypoints=False,
        bbox_params=None,
    ) -> None:
        mask_key = "polygon_mask" if mask_key is None else mask_key
        super(InstanceSegmentationDataset, self).__init__(
            input_csv_path=input_csv_path,
            root_dir=root_dir,
            augmentation_list=augmentation_list,
            data_loader=data_loader,
            image_key=image_key,
            mask_key=mask_key,
            bounding_box_key=bounding_box_key,
            n_first_rows_to_read=n_first_rows_to_read,
            bbox_format=bbox_format,
            bbox_output_format=bbox_output_format,
            bbox_params=bbox_params,
        )
        self.return_mask = return_mask
        self.return_keypoints = return_keypoints
        self.keypoint_key = "keypoints" if keypoint_key is None else keypoint_key

    def load_keypoints(self, idx):
        keypoints_path = self.get_path(idx, key=self.keypoint_key)
        with open(keypoints_path, "r") as f:
            json_file = json.load(f)
        return torch.as_tensor(json_file[self.keypoint_key], dtype=torch.float32)

    def __getitem__(
        self, index: int
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], int]:
        image = self.load_image(
            index, key=self.image_key, is_mask=False, force_rgb=True
        )
        bbox_list, label_list = self.load_bounding_boxes_and_labels(index)
        ds_item_dict = {"image": image, "bboxes": bbox_list, "labels": label_list}
        if self.return_mask:
            ds_item_dict["masks"] = [
                self.load_image(index, key=self.mask_key, is_mask=True)
            ]
        if self.return_keypoints:
            ds_item_dict["keypoints"] = self.load_keypoints(index)
        if self.transform is not None:
            ds_item_dict = self.transform(**ds_item_dict)
        image = ds_item_dict.pop("image")
        ds_item_dict["boxes"] = torch.as_tensor(
            [self.convert_bbox(bbox) for bbox in ds_item_dict.pop("bboxes")],
            dtype=torch.float32,
        )
        ds_item_dict["labels"] = torch.as_tensor(
            ds_item_dict["labels"], dtype=torch.int64
        )
        if self.return_mask:
            ds_item_dict["masks"] = torch.as_tensor(
                ds_item_dict["masks"], dtype=torch.uint8
            )
        return image, ds_item_dict, index


class NaiveModPolyMapperDataset(Dataset):
    def __init__(
        self,
        object_detection_dataset: ObjectDetectionDataset,
        polygon_rnn_dataset: PolygonRNNDataset,
    ) -> None:
        super(NaiveModPolyMapperDataset, self).__init__()
        self.object_detection_dataset = object_detection_dataset
        self.polygon_rnn_dataset = polygon_rnn_dataset

    def __len__(self) -> int:
        return len(self.object_detection_dataset)

    def get_polygonrnn_data(self, idx: int) -> Dict[str, Any]:
        (
            _,
            polygon_rnn_data,
        ) = self.polygon_rnn_dataset.get_training_images_from_image_path(
            self.object_detection_dataset.get_path(
                idx, key=self.object_detection_dataset.image_key, add_root_dir=False
            )
        )
        return {"polygon_rnn_data": polygon_rnn_data}

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, Dict[str, Any], int]:
        image, ds_item_dict, _ = self.object_detection_dataset[index]
        ds_item_dict.update(self.get_polygonrnn_data(index))
        return image, ds_item_dict, index

    @staticmethod
    def collate_fn(
        batch: List,
    ) -> Tuple[torch.Tensor, List[Dict[str, torch.Tensor]], torch.Tensor]:
        """
        :param batch: an iterable of N sets from __getitem__()
        :return: a tensor of images, lists of varying-size tensors of bounding boxes, labels, and difficulties
        """

        images, targets, indexes = tuple(zip(*batch))
        images = torch.stack(images, dim=0)
        indexes = torch.tensor(indexes, dtype=torch.int64)
        return images, list(targets), indexes


class ModPolyMapperDataset(NaiveModPolyMapperDataset):
    def __init__(
        self,
        object_detection_dataset: ObjectDetectionDataset,
        polygon_rnn_dataset: PolygonRNNDataset,
    ) -> None:
        super(ModPolyMapperDataset, self).__init__(
            object_detection_dataset=object_detection_dataset,
            polygon_rnn_dataset=polygon_rnn_dataset,
        )

    def get_polygonrnn_polygon_data(self, idx: int) -> List[Any]:
        image_path = self.object_detection_dataset.get_path(
            idx, key=self.object_detection_dataset.image_key, add_root_dir=False
        )
        entries = self.polygon_rnn_dataset.get_entries_from_image_path(image_path)
        return [
            self.polygon_rnn_dataset.__getitem__(idx) for idx in entries.index.tolist()
        ]

    def __getitem__(
        self, index: int
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor], int]:
        image, ds_item_dict, _ = self.object_detection_dataset[index]
        polygonrnn_data = self.get_polygonrnn_polygon_data(index)
        dict_items_to_update = {
            key: torch.stack([torch.tensor(item[key]) for item in polygonrnn_data])
            for key in polygonrnn_data[0].keys()
        }
        croped_images = dict_items_to_update.pop("image")
        ds_item_dict.update(dict_items_to_update)
        return image, ds_item_dict, index

    @staticmethod
    def collate_fn(
        batch: List,
    ) -> Tuple[torch.Tensor, List[Dict[str, torch.Tensor]], torch.Tensor]:
        """
        :param batch: an iterable of N sets from __getitem__()
        :return: a tensor of images, lists of varying-size tensors of bounding boxes, labels, and difficulties
        """

        images, targets, indexes = tuple(zip(*batch))
        images = torch.stack(images, dim=0)
        indexes = torch.tensor(indexes, dtype=torch.int64)
        return images, list(targets), indexes


if __name__ == "__main__":
    pass
