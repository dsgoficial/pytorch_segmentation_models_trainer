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
from abc import abstractmethod
import os
from pytorch_segmentation_models_trainer.utils.object_detection_utils import (
    bbox_xywh_to_xyxy,
    bbox_xyxy_to_xywh,
)
from albumentations.pytorch import ToTensorV2
from albumentations.augmentations.transforms import Normalize
import torch
from pathlib import Path
from typing import Any, Dict, List, Tuple

import albumentations as A
import numpy as np
import pandas as pd
from hydra.utils import instantiate
from PIL import Image
from torch.utils.data import Dataset
import json

from omegaconf import DictConfig, OmegaConf


def load_augmentation_object(input_list, bbox_params=None):
    try:
        aug_list = [instantiate(i, _recursive_=False) for i in input_list]
    except:
        aug_list = input_list
    return (
        A.Compose(aug_list)
        if bbox_params is None
        else A.Compose(aug_list, bbox_params=OmegaConf.to_container(bbox_params))
    )


class AbstractDataset(Dataset):
    def __init__(
        self,
        input_csv_path: Path,
        root_dir=None,
        augmentation_list=None,
        data_loader=None,
        image_key=None,
        mask_key=None,
        n_first_rows_to_read=None,
    ) -> None:
        self.input_csv_path = input_csv_path
        self.root_dir = root_dir
        self.df = (
            pd.read_csv(input_csv_path)
            if n_first_rows_to_read is None
            else pd.read_csv(input_csv_path, nrows=n_first_rows_to_read)
        )
        self.transform = (
            None
            if augmentation_list is None
            else load_augmentation_object(augmentation_list)
        )
        self.data_loader = data_loader
        self.len = len(self.df)
        self.image_key = image_key if image_key is not None else "image"
        self.mask_key = mask_key if mask_key is not None else "mask"

    def __len__(self) -> int:
        return self.len

    @abstractmethod
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Item getter. Must be reimplemented in each subclass.

        Args:
            idx (int): index of the item to be returned

        Returns:
            Dict[str, Any]: Loaded item.
        """
        pass

    def get_path(self, idx, key=None):
        key = self.image_key if key is None else key
        image_path = str(self.df.iloc[idx][key])
        if self.root_dir is not None:
            return os.path.join(
                self.root_dir,
                image_path
                if not image_path.startswith(os.path.sep)
                else image_path[1::],
            )
        return image_path

    def load_image(self, idx, key=None, is_mask=False, force_rgb=False):
        key = self.image_key if key is None else key
        image_path = self.get_path(idx, key=key)
        image = (
            Image.open(image_path)
            if not is_mask
            else Image.open(image_path).convert("L")
        )
        if force_rgb:
            image = image.convert("RGB")
        image = np.array(image)
        return (image > 0).astype(np.uint8) if is_mask else image

    def to_tensor(self, x):
        return x if isinstance(x, torch.Tensor) else torch.from_numpy(x)


class SegmentationDataset(AbstractDataset):
    """[summary]

    Args:
        Dataset ([type]): [description]
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

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        idx = idx % self.len

        image = self.load_image(idx, key=self.image_key)
        mask = self.load_image(idx, key=self.mask_key, is_mask=True)
        result = (
            {"image": image, "mask": mask}
            if self.transform is None
            else self.transform(image=image, mask=mask)
        )
        return result


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
            [A.Resize(image_height, image_width), A.Normalize(), A.pytorch.ToTensorV2()]
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
                idx, key=self.crossfield_mask_key, is_mask=True
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
        image_key=None,
        mask_key=None,
        n_first_rows_to_read=None,
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

    def load_polygon(self, idx):
        mask_name = os.path.join(self.root_dir, self.df.iloc[idx][self.mask_key])
        with open(mask_name, "r") as f:
            json_file = json.load(f)
        polygon = np.array(json_file["polygon"])
        point_num = len(json_file["polygon"])
        return polygon, point_num

    def __getitem__(self, index) -> Dict[str, torch.Tensor]:
        image = self.load_image(
            index, key=self.image_key, is_mask=False, force_rgb=True
        )
        polygon, num_vertexes = self.load_polygon(index)
        label_array, label_index_array = self.build_arrays(polygon, num_vertexes)

        if self.transform is not None:
            augmented = self.transform(image=image)
            image = augmented["image"]
        return {
            "image": self.to_tensor(image).float(),
            "x1": self.to_tensor(label_array[2]).float(),
            "x2": self.to_tensor(label_array[:-2]).float(),
            "x3": self.to_tensor(label_array[1:-1]).float(),
            "ta": self.to_tensor(label_index_array[2:]).long(),
        }

    def build_arrays(self, polygon, num_vertexes):
        point_count = 2
        label_array = np.zeros([self.sequence_length, 28 * 28 + 3])
        label_index_array = np.zeros([self.sequence_length])
        if num_vertexes < self.sequence_length - 3:
            for points in polygon:
                self.populate_label_index_array(
                    point_count, label_array, label_index_array, points
                )
                point_count += 1
            label_array[point_count, 28 * 28] = 1
            label_index_array[point_count] = 28 * 28
            for kkk in range(point_count + 1, self.sequence_length):
                if kkk % (num_vertexes + 3) == num_vertexes + 2:
                    index = 28 * 28
                elif kkk % (num_vertexes + 3) == 0:
                    index = 28 * 28 + 1
                elif kkk % (num_vertexes + 3) == 1:
                    index = 28 * 28 + 2
                else:
                    index_a = int(polygon[kkk % (num_vertexes + 3) - 2][0] / 8)
                    index_b = int(polygon[kkk % (num_vertexes + 3) - 2][1] / 8)
                    index = index_b * 28 + index_a
                label_array[kkk, index] = 1
                label_index_array[kkk] = index
        else:
            scale = num_vertexes * 1.0 / (self.sequence_length - 3)
            index_list = (np.arange(0, self.sequence_length - 3) * scale).astype(int)
            for points in polygon[index_list]:
                self.populate_label_index_array(
                    point_count, label_array, label_index_array, points
                )
                point_count += 1
            for kkk in range(point_count, self.sequence_length):
                index = 28 * 28
                label_array[kkk, index] = 1
                label_index_array[kkk] = index
        return label_array, label_index_array

    def populate_label_index_array(
        self, point_count, label_array, label_index_array, points
    ):
        index_a = int(points[0] / 8)
        index_b = int(points[1] / 8)
        index = index_b * 28 + index_a
        label_array[point_count, index] = 1
        label_index_array[point_count] = index


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

    def convert_bbox(self, bbox):
        if self.bbox_format == self.bbox_output_format:
            return bbox
        elif self.bbox_format == "xywh" and self.bbox_output_format == "xyxy":
            return bbox_xywh_to_xyxy(bbox)
        elif self.bbox_format == "xyxy" and self.bbox_output_format == "xywh":
            return bbox_xyxy_to_xywh(bbox)
        else:
            raise NotImplementedError

    def load_bounding_boxes_and_labels(self, idx):
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

    def __getitem__(self, index) -> Dict[str, torch.Tensor]:
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
        n_first_rows_to_read=None,
        bbox_format="xywh",
        bbox_output_format="xyxy",
        return_mask=True,
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

    def __getitem__(self, index) -> Dict[str, torch.Tensor]:
        image = self.load_image(
            index, key=self.image_key, is_mask=False, force_rgb=True
        )
        bbox_list, label_list = self.load_bounding_boxes_and_labels(index)
        ds_item_dict = {"image": image, "bboxes": bbox_list, "labels": label_list}
        if self.return_mask:
            ds_item_dict["masks"] = [
                self.load_image(index, key=self.mask_key, is_mask=True)
            ]
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


if __name__ == "__main__":
    pass
