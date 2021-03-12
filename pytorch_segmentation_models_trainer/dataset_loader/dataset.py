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
import os
from pathlib import Path
from typing import Any, Dict, List

import albumentations as A
import numpy as np
import pandas as pd
from hydra.utils import instantiate
from PIL import Image
from torch.utils.data import Dataset


def load_augmentation_object(input_list):
    try:
        aug_list = [
            instantiate(i) for i in input_list
        ]
    except:
        aug_list = input_list
    return A.Compose(aug_list)

class SegmentationDataset(Dataset):
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
        n_first_rows_to_read=None
    ) -> None:
        self.input_csv_path = input_csv_path
        self.root_dir = root_dir
        self.df = pd.read_csv(input_csv_path) if n_first_rows_to_read is None \
            else pd.read_csv(input_csv_path, nrows=n_first_rows_to_read)
        self.transform = None if augmentation_list is None \
            else load_augmentation_object(augmentation_list)
        self.data_loader = data_loader
        self.len = len(self.df)
        self.image_key = image_key if image_key is not None else 'image_path'
        self.mask_key = mask_key if mask_key is not None else 'label_path'

    def __len__(self) -> int:
        return self.len

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        idx = idx % self.len

        image = self.load_image(idx, key=self.image_key)
        mask = self.load_image(idx, key=self.mask_key, is_mask=True)
        result = {
            'image': image,
            'mask': mask
        } if self.transform is None else self.transform(
                image=image,
                mask=mask
            )
        return result

    def get_path(self, idx, key=None):
        key = self.image_key if key is None else key
        image_path = str(self.df.iloc[idx][key])
        if self.root_dir is not None:
            return os.path.join(
                self.root_dir,
                image_path if not image_path.startswith(os.path.sep) \
                    else image_path[1::]
            )
        return image_path

    def load_image(self, idx, key=None, is_mask=False):
        key = self.image_key if key is None else key
        image_path = self.get_path(idx, key=key)
        image = Image.open(image_path) if not is_mask \
            else Image.open(image_path).convert('L')
        image = np.array(image)
        return (image > 0).astype(np.uint8) if is_mask else image


if __name__ == '__main__':
    pass
