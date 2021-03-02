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
    ) -> None:
        self.input_csv_path = input_csv_path
        self.df = pd.read_csv(input_csv_path)
        self.root_dir = root_dir
        self.transform = None if augmentation_list is None \
            else load_augmentation_object(augmentation_list)
        self.data_loader = data_loader
        self.len = len(self.df)

    def __len__(self) -> int:
        return self.len

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        idx = idx % self.len

        image_path  = str(
            self.df.iloc[idx]['image_path']
        )
        mask_path = str(
            self.df.iloc[idx]['label_path']
        )

        if self.root_dir is not None:
            image_path = image_path.replace(
                '/data', str(self.root_dir)
            )
            mask_path = mask_path.replace(
                '/data', str(self.root_dir)
            )
        image = Image.open(image_path)
        image = np.array(image)
        mask = Image.open(mask_path).convert('L')
        mask = np.array(mask)
        result = {
            'image': image,
            'mask': mask
        }

        if self.transform is not None:
            result = self.transform(
                image=image,
                mask=mask
            )

        return result

if __name__ == '__main__':
    pass
