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
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from dataclasses_jsonschema import JsonSchemaMixin
from omegaconf import MISSING
from PIL import Image
from torch.utils.data import Dataset
from pathlib import Path
from typing import Any, Dict, List

class SegmentationDataset(Dataset):
    def __init__(
        self,
        input_csv_path: Path,
        root_dir=None, 
        transform=None,
    ) -> None:
        self.input_csv_path = input_csv_path
        self.df = pd.read_csv(input_csv_path)
        self.root_dir = root_dir
        self.transform = transform
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

@dataclass
class ImageAugumentation(JsonSchemaMixin):
    name: str
    parameters: dict

@dataclass
class DS:
    _target_: str = "pytorch_segmentation_models_trainer.dataset_loader.dataset.SegmentationDataset"
    input_csv_path: str = MISSING
    # n_classes: int
    # dataset_size: int
    # augmentation_list: List[ImageAugumentation]
    # file_path: str
    # base_path: str = ''
    # cache: Any = True
    # shuffle: bool = True
    # shuffle_buffer_size: int = 10000
    # shuffle_csv: bool = True
    # ignore_errors: bool = True
    # num_paralel_reads: int = 1
    # img_dtype: str = 'float32'
    # img_format: str = 'png'
    # img_width: int = 512
    # img_length: int = 512
    # img_bands: int = 3
    # mask_bands: int = 1
    # use_ds_width_len: bool = False

if __name__ == '__main__':
    pass