# -*- coding: utf-8 -*-
"""
/***************************************************************************
 pytorch_segmentation_models_trainer
                              -------------------
        begin                : 2021-03-10
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
import albumentations as A
import pytorch_lightning as pl
import torch
from hydra.utils import instantiate
#precision e recall com problema no pytorch lightning 1.2, 
# retirar e depois ver o que fazer
from torch.utils.data import DataLoader

from typing import List, Any

class ImageSegmentationResultCallback(pl.callbacks.base.Callback):
    pass